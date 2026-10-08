// HTTP + WebSocket API (native + OpenAI-compatible TTS).
// Dual NPU packs (base + dima) via EnginePool.

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <drogon/HttpController.h>
#include <drogon/WebSocketController.h>
#include <drogon/drogon.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <trantor/net/EventLoopThreadPool.h>

#include "OggOpusEncoder.hpp"
#include "engine_pool.hpp"
#include "kokoro.hpp"
#include "mp3_encoder.hpp"

using namespace drogon;

kokoro_server::EnginePool g_pool;
std::string g_authToken;
std::string g_defaultVoice;

// Size comes from --synth-threads (server/main.cpp), which runs before the first
// request, so the pool is built lazily on first use.
int g_synthThreads = 2;
static std::unique_ptr<trantor::EventLoopThreadPool> g_synthPool;
static std::once_flag g_synthPoolOnce;
static void ensureSynthPool() {
  std::call_once(g_synthPoolOnce, [] {
    g_synthPool = std::make_unique<trantor::EventLoopThreadPool>(g_synthThreads,
                                                                "kokoro-synth");
    g_synthPool->start();
  });
}

namespace {

constexpr std::size_t kMaxInputChars = 4096;

struct ApiParams {
  std::string text;
  std::string phonemes;
  std::string voice;
  std::optional<std::string> model;
  float speed = 1.0f;
  bool british = false;
  std::string audio_format = "opus";
};

float clampSpeed(float s) {
  return std::clamp(s, 0.25f, 4.0f);
}

ApiParams parseParams(std::string_view body, bool openai_defaults = false) {
  ApiParams p;
  auto j = nlohmann::json::parse(body);
  if (j.contains("text") && !j["text"].is_null())
    p.text = j["text"].get<std::string>();
  if (j.contains("phonemes") && !j["phonemes"].is_null())
    p.phonemes = j["phonemes"].get<std::string>();
  if (p.text.empty() && p.phonemes.empty())
    throw std::runtime_error("must provide 'text' or 'phonemes'");
  if (!p.text.empty() && !p.phonemes.empty())
    throw std::runtime_error("provide only one of 'text' or 'phonemes'");

  if (j.contains("model") && j["model"].is_string())
    p.model = j["model"].get<std::string>();

  if (j.contains("voice") && j["voice"].is_string()) {
    p.voice = j["voice"].get<std::string>();
  } else if (j.contains("speaker") && j["speaker"].is_string()) {
    p.voice = j["speaker"].get<std::string>();
  } else if (j.contains("speaker_id") && j["speaker_id"].is_number_integer()) {
    auto names = g_pool.listVoices();
    auto idx = j["speaker_id"].get<int64_t>();
    if (idx < 0 || static_cast<std::size_t>(idx) >= names.size())
      throw std::runtime_error("speaker_id out of range");
    p.voice = names[static_cast<std::size_t>(idx)];
  } else {
    p.voice = g_defaultVoice;
  }

  if (j.contains("speed") && j["speed"].is_number())
    p.speed = clampSpeed(j["speed"].get<float>());
  if (j.contains("british") && j["british"].is_boolean())
    p.british = j["british"].get<bool>();

  if (j.contains("audio_format") && j["audio_format"].is_string())
    p.audio_format = j["audio_format"].get<std::string>();
  else if (openai_defaults)
    p.audio_format = "mp3";

  return p;
}

HttpResponsePtr badRequest(const std::string& msg) {
  auto r = HttpResponse::newHttpResponse();
  r->setStatusCode(k400BadRequest);
  r->setContentTypeCode(CT_TEXT_PLAIN);
  r->setBody(msg);
  return r;
}

HttpResponsePtr unauthorized() {
  auto r = HttpResponse::newHttpResponse();
  r->setStatusCode(k401Unauthorized);
  r->setContentTypeCode(CT_APPLICATION_JSON);
  nlohmann::json err;
  err["error"]["message"] = "Incorrect API key provided";
  err["error"]["type"] = "invalid_request_error";
  err["error"]["param"] = nullptr;
  err["error"]["code"] = "invalid_api_key";
  r->setBody(err.dump());
  return r;
}

bool checkAuth(const HttpRequestPtr& req) {
  if (g_authToken.empty()) return true;
  return req->getHeader("Authorization") == "Bearer " + g_authToken;
}

bool checkAuthWs(const HttpRequestPtr& req) {
  return checkAuth(req);
}

void addCors(HttpResponsePtr& r, bool allow_auth = true) {
  r->addHeader("Access-Control-Allow-Origin", "*");
  r->addHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  r->addHeader("Access-Control-Allow-Headers",
               allow_auth ? "Content-Type, Authorization" : "Content-Type");
}

kokoro::SynthesisResult doSynth(kokoro::Engine& engine, const ApiParams& p,
                              kokoro::ChunkCallback emit) {
  if (!p.phonemes.empty())
    return engine.synthesizePhonemes(p.phonemes, p.voice, p.speed, std::move(emit));
  return engine.synthesizeText(p.text, p.voice, p.speed, p.british, std::move(emit));
}

void appendWavHeader(std::vector<uint8_t>& buf, int sr, int channels,
                     uint32_t pcm_bytes) {
  auto put32 = [&](uint32_t v) {
    buf.push_back(v & 0xFF);
    buf.push_back((v >> 8) & 0xFF);
    buf.push_back((v >> 16) & 0xFF);
    buf.push_back((v >> 24) & 0xFF);
  };
  auto put16 = [&](uint16_t v) {
    buf.push_back(v & 0xFF);
    buf.push_back((v >> 8) & 0xFF);
  };
  buf.insert(buf.end(), {'R', 'I', 'F', 'F'});
  put32(36 + pcm_bytes);
  buf.insert(buf.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
  put32(16);
  put16(1);
  put16(channels);
  put32(sr);
  put32(sr * channels * 2);
  put16(channels * 2);
  put16(16);
  buf.insert(buf.end(), {'d', 'a', 't', 'a'});
  put32(pcm_bytes);
}

HttpResponsePtr buildAudioResponse(const std::vector<int16_t>& audio,
                                   const std::string& format, int sr) {
  auto r = HttpResponse::newHttpResponse();
  if (format == "mp3") {
    auto mp3 = kokoro_server::encodeMp3(audio.data(), audio.size(), sr);
    r->addHeader("Content-Type", "audio/mpeg");
    r->setBody(std::string(reinterpret_cast<const char*>(mp3.data()), mp3.size()));
  } else if (format == "wav") {
    std::vector<uint8_t> buf;
    appendWavHeader(buf, sr, 1, static_cast<uint32_t>(audio.size() * 2));
    const auto* pp = reinterpret_cast<const uint8_t*>(audio.data());
    buf.insert(buf.end(), pp, pp + audio.size() * 2);
    r->addHeader("Content-Type", "audio/wav");
    r->setBody(std::string(reinterpret_cast<const char*>(buf.data()), buf.size()));
  } else if (format == "pcm" || format == "raw") {
    r->addHeader("Content-Type", "application/octet-stream");
    r->setBody(std::string(reinterpret_cast<const char*>(audio.data()),
                           audio.size() * sizeof(int16_t)));
  } else if (format == "opus" || format.empty()) {
    auto opus = encodeOgg(std::vector<short>(audio.begin(), audio.end()), sr, 1);
    r->addHeader("Content-Type", "audio/ogg; codecs=opus");
    r->setBody(std::string(reinterpret_cast<const char*>(opus.data()), opus.size()));
  } else if (format == "aac" || format == "flac") {
    throw std::runtime_error("response_format '" + format + "' is not supported");
  } else {
    throw std::runtime_error("unknown audio_format: " + format);
  }
  return r;
}

std::string mapOpenAiFormat(const std::string& fmt) {
  if (fmt == "pcm" || fmt == "wav" || fmt == "mp3" || fmt == "opus")
    return fmt;
  if (fmt == "aac" || fmt == "flac")
    throw std::runtime_error("response_format '" + fmt + "' is not supported");
  return "mp3";
}

} // namespace

namespace api {

class v1 : public HttpController<v1> {
 public:
  METHOD_LIST_BEGIN
  METHOD_ADD(v1::synthesise, "/synthesise", {Post, Options});
  METHOD_ADD(v1::voices, "/voices", Get);
  METHOD_ADD(v1::speakers, "/speakers", Get);
  METHOD_LIST_END

  void synthesise(const HttpRequestPtr& req,
                    std::function<void(const HttpResponsePtr&)>&& cb);
  void voices(const HttpRequestPtr& req,
              std::function<void(const HttpResponsePtr&)>&& cb);
  void speakers(const HttpRequestPtr& req,
                std::function<void(const HttpResponsePtr&)>&& cb);
};

void v1::synthesise(const HttpRequestPtr& req,
                    std::function<void(const HttpResponsePtr&)>&& cb) {
  if (req->method() == Options) {
    auto r = HttpResponse::newHttpResponse();
    addCors(r);
    cb(r);
    return;
  }
  if (req->getContentType() != CT_APPLICATION_JSON) {
    cb(badRequest("Content-Type must be application/json"));
    return;
  }
  if (!checkAuth(req)) {
    cb(unauthorized());
    return;
  }

  ApiParams p;
  try {
    p = parseParams(req->getBody());
  } catch (const std::exception& e) {
    cb(badRequest(e.what()));
    return;
  }

  ensureSynthPool();
  g_synthPool->getNextLoop()->queueInLoop([p = std::move(p),
                                          cb = std::move(cb)]() mutable {
    std::vector<int16_t> audio;
    try {
      auto& engine = g_pool.engineFor(p.voice, p.model);
      doSynth(engine, p, [&](const int16_t* d, std::size_t n) {
        audio.insert(audio.end(), d, d + n);
      });
      cb(buildAudioResponse(audio, p.audio_format, g_pool.sampleRate()));
    } catch (const std::exception& e) {
      cb(badRequest(std::string("synthesis failed: ") + e.what()));
    }
  });
}

void v1::voices(const HttpRequestPtr& req,
                std::function<void(const HttpResponsePtr&)>&& cb) {
  if (!checkAuth(req)) {
    cb(unauthorized());
    return;
  }
  nlohmann::json j = g_pool.listVoices();
  auto r = HttpResponse::newHttpResponse();
  r->setStatusCode(k200OK);
  r->setContentTypeCode(CT_APPLICATION_JSON);
  r->setBody(j.dump());
  cb(r);
}

void v1::speakers(const HttpRequestPtr& req,
                  std::function<void(const HttpResponsePtr&)>&& cb) {
  if (!checkAuth(req)) {
    cb(unauthorized());
    return;
  }
  auto names = g_pool.listVoices();
  nlohmann::json j = nlohmann::json::object();
  for (std::size_t i = 0; i < names.size(); ++i)
    j[names[i]] = static_cast<int64_t>(i);
  auto r = HttpResponse::newHttpResponse();
  r->setStatusCode(k200OK);
  r->setContentTypeCode(CT_APPLICATION_JSON);
  r->setBody(j.dump());
  cb(r);
}

class v1ws : public WebSocketController<v1ws> {
 public:
  void handleNewConnection(const HttpRequestPtr& req,
                           const WebSocketConnectionPtr& ws) override {
    if (!checkAuthWs(req)) {
      ws->forceClose();
      return;
    }
  }
  void handleNewMessage(const WebSocketConnectionPtr& ws, std::string&& msg,
                        const WebSocketMessageType& type) override;
  void handleConnectionClosed(const WebSocketConnectionPtr&) override {}

  WS_PATH_LIST_BEGIN
  WS_PATH_ADD("/api/v1/stream", Get);
  WS_PATH_LIST_END
};

void v1ws::handleNewMessage(const WebSocketConnectionPtr& ws, std::string&& msg,
                            const WebSocketMessageType& type) {
  if (type != WebSocketMessageType::Text) return;
  std::string message = std::move(msg);
  auto wsConn = ws;

  ensureSynthPool();
  g_synthPool->getNextLoop()->queueInLoop([wsConn, message]() mutable {
    ApiParams p;
    try {
      p = parseParams(message);
    } catch (const std::exception& e) {
      nlohmann::json j;
      j["status"] = "failed";
      j["message"] = e.what();
      wsConn->send(j.dump());
      return;
    }

    const int sr = g_pool.sampleRate();
    const bool send_opus = (p.audio_format == "opus" || p.audio_format.empty());
    StreamingOggOpusEncoder enc(sr, 1);

    try {
      auto& engine = g_pool.engineFor(p.voice, p.model);
      doSynth(engine, p, [&](const int16_t* d, std::size_t n) {
        if (send_opus) {
          std::vector<short> pcm(d, d + n);
          auto opus = enc.encode(pcm);
          if (!opus.empty())
            wsConn->send(reinterpret_cast<const char*>(opus.data()), opus.size(),
                         WebSocketMessageType::Binary);
        } else {
          if constexpr (std::endian::native == std::endian::big) {
            std::vector<int16_t> tmp(d, d + n);
            for (auto& s : tmp) s = static_cast<int16_t>((s >> 8) | (s << 8));
            wsConn->send(reinterpret_cast<const char*>(tmp.data()),
                         tmp.size() * sizeof(int16_t),
                         WebSocketMessageType::Binary);
          } else {
            wsConn->send(reinterpret_cast<const char*>(d), n * sizeof(int16_t),
                         WebSocketMessageType::Binary);
          }
        }
      });
    } catch (const std::exception& e) {
      nlohmann::json j;
      j["status"] = "failed";
      j["message"] = e.what();
      wsConn->send(j.dump());
      return;
    }

    if (send_opus) {
      auto tail = enc.finish();
      if (!tail.empty())
        wsConn->send(reinterpret_cast<const char*>(tail.data()), tail.size(),
                     WebSocketMessageType::Binary);
    }
    wsConn->send(R"({"status":"ok","message":"finished"})");
  });
}

} // namespace api

namespace v1 {

class models : public HttpController<models> {
 public:
  METHOD_LIST_BEGIN
  METHOD_ADD(models::listModels, "/models", Get);
  METHOD_LIST_END

  void listModels(const HttpRequestPtr& req,
                  std::function<void(const HttpResponsePtr&)>&& cb);
};

void models::listModels(const HttpRequestPtr& req,
                        std::function<void(const HttpResponsePtr&)>&& cb) {
  if (!checkAuth(req)) {
    cb(unauthorized());
    return;
  }

  nlohmann::json data = nlohmann::json::array();
  auto add_model = [&](const std::string& id) {
    data.push_back({{"id", id},
                    {"object", "model"},
                    {"created", 0},
                    {"owned_by", "kokoro-rknn"}});
  };

  for (const auto& pack : g_pool.loadedPacks())
    add_model("kokoro-" + pack);
  if (g_pool.hasPack("base"))
    add_model("tts-1");

  nlohmann::json body;
  body["object"] = "list";
  body["data"] = data;

  auto r = HttpResponse::newHttpResponse();
  r->setStatusCode(k200OK);
  r->setContentTypeCode(CT_APPLICATION_JSON);
  r->setBody(body.dump());
  cb(r);
}

class audio : public HttpController<audio> {
 public:
  METHOD_LIST_BEGIN
  METHOD_ADD(audio::speech, "/speech", {Post, Options});
  METHOD_LIST_END

  void speech(const HttpRequestPtr& req,
              std::function<void(const HttpResponsePtr&)>&& cb);
};

void audio::speech(const HttpRequestPtr& req,
                   std::function<void(const HttpResponsePtr&)>&& cb) {
  if (req->method() == Options) {
    auto r = HttpResponse::newHttpResponse();
    addCors(r);
    cb(r);
    return;
  }
  if (req->getContentType() != CT_APPLICATION_JSON) {
    cb(badRequest("Content-Type must be application/json"));
    return;
  }
  if (!checkAuth(req)) {
    cb(unauthorized());
    return;
  }

  nlohmann::json j;
  try {
    j = nlohmann::json::parse(req->getBody());
  } catch (...) {
    cb(badRequest("invalid JSON"));
    return;
  }

  if (!j.contains("input") || !j["input"].is_string()) {
    cb(badRequest("missing 'input'"));
    return;
  }

  ApiParams p;
  p.text = j["input"].get<std::string>();
  if (p.text.size() > kMaxInputChars) {
    cb(badRequest("input exceeds maximum length of 4096 characters"));
    return;
  }
  p.voice = j.value("voice", g_defaultVoice);
  if (j.contains("model") && j["model"].is_string())
    p.model = j["model"].get<std::string>();
  if (j.contains("speed") && j["speed"].is_number())
    p.speed = clampSpeed(j["speed"].get<float>());

  try {
    const std::string fmt = j.value("response_format", "mp3");
    p.audio_format = mapOpenAiFormat(fmt);
  } catch (const std::exception& e) {
    cb(badRequest(e.what()));
    return;
  }

  ensureSynthPool();
  g_synthPool->getNextLoop()->queueInLoop([p = std::move(p),
                                          cb = std::move(cb)]() mutable {
    std::vector<int16_t> audio;
    try {
      auto& engine = g_pool.engineFor(p.voice, p.model);
      doSynth(engine, p, [&](const int16_t* d, std::size_t n) {
        audio.insert(audio.end(), d, d + n);
      });
      cb(buildAudioResponse(audio, p.audio_format, g_pool.sampleRate()));
    } catch (const std::exception& e) {
      cb(badRequest(std::string("synthesis failed: ") + e.what()));
    }
  });
}

} // namespace v1
