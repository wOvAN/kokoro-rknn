#include "engine_pool.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <set>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include "paths.hpp"

// Set from --cpu-threads in server/main.cpp before the pool loads.
extern int g_cpuThreads;

namespace kokoro_server {
namespace {

bool fileExists(const std::string& path) {
  return std::filesystem::exists(path);
}

} // namespace

bool EnginePool::packAvailable(const char* pack) {
  try {
    const auto dir = kokoro::paths::packDir(pack);
    return fileExists((dir / "kokoro_decoder.rknn").string()) ||
           fileExists((dir / "kokoro_decoder.onnx").string());
  } catch (...) {
    return false;
  }
}

kokoro::EngineConfig EnginePool::makeConfig(const std::string& decoderPath) {
  kokoro::EngineConfig cfg;
  if (std::filesystem::path(decoderPath).extension() == ".rknn")
    cfg.decoderWorkers = 3;
  else
    cfg.decoderWorkers = 1;
  cfg.cpuThreads = g_cpuThreads;
  return cfg;
}

void EnginePool::load(const std::string& accelerator) {
  packs_.clear();
  voice_to_pack_.clear();

  static const char* kKnownPacks[] = {"base", "dima"};
  for (const char* pack : kKnownPacks) {
    if (!packAvailable(pack)) {
      spdlog::info("pack '{}' not found on disk, skipping", pack);
      continue;
    }

    const std::string vocab = kokoro::paths::packFile(pack, "config.json");
    // NPU-GEMM variant of the encoder wins when the pack ships it (build.py
    // --npu-matmul), same fallback style as the decoder below.
    std::string encoder = kokoro::paths::packFile(pack, "kokoro_encoder_npu.onnx");
    if (!fileExists(encoder))
      encoder = kokoro::paths::packFile(pack, "kokoro_encoder.onnx");
    const std::string har = kokoro::paths::packFile(pack, "har_generator.onnx");
    std::string decoder = kokoro::paths::packFile(pack, "kokoro_decoder.rknn");
    if (!fileExists(decoder))
      decoder = kokoro::paths::packFile(pack, "kokoro_decoder.onnx");
    const std::string voices = kokoro::paths::packFile(pack, "voices_npy");

    auto engine = std::make_unique<kokoro::Engine>();
    auto t0 = std::chrono::steady_clock::now();
    engine->load(vocab, encoder, har, decoder, voices, accelerator,
                 makeConfig(decoder));
    auto t1 = std::chrono::steady_clock::now();
    spdlog::info("pack '{}' loaded in {:.2f}s (decoder={})",
                 pack, std::chrono::duration<double>(t1 - t0).count(), decoder);

    for (const auto& voice : engine->listVoices())
      voice_to_pack_[voice] = pack;
    packs_.emplace(pack, std::move(engine));
  }

  if (packs_.empty())
    throw std::runtime_error(
        "no model packs loaded; check KOKORO_MODELS_DIR / --models-dir");
}

bool EnginePool::hasPack(const std::string& pack) const {
  return packs_.count(pack) != 0;
}

std::vector<std::string> EnginePool::loadedPacks() const {
  std::vector<std::string> out;
  out.reserve(packs_.size());
  for (const auto& [name, _] : packs_)
    out.push_back(name);
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<std::string> EnginePool::listVoices() const {
  std::vector<std::string> out;
  out.reserve(voice_to_pack_.size());
  for (const auto& [voice, _] : voice_to_pack_)
    out.push_back(voice);
  std::sort(out.begin(), out.end());
  return out;
}

std::string EnginePool::packForVoice(const std::string& voice) const {
  auto it = voice_to_pack_.find(voice);
  if (it == voice_to_pack_.end())
    throw std::runtime_error("unknown voice: " + voice);
  return it->second;
}

std::string EnginePool::resolveModelId(const std::string& model) const {
  if (model == "kokoro-base" || model == "base" || model == "tts-1")
    return hasPack("base") ? "base" : loadedPacks().front();
  if (model == "kokoro-dima" || model == "dima")
    return hasPack("dima") ? "dima" : "";
  if (hasPack(model))
    return model;
  return "";
}

kokoro::Engine& EnginePool::engineFor(
    const std::string& voice,
    const std::optional<std::string>& model) {
  std::string pack;
  if (model && !model->empty()) {
    pack = resolveModelId(*model);
    if (pack.empty())
      throw std::runtime_error("unknown model: " + *model);
    if (!voice.empty()) {
      const std::string voice_pack = packForVoice(voice);
      if (voice_pack != pack)
        throw std::runtime_error(
            "voice '" + voice + "' does not belong to model '" + *model + "'");
    }
  } else if (!voice.empty()) {
    pack = packForVoice(voice);
  } else {
    pack = hasPack("base") ? "base" : loadedPacks().front();
  }

  auto it = packs_.find(pack);
  if (it == packs_.end())
    throw std::runtime_error("pack not loaded: " + pack);
  return *it->second;
}

int EnginePool::sampleRate() const {
  if (packs_.empty())
    throw std::runtime_error("no engines loaded");
  return packs_.begin()->second->config().sampleRate;
}

} // namespace kokoro_server
