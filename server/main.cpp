#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include <drogon/drogon.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "engine_pool.hpp"
#include "g2p.hpp"
#include "paths.hpp"
#include "phonemizer.hpp"

using namespace drogon;

namespace kokoro_server {

struct RunConfig {
  std::optional<std::filesystem::path> espeakDataPath;
  std::optional<std::filesystem::path> lexiconDir;
  std::optional<std::filesystem::path> webRoot;
  std::string accelerator = "";
  std::string ip = "0.0.0.0";
  uint16_t port = 8848;
  std::string authToken = "";
  bool disableWebUI = false;
  std::string defaultVoice = "sveta";
  std::filesystem::path modelsDir;
  // ONNX Runtime intra-op threads (0 = ORT default, all cores) and the number
  // of syntheses that may run at once. See EngineConfig::cpuThreads.
  int cpuThreads = 0;
  int synthThreads = 0;
};

} // namespace kokoro_server

extern kokoro_server::EnginePool g_pool;
extern std::string g_authToken;
extern std::string g_defaultVoice;
// Defined here, read by engine_pool.cpp when it builds each engine.
int g_cpuThreads = 0;
extern int g_synthThreads;

namespace {

void printUsage(const char* prog) {
  std::cerr <<
    "usage: " << prog << " [options]\n\n"
    "models:\n"
    "  --models-dir DIR      model repo root (or set KOKORO_MODELS_DIR)\n"
    "                        loads packs/base and packs/dima when present\n"
    "\noptional:\n"
    "  --espeak-data DIR     espeak-ng-data directory (else next to executable)\n"
    "  --lexicon-dir DIR     misaki us/gb JSONs (else ./misaki-data)\n"
    "  --web-root DIR        web UI directory (else auto-detect; see below)\n"
    "  --accelerator STR     ONNX accelerator: cuda, tensorrt (default none)\n"
    "  --cpu-threads N       ONNX Runtime intra-op threads for encoder/har\n"
    "                        (default: all cores; use ~cores/synth-threads)\n"
    "  --synth-threads N     concurrent syntheses in flight (default 2)\n"
    "  --default-voice NAME  default voice name (default sveta)\n"
    "  --ip ADDR             bind address (default 0.0.0.0; env HOST/KOKORO_IP)\n"
    "  --port N              bind port (default 8848; env PORT)\n"
    "  --auth [TOKEN]        require bearer token (random if not given)\n"
    "  --disable-web-ui      disable demo web UI\n"
    "  --debug               enable debug logging\n"
    "  -q, --quiet           silence logs\n"
    "  -h, --help            show this help\n"
    "\nWeb-root auto-detect (relative to CWD, first hit wins):\n"
    "  ./server/web-content, ./web-content\n";
}

void parseArgs(int argc, char** argv, kokoro_server::RunConfig& rc) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const std::string& flag) {
      if (i + 1 >= argc) {
        printUsage(argv[0]);
        std::exit(1);
      }
      return std::string(argv[++i]);
    };
    if (a == "--models-dir") rc.modelsDir = need(a);
    else if (a == "--espeak-data") rc.espeakDataPath = std::filesystem::path(need(a));
    else if (a == "--lexicon-dir") rc.lexiconDir = std::filesystem::path(need(a));
    else if (a == "--web-root") rc.webRoot = std::filesystem::path(need(a));
    else if (a == "--accelerator") rc.accelerator = need(a);
    else if (a == "--cpu-threads") rc.cpuThreads = std::stoi(need(a));
    else if (a == "--synth-threads") rc.synthThreads = std::stoi(need(a));
    else if (a == "--default-voice") rc.defaultVoice = need(a);
    else if (a == "--ip") rc.ip = need(a);
    else if (a == "--port") rc.port = static_cast<uint16_t>(std::stoul(need(a)));
    else if (a == "--auth") {
      if (i + 1 < argc && argv[i + 1][0] != '-') rc.authToken = argv[++i];
      else rc.authToken = drogon::utils::secureRandomString(32);
    }
    else if (a == "--disable-web-ui") rc.disableWebUI = true;
    else if (a == "--debug") spdlog::set_level(spdlog::level::debug);
    else if (a == "-q" || a == "--quiet") spdlog::set_level(spdlog::level::off);
    else if (a == "-h" || a == "--help") {
      printUsage(argv[0]);
      std::exit(0);
    }
    else {
      std::cerr << "unknown arg: " << a << "\n";
      printUsage(argv[0]);
      std::exit(1);
    }
  }
}

} // namespace

void applyEnvDefaults(kokoro_server::RunConfig& rc) {
  if (const char* h = std::getenv("HOST")) rc.ip = h;
  else if (const char* h = std::getenv("KOKORO_IP")) rc.ip = h;
  if (const char* p = std::getenv("PORT")) {
    try {
      rc.port = static_cast<uint16_t>(std::stoul(p));
    } catch (...) {
      throw std::runtime_error("invalid PORT env value");
    }
  }
  if (const char* v = std::getenv("KOKORO_DEFAULT_VOICE"))
    rc.defaultVoice = v;
}

int main(int argc, char** argv) {
  spdlog::set_default_logger(spdlog::stderr_color_st("kokoro"));

  kokoro_server::RunConfig rc;
  applyEnvDefaults(rc);
  parseArgs(argc, argv, rc);

  if (!rc.modelsDir.empty())
    kokoro::paths::setModelsDir(rc.modelsDir);

  std::string espeakData;
  if (rc.espeakDataPath) {
    espeakData = std::filesystem::absolute(*rc.espeakDataPath).string();
  } else {
    espeakData = (kokoro::paths::exeDir() / "espeak-ng-data").string();
  }
  kokoro::Phonemizer::init(espeakData);

  std::string lexiconDir =
      rc.lexiconDir ? std::filesystem::absolute(*rc.lexiconDir).string()
                    : (kokoro::paths::exeDir() / "misaki-data").string();
  kokoro::G2P::init(lexiconDir, espeakData);

  g_cpuThreads = rc.cpuThreads;
  if (rc.synthThreads > 0) g_synthThreads = rc.synthThreads;
  g_pool.load(rc.accelerator);
  g_defaultVoice = rc.defaultVoice;

  if (const char* env = std::getenv("KOKORO_TOKEN")) {
    g_authToken = env;
    spdlog::info("Auth token from KOKORO_TOKEN env");
  } else if (const char* env = std::getenv("OPENAI_API_KEY")) {
    g_authToken = env;
    spdlog::info("Auth token from OPENAI_API_KEY env");
  } else if (!rc.authToken.empty()) {
    g_authToken = rc.authToken;
    spdlog::info("Auth token configured");
  }

  app().registerHandler(
      "/health",
      [](const HttpRequestPtr&,
         std::function<void(const HttpResponsePtr&)>&& cb) {
        auto r = HttpResponse::newHttpResponse();
        r->setStatusCode(k200OK);
        r->setContentTypeCode(CT_APPLICATION_JSON);
        r->setBody(R"({"status":"ok"})");
        cb(r);
      },
      {Get});

  if (!rc.disableWebUI) {
    std::filesystem::path webDir;
    if (rc.webRoot) {
      webDir = *rc.webRoot;
      if (!std::filesystem::exists(webDir)) {
        spdlog::error("--web-root {} does not exist", webDir.string());
        return 1;
      }
    } else {
      auto cwd = std::filesystem::current_path();
      auto root = kokoro::paths::projectRoot();
      for (const auto& candidate : {
               root / "server" / "web-content",
               cwd / "server" / "web-content",
               cwd / "web-content",
           }) {
        if (std::filesystem::exists(candidate)) {
          webDir = candidate;
          break;
        }
      }
    }
    if (!webDir.empty()) {
      app().setDocumentRoot(std::filesystem::absolute(webDir).string());
      spdlog::info("Web UI at http://{}:{}/  (root: {})", rc.ip, rc.port,
                   webDir.string());
    } else {
      spdlog::warn("Web UI requested but web-content/ not found "
                   "(pass --web-root DIR or --disable-web-ui)");
    }
  }

  app().addListener(rc.ip, rc.port).setThreadNum(3).run();

  kokoro::G2P::terminate();
  kokoro::Phonemizer::terminate();
  return 0;
}
