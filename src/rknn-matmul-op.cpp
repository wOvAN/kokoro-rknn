#include "rknn-matmul-op.hpp"

#include <cblas.h>
#include <onnxruntime_cxx_api.h>
#include <rknn_api.h>
#include <rknn_matmul_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>

#include "fp16.hpp"

namespace kokoro {
namespace {

// The encoder's token count is dynamic, so one context per distinct length
// would grow the pool forever: M is rounded up to a bucket. Past kMMax the CPU
// sgemm is at NPU speed already and has no fp32->fp16 round trip, so it wins.
constexpr int kMBucket = 64;
constexpr int kMMax    = 512;

inline int mBucket(int m) {
  return (m + kMBucket - 1) / kMBucket * kMBucket;
}

// fp32 <-> fp16 over a whole tensor. On aarch64 the __fp16 cast compiles to
// vcvtq, which is a few GB/s; the portable bit-twiddling fallback is ~50x
// slower and only matters on the host. These loops are the bulk of the NPU
// path's overhead, so they are the ones worth vectorising.
inline void cvtF32ToF16(const float* s, void* d, size_t n) {
#if defined(__aarch64__)
  __fp16* o = static_cast<__fp16*>(d);
  for (size_t i = 0; i < n; ++i) o[i] = static_cast<__fp16>(s[i]);
#else
  uint16_t* o = static_cast<uint16_t*>(d);
  for (size_t i = 0; i < n; ++i) o[i] = fp32_to_fp16(s[i]);
#endif
}

inline void cvtF16ToF32(const void* s, float* d, size_t n) {
#if defined(__aarch64__)
  const __fp16* o = static_cast<const __fp16*>(s);
  for (size_t i = 0; i < n; ++i) d[i] = static_cast<float>(o[i]);
#else
  const uint16_t* o = static_cast<const uint16_t*>(s);
  for (size_t i = 0; i < n; ++i) d[i] = fp16_to_fp32(o[i]);
#endif
}

// One (K,N) weight shape: a dynamic-shape matmul context covering every M
// bucket, its three DMA buffers, and the packed fp16 weight.
struct Gemm {
  rknn_matmul_ctx ctx = 0;
  rknn_matmul_info info{};
  std::vector<rknn_matmul_shape> shapes;
  std::vector<rknn_matmul_io_attr> ios;
  rknn_tensor_mem *A = nullptr, *B = nullptr, *C = nullptr;
  std::vector<uint16_t> bpack;  // fp32 -> fp16 staging for the weight
  int K = 0, N = 0;
  int cur = -1;                 // bucket the context is currently set to
  const void* bsrc = nullptr;   // weight buffer already packed into B

  Gemm(int k, int n) : K(k), N(n), bpack(static_cast<size_t>(k) * n) {
    for (int m = kMBucket; m <= kMMax; m += kMBucket)
      shapes.push_back(rknn_matmul_shape{m, K, N});
    ios.resize(shapes.size());

    info.M        = kMMax;  // buffers are sized for the largest bucket
    info.K        = K;
    info.N        = N;
    info.type     = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
    info.B_layout = 1;      // native: ~13% faster than normal layout, measured
    int ret = rknn_matmul_create_dynamic_shape(&ctx, &info, static_cast<int>(shapes.size()),
                                               shapes.data(), ios.data());
    if (ret < 0) {
      ctx = 0;
      throw std::runtime_error("rknn_matmul_create_dynamic_shape failed: " + std::to_string(ret));
    }
    // No rknn_matmul_set_core_mask here on purpose: RK3588's matmul API rejects
    // an explicit multi-core mask ("Not support core mask: 7, fallback to
    // single core auto mode"), so the default AUTO mask is the multi-core one.

    uint32_t sa = 0, sb = 0, sc = 0;
    for (const auto& io : ios) {
      sa = std::max(sa, io.A.size);
      sb = std::max(sb, io.B.size);
      sc = std::max(sc, io.C.size);
    }
    A = rknn_create_mem(ctx, sa);
    B = rknn_create_mem(ctx, sb);
    C = rknn_create_mem(ctx, sc);
    if (!A || !B || !C) {
      destroy();
      throw std::runtime_error("rknn_create_mem failed for NPU matmul");
    }
  }

  Gemm(const Gemm&)            = delete;
  Gemm& operator=(const Gemm&) = delete;

  ~Gemm() { destroy(); }

  void destroy() {
    if (!ctx) return;
    if (A) rknn_destroy_mem(ctx, A);
    if (B) rknn_destroy_mem(ctx, B);
    if (C) rknn_destroy_mem(ctx, C);
    rknn_matmul_destroy(ctx);
    ctx = 0;
  }

  void run(const float* a, const float* b, int M, float* out) {
    const int m   = mBucket(M);
    const int idx = m / kMBucket - 1;
    if (idx != cur) {
      if (rknn_matmul_set_dynamic_shape(ctx, &shapes[idx]) < 0)
        throw std::runtime_error("rknn_matmul_set_dynamic_shape failed");
      cur = idx;
    }
    if (bsrc != b) {  // ALBERT shares weights, so each tensor is packed once
      cvtF32ToF16(b, bpack.data(), bpack.size());
      if (rknn_B_normal_layout_to_native_layout(bpack.data(), B->virt_addr, K, N, &info) < 0)
        throw std::runtime_error("rknn_B_normal_layout_to_native_layout failed");
      bsrc = b;
      if (rknn_matmul_set_io_mem(ctx, B, &ios[idx].B) < 0)
        throw std::runtime_error("rknn_matmul_set_io_mem(B) failed");
    }

    uint16_t* da = static_cast<uint16_t*>(A->virt_addr);
    cvtF32ToF16(a, da, static_cast<size_t>(M) * K);
    if (m > M)
      std::memset(da + static_cast<size_t>(M) * K, 0, static_cast<size_t>(m - M) * K * sizeof(uint16_t));

    if (rknn_matmul_set_io_mem(ctx, A, &ios[idx].A) < 0)
      throw std::runtime_error("rknn_matmul_set_io_mem(A) failed");
    if (rknn_matmul_set_io_mem(ctx, C, &ios[idx].C) < 0)
      throw std::runtime_error("rknn_matmul_set_io_mem(C) failed");
    if (rknn_matmul_run(ctx) < 0)
      throw std::runtime_error("rknn_matmul_run failed");

    cvtF16ToF32(C->virt_addr, out, static_cast<size_t>(M) * N);
  }
};

struct Cache {
  std::mutex mu;
  std::unordered_map<uint64_t, std::unique_ptr<Gemm>> gemms;
};

Cache& cache() {
  static Cache c;
  return c;
}

struct Kernel {
  OrtStatusPtr ComputeV2(OrtKernelContext* kc) {
    try {
      compute(kc);
      return nullptr;
    } catch (const std::exception& e) {
      return Ort::Status(e.what(), ORT_RUNTIME_EXCEPTION).release();
    }
  }

  void compute(OrtKernelContext* kc) {
    Ort::KernelContext ctx{kc};
    auto ain = ctx.GetInput(0);
    auto bin = ctx.GetInput(1);
    auto as  = ain.GetTensorTypeAndShapeInfo().GetShape();
    auto bs  = bin.GetTensorTypeAndShapeInfo().GetShape();

    int64_t M = 1;
    for (size_t i = 0; i + 1 < as.size(); ++i) M *= as[i];
    const int64_t K = as.empty() ? 0 : as.back();
    const int64_t N = bs.empty() ? 0 : bs.back();

    std::vector<int64_t> od(as.begin(), as.end());
    if (!od.empty()) od.back() = N;
    auto outv = ctx.GetOutput(0, od);
    float*       out = outv.GetTensorMutableData<float>();
    const float* pa  = ain.GetTensorData<float>();
    const float* pb  = bin.GetTensorData<float>();

    // Anything the NPU path handles badly (tiny/large M, tile-misaligned K/N)
    // runs on the CPU, exactly as it did before this op existed.
    if (M < kMBucket || M > kMMax || K <= 0 || K % 32 || K > 10240 || N % 16) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(M), static_cast<int>(N),
                  static_cast<int>(K), 1.0f, pa, static_cast<int>(K), pb, static_cast<int>(N), 0.0f, out,
                  static_cast<int>(N));
      return;
    }

    Cache&                   c = cache();
    std::lock_guard<std::mutex> lk(c.mu);  // one NPU GEMM in flight, process-wide
    const uint64_t             key = (static_cast<uint64_t>(K) << 32) | static_cast<uint64_t>(N);
    auto                     it = c.gemms.find(key);
    if (it == c.gemms.end()) {
      spdlog::info("NPU matmul: {}x{} GEMM (M bucketed to {})", K, N, kMMax);
      it = c.gemms.emplace(key, std::make_unique<Gemm>(static_cast<int>(K), static_cast<int>(N))).first;
    }
    it->second->run(pa, pb, static_cast<int>(M), out);
  }
};

struct Op : Ort::CustomOpBase<Op, Kernel, true> {
  const char* GetName() const { return "RknnMatMul"; }
  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }

  OrtStatusPtr CreateKernelV2(const OrtApi&, const OrtKernelInfo*, void** kernel) const {
    *kernel = new Kernel();
    return nullptr;
  }
};

} // namespace

Ort::CustomOpDomain& rknnMatMulDomain() {
  static Ort::CustomOpDomain domain{"com.rknn"};
  static Op                  op;
  static bool                registered = false;
  if (!registered) {
    domain.Add(&op);
    registered = true;
  }
  return domain;
}

} // namespace kokoro
