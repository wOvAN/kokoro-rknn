#pragma once

namespace Ort {
struct CustomOpDomain;
}

namespace kokoro {

// com.rknn/RknnMatMul — encoder GEMMs on the NPU.
//
// build.py --npu-matmul rewrites the encoder MatMuls that multiply by a
// constant weight into this op; the kernel runs them through rknn_matmul_api
// (one fp16 GEMM per node, all three NPU cores) and falls back to cblas_sgemm
// for shapes the NPU path is not worth taking.
//
// The domain is a process-wide singleton: the OrtCustomOp it registers must
// outlive every session it is added to.
Ort::CustomOpDomain& rknnMatMulDomain();

} // namespace kokoro
