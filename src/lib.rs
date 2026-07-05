//! Chessformer/Maia-3-inspired win/draw/loss outcome predictor.
//!
//! See `DESIGN.md` for the architecture and rationale. The model predicts a
//! 3-way softmax over {win, draw, loss} from the side-to-move's perspective,
//! using only the current position (no history).

pub mod config;
pub mod data;
pub mod encoding;
pub mod fused;
pub mod metrics;
pub mod model;
pub mod runtime;

/// Link-time stub for Intel MKL's half-precision GEMM.
///
/// candle 0.10's CPU matmul references `hgemm_` (f16 BLAS) unconditionally, but
/// the `intel-mkl-src` 2020.1 static libs don't export it, so an MKL build fails
/// to link with `undefined symbol: hgemm_`. We only ever matmul in **f32**
/// (which resolves to `sgemm_`/`dgemm_`, both present), so the f16 path is never
/// executed — this stub just satisfies the linker. It must never be called.
#[cfg(not(target_os = "macos"))]
#[no_mangle]
pub extern "C" fn hgemm_() {
    unreachable!("hgemm_ (f16 MKL matmul) is stubbed out; f16 CPU matmul is unused");
}
