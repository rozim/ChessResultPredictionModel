//! Fused, rayon-parallel CPU ops for the hot element-wise kernels.
//!
//! Candle's CPU element-wise ops run single-threaded, and its *fused*
//! `softmax`/`layer_norm` custom ops are inference-only (no backward) and have
//! no Metal kernel. These `CustomOp1`s give a **multi-threaded fused forward**
//! plus a correct **differentiable backward** (composed from primitive ops), so
//! they speed up training and inference on CPU. They are CPU-only — callers must
//! fall back to primitive ops on Metal (see `model.rs`).

use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor, D};
use rayon::prelude::*;

/// Softmax over the last dimension.
struct FusedSoftmaxLastDim;

impl CustomOp1 for FusedSoftmaxLastDim {
    fn name(&self) -> &'static str {
        "fused-softmax-lastdim"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let data = match storage {
            CpuStorage::F32(d) => d,
            _ => candle_core::bail!("fused-softmax: expected f32"),
        };
        let dims = layout.dims();
        let ncols = *dims.last().unwrap();
        let n: usize = dims.iter().product();
        let off = layout.start_offset();
        let src = &data[off..off + n];
        let mut out = vec![0f32; n];
        out.par_chunks_mut(ncols)
            .zip(src.par_chunks(ncols))
            .for_each(|(o, s)| {
                let mut m = f32::NEG_INFINITY;
                for &v in s {
                    if v > m {
                        m = v;
                    }
                }
                let mut sum = 0f32;
                for (oi, &v) in o.iter_mut().zip(s) {
                    let e = (v - m).exp();
                    *oi = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for oi in o.iter_mut() {
                    *oi *= inv;
                }
            });
        Ok((CpuStorage::F32(out), layout.shape().clone()))
    }

    fn bwd(&self, _arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // softmax JVP: g = y ⊙ (dy − Σ(dy ⊙ y))
        let dot = (res * grad_res)?.sum_keepdim(D::Minus1)?;
        let grad = (res * grad_res.broadcast_sub(&dot)?)?;
        Ok(Some(grad))
    }
}

/// Standardize over the last dimension: `(x − mean) / sqrt(var + eps)` (no affine).
struct FusedLayerNormStandardize {
    eps: f32,
}

impl CustomOp1 for FusedLayerNormStandardize {
    fn name(&self) -> &'static str {
        "fused-layernorm-standardize"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let data = match storage {
            CpuStorage::F32(d) => d,
            _ => candle_core::bail!("fused-layernorm: expected f32"),
        };
        let dims = layout.dims();
        let ncols = *dims.last().unwrap();
        let n: usize = dims.iter().product();
        let off = layout.start_offset();
        let src = &data[off..off + n];
        let eps = self.eps;
        let inv_n = 1.0 / ncols as f32;
        let mut out = vec![0f32; n];
        out.par_chunks_mut(ncols)
            .zip(src.par_chunks(ncols))
            .for_each(|(o, s)| {
                let mut mean = 0f32;
                for &v in s {
                    mean += v;
                }
                mean *= inv_n;
                let mut var = 0f32;
                for &v in s {
                    let d = v - mean;
                    var += d * d;
                }
                var *= inv_n;
                let inv_std = 1.0 / (var + eps).sqrt();
                for (oi, &v) in o.iter_mut().zip(s) {
                    *oi = (v - mean) * inv_std;
                }
            });
        Ok((CpuStorage::F32(out), layout.shape().clone()))
    }

    fn bwd(&self, arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // xhat = res; recompute inv_std from arg. With N = last dim:
        // dx = inv_std * (dy − mean(dy) − xhat ⊙ mean(dy ⊙ xhat))
        let mean = arg.mean_keepdim(D::Minus1)?;
        let xc = arg.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let inv_std = (var + self.eps as f64)?.sqrt()?.recip()?;
        let mean_g = grad_res.mean_keepdim(D::Minus1)?;
        let mean_gx = (grad_res * res)?.mean_keepdim(D::Minus1)?;
        let inner = grad_res
            .broadcast_sub(&mean_g)?
            .sub(&res.broadcast_mul(&mean_gx)?)?;
        let grad = inner.broadcast_mul(&inv_std)?;
        Ok(Some(grad))
    }
}

/// Fused softmax over the last dim (CPU only — input must be on the CPU device).
pub fn softmax_lastdim_cpu(x: &Tensor) -> Result<Tensor> {
    x.contiguous()?.apply_op1(FusedSoftmaxLastDim)
}

/// Fused standardize (mean/var normalize) over the last dim (CPU only).
pub fn layernorm_standardize_cpu(x: &Tensor, eps: f32) -> Result<Tensor> {
    x.contiguous()?.apply_op1(FusedLayerNormStandardize { eps })
}
