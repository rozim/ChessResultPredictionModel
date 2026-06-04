//! Shared runtime helpers: device selection, checkpoints, loss, inference.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

use crate::config::ModelConfig;
use crate::encoding::Sample;
use crate::metrics::MaterialLogistic;
use crate::model::{Batch, ChessWdlModel};

/// Pick the Metal GPU when requested and available, else CPU.
pub fn select_device(prefer_metal: bool) -> Device {
    if prefer_metal {
        match Device::new_metal(0) {
            Ok(d) => return d,
            Err(e) => eprintln!("metal unavailable ({e}); falling back to CPU"),
        }
    }
    Device::Cpu
}

/// Sidecar metadata stored alongside the weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub temperature: f32,
    pub steps: usize,
    pub val_log_loss: f32,
    pub base_rate: [f32; 3],
    pub material: Option<MaterialLogistic>,
}

impl Default for Meta {
    fn default() -> Self {
        Meta {
            temperature: 1.0,
            steps: 0,
            val_log_loss: f32::INFINITY,
            base_rate: [1.0 / 3.0; 3],
            material: None,
        }
    }
}

const WEIGHTS: &str = "model.safetensors";
const CONFIG: &str = "model.toml";
const META: &str = "meta.json";

pub fn save_checkpoint(
    dir: impl AsRef<Path>,
    varmap: &VarMap,
    cfg: &ModelConfig,
    meta: &Meta,
) -> Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    varmap.save(dir.join(WEIGHTS)).context("saving weights")?;
    std::fs::write(dir.join(CONFIG), cfg.to_toml_string()?)?;
    std::fs::write(dir.join(META), serde_json::to_string_pretty(meta)?)?;
    Ok(())
}

/// Rebuild a model from a checkpoint directory and load its weights.
pub fn load_model(
    dir: impl AsRef<Path>,
    device: &Device,
) -> Result<(ChessWdlModel, ModelConfig, Meta)> {
    let dir = dir.as_ref();
    let cfg = ModelConfig::load(dir.join(CONFIG))?;
    let mut varmap = VarMap::new();
    let model = {
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        ChessWdlModel::new(&cfg, vb, device)?
    };
    varmap
        .load(dir.join(WEIGHTS))
        .with_context(|| format!("loading weights from {dir:?}"))?;
    let meta: Meta = match std::fs::read_to_string(dir.join(META)) {
        Ok(s) => serde_json::from_str(&s)?,
        Err(_) => Meta::default(),
    };
    Ok((model, cfg, meta))
}

/// Numerically-stable log-softmax over the last dim from primitive ops.
///
/// Candle's fused `log_softmax`/`cross_entropy` have no Metal kernel, so we
/// build the loss from primitives that do run on the M1 GPU.
fn log_softmax_lastdim(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let shifted = x.broadcast_sub(&max)?;
    let sumexp = shifted.exp()?.sum_keepdim(D::Minus1)?;
    Ok(shifted.broadcast_sub(&sumexp.log()?)?)
}

/// WDL cross-entropy with optional label smoothing. `targets`: [B] u32.
pub fn wdl_loss(
    logits: &Tensor,
    targets: &Tensor,
    smoothing: f32,
    class_weights: Option<&Tensor>,
) -> Result<Tensor> {
    let logp = log_softmax_lastdim(logits)?; // [B, 3]
    let picked = logp.gather(&targets.unsqueeze(1)?, 1)?.squeeze(1)?; // [B] log p[y]
    let per = if smoothing <= 0.0 {
        picked.neg()?
    } else {
        // Smoothing mixes in a uniform target: eps * mean over classes of -log p.
        let uniform = logp.mean(D::Minus1)?.neg()?; // [B]
        ((picked.neg()? * (1.0 - smoothing) as f64)? + (uniform * smoothing as f64)?)?
    };
    match class_weights {
        None => Ok(per.mean(0)?),
        Some(w) => {
            // Weighted mean: Σ(w[y]·loss) / Σ(w[y]).
            let wp = w.index_select(targets, 0)?; // [B]
            let num = (per * &wp)?.sum(0)?;
            let den = wp.sum(0)?;
            Ok(num.broadcast_div(&den)?)
        }
    }
}

/// Raw [B,3] logits for many samples, in batches, on the model's device.
pub fn predict_logits(
    model: &ChessWdlModel,
    samples: &[Sample],
    batch_size: usize,
) -> Result<Vec<[f32; 3]>> {
    let mut out = Vec::with_capacity(samples.len());
    for chunk in samples.chunks(batch_size.max(1)) {
        let batch = Batch::from_samples(chunk, model.device())?;
        let logits = model.forward(&batch, false)?.to_dtype(DType::F32)?;
        for r in logits.to_vec2::<f32>()? {
            out.push([r[0], r[1], r[2]]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn class_weights_change_the_loss() {
        let dev = Device::Cpu;
        // Sample 0 (label 0): confident & correct -> low loss.
        // Sample 1 (label 1): uniform logits -> high loss (ln 3).
        // Distinct per-sample losses, so class weighting shifts the mean.
        let logits = Tensor::from_vec(vec![5f32, 0., 0., 0., 0., 0.], (2, 3), &dev).unwrap();
        let targets = Tensor::from_vec(vec![0u32, 1], 2, &dev).unwrap();
        let unweighted = wdl_loss(&logits, &targets, 0.0, None)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        // Up-weight class 1 by 5x -> the weighted mean shifts toward sample 1's loss.
        let w = Tensor::from_vec(vec![1f32, 5., 1.], 3, &dev).unwrap();
        let weighted = wdl_loss(&logits, &targets, 0.0, Some(&w))
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(unweighted.is_finite() && weighted.is_finite());
        assert!(
            (unweighted - weighted).abs() > 1e-5,
            "weighting had no effect"
        );
    }
}
