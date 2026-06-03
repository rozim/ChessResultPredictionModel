//! chess-wdl-train — train the WDL model and save a calibrated checkpoint.

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{bail, Result};
use candle_core::DType;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use chess_wdl::config::ModelConfig;
use chess_wdl::data::read_shard_dir;
use chess_wdl::encoding::Sample;
use chess_wdl::metrics::{self, MaterialLogistic};
use chess_wdl::model::{Batch, ChessWdlModel};
use chess_wdl::runtime::{predict_logits, save_checkpoint, select_device, wdl_loss, Meta};

#[derive(Parser, Debug)]
#[command(name = "chess-wdl-train", about = "Train the WDL outcome model")]
struct Args {
    /// Model architecture TOML.
    #[arg(long, default_value = "configs/tiny.toml")]
    model_config: PathBuf,
    /// Directory of training shards.
    #[arg(long)]
    data: PathBuf,
    /// Directory of validation shards (drives early stopping).
    #[arg(long)]
    val_data: PathBuf,
    #[arg(long, default_value_t = 30)]
    epochs: usize,
    #[arg(long)]
    max_steps: Option<usize>,
    #[arg(long, default_value_t = 512)]
    batch_size: usize,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    #[arg(long, default_value_t = 0.05)]
    weight_decay: f64,
    #[arg(long, default_value_t = 200)]
    warmup_steps: usize,
    #[arg(long, default_value_t = 0.05)]
    label_smoothing: f32,
    #[arg(long, default_value_t = 10)]
    early_stop_patience: usize,
    /// Run validation every N optimizer steps.
    #[arg(long, default_value_t = 200)]
    val_interval: usize,
    #[arg(long, default_value_t = 50)]
    log_interval: usize,
    #[arg(long, default_value = "metal")]
    device: String,
    #[arg(long, default_value = "checkpoints/run")]
    checkpoint_dir: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn lr_at(step: usize, warmup: usize, total: usize, peak: f64) -> f64 {
    if warmup > 0 && step < warmup {
        peak * (step + 1) as f64 / warmup as f64
    } else {
        let denom = (total.saturating_sub(warmup)).max(1) as f64;
        let p = (step.saturating_sub(warmup)) as f64 / denom;
        peak * 0.5 * (1.0 + (std::f64::consts::PI * p.min(1.0)).cos())
    }
}

fn logits_to_probs(rows: &[[f32; 3]]) -> Vec<[f32; 3]> {
    rows.iter().map(|l| metrics::softmax3(*l)).collect()
}

fn val_log_loss(model: &ChessWdlModel, val: &[Sample], batch_size: usize) -> Result<f32> {
    let logits = predict_logits(model, val, batch_size)?;
    let probs = logits_to_probs(&logits);
    let labels: Vec<u8> = val.iter().map(|s| s.wdl).collect();
    Ok(metrics::evaluate(&probs, &labels).log_loss)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = ModelConfig::load(&args.model_config)?;
    println!("model `{}`: {:?}", cfg.name, cfg.model);

    let train = read_shard_dir(&args.data)?;
    let val = read_shard_dir(&args.val_data)?;
    if train.is_empty() {
        bail!("no training samples found in {:?}", args.data);
    }
    println!(
        "train samples: {} | val samples: {}",
        train.len(),
        val.len()
    );

    let device = select_device(args.device == "metal");
    println!("device: {:?}", device);

    let varmap = VarMap::new();
    let model = {
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        ChessWdlModel::new(&cfg, vb, &device)?
    };
    let n_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("parameters: {:.2}M", n_params as f64 / 1e6);

    // Early-stopping validation uses a fixed subset to keep val passes cheap;
    // the full val set is only used once at the end for temperature calibration.
    let val_es: Vec<Sample> = val.iter().take(16_384).cloned().collect();

    // Baselines (stored in the checkpoint for `eval --baseline`).
    let train_labels: Vec<u8> = train.iter().map(|s| s.wdl).collect();
    let base_rate = metrics::base_rate_probs(&train_labels);
    let feats: Vec<f32> = train.iter().map(|s| s.material_balance()).collect();
    let material = MaterialLogistic::fit(&feats, &train_labels, 300, 0.3);

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.lr,
            weight_decay: args.weight_decay,
            ..Default::default()
        },
    )?;

    let steps_per_epoch = train.len().div_ceil(args.batch_size);
    let total_steps = args
        .max_steps
        .unwrap_or(args.epochs * steps_per_epoch)
        .min(args.epochs * steps_per_epoch);
    println!("steps/epoch: {steps_per_epoch} | total steps: {total_steps}");

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut idx: Vec<usize> = (0..train.len()).collect();

    let mut step = 0usize;
    let mut best_val = f32::INFINITY;
    let mut patience = 0usize;
    let mut best_meta = Meta {
        base_rate,
        material: Some(material.clone()),
        ..Default::default()
    };
    'outer: for epoch in 0..args.epochs {
        idx.shuffle(&mut rng);
        for chunk in idx.chunks(args.batch_size) {
            let samples: Vec<Sample> = chunk.iter().map(|&i| train[i].clone()).collect();
            let batch = Batch::from_samples(&samples, &device)?;
            let logits = model.forward(&batch, true)?;
            let loss = wdl_loss(&logits, &batch.labels, args.label_smoothing)?;
            opt.set_learning_rate(lr_at(step, args.warmup_steps, total_steps, args.lr));
            opt.backward_step(&loss)?;

            if step % args.log_interval == 0 {
                let l = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
                println!(
                    "epoch {epoch} step {step}/{total_steps} lr {:.2e} loss {:.4}",
                    opt.learning_rate(),
                    l
                );
                std::io::stdout().flush().ok();
            }

            step += 1;
            let do_val = step % args.val_interval == 0 || step == total_steps;
            if do_val && !val_es.is_empty() {
                let vll = val_log_loss(&model, &val_es, args.batch_size)?;
                print!("  [val] step {step} log_loss {vll:.4}");
                if vll + 1e-5 < best_val {
                    best_val = vll;
                    patience = 0;
                    best_meta.steps = step;
                    best_meta.val_log_loss = vll;
                    save_checkpoint(&args.checkpoint_dir, &varmap, &cfg, &best_meta)?;
                    println!("  (best, saved)");
                } else {
                    patience += 1;
                    println!("  (no improve {patience}/{})", args.early_stop_patience);
                    if args.early_stop_patience > 0 && patience >= args.early_stop_patience {
                        println!("early stopping.");
                        break 'outer;
                    }
                }
            }
            if step >= total_steps {
                break 'outer;
            }
        }
    }

    // If no checkpoint was ever saved (e.g. no val), save the final state.
    if best_val.is_infinite() {
        save_checkpoint(&args.checkpoint_dir, &varmap, &cfg, &best_meta)?;
    } else {
        // Restore the best weights in place (Vars share storage with the model),
        // so calibration and the final save reflect the best checkpoint.
        let mut vm = varmap;
        vm.load(args.checkpoint_dir.join("model.safetensors"))?;
        let varmap = vm;
        // Calibrate: fit a temperature on the validation logits and persist it.
        if !val.is_empty() {
            let logits = predict_logits(&model, &val, args.batch_size)?;
            let labels: Vec<u8> = val.iter().map(|s| s.wdl).collect();
            let t = metrics::fit_temperature(&logits, &labels);
            best_meta.temperature = t;
            save_checkpoint(&args.checkpoint_dir, &varmap, &cfg, &best_meta)?;
            println!("fitted temperature T = {t:.3}");
        }
    }

    println!(
        "done. best val log_loss {:.4} -> checkpoint {:?}",
        best_val, args.checkpoint_dir
    );
    Ok(())
}
