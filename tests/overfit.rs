//! Overfit / memorization tests.
//!
//! The strongest end-to-end sanity check for a training stack: a model with
//! enough capacity must be able to drive the loss to ~0 on a handful of fixed
//! examples. These exercise the whole path — encoding → `Batch` → model forward
//! → `wdl_loss` → AdamW step — on CPU, and assert the model can perfectly
//! memorize (1) a single example and (2) a 16-sample batch.

use candle_core::{DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use chess_wdl::config::ModelConfig;
use chess_wdl::encoding::{encode_position, Sample};
use chess_wdl::metrics::argmax;
use chess_wdl::model::{logits_to_probs, Batch, ChessWdlModel};
use chess_wdl::runtime::wdl_loss;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use shakmaty::{Chess, Position};

/// `n` diverse, distinct positions from seeded random playouts of increasing
/// length, with labels cycling 0/1/2 so the batch covers all three classes.
fn distinct_samples(n: usize) -> Vec<Sample> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut pos = Chess::default();
        let plies = 6 + 2 * i; // varied lengths -> diverse boards
        for _ in 0..plies {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            let idx = rng.random_range(0..moves.len());
            pos.play_unchecked(moves[idx]);
        }
        let (squares, castling, ep_file) = encode_position(&pos);
        out.push(Sample {
            squares,
            castling,
            ep_file,
            self_elo: 2400,
            oppo_elo: 2400,
            wdl: (i % 3) as u8,
        });
    }
    out
}

/// Train `config` on `samples` until it memorizes them, then assert the
/// (dropout-free) predictions match every label and the loss is ~0.
fn assert_memorizes(config: &str, samples: &[Sample], steps: usize, lr: f64, max_loss: f32) {
    let device = Device::Cpu;
    let cfg = ModelConfig::load(config).unwrap_or_else(|e| panic!("load {config}: {e}"));

    let varmap = VarMap::new();
    let model = {
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        ChessWdlModel::new(&cfg, vb, &device).expect("build model")
    };
    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr,
            ..Default::default()
        },
    )
    .expect("optimizer");

    let batch = Batch::from_samples(samples, &device).expect("batch");
    for _ in 0..steps {
        let logits = model.forward(&batch, true).expect("forward");
        let loss = wdl_loss(&logits, &batch.labels, 0.0, None).expect("loss");
        opt.backward_step(&loss).expect("step");
    }

    // Evaluate with dropout off.
    let logits = model.forward(&batch, false).expect("eval forward");
    let eval_loss = wdl_loss(&logits, &batch.labels, 0.0, None)
        .expect("eval loss")
        .to_scalar::<f32>()
        .expect("scalar");
    let probs = logits_to_probs(&logits).expect("probs");

    for (i, (p, s)) in probs.iter().zip(samples.iter()).enumerate() {
        assert_eq!(
            argmax(p),
            s.wdl as usize,
            "sample {i}: predicted {:?} for label {} (probs {p:?})",
            argmax(p),
            s.wdl
        );
    }
    assert!(
        eval_loss < max_loss,
        "expected memorized loss < {max_loss}, got {eval_loss}"
    );
}

#[test]
fn nano_memorizes_single_example() {
    assert_memorizes("configs/nano.toml", &distinct_samples(1), 200, 1e-2, 0.02);
}

#[test]
fn nano_memorizes_batch_of_16() {
    let samples = distinct_samples(16);
    assert_eq!(samples.len(), 16);
    assert_memorizes("configs/nano.toml", &samples, 800, 1e-2, 0.10);
}

#[test]
fn tiny_memorizes_single_example() {
    assert_memorizes("configs/tiny.toml", &distinct_samples(1), 200, 1e-2, 0.02);
}

#[test]
fn tiny_memorizes_batch_of_16() {
    let samples = distinct_samples(16);
    assert_eq!(samples.len(), 16);
    // The deeper 0.93M model needs a gentler LR and more steps than nano.
    assert_memorizes("configs/tiny.toml", &samples, 1500, 3e-3, 0.10);
}
