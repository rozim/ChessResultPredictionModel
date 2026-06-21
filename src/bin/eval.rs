//! chess-wdl-eval — evaluate a checkpoint, optionally against trivial baselines.

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Parser;

use chess_wdl::data::{prepare_pgn, read_shard_dir, PrepareFilter};
use chess_wdl::encoding::Sample;
use chess_wdl::metrics::{self, Metrics};
use chess_wdl::runtime::{load_model, predict_logits, select_device};

#[derive(Parser, Debug)]
#[command(name = "chess-wdl-eval", about = "Evaluate a WDL checkpoint")]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    /// Directory of eval shards.
    #[arg(long)]
    data: Option<PathBuf>,
    /// Or evaluate a raw PGN directly.
    #[arg(long)]
    pgn: Option<PathBuf>,
    /// With --pgn: only score positions at least this many plies in.
    #[arg(long, default_value_t = 0)]
    min_ply: u32,
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,
    /// Also report base-rate and material-logistic baselines.
    #[arg(long, default_value_t = false)]
    baseline: bool,
    #[arg(long, default_value = "metal")]
    device: String,
}

fn print_metrics(name: &str, m: &Metrics) {
    println!(
        "{name:<22} n={} log_loss={:.4} acc={:.4} brier={:.4} ece={:.4}",
        m.n, m.log_loss, m.accuracy, m.brier, m.ece
    );
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = select_device(args.device == "metal");
    let (model, _cfg, meta) = load_model(&args.checkpoint, &device)?;

    let samples: Vec<Sample> = match (&args.data, &args.pgn) {
        (Some(dir), _) => read_shard_dir(dir)?,
        (None, Some(pgn)) => {
            let filter = PrepareFilter {
                min_ply: args.min_ply,
                ..Default::default()
            };
            prepare_pgn(pgn, filter)?.0
        }
        _ => bail!("provide --data <shard dir> or --pgn <file>"),
    };
    if samples.is_empty() {
        bail!("no eval samples");
    }
    let labels: Vec<u8> = samples.iter().map(|s| s.wdl).collect();
    println!(
        "eval samples: {} | temperature T={:.3}",
        samples.len(),
        meta.temperature
    );

    // Model predictions (apply the stored calibration temperature).
    let logits = predict_logits(&model, &samples, args.batch_size)?;
    let probs = metrics::apply_temperature(&logits, meta.temperature);
    let m = metrics::evaluate(&probs, &labels);
    println!();
    print_metrics("model (calibrated)", &m);

    // Uncalibrated for reference.
    let probs_raw = metrics::apply_temperature(&logits, 1.0);
    print_metrics("model (T=1)", &metrics::evaluate(&probs_raw, &labels));

    // Memorization split: how does the model do on positions it saw in training
    // vs. genuinely novel ones? Uses the `seen` flag stamped by
    // `prepare --seen-against`. Skipped if the eval set carries no seen flags.
    let seen_idx: Vec<usize> = (0..samples.len()).filter(|&i| samples[i].seen).collect();
    let unseen_idx: Vec<usize> = (0..samples.len()).filter(|&i| !samples[i].seen).collect();
    let eval_subset = |idxs: &[usize]| {
        let p: Vec<[f32; 3]> = idxs.iter().map(|&i| probs[i]).collect();
        let l: Vec<u8> = idxs.iter().map(|&i| labels[i]).collect();
        metrics::evaluate(&p, &l)
    };
    if !seen_idx.is_empty() {
        println!(
            "\nmemorization split (calibrated): {}/{} seen in training ({:.1}%)",
            seen_idx.len(),
            samples.len(),
            100.0 * seen_idx.len() as f32 / samples.len() as f32
        );
        print_metrics("  seen (memorized?)", &eval_subset(&seen_idx));
        print_metrics("  unseen (novel)", &eval_subset(&unseen_idx));
    }

    // Game-phase breakdown: metrics by ply band (buckets of 20 half-moves),
    // with everything past MAX_BAND lumped into a final bucket.
    const BAND: u16 = 20;
    const MAX_BAND: u16 = 200;
    println!("\nply bands (calibrated):");
    let mut start = 0u16;
    while start < MAX_BAND {
        let end = start + BAND;
        let idxs: Vec<usize> = (0..samples.len())
            .filter(|&i| samples[i].ply >= start && samples[i].ply < end)
            .collect();
        if !idxs.is_empty() {
            print_metrics(&format!("  ply {start:>3}-{:<3}", end - 1), &eval_subset(&idxs));
        }
        start = end;
    }
    let tail: Vec<usize> = (0..samples.len())
        .filter(|&i| samples[i].ply >= MAX_BAND)
        .collect();
    if !tail.is_empty() {
        print_metrics(&format!("  ply {MAX_BAND}+"), &eval_subset(&tail));
    }

    if args.baseline {
        println!();
        let base = vec![meta.base_rate; samples.len()];
        print_metrics("baseline: base-rate", &metrics::evaluate(&base, &labels));

        if let Some(mat) = &meta.material {
            let mp: Vec<[f32; 3]> = samples
                .iter()
                .map(|s| mat.predict(s.material_balance()))
                .collect();
            print_metrics("baseline: material", &metrics::evaluate(&mp, &labels));
        }
    }

    // Confusion matrix (rows = true win/draw/loss, cols = predicted).
    println!("\nconfusion [true\\pred]   win  draw  loss");
    let names = ["win ", "draw", "loss"];
    for t in 0..3 {
        println!(
            "  {}              {:6} {:5} {:5}",
            names[t], m.confusion[t][0], m.confusion[t][1], m.confusion[t][2]
        );
    }
    Ok(())
}
