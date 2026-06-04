//! chess-wdl-predict — WDL probabilities for a position (FEN) or whole PGN.

use std::io::BufRead;
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Parser;
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess};

use chess_wdl::data::{prepare_pgn, PrepareFilter};
use chess_wdl::encoding::{encode_position, Sample};
use chess_wdl::metrics;
use chess_wdl::runtime::{load_model, predict_logits, select_device};

#[derive(Parser, Debug)]
#[command(name = "chess-wdl-predict", about = "Predict WDL for a position")]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    /// A single FEN to score.
    #[arg(long)]
    fen: Option<String>,
    /// Score every position reached in a PGN.
    #[arg(long)]
    pgn: Option<PathBuf>,
    /// With --pgn: only score positions at least this many plies in.
    #[arg(long, default_value_t = 0)]
    min_ply: u32,
    /// Read FENs from stdin, one per line.
    #[arg(long, default_value_t = false)]
    batch: bool,
    #[arg(long, default_value_t = false)]
    json: bool,
    #[arg(long, default_value = "metal")]
    device: String,
}

fn sample_from_fen(fen: &str) -> Result<Sample> {
    let pos: Chess =
        Fen::from_ascii(fen.trim().as_bytes())?.into_position(CastlingMode::Standard)?;
    let (squares, castling, ep_file) = encode_position(&pos);
    // Elo fields are unused by the model; kept for shard-format compatibility.
    Ok(Sample {
        squares,
        castling,
        ep_file,
        self_elo: 0,
        oppo_elo: 0,
        wdl: 255,
    })
}

fn emit(label: &str, p: [f32; 3], json: bool) {
    if json {
        println!(
            "{{\"id\":\"{label}\",\"win\":{:.4},\"draw\":{:.4},\"loss\":{:.4}}}",
            p[0], p[1], p[2]
        );
    } else {
        println!(
            "{label}: win {:.3}  draw {:.3}  loss {:.3}",
            p[0], p[1], p[2]
        );
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = select_device(args.device == "metal");
    let (model, _cfg, meta) = load_model(&args.checkpoint, &device)?;
    let t = meta.temperature;

    if let Some(fen) = &args.fen {
        let s = sample_from_fen(fen)?;
        let logits = predict_logits(&model, &[s], 1)?;
        emit("fen", metrics::apply_temperature(&logits, t)[0], args.json);
    } else if let Some(pgn) = &args.pgn {
        let filter = PrepareFilter {
            min_ply: args.min_ply,
            ..Default::default()
        };
        let (samples, _) = prepare_pgn(pgn, filter)?;
        let logits = predict_logits(&model, &samples, 1024)?;
        let probs = metrics::apply_temperature(&logits, t);
        for (i, p) in probs.iter().enumerate() {
            emit(&format!("ply{i}"), *p, args.json);
        }
    } else if args.batch {
        let stdin = std::io::stdin();
        let mut fens = Vec::new();
        for line in stdin.lock().lines() {
            let line = line?;
            if !line.trim().is_empty() {
                fens.push(line);
            }
        }
        let samples: Result<Vec<Sample>> = fens.iter().map(|f| sample_from_fen(f)).collect();
        let samples = samples?;
        let logits = predict_logits(&model, &samples, 1024)?;
        let probs = metrics::apply_temperature(&logits, t);
        for (f, p) in fens.iter().zip(probs.iter()) {
            emit(f, *p, args.json);
        }
    } else {
        bail!("provide --fen, --pgn, or --batch (stdin FENs)");
    }
    Ok(())
}
