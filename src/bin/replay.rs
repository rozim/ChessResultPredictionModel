//! chess-wdl-replay — per-move WDL predictions for every game in a PGN.
//!
//! Replays each game move by move and, for the position *before* each move,
//! prints the model's win/draw/loss prediction **from the side-to-move's
//! perspective** (the same framing the model was trained on). Each game shows
//! its final result and flags the single most confident prediction.

use std::ops::ControlFlow;
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Parser;
use pgn_reader::{RawTag, Reader, SanPlus, Visitor};
use shakmaty::{Chess, Color, Position};

use chess_wdl::encoding::{encode_position, Sample};
use chess_wdl::metrics::{self, argmax};
use chess_wdl::runtime::{load_model, predict_logits, select_device};

#[derive(Parser, Debug)]
#[command(name = "chess-wdl-replay", about = "Per-move WDL predictions for a PGN")]
struct Args {
    /// Trained checkpoint directory.
    #[arg(long)]
    checkpoint: PathBuf,
    /// PGN file to replay.
    #[arg(long)]
    pgn: PathBuf,
    /// Stop after this many games.
    #[arg(long)]
    max_games: Option<usize>,
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,
    #[arg(long, default_value = "metal")]
    device: String,
}

/// One scored position: the move played from it and the side to move.
struct Row {
    ply: u32,
    mv: String,
    turn: Color,
    sample: Sample,
}

#[derive(Default)]
struct Tags {
    white: Option<String>,
    black: Option<String>,
    result: Option<String>,
}

/// Per-game accumulator (also the pgn-reader `Movetext` type).
struct Game {
    white: String,
    black: String,
    result: String,
    rows: Vec<Row>,
    pos: Chess,
    ply: u32,
    valid: bool,
}

#[derive(Default)]
struct Collector {
    games: Vec<Game>,
}

fn tag_string(value: &RawTag<'_>) -> String {
    String::from_utf8_lossy(value.as_bytes()).trim().to_string()
}

/// "12. Nf3" for White's move, "12... Nf6" for Black's (ply is 0-based).
fn move_label(ply: u32, san: &str) -> String {
    let n = ply / 2 + 1;
    if ply.is_multiple_of(2) {
        format!("{n}. {san}")
    } else {
        format!("{n}... {san}")
    }
}

fn class_name(cls: usize) -> &'static str {
    ["win", "draw", "loss"][cls]
}

/// Which color a side-to-move prediction favors, in absolute terms.
fn favored(turn: Color, cls: usize) -> &'static str {
    match cls {
        0 => {
            if turn == Color::White {
                "White"
            } else {
                "Black"
            }
        }
        2 => {
            if turn == Color::White {
                "Black"
            } else {
                "White"
            }
        }
        _ => "draw",
    }
}

impl Visitor for Collector {
    type Tags = Tags;
    type Movetext = Game;
    type Output = ();

    fn begin_tags(&mut self) -> ControlFlow<(), Tags> {
        ControlFlow::Continue(Tags::default())
    }

    fn tag(&mut self, tags: &mut Tags, name: &[u8], value: RawTag<'_>) -> ControlFlow<()> {
        match name {
            b"White" => tags.white = Some(tag_string(&value)),
            b"Black" => tags.black = Some(tag_string(&value)),
            b"Result" => tags.result = Some(tag_string(&value)),
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, tags: Tags) -> ControlFlow<(), Game> {
        ControlFlow::Continue(Game {
            white: tags.white.unwrap_or_else(|| "?".into()),
            black: tags.black.unwrap_or_else(|| "?".into()),
            result: tags.result.unwrap_or_else(|| "*".into()),
            rows: Vec::new(),
            pos: Chess::default(),
            ply: 0,
            valid: true,
        })
    }

    fn san(&mut self, g: &mut Game, san_plus: SanPlus) -> ControlFlow<()> {
        if !g.valid {
            return ControlFlow::Continue(());
        }
        let (squares, castling, ep_file) = encode_position(&g.pos);
        let turn = g.pos.turn();
        let mv = move_label(g.ply, &san_plus.to_string());
        match san_plus.san.to_move(&g.pos) {
            Ok(m) => {
                g.rows.push(Row {
                    ply: g.ply,
                    mv,
                    turn,
                    sample: Sample {
                        squares,
                        castling,
                        ep_file,
                        self_elo: 0,
                        oppo_elo: 0,
                        wdl: 255,
                        ply: g.ply.min(u16::MAX as u32) as u16,
                        seen: false,
                    },
                });
                g.pos.play_unchecked(m);
                g.ply += 1;
            }
            Err(_) => g.valid = false, // stop at the first unparseable move
        }
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, game: Game) {
        self.games.push(game);
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = select_device(args.device == "metal");
    let (model, _cfg, meta) = load_model(&args.checkpoint, &device)?;
    let t = meta.temperature;

    let file = std::fs::File::open(&args.pgn)?;
    let mut reader = Reader::new(std::io::BufReader::new(file));
    let mut collector = Collector::default();
    while reader.read_game(&mut collector)?.is_some() {
        if let Some(mg) = args.max_games {
            if collector.games.len() >= mg {
                break;
            }
        }
    }
    if collector.games.is_empty() {
        bail!("no games found in {:?}", args.pgn);
    }
    println!(
        "loaded {} game(s) from {:?} | temperature T={:.3}",
        collector.games.len(),
        args.pgn,
        t
    );

    for (gi, g) in collector.games.iter().enumerate() {
        println!(
            "\n=== Game {} : {} vs {}  [result {}] ===",
            gi + 1,
            g.white,
            g.black,
            g.result
        );
        if g.rows.is_empty() {
            println!("  (no moves)");
            continue;
        }

        let samples: Vec<Sample> = g.rows.iter().map(|r| r.sample.clone()).collect();
        let logits = predict_logits(&model, &samples, args.batch_size)?;
        let probs = metrics::apply_temperature(&logits, t);

        println!(
            " {:>4}  {:<11} {:^3}  {:>5} {:>5} {:>5}  {:<4} {:>5}",
            "ply", "move", "stm", "win", "draw", "loss", "pred", "conf"
        );
        let mut best = 0usize;
        for (i, (r, p)) in g.rows.iter().zip(probs.iter()).enumerate() {
            let cls = argmax(p);
            if p[cls] > probs[best][argmax(&probs[best])] {
                best = i;
            }
            println!(
                " {:>4}  {:<11} {:^3}  {:>5.3} {:>5.3} {:>5.3}  {:<4} {:>5.3}",
                r.ply,
                r.mv,
                if r.turn == Color::White { "w" } else { "b" },
                p[0],
                p[1],
                p[2],
                class_name(cls),
                p[cls],
            );
        }

        let r = &g.rows[best];
        let p = &probs[best];
        let cls = argmax(p);
        println!(
            " most confident: {} (ply {})  conf {:.3} -> {} ({})  | game result {}",
            r.mv,
            r.ply,
            p[cls],
            class_name(cls),
            favored(r.turn, cls),
            g.result,
        );
    }
    Ok(())
}
