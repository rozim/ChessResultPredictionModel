//! chess-wdl-prepare — convert PGN file(s) into compact training shards.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use chess_wdl::data::{prepare_pgn, write_shard, PrepareFilter};

#[derive(Parser, Debug)]
#[command(name = "chess-wdl-prepare", about = "PGN -> WDL training shards")]
struct Args {
    /// One or more PGN files.
    #[arg(long, required = true, num_args = 1..)]
    input: Vec<PathBuf>,
    /// Output directory for shards.
    #[arg(long)]
    output: PathBuf,
    /// Only keep positions at least this many plies into the game.
    #[arg(long, default_value_t = 0)]
    min_ply: u32,
    /// Drop entire games shorter than this many plies.
    #[arg(long, default_value_t = 0)]
    min_game_plies: u32,
    /// Drop games with either player rated below this.
    #[arg(long, default_value_t = 0)]
    min_elo: u16,
    /// Drop games with either player rated above this.
    #[arg(long, default_value_t = 4000)]
    max_elo: u16,
    /// Cap sampled positions per game (evenly spaced).
    #[arg(long)]
    positions_per_game: Option<usize>,
    /// Sentinel rating for games missing Elo headers.
    #[arg(long, default_value_t = 1500)]
    default_elo: u16,
    /// Positions per shard file.
    #[arg(long, default_value_t = 1_000_000)]
    shard_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output)?;

    let filter = PrepareFilter {
        min_ply: args.min_ply,
        min_game_plies: args.min_game_plies,
        min_elo: args.min_elo,
        max_elo: args.max_elo,
        positions_per_game: args.positions_per_game,
        default_elo: args.default_elo,
    };

    let mut all = Vec::new();
    for path in &args.input {
        println!("reading {:?} ...", path);
        let (samples, stats) = prepare_pgn(path, filter.clone())?;
        println!(
            "  games: {} kept / {} seen ({} unlabeled, {} filtered, {} illegal) -> {} samples",
            stats.games_kept,
            stats.games_seen,
            stats.games_dropped_unlabeled,
            stats.games_dropped_filtered,
            stats.games_dropped_illegal,
            samples.len()
        );
        all.extend(samples);
    }

    // Class distribution (sanity / draw-rate check).
    let mut dist = [0usize; 3];
    for s in &all {
        dist[s.wdl as usize] += 1;
    }
    let n = all.len().max(1) as f32;
    println!(
        "total samples: {} | win {:.1}% draw {:.1}% loss {:.1}%",
        all.len(),
        100.0 * dist[0] as f32 / n,
        100.0 * dist[1] as f32 / n,
        100.0 * dist[2] as f32 / n
    );

    for (i, chunk) in all.chunks(args.shard_size.max(1)).enumerate() {
        let path = args.output.join(format!("shard_{i:05}.bin"));
        write_shard(&path, chunk)?;
        println!("wrote {:?} ({} samples)", path, chunk.len());
    }
    Ok(())
}
