//! PGN ingestion and the on-disk shard format.
//!
//! `prepare_pgn` streams a PGN file, replays each game with shakmaty, and emits
//! one [`Sample`] per position (current position only — no history). Samples are
//! written to a compact fixed-record binary shard that loads via a single read.

use std::io::{BufWriter, Read, Write};
use std::ops::ControlFlow;
use std::path::Path;

use anyhow::{bail, Context, Result};
use pgn_reader::{RawTag, Reader, SanPlus, Visitor};
use shakmaty::{Chess, Color, Position};

use crate::encoding::{encode_position, wdl_label, GameResult, Sample, N_SQUARES};

/// Filters applied while reading a PGN. Mirrors the `prepare`/`eval` CLI flags.
#[derive(Debug, Clone)]
pub struct PrepareFilter {
    pub min_ply: u32,
    pub min_game_plies: u32,
    pub min_elo: u16,
    pub max_elo: u16,
    pub positions_per_game: Option<usize>,
    /// Sentinel rating used when a game is missing WhiteElo/BlackElo.
    pub default_elo: u16,
}

impl Default for PrepareFilter {
    fn default() -> Self {
        PrepareFilter {
            min_ply: 0,
            min_game_plies: 0,
            min_elo: 0,
            max_elo: 4000,
            positions_per_game: None,
            default_elo: 1500,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct PrepareStats {
    pub games_seen: usize,
    pub games_kept: usize,
    pub games_dropped_unlabeled: usize,
    pub games_dropped_filtered: usize,
    pub games_dropped_illegal: usize,
    pub samples: usize,
}

/// Per-game tag data we care about.
#[derive(Debug, Clone)]
struct GameMeta {
    result: Option<GameResult>,
    white_elo: Option<u16>,
    black_elo: Option<u16>,
}

/// Running movetext state: the live position plus the raw positions collected.
struct Movetext {
    pos: Chess,
    meta: GameMeta,
    /// (squares, castling, ep_file, side_to_move, ply) for each visited position.
    raw: Vec<([u8; N_SQUARES], u8, u8, Color, u32)>,
    plies: u32,
    valid: bool,
}

/// Visitor that accumulates finished [`Sample`]s across all games.
struct SampleCollector {
    filter: PrepareFilter,
    out: Vec<Sample>,
    stats: PrepareStats,
}

impl SampleCollector {
    fn new(filter: PrepareFilter) -> Self {
        SampleCollector {
            filter,
            out: Vec::new(),
            stats: PrepareStats::default(),
        }
    }

    fn record(&mut self, mt: &Movetext) {
        self.stats.games_seen += 1;

        let result = match mt.meta.result {
            Some(r) => r,
            None => {
                self.stats.games_dropped_unlabeled += 1;
                return;
            }
        };
        if !mt.valid {
            self.stats.games_dropped_illegal += 1;
            return;
        }

        let we = mt.meta.white_elo.unwrap_or(self.filter.default_elo);
        let be = mt.meta.black_elo.unwrap_or(self.filter.default_elo);
        let lo = we.min(be);
        let hi = we.max(be);
        if mt.plies < self.filter.min_game_plies
            || lo < self.filter.min_elo
            || hi > self.filter.max_elo
        {
            self.stats.games_dropped_filtered += 1;
            return;
        }

        // Candidate position indices passing the per-position ply filter.
        let mut idxs: Vec<usize> = (0..mt.raw.len())
            .filter(|&i| mt.raw[i].4 >= self.filter.min_ply)
            .collect();
        if let Some(cap) = self.filter.positions_per_game {
            if idxs.len() > cap && cap > 0 {
                // Evenly spaced subsample to decorrelate within a game.
                let step = idxs.len() as f64 / cap as f64;
                idxs = (0..cap).map(|k| idxs[(k as f64 * step) as usize]).collect();
            }
        }

        for i in idxs {
            let (squares, castling, ep_file, turn, _ply) = mt.raw[i];
            let (self_elo, oppo_elo) = if turn == Color::White {
                (we, be)
            } else {
                (be, we)
            };
            self.out.push(Sample {
                squares,
                castling,
                ep_file,
                self_elo,
                oppo_elo,
                wdl: wdl_label(turn, result),
            });
        }
        self.stats.games_kept += 1;
    }
}

fn parse_elo(value: &RawTag<'_>) -> Option<u16> {
    std::str::from_utf8(value.as_bytes())
        .ok()?
        .trim()
        .parse::<u16>()
        .ok()
}

impl Visitor for SampleCollector {
    type Tags = GameMeta;
    type Movetext = Movetext;
    type Output = ();

    fn begin_tags(&mut self) -> ControlFlow<(), GameMeta> {
        ControlFlow::Continue(GameMeta {
            result: None,
            white_elo: None,
            black_elo: None,
        })
    }

    fn tag(&mut self, tags: &mut GameMeta, name: &[u8], value: RawTag<'_>) -> ControlFlow<()> {
        match name {
            b"Result" => {
                tags.result = std::str::from_utf8(value.as_bytes())
                    .ok()
                    .and_then(GameResult::parse);
            }
            b"WhiteElo" => tags.white_elo = parse_elo(&value),
            b"BlackElo" => tags.black_elo = parse_elo(&value),
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, tags: GameMeta) -> ControlFlow<(), Movetext> {
        ControlFlow::Continue(Movetext {
            pos: Chess::default(),
            meta: tags,
            raw: Vec::new(),
            plies: 0,
            valid: true,
        })
    }

    fn san(&mut self, mt: &mut Movetext, san_plus: SanPlus) -> ControlFlow<()> {
        if !mt.valid {
            return ControlFlow::Continue(());
        }
        // Record the position *before* the move (side to move = mover).
        let (squares, castling, ep_file) = encode_position(&mt.pos);
        mt.raw
            .push((squares, castling, ep_file, mt.pos.turn(), mt.plies));

        match san_plus.san.to_move(&mt.pos) {
            Ok(m) => {
                mt.pos.play_unchecked(m);
                mt.plies += 1;
            }
            Err(_) => mt.valid = false, // malformed/illegal: stop recording this game
        }
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, mut mt: Movetext) -> () {
        // Include the terminal position (checkmate/stalemate or final position).
        if mt.valid {
            let (squares, castling, ep_file) = encode_position(&mt.pos);
            mt.raw
                .push((squares, castling, ep_file, mt.pos.turn(), mt.plies));
        }
        self.record(&mt);
    }
}

/// Stream a PGN file into `Sample`s, applying `filter`.
pub fn prepare_pgn(
    path: impl AsRef<Path>,
    filter: PrepareFilter,
) -> Result<(Vec<Sample>, PrepareStats)> {
    let file = std::fs::File::open(path.as_ref())
        .with_context(|| format!("opening PGN {:?}", path.as_ref()))?;
    let mut reader = Reader::new(std::io::BufReader::new(file));
    let mut collector = SampleCollector::new(filter);
    while reader.read_game(&mut collector)?.is_some() {}
    let stats = collector.stats.clone();
    let mut stats = stats;
    stats.samples = collector.out.len();
    Ok((collector.out, stats))
}

// ---------------------------------------------------------------------------
// Shard format: header + fixed 71-byte records.
//   magic "CWDL"(4) | version u32 | count u64    (16-byte header)
//   record: squares[64] castling[1] ep_file[1] self_elo[2] oppo_elo[2] wdl[1]
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"CWDL";
const VERSION: u32 = 1;
const RECORD_LEN: usize = N_SQUARES + 1 + 1 + 2 + 2 + 1; // 71

pub fn write_shard(path: impl AsRef<Path>, samples: &[Sample]) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())
        .with_context(|| format!("creating shard {:?}", path.as_ref()))?;
    let mut w = BufWriter::new(file);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(samples.len() as u64).to_le_bytes())?;
    let mut rec = [0u8; RECORD_LEN];
    for s in samples {
        rec[..N_SQUARES].copy_from_slice(&s.squares);
        rec[N_SQUARES] = s.castling;
        rec[N_SQUARES + 1] = s.ep_file;
        rec[N_SQUARES + 2..N_SQUARES + 4].copy_from_slice(&s.self_elo.to_le_bytes());
        rec[N_SQUARES + 4..N_SQUARES + 6].copy_from_slice(&s.oppo_elo.to_le_bytes());
        rec[N_SQUARES + 6] = s.wdl;
        w.write_all(&rec)?;
    }
    w.flush()?;
    Ok(())
}

pub fn read_shard(path: impl AsRef<Path>) -> Result<Vec<Sample>> {
    let mut f = std::fs::File::open(path.as_ref())
        .with_context(|| format!("opening shard {:?}", path.as_ref()))?;
    let mut header = [0u8; 16];
    f.read_exact(&mut header).context("reading shard header")?;
    if &header[..4] != MAGIC {
        bail!("bad shard magic in {:?}", path.as_ref());
    }
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if version != VERSION {
        bail!("unsupported shard version {version}");
    }
    let count = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    if buf.len() != count * RECORD_LEN {
        bail!(
            "shard body size mismatch: {} != {}",
            buf.len(),
            count * RECORD_LEN
        );
    }
    let mut out = Vec::with_capacity(count);
    for rec in buf.chunks_exact(RECORD_LEN) {
        let mut squares = [0u8; N_SQUARES];
        squares.copy_from_slice(&rec[..N_SQUARES]);
        out.push(Sample {
            squares,
            castling: rec[N_SQUARES],
            ep_file: rec[N_SQUARES + 1],
            self_elo: u16::from_le_bytes(rec[N_SQUARES + 2..N_SQUARES + 4].try_into().unwrap()),
            oppo_elo: u16::from_le_bytes(rec[N_SQUARES + 4..N_SQUARES + 6].try_into().unwrap()),
            wdl: rec[N_SQUARES + 6],
        });
    }
    Ok(out)
}

/// Load every `*.bin` shard in a directory into one vector.
pub fn read_shard_dir(dir: impl AsRef<Path>) -> Result<Vec<Sample>> {
    let mut paths: Vec<_> = std::fs::read_dir(dir.as_ref())
        .with_context(|| format!("reading shard dir {:?}", dir.as_ref()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "bin").unwrap_or(false))
        .collect();
    paths.sort();
    let mut all = Vec::new();
    for p in paths {
        all.extend(read_shard(p)?);
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(wdl: u8) -> Sample {
        let pos = Chess::default();
        let (squares, castling, ep_file) = encode_position(&pos);
        Sample {
            squares,
            castling,
            ep_file,
            self_elo: 2500,
            oppo_elo: 2400,
            wdl,
        }
    }

    #[test]
    fn shard_roundtrips() {
        let samples = vec![sample(0), sample(1), sample(2)];
        let dir = std::env::temp_dir().join(format!("cwdl-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.bin");
        write_shard(&path, &samples).unwrap();
        let back = read_shard(&path).unwrap();
        assert_eq!(samples, back);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn parses_a_tiny_pgn() {
        let pgn = "[Result \"1-0\"]\n[WhiteElo \"2500\"]\n[BlackElo \"2400\"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n";
        let dir = std::env::temp_dir().join(format!("cwdl-pgn-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("g.pgn");
        std::fs::write(&path, pgn).unwrap();

        let (samples, stats) = prepare_pgn(&path, PrepareFilter::default()).unwrap();
        assert_eq!(stats.games_kept, 1);
        // 7 plies played + 1 terminal position = 8 positions.
        assert_eq!(samples.len(), 8);
        // White to move at ply 0; White won => win label.
        assert_eq!(samples[0].wdl, crate::encoding::WDL_WIN);
        assert_eq!(samples[0].self_elo, 2500);
        // Terminal position: Black to move and checkmated => loss for Black.
        assert_eq!(samples.last().unwrap().wdl, crate::encoding::WDL_LOSS);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn drops_unlabeled_and_respects_min_ply() {
        let pgn = "[Result \"*\"]\n\n1. e4 e5 *\n";
        let dir = std::env::temp_dir().join(format!("cwdl-pgn2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("g.pgn");
        std::fs::write(&path, pgn).unwrap();
        let (samples, stats) = prepare_pgn(&path, PrepareFilter::default()).unwrap();
        assert_eq!(stats.games_dropped_unlabeled, 1);
        assert!(samples.is_empty());
        std::fs::remove_file(&path).ok();
    }
}
