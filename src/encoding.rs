//! Position encoding: a chess position becomes a fixed-size `Sample`.
//!
//! The board is always rendered from the **side-to-move's perspective**
//! ("own pieces play up the board"): when it is Black to move we flip the board
//! vertically and swap piece colors, so the network sees a canonical frame.
//! No move history is used — a `Sample` is a pure function of the position.

use shakmaty::{CastlingSide, Color, EnPassantMode, Position, Role, Square};

pub const N_SQUARES: usize = 64;
/// 6 piece types x 2 colors (own / opponent).
pub const N_PIECE_PLANES: usize = 12;
/// Aux features: 4 castling flags + ep-present + ep-file (normalized).
pub const AUX_DIM: usize = 6;

/// Win/Draw/Loss class index, from the side-to-move's perspective.
pub const WDL_WIN: u8 = 0;
pub const WDL_DRAW: u8 = 1;
pub const WDL_LOSS: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    WhiteWin,
    BlackWin,
    Draw,
}

impl GameResult {
    /// Parse a PGN `Result` header. `*` (unknown) returns `None`.
    pub fn parse(s: &str) -> Option<GameResult> {
        match s.trim() {
            "1-0" => Some(GameResult::WhiteWin),
            "0-1" => Some(GameResult::BlackWin),
            "1/2-1/2" | "1/2 - 1/2" => Some(GameResult::Draw),
            _ => None,
        }
    }
}

/// WDL label for the player to move given the final game result.
pub fn wdl_label(turn: Color, result: GameResult) -> u8 {
    match (turn, result) {
        (Color::White, GameResult::WhiteWin) | (Color::Black, GameResult::BlackWin) => WDL_WIN,
        (_, GameResult::Draw) => WDL_DRAW,
        _ => WDL_LOSS,
    }
}

/// A single tensorizable training/eval example.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sample {
    /// Per-square piece codes in the side-to-move frame:
    /// 0 = empty, 1..=6 = own {P,N,B,R,Q,K}, 7..=12 = opponent {P,N,B,R,Q,K}.
    pub squares: [u8; N_SQUARES],
    /// bit0 own-king-side, bit1 own-queen-side, bit2 opp-king-side, bit3 opp-queen-side.
    pub castling: u8,
    /// 0 = no en passant, else file+1 (1..=8).
    pub ep_file: u8,
    pub self_elo: u16,
    pub oppo_elo: u16,
    /// WDL class (see `WDL_*`). 255 = unlabeled (inference only).
    pub wdl: u8,
}

fn role_index(r: Role) -> usize {
    match r {
        Role::Pawn => 0,
        Role::Knight => 1,
        Role::Bishop => 2,
        Role::Rook => 3,
        Role::Queen => 4,
        Role::King => 5,
    }
}

/// Encode a position into the side-to-move canonical frame (without labels/elo).
pub fn encode_position(pos: &shakmaty::Chess) -> ([u8; N_SQUARES], u8, u8) {
    let board = pos.board();
    let stm = pos.turn();
    let flip = stm == Color::Black;

    let mut squares = [0u8; N_SQUARES];
    for i in 0..N_SQUARES as u32 {
        let canonical = Square::new(i);
        // Physical square to read: flip vertically (rank) when Black to move.
        let phys = if flip {
            canonical.flip_vertical()
        } else {
            canonical
        };
        if let Some(piece) = board.piece_at(phys) {
            let base = role_index(piece.role) as u8 + 1;
            let own = piece.color == stm;
            squares[i as usize] = if own { base } else { base + 6 };
        }
    }

    let castles = pos.castles();
    let opp = stm.other();
    let mut castling = 0u8;
    if castles.has(stm, CastlingSide::KingSide) {
        castling |= 1 << 0;
    }
    if castles.has(stm, CastlingSide::QueenSide) {
        castling |= 1 << 1;
    }
    if castles.has(opp, CastlingSide::KingSide) {
        castling |= 1 << 2;
    }
    if castles.has(opp, CastlingSide::QueenSide) {
        castling |= 1 << 3;
    }

    // Vertical flip preserves file, so no horizontal adjustment is needed.
    let ep_file = match pos.ep_square(EnPassantMode::Legal) {
        Some(sq) => u8::from(sq.file()) + 1,
        None => 0,
    };

    (squares, castling, ep_file)
}

impl Sample {
    /// One-hot occupancy planes, row-major `[plane][square]`, length 12*64.
    pub fn planes_f32(&self) -> Vec<f32> {
        let mut v = vec![0f32; N_PIECE_PLANES * N_SQUARES];
        for (sq, &code) in self.squares.iter().enumerate() {
            if code != 0 {
                let plane = (code - 1) as usize; // 0..=11
                v[plane * N_SQUARES + sq] = 1.0;
            }
        }
        v
    }

    /// Auxiliary feature vector (length `AUX_DIM`).
    pub fn aux_f32(&self) -> [f32; AUX_DIM] {
        [
            (self.castling & 1) as f32,
            ((self.castling >> 1) & 1) as f32,
            ((self.castling >> 2) & 1) as f32,
            ((self.castling >> 3) & 1) as f32,
            if self.ep_file > 0 { 1.0 } else { 0.0 },
            if self.ep_file > 0 {
                (self.ep_file - 1) as f32 / 7.0
            } else {
                0.0
            },
        ]
    }

    /// Material balance (own minus opponent), classic 1/3/3/5/9 weights.
    /// Positive favors the side to move. Used by the material-logistic baseline.
    pub fn material_balance(&self) -> f32 {
        const VAL: [f32; 6] = [1.0, 3.0, 3.0, 5.0, 9.0, 0.0]; // P N B R Q K
        let mut bal = 0.0;
        for &code in self.squares.iter() {
            if code == 0 {
                continue;
            }
            if code <= 6 {
                bal += VAL[(code - 1) as usize];
            } else {
                bal -= VAL[(code - 7) as usize];
            }
        }
        bal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Chess;

    #[test]
    fn startpos_encodes_symmetrically() {
        let pos = Chess::default();
        let (sq, castling, ep) = encode_position(&pos);
        // 32 pieces present.
        assert_eq!(sq.iter().filter(|&&c| c != 0).count(), 32);
        // All four castling rights available.
        assert_eq!(castling, 0b1111);
        assert_eq!(ep, 0);
        // a1 (index 0) is own rook for White to move => code 4.
        assert_eq!(sq[0], 4);
    }

    #[test]
    fn black_to_move_is_canonicalized() {
        // After 1. e4, Black to move. In the canonical frame the side-to-move's
        // pieces (Black) occupy the low ranks just like White's did at startpos.
        let m = shakmaty::san::San::from_ascii(b"e4")
            .unwrap()
            .to_move(&Chess::default())
            .unwrap();
        let pos: Chess = Chess::default().play(m).unwrap();
        assert_eq!(pos.turn(), Color::Black);
        let (sq, _, _) = encode_position(&pos);
        // a1 in canonical frame is Black's a8 rook => own rook, code 4.
        assert_eq!(sq[0], 4);
        // Same number of own (1..=6) and opponent (7..=12) pieces: 16 each.
        assert_eq!(sq.iter().filter(|&&c| (1..=6).contains(&c)).count(), 16);
        assert_eq!(sq.iter().filter(|&&c| (7..=12).contains(&c)).count(), 16);
    }

    #[test]
    fn planes_have_one_hot_occupancy() {
        let pos = Chess::default();
        let (squares, castling, ep_file) = encode_position(&pos);
        let s = Sample {
            squares,
            castling,
            ep_file,
            self_elo: 1500,
            oppo_elo: 1500,
            wdl: 255,
        };
        let planes = s.planes_f32();
        assert_eq!(planes.len(), N_PIECE_PLANES * N_SQUARES);
        // Exactly 32 set bits, all 1.0.
        assert_eq!(planes.iter().filter(|&&x| x == 1.0).count(), 32);
        assert!((planes.iter().sum::<f32>() - 32.0).abs() < 1e-6);
    }

    #[test]
    fn startpos_material_is_balanced() {
        let pos = Chess::default();
        let (squares, castling, ep_file) = encode_position(&pos);
        let s = Sample {
            squares,
            castling,
            ep_file,
            self_elo: 1500,
            oppo_elo: 1500,
            wdl: 255,
        };
        assert!((s.material_balance()).abs() < 1e-6);
    }

    #[test]
    fn wdl_labels_follow_perspective() {
        assert_eq!(wdl_label(Color::White, GameResult::WhiteWin), WDL_WIN);
        assert_eq!(wdl_label(Color::Black, GameResult::WhiteWin), WDL_LOSS);
        assert_eq!(wdl_label(Color::Black, GameResult::BlackWin), WDL_WIN);
        assert_eq!(wdl_label(Color::White, GameResult::Draw), WDL_DRAW);
        assert_eq!(wdl_label(Color::Black, GameResult::Draw), WDL_DRAW);
    }

    #[test]
    fn result_parsing() {
        assert_eq!(GameResult::parse("1-0"), Some(GameResult::WhiteWin));
        assert_eq!(GameResult::parse("0-1"), Some(GameResult::BlackWin));
        assert_eq!(GameResult::parse("1/2-1/2"), Some(GameResult::Draw));
        assert_eq!(GameResult::parse("*"), None);
    }
}
