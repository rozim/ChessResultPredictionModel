//! Chessformer/Maia-3-inspired win/draw/loss outcome predictor.
//!
//! See `DESIGN.md` for the architecture and rationale. The model predicts a
//! 3-way softmax over {win, draw, loss} from the side-to-move's perspective,
//! using only the current position (no history).

pub mod config;
pub mod data;
pub mod encoding;
pub mod fused;
pub mod metrics;
pub mod model;
pub mod runtime;
