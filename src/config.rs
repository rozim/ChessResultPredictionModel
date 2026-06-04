//! Model architecture configuration, loaded from a TOML file (see `configs/`).

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub model: Model,
    #[serde(default)]
    pub gab: Gab,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub d_model: usize,
    pub layers: usize,
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub head_hidden: usize,
    #[serde(default)]
    pub dropout: f64,
    #[serde(default = "default_pos_encoding")]
    pub pos_encoding: PosEncoding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PosEncoding {
    /// Geometric Attention Bias (the paper's contribution; default).
    Gab,
    /// A single static per-head 64x64 learned bias (ablation / pipeline check).
    LearnedBias,
    /// No positional encoding.
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gab {
    #[serde(default = "default_true")]
    pub avg_pool: bool,
    #[serde(default)]
    pub per_square_dim: usize,
    #[serde(default = "default_compress")]
    pub compress_dim: usize,
    #[serde(default = "default_templates")]
    pub templates: usize,
}

impl Default for Gab {
    fn default() -> Self {
        Gab {
            avg_pool: true,
            per_square_dim: 0,
            compress_dim: 128,
            templates: 16,
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_compress() -> usize {
    128
}
fn default_templates() -> usize {
    16
}
fn default_pos_encoding() -> PosEncoding {
    PosEncoding::Gab
}

impl ModelConfig {
    pub fn from_toml_str(s: &str) -> Result<Self> {
        let cfg: ModelConfig = toml::from_str(s)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let s = std::fs::read_to_string(path.as_ref())
            .map_err(|e| anyhow::anyhow!("reading model config {:?}: {e}", path.as_ref()))?;
        Self::from_toml_str(&s)
    }

    pub fn to_toml_string(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    pub fn validate(&self) -> Result<()> {
        let m = &self.model;
        if m.d_model == 0
            || m.layers == 0
            || m.num_heads == 0
            || m.ffn_dim == 0
            || m.head_hidden == 0
        {
            bail!("model dimensions must all be > 0");
        }
        if m.d_model % m.num_heads != 0 {
            bail!(
                "num_heads ({}) must divide d_model ({})",
                m.num_heads,
                m.d_model
            );
        }
        if m.pos_encoding == PosEncoding::Gab {
            if self.gab.compress_dim == 0 || self.gab.templates == 0 {
                bail!("gab.compress_dim and gab.templates must be > 0");
            }
            if !self.gab.avg_pool && self.gab.per_square_dim == 0 {
                bail!("gab.per_square_dim must be > 0 when gab.avg_pool = false");
            }
        }
        Ok(())
    }

    /// Per-token input feature width fed to the input projection:
    /// 12 occupancy planes + 6 aux features. The board is the only signal —
    /// player Elo is deliberately not used (the corpus is filtered to a fixed
    /// strong-GM band, so Elo carries no information).
    pub fn input_dim(&self) -> usize {
        crate::encoding::N_PIECE_PLANES + crate::encoding::AUX_DIM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_tiny_config() {
        let cfg = ModelConfig::load("configs/tiny.toml").unwrap();
        assert_eq!(cfg.name, "tiny");
        assert_eq!(cfg.model.d_model, 128);
        assert_eq!(cfg.model.num_heads, 4);
        assert_eq!(cfg.model.pos_encoding, PosEncoding::Gab);
    }

    #[test]
    fn all_shipped_configs_valid() {
        for name in ["tiny", "small", "medium", "large"] {
            let cfg = ModelConfig::load(format!("configs/{name}.toml")).unwrap();
            cfg.validate().unwrap();
            assert_eq!(cfg.name, name);
        }
    }

    #[test]
    fn rejects_indivisible_heads() {
        let s = r#"
            name = "bad"
            [model]
            d_model = 100
            layers = 2
            num_heads = 7
            ffn_dim = 64
            head_hidden = 16
        "#;
        assert!(ModelConfig::from_toml_str(s).is_err());
    }

    #[test]
    fn roundtrips_through_toml() {
        let cfg = ModelConfig::load("configs/small.toml").unwrap();
        let s = cfg.to_toml_string().unwrap();
        let back = ModelConfig::from_toml_str(&s).unwrap();
        assert_eq!(back.model.d_model, cfg.model.d_model);
        assert_eq!(back.model.pos_encoding, cfg.model.pos_encoding);
    }
}
