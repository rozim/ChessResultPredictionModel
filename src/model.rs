//! The Chessformer-style encoder and WDL value head, built on Candle.
//!
//! Encoder-only transformer over 64 square-tokens. Positional information comes
//! from Geometric Attention Bias (GAB): a content-dependent additive bias on the
//! attention logits. Output is a 3-way softmax over {win, draw, loss}.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear, Dropout, Init, Linear, Module, VarBuilder};

use crate::config::{ModelConfig, PosEncoding};
use crate::encoding::{Sample, AUX_DIM, N_PIECE_PLANES, N_SQUARES};

/// Softmax over the last dim. On CPU it uses the fused, rayon-parallel op
/// (`crate::fused`); on Metal it falls back to primitive ops (Metal-safe).
fn softmax_lastdim(x: &Tensor) -> Result<Tensor> {
    if matches!(x.device(), Device::Cpu) {
        return crate::fused::softmax_lastdim_cpu(x);
    }
    let max = x.max_keepdim(D::Minus1)?;
    let e = x.broadcast_sub(&max)?.exp()?;
    let sum = e.sum_keepdim(D::Minus1)?;
    e.broadcast_div(&sum)
}

/// Layer normalization over the last dimension, built from primitive ops only.
///
/// Candle's Metal backend has no fused layer-norm kernel, so we compose it from
/// mean/var/sqrt/broadcast ops (all of which run on the M1 GPU).
struct Ln {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl Ln {
    fn new(d: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(d, "weight", Init::Const(1.0))?;
        let bias = vb.get_with_hints(d, "bias", Init::Const(0.0))?;
        Ok(Ln {
            weight,
            bias,
            eps: 1e-5,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Standardize (mean/var normalize) over the last dim, then affine.
        let xn = if matches!(x.device(), Device::Cpu) {
            crate::fused::layernorm_standardize_cpu(x, self.eps as f32)?
        } else {
            let mean = x.mean_keepdim(D::Minus1)?;
            let xc = x.broadcast_sub(&mean)?;
            let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
            xc.broadcast_div(&(var + self.eps)?.sqrt()?)?
        };
        xn.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

/// A tensorized minibatch on a device. The board is the only model input —
/// player Elo is deliberately not used (corpus is a fixed strong-GM band).
pub struct Batch {
    /// [B, 64, 12] one-hot occupancy.
    pub planes: Tensor,
    /// [B, AUX_DIM] aux features (broadcast across tokens inside the model).
    pub aux: Tensor,
    /// [B] u32 labels (only meaningful for training/eval).
    pub labels: Tensor,
    pub len: usize,
}

impl Batch {
    pub fn from_samples(samples: &[Sample], device: &Device) -> Result<Batch> {
        let b = samples.len();
        let mut planes = vec![0f32; b * N_SQUARES * N_PIECE_PLANES];
        let mut aux = vec![0f32; b * AUX_DIM];
        let mut labels = vec![0u32; b];

        for (i, s) in samples.iter().enumerate() {
            // Square-major one-hot: token s carries 12 plane values.
            for sq in 0..N_SQUARES {
                let code = s.squares[sq];
                if code != 0 {
                    let plane = (code - 1) as usize;
                    planes[(i * N_SQUARES + sq) * N_PIECE_PLANES + plane] = 1.0;
                }
            }
            aux[i * AUX_DIM..(i + 1) * AUX_DIM].copy_from_slice(&s.aux_f32());
            labels[i] = if s.wdl == 255 { 0 } else { s.wdl as u32 };
        }

        Ok(Batch {
            planes: Tensor::from_vec(planes, (b, N_SQUARES, N_PIECE_PLANES), device)?,
            aux: Tensor::from_vec(aux, (b, AUX_DIM), device)?,
            labels: Tensor::from_vec(labels, b, device)?,
            len: b,
        })
    }
}

/// Geometric Attention Bias module (avg-pool variant or full projection).
struct Gab {
    avg_pool: bool,
    per_square: Option<Linear>, // d1 projection (full variant)
    compress: Linear,           // -> d2
    ln: Ln,
    gen: Linear,    // d2 -> heads*templates
    shared: Linear, // templates -> 64*64
    heads: usize,
    templates: usize,
}

impl Gab {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let g = &cfg.gab;
        let d = cfg.model.d_model;
        let heads = cfg.model.num_heads;
        let (per_square, gen_in) = if g.avg_pool {
            (None, d)
        } else {
            let ps = linear(d, g.per_square_dim, vb.pp("per_square"))?;
            (Some(ps), g.per_square_dim * N_SQUARES)
        };
        let compress = linear(gen_in, g.compress_dim, vb.pp("compress"))?;
        let ln = Ln::new(g.compress_dim, vb.pp("ln"))?;
        let gen = linear(g.compress_dim, heads * g.templates, vb.pp("gen"))?;
        let shared = linear(g.templates, N_SQUARES * N_SQUARES, vb.pp("shared"))?;
        Ok(Gab {
            avg_pool: g.avg_pool,
            per_square,
            compress,
            ln,
            gen,
            shared,
            heads,
            templates: g.templates,
        })
    }

    /// x: [B, 64, d] -> bias [B, heads, 64, 64].
    fn bias(&self, x: &Tensor) -> Result<Tensor> {
        let b = x.dim(0)?;
        let compressed = if self.avg_pool {
            x.mean(1)? // [B, d]
        } else {
            let ps = self.per_square.as_ref().unwrap().forward(x)?; // [B,64,d1]
            ps.reshape((b, ()))? // flatten -> [B, 64*d1]
        };
        let c = self
            .ln
            .forward(&self.compress.forward(&compressed)?.gelu()?)?; // [B, d2]
        let g = self
            .gen
            .forward(&c)?
            .reshape((b, self.heads, self.templates))?; // [B,h,d3]
        let bias = self.shared.forward(&g)?; // [B, h, 64*64]
        bias.reshape((b, self.heads, N_SQUARES, N_SQUARES))
    }
}

enum PosBias {
    Gab(Gab),
    Learned(Tensor), // [heads, 64, 64]
    None,
}

struct EncoderBlock {
    ln1: Ln,
    ln2: Ln,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ff1: Linear,
    ff2: Linear,
    heads: usize,
    head_dim: usize,
    dropout: Dropout,
}

impl EncoderBlock {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.model.d_model;
        let h = cfg.model.num_heads;
        Ok(EncoderBlock {
            ln1: Ln::new(d, vb.pp("ln1"))?,
            ln2: Ln::new(d, vb.pp("ln2"))?,
            q: linear(d, d, vb.pp("q"))?,
            k: linear(d, d, vb.pp("k"))?,
            v: linear(d, d, vb.pp("v"))?,
            o: linear(d, d, vb.pp("o"))?,
            ff1: linear(d, cfg.model.ffn_dim, vb.pp("ff1"))?,
            ff2: linear(cfg.model.ffn_dim, d, vb.pp("ff2"))?,
            heads: h,
            head_dim: d / h,
            dropout: Dropout::new(cfg.model.dropout as f32),
        })
    }

    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        x.reshape((b, t, self.heads, self.head_dim))?
            .transpose(1, 2)? // [B, h, T, dh]
            .contiguous()
    }

    fn attention(&self, x: &Tensor, bias: &Tensor, train: bool) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let q = self.split_heads(&self.q.forward(x)?)?;
        let k = self.split_heads(&self.k.forward(x)?)?;
        let v = self.split_heads(&self.v.forward(x)?)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * scale)?;
        let scores = scores.broadcast_add(bias)?; // bias broadcasts over batch if needed
        let attn = self.dropout.forward(&softmax_lastdim(&scores)?, train)?;
        let ctx = attn.matmul(&v)?; // [B,h,T,dh]
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, t, d))?;
        self.o.forward(&ctx)
    }

    fn forward(&self, x: &Tensor, bias: &Tensor, train: bool) -> Result<Tensor> {
        let a = self.attention(&self.ln1.forward(x)?, bias, train)?;
        let x = (x + self.dropout.forward(&a, train)?)?;
        let h = self.ln2.forward(&x)?;
        let ff = self.ff2.forward(&self.ff1.forward(&h)?.gelu()?)?;
        x + self.dropout.forward(&ff, train)?
    }
}

pub struct ChessWdlModel {
    input_proj: Linear,
    pos_bias: PosBias,
    blocks: Vec<EncoderBlock>,
    head_ln: Ln,
    head1: Linear,
    head2: Linear,
    cfg: ModelConfig,
    device: Device,
}

impl ChessWdlModel {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let d = cfg.model.d_model;
        let input_proj = linear(cfg.input_dim(), d, vb.pp("input_proj"))?;
        let pos_bias = match cfg.model.pos_encoding {
            PosEncoding::Gab => PosBias::Gab(Gab::new(cfg, vb.pp("gab"))?),
            PosEncoding::LearnedBias => {
                let t = vb.get_with_hints(
                    (cfg.model.num_heads, N_SQUARES, N_SQUARES),
                    "learned_bias",
                    Init::Const(0.0),
                )?;
                PosBias::Learned(t)
            }
            PosEncoding::None => PosBias::None,
        };
        let mut blocks = Vec::new();
        for i in 0..cfg.model.layers {
            blocks.push(EncoderBlock::new(cfg, vb.pp(format!("block{i}")))?);
        }
        let head_ln = Ln::new(d, vb.pp("head_ln"))?;
        let head1 = linear(d, cfg.model.head_hidden, vb.pp("head1"))?;
        let head2 = linear(cfg.model.head_hidden, 3, vb.pp("head2"))?;
        Ok(ChessWdlModel {
            input_proj,
            pos_bias,
            blocks,
            head_ln,
            head1,
            head2,
            cfg: cfg.clone(),
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// Build the per-token input features [B, 64, input_dim] then project.
    /// Features are occupancy planes + (broadcast) aux state only — no Elo.
    fn embed_input(&self, batch: &Batch) -> Result<Tensor> {
        let b = batch.len;
        let aux = batch
            .aux
            .unsqueeze(1)?
            .broadcast_as((b, N_SQUARES, AUX_DIM))?
            .contiguous()?;
        let x = Tensor::cat(&[batch.planes.clone(), aux], D::Minus1)?; // [B,64,18]
        self.input_proj.forward(&x)
    }

    fn pos_bias(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _, _) = x.dims3()?;
        match &self.pos_bias {
            PosBias::Gab(g) => g.bias(x),
            PosBias::Learned(t) => t
                .unsqueeze(0)?
                .broadcast_as((b, self.cfg.model.num_heads, N_SQUARES, N_SQUARES))?
                .contiguous(),
            PosBias::None => Tensor::zeros(
                (b, self.cfg.model.num_heads, N_SQUARES, N_SQUARES),
                DType::F32,
                &self.device,
            ),
        }
    }

    /// Forward pass returning raw [B, 3] logits.
    pub fn forward(&self, batch: &Batch, train: bool) -> Result<Tensor> {
        let mut x = self.embed_input(batch)?; // [B,64,d]
        let bias = self.pos_bias(&x)?; // [B,h,64,64]
        for block in &self.blocks {
            x = block.forward(&x, &bias, train)?;
        }
        let pooled = self.head_ln.forward(&x.mean(1)?)?; // [B,d]
        let h = self.head1.forward(&pooled)?.relu()?;
        self.head2.forward(&h) // [B,3]
    }
}

/// Convert a [B,3] logits tensor to host-side probability rows.
pub fn logits_to_probs(logits: &Tensor) -> Result<Vec<[f32; 3]>> {
    let p = softmax_lastdim(logits)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    Ok(p.into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::encode_position;
    use candle_nn::VarMap;
    use shakmaty::Chess;

    fn tiny_cfg() -> ModelConfig {
        ModelConfig::load("configs/tiny.toml").unwrap()
    }

    fn dummy_batch(n: usize, device: &Device) -> Batch {
        let pos = Chess::default();
        let (squares, castling, ep_file) = encode_position(&pos);
        let samples: Vec<Sample> = (0..n)
            .map(|i| Sample {
                squares,
                castling,
                ep_file,
                self_elo: 2500,
                oppo_elo: 2400,
                wdl: (i % 3) as u8,
                seen: false,
            })
            .collect();
        Batch::from_samples(&samples, device).unwrap()
    }

    #[test]
    fn forward_shapes_and_softmax() {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = ChessWdlModel::new(&cfg, vb, &device).unwrap();
        let batch = dummy_batch(4, &device);
        let logits = model.forward(&batch, false).unwrap();
        assert_eq!(logits.dims(), &[4, 3]);
        let probs = logits_to_probs(&logits).unwrap();
        for p in probs {
            let s: f32 = p.iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "probs sum to {s}");
            assert!(p.iter().all(|&x| x >= 0.0));
        }
    }

    #[test]
    fn learned_bias_variant_runs() {
        let device = Device::Cpu;
        let mut cfg = tiny_cfg();
        cfg.model.pos_encoding = PosEncoding::LearnedBias;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = ChessWdlModel::new(&cfg, vb, &device).unwrap();
        let batch = dummy_batch(2, &device);
        let logits = model.forward(&batch, true).unwrap();
        assert_eq!(logits.dims(), &[2, 3]);
    }
}
