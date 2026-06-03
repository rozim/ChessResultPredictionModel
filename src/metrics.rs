//! Evaluation metrics, trivial baselines, and probability calibration.
//!
//! Everything here operates on plain `[win, draw, loss]` probability rows and
//! `u8` labels, so it is independent of the model/Candle and easy to test.

pub const N_CLASSES: usize = 3;
const EPS: f32 = 1e-7;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Metrics {
    pub n: usize,
    pub log_loss: f32,
    pub accuracy: f32,
    pub brier: f32,
    pub ece: f32,
    /// confusion[true][pred].
    pub confusion: [[u32; N_CLASSES]; N_CLASSES],
}

fn clamp_norm(p: [f32; 3]) -> [f32; 3] {
    let mut q = [p[0].max(0.0), p[1].max(0.0), p[2].max(0.0)];
    let s: f32 = q.iter().sum::<f32>().max(EPS);
    for x in &mut q {
        *x /= s;
    }
    q
}

/// Compute all metrics for predicted probabilities vs. integer labels.
pub fn evaluate(probs: &[[f32; 3]], labels: &[u8]) -> Metrics {
    assert_eq!(probs.len(), labels.len());
    let n = probs.len();
    let mut log_loss = 0.0f64;
    let mut brier = 0.0f64;
    let mut correct = 0usize;
    let mut confusion = [[0u32; N_CLASSES]; N_CLASSES];

    // 10-bin reliability on the predicted-class confidence for ECE.
    let mut bin_conf = [0.0f64; 10];
    let mut bin_acc = [0.0f64; 10];
    let mut bin_cnt = [0u32; 10];

    for (p, &y) in probs.iter().zip(labels.iter()) {
        let q = clamp_norm(*p);
        let y = y as usize;
        log_loss += -(q[y].max(EPS) as f64).ln();
        for c in 0..N_CLASSES {
            let t = if c == y { 1.0 } else { 0.0 };
            brier += (q[c] as f64 - t).powi(2);
        }
        let pred = argmax(&q);
        if pred == y {
            correct += 1;
        }
        confusion[y][pred] += 1;

        let conf = q[pred];
        let b = ((conf * 10.0) as usize).min(9);
        bin_conf[b] += conf as f64;
        bin_acc[b] += if pred == y { 1.0 } else { 0.0 };
        bin_cnt[b] += 1;
    }

    let mut ece = 0.0f64;
    for b in 0..10 {
        if bin_cnt[b] > 0 {
            let acc = bin_acc[b] / bin_cnt[b] as f64;
            let conf = bin_conf[b] / bin_cnt[b] as f64;
            ece += (bin_cnt[b] as f64 / n as f64) * (acc - conf).abs();
        }
    }

    Metrics {
        n,
        log_loss: (log_loss / n as f64) as f32,
        accuracy: correct as f32 / n as f32,
        brier: (brier / n as f64) as f32,
        ece: ece as f32,
        confusion,
    }
}

pub fn argmax(p: &[f32; 3]) -> usize {
    let mut best = 0;
    for i in 1..3 {
        if p[i] > p[best] {
            best = i;
        }
    }
    best
}

pub fn softmax3(logits: [f32; 3]) -> [f32; 3] {
    let m = logits[0].max(logits[1]).max(logits[2]);
    let e = [
        (logits[0] - m).exp(),
        (logits[1] - m).exp(),
        (logits[2] - m).exp(),
    ];
    let s = e[0] + e[1] + e[2];
    [e[0] / s, e[1] / s, e[2] / s]
}

// --------------------------------------------------------------------------
// Baselines
// --------------------------------------------------------------------------

/// Constant predictor: the class base rates of the training labels.
pub fn base_rate_probs(train_labels: &[u8]) -> [f32; 3] {
    let mut c = [0f32; 3];
    for &y in train_labels {
        c[y as usize] += 1.0;
    }
    clamp_norm(c)
}

/// A 1-feature multinomial logistic regression on material balance, trained by
/// gradient descent. Returns weights `w[c]` and biases `b[c]` for `logit_c = w_c*x + b_c`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MaterialLogistic {
    pub w: [f32; 3],
    pub b: [f32; 3],
    pub feat_mean: f32,
    pub feat_std: f32,
}

impl MaterialLogistic {
    pub fn fit(features: &[f32], labels: &[u8], iters: usize, lr: f32) -> Self {
        let n = features.len().max(1);
        let mean = features.iter().sum::<f32>() / n as f32;
        let var = features.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let std = var.sqrt().max(1e-3);

        let mut w = [0f32; 3];
        let mut b = [0f32; 3];
        for _ in 0..iters {
            let mut gw = [0f32; 3];
            let mut gb = [0f32; 3];
            for (&x, &y) in features.iter().zip(labels.iter()) {
                let xn = (x - mean) / std;
                let probs = softmax3([w[0] * xn + b[0], w[1] * xn + b[1], w[2] * xn + b[2]]);
                for c in 0..3 {
                    let t = if c == y as usize { 1.0 } else { 0.0 };
                    let d = probs[c] - t;
                    gw[c] += d * xn;
                    gb[c] += d;
                }
            }
            for c in 0..3 {
                w[c] -= lr * gw[c] / n as f32;
                b[c] -= lr * gb[c] / n as f32;
            }
        }
        MaterialLogistic {
            w,
            b,
            feat_mean: mean,
            feat_std: std,
        }
    }

    pub fn predict(&self, feature: f32) -> [f32; 3] {
        let xn = (feature - self.feat_mean) / self.feat_std;
        softmax3([
            self.w[0] * xn + self.b[0],
            self.w[1] * xn + self.b[1],
            self.w[2] * xn + self.b[2],
        ])
    }
}

// --------------------------------------------------------------------------
// Temperature scaling
// --------------------------------------------------------------------------

/// Mean log-loss of `logits / T` against labels.
fn temp_log_loss(logits: &[[f32; 3]], labels: &[u8], t: f32) -> f32 {
    let mut ll = 0.0f64;
    for (l, &y) in logits.iter().zip(labels.iter()) {
        let p = softmax3([l[0] / t, l[1] / t, l[2] / t]);
        ll += -(p[y as usize].max(EPS) as f64).ln();
    }
    (ll / logits.len().max(1) as f64) as f32
}

/// Fit a single temperature minimizing log-loss via coarse-to-fine 1-D search.
pub fn fit_temperature(logits: &[[f32; 3]], labels: &[u8]) -> f32 {
    let mut best_t = 1.0f32;
    let mut best = temp_log_loss(logits, labels, 1.0);
    // Search T in [0.25, 5.0].
    let mut lo = 0.25f32;
    let mut hi = 5.0f32;
    for _ in 0..6 {
        let step = (hi - lo) / 20.0;
        let mut t = lo;
        while t <= hi {
            let v = temp_log_loss(logits, labels, t);
            if v < best {
                best = v;
                best_t = t;
            }
            t += step;
        }
        lo = (best_t - step).max(0.05);
        hi = best_t + step;
    }
    best_t
}

pub fn apply_temperature(logits: &[[f32; 3]], t: f32) -> Vec<[f32; 3]> {
    logits
        .iter()
        .map(|l| softmax3([l[0] / t, l[1] / t, l[2] / t]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_predictor_has_ln3_log_loss() {
        let probs = vec![[1.0 / 3.0; 3]; 30];
        let labels: Vec<u8> = (0..30).map(|i| (i % 3) as u8).collect();
        let m = evaluate(&probs, &labels);
        assert!((m.log_loss - (3.0f32).ln()).abs() < 1e-4);
        assert!((m.accuracy - 1.0 / 3.0).abs() < 0.05);
    }

    #[test]
    fn perfect_predictor_is_zero_loss() {
        let labels = [0u8, 1, 2, 1, 0];
        let probs: Vec<[f32; 3]> = labels
            .iter()
            .map(|&y| {
                let mut p = [0.0; 3];
                p[y as usize] = 1.0;
                p
            })
            .collect();
        let m = evaluate(&probs, &labels);
        assert!(m.log_loss < 1e-3);
        assert_eq!(m.accuracy, 1.0);
        assert!(m.brier < 1e-3);
    }

    #[test]
    fn base_rate_matches_frequencies() {
        let labels = [0u8, 0, 1, 2]; // 50% win, 25% draw, 25% loss
        let p = base_rate_probs(&labels);
        assert!((p[0] - 0.5).abs() < 1e-6);
        assert!((p[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn temperature_improves_overconfident_logits() {
        // Overconfident but often wrong -> T > 1 should reduce log loss.
        let logits = vec![[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]];
        let labels = [1u8, 2, 1]; // first two are wrong with high confidence
        let t = fit_temperature(&logits, &labels);
        let before = temp_log_loss(&logits, &labels, 1.0);
        let after = temp_log_loss(&logits, &labels, t);
        assert!(after <= before);
        assert!(t > 1.0);
    }

    #[test]
    fn material_logistic_learns_direction() {
        // Big material advantage -> win (0); big disadvantage -> loss (2).
        let mut feats = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..200 {
            feats.push(5.0);
            labels.push(0u8);
            feats.push(-5.0);
            labels.push(2u8);
        }
        let m = MaterialLogistic::fit(&feats, &labels, 300, 0.5);
        let win = m.predict(5.0);
        let loss = m.predict(-5.0);
        assert!(argmax(&win) == 0);
        assert!(argmax(&loss) == 2);
    }
}
