#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Distribution {
    pub min: f64,
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

impl Distribution {
    pub fn of(values: &[f64]) -> Option<Self> {
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return None;
        }
        let sorted = sorted(values)?;
        Some(Self {
            min: sorted[0],
            median: quantile(&sorted, 0.5),
            p95: quantile(&sorted, 0.95),
            p99: quantile(&sorted, 0.99),
            max: sorted[sorted.len() - 1],
        })
    }
}

pub fn median(values: &[f64]) -> Option<f64> {
    sorted(values).map(|sorted| quantile(&sorted, 0.5))
}

fn sorted(values: &[f64]) -> Option<Vec<f64>> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(sorted)
}

fn quantile(sorted: &[f64], fraction: f64) -> f64 {
    let rank = (sorted.len() - 1) as f64 * fraction;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - lower as f64)
}
