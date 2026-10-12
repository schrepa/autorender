//! The spread of measured values, which every benchmark, recording and
//! measured run reports: computed one way, here, for the page, the native
//! host and the measuring tools alike.

/// The spread of a set of measured values (milliseconds, counts): the
/// smallest and the largest, and the median, 95th and 99th percentiles,
/// each interpolated linearly between the two sorted values around its
/// rank.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Distribution {
    /// The smallest value.
    pub min: f64,
    /// The 50th percentile.
    pub median: f64,
    /// The 95th percentile.
    pub p95: f64,
    /// The 99th percentile.
    pub p99: f64,
    /// The largest value.
    pub max: f64,
}

impl Distribution {
    /// The spread of `values`: `None` when there are none, or one is not a
    /// finite number of at least 0, as no measurement is.
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

/// The median of `values`, interpolated as [`Distribution`]'s; `None`
/// without values.
pub fn median(values: &[f64]) -> Option<f64> {
    sorted(values).map(|sorted| quantile(&sorted, 0.5))
}

/// `values` in ascending order; `None` when there are none.
fn sorted(values: &[f64]) -> Option<Vec<f64>> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(sorted)
}

/// The value `fraction` (0 to 1) of the way through `sorted` (ascending,
/// not empty) by rank: rank `fraction × (len − 1)`, interpolated linearly
/// between the two values around it.
fn quantile(sorted: &[f64], fraction: f64) -> f64 {
    let rank = (sorted.len() - 1) as f64 * fraction;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - lower as f64)
}
