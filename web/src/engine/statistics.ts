/** The spread of a set of observations every benchmark, recording and report quotes. */
export type Distribution = { min: number; median: number; p95: number; p99: number; max: number };

/** Linear interpolation of sorted `values`, which must be nonempty. */
export function percentiles(values: number[]): Distribution {
  const sorted = [...values].sort((a, b) => a - b);
  const quantile = (fraction: number) => {
    const index = (sorted.length - 1) * fraction;
    const lower = Math.floor(index);
    return sorted[lower] + (sorted[Math.ceil(index)] - sorted[lower]) * (index - lower);
  };
  return { min: sorted[0], median: quantile(0.5), p95: quantile(0.95), p99: quantile(0.99), max: sorted.at(-1)! };
}
