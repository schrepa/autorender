/** Independent numerical reference for the radiance composition fixture.
 * Evaluates the published WBOIT operator, not sorted alpha OVER. Half stores
 * and UNORM revealage are explicit; the display grade follows composition. */
import assert from "node:assert/strict";

export const clamp = (x, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x));
export const unorm = (x, max) => Math.round(clamp(x) * max) / max;
const dot = rgb => rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
const mix = (a, b, t) => a * (1 - t) + b * t;

/** Round a finite value to binary16, nearest with ties to even. */
export function half(x) {
  assert.ok(Number.isFinite(x) && Math.abs(x) < 65520, `finite half input: ${x}`);
  if (x === 0) return x;
  const sign = Math.sign(x), magnitude = Math.abs(x);
  const step = 2 ** Math.max(-24, Math.floor(Math.log2(magnitude)) - 10);
  const units = magnitude / step, lo = Math.floor(units), fraction = units - lo;
  return sign * (fraction > 0.5 || (fraction === 0.5 && lo % 2) ? lo + 1 : lo) * step;
}

export const neutral = { contrast: 1, saturation: 1, vibrance: 0,
  shadow_tint: [1, 1, 1], highlight_tint: [1, 1, 1] };

/** Existing grade on already exposed linear RGB; no exposure is applied here. */
export function grade(exposed, settings = neutral) {
  const filmic = x => clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14));
  const peak = Math.max(...exposed);
  const curved = exposed.map(x => mix(filmic(x), x * filmic(peak) / Math.max(peak, 1e-6), 0.5));
  const contrast = clamp(2 * (settings.contrast - 1), -2, 1);
  let mapped = curved.map(x => {
    const root = Math.sqrt(x);
    return (root + contrast * (root * root * (3 - 2 * root) - root)) ** 2;
  });
  const t = clamp((Math.sqrt(dot(mapped)) - 0.2) / 0.6), tone = t * t * (3 - 2 * t);
  const normalize = tint => dot(tint) > 1e-6 ? tint.map(x => x / dot(tint)) : [1, 1, 1];
  const shadows = normalize(settings.shadow_tint), highlights = normalize(settings.highlight_tint);
  mapped = mapped.map((x, i) => x * mix(shadows[i], highlights[i], tone));
  const top = Math.max(...mapped), bottom = Math.min(...mapped);
  const amount = settings.saturation + settings.vibrance * (1 - (top - bottom) / Math.max(top, 1e-6));
  return mapped.map(x => clamp(mix(dot(mapped), x, amount)));
}

export const encode = rgb => rgb.map(x => Math.round(clamp(x <= 0.0031308
  ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055) * 255));

/** B and layer color are raw light. Layer alpha can be a computed coverage
 * (glass/particles); ordinary surface opacity is packed as UNORM16. */
export function compose(background, layers, exposure, settings = neutral, additive = []) {
  let mask = 0;
  const guard = (value, bit) => {
    assert.ok(Number.isFinite(value), "oracle never certifies upstream overflow");
    if (value < 0 || value >= 60000) mask |= bit;
    return clamp(value, 0, 60000);
  };
  let rgb = background.map(x => half(x * exposure / 64));
  const numerator = [0, 0, 0];
  let denominator = 0, reveal = 1;
  for (const layer of layers) {
    const alpha = layer.packed === false ? layer.alpha : unorm(layer.alpha, 65535);
    if (alpha === 0) continue;
    const weight = alpha * clamp(10 / (1e-5 + (layer.distance / 5) ** 2 + (layer.distance / 200) ** 6), 0.01, 3000);
    for (let channel = 0; channel < 3; channel++) {
      numerator[channel] = half(numerator[channel] + layer.color[channel] * exposure / 64 * alpha * weight);
    }
    denominator = half(denominator + alpha * weight);
    reveal = unorm(reveal * (1 - alpha), 255);
  }
  if (reveal < 1) {
    const sum = numerator.map(x => guard(x, 1)), alpha = guard(denominator, 1);
    const average = sum.map(x => guard(x / Math.max(alpha, 1e-5), 1));
    rgb = rgb.map((x, i) => half(average[i] * (1 - reveal) + x * reveal));
  }
  for (const light of additive) rgb = rgb.map((x, i) => half(x + light[i] * exposure / 64));
  const exposed = rgb.map(x => guard(x, 2) * 64);
  return { rgb: encode(grade(exposed, settings)), mask, exposed, reveal, denominator, numerator };
}

/** Grey grade is monotonic in the fixture's unclipped interval. */
export function inverseGrey(code) {
  const encoded = code / 255;
  const display = encoded <= 0.04045 ? encoded / 12.92 : ((encoded + 0.055) / 1.055) ** 2.4;
  let lo = 0, hi = 2;
  for (let step = 0; step < 60; step++) {
    const mid = (lo + hi) / 2;
    if (grade([mid, mid, mid])[0] < display) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}
