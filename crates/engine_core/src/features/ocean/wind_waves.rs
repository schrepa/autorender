//! The wind's waves over water (Horvath 2015, "Empirical directional wave
//! spectra for computer graphics"): JONSWAP's spectrum (Hasselmann et al.
//! 1973), whose energy and peak the fetch sets, how far the wind has blown
//! over the water, never past the fully raised sea's (Pierson and Moskowitz
//! 1964), and cut where the water is too shallow to hold its longer waves by
//! Kitaigorodskii's depth factor (TMA, Bouws et al. 1985). The sea draws its
//! waves from it over unlimited fetch ([`crate::Ocean::spectrum`]), and
//! water's shaders size the wind's tiles by it where they read them
//! (`meshes/water_spectrum.wgsl`, which mirrors its constants).

use crate::math::{exp, max, pow};
use crate::terrain_data::hydraulics::GRAVITY;

/// Phillips's constant α of the sea the wind raises over a fetch F,
/// 0.076 (U²/(F g))^0.22.
pub const FETCH_ALPHA: f32 = 0.076;
pub const FETCH_ALPHA_POWER: f32 = 0.22;
/// The angular frequency of the waves the wind raises most over a fetch F,
/// 22 (g²/(U F))^(1/3).
pub const FETCH_PEAK: f32 = 22.0;
/// The fully raised sea's α, and its peak's angular frequency over g/U:
/// however far the wind blows, its sea's α falls and its peak lowers no
/// further.
pub const RAISED_ALPHA: f32 = 0.0081;
pub const RAISED_PEAK: f32 = 0.877;
/// How much higher the spectrum stands at its peak than the fully raised
/// sea's shape alone (γ), and the widths of the peak below and above its
/// frequency, as shares of it (σ).
pub const PEAK_ENHANCEMENT: f32 = 3.3;
pub const PEAK_BELOW: f32 = 0.07;
pub const PEAK_ABOVE: f32 = 0.09;

/// The wind's sea over a stretch of water: Phillips's constant α, the
/// angular frequency of its peak (radians a second), and the water's depth
/// (meters), which holds back its longer waves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindSea {
    pub alpha: f32,
    pub peak: f32,
    pub depth: f32,
}

impl WindSea {
    /// The sea wind of `speed` meters a second (more than none) raises over
    /// `fetch` meters of water `depth` meters deep, both infinite for the
    /// open sea.
    pub fn new(speed: f32, fetch: f32, depth: f32) -> Self {
        // The fetch over the length the wind's speed sets, g F / U².
        let reach = GRAVITY * fetch / (speed * speed);
        Self {
            alpha: max(FETCH_ALPHA * pow(reach, -FETCH_ALPHA_POWER), RAISED_ALPHA),
            peak: max(FETCH_PEAK * pow(reach, -1.0 / 3.0), RAISED_PEAK) * GRAVITY / speed,
            depth,
        }
    }

    /// The spectrum's density at angular frequency `frequency` (radians a
    /// second): the variance of the surface's height its waves of every
    /// heading hold per radian a second (square meters seconds). JONSWAP's
    /// α g² ω⁻⁵ e^(−5/4 (ω_p/ω)⁴) γ^r, r = e^(−(ω − ω_p)² / (2 σ² ω_p²)),
    /// times Kitaigorodskii's factor of ω √(h/g).
    pub fn density(&self, frequency: f32) -> f32 {
        let ratio = self.peak / frequency;
        let squared = ratio * ratio;
        let width = if frequency <= self.peak {
            PEAK_BELOW
        } else {
            PEAK_ABOVE
        };
        let off = (frequency - self.peak) / (width * self.peak);
        let enhanced = pow(PEAK_ENHANCEMENT, exp(-0.5 * off * off));
        let fifth = frequency * frequency * frequency * frequency * frequency;
        self.alpha * GRAVITY * GRAVITY / fifth
            * exp(-1.25 * squared * squared)
            * enhanced
            * held(frequency * (self.depth / GRAVITY).sqrt())
    }
}

/// Kitaigorodskii's depth factor at `reach`, a wave's angular frequency
/// times √(h/g): the share of deep water's spectrum water h meters deep
/// holds there, ω_h²/2 up to 1, then 1 − (2 − ω_h)²/2 up to 2, and all of it
/// past (Thompson and Vincent's fit).
fn held(reach: f32) -> f32 {
    if reach <= 1.0 {
        0.5 * reach * reach
    } else if reach < 2.0 {
        let short = 2.0 - reach;
        1.0 - 0.5 * short * short
    } else {
        1.0
    }
}
