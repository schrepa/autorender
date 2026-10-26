//! The wind's waves over water: Elfouhaily, Chapron, Katsaros and
//! Vandemark's unified spectrum (1997, "A unified directional spectrum for
//! long and short wind-driven waves"), whose curvature, the variance of the
//! slopes by the logarithm of the wave number, sums the long waves about the
//! peak a sea as young as its fetch lets it be raises, and the short waves
//! about the slowest ones, as strong as the wind's friction makes them:
//! light wind raises no capillary ripples. Water too shallow to hold its
//! longer waves cuts them by Kitaigorodskii's depth factor (TMA, Bouws et
//! al. 1985), and it holds no waves past the viscous end, where viscosity
//! takes a wave's energy faster than the wind feeds it (Donelan and Pierson
//! 1987). The sea draws its waves from it over unlimited fetch
//! ([`crate::Ocean::spectrum`]), and water's shaders size the wind's tiles
//! and roughen the water by it where they read them
//! (`meshes/water_spectrum.wgsl`, which mirrors its constants).

use crate::math::{clamp, exp, ln, max, pow};
use crate::terrain_data::hydraulics::GRAVITY;

/// The inverse wave age U/c_p of the sea the wind raises over a fetch F,
/// Ω = 0.84 tanh((X/X₀)^0.4)^−0.75 of the fetch over the length the wind's
/// speed sets, X = g F / U², with X₀ = 2.2×10⁴: 0.84 for the fully raised
/// sea, more for younger ones, whose peak runs slower.
pub const RAISED_AGE: f32 = 0.84;
pub const AGE_FETCH: f32 = 2.2e4;
pub const AGE_FETCH_POWER: f32 = 0.4;
pub const AGE_POWER: f32 = -0.75;
/// The long waves' strength, α_p = 6×10⁻³ √Ω.
pub const LONG_STRENGTH: f32 = 6.0e-3;
/// How much higher the spectrum stands at its peak (γ), 1.7 up to Ω = 1
/// and 6 log₁₀ Ω more past it, and the peak's width in √(k/k_p) (σ),
/// 0.08 (1 + 4 Ω⁻³).
pub const PEAK_ENHANCEMENT: f32 = 1.7;
pub const PEAK_ENHANCEMENT_GROWTH: f32 = 6.0;
pub const PEAK_WIDTH: f32 = 0.08;
pub const PEAK_WIDTH_GROWTH: f32 = 4.0;
/// The wave number (radians a meter) at which surface tension speeds waves
/// as much as gravity does, ω² = g k (1 + (k/k_m)²), and the phase speed
/// (meters a second) of the slowest waves, there: k_m and c_m, about which
/// the short waves gather (capillary waves, shorter than about 1.7 cm, lie
/// past it). Water's wave tiles evolve by the same dispersion
/// (`water_waves.wgsl`).
pub const RIPPLE_NUMBER: f32 = 370.0;
pub const RIPPLE_SPEED: f32 = 0.23;
/// The short waves' strength, α_m = 0.01 (1 + ln(u*/c_m)) for the wind's
/// friction velocity u* up to c_m and 0.01 (1 + 3 ln(u*/c_m)) past it, none
/// where that is negative.
pub const SHORT_STRENGTH: f32 = 0.01;
pub const SHORT_GROWTH: f32 = 3.0;
/// The sea's drag on the wind 10 meters up, C_D = 8×10⁻⁴ + 6.5×10⁻⁵ U (Wu
/// 1982), by which the wind's friction velocity is u* = √C_D U.
pub const DRAG: f32 = 8.0e-4;
pub const DRAG_GROWTH: f32 = 6.5e-5;
/// How fast the wind feeds a wave's energy, 0.04 (u*/c)² ω (Plant 1982),
/// and water's kinematic viscosity ν (square meters a second, at 20 °C), by
/// which it takes it at 4 ν k² (Lamb): past the angular frequency where the
/// two meet, 0.04 u*² / (4 ν), the sea holds no waves.
pub const WIND_GROWTH: f32 = 0.04;
pub const VISCOSITY: f32 = 1.0e-6;

/// The wind's sea over a stretch of water: the wave number of its peak
/// (radians a meter); its long and its short waves' strengths times their
/// phase speeds (α_p c_p / 2 and α_m c_m / 2, meters a second); how fast
/// its long waves fall away past its peak (Ω/√10); how much higher it
/// stands at its peak (γ) and how wide its peak is (σ); the wave number of
/// its viscous end (radians a meter), past which it holds no waves; and the
/// water's depth (meters), which holds back its longer waves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindSea {
    pub peak: f32,
    pub long: f32,
    pub short: f32,
    pub fall: f32,
    pub enhancement: f32,
    pub width: f32,
    pub viscous: f32,
    pub depth: f32,
}

impl WindSea {
    /// The sea wind of `speed` meters a second (more than none) raises over
    /// `fetch` meters of water `depth` meters deep, both infinite for the
    /// open sea: its inverse wave age Ω by the fetch, its peak k_p = g Ω²/U²,
    /// which runs at c_p = U/Ω, its short waves' strength by the wind's
    /// friction velocity u*, and its viscous end, the wave number whose
    /// angular frequency ω_v = 0.04 u*² / (4 ν) solves ω² = g k (1 + (k/k_m)²):
    /// k = (k_m/√3)(w − 1/w), w = ∛(z + √(z² + 1)), z = (3√3/2) ω_v² / (g k_m)
    /// (Cardano's root).
    pub fn new(speed: f32, fetch: f32, depth: f32) -> Self {
        // The fetch over the length the wind's speed sets, g F / U², and the
        // sea's inverse wave age, its tanh by the exponential, which an
        // unlimited fetch takes to 1.
        let reach = GRAVITY * fetch / (speed * speed);
        let grown = exp(-2.0 * pow(reach / AGE_FETCH, AGE_FETCH_POWER));
        let age = RAISED_AGE * pow((1.0 - grown) / (1.0 + grown), AGE_POWER);
        // The wind's friction velocity, and the short waves' strength by its
        // ratio to the slowest waves' speed.
        let friction = (DRAG + DRAG_GROWTH * speed).sqrt() * speed;
        let rise = ln(friction / RIPPLE_SPEED);
        let growth = if rise > 0.0 { SHORT_GROWTH } else { 1.0 };
        let short = SHORT_STRENGTH * (1.0 + growth * rise);
        // The viscous end's angular frequency, and its wave number.
        let frequency = WIND_GROWTH * friction * friction / (4.0 * VISCOSITY);
        let cubic = 1.5 * 3.0_f32.sqrt() * frequency * frequency / (GRAVITY * RIPPLE_NUMBER);
        let turned = pow(cubic + (cubic * cubic + 1.0).sqrt(), 1.0 / 3.0);
        Self {
            peak: GRAVITY * age * age / (speed * speed),
            long: 0.5 * LONG_STRENGTH * age.sqrt() * speed / age,
            short: 0.5 * max(short, 0.0) * RIPPLE_SPEED,
            fall: age / 10.0_f32.sqrt(),
            enhancement: PEAK_ENHANCEMENT
                + PEAK_ENHANCEMENT_GROWTH * max(ln(age) / std::f32::consts::LN_10, 0.0),
            width: PEAK_WIDTH * (1.0 + PEAK_WIDTH_GROWTH / (age * age * age)),
            viscous: RIPPLE_NUMBER / 3.0_f32.sqrt() * (turned - 1.0 / turned),
            depth,
        }
    }

    /// Its curvature at wave number `number` (radians a meter): the
    /// variance of its slopes by the logarithm of the wave number, its waves
    /// of every heading counted, B = B_l + B_h. Its long waves'
    /// B_l = (α_p/2)(c_p/c) L_PM J_p e^(−(Ω/√10)(√(k/k_p) − 1)) and its short
    /// waves' B_h = (α_m/2)(c_m/c) L_PM J_p e^(−(k/k_m − 1)²/4), where
    /// c = √((g/k)(1 + (k/k_m)²)) is the waves' phase speed,
    /// L_PM = e^(−5/4 (k_p/k)²) the fully raised sea's shape and J_p = γ^Γ,
    /// Γ = e^(−(√(k/k_p) − 1)² / (2σ²)), its peak's enhancement; times
    /// Kitaigorodskii's factor of √(k h).
    pub fn curvature(&self, number: f32) -> f32 {
        // √(k/k_p), ω/ω_p in deep water, and k_p/k, by which the long and the
        // short waves both fade below the peak.
        let over = (number / self.peak).sqrt();
        let ratio = self.peak / number;
        let fade = -1.25 * ratio * ratio;
        let off = (over - 1.0) / self.width;
        let enhanced = pow(self.enhancement, exp(-0.5 * off * off));
        // k/k_m, and the waves' slowness 1/c (seconds a meter).
        let scaled = number / RIPPLE_NUMBER;
        let slowness = (number / (GRAVITY * (1.0 + scaled * scaled))).sqrt();
        let waves = self.long * exp(fade - self.fall * (over - 1.0))
            + self.short * exp(fade - 0.25 * (scaled - 1.0) * (scaled - 1.0));
        waves * enhanced * slowness * held((number * self.depth).sqrt())
    }

    /// The variance of the slopes it holds over a band of wave numbers about
    /// `number` (radians a meter) `span` wide in their logarithm: its
    /// curvature there times the span, as far as the band lies short of its
    /// viscous end.
    pub fn slopes(&self, number: f32, span: f32) -> f32 {
        let share = clamp(ln(self.viscous / number) / span + 0.5, 0.0, 1.0);
        self.curvature(number) * share * span
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
