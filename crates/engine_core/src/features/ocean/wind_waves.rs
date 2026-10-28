use crate::math::{clamp, exp, ln, max, pow};
use crate::terrain_data::hydraulics::GRAVITY;

pub const RAISED_AGE: f32 = 0.84;
pub const AGE_FETCH: f32 = 2.2e4;
pub const AGE_FETCH_POWER: f32 = 0.4;
pub const AGE_POWER: f32 = -0.75;
pub const LONG_STRENGTH: f32 = 6.0e-3;
pub const PEAK_ENHANCEMENT: f32 = 1.7;
pub const PEAK_ENHANCEMENT_GROWTH: f32 = 6.0;
pub const PEAK_WIDTH: f32 = 0.08;
pub const PEAK_WIDTH_GROWTH: f32 = 4.0;
pub const RIPPLE_NUMBER: f32 = 370.0;
pub const RIPPLE_SPEED: f32 = 0.23;
pub const SHORT_STRENGTH: f32 = 0.01;
pub const SHORT_GROWTH: f32 = 3.0;
pub const DRAG: f32 = 8.0e-4;
pub const DRAG_GROWTH: f32 = 6.5e-5;
pub const WIND_GROWTH: f32 = 0.04;
pub const VISCOSITY: f32 = 1.0e-6;

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
    pub fn new(speed: f32, fetch: f32, depth: f32) -> Self {
        let reach = GRAVITY * fetch / (speed * speed);
        let grown = exp(-2.0 * pow(reach / AGE_FETCH, AGE_FETCH_POWER));
        let age = RAISED_AGE * pow((1.0 - grown) / (1.0 + grown), AGE_POWER);
        let friction = (DRAG + DRAG_GROWTH * speed).sqrt() * speed;
        let rise = ln(friction / RIPPLE_SPEED);
        let growth = if rise > 0.0 { SHORT_GROWTH } else { 1.0 };
        let short = SHORT_STRENGTH * (1.0 + growth * rise);
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

    pub fn curvature(&self, number: f32) -> f32 {
        let over = (number / self.peak).sqrt();
        let ratio = self.peak / number;
        let fade = -1.25 * ratio * ratio;
        let off = (over - 1.0) / self.width;
        let enhanced = pow(self.enhancement, exp(-0.5 * off * off));
        let scaled = number / RIPPLE_NUMBER;
        let slowness = (number / (GRAVITY * (1.0 + scaled * scaled))).sqrt();
        let waves = self.long * exp(fade - self.fall * (over - 1.0))
            + self.short * exp(fade - 0.25 * (scaled - 1.0) * (scaled - 1.0));
        waves * enhanced * slowness * held((number * self.depth).sqrt())
    }

    pub fn slopes(&self, number: f32, span: f32) -> f32 {
        let share = clamp(ln(self.viscous / number) / span + 0.5, 0.0, 1.0);
        self.curvature(number) * share * span
    }
}

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
