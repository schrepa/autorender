use glam::{Vec2, Vec3};

use super::numbers::{LevelNumbers, PlayNumbers};
use super::rng::Rng;

pub const PI: f32 = std::f32::consts::PI;
const HALF_PI: f32 = std::f32::consts::FRAC_PI_2;
const TAU: f32 = std::f32::consts::TAU;

pub fn min(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

pub fn max(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

pub fn clamp(x: f32, low: f32, high: f32) -> f32 {
    min(max(x, low), high)
}

pub fn radians(degrees: f32) -> f32 {
    degrees * (PI / 180.0)
}

pub fn sin_cos(angle: f32) -> (f32, f32) {
    let turns = (angle * (1.0 / TAU)).round();
    let mut r = angle - turns * TAU;
    let mut cos_sign = 1.0;
    if r > HALF_PI {
        r = PI - r;
        cos_sign = -1.0;
    } else if r < -HALF_PI {
        r = -PI - r;
        cos_sign = -1.0;
    }
    let r2 = r * r;
    let sin = r
        * (1.0
            + r2 * (-1.0 / 6.0
                + r2 * (1.0 / 120.0 + r2 * (-1.0 / 5_040.0 + r2 * (1.0 / 362_880.0)))));
    let cos = 1.0
        + r2 * (-0.5
            + r2 * (1.0 / 24.0
                + r2 * (-1.0 / 720.0 + r2 * (1.0 / 40_320.0 - r2 * (1.0 / 3_628_800.0)))));
    (sin, cos * cos_sign)
}

fn atan_unit(t: f32) -> f32 {
    let t2 = t * t;
    t * (0.999_866 + t2 * (-0.330_299_5 + t2 * (0.180_141 + t2 * (-0.085_133 + t2 * 0.020_835_1))))
}

pub fn atan2(y: f32, x: f32) -> f32 {
    let ax = x.abs();
    let ay = y.abs();
    if ax == 0.0 && ay == 0.0 {
        return 0.0;
    }
    let base = if ax >= ay {
        atan_unit(ay / ax)
    } else {
        HALF_PI - atan_unit(ax / ay)
    };
    let quadrant = if x < 0.0 { PI - base } else { base };
    if y < 0.0 { -quadrant } else { quadrant }
}

pub fn wrap(angle: f32) -> f32 {
    angle - (angle * (1.0 / TAU)).round() * TAU
}

pub fn look_angles(from: Vec3, to: Vec3) -> (f32, f32) {
    let d = to - from;
    let flat = (d.x * d.x + d.z * d.z).sqrt();
    (atan2(-d.x, -d.z), atan2(d.y, flat))
}

pub fn direction(yaw: f32, pitch: f32) -> Vec3 {
    let (sin_yaw, cos_yaw) = sin_cos(yaw);
    let (sin_pitch, cos_pitch) = sin_cos(pitch);
    Vec3::new(-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch)
}

pub fn flat(v: Vec3) -> Vec3 {
    Vec3::new(v.x, 0.0, v.z)
}

pub fn flat_distance(a: Vec3, b: Vec3) -> f32 {
    flat(b - a).length()
}

pub fn angle_between(yaw_a: f32, pitch_a: f32, yaw_b: f32, pitch_b: f32) -> f32 {
    let dy = wrap(yaw_b - yaw_a);
    let dp = pitch_b - pitch_a;
    (dy * dy + dp * dp).sqrt()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aim {
    pub yaw: f32,
    pub pitch: f32,
    pub synced: bool,
    from: Vec2,
    to: Vec2,
    wander_start: u64,
    wander_ticks: u64,
}

impl Aim {
    pub fn new() -> Aim {
        Aim {
            yaw: 0.0,
            pitch: 0.0,
            synced: false,
            from: Vec2::ZERO,
            to: Vec2::ZERO,
            wander_start: 0,
            wander_ticks: 1,
        }
    }

    pub fn sync(&mut self, yaw: f32, pitch: f32) {
        self.yaw = yaw;
        self.pitch = pitch;
        self.synced = true;
    }

    pub fn turn_toward(
        &mut self,
        yaw: f32,
        pitch: f32,
        turn_deg: f32,
        play: &PlayNumbers,
        dt: f32,
    ) {
        let dy = wrap(yaw - self.yaw);
        let dp = pitch - self.pitch;
        let distance = (dy * dy + dp * dp).sqrt();
        if distance <= f32::EPSILON {
            return;
        }
        let ease = radians(play.aim_ease_deg);
        let share = clamp(distance / ease, play.aim_min_turn_share, 1.0);
        let step = min(radians(turn_deg) * share * dt, distance);
        self.yaw = wrap(self.yaw + dy * (step / distance));
        self.pitch += dp * (step / distance);
    }

    fn polar(rng: &mut Rng) -> Vec2 {
        let (sin, cos) = sin_cos(rng.unit() * TAU);
        let magnitude = 1.5 * rng.unit().sqrt();
        Vec2::new(cos * magnitude, sin * magnitude)
    }

    pub fn new_error(&mut self, rng: &mut Rng, tick: u64, wander_ticks: u64) {
        self.from = Aim::polar(rng);
        self.to = Aim::polar(rng);
        self.wander_start = tick;
        self.wander_ticks = wander_ticks.max(1);
    }

    pub fn error_unit(&mut self, rng: &mut Rng, tick: u64) -> Vec2 {
        while tick >= self.wander_start + self.wander_ticks {
            self.from = self.to;
            self.to = Aim::polar(rng);
            self.wander_start += self.wander_ticks;
        }
        let s = (tick - self.wander_start) as f32 / self.wander_ticks as f32;
        let from_length = self.from.length();
        let to_length = self.to.length();
        let from_angle = atan2(self.from.y, self.from.x);
        let to_angle = atan2(self.to.y, self.to.x);
        let angle = from_angle + wrap(to_angle - from_angle) * s;
        let length = from_length + (to_length - from_length) * s;
        let (sin, cos) = sin_cos(angle);
        Vec2::new(cos * length, sin * length)
    }
}

pub fn error_radians(level: &LevelNumbers, seen_seconds: f32, speed_share: f32, blind: f32) -> f32 {
    let extra = max(level.first_shot_error_deg - level.tracking_error_deg, 0.0);
    let settled = extra / (1.0 + max(seen_seconds, 0.0) / level.settle_seconds);
    let moving = 1.0 + level.move_error * clamp(speed_share, 0.0, 1.0);
    radians(level.tracking_error_deg + settled) * moving * (1.0 + blind)
}
