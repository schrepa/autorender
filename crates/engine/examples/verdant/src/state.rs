use engine::math;
use engine::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TimePreset {
    Noon,
    #[default]
    Gold,
    Dusk,
    Night,
}

impl TimePreset {
    pub const ALL: [Self; 4] = [Self::Noon, Self::Gold, Self::Dusk, Self::Night];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Noon => "NOON",
            Self::Gold => "GOLD",
            Self::Dusk => "DUSK",
            Self::Night => "NIGHT",
        }
    }

    pub const fn value(self) -> f32 {
        match self {
            Self::Noon => 0.0,
            Self::Gold => 1.0,
            Self::Dusk => 2.0,
            Self::Night => 3.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WindPreset {
    Calm,
    #[default]
    Breeze,
    Gale,
}

impl WindPreset {
    pub const ALL: [Self; 3] = [Self::Calm, Self::Breeze, Self::Gale];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Calm => "CALM",
            Self::Breeze => "BREEZE",
            Self::Gale => "GALE",
        }
    }

    pub const fn strength(self) -> f32 {
        match self {
            Self::Calm => 0.32,
            Self::Breeze => 0.92,
            Self::Gale => 1.95,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DensityPreset {
    K250,
    K500,
    #[default]
    M1,
}

impl DensityPreset {
    pub const ALL: [Self; 3] = [Self::K250, Self::K500, Self::M1];

    pub const fn label(self) -> &'static str {
        match self {
            Self::K250 => "250K",
            Self::K500 => "500K",
            Self::M1 => "1M",
        }
    }

    pub const fn short_label(self) -> &'static str {
        match self {
            Self::K250 => "250K",
            Self::K500 => "500K",
            Self::M1 => "1.0M",
        }
    }

    pub const fn count(self) -> u32 {
        match self {
            Self::K250 => 250_000,
            Self::K500 => 500_000,
            Self::M1 => 1_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CamMode {
    #[default]
    Orbit,
    Roam,
}

impl CamMode {
    pub const ALL: [Self; 2] = [Self::Orbit, Self::Roam];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Orbit => "ORBIT",
            Self::Roam => "ROAM",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LightingMood {
    pub sun_dir: Vec3,
    pub sun_color: Vec3,
    pub sky_zenith: Vec3,
    pub sky_horizon: Vec3,
    pub haze_color: Vec3,
    pub ground_ambient: Vec3,
    pub cloud_lit: Vec3,
    pub cloud_shadow: Vec3,
    pub lantern_power: f32,
    pub firefly_power: f32,
    pub star_power: f32,
    pub bloom_strength: f32,
    pub exposure: f32,
    pub shaft_strength: f32,
    pub moon_factor: f32,
}

fn mix3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

fn mix1(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn preset_mood(index: usize) -> LightingMood {
    match index {
        0 => LightingMood {
            sun_dir: Vec3::new(-0.42, 0.78, -0.46).normalize(),
            sun_color: Vec3::new(3.25, 3.18, 2.85),
            sky_zenith: Vec3::new(0.042, 0.21, 0.74),
            sky_horizon: Vec3::new(0.42, 0.72, 0.96),
            haze_color: Vec3::new(0.46, 0.72, 0.94),
            ground_ambient: Vec3::new(0.08, 0.16, 0.08),
            cloud_lit: Vec3::new(1.58, 1.56, 1.50),
            cloud_shadow: Vec3::new(0.20, 0.34, 0.58),
            lantern_power: 0.02,
            firefly_power: 0.0,
            star_power: 0.0,
            bloom_strength: 0.14,
            exposure: 0.96,
            shaft_strength: 0.10,
            moon_factor: 0.0,
        },
        1 => LightingMood {
            sun_dir: Vec3::new(-0.56, 0.175, -0.810).normalize(),
            sun_color: Vec3::new(3.55, 2.42, 1.24),
            sky_zenith: Vec3::new(0.024, 0.082, 0.29),
            sky_horizon: Vec3::new(1.24, 0.44, 0.14),
            haze_color: Vec3::new(0.54, 0.24, 0.22),
            ground_ambient: Vec3::new(0.048, 0.105, 0.048),
            cloud_lit: Vec3::new(2.05, 0.86, 0.34),
            cloud_shadow: Vec3::new(0.095, 0.075, 0.20),
            lantern_power: 1.15,
            firefly_power: 0.18,
            star_power: 0.04,
            bloom_strength: 0.25,
            exposure: 1.00,
            shaft_strength: 0.52,
            moon_factor: 0.0,
        },
        2 => LightingMood {
            sun_dir: Vec3::new(-0.54, 0.06, -0.839).normalize(),
            sun_color: Vec3::new(1.85, 0.72, 0.62),
            sky_zenith: Vec3::new(0.016, 0.022, 0.115),
            sky_horizon: Vec3::new(0.84, 0.21, 0.32),
            haze_color: Vec3::new(0.26, 0.11, 0.24),
            ground_ambient: Vec3::new(0.028, 0.048, 0.088),
            cloud_lit: Vec3::new(1.32, 0.38, 0.36),
            cloud_shadow: Vec3::new(0.045, 0.040, 0.125),
            lantern_power: 2.45,
            firefly_power: 1.45,
            star_power: 0.58,
            bloom_strength: 0.35,
            exposure: 1.08,
            shaft_strength: 0.24,
            moon_factor: 0.25,
        },
        _ => LightingMood {
            sun_dir: Vec3::new(0.28, 0.24, -0.928).normalize(),
            sun_color: Vec3::new(0.34, 0.50, 0.90),
            sky_zenith: Vec3::new(0.004, 0.009, 0.034),
            sky_horizon: Vec3::new(0.018, 0.044, 0.125),
            haze_color: Vec3::new(0.016, 0.036, 0.100),
            ground_ambient: Vec3::new(0.008, 0.020, 0.040),
            cloud_lit: Vec3::new(0.14, 0.22, 0.40),
            cloud_shadow: Vec3::new(0.008, 0.014, 0.038),
            lantern_power: 3.15,
            firefly_power: 2.55,
            star_power: 1.0,
            bloom_strength: 0.44,
            exposure: 1.22,
            shaft_strength: 0.10,
            moon_factor: 1.0,
        },
    }
}

pub fn sample_mood(phase: f32) -> LightingMood {
    let clamped = math::clamp(phase, 0.0, 3.0);
    let idx = (clamped as usize).min(2);
    let frac = clamped - idx as f32;
    let t = frac * frac * (3.0 - 2.0 * frac);
    let a = preset_mood(idx);
    let b = preset_mood(idx + 1);
    LightingMood {
        sun_dir: mix3(a.sun_dir, b.sun_dir, t).normalize_or_zero(),
        sun_color: mix3(a.sun_color, b.sun_color, t),
        sky_zenith: mix3(a.sky_zenith, b.sky_horizon * 0.0 + b.sky_zenith, t),
        sky_horizon: mix3(a.sky_horizon, b.sky_horizon, t),
        haze_color: mix3(a.haze_color, b.haze_color, t),
        ground_ambient: mix3(a.ground_ambient, b.ground_ambient, t),
        cloud_lit: mix3(a.cloud_lit, b.cloud_lit, t),
        cloud_shadow: mix3(a.cloud_shadow, b.cloud_shadow, t),
        lantern_power: mix1(a.lantern_power, b.lantern_power, t),
        firefly_power: mix1(a.firefly_power, b.firefly_power, t),
        star_power: mix1(a.star_power, b.star_power, t),
        bloom_strength: mix1(a.bloom_strength, b.bloom_strength, t),
        exposure: mix1(a.exposure, b.exposure, t),
        shaft_strength: mix1(a.shaft_strength, b.shaft_strength, t),
        moon_factor: mix1(a.moon_factor, b.moon_factor, t),
    }
}

#[derive(Resource)]
pub struct VerdantState {
    pub time_preset: TimePreset,
    pub wind_preset: WindPreset,
    pub density: DensityPreset,
    pub cam_mode: CamMode,
    pub density_auto_set: bool,
    pub time_phase: f32,
    pub wind_strength: f32,
    pub sim_time: f32,
    pub dt: f32,
    pub hud_hidden: bool,
    pub ui_captured_pointer: bool,
    pub ui_hovered: bool,
    pub wake_body_pos: Vec3,
    pub wake_body_vel: Vec2,
    pub wake_cursor_pos: Vec3,
    pub wake_cursor_vel: Vec2,
    pub wake_cursor_active: bool,
    pub fps_smoothed: f32,
    pub frame_count: u64,
    pub camera_eye: Vec3,
    pub camera_forward: Vec3,
    pub camera_right: Vec3,
    pub camera_up: Vec3,
    pub view_projection: Mat4,
    pub inv_view_projection: Mat4,
    pub render_scale: f32,
}

impl Default for VerdantState {
    fn default() -> Self {
        Self {
            time_preset: TimePreset::Gold,
            wind_preset: WindPreset::Breeze,
            density: DensityPreset::M1,
            cam_mode: CamMode::Orbit,
            density_auto_set: false,
            time_phase: TimePreset::Gold.value(),
            wind_strength: WindPreset::Breeze.strength(),
            sim_time: 0.0,
            dt: 1.0 / 60.0,
            hud_hidden: false,
            ui_captured_pointer: false,
            ui_hovered: false,
            wake_body_pos: Vec3::new(0.0, 4.0, 14.0),
            wake_body_vel: Vec2::ZERO,
            wake_cursor_pos: Vec3::ZERO,
            wake_cursor_vel: Vec2::ZERO,
            wake_cursor_active: false,
            fps_smoothed: 60.0,
            frame_count: 0,
            camera_eye: Vec3::new(3.8, 6.4, 16.5),
            camera_forward: Vec3::NEG_Z,
            camera_right: Vec3::X,
            camera_up: Vec3::Y,
            view_projection: Mat4::IDENTITY,
            inv_view_projection: Mat4::IDENTITY,
            render_scale: 1.0,
        }
    }
}

impl VerdantState {
    pub fn mood(&self) -> LightingMood {
        sample_mood(self.time_phase)
    }

    pub fn active_blades(&self) -> u32 {
        self.density.count()
    }
}

impl brick_present::Scale for VerdantState {
    fn fraction(&self) -> f32 {
        self.render_scale
    }

    fn direct_supported() -> bool {
        true
    }
}
