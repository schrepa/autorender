use engine::math;
use engine::prelude::*;

pub const MAX_PARTICLES: u32 = 250_000;
pub const GRID_X: u32 = 84;
pub const GRID_Y: u32 = 48;
pub const GRID_Z: u32 = 56;
pub const CELL_SIZE: f32 = 0.056;
pub const BASIN_HALF_X: f32 = 2.18;
pub const BASIN_HALF_Z: f32 = 1.40;
pub const BASIN_HEIGHT: f32 = 2.35;
pub const BASIN_FLOOR_Y: f32 = 0.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum MaterialKind {
    #[default]
    Chrome = 0,
    Water = 1,
    Gold = 2,
    Plasma = 3,
}

impl MaterialKind {
    pub const ALL: [Self; 4] = [Self::Chrome, Self::Water, Self::Gold, Self::Plasma];

    pub const fn index(self) -> u32 {
        self as u32
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Chrome => "CHROME",
            Self::Water => "WATER",
            Self::Gold => "GOLD",
            Self::Plasma => "PLASMA",
        }
    }

    pub const fn swatch(self) -> [f32; 4] {
        match self {
            Self::Chrome => [0.88, 0.91, 0.97, 1.0],
            Self::Water => [0.14, 0.86, 0.96, 1.0],
            Self::Gold => [1.0, 0.74, 0.20, 1.0],
            Self::Plasma => [0.72, 0.30, 1.0, 1.0],
        }
    }

    pub const fn viscosity(self) -> f32 {
        match self {
            Self::Chrome => 0.028,
            Self::Water => 0.015,
            Self::Gold => 0.068,
            Self::Plasma => 0.022,
        }
    }

    pub const fn stiffness(self) -> f32 {
        match self {
            Self::Chrome => 16.0,
            Self::Water => 18.0,
            Self::Gold => 14.5,
            Self::Plasma => 14.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScalePreset {
    Low,
    Mid,
    #[default]
    High,
}

impl ScalePreset {
    pub const ALL: [Self; 3] = [Self::Low, Self::Mid, Self::High];

    pub const fn count(self) -> u32 {
        match self {
            Self::Low => 60_000,
            Self::Mid => 150_000,
            Self::High => 250_000,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Low => "60K",
            Self::Mid => "150K",
            Self::High => "250K",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CameraRig {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub goal_yaw: f32,
    pub goal_pitch: f32,
    pub goal_distance: f32,
    pub target: Vec3,
    pub eye: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
    pub user_moved: bool,
    pub portrait_init: bool,
}

impl Default for CameraRig {
    fn default() -> Self {
        let yaw = 0.38;
        let pitch = 0.46;
        let distance = 5.35;
        let target = Vec3::new(0.0, 0.44, 0.0);
        Self {
            yaw,
            pitch,
            distance,
            goal_yaw: yaw,
            goal_pitch: pitch,
            goal_distance: distance,
            target,
            eye: Vec3::new(1.8, 2.5, 4.6),
            forward: Vec3::NEG_Z,
            right: Vec3::X,
            up: Vec3::Y,
            fov_y: 0.70,
            user_moved: false,
            portrait_init: false,
        }
    }
}

impl CameraRig {
    pub fn update(&mut self, dt: f32, elapsed: f32, zero_g: bool, aspect: f32) {
        let portrait = aspect < 0.85;
        let want_target = if zero_g {
            Vec3::new(0.0, if portrait { 0.92 } else { 1.08 }, 0.0)
        } else {
            Vec3::new(0.0, if portrait { 0.26 } else { 0.42 }, 0.0)
        };
        let blend = 1.0 - math::exp(-dt * 7.5);
        self.target = self.target.lerp(want_target, blend);

        let auto_dist = if portrait {
            math::clamp(2.74 / (0.495 * math::max(aspect, 0.40)), 6.4, 11.8)
        } else {
            5.35
        };
        if portrait && !self.portrait_init {
            self.portrait_init = true;
            self.distance = auto_dist;
            self.goal_distance = auto_dist;
            self.target = want_target;
        }
        if !self.user_moved {
            let drift_yaw = if portrait {
                0.34 + math::sin(elapsed * 0.22) * 0.12
            } else {
                0.38 + math::sin(elapsed * 0.22) * 0.14
            };
            let drift_pitch = if portrait {
                0.52 + math::cos(elapsed * 0.17) * 0.04
            } else {
                0.46 + math::cos(elapsed * 0.17) * 0.04
            };
            self.goal_yaw = drift_yaw;
            self.goal_pitch = drift_pitch;
            self.goal_distance = auto_dist;
        }

        let min_dist = if portrait { 5.8 } else { 3.2 };
        let max_dist = if portrait { 14.0 } else { 10.5 };
        self.goal_distance = math::clamp(self.goal_distance, min_dist, max_dist);
        self.goal_pitch = math::clamp(self.goal_pitch, 0.12, 1.38);
        let smooth = 1.0 - math::exp(-dt * 12.0);
        self.yaw += (self.goal_yaw - self.yaw) * smooth;
        self.pitch += (self.goal_pitch - self.pitch) * smooth;
        self.distance += (self.goal_distance - self.distance) * smooth;
        let (sy, cy) = math::sin_cos(self.yaw);
        let (sp, cp) = math::sin_cos(self.pitch);
        let offset = Vec3::new(sy * cp, sp, cy * cp) * self.distance;
        self.eye = self.target + offset;
        self.forward = (self.target - self.eye).try_normalize().unwrap_or(Vec3::NEG_Z);
        self.right = self
            .forward
            .cross(Vec3::Y)
            .try_normalize()
            .unwrap_or(Vec3::X);
        self.up = self.right.cross(self.forward).normalize_or_zero();
        self.fov_y = if portrait { 0.92 } else { 0.70 };
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StirState {
    pub active: bool,
    pub world_pos: Vec3,
    pub world_vel: Vec3,
    pub radius: f32,
    pub strength: f32,
    pub had_prev: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PerfMeter {
    pub frames: u32,
    pub accum: f64,
    pub displayed_fps: u32,
}

impl Default for PerfMeter {
    fn default() -> Self {
        Self {
            frames: 0,
            accum: 0.0,
            displayed_fps: 60,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UiFocus {
    pub hovered: Option<usize>,
    pub pressed: Option<usize>,
    pub captured_pointer: bool,
    pub last_touch_mid: Option<Vec2>,
    pub last_pinch_dist: Option<f32>,
}

#[derive(Resource)]
pub struct QuicksilverState {
    pub material: MaterialKind,
    pub vortex: bool,
    pub zero_g: bool,
    pub orbit_mode: bool,
    pub preset: ScalePreset,
    pub preset_locked: bool,
    pub auto_checked: bool,
    pub drop_counter: u32,
    pub drop_seed: u32,
    pub reset_counter: u32,
    pub camera: CameraRig,
    pub stir: StirState,
    pub perf: PerfMeter,
    pub ui: UiFocus,
    pub viewport: [f32; 2],
    pub pixel_ratio: f32,
    pub is_coarse: bool,
    pub elapsed: f32,
    pub ui_revision: u64,
}

impl Default for QuicksilverState {
    fn default() -> Self {
        Self {
            material: MaterialKind::Chrome,
            vortex: false,
            zero_g: false,
            orbit_mode: false,
            preset: ScalePreset::High,
            preset_locked: false,
            auto_checked: false,
            drop_counter: 0,
            drop_seed: 1,
            reset_counter: 1,
            camera: CameraRig::default(),
            stir: StirState {
                radius: 0.44,
                ..StirState::default()
            },
            perf: PerfMeter::default(),
            ui: UiFocus::default(),
            viewport: [1280.0, 720.0],
            pixel_ratio: 1.0,
            is_coarse: false,
            elapsed: 0.0,
            ui_revision: 1,
        }
    }
}

impl QuicksilverState {
    pub fn trigger_drop(&mut self) {
        self.drop_counter = self.drop_counter.wrapping_add(1);
        self.drop_seed = self.drop_seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.ui_revision = self.ui_revision.wrapping_add(1);
    }

    pub fn set_preset(&mut self, preset: ScalePreset) {
        self.preset_locked = true;
        if self.preset != preset {
            self.preset = preset;
            self.reset_counter = self.reset_counter.wrapping_add(1);
            self.ui_revision = self.ui_revision.wrapping_add(1);
        }
    }

    pub fn trigger_reset(&mut self) {
        self.reset_counter = self.reset_counter.wrapping_add(1);
        self.ui_revision = self.ui_revision.wrapping_add(1);
    }

    pub fn world_to_grid(p: Vec3) -> Vec3 {
        Vec3::new(
            p.x / CELL_SIZE + (GRID_X as f32) * 0.5,
            (p.y - BASIN_FLOOR_Y) / CELL_SIZE + 3.0,
            p.z / CELL_SIZE + (GRID_Z as f32) * 0.5,
        )
    }
}
