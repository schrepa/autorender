use brick_nanite::{ViewMode, pages_for_mb};
use engine::prelude::*;

#[must_use]
pub const fn view_mode_label(mode: ViewMode) -> &'static str {
    match mode {
        ViewMode::Lit => "LIT",
        ViewMode::Meshlets => "MESHLETS",
        ViewMode::Triangles => "TRIS",
        ViewMode::Lod => "LOD",
        ViewMode::HwSw => "HW/SW",
        ViewMode::Pages => "PAGES",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HeapPreset {
    Mb16,
    Mb32,
    Mb64,
    #[default]
    Mb128,
}

impl HeapPreset {
    #[must_use]
    pub const fn from_mb(mb: u32) -> Self {
        match mb {
            0..=16 => Self::Mb16,
            17..=32 => Self::Mb32,
            33..=64 => Self::Mb64,
            _ => Self::Mb128,
        }
    }

    #[must_use]
    pub const fn mb(self) -> u32 {
        match self {
            Self::Mb16 => 16,
            Self::Mb32 => 32,
            Self::Mb64 => 64,
            Self::Mb128 => 128,
        }
    }

    #[must_use]
    pub const fn pages(self) -> u32 {
        pages_for_mb(self.mb())
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Mb16 => "HEAP 16M",
            Self::Mb32 => "HEAP 32M",
            Self::Mb64 => "HEAP 64M",
            Self::Mb128 => "HEAP 128M",
        }
    }

    #[must_use]
    pub const fn next_within(self, max_pages: u32) -> Self {
        let cand = match self {
            Self::Mb16 => Self::Mb32,
            Self::Mb32 => Self::Mb64,
            Self::Mb64 => Self::Mb128,
            Self::Mb128 => Self::Mb16,
        };
        if cand.pages() <= max_pages {
            cand
        } else {
            Self::Mb16
        }
    }

    #[must_use]
    pub const fn clamp_to_pages(self, max_pages: u32) -> Self {
        if self.pages() <= max_pages {
            self
        } else if Self::Mb64.pages() <= max_pages {
            Self::Mb64
        } else if Self::Mb32.pages() <= max_pages {
            Self::Mb32
        } else {
            Self::Mb16
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LodPreset {
    Px1,
    #[default]
    Px2,
    Px4,
}

impl LodPreset {
    #[must_use]
    pub const fn base_px(self) -> f32 {
        match self {
            Self::Px1 => 1.0,
            Self::Px2 => 2.0,
            Self::Px4 => 4.0,
        }
    }

    #[must_use]
    pub fn from_scaled_px(px: f32, scale: f32) -> Self {
        let base = px / engine::math::max(scale, 0.01);
        if base <= 1.5 {
            Self::Px1
        } else if base <= 3.0 {
            Self::Px2
        } else {
            Self::Px4
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Px1 => "LOD 1PX",
            Self::Px2 => "LOD 2PX",
            Self::Px4 => "LOD 4PX",
        }
    }

    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Px1 => Self::Px2,
            Self::Px2 => Self::Px4,
            Self::Px4 => Self::Px1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CamBookmark {
    #[default]
    Atrium,
    Gallery,
    Ivy,
    Lion,
}

impl CamBookmark {
    pub const ALL: [Self; 4] = [Self::Atrium, Self::Gallery, Self::Ivy, Self::Lion];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Atrium => "ATRIUM",
            Self::Gallery => "GALLERY",
            Self::Ivy => "IVY",
            Self::Lion => "LION",
        }
    }

    #[must_use]
    pub fn pose(self) -> (Vec3, Vec3, f32) {
        match self {
            Self::Atrium => (
                Vec3::new(7.1, 1.95, -0.10),
                Vec3::new(-15.0, 2.8, 0.0),
                50.0,
            ),
            Self::Gallery => (
                Vec3::new(8.8, 6.9, -4.4),
                Vec3::new(-7.5, 5.1, 1.8),
                50.0,
            ),
            Self::Ivy => (
                Vec3::new(-1.6, 5.6, 1.2),
                Vec3::new(-9.0, 8.2, 2.2),
                48.0,
            ),
            Self::Lion => (
                Vec3::new(-10.6, 2.35, 0.35),
                Vec3::new(-15.2, 2.25, 0.0),
                44.0,
            ),
        }
    }
}

#[derive(Resource)]
pub struct MonolithState {
    pub bookmark: CamBookmark,
    pub requested_bookmark: Option<CamBookmark>,
    pub dolly_enabled: bool,
    pub hud_hidden: bool,
    pub ui_captured_pointer: bool,
    pub ui_hovered: bool,
    pub sim_time: f32,
    pub dt: f32,
    pub fps_smoothed: f32,
    pub frame_count: u64,
    pub render_scale: f32,
    pub camera_eye: Vec3,
    pub camera_forward: Vec3,
    pub camera_right: Vec3,
    pub camera_up: Vec3,
    pub fov_deg: f32,
    pub view_projection: Mat4,
    pub inv_view_projection: Mat4,
    pub prev_unjittered_vp: Mat4,
    pub jitter_px: [f32; 2],
    pub taa_reset: bool,
    pub frozen_cull_pos: Vec3,
    pub frozen_cull_view: Mat4,
    pub frozen_cull_proj: Mat4,
    pub was_frozen: bool,
    pub assets_loaded: bool,
    pub lod_error_scale: f32,
}

impl Default for MonolithState {
    fn default() -> Self {
        let (eye, target, fov) = CamBookmark::Atrium.pose();
        let fwd = (target - eye).normalize_or_zero();
        let right = fwd.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(fwd).normalize_or_zero();
        Self {
            bookmark: CamBookmark::Atrium,
            requested_bookmark: None,
            dolly_enabled: false,
            hud_hidden: false,
            ui_captured_pointer: false,
            ui_hovered: false,
            sim_time: 0.0,
            dt: 1.0 / 60.0,
            fps_smoothed: 60.0,
            frame_count: 0,
            render_scale: 1.0,
            camera_eye: eye,
            camera_forward: fwd,
            camera_right: right,
            camera_up: up,
            fov_deg: fov,
            view_projection: Mat4::IDENTITY,
            inv_view_projection: Mat4::IDENTITY,
            prev_unjittered_vp: Mat4::IDENTITY,
            jitter_px: [0.0, 0.0],
            taa_reset: true,
            frozen_cull_pos: eye,
            frozen_cull_view: Mat4::IDENTITY,
            frozen_cull_proj: Mat4::IDENTITY,
            was_frozen: false,
            assets_loaded: false,
            lod_error_scale: 1.0,
        }
    }
}

impl brick_present::Scale for MonolithState {
    fn fraction(&self) -> f32 {
        self.render_scale
    }

    fn direct_supported() -> bool {
        true
    }
}
