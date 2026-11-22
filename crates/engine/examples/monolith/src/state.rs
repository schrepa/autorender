use engine::prelude::*;

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
