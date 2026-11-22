use engine::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CamBookmark {
    #[default]
    Vista,
    Sentinel,
    Canyon,
    Corridor,
}

impl CamBookmark {
    pub const ALL: [Self; 4] = [Self::Vista, Self::Sentinel, Self::Canyon, Self::Corridor];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Vista => "VISTA",
            Self::Sentinel => "SENTINEL",
            Self::Canyon => "CANYON",
            Self::Corridor => "CORRIDOR",
        }
    }

    #[must_use]
    pub fn pose(self) -> (Vec3, Vec3, f32) {
        match self {
            Self::Vista => (
                Vec3::new(0.0, 7.5, 160.0),
                Vec3::new(0.0, 14.0, -215.0),
                42.0,
            ),
            Self::Sentinel => (
                Vec3::new(6.5, 15.0, -186.0),
                Vec3::new(0.0, 18.4, -214.0),
                30.0,
            ),
            Self::Canyon => (
                Vec3::new(-150.0, 110.0, 170.0),
                Vec3::new(0.0, 0.0, -120.0),
                50.0,
            ),
            Self::Corridor => (
                Vec3::new(60.0, 4.2, 28.0),
                Vec3::new(60.0, 6.5, -120.0),
                70.0,
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
    pub frozen_cull_pos: Vec3,
    pub frozen_cull_view: Mat4,
    pub frozen_cull_proj: Mat4,
    pub was_frozen: bool,
    pub assets_loaded: bool,
}

impl Default for MonolithState {
    fn default() -> Self {
        let (eye, target, fov) = CamBookmark::Vista.pose();
        let fwd = (target - eye).normalize_or_zero();
        let right = fwd.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(fwd).normalize_or_zero();
        Self {
            bookmark: CamBookmark::Vista,
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
