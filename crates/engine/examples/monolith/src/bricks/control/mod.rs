engine::brick!(control);

use brick_nanite::{
    NaniteCamera, NaniteConfig, NaniteTelemetry, ViewMode, reverse_z_infinite_proj,
};
use engine::math;
use engine::prelude::*;
use engine_core::{DevSetting, DeviceLimits, KeyCode, Stats};
use engine_touch::{LookPadSpec, TouchLayout, TouchState, VirtualStickSpec};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::names::{NEAR_Z, SUN_DIR};
use crate::state::{CamBookmark, HeapPreset, LodPreset, MonolithState, view_mode_label};

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Control {}

settings! {
    #[dev]
    #[section("monolith")]
    pub struct MonolithDevControl {
        pub bookmark: String = "atrium".to_owned() => any,
        pub view_mode: String = "lit".to_owned() => any,
        pub freeze_culling: bool = false => any,
        pub hw_only: bool = true => any,
        pub heap_mb: u32 = 128 => any,
        pub lod_px: u32 = 1 => any,
        pub hud_hidden: bool = false => any,
        pub shadows: bool = true => any,
        pub dolly: bool = false => any,
    }
}

stats! {
    #[section("monolith_stats")]
    pub struct MonolithStats {
        pub assets_loaded: bool,
        pub frame_count: u64,
        pub fps: f32,
        pub bookmark: String,
        pub view_mode: String,
        pub freeze_culling: bool,
        pub total_virtual_tris: u64,
        pub visible_tris: u32,
        pub tris_hw: u32,
        pub tris_sw: u32,
        pub hw_percent: u32,
        pub sw_percent: u32,
        pub resident_pages: u32,
        pub total_pages: u32,
        pub budget_pages: u32,
        pub pending_fetches: u32,
        pub inst_p1_vis: u32,
        pub inst_p1_occ: u32,
        pub inst_p2_vis: u32,
        pub clusters_tested: u32,
        pub hw_clusters: u32,
        pub sw_clusters: u32,
        pub page_faults: u32,
        pub fallback_draws: u32,
        pub cone_culled: u32,
        pub frustum_culled: u32,
        pub hiz_culled: u32,
    }
}

#[derive(Resource)]
pub struct CameraRig {
    pub eye: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_deg: f32,
    pub trans_active: bool,
    pub trans_t: f32,
    pub trans_dur: f32,
    pub start_eye: Vec3,
    pub start_yaw: f32,
    pub start_pitch: f32,
    pub start_fov: f32,
    pub end_eye: Vec3,
    pub end_yaw: f32,
    pub end_pitch: f32,
    pub end_fov: f32,
    pub dragging: bool,
    pub dolly_time: f32,
    pub last_view_mode: ViewMode,
    pub last_dev: Option<MonolithDevControl>,
}

fn look_angles(eye: Vec3, target: Vec3) -> (f32, f32) {
    let dir = (target - eye).normalize_or_zero();
    let yaw = math::atan2(-dir.x, -dir.z);
    let pitch = math::clamp(dir.y, -0.995, 0.995).asin();
    (yaw, pitch)
}

fn shortest_angle_diff(from: f32, to: f32) -> f32 {
    let pi = std::f32::consts::PI;
    let tau = std::f32::consts::TAU;
    let mut d = (to - from) % tau;
    if d > pi {
        d -= tau;
    } else if d < -pi {
        d += tau;
    }
    d
}

impl Default for CameraRig {
    fn default() -> Self {
        let (eye, target, fov) = CamBookmark::Atrium.pose();
        let (yaw, pitch) = look_angles(eye, target);
        Self {
            eye,
            yaw,
            pitch,
            fov_deg: fov,
            trans_active: false,
            trans_t: 0.0,
            trans_dur: 0.85,
            start_eye: eye,
            start_yaw: yaw,
            start_pitch: pitch,
            start_fov: fov,
            end_eye: eye,
            end_yaw: yaw,
            end_pitch: pitch,
            end_fov: fov,
            dragging: false,
            dolly_time: 0.0,
            last_view_mode: ViewMode::Lit,
            last_dev: None,
        }
    }
}

impl CameraRig {
    fn snap_bookmark(&mut self, bm: CamBookmark) {
        let (eye, target, fov) = bm.pose();
        let (yaw, pitch) = look_angles(eye, target);
        self.eye = eye;
        self.yaw = yaw;
        self.pitch = pitch;
        self.fov_deg = fov;
        self.trans_active = false;
        self.dolly_time = 0.0;
    }

    fn start_transition(&mut self, bm: CamBookmark) {
        let (eye, target, fov) = bm.pose();
        let (yaw, pitch) = look_angles(eye, target);
        self.start_eye = self.eye;
        self.start_yaw = self.yaw;
        self.start_pitch = self.pitch;
        self.start_fov = self.fov_deg;
        self.end_eye = eye;
        self.end_yaw = self.yaw + shortest_angle_diff(self.yaw, yaw);
        self.end_pitch = pitch;
        self.end_fov = fov;
        self.trans_t = 0.0;
        self.trans_dur = 0.85;
        self.trans_active = true;
        self.dolly_time = 0.0;
    }
}

pub struct ControlBrick;

impl Brick for ControlBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<Control>("control").schema(engine_core::schema_of::<Control>),
            )
            .moves::<Control>(Pace::Frame)
            .resource::<MonolithState>()
            .resource::<CameraRig>()
            .systems(Schedule::Presentation, ensure_camera)
            .systems(Schedule::Presentation, update_control)
            .dev(|dev| {
                dev.settings::<MonolithDevControl>()
                    .stats::<MonolithStats>()
                    .systems(Schedule::Presentation, publish_dev_stats);
            });
    }
}

fn lens_of(proj: Mat4) -> Camera {
    Camera {
        near: NEAR_Z,
        far: 10_000.0,
        projection: Projection::Fitted {
            clip: proj.to_cols_array_2d(),
        },
    }
}

fn ensure_camera(
    eyes: Query<Entity, (With<Control>, Without<Camera>)>,
    mut spawner: Spawner<ControlBrick>,
) {
    let proj = reverse_z_infinite_proj(42.0_f32.to_radians(), 16.0 / 9.0, NEAR_Z);
    for eye in &eyes {
        spawner.insert(eye, lens_of(proj));
    }
}

fn parse_bookmark(s: &str) -> Option<CamBookmark> {
    match s.to_ascii_lowercase().as_str() {
        "atrium" | "vista" => Some(CamBookmark::Atrium),
        "gallery" | "sentinel" => Some(CamBookmark::Gallery),
        "ivy" | "canyon" => Some(CamBookmark::Ivy),
        "lion" | "corridor" => Some(CamBookmark::Lion),
        _ => None,
    }
}

fn parse_view_mode(s: &str) -> Option<ViewMode> {
    match s.to_ascii_lowercase().as_str() {
        "lit" => Some(ViewMode::Lit),
        "meshlets" => Some(ViewMode::Meshlets),
        "tris" | "triangles" => Some(ViewMode::Triangles),
        "lod" => Some(ViewMode::Lod),
        "hw_sw" | "hwsw" | "hw/sw" => Some(ViewMode::HwSw),
        "pages" => Some(ViewMode::Pages),
        _ => None,
    }
}

fn apply_dev_control(
    dev: &MonolithDevControl,
    state: &mut MonolithState,
    cfg: &mut NaniteConfig,
    rig: &mut CameraRig,
) {
    let prev = rig.last_dev.clone().unwrap_or_default();
    if dev.bookmark != prev.bookmark
        && let Some(bm) = parse_bookmark(&dev.bookmark)
    {
        state.bookmark = bm;
        state.requested_bookmark = None;
        state.taa_reset = true;
        rig.snap_bookmark(bm);
    }
    if dev.view_mode != prev.view_mode
        && let Some(vm) = parse_view_mode(&dev.view_mode)
    {
        cfg.view_mode = vm;
        state.taa_reset = true;
    }
    if dev.freeze_culling != prev.freeze_culling {
        cfg.freeze_culling = dev.freeze_culling;
    }
    if dev.hw_only != prev.hw_only {
        cfg.hw_only = dev.hw_only;
    }
    if dev.heap_mb != prev.heap_mb {
        cfg.heap_budget_mb = HeapPreset::from_mb(dev.heap_mb).mb();
    }
    if dev.lod_px != prev.lod_px {
        let preset = match dev.lod_px {
            1 => LodPreset::Px1,
            4 => LodPreset::Px4,
            _ => LodPreset::Px2,
        };
        cfg.lod_error_px = preset.base_px() * state.lod_error_scale;
    }
    if dev.hud_hidden != prev.hud_hidden {
        state.hud_hidden = dev.hud_hidden;
    }
    if dev.shadows != prev.shadows {
        cfg.shadow.enabled = dev.shadows;
    }
    if dev.dolly != prev.dolly {
        state.dolly_enabled = dev.dolly;
    }
    rig.last_dev = Some(dev.clone());
}

fn handle_shortcuts(
    devices: &Devices,
    state: &mut MonolithState,
    cfg: &mut NaniteConfig,
    heap_cap_pages: u32,
) {
    let kb = &devices.keyboard;
    if kb.pressed(KeyCode::Digit1) {
        cfg.view_mode = ViewMode::Lit;
    } else if kb.pressed(KeyCode::Digit2) {
        cfg.view_mode = ViewMode::Meshlets;
    } else if kb.pressed(KeyCode::Digit3) {
        cfg.view_mode = ViewMode::Triangles;
    } else if kb.pressed(KeyCode::Digit4) {
        cfg.view_mode = ViewMode::Lod;
    } else if kb.pressed(KeyCode::Digit5) {
        cfg.view_mode = ViewMode::HwSw;
    } else if kb.pressed(KeyCode::Digit6) {
        cfg.view_mode = ViewMode::Pages;
    }

    if kb.pressed(KeyCode::KeyV) || kb.pressed(KeyCode::F1) {
        state.bookmark = CamBookmark::Atrium;
        state.requested_bookmark = Some(CamBookmark::Atrium);
    } else if kb.pressed(KeyCode::KeyB) || kb.pressed(KeyCode::F2) {
        state.bookmark = CamBookmark::Gallery;
        state.requested_bookmark = Some(CamBookmark::Gallery);
    } else if kb.pressed(KeyCode::KeyN) || kb.pressed(KeyCode::F3) {
        state.bookmark = CamBookmark::Ivy;
        state.requested_bookmark = Some(CamBookmark::Ivy);
    } else if kb.pressed(KeyCode::KeyM) || kb.pressed(KeyCode::F4) {
        state.bookmark = CamBookmark::Lion;
        state.requested_bookmark = Some(CamBookmark::Lion);
    }

    if kb.pressed(KeyCode::KeyF) {
        cfg.freeze_culling = !cfg.freeze_culling;
    }
    if kb.pressed(KeyCode::KeyG) {
        cfg.heap_budget_mb = HeapPreset::from_mb(cfg.heap_budget_mb)
            .next_within(heap_cap_pages)
            .mb();
    }
    if kb.pressed(KeyCode::KeyL) {
        let next = LodPreset::from_scaled_px(cfg.lod_error_px, state.lod_error_scale).next();
        cfg.lod_error_px = next.base_px() * state.lod_error_scale;
    }
    if kb.pressed(KeyCode::KeyR) {
        cfg.hw_only = !cfg.hw_only;
    }
    if kb.pressed(KeyCode::KeyH) {
        state.hud_hidden = !state.hud_hidden;
    }
    if kb.pressed(KeyCode::Space) {
        state.dolly_enabled = !state.dolly_enabled;
    }
}

fn sync_touch_layout(
    state: &MonolithState,
    profile: Option<InputProfile>,
    layout: &mut TouchLayout,
) {
    let touch_active = profile.is_some_and(|p| p.active == InputModality::Touch);
    let want = touch_active && !state.hud_hidden;
    if layout.enabled != want {
        layout.enabled = want;
        if want {
            layout.stick = Some(VirtualStickSpec {
                zone_max_x_frac: 0.44,
                anchor_dp: [84.0, 168.0],
                radius_dp: 50.0,
                knob_radius_dp: 21.0,
                deadzone: 0.12,
                floating: true,
            });
            layout.look = Some(LookPadSpec {
                zone_min_x_frac: 0.44,
                sensitivity: 1.0,
            });
            layout.buttons.clear();
        } else {
            layout.stick = None;
            layout.look = None;
            layout.buttons.clear();
        }
    }
}

fn clamp_floor(eye: Vec3) -> Vec3 {
    Vec3::new(
        math::clamp(eye.x, -15.2, 18.8),
        math::clamp(eye.y, 0.55, 16.5),
        math::clamp(eye.z, -8.5, 11.5),
    )
}

fn step_camera(
    state: &mut MonolithState,
    rig: &mut CameraRig,
    devices: &Devices,
    pointer: &Pointer,
    touch: Option<&mut TouchState>,
    dt: f32,
) -> (Vec3, Quat, f32) {
    if let Some(bm) = state.requested_bookmark.take() {
        rig.start_transition(bm);
    }

    let kb = &devices.keyboard;
    let move_f = f32::from(kb.held(KeyCode::KeyW) || kb.held(KeyCode::ArrowUp))
        - f32::from(kb.held(KeyCode::KeyS) || kb.held(KeyCode::ArrowDown));
    let move_r = f32::from(kb.held(KeyCode::KeyD) || kb.held(KeyCode::ArrowRight))
        - f32::from(kb.held(KeyCode::KeyA) || kb.held(KeyCode::ArrowLeft));
    let move_u = f32::from(kb.held(KeyCode::KeyE)) - f32::from(kb.held(KeyCode::KeyQ));
    let stick = touch.as_ref().map_or(Vec2::ZERO, |t| t.move_axis);

    let allow_drag = !state.ui_captured_pointer;
    if pointer.pressed && allow_drag {
        rig.dragging = true;
    }
    if !pointer.down {
        rig.dragging = false;
    }
    let touch_look = touch.map_or(Vec2::ZERO, TouchState::take_look_delta_dp);
    let drag_delta = if rig.dragging && !touch_look.length_squared().is_normal() {
        pointer.delta_pixels
    } else {
        Vec2::ZERO
    };
    let right_drag = if devices.mouse.buttons.held(MouseButton::Right) {
        devices.mouse.motion_pixels
    } else {
        Vec2::ZERO
    };
    let look = drag_delta + right_drag + touch_look * 1.6;
    let wheel = devices.mouse.wheel_notches.y;

    let manual = move_f.abs() > 0.01
        || move_r.abs() > 0.01
        || move_u.abs() > 0.01
        || stick.length_squared() > 0.01
        || look.length_squared() > 0.25
        || wheel.abs() > 0.001;
    if manual {
        rig.trans_active = false;
        state.dolly_enabled = false;
    }

    if rig.trans_active {
        rig.trans_t = math::min(rig.trans_t + dt / math::max(rig.trans_dur, 0.05), 1.0);
        let u = rig.trans_t * rig.trans_t * (3.0 - 2.0 * rig.trans_t);
        rig.eye = rig.start_eye.lerp(rig.end_eye, u);
        rig.yaw = rig.start_yaw + (rig.end_yaw - rig.start_yaw) * u;
        rig.pitch = rig.start_pitch + (rig.end_pitch - rig.start_pitch) * u;
        rig.fov_deg = rig.start_fov + (rig.end_fov - rig.start_fov) * u;
        if rig.trans_t >= 1.0 {
            rig.trans_active = false;
        }
    } else if state.dolly_enabled {
        rig.dolly_time += dt;
        match state.bookmark {
            CamBookmark::Atrium => {
                let phase = 0.5 - 0.5 * math::cos(rig.dolly_time * std::f32::consts::PI / 32.0);
                rig.eye = Vec3::new(7.1 - 8.5 * phase, 1.95 + 0.30 * phase, -0.10);
                let (yaw, pitch) = look_angles(rig.eye, Vec3::new(-15.0, 2.8, 0.0));
                rig.yaw = yaw;
                rig.pitch = pitch;
            }
            CamBookmark::Gallery => {
                let phase = 0.5 - 0.5 * math::cos(rig.dolly_time * std::f32::consts::PI / 28.0);
                rig.eye = Vec3::new(8.8 - 8.0 * phase, 6.9, -4.4);
                let (yaw, pitch) = look_angles(rig.eye, Vec3::new(-7.5, 5.1, 1.8));
                rig.yaw = yaw;
                rig.pitch = pitch;
            }
            CamBookmark::Ivy | CamBookmark::Lion => {}
        }
    } else if manual {
        rig.yaw -= look.x * 0.0038;
        rig.pitch = math::clamp(rig.pitch - look.y * 0.0032, -1.42, 1.42);
        let rot = Quat::from_rotation_y(rig.yaw) * Quat::from_rotation_x(rig.pitch);
        let fwd = (rot * Vec3::NEG_Z).normalize_or_zero();
        let right = (rot * Vec3::X).normalize_or_zero();
        let speed = if kb.held(KeyCode::ShiftLeft) || kb.held(KeyCode::ShiftRight) {
            12.0
        } else {
            4.5
        };
        let wish = fwd * math::clamp(move_f + stick.y, -1.0, 1.0)
            + right * math::clamp(move_r + stick.x, -1.0, 1.0)
            + Vec3::Y * move_u;
        if wish.length_squared() > 0.0001 {
            rig.eye += wish.normalize() * (speed * dt);
        }
        if wheel.abs() > 0.001 {
            rig.eye += fwd * (wheel * 1.2);
        }
        rig.eye = clamp_floor(rig.eye);
    }

    let rot = Quat::from_rotation_y(rig.yaw) * Quat::from_rotation_x(rig.pitch);
    (rig.eye, rot, rig.fov_deg)
}

#[derive(SystemParam)]
struct ControlInput<'w> {
    time: Res<'w, FrameTime>,
    devices: Res<'w, Devices>,
    pointer: Res<'w, Pointer>,
    views: Res<'w, Views>,
    limits: Res<'w, DeviceLimits>,
    profile: Option<Res<'w, InputProfile>>,
    touch_layout: Option<ResMut<'w, TouchLayout>>,
    touch_state: Option<ResMut<'w, TouchState>>,
    dev: DevSetting<'w, MonolithDevControl>,
}

fn halton(mut idx: u32, base: u32) -> f32 {
    let mut f = 1.0_f32;
    let mut r = 0.0_f32;
    let b = base as f32;
    while idx > 0 {
        f /= b;
        r += f * (idx % base) as f32;
        idx /= base;
    }
    r
}

struct CameraFrame {
    eye: Vec3,
    rot: Quat,
    eff_fov: f32,
    aspect: f32,
    display: [u32; 2],
}

fn apply_camera_matrices(
    cf: CameraFrame,
    state: &mut MonolithState,
    cfg: &NaniteConfig,
    nanite_cam: &mut NaniteCamera,
) -> Mat4 {
    let CameraFrame {
        eye,
        rot,
        eff_fov,
        aspect,
        display,
    } = cf;
    let forward = (rot * Vec3::NEG_Z).normalize_or_zero();
    let right = (rot * Vec3::X).normalize_or_zero();
    let up = (rot * Vec3::Y).normalize_or_zero();
    let view = Mat4::look_to_rh(eye, forward, up);
    let proj = reverse_z_infinite_proj(eff_fov.to_radians(), aspect, NEAR_Z);
    let vp = proj * view;
    let inv_vp = vp.inverse();

    let (jx, jy) = if cfg.view_mode == ViewMode::Lit {
        let idx = ((state.frame_count % 16) as u32) + 1;
        (halton(idx, 2) - 0.5, halton(idx, 3) - 0.5)
    } else {
        (0.0, 0.0)
    };
    let rw = math::max((display[0] as f32 * state.render_scale).round(), 1.0);
    let rh = math::max((display[1] as f32 * state.render_scale).round(), 1.0);
    let jitter_mat = Mat4::from_translation(Vec3::new(jx * 2.0 / rw, -jy * 2.0 / rh, 0.0));
    let render_proj = jitter_mat * proj;

    state.prev_unjittered_vp = if state.frame_count <= 1 {
        vp
    } else {
        state.view_projection
    };
    state.jitter_px = [jx, jy];
    state.camera_eye = eye;
    state.camera_forward = forward;
    state.camera_right = right;
    state.camera_up = up;
    state.fov_deg = eff_fov;
    state.view_projection = vp;
    state.inv_view_projection = inv_vp;

    if cfg.freeze_culling {
        if !state.was_frozen {
            state.frozen_cull_pos = eye;
            state.frozen_cull_view = view;
            state.frozen_cull_proj = proj;
            state.was_frozen = true;
        }
    } else {
        state.was_frozen = false;
    }

    nanite_cam.render_pos = eye;
    nanite_cam.render_view = view;
    nanite_cam.render_proj = render_proj;
    nanite_cam.near_z = NEAR_Z;
    nanite_cam.sim_time = state.sim_time;
    nanite_cam.sun_dir = Vec3::from_array(SUN_DIR).normalize();
    if cfg.freeze_culling {
        nanite_cam.cull_pos = state.frozen_cull_pos;
        nanite_cam.cull_view = state.frozen_cull_view;
        nanite_cam.cull_proj = state.frozen_cull_proj;
    } else {
        nanite_cam.cull_pos = eye;
        nanite_cam.cull_view = view;
        nanite_cam.cull_proj = proj;
    }
    proj
}

fn update_control(
    mut input: ControlInput<'_>,
    mut state: ResMut<MonolithState>,
    mut cfg: ResMut<NaniteConfig>,
    mut nanite_cam: ResMut<NaniteCamera>,
    mut rig: ResMut<CameraRig>,
    mut eyes: Query<(Entity, &mut Camera), With<Control>>,
    mut moves: Moves<ControlBrick>,
) {
    let raw_dt = input.time.real_seconds as f32;
    let dt = math::clamp(raw_dt, 0.001, 0.05);
    state.dt = dt;
    state.sim_time += dt;
    state.frame_count = state.frame_count.saturating_add(1);
    let inst_fps = math::clamp(1.0 / math::max(raw_dt, 0.001), 1.0, 480.0);
    state.fps_smoothed += (inst_fps - state.fps_smoothed) * math::clamp(dt * 4.5, 0.02, 0.25);

    state.taa_reset = state.frame_count <= 2;

    if let Some(dev) = input.dev.get()
        && rig.last_dev.as_ref() != Some(dev)
    {
        apply_dev_control(dev, &mut state, &mut cfg, &mut rig);
    }

    let heap_cap_pages = cfg.effective_heap_cap_pages(input.limits.storage_room);
    handle_shortcuts(&input.devices, &mut state, &mut cfg, heap_cap_pages);
    cfg.heap_budget_mb = HeapPreset::from_mb(cfg.heap_budget_mb)
        .clamp_to_pages(heap_cap_pages)
        .mb();
    if cfg.view_mode != rig.last_view_mode {
        state.taa_reset = true;
        rig.last_view_mode = cfg.view_mode;
    }

    if let Some(layout) = input.touch_layout.as_deref_mut() {
        sync_touch_layout(&state, input.profile.as_deref().copied(), layout);
    }

    let (eye, rot, fov_deg) = step_camera(
        &mut state,
        &mut rig,
        &input.devices,
        &input.pointer,
        input.touch_state.as_deref_mut(),
        dt,
    );

    let frame = input.views.frame();
    let logical = frame.logical_size();
    let touch_active = input
        .profile
        .as_deref()
        .is_some_and(|p| p.active == InputModality::Touch);
    let constrained = touch_active
        || (input.limits.storage_room > 0 && input.limits.storage_room <= 128 * 1024 * 1024);
    #[cfg(target_arch = "wasm32")]
    {
        let phys_px = (logical.x * frame.pixel_ratio) * (logical.y * frame.pixel_ratio);
        let max_px = if constrained { 1_050_000.0_f32 } else { 2_600_000.0_f32 };
        state.render_scale = if phys_px > max_px {
            math::clamp((max_px / phys_px).sqrt(), 0.5, 1.0)
        } else {
            1.0
        };
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        state.render_scale = 1.0;
    }
    cfg.render_scale = state.render_scale;
    cfg.lod_max_viewport_h = if constrained { 1080 } else { 1440 };
    cfg.shadow.c0_half = 14.0;
    cfg.shadow.c0_forward = 6.0;
    cfg.shadow.c0_depth = 48.0;
    cfg.shadow.c0_y_bounds = Some([-2.0, 26.0]);
    cfg.shadow.c1_half = 32.0;
    cfg.shadow.c1_center = Some(Vec3::new(1.0, 6.0, 0.0));
    cfg.shadow.c1_depth = 64.0;
    let eff_fov = if logical.x > 0.0 && logical.x < logical.y {
        math::min(fov_deg + 14.0, 78.0)
    } else {
        fov_deg
    };
    let aspect = input.views.main().map_or(16.0 / 9.0, ViewFacts::aspect);
    let proj = apply_camera_matrices(
        CameraFrame {
            eye,
            rot,
            eff_fov,
            aspect,
            display: [frame.display_size.x, frame.display_size.y],
        },
        &mut state,
        &cfg,
        &mut nanite_cam,
    );

    let lens = lens_of(proj);
    for (entity, mut camera) in &mut eyes {
        camera.set_if_neq(lens);
        if let Some(mut transform) = moves.get_mut(entity) {
            let placed = Transform {
                translation: eye,
                rotation: rot,
                scale: transform.scale,
            };
            transform.set_if_neq(placed);
        }
    }
}

fn publish_dev_stats(
    state: Res<MonolithState>,
    cfg: Res<NaniteConfig>,
    tel: Res<NaniteTelemetry>,
    mut stats: Stats<'_, MonolithStats>,
) {
    stats.show(|| MonolithStats {
        assets_loaded: state.assets_loaded,
        frame_count: state.frame_count,
        fps: state.fps_smoothed,
        bookmark: state.bookmark.label().to_owned(),
        view_mode: view_mode_label(cfg.view_mode).to_owned(),
        freeze_culling: cfg.freeze_culling,
        total_virtual_tris: tel.total_virtual_tris,
        visible_tris: tel.visible_tris(),
        tris_hw: tel.tris_hw,
        tris_sw: tel.tris_sw,
        hw_percent: tel.hw_percent(),
        sw_percent: tel.sw_percent(),
        resident_pages: tel.resident_pages,
        total_pages: tel.total_pages,
        budget_pages: tel.budget_pages,
        pending_fetches: tel.pending_fetches,
        inst_p1_vis: tel.inst_p1_vis,
        inst_p1_occ: tel.inst_p1_occ,
        inst_p2_vis: tel.inst_p2_vis,
        clusters_tested: tel.clusters_tested,
        hw_clusters: tel.hw_clusters,
        sw_clusters: tel.sw_clusters,
        page_faults: tel.page_faults,
        fallback_draws: tel.fallback_draws,
        cone_culled: tel.cone_culled,
        frustum_culled: tel.frustum_culled,
        hiz_culled: tel.hiz_culled,
    });
}
