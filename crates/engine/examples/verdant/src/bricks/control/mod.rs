engine::brick!(control);

use engine::math;
use engine::prelude::*;
use engine_core::KeyCode;
use engine_touch::{LookPadSpec, TouchLayout, TouchState, VirtualStickSpec};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::names::{DEFAULT_FOV_DEGREES, FAR, NEAR};
use crate::state::{CamMode, DensityPreset, TimePreset, VerdantState};
use crate::terrain::terrain_height;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Control {}

#[derive(Resource)]
pub struct CameraRig {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target_yaw: f32,
    pub target_pitch: f32,
    pub target_distance: f32,
    pub roam_pos: Vec3,
    pub roam_yaw: f32,
    pub roam_pitch: f32,
    pub user_interacted: bool,
    pub dragging: bool,
    pub last_cursor_world: Option<Vec3>,
    pub prev_body_xz: Vec2,
    pub prev_mode: CamMode,
}

impl Default for CameraRig {
    fn default() -> Self {
        let start_yaw = 0.12;
        let start_pitch = 0.125;
        let start_dist = 21.8;
        Self {
            target: Vec3::new(-0.2, 5.15, -0.8),
            yaw: start_yaw,
            pitch: start_pitch,
            distance: start_dist,
            target_yaw: start_yaw,
            target_pitch: start_pitch,
            target_distance: start_dist,
            roam_pos: Vec3::new(-0.5, terrain_height(-0.5, 16.5) + 1.68, 16.5),
            roam_yaw: 0.04,
            roam_pitch: 0.08,
            user_interacted: false,
            dragging: false,
            last_cursor_world: None,
            prev_body_xz: Vec2::new(-0.5, 16.5),
            prev_mode: CamMode::Orbit,
        }
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
            .resource::<VerdantState>()
            .resource::<CameraRig>()
            .systems(Schedule::Presentation, ensure_camera)
            .systems(Schedule::Presentation, update_control);
    }
}

fn lens_of(fov_degrees: f32) -> Camera {
    Camera {
        near: NEAR,
        far: FAR,
        projection: Projection::Perspective {
            vertical_fov_radians: fov_degrees.to_radians(),
            fit: FovFit::MinSpan,
        },
    }
}

fn ensure_camera(
    eyes: Query<Entity, (With<Control>, Without<Camera>)>,
    mut spawner: Spawner<ControlBrick>,
) {
    for eye in &eyes {
        spawner.insert(eye, lens_of(DEFAULT_FOV_DEGREES));
    }
}

fn ray_terrain_hit(origin: Vec3, dir: Vec3) -> Option<Vec3> {
    if dir.y > 0.25 {
        return None;
    }
    let mut t = 1.0_f32;
    for _ in 0..24 {
        let p = origin + dir * t;
        let h = terrain_height(p.x, p.z);
        let diff = p.y - h;
        if diff < 0.08 {
            return Some(Vec3::new(p.x, h, p.z));
        }
        t += math::clamp(diff * 0.72, 0.35, 3.8);
        if t > 52.0 {
            break;
        }
    }
    None
}

fn cursor_ray(pixel: Vec2, screen: Vec2, inv_vp: Mat4) -> (Vec3, Vec3) {
    let ndc_x = (pixel.x / math::max(screen.x, 1.0)) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel.y / math::max(screen.y, 1.0)) * 2.0;
    let near_h = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let far_h = inv_vp * Vec4::new(ndc_x, ndc_y, 0.01, 1.0);
    let near = near_h.truncate() / math::max(near_h.w.abs(), 1.0e-5) * near_h.w.signum();
    let far = far_h.truncate() / math::max(far_h.w.abs(), 1.0e-5) * far_h.w.signum();
    (near, (far - near).normalize_or_zero())
}

fn handle_shortcuts(devices: &Devices, state: &mut VerdantState, rig: &mut CameraRig) {
    let kb = &devices.keyboard;
    if kb.pressed(KeyCode::Digit1) {
        state.time_preset = TimePreset::Noon;
    } else if kb.pressed(KeyCode::Digit2) {
        state.time_preset = TimePreset::Gold;
    } else if kb.pressed(KeyCode::Digit3) {
        state.time_preset = TimePreset::Dusk;
    } else if kb.pressed(KeyCode::Digit4) {
        state.time_preset = TimePreset::Night;
    }
    if kb.pressed(KeyCode::KeyH) {
        state.hud_hidden = !state.hud_hidden;
    }
    if kb.pressed(KeyCode::KeyR) {
        *rig = CameraRig::default();
        state.cam_mode = CamMode::Orbit;
    }
    if kb.pressed(KeyCode::KeyV) {
        state.cam_mode = match state.cam_mode {
            CamMode::Orbit => CamMode::Roam,
            CamMode::Roam => CamMode::Orbit,
        };
    }
    if kb.pressed(KeyCode::KeyC) {
        rig.user_interacted = true;
        rig.target = Vec3::new(-0.15, 5.75, -1.8);
        rig.target_yaw = -0.14;
        rig.target_pitch = 0.07;
        rig.target_distance = 11.4;
        state.cam_mode = CamMode::Orbit;
    }
}

fn sync_touch_layout(
    state: &VerdantState,
    profile: Option<InputProfile>,
    layout: &mut TouchLayout,
) {
    let touch_active = profile.is_some_and(|p| p.active == InputModality::Touch);
    let want_roam = touch_active && state.cam_mode == CamMode::Roam && !state.hud_hidden;
    if layout.enabled != want_roam {
        layout.enabled = want_roam;
        if want_roam {
            layout.stick = Some(VirtualStickSpec {
                zone_max_x_frac: 0.44,
                anchor_dp: [84.0, 148.0],
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

fn compute_camera_pose(
    state: &mut VerdantState,
    rig: &mut CameraRig,
    devices: &Devices,
    pointer: &Pointer,
    touch: Option<&mut TouchState>,
    dt: f32,
) -> (Vec3, Quat) {
    let kb = &devices.keyboard;
    let move_f = f32::from(kb.held(KeyCode::KeyW) || kb.held(KeyCode::ArrowUp))
        - f32::from(kb.held(KeyCode::KeyS) || kb.held(KeyCode::ArrowDown));
    let move_r = f32::from(kb.held(KeyCode::KeyD) || kb.held(KeyCode::ArrowRight))
        - f32::from(kb.held(KeyCode::KeyA) || kb.held(KeyCode::ArrowLeft));
    let stick = touch.as_ref().map_or(Vec2::ZERO, |t| t.move_axis);
    let has_move = move_f.abs() > 0.01 || move_r.abs() > 0.01 || stick.length_squared() > 0.01;

    if has_move && state.cam_mode == CamMode::Orbit {
        state.cam_mode = CamMode::Roam;
        rig.user_interacted = true;
    }

    if state.cam_mode != rig.prev_mode {
        if state.cam_mode == CamMode::Roam {
            let eye = state.camera_eye;
            let ground = terrain_height(eye.x, eye.z);
            rig.roam_pos = Vec3::new(
                math::clamp(eye.x, -34.0, 34.0),
                ground + 1.68,
                math::clamp(eye.z, -24.0, 34.0),
            );
            let fwd = state.camera_forward;
            rig.roam_yaw = math::atan2(-fwd.x, -fwd.z);
            rig.roam_pitch = math::clamp(fwd.y, -0.8, 0.8).asin();
        } else {
            let to_cam = state.camera_eye - rig.target;
            let dist = math::clamp(to_cam.length(), 6.5, 36.0);
            rig.target_distance = dist;
            rig.distance = dist;
            rig.target_yaw = math::atan2(to_cam.x, to_cam.z);
            rig.yaw = rig.target_yaw;
            let horiz = math::max(Vec2::new(to_cam.x, to_cam.z).length(), 0.001);
            rig.target_pitch = math::clamp(math::atan2(to_cam.y, horiz), -0.02, 0.72);
            rig.pitch = rig.target_pitch;
        }
        rig.prev_mode = state.cam_mode;
    }

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
    let look = drag_delta + right_drag + touch_look * 1.65;
    if look.length_squared() > 0.25 {
        rig.user_interacted = true;
    }

    let wheel = devices.mouse.wheel_notches.y;
    if wheel.abs() > 0.001 {
        rig.user_interacted = true;
        rig.target_distance = math::clamp(rig.target_distance * (1.0 - wheel * 0.09), 5.5, 38.0);
    }

    let smooth = 1.0 - math::exp(-dt * 8.5);
    match state.cam_mode {
        CamMode::Orbit => {
            if !rig.user_interacted {
                let drift = math::sin(state.sim_time * 0.14) * 0.20;
                rig.target_yaw = 0.12 + drift;
                rig.target_pitch = 0.125 + math::sin(state.sim_time * 0.21 + 0.6) * 0.018;
                rig.target_distance = 21.8 + math::cos(state.sim_time * 0.17) * 0.85;
            } else {
                rig.target_yaw -= look.x * 0.0042;
                rig.target_pitch = math::clamp(rig.target_pitch + look.y * 0.0034, -0.02, 0.78);
            }
            rig.yaw += (rig.target_yaw - rig.yaw) * smooth;
            rig.pitch += (rig.target_pitch - rig.pitch) * smooth;
            rig.distance += (rig.target_distance - rig.distance) * smooth;

            let (sy, cy) = math::sin_cos(rig.yaw);
            let (sp, cp) = math::sin_cos(rig.pitch);
            let offset = Vec3::new(sy * cp, sp, cy * cp) * rig.distance;
            let mut eye = rig.target + offset;
            let floor_y = terrain_height(eye.x, eye.z) + 1.08;
            if eye.y < floor_y {
                eye.y = floor_y;
            }
            let dir = (rig.target - eye).normalize_or_zero();
            let look_yaw = math::atan2(-dir.x, -dir.z);
            let look_pitch = math::clamp(dir.y, -0.98, 0.98).asin();
            let rot = Quat::from_rotation_y(look_yaw) * Quat::from_rotation_x(look_pitch);
            (eye, rot)
        }
        CamMode::Roam => {
            rig.roam_yaw -= look.x * 0.0040;
            rig.roam_pitch = math::clamp(rig.roam_pitch - look.y * 0.0032, -0.85, 0.85);
            let (sy, cy) = math::sin_cos(rig.roam_yaw);
            let fwd_xz = Vec3::new(-sy, 0.0, -cy);
            let right_xz = Vec3::new(cy, 0.0, -sy);
            let input_f = math::clamp(move_f + stick.y, -1.0, 1.0);
            let input_r = math::clamp(move_r + stick.x, -1.0, 1.0);
            let wish = fwd_xz * input_f + right_xz * input_r;
            let speed = if kb.held(KeyCode::ShiftLeft) || kb.held(KeyCode::ShiftRight) {
                8.2
            } else {
                4.8
            };
            if wish.length_squared() > 0.0001 {
                let step = wish.normalize() * speed * dt;
                rig.roam_pos.x = math::clamp(rig.roam_pos.x + step.x, -35.0, 35.0);
                rig.roam_pos.z = math::clamp(rig.roam_pos.z + step.z, -24.0, 35.0);
            }
            let target_y = terrain_height(rig.roam_pos.x, rig.roam_pos.z) + 1.65;
            rig.roam_pos.y += (target_y - rig.roam_pos.y) * math::clamp(dt * 12.0, 0.0, 1.0);
            let rot = Quat::from_rotation_y(rig.roam_yaw) * Quat::from_rotation_x(rig.roam_pitch);
            (rig.roam_pos, rot)
        }
    }
}

#[derive(SystemParam)]
struct ControlInput<'w> {
    time: Res<'w, FrameTime>,
    devices: Res<'w, Devices>,
    pointer: Res<'w, Pointer>,
    views: Res<'w, Views>,
    profile: Option<Res<'w, InputProfile>>,
    touch_layout: Option<ResMut<'w, TouchLayout>>,
    touch_state: Option<ResMut<'w, TouchState>>,
}

fn update_control(
    mut input: ControlInput<'_>,
    mut state: ResMut<VerdantState>,
    mut rig: ResMut<CameraRig>,
    mut eyes: Query<(Entity, &mut Camera), With<Control>>,
    mut moves: Moves<ControlBrick>,
) {
    let raw_dt = input.time.real_seconds as f32;
    let dt = math::clamp(raw_dt, 0.001, 0.05);
    state.dt = dt;
    state.sim_time += dt;
    state.frame_count = state.frame_count.saturating_add(1);
    let inst_fps = math::clamp(1.0 / math::max(raw_dt, 0.001), 1.0, 240.0);
    state.fps_smoothed += (inst_fps - state.fps_smoothed) * math::clamp(dt * 4.0, 0.02, 0.25);

    let frame = input.views.frame();
    if !state.density_auto_set && frame.display_size.x > 0 {
        let coarse = input.profile.as_deref().is_some_and(|p| p.capabilities.coarse_pointer);
        let logical = frame.logical_size();
        if coarse || logical.x < 700.0 {
            state.density = DensityPreset::K250;
        } else {
            state.density = DensityPreset::M1;
        }
        state.density_auto_set = true;
    }

    handle_shortcuts(&input.devices, &mut state, &mut rig);

    let time_target = state.time_preset.value();
    let time_rate = 1.0 - math::exp(-dt * 2.8);
    state.time_phase += (time_target - state.time_phase) * time_rate;
    let wind_target = state.wind_preset.strength();
    state.wind_strength += (wind_target - state.wind_strength) * (1.0 - math::exp(-dt * 3.0));

    if let Some(layout) = input.touch_layout.as_deref_mut() {
        sync_touch_layout(&state, input.profile.as_deref().copied(), layout);
    }

    let (eye, rot) = compute_camera_pose(
        &mut state,
        &mut rig,
        &input.devices,
        &input.pointer,
        input.touch_state.as_deref_mut(),
        dt,
    );

    let fov_deg = if frame.logical_size().x > 0.0 && frame.logical_size().x < frame.logical_size().y {
        56.0
    } else {
        DEFAULT_FOV_DEGREES
    };
    let lens = lens_of(fov_deg);
    let aspect = input.views.main().map_or(16.0 / 9.0, ViewFacts::aspect);
    let forward = (rot * Vec3::NEG_Z).normalize_or_zero();
    let right = (rot * Vec3::X).normalize_or_zero();
    let up = (rot * Vec3::Y).normalize_or_zero();
    let vp = lens.clip(aspect) * Mat4::look_to_rh(eye, forward, up);
    let inv_vp = vp.inverse();

    state.camera_eye = eye;
    state.camera_forward = forward;
    state.camera_right = right;
    state.camera_up = up;
    state.view_projection = vp;
    state.inv_view_projection = inv_vp;

    let body_xz = Vec2::new(eye.x, eye.z);
    let body_vel = (body_xz - rig.prev_body_xz) / math::max(dt, 0.004);
    rig.prev_body_xz = body_xz;
    state.wake_body_pos = Vec3::new(eye.x, terrain_height(eye.x, eye.z), eye.z);
    state.wake_body_vel = body_vel.clamp_length_max(12.0);

    let screen = Vec2::new(frame.display_size.x as f32, frame.display_size.y as f32);
    if let Some(px) = input.pointer.position_pixels
        && !state.ui_hovered
        && !state.ui_captured_pointer
        && screen.x > 1.0
        && screen.y > 1.0
    {
        let (ray_o, ray_d) = cursor_ray(px, screen, inv_vp);
        if let Some(hit) = ray_terrain_hit(ray_o, ray_d) {
            let prev = rig.last_cursor_world.unwrap_or(hit);
            let vel = Vec2::new(hit.x - prev.x, hit.z - prev.z) / math::max(dt, 0.008);
            state.wake_cursor_pos = hit;
            state.wake_cursor_vel = vel.clamp_length_max(16.0);
            state.wake_cursor_active = true;
            rig.last_cursor_world = Some(hit);
        } else {
            state.wake_cursor_active = false;
            rig.last_cursor_world = None;
        }
    } else {
        state.wake_cursor_active = false;
        rig.last_cursor_world = None;
    }

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
