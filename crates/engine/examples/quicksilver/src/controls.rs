use engine::math;
use engine::prelude::*;
use engine_core::{Devices, InputProfile, KeyCode, MouseButton, Pointer, TouchPhase, Views};

use crate::bricks::fluid::FluidBrick;
use crate::state::{BASIN_FLOOR_Y, MaterialKind, QuicksilverState, ScalePreset};
use crate::ui_layout::{DockAction, DockLayout, compute_layout};

#[derive(
    Component,
    Clone,
    Copy,
    Default,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct StudioCamera {}

pub fn attach_camera_lens(
    eyes: Query<Entity, (With<StudioCamera>, Without<Camera>)>,
    mut spawner: Spawner<FluidBrick>,
) {
    for eye in &eyes {
        spawner.insert(
            eye,
            Camera {
                near: 0.05,
                far: 80.0,
                projection: Projection::Perspective {
                    vertical_fov_radians: 0.72,
                    fit: FovFit::Vertical,
                },
            },
        );
    }
}

pub fn handle_input_and_camera(
    (devices, pointer, profile): (Res<Devices>, Res<Pointer>, Option<Res<InputProfile>>),
    views: Res<Views>,
    time: Res<FrameTime>,
    mut state: ResMut<QuicksilverState>,
    mut cameras: Query<(Entity, &mut Camera), With<StudioCamera>>,
    mut moves: Moves<FluidBrick>,
) {
    let dt = math::clamp(time.real_seconds as f32, 0.001, 0.05);
    state.elapsed += dt;
    update_fps(&mut state, time.real_seconds);

    let frame = views.frame();
    let size = frame.display_size;
    let w = math::max(size.x as f32, 1.0);
    let h = math::max(size.y as f32, 1.0);
    let dpr = math::clamp(frame.pixel_ratio, 1.0, 3.0);
    let coarse_profile = profile
        .as_deref()
        .is_some_and(|p| p.capabilities.coarse_pointer);
    let coarse = pointer.is_coarse() || coarse_profile || (w / dpr) < 640.0;

    if (state.viewport[0] - w).abs() > 0.5
        || (state.viewport[1] - h).abs() > 0.5
        || (state.pixel_ratio - dpr).abs() > 0.01
    {
        state.viewport = [w, h];
        state.pixel_ratio = dpr;
        state.ui_revision = state.ui_revision.wrapping_add(1);
    }
    if state.is_coarse != coarse {
        state.is_coarse = coarse;
        state.ui_revision = state.ui_revision.wrapping_add(1);
    }
    if !state.auto_checked && !state.preset_locked && size.x > 1 {
        state.auto_checked = true;
        if coarse {
            state.preset = ScalePreset::Low;
            state.reset_counter = state.reset_counter.wrapping_add(1);
            state.ui_revision = state.ui_revision.wrapping_add(1);
        }
    }

    handle_keys(&devices, &mut state);
    let layout = compute_layout(&state);
    handle_pointer_and_touches(&devices, &pointer, &layout, &mut state, dt);

    let aspect = w / h;
    let (elapsed, zero_g) = (state.elapsed, state.zero_g);
    state.camera.update(dt, elapsed, zero_g, aspect);

    let lens = Camera {
        near: 0.05,
        far: 80.0,
        projection: Projection::Perspective {
            vertical_fov_radians: state.camera.fov_y,
            fit: FovFit::Vertical,
        },
    };
    let placed = Transform::looking_at(state.camera.eye, state.camera.target);
    for (eye, mut camera) in &mut cameras {
        camera.set_if_neq(lens);
        if let Some(mut transform) = moves.get_mut(eye) {
            let next = Transform {
                translation: placed.translation,
                rotation: placed.rotation,
                scale: transform.scale,
            };
            transform.set_if_neq(next);
        }
    }
}

fn update_fps(state: &mut QuicksilverState, real_delta: f64) {
    if state.elapsed < 0.35 {
        return;
    }
    state.perf.frames += 1;
    state.perf.accum += real_delta;
    if state.perf.accum >= 0.40 {
        let fps = (f64::from(state.perf.frames) / state.perf.accum).round() as u32;
        let clamped = fps.clamp(1, 165);
        if state.perf.displayed_fps != clamped {
            state.perf.displayed_fps = clamped;
            state.ui_revision = state.ui_revision.wrapping_add(1);
        }
        state.perf.frames = 0;
        state.perf.accum = 0.0;
    }
}

fn handle_keys(devices: &Devices, state: &mut QuicksilverState) {
    let kb = &devices.keyboard;
    if kb.pressed(KeyCode::Digit1) {
        set_material(state, MaterialKind::Chrome);
    }
    if kb.pressed(KeyCode::Digit2) {
        set_material(state, MaterialKind::Water);
    }
    if kb.pressed(KeyCode::Digit3) {
        set_material(state, MaterialKind::Gold);
    }
    if kb.pressed(KeyCode::Digit4) {
        set_material(state, MaterialKind::Plasma);
    }
    if kb.pressed(KeyCode::Space) || kb.pressed(KeyCode::KeyD) {
        state.trigger_drop();
    }
    if kb.pressed(KeyCode::KeyV) {
        state.vortex = !state.vortex;
        state.ui_revision = state.ui_revision.wrapping_add(1);
    }
    if kb.pressed(KeyCode::KeyG) {
        state.zero_g = !state.zero_g;
        state.ui_revision = state.ui_revision.wrapping_add(1);
    }
    if kb.pressed(KeyCode::KeyR) {
        state.trigger_reset();
    }
    if kb.pressed(KeyCode::KeyQ) {
        state.set_preset(ScalePreset::Low);
    }
    if kb.pressed(KeyCode::KeyW) {
        state.set_preset(ScalePreset::Mid);
    }
    if kb.pressed(KeyCode::KeyE) {
        state.set_preset(ScalePreset::High);
    }
}

fn set_material(state: &mut QuicksilverState, mat: MaterialKind) {
    if state.material != mat {
        state.material = mat;
        state.ui_revision = state.ui_revision.wrapping_add(1);
    }
}

fn apply_action(state: &mut QuicksilverState, action: DockAction) {
    match action {
        DockAction::Material(mat) => set_material(state, mat),
        DockAction::Drop => state.trigger_drop(),
        DockAction::Vortex => {
            state.vortex = !state.vortex;
            state.ui_revision = state.ui_revision.wrapping_add(1);
        }
        DockAction::ZeroG => {
            state.zero_g = !state.zero_g;
            state.ui_revision = state.ui_revision.wrapping_add(1);
        }
        DockAction::Scale(preset) => state.set_preset(preset),
        DockAction::OrbitMode => {
            state.orbit_mode = !state.orbit_mode;
            state.ui_revision = state.ui_revision.wrapping_add(1);
        }
    }
}

fn handle_pointer_and_touches(
    devices: &Devices,
    pointer: &Pointer,
    layout: &DockLayout,
    state: &mut QuicksilverState,
    dt: f32,
) {
    let hovered = if pointer.hovering {
        pointer.position_pixels.and_then(|p| layout.hit_test(p.x, p.y))
    } else {
        None
    };
    if state.ui.hovered != hovered {
        state.ui.hovered = hovered;
        state.ui_revision = state.ui_revision.wrapping_add(1);
    }

    if pointer.pressed {
        let hit = pointer.position_pixels.and_then(|p| layout.hit_test(p.x, p.y));
        let on_panel = pointer
            .position_pixels
            .is_some_and(|p| layout.covers_point(p.x, p.y));
        state.ui.pressed = hit;
        state.ui.captured_pointer = on_panel;
        if hit.is_some() {
            state.ui_revision = state.ui_revision.wrapping_add(1);
        }
    }

    if pointer.released {
        if let Some(pressed_idx) = state.ui.pressed {
            let hit = pointer.position_pixels.and_then(|p| layout.hit_test(p.x, p.y));
            if hit == Some(pressed_idx)
                && let Some(btn) = layout.buttons.get(pressed_idx)
            {
                apply_action(state, btn.action);
            }
            state.ui_revision = state.ui_revision.wrapping_add(1);
        }
        state.ui.pressed = None;
        state.ui.captured_pointer = false;
    }

    if pointer.cancelled {
        state.ui.pressed = None;
        state.ui.captured_pointer = false;
    }

    let wheel = devices.mouse.wheel_notches.y;
    if wheel.abs() > 0.001 {
        state.camera.goal_distance -= wheel * 0.42;
        state.camera.user_moved = true;
    }

    let active_touches: Vec<Vec2> = devices
        .touches
        .iter()
        .filter(|t| matches!(t.phase, TouchPhase::Began | TouchPhase::Moved))
        .map(|t| t.position_pixels)
        .collect();

    if active_touches.len() >= 2 {
        let mid = (active_touches[0] + active_touches[1]) * 0.5;
        let dist = active_touches[0].distance(active_touches[1]);
        if let Some(prev_mid) = state.ui.last_touch_mid {
            let delta = mid - prev_mid;
            state.camera.goal_yaw -= delta.x * 0.0055 / state.pixel_ratio;
            state.camera.goal_pitch += delta.y * 0.0055 / state.pixel_ratio;
            state.camera.user_moved = true;
        }
        if let Some(prev_dist) = state.ui.last_pinch_dist {
            let dd = dist - prev_dist;
            state.camera.goal_distance -= dd * 0.01 / state.pixel_ratio;
            state.camera.user_moved = true;
        }
        state.ui.last_touch_mid = Some(mid);
        state.ui.last_pinch_dist = Some(dist);
        state.stir.active = false;
        state.stir.had_prev = false;
        return;
    }
    state.ui.last_touch_mid = None;
    state.ui.last_pinch_dist = None;

    let shift = devices.keyboard.held(KeyCode::ShiftLeft)
        || devices.keyboard.held(KeyCode::ShiftRight);
    let right_or_mid = devices.mouse.buttons.held(MouseButton::Right)
        || devices.mouse.buttons.held(MouseButton::Middle);

    if right_or_mid || (shift && pointer.down && !state.ui.captured_pointer) {
        let delta = devices.mouse.motion_pixels;
        state.camera.goal_yaw -= delta.x * 0.0065;
        state.camera.goal_pitch += delta.y * 0.0065;
        state.camera.user_moved = true;
        state.stir.active = false;
        state.stir.had_prev = false;
        return;
    }

    if pointer.down && !state.ui.captured_pointer {
        if state.orbit_mode {
            let delta = pointer.delta_pixels;
            state.camera.goal_yaw -= delta.x * 0.006 / state.pixel_ratio;
            state.camera.goal_pitch += delta.y * 0.006 / state.pixel_ratio;
            state.camera.user_moved = true;
            state.stir.active = false;
            state.stir.had_prev = false;
        } else if let Some(pos_px) = pointer.position_pixels {
            update_stir_ray(state, pos_px, dt);
        }
    } else {
        state.stir.active = false;
        state.stir.had_prev = false;
        state.stir.world_vel *= 0.8;
    }
}

fn update_stir_ray(state: &mut QuicksilverState, pos_px: Vec2, dt: f32) {
    let w = math::max(state.viewport[0], 1.0);
    let h = math::max(state.viewport[1], 1.0);
    let ndc_x = (pos_px.x / w) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pos_px.y / h) * 2.0;
    let aspect = w / h;
    let tan_half = (state.camera.fov_y * 0.5).tan();
    let ray_dir = (state.camera.forward
        + state.camera.right * (ndc_x * aspect * tan_half)
        + state.camera.up * (ndc_y * tan_half))
        .normalize_or_zero();
    let eye = state.camera.eye;

    let hit = if state.zero_g {
        let plane_n = -state.camera.forward;
        let denom = ray_dir.dot(plane_n);
        if denom.abs() > 1.0e-4 {
            let t = (state.camera.target - eye).dot(plane_n) / denom;
            if t > 0.2 { Some(eye + ray_dir * t) } else { None }
        } else {
            None
        }
    } else {
        let plane_y = BASIN_FLOOR_Y + 0.42;
        if ray_dir.y.abs() > 0.08 {
            let t = (plane_y - eye.y) / ray_dir.y;
            if t > 0.2 && t < 25.0 {
                Some(eye + ray_dir * t)
            } else {
                None
            }
        } else {
            let t = (state.camera.target - eye).dot(state.camera.forward);
            Some(eye + ray_dir * math::max(t, 1.0))
        }
    };

    if let Some(world) = hit {
        let clamped = Vec3::new(
            math::clamp(world.x, -2.15, 2.15),
            math::clamp(world.y, 0.12, 2.35),
            math::clamp(world.z, -1.35, 1.35),
        );
        if state.stir.had_prev {
            let raw_vel = (clamped - state.stir.world_pos) / math::max(dt, 0.004);
            let max_v = 14.0;
            let len = raw_vel.length();
            let vel = if len > max_v {
                raw_vel * (max_v / len)
            } else {
                raw_vel
            };
            state.stir.world_vel = state.stir.world_vel.lerp(vel, 0.55);
        } else {
            state.stir.world_vel = Vec3::ZERO;
            state.stir.had_prev = true;
        }
        state.stir.world_pos = clamped;
        state.stir.active = true;
        state.stir.strength = 1.0;
    } else {
        state.stir.active = false;
        state.stir.had_prev = false;
    }
}
