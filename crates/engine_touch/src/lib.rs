#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::{Devices, FrameFacts, InputModality, InputProfile, TouchPhase, math};
use engine_ecs as bevy_ecs;
use engine_ecs::prelude::Resource;
use glam::Vec2;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TouchAnchor {
    #[default]
    BottomLeft,
    BottomRight,
    TopLeft,
    TopRight,
    BottomCenter,
    TopCenter,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VirtualStickSpec {
    pub zone_max_x_frac: f32,
    pub anchor_dp: [f32; 2],
    pub radius_dp: f32,
    pub knob_radius_dp: f32,
    pub deadzone: f32,
    pub floating: bool,
}

impl Default for VirtualStickSpec {
    fn default() -> Self {
        Self {
            zone_max_x_frac: 0.45,
            anchor_dp: [96.0, 92.0],
            radius_dp: 56.0,
            knob_radius_dp: 24.0,
            deadzone: 0.12,
            floating: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LookPadSpec {
    pub zone_min_x_frac: f32,
    pub sensitivity: f32,
}

impl Default for LookPadSpec {
    fn default() -> Self {
        Self {
            zone_min_x_frac: 0.42,
            sensitivity: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TouchButtonSpec {
    pub id: u32,
    pub label: String,
    pub anchor: TouchAnchor,
    pub offset_dp: [f32; 2],
    pub radius_dp: f32,
    pub drag_aim: bool,
    pub accent: bool,
    pub active: bool,
    pub visible: bool,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct TouchLayout {
    pub enabled: bool,
    pub stick: Option<VirtualStickSpec>,
    pub look: Option<LookPadSpec>,
    pub buttons: Vec<TouchButtonSpec>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StickVisual {
    pub center_px: Vec2,
    pub knob_px: Vec2,
    pub radius_px: f32,
    pub knob_radius_px: f32,
    pub active: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ButtonVisual {
    pub id: u32,
    pub center_px: Vec2,
    pub radius_px: f32,
    pub label: String,
    pub held: bool,
    pub accent: bool,
    pub active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Claim {
    Stick,
    Look,
    Button(u32),
}

#[derive(Resource, Clone, Debug, Default)]
pub struct TouchState {
    pub visible: bool,
    pub move_axis: Vec2,
    look_delta_dp: Vec2,
    claims: Vec<(u32, Claim)>,
    stick_origin_px: Option<Vec2>,
    stick_knob_px: Vec2,
    held_ids: Vec<u32>,
    frame_pressed: Vec<u32>,
    frame_released: Vec<u32>,
    pending_pressed: Vec<u32>,
    pending_released: Vec<u32>,
    tick_pressed: Vec<u32>,
    tick_released: Vec<u32>,
    stick_visual: Option<StickVisual>,
    button_visuals: Vec<ButtonVisual>,
    revision: u64,
}

fn push_unique(list: &mut Vec<u32>, id: u32) {
    if !list.contains(&id) {
        list.push(id);
    }
}

fn safe_scale(frame: FrameFacts) -> f32 {
    let ratio = frame.effective_pixel_ratio();
    let logical = frame.logical_size();
    let short = math::min(logical.x, logical.y);
    let compact = if short > 0.0 && short < 420.0 {
        math::clamp(short / 420.0, 0.72, 1.0)
    } else {
        1.0
    };
    ratio * compact
}

fn anchor_position(anchor: TouchAnchor, offset_dp: [f32; 2], safe: [f32; 4], scale: f32) -> Vec2 {
    let [sx, sy, sw, sh] = safe;
    let dx = offset_dp[0] * scale;
    let dy = offset_dp[1] * scale;
    match anchor {
        TouchAnchor::BottomLeft => Vec2::new(sx + dx, sy + sh - dy),
        TouchAnchor::BottomRight => Vec2::new(sx + sw - dx, sy + sh - dy),
        TouchAnchor::TopLeft => Vec2::new(sx + dx, sy + dy),
        TouchAnchor::TopRight => Vec2::new(sx + sw - dx, sy + dy),
        TouchAnchor::BottomCenter => Vec2::new(sx + sw * 0.5 + dx, sy + sh - dy),
        TouchAnchor::TopCenter => Vec2::new(sx + sw * 0.5 + dx, sy + dy),
    }
}

fn stick_rest_center(spec: VirtualStickSpec, safe: [f32; 4], scale: f32) -> Vec2 {
    anchor_position(TouchAnchor::BottomLeft, spec.anchor_dp, safe, scale)
}

fn clamp_in_safe(point: Vec2, radius: f32, safe: [f32; 4]) -> Vec2 {
    let [sx, sy, sw, sh] = safe;
    let min_x = sx + radius;
    let max_x = math::max(min_x, sx + sw - radius);
    let min_y = sy + radius;
    let max_y = math::max(min_y, sy + sh - radius);
    Vec2::new(
        math::clamp(point.x, min_x, max_x),
        math::clamp(point.y, min_y, max_y),
    )
}

impl TouchState {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn stick_visual(&self) -> Option<StickVisual> {
        self.stick_visual
    }

    pub fn button_visuals(&self) -> &[ButtonVisual] {
        &self.button_visuals
    }

    pub fn held(&self, id: u32) -> bool {
        self.held_ids.contains(&id)
    }

    pub fn pressed(&self, id: u32) -> bool {
        self.frame_pressed.contains(&id)
    }

    pub fn released(&self, id: u32) -> bool {
        self.frame_released.contains(&id)
    }

    pub fn tick_pressed(&self, id: u32) -> bool {
        self.tick_pressed.contains(&id)
    }

    pub fn tick_released(&self, id: u32) -> bool {
        self.tick_released.contains(&id)
    }

    pub fn take_look_delta_dp(&mut self) -> Vec2 {
        std::mem::take(&mut self.look_delta_dp)
    }

    pub fn step_tick(&mut self) {
        self.tick_pressed.clear();
        self.tick_pressed.append(&mut self.pending_pressed);
        self.tick_released.clear();
        self.tick_released.append(&mut self.pending_released);
    }

    pub fn clear(&mut self) {
        let changed = self.visible || self.stick_visual.is_some() || !self.button_visuals.is_empty();
        self.visible = false;
        self.move_axis = Vec2::ZERO;
        self.look_delta_dp = Vec2::ZERO;
        self.claims.clear();
        self.stick_origin_px = None;
        self.stick_knob_px = Vec2::ZERO;
        self.held_ids.clear();
        self.frame_pressed.clear();
        self.frame_released.clear();
        self.pending_pressed.clear();
        self.pending_released.clear();
        self.tick_pressed.clear();
        self.tick_released.clear();
        self.stick_visual = None;
        self.button_visuals.clear();
        if changed {
            self.revision = self.revision.wrapping_add(1);
        }
    }

    pub fn update(
        &mut self,
        devices: &Devices,
        profile: InputProfile,
        frame: FrameFacts,
        layout: &TouchLayout,
    ) {
        self.frame_pressed.clear();
        self.frame_released.clear();
        let active = layout.enabled
            && profile.active == InputModality::Touch
            && frame.display_size.x > 0
            && frame.display_size.y > 0;
        if !active {
            self.clear();
            return;
        }
        let safe = frame.safe_rect_pixels();
        let scale = safe_scale(frame);
        let dpr = frame.effective_pixel_ratio();
        let prev_stick = self.stick_visual;
        let prev_buttons = self.button_visuals.clone();
        let prev_visible = self.visible;
        self.visible = true;

        let mut dropped = Vec::new();
        self.claims.retain(|&(touch_id, claim)| {
            let touch_alive = devices.touches.iter().any(|touch| touch.id == touch_id);
            let target_alive = match claim {
                Claim::Stick => layout.stick.is_some(),
                Claim::Look => layout.look.is_some(),
                Claim::Button(id) => layout
                    .buttons
                    .iter()
                    .any(|button| button.visible && button.id == id),
            };
            let keep = touch_alive && target_alive;
            if !keep {
                dropped.push(claim);
            }
            keep
        });
        for claim in dropped {
            self.release_claim(claim, false);
        }

        for touch in &devices.touches {
            match touch.phase {
                TouchPhase::Began => {
                    if let Some(old) = self.take_claim(touch.id) {
                        self.release_claim(old, false);
                    }
                    if let Some(claim) = self.pick_claim(touch.position_pixels, safe, scale, layout)
                    {
                        self.claims.push((touch.id, claim));
                        self.begin_claim(claim, touch.position_pixels, safe, scale, layout);
                    }
                }
                TouchPhase::Moved => {
                    if let Some(claim) = self.claim_of(touch.id) {
                        self.move_claim(
                            claim,
                            touch.position_pixels,
                            touch.motion_pixels,
                            dpr,
                            scale,
                            layout,
                        );
                    }
                }
                TouchPhase::Ended => {
                    if let Some(claim) = self.take_claim(touch.id) {
                        self.release_claim(claim, true);
                    }
                }
                TouchPhase::Cancelled => {
                    if let Some(claim) = self.take_claim(touch.id) {
                        self.release_claim(claim, false);
                    }
                }
            }
        }

        self.held_ids.clear();
        for &(_, claim) in &self.claims {
            if let Claim::Button(id) = claim {
                push_unique(&mut self.held_ids, id);
            }
        }
        if !self.claims.iter().any(|&(_, claim)| claim == Claim::Stick) {
            self.stick_origin_px = None;
            self.stick_knob_px = Vec2::ZERO;
            self.move_axis = Vec2::ZERO;
        }

        self.rebuild_visuals(safe, scale, layout);
        if prev_visible != self.visible
            || prev_stick != self.stick_visual
            || prev_buttons != self.button_visuals
        {
            self.revision = self.revision.wrapping_add(1);
        }
    }

    fn claim_of(&self, touch_id: u32) -> Option<Claim> {
        self.claims
            .iter()
            .find(|&&(id, _)| id == touch_id)
            .map(|&(_, claim)| claim)
    }

    fn take_claim(&mut self, touch_id: u32) -> Option<Claim> {
        let index = self.claims.iter().position(|&(id, _)| id == touch_id)?;
        Some(self.claims.swap_remove(index).1)
    }

    fn has_claim(&self, target: Claim) -> bool {
        self.claims.iter().any(|&(_, claim)| claim == target)
    }

    fn pick_claim(
        &self,
        pos: Vec2,
        safe: [f32; 4],
        scale: f32,
        layout: &TouchLayout,
    ) -> Option<Claim> {
        let mut best_button: Option<(u32, f32)> = None;
        for button in layout.buttons.iter().filter(|button| button.visible) {
            let center = anchor_position(button.anchor, button.offset_dp, safe, scale);
            let hit_radius = math::max(button.radius_dp * scale * 1.18, 24.0 * scale);
            let dist_sq = pos.distance_squared(center);
            if dist_sq <= hit_radius * hit_radius
                && best_button.is_none_or(|(_, best_sq)| dist_sq < best_sq)
            {
                best_button = Some((button.id, dist_sq));
            }
        }
        if let Some((id, _)) = best_button {
            return Some(Claim::Button(id));
        }
        if let Some(stick) = layout.stick
            && !self.has_claim(Claim::Stick)
        {
            let zone_max = safe[0] + safe[2] * stick.zone_max_x_frac;
            if pos.x <= zone_max {
                return Some(Claim::Stick);
            }
        }
        if let Some(look) = layout.look
            && !self.has_claim(Claim::Look)
        {
            let zone_min = safe[0] + safe[2] * look.zone_min_x_frac;
            if pos.x >= zone_min {
                return Some(Claim::Look);
            }
        }
        None
    }

    fn begin_claim(
        &mut self,
        claim: Claim,
        pos: Vec2,
        safe: [f32; 4],
        scale: f32,
        layout: &TouchLayout,
    ) {
        match claim {
            Claim::Stick => {
                if let Some(spec) = layout.stick {
                    let radius = spec.radius_dp * scale;
                    let origin = if spec.floating {
                        clamp_in_safe(pos, radius, safe)
                    } else {
                        stick_rest_center(spec, safe, scale)
                    };
                    self.stick_origin_px = Some(origin);
                    self.update_stick(pos, origin, spec, scale);
                }
            }
            Claim::Look => {}
            Claim::Button(id) => {
                push_unique(&mut self.frame_pressed, id);
                push_unique(&mut self.pending_pressed, id);
            }
        }
    }

    fn move_claim(
        &mut self,
        claim: Claim,
        pos: Vec2,
        motion_px: Vec2,
        dpr: f32,
        scale: f32,
        layout: &TouchLayout,
    ) {
        match claim {
            Claim::Stick => {
                if let (Some(spec), Some(origin)) = (layout.stick, self.stick_origin_px) {
                    self.update_stick(pos, origin, spec, scale);
                }
            }
            Claim::Look => {
                let gain = layout.look.map_or(1.0, |look| look.sensitivity);
                self.look_delta_dp += (motion_px / dpr) * gain;
            }
            Claim::Button(id) => {
                if let Some(button) = layout.buttons.iter().find(|button| button.id == id)
                    && button.drag_aim
                {
                    let gain = layout.look.map_or(1.0, |look| look.sensitivity);
                    self.look_delta_dp += (motion_px / dpr) * gain;
                }
            }
        }
    }

    fn release_claim(&mut self, claim: Claim, ended: bool) {
        match claim {
            Claim::Stick => {
                self.stick_origin_px = None;
                self.stick_knob_px = Vec2::ZERO;
                self.move_axis = Vec2::ZERO;
            }
            Claim::Look => {}
            Claim::Button(id) => {
                if ended {
                    push_unique(&mut self.frame_released, id);
                    push_unique(&mut self.pending_released, id);
                }
            }
        }
    }

    fn update_stick(&mut self, pos: Vec2, origin: Vec2, spec: VirtualStickSpec, scale: f32) {
        let radius = math::max(spec.radius_dp * scale, 1.0);
        let offset = pos - origin;
        let len = offset.length();
        let clamped = if len > radius {
            offset * (radius / len)
        } else {
            offset
        };
        self.stick_knob_px = origin + clamped;
        let frac = math::clamp(len / radius, 0.0, 1.0);
        if frac <= spec.deadzone {
            self.move_axis = Vec2::ZERO;
        } else {
            let scaled = (frac - spec.deadzone) / math::max(1.0 - spec.deadzone, 0.001);
            let dir = clamped / math::max(clamped.length(), 1.0e-5);
            self.move_axis = Vec2::new(dir.x * scaled, -dir.y * scaled);
        }
    }

    fn rebuild_visuals(&mut self, safe: [f32; 4], scale: f32, layout: &TouchLayout) {
        self.stick_visual = layout.stick.map(|spec| {
            let active = self.stick_origin_px.is_some();
            let center_px = self
                .stick_origin_px
                .unwrap_or_else(|| stick_rest_center(spec, safe, scale));
            let knob_px = if active {
                self.stick_knob_px
            } else {
                center_px
            };
            StickVisual {
                center_px,
                knob_px,
                radius_px: spec.radius_dp * scale,
                knob_radius_px: spec.knob_radius_dp * scale,
                active,
            }
        });
        self.button_visuals.clear();
        for button in layout.buttons.iter().filter(|button| button.visible) {
            self.button_visuals.push(ButtonVisual {
                id: button.id,
                center_px: anchor_position(button.anchor, button.offset_dp, safe, scale),
                radius_px: button.radius_dp * scale,
                label: button.label.clone(),
                held: self.held_ids.contains(&button.id),
                accent: button.accent,
                active: button.active,
            });
        }
    }
}
