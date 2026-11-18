use engine::math;
use glam::Vec3;

use super::Loaded;
use super::senses::Who;
use crate::bots::{Brain, Gait, Intent};
use crate::play::arms::{Dropped, Drops};
use crate::play::world::Arena;
use crate::weapons::GunKind;

pub(super) struct Ground<'a> {
    pub drops: Option<&'a Drops>,
    pub arena: &'a Arena,
    pub loaded: &'a Loaded,
}

fn wanted(ground: &Ground<'_>, who: &Who) -> Option<Dropped> {
    let reach = ground.loaded.bots.play.pickup_detour_m;
    let lift = Vec3::Y * ground.loaded.leg_radius_m;
    ground
        .drops?
        .items
        .iter()
        .filter(|item| item.weapon != GunKind::Sidearm)
        .filter(|item| {
            let flat = Vec3::new(
                item.position.x - who.feet.x,
                0.0,
                item.position.z - who.feet.z,
            );
            flat.length() <= reach && ground.arena.sees(who.eye, item.position + lift)
        })
        .min_by(|a, b| {
            let da = (a.position - who.feet).length();
            let db = (b.position - who.feet).length();
            da.total_cmp(&db).then(a.id.cmp(&b.id))
        })
        .copied()
}

fn wrap(angle: f32) -> f32 {
    let turn = std::f32::consts::TAU;
    angle - turn * ((angle + std::f32::consts::PI) / turn).floor()
}

fn toward(from: f32, to: f32, limit: f32) -> f32 {
    let gap = wrap(to - from);
    from + math::clamp(gap, -limit, limit)
}

pub(super) fn consider(
    ground: &Ground<'_>,
    who: &Who,
    brain: &mut Brain,
    pressed_before: bool,
    intent: &mut Intent,
) -> bool {
    let busy = intent.fire || intent.plant || intent.defuse || intent.ability.is_some();
    if !who.alive || who.gun.has_primary || busy {
        return false;
    }
    let Some(item) = wanted(ground, who) else {
        return false;
    };
    let to = item.position - who.eye;
    let flat = Vec3::new(to.x, 0.0, to.z);
    let yaw = math::atan2(-to.x, -to.z);
    let pitch = math::atan2(to.y, flat.length());
    let level = ground.loaded.bots.level(brain.debug().level);
    let limit = level.turn_deg_per_second.to_radians() * ground.loaded.bots.dt();
    intent.yaw = toward(who.yaw, yaw, limit);
    intent.pitch = toward(who.pitch, pitch, limit);
    intent.goal = item.position;
    brain.resync();
    if who.pickable == Some(item.id) {
        intent.wish = Vec3::ZERO;
        intent.gait = Gait::Still;
        return !pressed_before;
    }
    intent.wish = flat.normalize_or_zero();
    intent.gait = Gait::Run;
    false
}
