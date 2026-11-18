use glam::Vec3;

use super::aim::{
    angle_between, clamp, direction, error_radians, flat, flat_distance, look_angles, max, radians,
    sin_cos, wrap,
};
use super::debug::{Activity, Why};
use super::memory::{Source, Track};
use super::sense::{
    CallKind, DeviceSense, Gait, GamePhase, Intent, ObjectKind, Purchase, Side, WeaponChoice,
};
use super::team::{Duty, Step};
use super::{Brain, Ctx};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fight {
    target: Option<u32>,
    head: bool,
    sloppy: bool,
    over_peek: bool,
    burst_left: u32,
    pause_until: u64,
    strafe_sign: f32,
    strafe_until: u64,
    cover: Option<Vec3>,
    hunt_at: Option<Vec3>,
    peek_until: u64,
    hunt_done: bool,
    scope_tick: u64,
    spray_yaw: f32,
    spray_pitch: f32,
}

impl Fight {
    pub fn new() -> Fight {
        Fight {
            target: None,
            head: false,
            sloppy: false,
            over_peek: false,
            burst_left: 0,
            pause_until: 0,
            strafe_sign: 1.0,
            strafe_until: 0,
            cover: None,
            hunt_at: None,
            peek_until: 0,
            hunt_done: false,
            scope_tick: 0,
            spray_yaw: 0.0,
            spray_pitch: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Glance {
    cleared: Vec<(usize, u64)>,
    current: Option<(usize, u64)>,
    next_scan: u64,
    scan_yaw: f32,
    next_hold_scan: u64,
}

impl Glance {
    pub fn new() -> Glance {
        Glance {
            cleared: Vec::with_capacity(32),
            current: None,
            next_scan: 0,
            scan_yaw: 0.0,
            next_hold_scan: 0,
        }
    }
}

fn turn_to(brain: &mut Brain, ctx: &Ctx<'_, '_>, point: Vec3, share: f32) {
    let (yaw, pitch) = look_angles(ctx.eye, point);
    let turn = ctx.level.turn_deg_per_second
        * share
        * (1.0 - ctx.sense.blind * ctx.numbers.play.blind_turn);
    brain
        .aim
        .turn_toward(yaw, pitch, turn, &ctx.numbers.play, ctx.dt);
}

fn calm_turn(brain: &mut Brain, ctx: &Ctx<'_, '_>, point: Vec3) {
    turn_to(brain, ctx, point, ctx.numbers.play.idle_turn_share);
}

pub fn buy(brain: &mut Brain, ctx: &Ctx<'_, '_>, intent: &mut Intent) {
    let tick = ctx.tick;
    let play = &ctx.numbers.play;
    let delay = brain.rng.range(play.hesitate_seconds);
    let due = *brain
        .buy_tick
        .get_or_insert(tick + ctx.numbers.ticks(delay));
    let late = u64::from(ctx.sense.game.ticks_left) <= ctx.numbers.ticks(play.buy_late_seconds);
    let due = if late { due.min(tick) } else { due };
    if !brain.bought && tick >= due {
        intent.buy = Purchase::Own;
        brain.bought = true;
    }
    turn_to(brain, ctx, ctx.sense.order.face, play.idle_turn_share);
    switch_primary(ctx, intent);
    brain.mover.reset_progress(tick);
    brain.mind(Activity::Wait, Why::RoundStart, tick);
}

fn switch_primary(ctx: &Ctx<'_, '_>, intent: &mut Intent) {
    let gun = ctx.sense.gun;
    if gun.has_primary && gun.holding != WeaponChoice::Primary {
        intent.switch = Some(WeaponChoice::Primary);
    } else if !gun.has_primary && gun.holding == WeaponChoice::Melee {
        intent.switch = Some(WeaponChoice::Sidearm);
    }
}

pub fn act(brain: &mut Brain, ctx: &Ctx<'_, '_>, intent: &mut Intent) {
    let play = &ctx.numbers.play;
    if ctx.sense.blind >= play.blind_retreat {
        blinded(brain, ctx, intent);
        return;
    }
    if let Some(target) = brain.memory.target(ctx.eye, ctx.tick) {
        engage(brain, ctx, &target, intent);
        return;
    }
    if brain.fight.target.is_some() {
        brain.fight.target = None;
        brain.fight.burst_left = 0;
        brain.fight.cover = None;
    }
    brain.debug.target = None;
    if hunt(brain, ctx, intent) {
        return;
    }
    follow(brain, ctx, intent);
}

fn blinded(brain: &mut Brain, ctx: &Ctx<'_, '_>, intent: &mut Intent) {
    let away = brain
        .memory
        .lost(ctx.tick, ctx.numbers.ticks(ctx.level.memory_seconds))
        .map(|t| flat(ctx.sense.position - t.at).normalize_or_zero());
    if let Some(away) = away {
        intent.wish = away;
        intent.gait = Gait::Walk;
    }
    brain.mover.reset_progress(ctx.tick);
    brain.mind(Activity::Retreat, Why::Blinded, ctx.tick);
}

fn mate_in_line(ctx: &Ctx<'_, '_>, toward: Vec3, reach: f32) -> bool {
    let clearance = ctx.numbers.play.mate_clearance_m;
    let chest = ctx.up * 0.75;
    ctx.sense.mates.iter().filter(|m| m.alive).any(|mate| {
        let v = mate.position + chest - ctx.eye;
        let along = v.dot(toward);
        along > 0.0 && along < reach && (v - toward * along).length() < clearance
    })
}

fn acquire(brain: &mut Brain, ctx: &Ctx<'_, '_>, target: &Track) {
    if brain.fight.target == target.id {
        return;
    }
    brain.fight.target = target.id;
    brain.fight.head = brain.rng.chance(ctx.level.head_share);
    brain.fight.sloppy = brain.rng.chance(ctx.level.run_and_gun);
    brain.fight.over_peek = brain.rng.chance(peek_share(brain, ctx));
    brain.fight.burst_left = 0;
    brain.fight.cover = None;
    brain.fight.hunt_at = None;
    let wander = ctx.numbers.ticks(ctx.numbers.play.error_wander_seconds);
    brain.aim.new_error(&mut brain.rng, ctx.tick, wander);
    brain.debug.target = target.id;
    brain.mind(Activity::Fight, Why::Spotted, ctx.tick);
}

fn needs_cover(brain: &Brain, ctx: &Ctx<'_, '_>, distance: f32) -> Option<Why> {
    let gun = ctx.sense.gun;
    let play = &ctx.numbers.play;
    if gun.reloading || (gun.magazine == 0 && gun.reserve > 0) {
        return Some(Why::Reload);
    }
    let objective =
        matches!(brain.order.step, Step::Plant | Step::Defuse) && brain.order.duty != Duty::Cover;
    if ctx.sense.health < play.retreat_health && distance > play.spray_m && !objective {
        return Some(Why::LowHealth);
    }
    None
}

fn engage(brain: &mut Brain, ctx: &Ctx<'_, '_>, target: &Track, intent: &mut Intent) {
    acquire(brain, ctx, target);
    let play = &ctx.numbers.play;
    let level = ctx.level;
    let point = if brain.fight.head {
        target.head
    } else {
        target.chest
    };
    let distance = max((point - ctx.eye).length(), play.near_m);
    let seen = ctx
        .numbers
        .seconds(ctx.tick.saturating_sub(target.react_tick));
    let error = error_radians(level, seen, ctx.speed_share, ctx.sense.blind);
    let unit = brain.aim.error_unit(&mut brain.rng, ctx.tick);
    let (yaw, pitch) = look_angles(ctx.eye, point);
    let want_yaw = wrap(yaw + unit.x * error);
    let want_pitch = pitch + unit.y * error;
    let turn = level.turn_deg_per_second * (1.0 - ctx.sense.blind * play.blind_turn);
    brain
        .aim
        .turn_toward(want_yaw, want_pitch, turn, play, ctx.dt);
    let gun = ctx.sense.gun;
    let empty = gun.magazine == 0 && !gun.reloading && gun.holding == WeaponChoice::Primary;
    if empty && (distance <= play.spray_m || gun.reserve == 0) {
        intent.switch = Some(WeaponChoice::Sidearm);
    }
    if let Some(why) = needs_cover(brain, ctx, distance) {
        retreat(brain, ctx, target, why, intent);
    } else {
        brain.fight.cover = None;
        brain.mover.reset_progress(ctx.tick);
        fight_moves(brain, ctx, target, distance, intent);
        brain.mind(Activity::Fight, Why::Spotted, ctx.tick);
    }
    let off = angle_between(brain.aim.yaw, brain.aim.pitch, want_yaw, want_pitch);
    trigger(brain, ctx, point, off, intent);
}

fn retreat(brain: &mut Brain, ctx: &Ctx<'_, '_>, target: &Track, why: Why, intent: &mut Intent) {
    let play = &ctx.numbers.play;
    let cover = *brain.fight.cover.get_or_insert_with(|| {
        ctx.world
            .cover_from(target.head, ctx.sense.position, play.cover_search_m)
            .unwrap_or(
                ctx.sense.position
                    - flat(target.at - ctx.sense.position).normalize_or_zero()
                        * play.cover_search_m,
            )
    });
    let steering = brain.mover.steer(
        ctx.sense.position,
        cover,
        ctx.world,
        ctx.numbers,
        &mut brain.rng,
        ctx.tick,
    );
    intent.wish = steering.wish;
    intent.gait = if steering.arrived {
        Gait::Still
    } else {
        Gait::Run
    };
    intent.goal = cover;
    if steering.arrived
        && ctx.sense.gun.magazine < ctx.sense.gun.magazine_size
        && !ctx.sense.gun.reloading
    {
        intent.reload = ctx.sense.gun.reserve > 0;
    }
    let activity = if why == Why::Reload {
        Activity::Reload
    } else {
        Activity::Retreat
    };
    brain.mind(activity, why, ctx.tick);
}

fn fight_moves(
    brain: &mut Brain,
    ctx: &Ctx<'_, '_>,
    target: &Track,
    distance: f32,
    intent: &mut Intent,
) {
    let play = &ctx.numbers.play;
    let at_range = distance > play.spray_m;
    let pausing = brain.fight.burst_left == 0 && ctx.tick < brain.fight.pause_until;
    let objective = matches!(brain.order.step, Step::Plant | Step::Defuse)
        && matches!(brain.order.duty, Duty::Carrier | Duty::Defuser);
    if objective {
        return;
    }
    let to_target = flat(target.at - ctx.sense.position).normalize_or_zero();
    if brain.fight.over_peek && at_range {
        intent.wish = to_target;
        intent.gait = Gait::Run;
        return;
    }
    if !pausing && !(brain.fight.sloppy && !at_range) {
        intent.wish = if ctx.speed_share >= ctx.numbers.combat.deadzone_share {
            -flat(ctx.sense.velocity).normalize_or_zero()
        } else {
            Vec3::ZERO
        };
        intent.gait = Gait::Run;
        return;
    }
    if ctx.tick >= brain.fight.strafe_until {
        brain.fight.strafe_sign = -brain.fight.strafe_sign;
        let span = brain.rng.range(play.strafe_seconds);
        brain.fight.strafe_until = ctx.tick + ctx.numbers.ticks(span);
    }
    intent.wish = Vec3::new(-to_target.z, 0.0, to_target.x) * brain.fight.strafe_sign;
    intent.gait = if distance <= play.spray_m {
        Gait::Walk
    } else {
        Gait::Run
    };
}

fn shot_line_blocked(brain: &Brain, ctx: &Ctx<'_, '_>, point: Vec3, distance: f32) -> bool {
    let gun = ctx.sense.gun;
    let shot = direction(
        brain.aim.yaw + gun.recoil_yaw - brain.fight.spray_yaw,
        brain.aim.pitch + gun.recoil_pitch - brain.fight.spray_pitch,
    );
    let toward = (point - ctx.eye).normalize_or_zero();
    let reach = max(distance, gun.range_m);
    mate_in_line(ctx, shot, reach) || mate_in_line(ctx, toward, reach)
}

fn trigger(brain: &mut Brain, ctx: &Ctx<'_, '_>, point: Vec3, off: f32, intent: &mut Intent) {
    let play = &ctx.numbers.play;
    let gun = ctx.sense.gun;
    let distance = max((point - ctx.eye).length(), play.near_m);
    let fired = if brain.last_fire {
        brain.last_magazine.saturating_sub(gun.magazine)
    } else {
        0
    };
    if brain.fight.burst_left > 0 && fired > 0 {
        let hold = play.head_radius_m * play.spray_hold_share / distance;
        let grown = radians(gun.spread_deg) > hold;
        brain.fight.burst_left = if grown {
            0
        } else {
            brain.fight.burst_left.saturating_sub(fired)
        };
        if brain.fight.burst_left == 0 {
            let stretch = clamp(distance / play.spray_m, 1.0, play.burst_pause_stretch);
            let pause = play.burst_pause_seconds * stretch;
            brain.fight.pause_until = ctx.tick + ctx.numbers.ticks(pause);
        }
    }
    let at_range = distance > play.spray_m;
    let scoped = gun.scope_level > 0;
    if gun.has_scope && scoped != at_range && ctx.tick >= brain.fight.scope_tick {
        intent.scope = true;
        brain.fight.scope_tick = ctx.tick + ctx.numbers.ticks(play.utility_seconds);
    }
    let size = if brain.fight.head {
        play.head_radius_m
    } else {
        play.body_radius_m
    };
    let tolerance = size / distance * ctx.level.fire_tolerance;
    let bursting = brain.fight.burst_left > 0;
    let on = off
        <= if bursting {
            tolerance * play.burst_hold_share
        } else {
            tolerance
        };
    let still = ctx.speed_share < ctx.numbers.combat.deadzone_share;
    let moving_ok = (still || (brain.fight.sloppy && !at_range)) && ctx.sense.grounded;
    let ready = gun.ready && gun.magazine > 0 && !gun.reloading;
    if shot_line_blocked(brain, ctx, point, distance) {
        brain.mind(Activity::Fight, Why::MateInLine, ctx.tick);
        return;
    }
    if !(on && moving_ok && ready) {
        return;
    }
    if !bursting && ctx.tick >= brain.fight.pause_until {
        brain.fight.burst_left = gun.magazine;
    }
    if brain.fight.burst_left > 0 {
        intent.fire = gun.auto || !brain.last_fire;
        intent.crouch = gun.auto && brain.last_fire && distance <= play.spray_m;
    }
}

fn smoke_between(ctx: &Ctx<'_, '_>, to: Vec3) -> bool {
    ctx.sense.smokes.iter().any(|smoke| {
        let segment = to - ctx.eye;
        let length = segment.length();
        if length <= f32::EPSILON {
            return false;
        }
        let along = clamp((smoke.center - ctx.eye).dot(segment / length), 0.0, length);
        (ctx.eye + segment / length * along - smoke.center).length() <= smoke.radius
    })
}

fn through_smoke(brain: &mut Brain, ctx: &Ctx<'_, '_>, track: &Track, intent: &mut Intent) -> bool {
    let play = &ctx.numbers.play;
    let fresh =
        ctx.tick.saturating_sub(track.seen_tick) <= ctx.numbers.ticks(play.smoke_spray_seconds);
    if !play.shoot_through_smoke
        || track.source != Source::Seen
        || !fresh
        || !smoke_between(ctx, track.chest)
    {
        return false;
    }
    let gun = ctx.sense.gun;
    let lead = track.velocity * ctx.numbers.seconds(ctx.tick - track.seen_tick);
    let distance = max((track.chest - ctx.eye).length(), play.near_m);
    if shot_line_blocked(brain, ctx, track.chest + lead, distance) {
        return false;
    }
    turn_to(brain, ctx, track.chest + lead, 1.0);
    let (yaw, pitch) = look_angles(ctx.eye, track.chest + lead);
    let off = angle_between(brain.aim.yaw, brain.aim.pitch, yaw, pitch);
    if off <= play.body_radius_m / distance * ctx.level.fire_tolerance
        && gun.ready
        && gun.magazine > 0
    {
        intent.fire = gun.auto || !brain.last_fire;
    }
    brain.mind(Activity::Fight, Why::LostTarget, ctx.tick);
    true
}

fn hunt(brain: &mut Brain, ctx: &Ctx<'_, '_>, intent: &mut Intent) -> bool {
    let level = ctx.level;
    let play = &ctx.numbers.play;
    let Some(track) = brain
        .memory
        .lost(ctx.tick, ctx.numbers.ticks(level.memory_seconds))
    else {
        brain.fight.hunt_at = None;
        return false;
    };
    if through_smoke(brain, ctx, &track, intent) {
        return true;
    }
    let order = brain.order;
    let busy = matches!(
        order.step,
        Step::Plant | Step::Defuse | Step::Save | Step::Pickup
    ) && matches!(order.duty, Duty::Carrier | Duty::Defuser | Duty::Saver);
    let taking_orb = track.source == Source::Called
        && brain
            .orb_spot
            .is_some_and(|stand| flat_distance(stand, ctx.sense.position) <= play.arrive_m);
    if busy || taking_orb {
        return false;
    }
    let new_episode = brain
        .fight
        .hunt_at
        .is_none_or(|at| (at - track.at).length() > play.merge_m);
    if new_episode {
        brain.fight.hunt_at = Some(track.at);
        brain.fight.hunt_done = false;
        brain.fight.peek_until =
            ctx.tick + ctx.numbers.ticks(brain.rng.range(play.peek_wait_seconds));
        brain.fight.over_peek = brain.rng.chance(peek_share(brain, ctx));
        let why = if track.source == Source::Called {
            Why::Called
        } else {
            Why::LostTarget
        };
        brain.mind(Activity::Hunt, why, ctx.tick);
    }
    if brain.fight.hunt_done {
        return false;
    }
    let post_reach = if order.wait_contact && order.step == Step::Retake {
        play.spacing_m * 2.0
    } else {
        play.walk_near_m
    };
    let far_from_post = flat_distance(ctx.sense.position, order.goal) > post_reach;
    let pushes = (matches!(order.step, Step::Execute | Step::Retake | Step::Hunt)
        || order.duty == Duty::Entry
        || brain.fight.over_peek)
        && !order.wait_contact;
    if !pushes && far_from_post && ctx.tick >= order.go_tick {
        return false;
    }
    calm_turn(brain, ctx, track.head);
    brain.mind(Activity::Hunt, brain.debug.why, ctx.tick);
    if ctx.tick < brain.fight.peek_until || !pushes {
        brain.mover.reset_progress(ctx.tick);
        if !pushes
            && ctx.tick
                >= brain.fight.peek_until
                    + ctx
                        .numbers
                        .ticks(level.memory_seconds * play.hold_lost_share)
        {
            brain.fight.hunt_done = true;
        }
        return true;
    }
    let sees = ctx.world.sees(ctx.eye, track.head);
    let close = flat_distance(ctx.sense.position, track.at) <= play.arrive_m * 2.0;
    if close || (sees && !brain.fight.over_peek) {
        brain.fight.hunt_done = true;
        return true;
    }
    let steering = brain.mover.steer(
        ctx.sense.position,
        track.at,
        ctx.world,
        ctx.numbers,
        &mut brain.rng,
        ctx.tick,
    );
    intent.wish = steering.wish;
    intent.gait = if brain.fight.over_peek {
        Gait::Run
    } else {
        Gait::Walk
    };
    intent.goal = track.at;
    true
}

fn gait_for(brain: &Brain, ctx: &Ctx<'_, '_>) -> Gait {
    let play = &ctx.numbers.play;
    let position = ctx.sense.position;
    let near_enemy = brain
        .memory
        .nearest(position)
        .is_some_and(|d| d <= play.walk_near_m);
    let near_goal = flat_distance(position, brain.order.goal) <= play.walk_near_m;
    let lurking = brain.order.duty == Duty::Lurk;
    let careful = brain.order.careful && (near_goal || near_enemy || lurking);
    if careful
        || (near_enemy && !matches!(brain.order.step, Step::Execute | Step::Rotate | Step::Save))
    {
        Gait::Walk
    } else {
        Gait::Run
    }
}

fn preaim(brain: &mut Brain, ctx: &Ctx<'_, '_>, heading: Vec3) -> Option<Vec3> {
    let play = &ctx.numbers.play;
    let tick = ctx.tick;
    let angles = &ctx.world.places().angles;
    if let Some((index, start)) = brain.glance.current {
        if tick.saturating_sub(start) < ctx.numbers.ticks(play.angle_look_seconds) {
            return angles.get(index).map(|a| a.at + ctx.up);
        }
        brain.glance.cleared.push((index, tick));
        brain.glance.current = None;
    }
    if tick < brain.glance.next_scan || heading == Vec3::ZERO {
        return None;
    }
    brain.glance.next_scan = tick + ctx.numbers.ticks(play.angle_scan_seconds).max(1);
    let forget = ctx.numbers.ticks(play.angle_forget_seconds);
    brain
        .glance
        .cleared
        .retain(|&(_, t)| tick.saturating_sub(t) <= forget);
    let position = ctx.sense.position;
    let mut tried = [usize::MAX; 3];
    for attempt in 0..tried.len() {
        let mut best: Option<(usize, f32)> = None;
        for (index, angle) in angles.iter().enumerate() {
            let d = flat_distance(position, angle.at);
            if d > play.preaim_m || d < play.arrive_m * 2.0 || tried.contains(&index) {
                continue;
            }
            if flat(angle.at - position).dot(heading) < d * play.preaim_front {
                continue;
            }
            if brain.glance.cleared.iter().any(|&(i, _)| i == index) {
                continue;
            }
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((index, d));
            }
        }
        let (index, _) = best?;
        tried[attempt] = index;
        let point = angles[index].at + ctx.up;
        if !ctx.world.sees(ctx.eye, point) {
            continue;
        }
        if ctx.level.checks_corners {
            brain.glance.current = Some((index, tick));
            return Some(point);
        }
        brain.glance.cleared.push((index, tick));
        return None;
    }
    None
}

fn hold_look(brain: &mut Brain, ctx: &Ctx<'_, '_>) {
    let play = &ctx.numbers.play;
    if ctx.tick >= brain.glance.next_hold_scan {
        let span = brain.rng.range(play.hold_scan_seconds);
        brain.glance.next_hold_scan = ctx.tick + ctx.numbers.ticks(span);
        brain.glance.scan_yaw = radians(play.hold_scan_deg) * (2.0 * brain.rng.unit() - 1.0);
    }
    let face = brain.order.face;
    let (yaw, pitch) = look_angles(ctx.eye, face);
    let scan = brain.glance.scan_yaw;
    let level = ctx.level.turn_deg_per_second * play.idle_turn_share;
    brain
        .aim
        .turn_toward(wrap(yaw + scan), pitch, level, play, ctx.dt);
}

fn idle_reload(brain: &Brain, ctx: &Ctx<'_, '_>, intent: &mut Intent) {
    let gun = ctx.sense.gun;
    let play = &ctx.numbers.play;
    let low = (gun.magazine as f32) < gun.magazine_size as f32 * play.low_ammo_share;
    let calm = brain
        .memory
        .lost(ctx.tick, ctx.numbers.ticks(play.reaction_reset_seconds))
        .is_none();
    let exposed = brain
        .memory
        .lost(ctx.tick, ctx.numbers.ticks(ctx.level.memory_seconds))
        .map(|track| track.head)
        .into_iter()
        .chain(
            ctx.sense
                .sounds
                .iter()
                .map(|heard| heard.at + (ctx.eye - ctx.sense.position)),
        )
        .any(|from| ctx.world.sees(from, ctx.eye));
    if low && calm && !exposed && !gun.reloading && gun.reserve > 0 {
        intent.reload = true;
    }
}

fn quiet(brain: &Brain, ctx: &Ctx<'_, '_>) -> bool {
    let fresh = ctx.numbers.ticks(ctx.level.memory_seconds);
    let quiet_calls = ctx.sense.calls.iter().all(|call| {
        ctx.tick.saturating_sub(call.tick) > fresh
            || !matches!(
                call.kind,
                CallKind::Spotted | CallKind::Heard | CallKind::Down | CallKind::Device
            )
    });
    quiet_calls
        && ctx.sense.sounds.is_empty()
        && brain.memory.visible_count() == 0
        && brain.memory.lost(ctx.tick, fresh).is_none()
}

fn quiet_near(brain: &Brain, ctx: &Ctx<'_, '_>, reach: f32) -> bool {
    let fresh = ctx.numbers.ticks(ctx.level.memory_seconds);
    let me = ctx.sense.position;
    let near = |at: Vec3| flat_distance(at, me) <= reach;
    let calls = ctx.sense.calls.iter().any(|call| {
        ctx.tick.saturating_sub(call.tick) <= fresh
            && matches!(
                call.kind,
                CallKind::Spotted | CallKind::Heard | CallKind::Down | CallKind::Device
            )
            && near(call.at)
    });
    let tracks = brain
        .memory
        .tracks
        .iter()
        .any(|track| ctx.tick.saturating_sub(track.seen_tick) <= fresh && near(track.at));
    let sounds = ctx.sense.sounds.iter().any(|heard| near(heard.at));
    brain.memory.visible_count() == 0 && !calls && !tracks && !sounds
}

fn object_in_way(brain: &Brain, ctx: &Ctx<'_, '_>) -> Option<Vec3> {
    let play = &ctx.numbers.play;
    let me = ctx.sense.position;
    let ahead = flat(brain.order.goal - me).normalize_or_zero();
    let stuck = ctx.tick < brain.stuck_tick + ctx.numbers.ticks(play.wall_stuck_seconds)
        && brain.stuck_tick > 0;
    ctx.sense
        .objects
        .iter()
        .filter(|object| match object.kind {
            ObjectKind::Wall => stuck,
            ObjectKind::Trap | ObjectKind::Wire | ObjectKind::Sensor => object.enemy,
        })
        .filter(|object| {
            let to = flat(object.at - me);
            let along = to.dot(ahead);
            let reach = if object.kind == ObjectKind::Wall {
                play.object_block_m * 3.0
            } else {
                play.object_shoot_m
            };
            along > 0.0
                && along <= reach
                && (to - ahead * along).length() <= play.object_block_m * 2.0
        })
        .map(|object| object.at)
        .min_by(|a, b| flat_distance(*a, me).total_cmp(&flat_distance(*b, me)))
}

fn shoot_object(brain: &mut Brain, ctx: &Ctx<'_, '_>, at: Vec3, intent: &mut Intent) {
    turn_to(brain, ctx, at, 1.0);
    let (yaw, pitch) = look_angles(ctx.eye, at);
    let on = angle_between(brain.aim.yaw, brain.aim.pitch, yaw, pitch) <= radians(1.0);
    let gun = ctx.sense.gun;
    let still = ctx.speed_share < ctx.numbers.combat.deadzone_share;
    intent.wish = Vec3::ZERO;
    intent.gait = Gait::Still;
    intent.fire = on && still && gun.ready && gun.magazine > 0 && !brain.last_fire;
    brain.mind(Activity::Fight, Why::Object, ctx.tick);
}

fn orb_wanted(brain: &Brain, ctx: &Ctx<'_, '_>) -> Option<Vec3> {
    let order = brain.order;
    let taker = brain.orb_spot.is_some()
        || match ctx.sense.game.side {
            Side::Attack => matches!(order.step, Step::Wait | Step::Gather),
            Side::Defend => matches!(order.step, Step::Setup | Step::Rotate),
        };
    if !taker
        || ctx.sense.game.phase != GamePhase::Live
        || ctx.sense.carrying_device
        || ctx.sense.abilities[3].ready
        || !quiet_near(brain, ctx, ctx.numbers.play.cast_clear_m)
    {
        return None;
    }
    let me = ctx.sense.position;
    ctx.sense
        .orbs
        .iter()
        .filter(|orb| orb.waiting)
        .map(|orb| orb.at)
        .filter(|at| flat_distance(*at, me) <= ctx.numbers.play.orb_detour_m)
        .find(|at| {
            let mine = flat_distance(*at, me);
            !ctx.sense
                .mates
                .iter()
                .any(|mate| mate.alive && flat_distance(*at, mate.position) < mine)
        })
}

fn orb_stand(brain: &Brain, ctx: &Ctx<'_, '_>, at: Vec3) -> Option<Vec3> {
    let reach = ctx.numbers.play.orb_stand_m;
    if let Some(stand) = brain
        .orb_spot
        .filter(|stand| flat_distance(*stand, at) <= reach)
    {
        return Some(stand);
    }
    let mut corners = Vec::new();
    if !ctx.world.path(ctx.sense.position, at, &mut corners) {
        return None;
    }
    let stand = corners.last().copied().unwrap_or(at);
    (flat_distance(stand, at) <= reach).then_some(stand)
}

fn peek_share(brain: &Brain, ctx: &Ctx<'_, '_>) -> f32 {
    if brain.order.step == Step::PostPlant {
        ctx.level.post_peek
    } else {
        ctx.numbers.play.over_peek
    }
}

fn post_goal(brain: &Brain, ctx: &Ctx<'_, '_>) -> Vec3 {
    let order = brain.order;
    match ctx.sense.game.device {
        DeviceSense::Planted { at, .. } if order.step == Step::PostPlant => order
            .goal
            .lerp(Vec3::new(at.x, order.goal.y, at.z), ctx.level.post_close),
        DeviceSense::Planted { .. }
        | DeviceSense::Unknown
        | DeviceSense::Carried { .. }
        | DeviceSense::Dropped { .. }
        | DeviceSense::Over => order.goal,
    }
}

fn hazard_dodge(brain: &mut Brain, ctx: &Ctx<'_, '_>, goal: Vec3) -> Option<Vec3> {
    let order = brain.order;
    let holding = matches!(
        order.step,
        Step::Setup | Step::Wait | Step::Gather | Step::PostPlant | Step::Retake
    );
    let reach = ctx.numbers.player.radius_m;
    let hazard = ctx
        .sense
        .hazards
        .iter()
        .find(|hazard| flat_distance(hazard.center, goal) <= hazard.radius + reach)
        .filter(|_| holding);
    let Some(hazard) = hazard else {
        brain.hazard_spot = None;
        return None;
    };
    let play = &ctx.numbers.play;
    let spot = *brain.hazard_spot.get_or_insert_with(|| {
        let out = flat(goal - hazard.center).normalize_or_zero();
        let out = if out == Vec3::ZERO {
            let side = flat(order.face - goal).normalize_or_zero();
            Vec3::new(-side.z, 0.0, side.x)
        } else {
            out
        };
        let near = hazard.center + out * (hazard.radius + play.spacing_m);
        ctx.world
            .cover_from(order.face, near, play.cover_search_m)
            .filter(|spot| flat_distance(*spot, hazard.center) > hazard.radius + reach)
            .unwrap_or(near)
    });
    Some(spot)
}

fn reveal_dodge(brain: &mut Brain, ctx: &Ctx<'_, '_>, goal: Vec3) -> Vec3 {
    let order = brain.order;
    if !ctx.sense.revealed || !matches!(order.step, Step::Setup | Step::Wait | Step::PostPlant) {
        brain.reveal_spot = None;
        return goal;
    }
    let play = &ctx.numbers.play;
    *brain.reveal_spot.get_or_insert_with(|| {
        let side = flat(order.face - goal).normalize_or_zero();
        let near = goal + Vec3::new(-side.z, 0.0, side.x) * play.reveal_shift_m;
        ctx.world
            .cover_from(order.face, near, play.cover_search_m)
            .unwrap_or(near)
    })
}

fn follow(brain: &mut Brain, ctx: &Ctx<'_, '_>, intent: &mut Intent) {
    let mut order = brain.order;
    let tick = ctx.tick;
    switch_primary(ctx, intent);
    idle_reload(brain, ctx, intent);
    if let Some(at) = object_in_way(brain, ctx).filter(|_| quiet(brain, ctx)) {
        shoot_object(brain, ctx, at, intent);
        return;
    }
    let wanted = orb_wanted(brain, ctx);
    let orb = wanted.and_then(|at| orb_stand(brain, ctx, at));
    brain.orb_spot = orb;
    if let (Some(at), Some(stand)) = (wanted, orb)
        && flat_distance(stand, ctx.sense.position) <= ctx.numbers.play.arrive_m
    {
        let to = flat(at - ctx.sense.position);
        intent.goal = at;
        if to.length() > ctx.numbers.play.orb_stand_m {
            intent.wish = to.normalize_or_zero();
            intent.gait = Gait::Walk;
        } else {
            intent.wish = Vec3::ZERO;
            intent.gait = Gait::Still;
            intent.interact = true;
        }
        brain.mind(Activity::Wait, Why::Orb, tick);
        return;
    }
    let held = post_goal(brain, ctx);
    let dodged = hazard_dodge(brain, ctx, held);
    order.goal = orb
        .or(dodged)
        .unwrap_or_else(|| reveal_dodge(brain, ctx, held));
    if dodged.is_some() {
        brain.debug.why = Why::Hazard;
    } else if ctx.sense.revealed && brain.reveal_spot.is_some() {
        brain.debug.why = Why::Revealed;
    }
    intent.goal = order.goal;
    if tick < brain.hesitate_until || (orb.is_none() && tick < order.go_tick) {
        brain.mover.reset_progress(tick);
        hold_look(brain, ctx);
        let why = if tick < brain.hesitate_until {
            Why::Hesitating
        } else {
            Why::Order
        };
        brain.mind(Activity::Wait, why, tick);
        return;
    }
    let steering = brain.mover.steer(
        ctx.sense.position,
        order.goal,
        ctx.world,
        ctx.numbers,
        &mut brain.rng,
        tick,
    );
    if steering.stuck {
        intent.jump = true;
        brain.stuck_tick = tick;
        brain.debug.why = Why::Stuck;
    }
    if !steering.arrived {
        intent.wish = around_mates(ctx, steering.wish);
        intent.gait = gait_for(brain, ctx);
        if intent.gait == Gait::Run && knife_safe(brain, ctx) {
            intent.switch =
                (ctx.sense.gun.holding != WeaponChoice::Melee).then_some(WeaponChoice::Melee);
        }
        moving_look(brain, ctx, steering.ahead, steering.wish);
        let why = if steering.stuck {
            Why::Stuck
        } else {
            Why::Order
        };
        let activity = if order.step == Step::Save {
            Activity::Save
        } else {
            Activity::Move
        };
        brain.mind(activity, why, tick);
        return;
    }
    arrived(brain, ctx, intent);
}

fn knife_safe(brain: &Brain, ctx: &Ctx<'_, '_>) -> bool {
    let play = &ctx.numbers.play;
    let fresh = ctx.numbers.ticks(ctx.level.memory_seconds);
    let walk = matches!(
        brain.order.step,
        Step::Setup | Step::Rotate | Step::Gather | Step::Wait | Step::Save
    );
    let quiet_calls = ctx.sense.calls.iter().all(|call| {
        ctx.tick.saturating_sub(call.tick) > fresh
            || !matches!(
                call.kind,
                CallKind::Spotted | CallKind::Heard | CallKind::Down | CallKind::Device
            )
    });
    walk && ctx.sense.game.phase == GamePhase::Live
        && !ctx.sense.carrying_device
        && ctx.sense.sounds.is_empty()
        && quiet_calls
        && brain.memory.visible_count() == 0
        && brain.memory.lost(ctx.tick, fresh).is_none()
        && brain.mover.remaining(ctx.sense.position) >= play.knife_path_m
}

pub fn clear_line(brain: &mut Brain, ctx: &Ctx<'_, '_>, intent: &mut Intent) {
    let play = &ctx.numbers.play;
    let me = ctx.sense.position;
    let (sin, cos) = sin_cos(radians(play.line_of_fire_deg));
    let reach = ctx.numbers.player.radius_m;
    let blocking = ctx
        .sense
        .mates
        .iter()
        .filter(|mate| mate.alive && mate.human)
        .find_map(|mate| {
            let to = flat(me - mate.position);
            let ahead = flat(direction(mate.yaw, 0.0)).normalize_or_zero();
            let along = to.dot(ahead);
            let off = to - ahead * along;
            let inside = along > 0.0
                && along <= play.line_of_fire_m
                && off.length() <= along * sin / cos + reach;
            inside.then_some((ahead, off))
        });
    let Some((ahead, off)) = blocking else {
        brain.line_since = None;
        return;
    };
    let since = *brain.line_since.get_or_insert(ctx.tick);
    let busy = intent.plant || intent.defuse || intent.drop_device;
    if busy || ctx.tick < since + ctx.numbers.ticks(play.line_of_fire_seconds) {
        return;
    }
    let side = if off.length() > reach * 0.1 {
        off.normalize_or_zero()
    } else if ctx.sense.me.is_multiple_of(2) {
        Vec3::new(-ahead.z, 0.0, ahead.x)
    } else {
        Vec3::new(ahead.z, 0.0, -ahead.x)
    };
    intent.wish = side;
    intent.gait = Gait::Run;
    brain.debug.why = Why::InLine;
}

fn around_mates(ctx: &Ctx<'_, '_>, wish: Vec3) -> Vec3 {
    let reach = ctx.numbers.player.radius_m * 4.0;
    let me = ctx.sense.position;
    let ahead = ctx
        .sense
        .mates
        .iter()
        .filter(|mate| mate.alive)
        .map(|mate| flat(mate.position - me))
        .filter(|to| {
            let d = to.length();
            d < reach && to.dot(wish) > d * 0.5
        })
        .min_by(|a, b| a.length().total_cmp(&b.length()));
    let Some(to) = ahead else {
        return wish;
    };
    let side = Vec3::new(-wish.z, 0.0, wish.x);
    let lean = side.dot(to);
    let away = if lean > 0.0 || (lean == 0.0 && ctx.sense.me % 2 == 1) {
        -side
    } else {
        side
    };
    (wish + away).normalize_or_zero()
}

fn moving_look(brain: &mut Brain, ctx: &Ctx<'_, '_>, ahead: Vec3, heading: Vec3) {
    let play = &ctx.numbers.play;
    if let Some(track) = brain
        .memory
        .heard(ctx.tick, ctx.numbers.ticks(play.heard_seconds))
    {
        calm_turn(brain, ctx, track.head);
        return;
    }
    if let Some(point) = preaim(brain, ctx, heading) {
        calm_turn(brain, ctx, point);
        return;
    }
    let near_goal = flat_distance(ctx.sense.position, brain.order.goal) <= play.arrive_m * 4.0;
    let point = if near_goal {
        brain.order.face
    } else {
        flat(ahead) + Vec3::new(0.0, ctx.eye.y, 0.0)
    };
    if flat_distance(ctx.sense.position, point) > play.arrive_m {
        calm_turn(brain, ctx, point);
    }
}

fn arrived(brain: &mut Brain, ctx: &Ctx<'_, '_>, intent: &mut Intent) {
    let order = brain.order;
    let tick = ctx.tick;
    let play = &ctx.numbers.play;
    let places = ctx.world.places();
    if order.step == Step::Plant
        && ctx.sense.carrying_device
        && places.site_at(ctx.sense.position).is_some()
    {
        intent.plant = true;
        hold_look(brain, ctx);
        brain.mind(Activity::Plant, Why::Planting, tick);
        return;
    }
    if order.step == Step::Defuse
        && order.duty == Duty::Defuser
        && ctx.sense.game.side == Side::Defend
        && let DeviceSense::Planted { at, .. } = ctx.sense.game.device
        && flat_distance(ctx.sense.position, at) <= play.reach_m
    {
        intent.defuse = true;
        calm_turn(brain, ctx, at);
        brain.mind(Activity::Defuse, Why::Defusing, tick);
        return;
    }
    let crowded = ctx
        .sense
        .mates
        .iter()
        .filter(|m| m.alive)
        .find(|m| flat_distance(m.position, ctx.sense.position) < play.spacing_m * 0.5);
    if let Some(mate) = crowded {
        let away = flat(ctx.sense.position - mate.position).normalize_or_zero();
        intent.wish = if away == Vec3::ZERO { Vec3::X } else { away };
        intent.gait = Gait::Walk;
    }
    if let Some(track) = brain
        .memory
        .heard(tick, ctx.numbers.ticks(play.heard_seconds))
    {
        calm_turn(brain, ctx, track.head);
        brain.mind(Activity::Hold, Why::Heard, tick);
        return;
    }
    hold_look(brain, ctx);
    let activity = if order.step == Step::Save {
        Activity::Save
    } else {
        Activity::Hold
    };
    let why = if order.step == Step::Save {
        Why::Saving
    } else {
        Why::Arrived
    };
    brain.mind(activity, why, tick);
}

pub fn finish(brain: &mut Brain, ctx: &Ctx<'_, '_>, intent: &mut Intent) {
    let gun = ctx.sense.gun;
    let follow = clamp(
        ctx.dt / max(ctx.level.spray_settle_seconds, ctx.dt),
        0.0,
        1.0,
    );
    brain.fight.spray_yaw += (gun.recoil_yaw - brain.fight.spray_yaw) * follow;
    brain.fight.spray_pitch += (gun.recoil_pitch - brain.fight.spray_pitch) * follow;
    intent.yaw = wrap(brain.aim.yaw - brain.fight.spray_yaw);
    intent.pitch = brain.aim.pitch - brain.fight.spray_pitch;
    if intent.wish == Vec3::ZERO {
        intent.gait = Gait::Still;
    }
    if intent.fire {
        intent.jump = false;
    }
    brain.last_fire = intent.fire;
    brain.last_magazine = gun.magazine;
    brain.last_health = ctx.sense.health;
}
