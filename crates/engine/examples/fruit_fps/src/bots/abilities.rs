use glam::Vec3;

use super::aim::{angle_between, flat, flat_distance, look_angles, max, radians};
use super::debug::{Activity, Why};
use super::sense::{AbilityUse, Block, DeviceSense, Intent, Side, SoundKind};
use super::team::{Duty, Step};
use super::{Brain, Ctx};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Pending {
    slot: usize,
    target: Vec3,
    aimed: bool,
    start: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Utilities {
    pending: Option<Pending>,
    done_order: Option<(Block, u64)>,
    next_check: u64,
    quiet_until: [u64; 4],
    avert: Option<(Vec3, u64)>,
}

impl Utilities {
    pub fn new() -> Utilities {
        Utilities {
            pending: None,
            done_order: None,
            next_check: 0,
            quiet_until: [0; 4],
            avert: None,
        }
    }
}

fn slot_with(ctx: &Ctx<'_, '_>, brain: &Brain, block: Block) -> Option<usize> {
    ctx.sense
        .abilities
        .iter()
        .enumerate()
        .find(|(i, a)| {
            a.block == Some(block) && a.ready && ctx.tick >= brain.utilities.quiet_until[*i]
        })
        .map(|(i, _)| i)
}

fn self_cast(block: Block) -> bool {
    matches!(block, Block::Dash | Block::Buff)
}

fn in_range(ctx: &Ctx<'_, '_>, slot: usize, target: Vec3) -> bool {
    let range = ctx.sense.abilities[slot].range_m;
    range <= 0.0 || flat_distance(ctx.sense.position, target) <= range
}

const AROUND: [(f32, f32); 9] = [
    (0.0, 0.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (0.707, 0.707),
    (-0.707, 0.707),
    (0.707, -0.707),
    (-0.707, -0.707),
];

fn flash_spot(ctx: &Ctx<'_, '_>, slot: usize, target: Vec3) -> Option<Vec3> {
    let play = &ctx.numbers.play;
    let reach = max(ctx.sense.abilities[slot].range_m, play.flash_mate_m);
    AROUND
        .iter()
        .map(|&(x, z)| target + Vec3::new(x, 0.0, z) * play.flash_shift_m)
        .filter(|&spot| in_range(ctx, slot, spot))
        .find(|&spot| {
            let mine =
                flat_distance(ctx.sense.position, spot) <= reach && ctx.world.sees(ctx.eye, spot);
            !mine
                && !ctx.sense.mates.iter().filter(|m| m.alive).any(|mate| {
                    flat_distance(mate.position, spot) <= reach
                        && ctx.world.sees(mate.position + ctx.up, spot)
                })
        })
}

fn from_order(brain: &mut Brain, ctx: &Ctx<'_, '_>) -> Option<(usize, Vec3)> {
    let utility = brain.order.utility?;
    if ctx.tick < utility.after_tick
        || brain.utilities.done_order == Some((utility.block, utility.after_tick))
    {
        return None;
    }
    brain.utilities.done_order = Some((utility.block, utility.after_tick));
    let slot = slot_with(ctx, brain, utility.block)?;
    if utility.block == Block::Flash {
        return flash_spot(ctx, slot, utility.target).map(|spot| (slot, spot));
    }
    Some((slot, utility.target))
}

fn heal(brain: &Brain, ctx: &Ctx<'_, '_>) -> Option<(usize, Vec3)> {
    let slot = slot_with(ctx, brain, Block::Heal)?;
    if brain.memory.visible_count() > 0 {
        return None;
    }
    let below = ctx.numbers.play.heal_below;
    if ctx.sense.health < below {
        return Some((slot, ctx.sense.position));
    }
    let radius = ctx.sense.abilities[slot].radius_m;
    ctx.sense
        .mates
        .iter()
        .filter(|m| m.alive && m.health < below)
        .find(|m| {
            flat_distance(m.position, ctx.sense.position)
                <= max(radius, ctx.sense.abilities[slot].range_m)
        })
        .map(|m| (slot, m.position))
}

fn fight_tools(brain: &Brain, ctx: &Ctx<'_, '_>) -> Option<(usize, Vec3)> {
    let target = brain.memory.target(ctx.eye, ctx.tick)?;
    let play = &ctx.numbers.play;
    if ctx.sense.health < play.retreat_health {
        let slot = slot_with(ctx, brain, Block::Dash)?;
        let away = flat(ctx.sense.position - target.at).normalize_or_zero();
        return Some((
            slot,
            ctx.sense.position + away * max(ctx.sense.abilities[slot].range_m, play.cover_search_m),
        ));
    }
    let slot = slot_with(ctx, brain, Block::Buff)?;
    Some((slot, ctx.sense.position))
}

fn push_tools(brain: &Brain, ctx: &Ctx<'_, '_>) -> Option<(usize, Vec3)> {
    let order = brain.order;
    let play = &ctx.numbers.play;
    let going = matches!(order.step, Step::Execute | Step::Retake) && ctx.tick >= order.go_tick;
    if !going {
        return None;
    }
    let within = ctx.numbers.ticks(play.heard_seconds);
    let ahead = brain
        .memory
        .lost(ctx.tick, within)
        .or_else(|| brain.memory.heard(ctx.tick, within));
    if let Some(track) = ahead {
        if let Some(slot) = slot_with(ctx, brain, Block::Flash)
            && let Some(spot) = flash_spot(ctx, slot, track.head)
        {
            return Some((slot, spot));
        }
        if let Some(slot) = slot_with(ctx, brain, Block::DamageZone)
            && in_range(ctx, slot, track.at)
        {
            return Some((slot, track.at));
        }
    }
    if order.duty == Duty::Entry
        && ctx.tick <= order.go_tick + ctx.numbers.ticks(play.utility_seconds)
    {
        if let Some(slot) = slot_with(ctx, brain, Block::Buff) {
            return Some((slot, ctx.sense.position));
        }
        if let Some(slot) = slot_with(ctx, brain, Block::Dash) {
            let toward = flat(order.goal - ctx.sense.position).normalize_or_zero();
            return Some((
                slot,
                ctx.sense.position + toward * ctx.sense.abilities[slot].range_m,
            ));
        }
    }
    if order.step == Step::Retake
        && let DeviceSense::Planted { at, .. } = ctx.sense.game.device
        && let Some(slot) = slot_with(ctx, brain, Block::Reveal)
    {
        return Some((slot, at));
    }
    None
}

fn nearest_choke(ctx: &Ctx<'_, '_>, skip_face: bool) -> Option<Vec3> {
    let places = ctx.world.places();
    let face = flat(ctx.sense.order.face);
    places
        .chokes
        .iter()
        .map(|c| c.middle())
        .filter(|m| !skip_face || flat_distance(*m, face) > ctx.numbers.play.walk_near_m * 0.5)
        .min_by(|a, b| {
            flat_distance(*a, ctx.sense.position).total_cmp(&flat_distance(*b, ctx.sense.position))
        })
}

fn hold_tools(brain: &Brain, ctx: &Ctx<'_, '_>) -> Option<(usize, Vec3)> {
    let order = brain.order;
    let play = &ctx.numbers.play;
    let holding = brain.debug.activity == Activity::Hold;
    let side = ctx.sense.game.side;
    if holding
        && matches!(order.step, Step::Setup | Step::PostPlant)
        && let Some(slot) = slot_with(ctx, brain, Block::Trap)
    {
        let flank = nearest_choke(ctx, true).or_else(|| nearest_choke(ctx, false))?;
        if in_range(ctx, slot, flank) || ctx.sense.abilities[slot].range_m <= 0.0 {
            return Some((slot, flank));
        }
    }
    if side == Side::Defend && matches!(order.step, Step::Setup | Step::Rotate) {
        let within = ctx.numbers.ticks(play.heard_seconds);
        let pushing = brain
            .memory
            .tracks
            .iter()
            .filter(|t| ctx.tick.saturating_sub(t.seen_tick) <= within)
            .count();
        if pushing >= play.push_count as usize {
            let choke = nearest_choke(ctx, false)?;
            for block in [Block::Smoke, Block::Wall, Block::DamageZone] {
                if let Some(slot) = slot_with(ctx, brain, block)
                    && in_range(ctx, slot, choke)
                {
                    return Some((slot, choke));
                }
            }
        }
    }
    if order.step == Step::Plant && brain.debug.activity == Activity::Plant {
        let slot = slot_with(ctx, brain, Block::Wall)?;
        let toward = flat(order.face - ctx.sense.position).normalize_or_zero();
        return Some((slot, ctx.sense.position + toward * play.reach_m * 2.0));
    }
    if side == Side::Attack
        && order.step == Step::PostPlant
        && let DeviceSense::Planted { at, defuser, .. } = ctx.sense.game.device
    {
        let heard = brain
            .memory
            .tracks
            .iter()
            .any(|t| t.sound == Some(SoundKind::Defuse));
        if defuser.is_some() || heard {
            let slot = slot_with(ctx, brain, Block::DamageZone)?;
            return in_range(ctx, slot, at).then_some((slot, at));
        }
    }
    None
}

fn choose(brain: &mut Brain, ctx: &Ctx<'_, '_>) -> Option<(usize, Vec3)> {
    if let Some(choice) = from_order(brain, ctx) {
        return Some(choice);
    }
    heal(brain, ctx)
        .or_else(|| fight_tools(brain, ctx))
        .or_else(|| push_tools(brain, ctx))
        .or_else(|| hold_tools(brain, ctx))
}

pub fn consider(brain: &mut Brain, ctx: &Ctx<'_, '_>, intent: &mut Intent) {
    let play = &ctx.numbers.play;
    if let Some((from, until)) = brain.utilities.avert {
        if ctx.tick < until {
            let away = ctx.eye + flat(ctx.eye - from).normalize_or_zero() * play.flash_shift_m;
            let (yaw, pitch) = look_angles(ctx.eye, away);
            brain
                .aim
                .turn_toward(yaw, pitch, ctx.level.turn_deg_per_second, play, ctx.dt);
            return;
        }
        brain.utilities.avert = None;
    }
    if let Some(pending) = brain.utilities.pending {
        release(brain, ctx, pending, intent);
        return;
    }
    if ctx.tick < brain.utilities.next_check || intent.defuse {
        return;
    }
    brain.utilities.next_check = ctx.tick + ctx.numbers.ticks(play.utility_seconds).max(1);
    let Some((slot, target)) = choose(brain, ctx) else {
        return;
    };
    let quiet = ctx.tick + ctx.numbers.ticks(play.utility_repeat_seconds);
    brain.utilities.quiet_until[slot] = quiet;
    if !brain.rng.chance(ctx.level.ability_use) {
        return;
    }
    let block = ctx.sense.abilities[slot].block;
    let aimed =
        block.is_some_and(self_cast) || block == Some(Block::Heal) && target == ctx.sense.position;
    brain.utilities.pending = Some(Pending {
        slot,
        target,
        aimed,
        start: ctx.tick,
    });
}

fn release(brain: &mut Brain, ctx: &Ctx<'_, '_>, pending: Pending, intent: &mut Intent) {
    let play = &ctx.numbers.play;
    let fighting = brain.debug.activity == Activity::Fight;
    let stale = ctx.tick.saturating_sub(pending.start)
        > ctx
            .numbers
            .ticks(play.utility_seconds * play.utility_patience);
    if stale || (fighting && !pending.aimed) {
        brain.utilities.pending = None;
        return;
    }
    let ready = if pending.aimed {
        true
    } else {
        let (yaw, pitch) = look_angles(ctx.eye, pending.target);
        brain
            .aim
            .turn_toward(yaw, pitch, ctx.level.turn_deg_per_second, play, ctx.dt);
        angle_between(brain.aim.yaw, brain.aim.pitch, yaw, pitch) <= radians(play.utility_aim_deg)
    };
    if ready {
        intent.ability = Some(AbilityUse {
            slot: pending.slot,
            target: pending.target,
        });
        brain.utilities.pending = None;
        if ctx.sense.abilities[pending.slot].block == Some(Block::Flash) {
            let until = ctx.tick + ctx.numbers.ticks(play.flash_avert_seconds);
            brain.utilities.avert = Some((pending.target, until));
        }
        brain.mind(Activity::Utility, Why::Utility, ctx.tick);
    }
}
