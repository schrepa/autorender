use glam::Vec3;

use super::aim::{angle_between, flat, flat_distance, look_angles, max, radians, sin_cos, wrap};
use super::debug::{Activity, Why};
use super::sense::{AbilityUse, Block, DeviceSense, Gait, Intent, Side, SoundKind};
use super::team::{Duty, Line, LineKind, Step};
use super::throws::{Progress, Search, Shot};
use super::{Brain, Ctx};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Pending {
    slot: usize,
    target: Vec3,
    aimed: bool,
    start: u64,
    line: Option<Line>,
    ordered: bool,
    search: Option<Search>,
    shot: Option<Shot>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThrowCounts {
    pub solved: u32,
    pub refused: u32,
    pub thrown: u32,
    pub abandoned: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThrowState {
    Solving,
    Aiming,
    Thrown,
    Refused,
}

impl ThrowState {
    pub fn key(self) -> &'static str {
        match self {
            ThrowState::Solving => "solving",
            ThrowState::Aiming => "aiming",
            ThrowState::Thrown => "thrown",
            ThrowState::Refused => "refused",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Utilities {
    pending: Option<Pending>,
    done_order: Option<(Block, u64)>,
    next_check: u64,
    quiet_until: [u64; 4],
    avert: Option<(Vec3, u64)>,
    pub counts: ThrowCounts,
    pub last: Option<(Line, ThrowState)>,
}

impl Utilities {
    pub fn owns_view(&self) -> bool {
        self.pending.is_some_and(|pending| pending.shot.is_some())
    }

    pub fn new() -> Utilities {
        Utilities {
            pending: None,
            done_order: None,
            next_check: 0,
            quiet_until: [0; 4],
            avert: None,
            counts: ThrowCounts::default(),
            last: None,
        }
    }
}

type Choice = (usize, Vec3, Option<Line>);

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

fn from_order(brain: &mut Brain, ctx: &Ctx<'_, '_>) -> Option<Choice> {
    let utility = brain.order.utility?;
    if ctx.tick < utility.after_tick
        || brain.utilities.done_order == Some((utility.block, utility.after_tick))
    {
        return None;
    }
    brain.utilities.done_order = Some((utility.block, utility.after_tick));
    let slot = slot_with(ctx, brain, utility.block)?;
    let line = Some(utility.line);
    if utility.block == Block::Flash {
        return flash_spot(ctx, slot, utility.target).map(|spot| (slot, spot, line));
    }
    Some((slot, utility.target, line))
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

fn line_of(ctx: &Ctx<'_, '_>, slot: usize, target: Vec3) -> Option<Line> {
    let block = ctx.sense.abilities[slot].block?;
    let places = ctx.world.places();
    let kind = match (block, ctx.sense.game.side, ctx.sense.order.step) {
        (Block::DamageZone, Side::Attack, Step::PostPlant) => LineKind::DamagePlant,
        (Block::Smoke | Block::Wall | Block::DamageZone, Side::Defend, _) => LineKind::StallChoke,
        (Block::Flash, Side::Defend, Step::Retake) => LineKind::RetakeFlash,
        (Block::Flash, _, _) => LineKind::FlashCorner,
        (Block::Reveal, _, _) => LineKind::RevealSite,
        (Block::Smoke, _, _) => LineKind::SmokeHold,
        _ => return None,
    };
    let choke = places
        .chokes
        .iter()
        .enumerate()
        .map(|(index, gate)| (index, flat_distance(gate.middle(), target)))
        .filter(|(_, d)| *d <= ctx.numbers.play.walk_near_m * 0.5)
        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
        .map(|(index, _)| index);
    Some(Line {
        kind,
        site: places.site_at(target),
        choke,
    })
}

fn choose(brain: &mut Brain, ctx: &Ctx<'_, '_>) -> Option<(Choice, bool)> {
    if let Some(choice) = from_order(brain, ctx) {
        return Some((choice, true));
    }
    let (slot, target) = heal(brain, ctx)
        .or_else(|| fight_tools(brain, ctx))
        .or_else(|| push_tools(brain, ctx))
        .or_else(|| hold_tools(brain, ctx))?;
    Some(((slot, target, line_of(ctx, slot, target)), false))
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
    let Some(((slot, target, line), ordered)) = choose(brain, ctx) else {
        return;
    };
    let quiet = ctx.tick + ctx.numbers.ticks(play.utility_repeat_seconds);
    brain.utilities.quiet_until[slot] = quiet;
    let sense = ctx.sense.abilities[slot];
    let aimed = !sense.thrown
        && (sense.block.is_some_and(self_cast)
            || sense.block == Some(Block::Heal) && target == ctx.sense.position);
    brain.utilities.pending = Some(Pending {
        slot,
        target,
        aimed,
        start: ctx.tick,
        line,
        ordered,
        search: None,
        shot: None,
    });
}

fn refuse(brain: &mut Brain, pending: Pending) {
    brain.utilities.pending = None;
    brain.utilities.counts.refused += 1;
    if pending.ordered {
        brain.utilities.quiet_until[pending.slot] = 0;
        brain.utilities.done_order = None;
    }
    if let Some(line) = pending.line {
        brain.utilities.last = Some((line, ThrowState::Refused));
    }
}

fn solve(brain: &mut Brain, ctx: &Ctx<'_, '_>, mut pending: Pending) -> Option<Shot> {
    let play = &ctx.numbers.play;
    let moved = pending
        .search
        .is_some_and(|search| (search.origin() - ctx.eye).length() > play.throw_moved_m);
    if moved {
        pending.search = None;
        pending.shot = None;
    }
    if let Some(shot) = pending.shot {
        return Some(shot);
    }
    let search = pending
        .search
        .get_or_insert_with(|| Search::new(pending.slot, pending.target, ctx.eye));
    let progress = search.step(ctx.world, play);
    if let Some(line) = pending.line {
        brain.utilities.last = Some((line, ThrowState::Solving));
    }
    match progress {
        Progress::Working => {
            brain.utilities.pending = Some(pending);
            None
        }
        Progress::Refused => {
            refuse(brain, pending);
            None
        }
        Progress::Solved(shot) => {
            brain.utilities.counts.solved += 1;
            pending.shot = Some(shot);
            brain.utilities.pending = Some(pending);
            Some(shot)
        }
    }
}

fn throw(brain: &mut Brain, ctx: &Ctx<'_, '_>, pending: Pending, intent: &mut Intent) {
    intent.wish = Vec3::ZERO;
    intent.gait = Gait::Still;
    let Some(shot) = solve(brain, ctx, pending) else {
        return;
    };
    let play = &ctx.numbers.play;
    brain
        .aim
        .turn_toward(shot.yaw, shot.pitch, ctx.level.turn_deg_per_second, play, ctx.dt);
    let on = angle_between(brain.aim.yaw, brain.aim.pitch, shot.yaw, shot.pitch)
        <= radians(play.throw_aim_deg);
    let still = ctx.speed_share < ctx.numbers.combat.deadzone_share;
    if let Some(line) = pending.line {
        brain.utilities.last = Some((line, ThrowState::Aiming));
    }
    brain.mind(Activity::Utility, Why::Utility, ctx.tick);
    if !(on && still) {
        return;
    }
    let slip = radians(play.throw_error_deg * (1.0 - ctx.level.ability_skill)) * brain.rng.unit();
    let (sin, cos) = sin_cos(brain.rng.unit() * std::f32::consts::TAU);
    brain.aim.yaw = wrap(shot.yaw + cos * slip);
    brain.aim.pitch = shot.pitch + sin * slip;
    intent.ability = Some(AbilityUse {
        slot: pending.slot,
        target: pending.target,
    });
    brain.utilities.pending = None;
    brain.utilities.counts.thrown += 1;
    if let Some(line) = pending.line {
        brain.utilities.last = Some((line, ThrowState::Thrown));
    }
    if ctx.sense.abilities[pending.slot].block == Some(Block::Flash) {
        let until = ctx.tick + ctx.numbers.ticks(play.flash_avert_seconds);
        brain.utilities.avert = Some((pending.target, until));
    }
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
        if pending.shot.is_some() {
            brain.utilities.counts.abandoned += 1;
        }
        return;
    }
    if ctx.sense.abilities[pending.slot].thrown {
        throw(brain, ctx, pending, intent);
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
