mod abilities;
mod aim;
mod debug;
mod fighter;
mod memory;
mod moving;
mod numbers;
mod places;
mod rng;
mod sense;
mod team;

use glam::Vec3;

pub use debug::{Activity, BotDebug, SquadDebug, Why};
pub use numbers::{
    BotNumbers, BotPlayer, BotRound, BotsTable, Level, LevelNumbers, Levels, PlayNumbers,
};
pub use places::{
    Angle, CoverBox, Gate, LayoutCallout, LayoutChoke, LayoutCover, LayoutPlant, LayoutPortal,
    LayoutSpawn, LayoutZone, MapLayout, Places, PlacesError, Site, Spot, Zone, ZoneId,
};
pub use sense::{
    AbilitySense, AbilityUse, Block, Call, CallKind, DeviceSense, Gait, GamePhase, GunClass,
    GunSense, Heard, Intent, MatchSense, Mate, Seen, Sense, Side, Smoke, SoundKind, WeaponChoice,
    World,
};
pub use team::{Approach, Duty, Member, Order, Role, Squad, SquadSense, Step, Utility};

use abilities::Utilities;
use aim::Aim;
use fighter::{Fight, Glance};
use memory::{Look, Memory};
use moving::Mover;
use rng::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct Brain {
    level: Level,
    rng: Rng,
    memory: Memory,
    aim: Aim,
    mover: Mover,
    fight: Fight,
    glance: Glance,
    utilities: Utilities,
    order: Order,
    round: Option<u32>,
    hesitate_until: u64,
    buy_tick: Option<u64>,
    debug: BotDebug,
    last_magazine: u32,
    last_fire: bool,
    last_health: f32,
    was_alive: bool,
    last_heard_call: Option<u64>,
}

pub(crate) struct Ctx<'a, 'b> {
    pub sense: &'a Sense<'b>,
    pub world: &'a dyn World,
    pub numbers: &'a BotNumbers,
    pub level: &'a LevelNumbers,
    pub tick: u64,
    pub eye: Vec3,
    pub up: Vec3,
    pub dt: f32,
    pub speed_share: f32,
}

impl Brain {
    pub fn new(seed: u64, level: Level) -> Brain {
        let order = Order::idle(Vec3::ZERO);
        Brain {
            level,
            rng: Rng::new(seed),
            memory: Memory::new(),
            aim: Aim::new(),
            mover: Mover::new(),
            fight: Fight::new(),
            glance: Glance::new(),
            utilities: Utilities::new(),
            order,
            round: None,
            hesitate_until: 0,
            buy_tick: None,
            debug: BotDebug {
                level,
                duty: order.duty,
                step: order.step,
                activity: Activity::Wait,
                goal_zone: None,
                target: None,
                why: Why::RoundStart,
                since_tick: 0,
            },
            last_magazine: 0,
            last_fire: false,
            last_health: 0.0,
            was_alive: false,
            last_heard_call: None,
        }
    }

    pub fn round_start(&mut self) {
        self.memory.clear();
        self.mover.clear();
        self.fight = Fight::new();
        self.glance = Glance::new();
        self.utilities = Utilities::new();
        self.aim.synced = false;
        self.buy_tick = None;
        self.hesitate_until = 0;
        self.last_heard_call = None;
        self.debug.why = Why::RoundStart;
    }

    pub fn debug(&self) -> BotDebug {
        self.debug
    }

    pub(crate) fn mind(&mut self, activity: Activity, why: Why, tick: u64) {
        if self.debug.activity != activity {
            self.debug.activity = activity;
            self.debug.why = why;
            self.debug.since_tick = tick;
        }
    }

    fn take_order(&mut self, ctx: &Ctx<'_, '_>) {
        let order = ctx.sense.order;
        let moved = (order.goal - self.order.goal).length() > ctx.numbers.play.repath_m;
        if order.step != self.order.step || order.duty != self.order.duty || moved {
            let pause = self.rng.range(ctx.level.hesitate_seconds);
            self.hesitate_until = ctx.tick + ctx.numbers.ticks(pause);
            self.debug.why = Why::Order;
            self.debug.since_tick = ctx.tick;
        }
        self.order = order;
        self.debug.duty = order.duty;
        self.debug.step = order.step;
        self.debug.goal_zone = order.zone;
    }

    fn dead(&mut self, sense: &Sense<'_>, world: &dyn World, tick: u64, intent: &mut Intent) {
        if self.was_alive {
            let at = self
                .memory
                .lost(tick, u64::MAX)
                .map_or(sense.position, |t| t.at);
            intent.call = Some(Call {
                from: sense.me as u32,
                kind: CallKind::Down,
                at,
                zone: world.places().zone_at(at),
                count: 1,
                tick,
            });
        }
        self.was_alive = false;
        self.debug.target = None;
        self.mind(Activity::Dead, Why::Order, tick);
    }

    fn call(&mut self, ctx: &Ctx<'_, '_>, intent: &mut Intent) {
        if intent.call.is_some() {
            return;
        }
        let play = &ctx.numbers.play;
        let repeat = ctx.numbers.ticks(play.call_repeat_seconds);
        let places = ctx.world.places();
        let me = ctx.sense.me as u32;
        if let Some(track) = self.memory.take_call(ctx.tick, repeat) {
            let kind = if track.carries_device {
                CallKind::Device
            } else {
                CallKind::Spotted
            };
            let count = u8::try_from(self.memory.visible_count()).unwrap_or(u8::MAX);
            intent.call = Some(Call {
                from: me,
                kind,
                at: track.at,
                zone: places.zone_at(track.at),
                count,
                tick: ctx.tick,
            });
            return;
        }
        let health = ctx.sense.health;
        if self.last_health >= play.hurt_call_health && health < play.hurt_call_health {
            let at = ctx.sense.position;
            intent.call = Some(Call {
                from: me,
                kind: CallKind::Hurt,
                at,
                zone: places.zone_at(at),
                count: 1,
                tick: ctx.tick,
            });
            return;
        }
        let quiet = self
            .last_heard_call
            .is_none_or(|t| ctx.tick.saturating_sub(t) >= repeat);
        if let Some(track) = self.memory.heard(ctx.tick, 0).filter(|_| quiet) {
            self.last_heard_call = Some(ctx.tick);
            intent.call = Some(Call {
                from: me,
                kind: CallKind::Heard,
                at: track.at,
                zone: places.zone_at(track.at),
                count: 1,
                tick: ctx.tick,
            });
        }
    }
}

pub fn think(
    brain: &mut Brain,
    sense: &Sense<'_>,
    world: &dyn World,
    numbers: &BotNumbers,
    tick: u64,
) -> Intent {
    let mut intent = Intent::rest(sense);
    if brain.round != Some(sense.game.round) {
        brain.round = Some(sense.game.round);
        brain.round_start();
    }
    if !sense.alive {
        brain.dead(sense, world, tick, &mut intent);
        return intent;
    }
    if !brain.was_alive {
        brain.aim.synced = false;
        brain.last_health = sense.health;
        brain.was_alive = true;
    }
    if !brain.aim.synced {
        brain.aim.sync(sense.yaw, sense.pitch);
    }
    let level = numbers.level(brain.level);
    let player = &numbers.player;
    let eye_height = if sense.crouched {
        player.crouch_eye_height_m
    } else {
        player.eye_height_m
    };
    let ctx = Ctx {
        sense,
        world,
        numbers,
        level,
        tick,
        eye: sense.position + Vec3::new(0.0, eye_height, 0.0),
        up: Vec3::new(0.0, player.eye_height_m, 0.0),
        dt: numbers.dt(),
        speed_share: aim::flat(sense.velocity).length() / player.run_speed_mps,
    };
    brain.take_order(&ctx);
    let look = Look {
        eye: ctx.eye,
        yaw: brain.aim.yaw,
        pitch: brain.aim.pitch,
    };
    brain
        .memory
        .observe(sense, level, numbers, &look, &mut brain.rng, tick);
    if sense.game.phase == GamePhase::Buy {
        fighter::buy(brain, &ctx, &mut intent);
    } else {
        fighter::act(brain, &ctx, &mut intent);
        abilities::consider(brain, &ctx, &mut intent);
    }
    brain.call(&ctx, &mut intent);
    fighter::finish(brain, &ctx, &mut intent);
    intent
}
