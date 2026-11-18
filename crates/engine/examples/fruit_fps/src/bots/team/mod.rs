use glam::Vec3;

use super::aim::{flat, flat_distance, min};
use super::debug::{SquadDebug, Why};
use super::numbers::BotNumbers;
use super::places::{Places, ZoneId};
use super::rng::Rng;
use super::sense::{Block, Call, CallKind, GamePhase, MatchSense, Side, World};
use tactics::{Hold, SiteTactics, Tactics};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    Entry,
    Support,
    Controller,
    Initiator,
    Denial,
}

impl Role {
    pub fn from_data_name(name: &str) -> Option<Role> {
        match name {
            "entry" => Some(Role::Entry),
            "support" => Some(Role::Support),
            "controller" => Some(Role::Controller),
            "initiator" => Some(Role::Initiator),
            "denial" => Some(Role::Denial),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Duty {
    Entry,
    Smoke,
    Carrier,
    Follow,
    Lurk,
    Anchor,
    Mid,
    Retaker,
    Defuser,
    Cover,
    Saver,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Step {
    Buy,
    Setup,
    Wait,
    Gather,
    Execute,
    Plant,
    PostPlant,
    Rotate,
    Retake,
    Defuse,
    Pickup,
    Save,
    Hunt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Approach {
    Fast,
    Split,
    Default,
    Eco,
    Hold,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Utility {
    pub block: Block,
    pub target: Vec3,
    pub after_tick: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Order {
    pub duty: Duty,
    pub step: Step,
    pub goal: Vec3,
    pub zone: Option<ZoneId>,
    pub face: Vec3,
    pub go_tick: u64,
    pub careful: bool,
    pub utility: Option<Utility>,
}

impl Order {
    pub fn idle(at: Vec3) -> Order {
        Order {
            duty: Duty::Follow,
            step: Step::Buy,
            goal: at,
            zone: None,
            face: at,
            go_tick: 0,
            careful: false,
            utility: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Member {
    pub id: u32,
    pub bot: bool,
    pub alive: bool,
    pub position: Vec3,
    pub health: f32,
    pub role: Role,
    pub carrying_device: bool,
    pub gun_cost: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SquadSense<'a> {
    pub game: MatchSense,
    pub members: &'a [Member],
    pub calls: &'a [Call],
    pub enemies_alive: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Intel {
    site: Option<usize>,
    tick: u64,
    kind: CallKind,
    count: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Slot {
    duty: Duty,
    group: usize,
    rank: usize,
    hold: Hold,
}

const BLANK: Slot = Slot {
    duty: Duty::Follow,
    group: 0,
    rank: 0,
    hold: Hold {
        at: Vec3::ZERO,
        face: Vec3::ZERO,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub struct Squad {
    seed: u64,
    rng: Rng,
    orders: Vec<Order>,
    slots: Vec<Slot>,
    tactics: Option<Tactics>,
    buffer: Vec<Vec3>,
    intel: Vec<Intel>,
    round: u32,
    phase: GamePhase,
    approach: Approach,
    site: Option<usize>,
    step: Step,
    why: Why,
    since: u64,
    go_tick: u64,
    deadline: u64,
    pick_tick: u64,
    next_plan: u64,
    alive_at_go: usize,
    rotated: Option<usize>,
    defuser: Option<usize>,
    entry_way: Vec<Option<usize>>,
}

pub struct Plan<'a, 'b> {
    pub sense: &'a SquadSense<'b>,
    pub world: &'a dyn World,
    pub numbers: &'a BotNumbers,
    pub tactics: &'a Tactics,
    pub tick: u64,
}

impl Plan<'_, '_> {
    fn places(&self) -> &Places {
        self.world.places()
    }

    fn alive(&self) -> Vec<usize> {
        (0..self.sense.members.len())
            .filter(|&m| self.sense.members[m].alive)
            .collect()
    }

    fn eye(&self) -> Vec3 {
        Vec3::new(0.0, self.numbers.player.eye_height_m, 0.0)
    }

    fn site(&self, site: usize) -> &SiteTactics {
        &self.tactics.sites[site.min(self.tactics.sites.len().saturating_sub(1))]
    }

    fn seconds_left(&self) -> f32 {
        self.numbers.seconds(u64::from(self.sense.game.ticks_left))
    }
}

fn spaced(at: Vec3, face: Vec3, rank: usize, spacing: f32) -> Vec3 {
    let towards = flat(face - at).normalize_or_zero();
    let side = Vec3::new(-towards.z, 0.0, towards.x);
    let offset = (rank as f32 - 1.0) * spacing;
    at + side * offset - towards * (rank as f32 * spacing * 0.5)
}

impl Squad {
    pub fn new(seed: u64, members: usize) -> Squad {
        Squad {
            seed,
            rng: Rng::new(seed),
            orders: vec![Order::idle(Vec3::ZERO); members],
            slots: vec![BLANK; members],
            tactics: None,
            buffer: Vec::with_capacity(64),
            intel: Vec::with_capacity(32),
            round: u32::MAX,
            phase: GamePhase::Ended,
            approach: Approach::Hold,
            site: None,
            step: Step::Buy,
            why: Why::RoundStart,
            since: 0,
            go_tick: 0,
            deadline: 0,
            pick_tick: 0,
            next_plan: 0,
            alive_at_go: 0,
            rotated: None,
            defuser: None,
            entry_way: vec![None; members],
        }
    }

    pub fn order(&self, member: usize) -> Order {
        self.orders
            .get(member)
            .copied()
            .unwrap_or(Order::idle(Vec3::ZERO))
    }

    pub fn debug(&self) -> SquadDebug {
        SquadDebug {
            approach: self.approach,
            site: self.site,
            step: self.step,
            why: self.why,
            since_tick: self.since,
        }
    }

    fn set_step(&mut self, step: Step, why: Why, tick: u64) {
        if self.step != step {
            self.step = step;
            self.why = why;
            self.since = tick;
        }
    }

    pub fn plan(
        &mut self,
        sense: &SquadSense<'_>,
        world: &dyn World,
        numbers: &BotNumbers,
        tick: u64,
    ) {
        if self.orders.len() < sense.members.len() {
            self.orders
                .resize(sense.members.len(), Order::idle(Vec3::ZERO));
            self.entry_way.resize(sense.members.len(), None);
            self.slots.resize(sense.members.len(), BLANK);
        }
        let built = match self.tactics.take() {
            Some(t) => t,
            None => tactics::build(world, numbers, &mut self.buffer),
        };
        self.absorb(sense, world, &built, numbers, tick);
        let changed = sense.game.phase != self.phase || sense.game.round != self.round;
        if changed || tick >= self.next_plan {
            self.next_plan = tick + numbers.ticks(numbers.play.plan_seconds).max(1);
            let plan = Plan {
                sense,
                world,
                numbers,
                tactics: &built,
                tick,
            };
            if changed {
                self.enter(&plan);
            }
            self.revise(&plan);
        }
        self.tactics = Some(built);
    }

    fn absorb(
        &mut self,
        sense: &SquadSense<'_>,
        world: &dyn World,
        tactics: &Tactics,
        numbers: &BotNumbers,
        tick: u64,
    ) {
        let keep = numbers.ticks(numbers.play.info_seconds);
        self.intel.retain(|i| tick.saturating_sub(i.tick) <= keep);
        for call in sense.calls {
            let site = site_near(world.places(), tactics, call.at, numbers.play.site_near_m);
            self.intel.push(Intel {
                site,
                tick: call.tick,
                kind: call.kind,
                count: call.count.max(1),
            });
        }
    }

    fn pressure(&self, site: usize, numbers: &BotNumbers) -> u32 {
        self.intel
            .iter()
            .filter(|i| i.site == Some(site))
            .map(|i| match i.kind {
                CallKind::Device => numbers.play.rotate_seen,
                CallKind::Spotted | CallKind::Down => u32::from(i.count),
                CallKind::Heard => 1,
                CallKind::Hurt => 0,
            })
            .sum()
    }

    fn last_spotted(&self, site: usize) -> Option<u64> {
        self.intel
            .iter()
            .filter(|i| {
                i.site == Some(site)
                    && matches!(
                        i.kind,
                        CallKind::Spotted | CallKind::Device | CallKind::Down
                    )
            })
            .map(|i| i.tick)
            .max()
    }

    fn enter(&mut self, plan: &Plan<'_, '_>) {
        let game = plan.sense.game;
        if game.round != self.round {
            self.round = game.round;
            self.rng =
                Rng::new(self.seed ^ u64::from(game.round).wrapping_mul(0x2545_F491_4F6C_DD1D));
            self.intel.clear();
            self.site = None;
            self.rotated = None;
            self.defuser = None;
            self.approach = Approach::Hold;
            self.why = Why::RoundStart;
        }
        self.phase = game.phase;
        match (game.side, game.phase) {
            (_, GamePhase::Buy) => self.set_step(Step::Buy, Why::RoundStart, plan.tick),
            (Side::Attack, GamePhase::Live) => self.start_attack(plan),
            (Side::Defend, GamePhase::Live) => self.start_defense(plan),
            (_, GamePhase::Planted) => {
                self.deadline =
                    plan.tick + plan.numbers.ticks(plan.numbers.play.retake_wait_seconds);
                self.go_tick = u64::MAX;
                self.defuser = None;
                self.entry_way.iter_mut().for_each(|w| *w = None);
                let step = if game.side == Side::Attack {
                    Step::PostPlant
                } else {
                    Step::Retake
                };
                self.set_step(step, Why::Planting, plan.tick);
            }
            (_, GamePhase::Ended) => self.set_step(Step::Hunt, Why::Order, plan.tick),
        }
    }

    fn revise(&mut self, plan: &Plan<'_, '_>) {
        let game = plan.sense.game;
        match (game.side, game.phase) {
            (_, GamePhase::Buy) => self.buy_orders(plan),
            (Side::Attack, GamePhase::Live) => self.attack(plan),
            (Side::Defend, GamePhase::Live) => self.defend(plan),
            (Side::Attack, GamePhase::Planted) => self.post_plant(plan),
            (Side::Defend, GamePhase::Planted) => self.retake(plan),
            (_, GamePhase::Ended) => self.ended(plan),
        }
    }

    fn put(&mut self, plan: &Plan<'_, '_>, member: usize, order: Order) {
        let zone = plan.places().zone_at(order.goal);
        self.orders[member] = Order { zone, ..order };
    }

    fn simple(
        &mut self,
        plan: &Plan<'_, '_>,
        member: usize,
        duty: Duty,
        step: Step,
        hold: Hold,
        careful: bool,
    ) {
        self.put(
            plan,
            member,
            Order {
                duty,
                step,
                goal: hold.at,
                zone: None,
                face: hold.face,
                go_tick: 0,
                careful,
                utility: None,
            },
        );
    }

    fn buy_orders(&mut self, plan: &Plan<'_, '_>) {
        let center = plan.tactics.mid_spot.face;
        for member in 0..plan.sense.members.len() {
            let at = plan.sense.members[member].position;
            self.simple(
                plan,
                member,
                Duty::Follow,
                Step::Buy,
                Hold { at, face: center },
                false,
            );
        }
    }

    fn ended(&mut self, plan: &Plan<'_, '_>) {
        for member in plan.alive() {
            let at = plan.sense.members[member].position;
            let keep = self.orders[member];
            let face = if keep.step == Step::Save {
                keep.face
            } else {
                at + (keep.face - keep.goal)
            };
            self.simple(plan, member, keep.duty, Step::Hunt, Hold { at, face }, true);
        }
    }

    fn lost_round(&self, plan: &Plan<'_, '_>) -> bool {
        let odds = plan.numbers.play.lose_odds;
        plan.alive().len() as u32 <= odds[0] && plan.sense.enemies_alive >= odds[1]
    }

    fn saves(&self, plan: &Plan<'_, '_>, member: usize) -> bool {
        plan.sense.members[member].gun_cost >= plan.numbers.play.save_gun_cost
    }

    fn save_order(&mut self, plan: &Plan<'_, '_>, member: usize, home: Vec3) {
        let face = plan.sense.members[member].position + plan.eye();
        self.simple(
            plan,
            member,
            Duty::Saver,
            Step::Save,
            Hold { at: home, face },
            true,
        );
    }
}

fn site_near(places: &Places, tactics: &Tactics, at: Vec3, near: f32) -> Option<usize> {
    if let Some(site) = places.site_zone_at(at) {
        return Some(site);
    }
    let mut best: Option<(usize, f32)> = None;
    for (index, site) in tactics.sites.iter().enumerate() {
        for way in &site.ways {
            let distance = min(
                flat_distance(way.outside, at),
                flat_distance(way.attack_stage, at),
            );
            if distance <= near && best.is_none_or(|(_, d)| distance < d) {
                best = Some((index, distance));
            }
        }
    }
    best.map(|(index, _)| index)
}

mod attack;
mod defend;
mod tactics;
