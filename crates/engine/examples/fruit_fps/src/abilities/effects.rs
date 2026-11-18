use std::collections::BTreeMap;

use glam::{Mat3, Vec3};

use super::numbers::{
    AbilityId, AbilitySpec, Action, Delivery, Falloff, FlashSpec, Lay, Modifiers, Onto, Place,
    SensorSpec, SmokeSpec, Throw, TrapShape, TrapSpec, WallSpec, ZoneSpec,
};
use super::shapes::{
    closest_on_segment, flat, frame, greater, lesser, segment_distance, segment_hits_box,
    segment_hits_sphere,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ActorId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EffectId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Actor {
    pub id: ActorId,
    pub team: usize,
    pub feet: Vec3,
    pub eye: Vec3,
    pub view: Vec3,
    pub radius: f32,
    pub height: f32,
}

impl Actor {
    pub fn centre(&self) -> Vec3 {
        self.feet + Vec3::Y * (self.height * 0.5)
    }

    pub fn top(&self) -> Vec3 {
        self.feet + Vec3::Y * self.height
    }

    pub fn distance_to(&self, point: Vec3) -> f32 {
        (point - closest_on_segment(point, self.feet, self.top())).length()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aim {
    pub origin: Vec3,
    pub direction: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldHit {
    pub distance: f32,
    pub normal: Vec3,
    pub bounce: f32,
}

pub trait World {
    fn ray(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<WorldHit>;
    fn sphere_sweep(&self, centre: Vec3, radius: f32, motion: Vec3) -> Option<WorldHit>;
    fn add_box(
        &mut self,
        id: u32,
        ability: AbilityId,
        centre: Vec3,
        half_extents: Vec3,
        axes: Mat3,
    );
    fn remove_box(&mut self, id: u32);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CueKind {
    Cast,
    Throw,
    Bounce,
    FlashBurst,
    Explosion,
    SmokeBurst,
    ZoneStart,
    WallUp,
    WallDown,
    TrapPlaced,
    TrapArmed,
    TrapTriggered,
    SensorPlaced,
    Destroyed,
    StrikeWarning,
    StrikeImpact,
    Pulse,
    Dash,
    TransformStart,
    TransformEnd,
    HealStart,
    Glare,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Damage {
        target: ActorId,
        amount: f32,
        by: ActorId,
        ability: AbilityId,
    },
    Heal {
        target: ActorId,
        amount: f32,
        by: ActorId,
        ability: AbilityId,
    },
    Shield {
        target: ActorId,
        amount: f32,
        until: Option<u64>,
        by: ActorId,
        ability: AbilityId,
    },
    BuffStarted {
        actor: ActorId,
        effect: EffectId,
        modifiers: Modifiers,
        until: u64,
        ability: AbilityId,
    },
    BuffEnded {
        actor: ActorId,
        effect: EffectId,
    },
    Blinded {
        actor: ActorId,
        amount: f32,
        until: u64,
        by: ActorId,
        ability: AbilityId,
    },
    RevealStarted {
        actor: ActorId,
        team: usize,
        until: u64,
        by: ActorId,
        ability: AbilityId,
    },
    RevealEnded {
        actor: ActorId,
        team: usize,
    },
    Moved {
        actor: ActorId,
        by: Vec3,
    },
    SolidAdded {
        effect: EffectId,
        id: u32,
        centre: Vec3,
        half_extents: Vec3,
        axes: Mat3,
    },
    SolidRemoved {
        effect: EffectId,
        id: u32,
    },
    TrapArmed {
        effect: EffectId,
    },
    TrapTriggered {
        effect: EffectId,
        actor: ActorId,
    },
    Destroyed {
        effect: EffectId,
        by: ActorId,
    },
    Transformed {
        actor: ActorId,
        until: u64,
        refresh_signature_on_kill: bool,
        ability: AbilityId,
    },
    TransformEnded {
        actor: ActorId,
    },
    Ended {
        effect: EffectId,
    },
    Cue {
        cue: CueKind,
        ability: AbilityId,
        at: Vec3,
        size: f32,
        ticks: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Flight {
    pub(super) position: Vec3,
    pub(super) velocity: Vec3,
    pub(super) bounces_left: u32,
    pub(super) travelled: f32,
    pub(super) resting: bool,
    pub(super) throw: Throw,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct WallState {
    pub(super) centre: Vec3,
    pub(super) half_extents: Vec3,
    pub(super) axes: Mat3,
    pub(super) spec: WallSpec,
    pub(super) health: Option<f32>,
    pub(super) solid_id: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct TrapState {
    pub(super) a: Vec3,
    pub(super) b: Vec3,
    pub(super) normal_a: Vec3,
    pub(super) normal_b: Vec3,
    pub(super) spec: TrapSpec,
    pub(super) health: f32,
    pub(super) armed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SensorState {
    pub(super) at: Vec3,
    pub(super) normal: Vec3,
    pub(super) spec: SensorSpec,
    pub(super) health: f32,
    pub(super) armed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct GlareState {
    pub(super) origin: Vec3,
    pub(super) direction: Vec3,
    pub(super) windup_ticks: u32,
    pub(super) cos_half_angle: f32,
    pub(super) flash: FlashSpec,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum State {
    Pending,
    Flight(Flight),
    Smoke {
        centre: Vec3,
        spec: SmokeSpec,
    },
    Zone {
        centre: Vec3,
        spec: ZoneSpec,
    },
    Wall(WallState),
    Trap(TrapState),
    Sensor(SensorState),
    Strike {
        centre: Vec3,
        warn_ticks: u32,
        falloff: Falloff,
    },
    Glare(GlareState),
    Heal {
        targets: Vec<ActorId>,
        heal_per_tick: f32,
        ticks: u32,
    },
    Dash {
        direction: Vec3,
        step: f32,
        ticks: u32,
    },
    Transform {
        ticks: u32,
        refresh_signature_on_kill: bool,
    },
    Done,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Living {
    pub(super) id: EffectId,
    pub(super) ability: AbilityId,
    pub(super) caster: Actor,
    pub(super) aim: Aim,
    pub(super) delivery: Delivery,
    pub(super) action: Action,
    pub(super) born: u64,
    pub(super) state: State,
}

impl Living {
    pub(super) fn age(&self, now: u64) -> u32 {
        now.saturating_sub(self.born).min(u32::MAX as u64) as u32
    }

    pub(super) fn smoke_radius(spec: &SmokeSpec, age: u32) -> f32 {
        if spec.grow_ticks == 0 || age + 1 >= spec.grow_ticks {
            spec.radius_m
        } else {
            spec.radius_m * (age + 1) as f32 / spec.grow_ticks as f32
        }
    }

    pub(super) fn blocks_sight(&self, now: u64, a: Vec3, b: Vec3) -> bool {
        match &self.state {
            State::Smoke { centre, spec } => {
                let age = self.age(now);
                age < spec.ticks
                    && segment_hits_sphere(a, b, *centre, Living::smoke_radius(spec, age))
            }
            State::Wall(wall) => {
                !wall.spec.solid
                    && segment_hits_box(a, b, wall.centre, wall.half_extents, wall.axes)
            }
            State::Pending
            | State::Flight(_)
            | State::Zone { .. }
            | State::Trap(_)
            | State::Sensor(_)
            | State::Strike { .. }
            | State::Glare(_)
            | State::Heal { .. }
            | State::Dash { .. }
            | State::Transform { .. }
            | State::Done => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Buff {
    pub(super) actor: ActorId,
    pub(super) effect: EffectId,
    pub(super) modifiers: Modifiers,
    pub(super) until: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Blind {
    pub(super) actor: ActorId,
    pub(super) amount: f32,
    pub(super) start: u64,
    pub(super) ticks: u32,
    pub(super) fade_ticks: u32,
}

impl Blind {
    pub(super) fn value(&self, now: u64) -> f32 {
        let elapsed = now.saturating_sub(self.start);
        if elapsed >= self.ticks as u64 {
            return 0.0;
        }
        let left = (self.ticks as u64 - elapsed) as u32;
        let fade = self.fade_ticks.min(self.ticks);
        if fade == 0 || left >= fade {
            self.amount
        } else {
            self.amount * left as f32 / fade as f32
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Marks {
    pub(super) buffs: Vec<Buff>,
    pub(super) blinds: Vec<Blind>,
    pub(super) reveals: BTreeMap<(usize, ActorId), u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Effects {
    solid_id_base: u32,
    next: u32,
    pub(super) now: u64,
    pub(super) livings: Vec<Living>,
    pub(super) marks: Marks,
}

enum Step {
    Stay,
    Launch(Flight),
    Burst { point: Vec3, normal: Vec3 },
    Finish,
}

struct Run<'a, W: World> {
    world: &'a mut W,
    actors: &'a [Actor],
    tick: u64,
    sight: &'a [Living],
    marks: &'a mut Marks,
    solid_id_base: u32,
    events: Vec<Event>,
}

impl Effects {
    pub fn new(solid_id_base: u32) -> Effects {
        Effects {
            solid_id_base,
            next: 0,
            now: 0,
            livings: Vec::new(),
            marks: Marks::default(),
        }
    }

    pub fn cast(
        &mut self,
        caster: &Actor,
        ability: &AbilitySpec,
        aim: Aim,
        tick: u64,
    ) -> Vec<Event> {
        let id = EffectId(self.next);
        self.next += 1;
        self.livings.push(Living {
            id,
            ability: ability.id,
            caster: *caster,
            aim,
            delivery: ability.delivery,
            action: ability.action,
            born: tick,
            state: State::Pending,
        });
        vec![Event::Cue {
            cue: CueKind::Cast,
            ability: ability.id,
            at: caster.eye,
            size: caster.radius,
            ticks: 0,
        }]
    }

    pub fn tick(&mut self, world: &mut impl World, actors: &[Actor], tick: u64) -> Vec<Event> {
        self.now = tick;
        let mut livings = std::mem::take(&mut self.livings);
        let sight: Vec<Living> = livings
            .iter()
            .filter(|living| matches!(living.state, State::Smoke { .. } | State::Wall(_)))
            .cloned()
            .collect();
        let mut run = Run {
            world,
            actors,
            tick,
            sight: &sight,
            marks: &mut self.marks,
            solid_id_base: self.solid_id_base,
            events: Vec::new(),
        };
        for living in livings.iter_mut() {
            run.advance(living);
        }
        livings.retain(|living| living.state != State::Done);
        run.expire();
        let events = run.events;
        self.livings = livings;
        events
    }

    pub fn end_round(&mut self, world: &mut impl World) -> Vec<Event> {
        let mut events = Vec::new();
        for living in std::mem::take(&mut self.livings) {
            if let State::Wall(WallState {
                solid_id: Some(id), ..
            }) = living.state
            {
                world.remove_box(id);
                events.push(Event::SolidRemoved {
                    effect: living.id,
                    id,
                });
            }
            if let State::Transform { .. } = living.state {
                events.push(Event::TransformEnded {
                    actor: living.caster.id,
                });
            }
            events.push(Event::Ended { effect: living.id });
        }
        for buff in std::mem::take(&mut self.marks.buffs) {
            events.push(Event::BuffEnded {
                actor: buff.actor,
                effect: buff.effect,
            });
        }
        for (team, actor) in std::mem::take(&mut self.marks.reveals).into_keys() {
            events.push(Event::RevealEnded { actor, team });
        }
        self.marks.blinds.clear();
        events
    }

    pub fn damage_effect(
        &mut self,
        world: &mut impl World,
        effect: EffectId,
        amount: f32,
        by: ActorId,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        let Some(index) = self.livings.iter().position(|living| living.id == effect) else {
            return events;
        };
        let living = &mut self.livings[index];
        let (health, at, solid) = match &mut living.state {
            State::Wall(WallState {
                health: Some(health),
                centre,
                solid_id,
                ..
            }) => (health, *centre, *solid_id),
            State::Trap(trap) => (&mut trap.health, trap.a, None),
            State::Sensor(sensor) => (&mut sensor.health, sensor.at, None),
            State::Wall(WallState { health: None, .. })
            | State::Pending
            | State::Flight(_)
            | State::Smoke { .. }
            | State::Zone { .. }
            | State::Strike { .. }
            | State::Glare(_)
            | State::Heal { .. }
            | State::Dash { .. }
            | State::Transform { .. }
            | State::Done => return events,
        };
        *health -= amount;
        if *health > 0.0 {
            return events;
        }
        if let Some(id) = solid {
            world.remove_box(id);
            events.push(Event::SolidRemoved { effect, id });
        }
        events.push(Event::Destroyed { effect, by });
        events.push(Event::Cue {
            cue: CueKind::Destroyed,
            ability: living.ability,
            at,
            size: 0.0,
            ticks: 0,
        });
        events.push(Event::Ended { effect });
        self.livings.remove(index);
        events
    }
}

impl<W: World> Run<'_, W> {
    fn cue(&mut self, cue: CueKind, ability: AbilityId, at: Vec3, size: f32, ticks: u32) {
        self.events.push(Event::Cue {
            cue,
            ability,
            at,
            size,
            ticks,
        });
    }

    fn present(&self, actor: ActorId) -> Option<&Actor> {
        self.actors.iter().find(|other| other.id == actor)
    }

    fn sight_blocked(&self, a: Vec3, b: Vec3) -> bool {
        self.sight
            .iter()
            .any(|living| living.blocks_sight(self.tick, a, b))
    }

    fn clear_line(&self, from: Vec3, to: Vec3) -> bool {
        let span = to - from;
        let length = span.length();
        length <= 0.0 || self.world.ray(from, span / length, length).is_none()
    }

    fn buff(&mut self, living: &Living, actor: ActorId, modifiers: Modifiers, until: u64) {
        let effect = living.id;
        if let Some(buff) = self
            .marks
            .buffs
            .iter_mut()
            .find(|buff| buff.actor == actor && buff.effect == effect)
        {
            buff.until = buff.until.max(until);
            buff.modifiers = modifiers;
            return;
        }
        self.marks.buffs.push(Buff {
            actor,
            effect,
            modifiers,
            until,
        });
        self.events.push(Event::BuffStarted {
            actor,
            effect,
            modifiers,
            until,
            ability: living.ability,
        });
    }

    fn reveal(&mut self, living: &Living, actor: ActorId, until: u64) {
        let team = living.caster.team;
        match self.marks.reveals.get_mut(&(team, actor)) {
            Some(existing) => *existing = (*existing).max(until),
            None => {
                self.marks.reveals.insert((team, actor), until);
                self.events.push(Event::RevealStarted {
                    actor,
                    team,
                    until,
                    by: living.caster.id,
                    ability: living.ability,
                });
            }
        }
    }

    fn flash(&mut self, living: &Living, origin: Vec3, spec: FlashSpec, cone: Option<(Vec3, f32)>) {
        for actor in self.actors {
            if cone.is_some() && actor.id == living.caster.id {
                continue;
            }
            let to = origin - actor.eye;
            let distance = to.length();
            if distance > spec.range_m {
                continue;
            }
            let amount = if distance > 0.0 {
                let toward = to / distance;
                if let Some((direction, cos_half_angle)) = cone
                    && direction.dot(-toward) < cos_half_angle
                {
                    continue;
                }
                if self.world.ray(actor.eye, toward, distance).is_some() {
                    continue;
                }
                let facing = greater(actor.view.dot(toward), 0.0);
                (spec.away_factor + (1.0 - spec.away_factor) * facing)
                    * (1.0 - distance / spec.range_m)
            } else {
                1.0
            };
            let ticks = (amount * spec.max_blind_ticks as f32).round() as u32;
            if ticks == 0 {
                continue;
            }
            self.marks.blinds.push(Blind {
                actor: actor.id,
                amount,
                start: self.tick,
                ticks,
                fade_ticks: spec.fade_ticks,
            });
            self.events.push(Event::Blinded {
                actor: actor.id,
                amount,
                until: self.tick + ticks as u64,
                by: living.caster.id,
                ability: living.ability,
            });
        }
    }

    fn falloff(&mut self, living: &Living, centre: Vec3, falloff: Falloff) {
        for actor in self.actors {
            if actor.team == living.caster.team && !falloff.hurts_team {
                continue;
            }
            let distance = actor.distance_to(centre);
            if distance > falloff.radius_m {
                continue;
            }
            if falloff.line_of_sight && !self.clear_line(centre, actor.centre()) {
                continue;
            }
            let share = if falloff.radius_m > 0.0 {
                distance / falloff.radius_m
            } else {
                0.0
            };
            self.events.push(Event::Damage {
                target: actor.id,
                amount: falloff.max + (falloff.min - falloff.max) * share,
                by: living.caster.id,
                ability: living.ability,
            });
        }
    }

    fn place(&self, living: &Living, place: Place) -> (Vec3, Vec3) {
        let aim = living.aim;
        let (point, normal) = match self.world.ray(aim.origin, aim.direction, place.range_m) {
            Some(hit) => (
                aim.origin + aim.direction * hit.distance + hit.normal * place.lift_m,
                hit.normal,
            ),
            None => (aim.origin + aim.direction * place.range_m, -aim.direction),
        };
        match place.onto {
            Onto::Surface => (point, normal),
            Onto::Ground => match self.world.ray(point, Vec3::NEG_Y, place.range_m) {
                Some(ground) => (
                    point + Vec3::NEG_Y * ground.distance + Vec3::Y * place.lift_m,
                    ground.normal,
                ),
                None => (point, Vec3::Y),
            },
        }
    }

    fn advance(&mut self, living: &mut Living) {
        loop {
            if living.born > self.tick || living.state == State::Done {
                return;
            }
            match self.step(living) {
                Step::Stay => return,
                Step::Launch(flight) => {
                    living.state = State::Flight(flight);
                    living.born = self.tick;
                    self.cue(
                        CueKind::Throw,
                        living.ability,
                        flight.position,
                        flight.throw.radius_m,
                        0,
                    );
                    return;
                }
                Step::Burst { point, normal } => {
                    living.state = self.act(living, point, normal);
                    living.born = self.tick;
                    if living.state == State::Done {
                        self.events.push(Event::Ended { effect: living.id });
                        return;
                    }
                }
                Step::Finish => {
                    living.state = State::Done;
                    self.events.push(Event::Ended { effect: living.id });
                    return;
                }
            }
        }
    }

    fn step(&mut self, living: &mut Living) -> Step {
        let age = living.age(self.tick);
        let snapshot = living.clone();
        let ability = living.ability;
        match &mut living.state {
            State::Pending => self.resolve(&snapshot),
            State::Flight(flight) => self.fly(flight, age, ability),
            State::Smoke { spec, .. } => finish_at(age, spec.ticks),
            State::Zone { centre, spec } => {
                let (centre, spec) = (*centre, *spec);
                if age >= spec.ticks {
                    return Step::Finish;
                }
                self.zone(&snapshot, centre, spec);
                Step::Stay
            }
            State::Wall(wall) => self.wall_step(&snapshot, *wall, age),
            State::Trap(trap) => self.trap_step(&snapshot, trap, age),
            State::Sensor(sensor) => self.sensor_step(&snapshot, sensor, age),
            State::Strike {
                centre,
                warn_ticks,
                falloff,
            } => {
                if age < *warn_ticks {
                    return Step::Stay;
                }
                let (centre, falloff) = (*centre, *falloff);
                self.falloff(&snapshot, centre, falloff);
                self.cue(CueKind::StrikeImpact, ability, centre, falloff.radius_m, 0);
                Step::Finish
            }
            State::Glare(glare) => self.glare_step(&snapshot, *glare, age),
            State::Heal {
                targets,
                heal_per_tick,
                ticks,
            } => self.heal_step(&snapshot, targets, *heal_per_tick, finish_at(age, *ticks)),
            State::Dash {
                direction,
                step,
                ticks,
            } => self.dash_step(&snapshot, *direction, *step, finish_at(age, *ticks)),
            State::Transform { ticks, .. } => self.transform_step(&snapshot, age >= *ticks),
            State::Done => Step::Stay,
        }
    }

    fn resolve(&mut self, living: &Living) -> Step {
        match living.delivery {
            Delivery::Caster => Step::Burst {
                point: living.caster.feet,
                normal: Vec3::Y,
            },
            Delivery::Place(place) => {
                let (point, normal) = self.place(living, place);
                Step::Burst { point, normal }
            }
            Delivery::Throw(throw) => Step::Launch(Flight {
                position: living.aim.origin,
                velocity: living.aim.direction * throw.speed,
                bounces_left: throw.bounces,
                travelled: 0.0,
                resting: false,
                throw,
            }),
        }
    }

    fn fly(&mut self, flight: &mut Flight, age: u32, ability: AbilityId) -> Step {
        if age == 0 {
            return Step::Stay;
        }
        let throw = flight.throw;
        if !flight.resting
            && let Some(step) = self.move_flight(flight, ability)
        {
            return step;
        }
        if throw.fuse_ticks.is_some_and(|fuse| age >= fuse) {
            return Step::Burst {
                point: flight.position,
                normal: Vec3::Y,
            };
        }
        Step::Stay
    }

    fn move_flight(&mut self, flight: &mut Flight, ability: AbilityId) -> Option<Step> {
        let throw = flight.throw;
        let mut motion = flight.velocity + Vec3::NEG_Y * (throw.gravity * 0.5);
        let full = motion.length();
        let mut capped = false;
        if let Some(max) = throw.max_travel_m
            && full > 0.0
            && flight.travelled + full >= max
        {
            motion *= greater(max - flight.travelled, 0.0) / full;
            capped = true;
        }
        let length = motion.length();
        let Some(hit) = self
            .world
            .sphere_sweep(flight.position, throw.radius_m, motion)
        else {
            flight.position += motion;
            flight.travelled += length;
            flight.velocity += Vec3::NEG_Y * throw.gravity;
            return capped.then_some(Step::Burst {
                point: flight.position,
                normal: Vec3::Y,
            });
        };
        let along = if length > 0.0 {
            motion / length
        } else {
            Vec3::ZERO
        };
        let contact = flight.position + along * hit.distance;
        flight.travelled += hit.distance;
        flight.position = contact;
        let share = if full > 0.0 { hit.distance / full } else { 0.0 };
        let velocity = flight.velocity + Vec3::NEG_Y * (throw.gravity * share);
        if flight.bounces_left > 0 && hit.bounce > 0.0 {
            flight.bounces_left -= 1;
            flight.velocity = (velocity - hit.normal * (2.0 * velocity.dot(hit.normal)))
                * throw.bounce_keep
                * hit.bounce;
            self.cue(CueKind::Bounce, ability, contact, throw.radius_m, 0);
            None
        } else if throw.fuse_ticks.is_some() {
            flight.resting = true;
            flight.velocity = Vec3::ZERO;
            None
        } else {
            Some(Step::Burst {
                point: contact,
                normal: hit.normal,
            })
        }
    }

    fn zone(&mut self, living: &Living, centre: Vec3, spec: ZoneSpec) {
        let hostile = Modifiers {
            speed: spec.slow.map_or(1.0, |slow| 1.0 - slow),
            sight_m: spec.sight_m,
            ..Modifiers::NONE
        };
        let slows_or_blinds = spec.slow.is_some() || spec.sight_m.is_some();
        for actor in self.actors {
            if actor.distance_to(centre) > spec.radius_m {
                continue;
            }
            let enemy = actor.team != living.caster.team;
            if let Some(amount) = spec.damage_per_tick
                && (enemy || spec.hurts_team)
            {
                self.events.push(Event::Damage {
                    target: actor.id,
                    amount,
                    by: living.caster.id,
                    ability: living.ability,
                });
            }
            if let Some(amount) = spec.heal_per_tick
                && !enemy
            {
                self.events.push(Event::Heal {
                    target: actor.id,
                    amount,
                    by: living.caster.id,
                    ability: living.ability,
                });
            }
            if enemy && slows_or_blinds {
                self.buff(living, actor.id, hostile, self.tick + 1);
            }
            if enemy && spec.reveal {
                self.reveal(living, actor.id, self.tick + 1);
            }
        }
    }

    fn wall_step(&mut self, living: &Living, wall: WallState, age: u32) -> Step {
        if age < wall.spec.ticks {
            return Step::Stay;
        }
        if let Some(id) = wall.solid_id {
            self.world.remove_box(id);
            self.events.push(Event::SolidRemoved {
                effect: living.id,
                id,
            });
        }
        let size = greater(wall.half_extents.x, wall.half_extents.z);
        self.cue(CueKind::WallDown, living.ability, wall.centre, size, 0);
        Step::Finish
    }

    fn trap_step(&mut self, living: &Living, trap: &mut TrapState, age: u32) -> Step {
        if trap.spec.life_ticks.is_some_and(|life| age >= life) {
            return Step::Finish;
        }
        if !trap.armed {
            if age < trap.spec.arm_ticks {
                return Step::Stay;
            }
            trap.armed = true;
            self.events.push(Event::TrapArmed { effect: living.id });
            self.cue(
                CueKind::TrapArmed,
                living.ability,
                trap.a,
                trap.spec.hit_radius_m,
                0,
            );
        }
        let trap = *trap;
        let caught = self.actors.iter().find(|actor| {
            actor.team != living.caster.team
                && match trap.spec.shape {
                    TrapShape::Radius(radius) => actor.distance_to(trap.a) <= radius,
                    TrapShape::Wire { .. } => {
                        segment_distance(trap.a, trap.b, actor.feet, actor.top()) <= actor.radius
                    }
                }
        });
        let Some(actor) = caught.copied() else {
            return Step::Stay;
        };
        self.events.push(Event::TrapTriggered {
            effect: living.id,
            actor: actor.id,
        });
        let slowed = Modifiers {
            speed: 1.0 - trap.spec.slow,
            ..Modifiers::NONE
        };
        self.buff(
            living,
            actor.id,
            slowed,
            self.tick + trap.spec.slow_ticks as u64,
        );
        if let Some(ticks) = trap.spec.reveal_ticks {
            self.reveal(living, actor.id, self.tick + ticks as u64);
        }
        self.cue(
            CueKind::TrapTriggered,
            living.ability,
            actor.feet,
            trap.spec.hit_radius_m,
            trap.spec.slow_ticks,
        );
        Step::Finish
    }

    fn sensor_step(&mut self, living: &Living, sensor: &mut SensorState, age: u32) -> Step {
        if sensor.spec.life_ticks.is_some_and(|life| age >= life) {
            return Step::Finish;
        }
        if !sensor.armed {
            if age < sensor.spec.arm_ticks {
                return Step::Stay;
            }
            sensor.armed = true;
            self.events.push(Event::TrapArmed { effect: living.id });
            self.cue(
                CueKind::TrapArmed,
                living.ability,
                sensor.at,
                sensor.spec.radius_m,
                0,
            );
        }
        let sensor = *sensor;
        for actor in self.actors {
            if actor.team == living.caster.team
                || actor.distance_to(sensor.at) > sensor.spec.radius_m
            {
                continue;
            }
            let target = actor.centre();
            if self.clear_line(sensor.at, target) && !self.sight_blocked(sensor.at, target) {
                self.reveal(
                    living,
                    actor.id,
                    self.tick + sensor.spec.reveal_ticks as u64,
                );
            }
        }
        Step::Stay
    }

    fn glare_step(&mut self, living: &Living, glare: GlareState, age: u32) -> Step {
        if age < glare.windup_ticks {
            return Step::Stay;
        }
        self.flash(
            living,
            glare.origin,
            glare.flash,
            Some((glare.direction, glare.cos_half_angle)),
        );
        self.cue(
            CueKind::Glare,
            living.ability,
            glare.origin,
            glare.flash.range_m,
            glare.flash.max_blind_ticks,
        );
        Step::Finish
    }

    fn heal_step(
        &mut self,
        living: &Living,
        targets: &[ActorId],
        heal_per_tick: f32,
        after: Step,
    ) -> Step {
        if matches!(after, Step::Finish) {
            return after;
        }
        for target in targets {
            if self.present(*target).is_some() {
                self.events.push(Event::Heal {
                    target: *target,
                    amount: heal_per_tick,
                    by: living.caster.id,
                    ability: living.ability,
                });
            }
        }
        Step::Stay
    }

    fn dash_step(&mut self, living: &Living, direction: Vec3, step: f32, after: Step) -> Step {
        if matches!(after, Step::Finish) {
            return after;
        }
        let Some(actor) = self.present(living.caster.id).copied() else {
            return Step::Finish;
        };
        let motion = direction * step;
        let (moved, stopped) = match self
            .world
            .sphere_sweep(actor.centre(), actor.radius, motion)
        {
            Some(hit) => (direction * lesser(hit.distance, step), true),
            None => (motion, false),
        };
        self.events.push(Event::Moved {
            actor: actor.id,
            by: moved,
        });
        if stopped { Step::Finish } else { Step::Stay }
    }

    fn transform_step(&mut self, living: &Living, over: bool) -> Step {
        if !over && self.present(living.caster.id).is_some() {
            return Step::Stay;
        }
        self.events.push(Event::TransformEnded {
            actor: living.caster.id,
        });
        self.cue(
            CueKind::TransformEnd,
            living.ability,
            living.caster.feet,
            living.caster.radius,
            0,
        );
        Step::Finish
    }

    fn act(&mut self, living: &Living, point: Vec3, normal: Vec3) -> State {
        let ability = living.ability;
        match living.action {
            Action::Flash(spec) => {
                self.flash(living, point, spec, None);
                self.cue(
                    CueKind::FlashBurst,
                    ability,
                    point,
                    spec.range_m,
                    spec.max_blind_ticks,
                );
                State::Done
            }
            Action::FlashCone {
                windup_ticks,
                cos_half_angle,
                flash,
            } => State::Glare(GlareState {
                origin: living.caster.eye,
                direction: living.aim.direction,
                windup_ticks,
                cos_half_angle,
                flash,
            }),
            Action::Explosion(falloff) => {
                self.falloff(living, point, falloff);
                self.cue(CueKind::Explosion, ability, point, falloff.radius_m, 0);
                State::Done
            }
            Action::Strike {
                warn_ticks,
                falloff,
            } => {
                self.cue(
                    CueKind::StrikeWarning,
                    ability,
                    point,
                    falloff.radius_m,
                    warn_ticks,
                );
                State::Strike {
                    centre: point,
                    warn_ticks,
                    falloff,
                }
            }
            Action::Smoke(spec) => {
                self.cue(
                    CueKind::SmokeBurst,
                    ability,
                    point,
                    spec.radius_m,
                    spec.ticks,
                );
                State::Smoke {
                    centre: point,
                    spec,
                }
            }
            Action::Zone(spec) => {
                self.cue(
                    CueKind::ZoneStart,
                    ability,
                    point,
                    spec.radius_m,
                    spec.ticks,
                );
                State::Zone {
                    centre: point,
                    spec,
                }
            }
            Action::Wall(spec) => self.wall(living, point, spec),
            Action::Trap(spec) => self.trap(living, point, normal, spec),
            Action::Sensor(spec) => self.sensor(living, point, normal, spec),
            Action::Pulse {
                radius_m,
                reveal_ticks,
            } => self.pulse(living, point, radius_m, reveal_ticks),
            Action::TeamHeal {
                radius_m,
                heal_per_tick,
                ticks,
                overshield,
            } => self.team_heal(living, point, radius_m, overshield, (heal_per_tick, ticks)),
            Action::Dash { distance_m, ticks } => self.dash(living, distance_m, ticks),
            Action::Transform {
                ticks,
                modifiers,
                overshield,
                refresh_signature_on_kill,
            } => self.transform(
                living,
                ticks,
                modifiers,
                overshield,
                refresh_signature_on_kill,
            ),
        }
    }

    fn sensor(&mut self, living: &Living, point: Vec3, normal: Vec3, spec: SensorSpec) -> State {
        self.cue(
            CueKind::SensorPlaced,
            living.ability,
            point,
            spec.radius_m,
            spec.arm_ticks,
        );
        State::Sensor(SensorState {
            at: point,
            normal,
            spec,
            health: spec.health,
            armed: false,
        })
    }

    fn dash(&mut self, living: &Living, distance_m: f32, ticks: u32) -> State {
        self.cue(
            CueKind::Dash,
            living.ability,
            living.caster.feet,
            living.caster.radius,
            ticks,
        );
        let ticks = ticks.max(1);
        match flat(living.aim.direction) {
            Some(direction) => State::Dash {
                direction,
                step: distance_m / ticks as f32,
                ticks,
            },
            None => State::Done,
        }
    }

    fn trap(&mut self, living: &Living, point: Vec3, normal: Vec3, spec: TrapSpec) -> State {
        let (b, normal_b) = match spec.shape {
            TrapShape::Radius(_) => (point, normal),
            TrapShape::Wire { max_length_m } => {
                match flat(normal).or_else(|| flat(-living.aim.direction)) {
                    Some(across) => match self.world.ray(point, across, max_length_m) {
                        Some(hit) => (point + across * hit.distance, hit.normal),
                        None => (point + across * max_length_m, -across),
                    },
                    None => (point, normal),
                }
            }
        };
        self.cue(
            CueKind::TrapPlaced,
            living.ability,
            point,
            spec.hit_radius_m,
            spec.arm_ticks,
        );
        State::Trap(TrapState {
            a: point,
            b,
            normal_a: normal,
            normal_b,
            spec,
            health: spec.health,
            armed: false,
        })
    }

    fn pulse(&mut self, living: &Living, point: Vec3, radius_m: f32, reveal_ticks: u32) -> State {
        for actor in self.actors {
            if actor.team != living.caster.team && actor.distance_to(point) <= radius_m {
                self.reveal(living, actor.id, self.tick + reveal_ticks as u64);
            }
        }
        self.cue(
            CueKind::Pulse,
            living.ability,
            point,
            radius_m,
            reveal_ticks,
        );
        State::Done
    }

    fn team_heal(
        &mut self,
        living: &Living,
        point: Vec3,
        radius_m: f32,
        overshield: Option<f32>,
        (heal_per_tick, ticks): (f32, u32),
    ) -> State {
        let targets: Vec<ActorId> = self
            .actors
            .iter()
            .filter(|actor| {
                actor.team == living.caster.team && actor.distance_to(point) <= radius_m
            })
            .map(|actor| actor.id)
            .collect();
        if let Some(amount) = overshield {
            for target in &targets {
                self.events.push(Event::Shield {
                    target: *target,
                    amount,
                    until: None,
                    by: living.caster.id,
                    ability: living.ability,
                });
            }
        }
        self.cue(CueKind::HealStart, living.ability, point, radius_m, ticks);
        State::Heal {
            targets,
            heal_per_tick,
            ticks,
        }
    }

    fn transform(
        &mut self,
        living: &Living,
        ticks: u32,
        modifiers: Modifiers,
        overshield: Option<f32>,
        refresh_signature_on_kill: bool,
    ) -> State {
        let caster = living.caster;
        let until = self.tick + ticks as u64;
        self.buff(living, caster.id, modifiers, until);
        self.events.push(Event::Transformed {
            actor: caster.id,
            until,
            refresh_signature_on_kill,
            ability: living.ability,
        });
        if let Some(amount) = overshield {
            self.events.push(Event::Shield {
                target: caster.id,
                amount,
                until: Some(until),
                by: caster.id,
                ability: living.ability,
            });
        }
        self.cue(
            CueKind::TransformStart,
            living.ability,
            caster.feet,
            caster.radius,
            ticks,
        );
        State::Transform {
            ticks,
            refresh_signature_on_kill,
        }
    }

    fn wall(&mut self, living: &Living, point: Vec3, spec: WallSpec) -> State {
        let forward = flat(living.aim.direction).or_else(|| flat(living.caster.view));
        let axes = forward.map_or(Mat3::IDENTITY, frame);
        let up = Vec3::Y * (spec.height_m * 0.5);
        let (centre, half_extents) = match spec.lay {
            Lay::AlongAim => (
                point + axes.z_axis * (spec.length_m * 0.5) + up,
                Vec3::new(
                    spec.thickness_m * 0.5,
                    spec.height_m * 0.5,
                    spec.length_m * 0.5,
                ),
            ),
            Lay::AcrossAim => (
                point + up,
                Vec3::new(
                    spec.length_m * 0.5,
                    spec.height_m * 0.5,
                    spec.thickness_m * 0.5,
                ),
            ),
        };
        let solid_id = spec.solid.then(|| self.solid_id_base + living.id.0);
        if let Some(id) = solid_id {
            self.world
                .add_box(id, living.ability, centre, half_extents, axes);
            self.events.push(Event::SolidAdded {
                effect: living.id,
                id,
                centre,
                half_extents,
                axes,
            });
        }
        self.cue(
            CueKind::WallUp,
            living.ability,
            centre,
            greater(half_extents.x, half_extents.z),
            spec.ticks,
        );
        State::Wall(WallState {
            centre,
            half_extents,
            axes,
            spec,
            health: spec.health,
            solid_id,
        })
    }

    fn expire(&mut self) {
        let tick = self.tick;
        let actors = self.actors;
        let alive = |actor: ActorId| actors.iter().any(|other| other.id == actor);
        let mut ended = Vec::new();
        self.marks.buffs.retain(|buff| {
            let keep = buff.until > tick && alive(buff.actor);
            if !keep {
                ended.push(Event::BuffEnded {
                    actor: buff.actor,
                    effect: buff.effect,
                });
            }
            keep
        });
        self.marks
            .blinds
            .retain(|blind| blind.start + blind.ticks as u64 > tick && alive(blind.actor));
        self.marks.reveals.retain(|(team, actor), until| {
            let keep = *until > tick && alive(*actor);
            if !keep {
                ended.push(Event::RevealEnded {
                    actor: *actor,
                    team: *team,
                });
            }
            keep
        });
        self.events.append(&mut ended);
    }
}

fn finish_at(age: u32, ticks: u32) -> Step {
    if age >= ticks {
        Step::Finish
    } else {
        Step::Stay
    }
}
