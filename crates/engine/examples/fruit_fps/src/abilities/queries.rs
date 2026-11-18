use glam::{Mat3, Vec3};

use super::effects::{ActorId, EffectId, Effects, Living, State};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusKind {
    Chained,
    Slowed,
    SightCut,
    Vulnerable,
    Revealed,
    Boosted,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Status {
    pub kind: StatusKind,
    pub ability: AbilityId,
    pub effect: Option<EffectId>,
    pub to_team: Option<usize>,
    pub until: u64,
    pub ticks_left: u64,
    pub strength: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WireChain {
    pub wire: EffectId,
    pub ability: AbilityId,
    pub since: u64,
    pub until: u64,
    pub slow: f32,
}
use super::numbers::{AbilityId, Modifiers, TrapShape};
use super::shapes::{ray_box, ray_sphere};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transformation {
    pub ability: AbilityId,
    pub until: u64,
    pub refresh_signature_on_kill: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DangerKind {
    Damage,
    Strike,
    Slow,
    NearSight,
    Reveal,
    Smoke,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Danger {
    pub kind: DangerKind,
    pub centre: Vec3,
    pub radius: f32,
    pub weight: f32,
    pub until: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawKind {
    Projectile,
    Smoke,
    Zone,
    Curtain,
    Wall,
    Trap,
    Wire,
    Sensor,
    StrikeMark,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DrawShape {
    Sphere {
        centre: Vec3,
        radius: f32,
    },
    Box {
        centre: Vec3,
        half_extents: Vec3,
        axes: Mat3,
    },
    Segment {
        a: Vec3,
        b: Vec3,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drawable {
    pub effect: EffectId,
    pub kind: DrawKind,
    pub ability: AbilityId,
    pub owner: ActorId,
    pub team: usize,
    pub shape: DrawShape,
    pub age: u32,
    pub lifetime: Option<u32>,
    pub strength: f32,
    pub armed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShootableHit {
    pub effect: EffectId,
    pub distance: f32,
    pub team: usize,
    pub wall: bool,
}

impl ShootableHit {
    pub fn shootable_by(&self, team: usize) -> bool {
        self.wall || self.team != team
    }
}

fn nearer(best: Option<f32>, candidate: Option<f32>) -> Option<f32> {
    match (best, candidate) {
        (Some(a), Some(b)) => Some(if b < a { b } else { a }),
        (a, None) => a,
        (None, b) => b,
    }
}

fn smoke_strength(age: u32, ticks: u32, grow_ticks: u32, fade_ticks: u32) -> f32 {
    let mut strength = 1.0f32;
    if grow_ticks > 0 && age < grow_ticks {
        strength = (age + 1) as f32 / grow_ticks as f32;
    }
    let left = ticks.saturating_sub(age);
    if fade_ticks > 0 && left < fade_ticks {
        let fading = left as f32 / fade_ticks as f32;
        if fading < strength {
            strength = fading;
        }
    }
    strength
}

impl Effects {
    pub fn sight_blocked(&self, a: Vec3, b: Vec3) -> bool {
        self.livings
            .iter()
            .any(|living| living.blocks_sight(self.now, a, b))
    }

    pub fn blindness(&self, actor: ActorId) -> f32 {
        self.blindness_at(actor, self.now as f64)
    }

    pub fn blindness_at(&self, actor: ActorId, tick: f64) -> f32 {
        self.marks.blinds.get(&actor).map_or(0.0, |blinds| {
            blinds
                .iter()
                .map(|blind| blind.value(tick))
                .fold(0.0, |most, value| if value > most { value } else { most })
        })
    }

    pub fn sight_limit(&self, actor: ActorId) -> Option<f32> {
        self.modifiers(actor).sight_m
    }

    pub fn modifiers(&self, actor: ActorId) -> Modifiers {
        self.marks
            .buffs
            .range((actor, EffectId(0))..=(actor, EffectId(u32::MAX)))
            .fold(Modifiers::NONE, |all, (_, buff)| {
                all.combine(buff.modifiers)
            })
    }

    pub fn wire_chain(&self, actor: ActorId) -> Option<WireChain> {
        self.livings.iter().find_map(|living| {
            let State::Trap(trap) = &living.state else {
                return None;
            };
            let (held, since) = trap.chained?;
            let chain = trap.spec.chain?;
            (held == actor).then_some(WireChain {
                wire: living.id,
                ability: living.ability,
                since,
                until: since + u64::from(chain.windup_ticks),
                slow: chain.slow,
            })
        })
    }

    pub fn statuses(&self, actor: ActorId) -> Vec<Status> {
        let now = self.now;
        let status = |kind, ability, effect, until: u64, strength| Status {
            kind,
            ability,
            effect,
            to_team: None,
            until,
            ticks_left: until.saturating_sub(now),
            strength,
        };
        let chain = self.wire_chain(actor);
        let mut out: Vec<Status> = chain
            .iter()
            .map(|chain| {
                status(
                    StatusKind::Chained,
                    chain.ability,
                    Some(chain.wire),
                    chain.until,
                    Some(chain.slow),
                )
            })
            .collect();
        let buffs = self
            .marks
            .buffs
            .range((actor, EffectId(0))..=(actor, EffectId(u32::MAX)))
            .map(|(_, buff)| buff)
            .filter(|buff| chain.is_none_or(|chain| chain.wire != buff.effect));
        for buff in buffs {
            let made = |kind, strength| {
                status(
                    kind,
                    buff.ability,
                    Some(buff.effect),
                    buff.until,
                    Some(strength),
                )
            };
            let m = buff.modifiers;
            if m.speed < 1.0 {
                out.push(made(StatusKind::Slowed, 1.0 - m.speed));
            }
            if let Some(sight) = m.sight_m {
                out.push(made(StatusKind::SightCut, sight));
            }
            if m.damage_taken > 1.0 {
                out.push(made(StatusKind::Vulnerable, m.damage_taken - 1.0));
            }
            let boost = [m.fire_rate, m.reload, m.speed]
                .into_iter()
                .fold(1.0, |best, value| if value > best { value } else { best });
            if boost > 1.0 {
                out.push(made(StatusKind::Boosted, boost - 1.0));
            }
        }
        for (&(team, _), &(until, ability)) in self
            .marks
            .reveals
            .iter()
            .filter(|((_, revealed), _)| *revealed == actor)
        {
            out.push(Status {
                to_team: Some(team),
                ..status(StatusKind::Revealed, ability, None, until, None)
            });
        }
        out
    }

    pub fn revealed(&self, actor: ActorId, team: usize) -> bool {
        self.marks.reveals.contains_key(&(team, actor))
    }

    pub fn reveals(&self) -> Vec<(usize, ActorId, u64)> {
        self.marks
            .reveals
            .iter()
            .map(|(&(team, actor), &(until, _))| (team, actor, until))
            .collect()
    }

    pub fn transformation(&self, actor: ActorId) -> Option<Transformation> {
        self.livings.iter().find_map(|living| {
            let State::Transform {
                ticks,
                refresh_signature_on_kill,
            } = living.state
            else {
                return None;
            };
            (living.caster.id == actor).then_some(Transformation {
                ability: living.ability,
                until: living.born + ticks as u64,
                refresh_signature_on_kill,
            })
        })
    }

    pub fn dangers(&self, team: usize) -> Vec<Danger> {
        let mut dangers = Vec::new();
        for living in &self.livings {
            let enemy = living.caster.team != team;
            match living.state {
                State::Zone { centre, spec } => {
                    let until = living.born + spec.ticks as u64;
                    let mut add = |kind, weight| {
                        dangers.push(Danger {
                            kind,
                            centre,
                            radius: spec.radius_m,
                            weight,
                            until,
                        })
                    };
                    if let Some(per_tick) = spec.damage_per_tick
                        && (enemy || spec.hurts_team)
                    {
                        add(DangerKind::Damage, per_tick);
                    }
                    if enemy {
                        if let Some(slow) = spec.slow {
                            add(DangerKind::Slow, slow);
                        }
                        if let Some(sight) = spec.sight_m {
                            add(DangerKind::NearSight, sight);
                        }
                        if spec.reveal {
                            add(DangerKind::Reveal, 0.0);
                        }
                    }
                }
                State::Strike {
                    centre,
                    warn_ticks,
                    falloff,
                } if enemy || falloff.hurts_team => dangers.push(Danger {
                    kind: DangerKind::Strike,
                    centre,
                    radius: falloff.radius_m,
                    weight: falloff.max,
                    until: living.born + warn_ticks as u64,
                }),
                State::Smoke { centre, spec } => dangers.push(Danger {
                    kind: DangerKind::Smoke,
                    centre,
                    radius: Living::smoke_radius(&spec, living.age(self.now)),
                    weight: spec.radius_m,
                    until: living.born + spec.ticks as u64,
                }),
                State::Strike { .. }
                | State::Pending
                | State::Flight(_)
                | State::Wall(_)
                | State::Trap(_)
                | State::Sensor(_)
                | State::Glare(_)
                | State::Heal { .. }
                | State::Dash { .. }
                | State::Transform { .. }
                | State::Done => {}
            }
        }
        dangers
    }

    pub fn drawables(&self) -> Vec<Drawable> {
        self.livings
            .iter()
            .filter_map(|living| self.drawable(living))
            .collect()
    }

    fn drawable(&self, living: &Living) -> Option<Drawable> {
        let age = living.age(self.now);
        let (kind, shape, lifetime, strength, armed) = match &living.state {
            State::Flight(flight) => (
                DrawKind::Projectile,
                DrawShape::Sphere {
                    centre: flight.position,
                    radius: flight.throw.radius_m,
                },
                flight.throw.fuse_ticks,
                1.0,
                false,
            ),
            State::Smoke { centre, spec } => (
                DrawKind::Smoke,
                DrawShape::Sphere {
                    centre: *centre,
                    radius: Living::smoke_radius(spec, age),
                },
                Some(spec.ticks),
                smoke_strength(age, spec.ticks, spec.grow_ticks, spec.fade_ticks),
                false,
            ),
            State::Zone { centre, spec } => (
                DrawKind::Zone,
                DrawShape::Sphere {
                    centre: *centre,
                    radius: spec.radius_m,
                },
                Some(spec.ticks),
                1.0,
                false,
            ),
            State::Wall(wall) => (
                if wall.spec.solid {
                    DrawKind::Wall
                } else {
                    DrawKind::Curtain
                },
                DrawShape::Box {
                    centre: wall.centre,
                    half_extents: wall.half_extents,
                    axes: wall.axes,
                },
                Some(wall.spec.ticks),
                match (wall.health, wall.spec.health) {
                    (Some(now), Some(full)) if full > 0.0 => now / full,
                    _ => 1.0,
                },
                false,
            ),
            State::Trap(trap) => match trap.spec.shape {
                TrapShape::Radius(radius) => (
                    DrawKind::Trap,
                    DrawShape::Sphere {
                        centre: trap.a,
                        radius,
                    },
                    trap.spec.life_ticks,
                    1.0,
                    trap.armed,
                ),
                TrapShape::Wire { .. } => (
                    DrawKind::Wire,
                    DrawShape::Segment {
                        a: trap.a,
                        b: trap.b,
                    },
                    trap.spec.life_ticks,
                    1.0,
                    trap.armed,
                ),
            },
            State::Sensor(sensor) => (
                DrawKind::Sensor,
                DrawShape::Sphere {
                    centre: sensor.at,
                    radius: sensor.spec.radius_m,
                },
                sensor.spec.life_ticks,
                1.0,
                sensor.armed,
            ),
            State::Strike {
                centre,
                warn_ticks,
                falloff,
            } => (
                DrawKind::StrikeMark,
                DrawShape::Sphere {
                    centre: *centre,
                    radius: falloff.radius_m,
                },
                Some(*warn_ticks),
                1.0,
                false,
            ),
            State::Pending
            | State::Glare(_)
            | State::Heal { .. }
            | State::Dash { .. }
            | State::Transform { .. }
            | State::Done => return None,
        };
        Some(Drawable {
            effect: living.id,
            kind,
            ability: living.ability,
            owner: living.caster.id,
            team: living.caster.team,
            shape,
            age,
            lifetime,
            strength,
            armed,
        })
    }

    pub fn shootable_ray(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<ShootableHit> {
        let mut best: Option<ShootableHit> = None;
        for living in &self.livings {
            let distance = match &living.state {
                State::Wall(wall) if wall.health.is_some() => ray_box(
                    origin,
                    direction,
                    max,
                    wall.centre,
                    wall.half_extents,
                    wall.axes,
                ),
                State::Trap(trap) => {
                    let near_a = ray_sphere(origin, direction, max, trap.a, trap.spec.hit_radius_m);
                    match trap.spec.shape {
                        TrapShape::Radius(_) => near_a,
                        TrapShape::Wire { .. } => nearer(
                            near_a,
                            ray_sphere(origin, direction, max, trap.b, trap.spec.hit_radius_m),
                        ),
                    }
                }
                State::Sensor(sensor) => {
                    ray_sphere(origin, direction, max, sensor.at, sensor.spec.hit_radius_m)
                }
                State::Wall(_)
                | State::Pending
                | State::Flight(_)
                | State::Smoke { .. }
                | State::Zone { .. }
                | State::Strike { .. }
                | State::Glare(_)
                | State::Heal { .. }
                | State::Dash { .. }
                | State::Transform { .. }
                | State::Done => None,
            };
            if let Some(distance) = distance
                && best.is_none_or(|hit| distance < hit.distance)
            {
                best = Some(ShootableHit {
                    effect: living.id,
                    distance,
                    team: living.caster.team,
                    wall: matches!(living.state, State::Wall(_)),
                });
            }
        }
        best
    }

    pub fn effect_of_solid(&self, id: u32) -> Option<EffectId> {
        self.livings.iter().find_map(|living| {
            let State::Wall(wall) = living.state else {
                return None;
            };
            (wall.solid_id == Some(id)).then_some(living.id)
        })
    }
}
