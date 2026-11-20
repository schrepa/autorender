use std::collections::{BTreeMap, BTreeSet};

use glam::Vec3;
use serde::Deserialize;
use serde_json::Value;

use super::effects::{Actor, ActorId};

#[derive(Clone, Debug, PartialEq)]
pub struct OrbNumbers {
    pub rules: OrbRules,
    pub spots: Vec<String>,
}

#[derive(Deserialize)]
struct Root {
    ultimate_orbs: RawOrbs,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOrbs {
    spots: Vec<String>,
    channel_seconds: f32,
    reach_m: f32,
}

impl OrbNumbers {
    pub fn parse(whole: &Value, ticks_per_second: u32) -> Result<OrbNumbers, String> {
        let raw = Root::deserialize(whole)
            .map_err(|error| format!("ultimate_orbs: {error}"))?
            .ultimate_orbs;
        Ok(OrbNumbers {
            rules: OrbRules {
                channel_ticks: engine_math::ticks(raw.channel_seconds, ticks_per_second),
                reach_m: raw.reach_m,
            },
            spots: raw.spots,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbRules {
    pub channel_ticks: u32,
    pub reach_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrbState {
    Waiting,
    Taken { by: ActorId, tick: u64 },
    Gone,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Channel {
    pub actor: ActorId,
    pub since: u64,
    pub done: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Orb {
    pub name: String,
    pub at: Vec3,
    pub state: OrbState,
    pub channels: Vec<Channel>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    Released,
    Moved,
    Died,
    TakenByOther,
    Gone,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrbEvent {
    Started {
        orb: usize,
        actor: ActorId,
    },
    Stopped {
        orb: usize,
        actor: ActorId,
        why: Stop,
    },
    Taken {
        orb: usize,
        actor: ActorId,
    },
    Vanished {
        orb: usize,
    },
    Restored {
        orb: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Taker {
    pub actor: Actor,
    pub holding: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Orbs {
    pub rules: Option<OrbRules>,
    pub orbs: Vec<Orb>,
}

impl Orbs {
    pub fn place(&mut self, rules: OrbRules, spots: Vec<(String, Vec3)>) {
        self.rules = Some(rules);
        self.orbs = spots
            .into_iter()
            .map(|(name, at)| Orb {
                name,
                at,
                state: OrbState::Waiting,
                channels: Vec::new(),
            })
            .collect();
    }

    pub fn restore(&mut self) -> Vec<OrbEvent> {
        let mut events = Vec::new();
        for (index, orb) in self.orbs.iter_mut().enumerate() {
            for channel in orb.channels.drain(..) {
                events.push(OrbEvent::Stopped {
                    orb: index,
                    actor: channel.actor,
                    why: Stop::Gone,
                });
            }
            if orb.state != OrbState::Waiting {
                orb.state = OrbState::Waiting;
                events.push(OrbEvent::Restored { orb: index });
            }
        }
        events
    }

    pub fn vanish(&mut self) -> Vec<OrbEvent> {
        let mut events = Vec::new();
        for (index, orb) in self.orbs.iter_mut().enumerate() {
            if orb.state != OrbState::Waiting {
                continue;
            }
            for channel in orb.channels.drain(..) {
                events.push(OrbEvent::Stopped {
                    orb: index,
                    actor: channel.actor,
                    why: Stop::Gone,
                });
            }
            orb.state = OrbState::Gone;
            events.push(OrbEvent::Vanished { orb: index });
        }
        events
    }

    pub fn tick(&mut self, takers: &[Taker], tick: u64) -> Vec<OrbEvent> {
        let mut events = Vec::new();
        let Some(rules) = self.rules else {
            return events;
        };
        let by_id: BTreeMap<ActorId, &Taker> =
            takers.iter().map(|taker| (taker.actor.id, taker)).collect();
        for index in 0..self.orbs.len() {
            self.keep_channels(index, &by_id, rules, &mut events);
            self.finish(index, tick, &mut events);
        }
        let channelling: BTreeSet<ActorId> = self
            .orbs
            .iter()
            .flat_map(|orb| orb.channels.iter().map(|channel| channel.actor))
            .collect();
        for taker in takers {
            if !taker.holding || channelling.contains(&taker.actor.id) {
                continue;
            }
            let Some(index) = self.orbs.iter().position(|orb| {
                orb.state == OrbState::Waiting && taker.actor.distance_to(orb.at) <= rules.reach_m
            }) else {
                continue;
            };
            self.orbs[index].channels.push(Channel {
                actor: taker.actor.id,
                since: tick,
                done: tick + u64::from(rules.channel_ticks),
            });
            events.push(OrbEvent::Started {
                orb: index,
                actor: taker.actor.id,
            });
        }
        events
    }

    fn keep_channels(
        &mut self,
        index: usize,
        takers: &BTreeMap<ActorId, &Taker>,
        rules: OrbRules,
        events: &mut Vec<OrbEvent>,
    ) {
        let orb = &mut self.orbs[index];
        let at = orb.at;
        orb.channels.retain(|channel| {
            let why = match takers.get(&channel.actor) {
                None => Some(Stop::Died),
                Some(taker) if !taker.holding => Some(Stop::Released),
                Some(taker) if taker.actor.distance_to(at) > rules.reach_m => Some(Stop::Moved),
                Some(_) => None,
            };
            match why {
                Some(why) => {
                    events.push(OrbEvent::Stopped {
                        orb: index,
                        actor: channel.actor,
                        why,
                    });
                    false
                }
                None => true,
            }
        });
    }

    fn finish(&mut self, index: usize, tick: u64, events: &mut Vec<OrbEvent>) {
        let orb = &mut self.orbs[index];
        let Some(winner) = orb
            .channels
            .iter()
            .filter(|channel| tick >= channel.done)
            .min_by_key(|channel| (channel.since, channel.actor))
            .copied()
        else {
            return;
        };
        orb.state = OrbState::Taken {
            by: winner.actor,
            tick,
        };
        events.push(OrbEvent::Taken {
            orb: index,
            actor: winner.actor,
        });
        for channel in orb.channels.drain(..) {
            if channel.actor != winner.actor {
                events.push(OrbEvent::Stopped {
                    orb: index,
                    actor: channel.actor,
                    why: Stop::TakenByOther,
                });
            }
        }
    }

    fn channel_of(&self, actor: ActorId) -> Option<(usize, Channel)> {
        self.orbs.iter().enumerate().find_map(|(index, orb)| {
            orb.channels
                .iter()
                .find(|channel| channel.actor == actor)
                .map(|channel| (index, *channel))
        })
    }

    pub fn progress(&self, actor: ActorId, tick: u64) -> Option<(usize, f32)> {
        let (index, channel) = self.channel_of(actor)?;
        let span = channel.done.saturating_sub(channel.since);
        let share = if span == 0 {
            1.0
        } else {
            tick.saturating_sub(channel.since) as f32 / span as f32
        };
        Some((index, if share > 1.0 { 1.0 } else { share }))
    }
}
