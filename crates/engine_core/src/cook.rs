use std::any::Any;
use std::sync::Arc;

use bevy_ecs::change_detection::Mut;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_registry::{BrickStates, JobBatch, JobOrder, JobQueue, MembersOf};

use crate::collections::HashSet;
use crate::jobs::RecordSet;
use crate::{
    Interest, JobExchange, JobKind, JobRef, List, RecordKey, Records, Request, VIEW_JOBS,
    job_priority,
};

pub struct CookList;

impl List for CookList {
    const NAME: &'static str = "cooks";
    type Member = Cook;
}

#[derive(Clone, Copy)]
pub struct Cook(fn(&World, &mut CookAsks<'_>));

impl Cook {
    pub const fn new(list: fn(&World, &mut CookAsks<'_>)) -> Self {
        Self(list)
    }
}

struct Ask {
    key: RecordKey,
    kind: &'static JobKind,
    input: Vec<u8>,
    reads: Vec<RecordKey>,
}

pub struct CookAsks<'c> {
    known: &'c HashSet<RecordKey>,
    fresh: HashSet<RecordKey>,
    asks: Vec<Ask>,
    holds: Vec<Box<dyn Any + Send + Sync>>,
    left: u64,
}

impl CookAsks<'_> {
    pub fn asked(&self, key: &RecordKey) -> bool {
        self.known.contains(key) || self.fresh.contains(key)
    }

    pub fn ask(&mut self, kind: &'static JobKind, input: Vec<u8>) {
        self.ask_reading(kind, input, Vec::new());
    }

    pub fn ask_reading(&mut self, kind: &'static JobKind, input: Vec<u8>, reads: Vec<RecordKey>) {
        let key = RecordKey::of(kind, &input);
        self.push(Ask {
            key,
            kind,
            input,
            reads,
        });
    }

    pub fn ask_as(&mut self, key: RecordKey, kind: &'static JobKind, input: Vec<u8>) {
        self.push(Ask {
            key,
            kind,
            input,
            reads: Vec::new(),
        });
    }

    pub fn hold(&mut self, held: impl Any + Send + Sync) {
        self.holds.push(Box::new(held));
    }

    pub fn left(&mut self, left: u64) {
        self.left = self.left.saturating_add(left);
    }

    fn push(&mut self, ask: Ask) {
        if !self.asked(&ask.key) {
            self.fresh.insert(ask.key);
            self.asks.push(ask);
        }
    }
}

struct Asked {
    ask: Ask,
    interest: Interest<RecordKey>,
    started: bool,
    reading: bool,
}

#[derive(Resource, Default)]
pub(crate) struct Cooking {
    known: HashSet<RecordKey>,
    asked: Vec<Asked>,
    holds: Vec<Box<dyn Any + Send + Sync>>,
    left: u64,
    runs: u64,
}

impl Cooking {
    pub(crate) fn left(&self) -> u64 {
        self.left
    }

    fn step(
        &mut self,
        fresh: Vec<Ask>,
        records: &mut RecordSet,
        exchange: &mut JobExchange,
    ) -> Vec<JobOrder> {
        let Self { known, asked, .. } = self;
        for ask in fresh {
            known.insert(ask.key);
            let interest = records.ask(ask.key);
            asked.push(Asked {
                ask,
                interest,
                started: false,
                reading: false,
            });
        }
        let held: HashSet<RecordKey> = asked.iter().map(|each| *each.interest.key()).collect();
        for each in asked.iter() {
            for read in &each.ask.reads {
                if !held.contains(read) {
                    known.remove(read);
                }
            }
        }
        let mut orders = Vec::new();
        for each in asked.iter_mut() {
            let key = *each.interest.key();
            match records.get(&key) {
                Some(Request::Asked(_)) if !each.started => {
                    let reads: Option<Vec<Arc<[u8]>>> = each
                        .ask
                        .reads
                        .iter()
                        .map(|read| records.get(read).and_then(Request::ready).cloned())
                        .collect();
                    let Some(reads) = reads else {
                        continue;
                    };
                    for (read, bytes) in each.ask.reads.iter().zip(reads) {
                        exchange.hold_read(*read, bytes);
                    }
                    each.reading = !each.ask.reads.is_empty();
                    each.started = true;
                    orders.push(order(&each.ask, orders.len()));
                }
                Some(Request::Missing) => {
                    release(each, exchange);
                    if records.ask_again(&key).is_some() {
                        each.started = false;
                    }
                }
                Some(Request::Ready(_) | Request::Failed(_)) => release(each, exchange),
                Some(Request::Asked(_)) | None => {}
            }
        }
        let reading: HashSet<RecordKey> = asked
            .iter()
            .filter(|each| {
                records
                    .get(each.interest.key())
                    .is_none_or(|request| matches!(request, Request::Asked(_) | Request::Missing))
            })
            .flat_map(|each| each.ask.reads.iter().copied())
            .collect();
        asked.retain(|each| {
            let key = each.interest.key();
            reading.contains(key) || records.get(key).and_then(Request::ready).is_none()
        });
        orders
    }
}

fn release(asked: &mut Asked, exchange: &mut JobExchange) {
    if asked.reading {
        for read in &asked.ask.reads {
            exchange.release_read(*read);
        }
        asked.reading = false;
    }
}

fn order(ask: &Ask, place: usize) -> JobOrder {
    JobOrder::Start {
        key: ask.key.hash(),
        kind: JobRef::from(ask.kind),
        input: ask.input.clone(),
        priority: job_priority(VIEW_JOBS, u32::try_from(place).unwrap_or(u32::MAX)),
        admitted: false,
        cause: engine_profile::carry(),
    }
}

pub fn cook(world: &mut World) {
    let members: Vec<Cook> = if let (Some(members), Some(states)) = (
        world.get_resource::<MembersOf<CookList>>(),
        world.get_resource::<BrickStates>(),
    ) {
        members.live(states).copied().collect()
    } else {
        Vec::new()
    };
    let mut cooking = world.remove_resource::<Cooking>().unwrap_or_default();
    let mut asks = CookAsks {
        known: &cooking.known,
        fresh: HashSet::default(),
        asks: Vec::new(),
        holds: Vec::new(),
        left: 0,
    };
    for member in members {
        (member.0)(world, &mut asks);
    }
    let CookAsks {
        asks: fresh,
        holds,
        left,
        ..
    } = asks;
    cooking.holds = holds;
    cooking.left = left;
    cooking.runs = cooking.runs.wrapping_add(1);
    let batch = JobBatch {
        owner: 0,
        run: cooking.runs,
    };
    let mut orders = Vec::new();
    world.resource_scope(|world, mut exchange: Mut<JobExchange>| {
        crate::modify(world, |records: &mut Records| {
            orders = cooking.step(fresh, &mut records.requests, &mut exchange);
        });
    });
    if !orders.is_empty() {
        world
            .get_resource_or_init::<JobQueue>()
            .orders
            .extend(orders.into_iter().map(|order| (batch, order)));
    }
    world.insert_resource(cooking);
}
