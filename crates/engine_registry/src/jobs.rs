use std::any::{Any, TypeId};
use std::sync::Arc;

use bevy_ecs::resource::Resource;
use bevy_ecs::system::{SystemBuffer, SystemMeta};
use bevy_ecs::world::DeferredWorld;
use engine_math::collections::HashMap;
use engine_place::{Error, ErrorCode, JobRef, List};
use sha2::{Digest, Sha256};

pub type JobKindRef = &'static (dyn Any + Send + Sync);

pub type JobMember = (TypeId, Box<dyn Any + Send + Sync>);

#[derive(Clone, Copy, Debug)]
pub enum RecordParity {
    Exact,
    Within(fn(&[u8], &[u8]) -> bool),
}

impl RecordParity {
    pub fn holds(self, handed: &[u8], made: &[u8]) -> bool {
        match self {
            Self::Exact => handed == made,
            Self::Within(alike) => alike(handed, made),
        }
    }
}

#[derive(Default)]
pub struct JobList {
    pub names: Vec<&'static str>,
    sources: Vec<[u8; 32]>,
    kinds: Vec<JobKindRef>,
    members: Vec<JobMember>,
    hash: [u8; 32],
}

impl JobList {
    pub fn new(
        kinds: impl IntoIterator<Item = (&'static str, [u8; 32], JobKindRef)>,
    ) -> Result<Self, Error> {
        let mut listed: Vec<(&'static str, [u8; 32], JobKindRef)> = kinds.into_iter().collect();
        listed.sort_by(|a, b| a.0.cmp(b.0));
        if let Some(pair) = listed.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(
                Error::new(ErrorCode::NameTwice, format!("$.jobs.{}", pair[1].0)).got(pair[0].0),
            );
        }
        let mut hash = Sha256::new();
        for &(name, source, _) in &listed {
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            hash.update(source);
        }
        Ok(Self {
            names: listed.iter().map(|&(name, _, _)| name).collect(),
            sources: listed.iter().map(|&(_, source, _)| source).collect(),
            kinds: listed.iter().map(|&(_, _, kind)| kind).collect(),
            members: Vec::new(),
            hash: hash.finalize().into(),
        })
    }

    #[must_use]
    pub fn with_members(mut self, members: Vec<JobMember>) -> Self {
        self.members = members;
        self
    }

    pub fn find(&self, name: &str, source: &[u8; 32]) -> Option<JobKindRef> {
        let index = self.names.binary_search_by(|probe| probe.cmp(&name)).ok()?;
        (self.sources[index] == *source).then_some(self.kinds[index])
    }

    pub fn members<L: List>(&self) -> impl Iterator<Item = &L::Member> + '_ {
        let list = TypeId::of::<L>();
        self.members
            .iter()
            .filter(move |(of, _)| *of == list)
            .filter_map(|(_, member)| member.downcast_ref::<L::Member>())
    }

    pub fn named(&self, name: &str) -> Option<JobRef> {
        let index = self.names.binary_search_by(|probe| probe.cmp(&name)).ok()?;
        Some(JobRef::named(self.names[index], self.sources[index]))
    }

    pub fn hash(&self) -> [u8; 32] {
        self.hash
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static str, [u8; 32], JobKindRef)> + '_ {
        self.names
            .iter()
            .zip(&self.sources)
            .zip(&self.kinds)
            .map(|((&name, &source), &kind)| (name, source, kind))
    }
}

pub trait Shares: Send + Sync {
    fn share(&self, key: [u8; 32]) -> Option<&[u8]>;
}

#[derive(Resource, Default)]
pub struct JobListed {
    pub hash: [u8; 32],
    names: Vec<&'static str>,
    sources: Vec<[u8; 32]>,
}

impl JobListed {
    pub fn of(list: &JobList) -> Self {
        Self {
            hash: list.hash,
            names: list.names.clone(),
            sources: list.sources.clone(),
        }
    }

    pub fn listed(&self, name: &str, source: &[u8; 32]) -> bool {
        self.names
            .binary_search_by(|probe| probe.cmp(&name))
            .is_ok_and(|index| self.sources[index] == *source)
    }
}

pub enum JobOrder {
    Start {
        key: [u8; 32],
        kind: &'static str,
        source: [u8; 32],
        input: Vec<u8>,
        priority: u32,
    },
    Fetch {
        key: [u8; 32],
        kind: &'static str,
        source: [u8; 32],
        input: Vec<u8>,
        priority: u32,
    },
    Cancel {
        key: [u8; 32],
    },
    Share {
        key: [u8; 32],
        bytes: Arc<[u8]>,
    },
    Keep {
        key: [u8; 32],
        kind: &'static str,
        source: [u8; 32],
        bytes: Vec<u8>,
        parity: RecordParity,
    },
}

#[derive(Default)]
pub struct JobOrders {
    pub orders: Vec<JobOrder>,
}

impl SystemBuffer for JobOrders {
    fn queue(&mut self, _: &SystemMeta, mut world: DeferredWorld) {
        if self.orders.is_empty() {
            return;
        }
        if let Some(mut queue) = world.get_resource_mut::<JobQueue>() {
            queue.orders.append(&mut self.orders);
        }
    }
}

#[derive(Resource, Default)]
pub struct JobQueue {
    pub orders: Vec<JobOrder>,
}

struct Held {
    record: Option<Result<Arc<[u8]>, Error>>,
    interests: u32,
    idle: u32,
}

#[derive(Default)]
pub struct RecordStore {
    held: HashMap<[u8; 32], Held>,
}

impl RecordStore {
    pub fn get(&self, key: &[u8; 32]) -> Option<Result<Arc<[u8]>, Error>> {
        self.held.get(key)?.record.clone()
    }

    pub fn waiting(&self, key: &[u8; 32]) -> bool {
        self.held
            .get(key)
            .is_some_and(|held| held.record.is_none() && held.interests > 0)
    }

    pub fn want(&mut self, key: [u8; 32]) {
        let held = self.held.entry(key).or_insert(Held {
            record: None,
            interests: 0,
            idle: 0,
        });
        held.interests = held.interests.saturating_add(1);
        held.idle = 0;
    }

    pub fn release(&mut self, key: &[u8; 32]) -> bool {
        let Some(held) = self.held.get_mut(key) else {
            return false;
        };
        held.interests = held.interests.saturating_sub(1);
        held.interests == 0
    }

    pub fn arrive(&mut self, key: [u8; 32], record: Result<Arc<[u8]>, Error>) {
        let held = self.held.entry(key).or_insert(Held {
            record: None,
            interests: 0,
            idle: 0,
        });
        held.record = Some(record);
    }

    pub fn evict(&mut self, keep_frames: u32) -> usize {
        let before = self.held.len();
        self.held.retain(|_, held| {
            if held.interests > 0 {
                return true;
            }
            held.idle = held.idle.saturating_add(1);
            held.idle <= keep_frames
        });
        before - self.held.len()
    }

    pub fn len(&self) -> usize {
        self.held.len()
    }

    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }
}
