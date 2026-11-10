use crate::collections::HashMap;
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bevy_ecs::change_detection::Mut;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Deferred, Res, SystemParam};
use bevy_ecs::world::World;
use engine_registry::{
    JobBatch, JobKindRef, JobList, JobListed, JobMember, JobOrder, JobOrders, RecordParity, Shares,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    Background, Census, Error, ErrorCode, Interest, JobRef, List, Request, Requests, Source, Ticket,
};

type JobStart = fn(&[u8], &mut JobCx<'_>) -> Result<Box<dyn Job>, Error>;

pub struct JobKind {
    name: &'static str,
    source: &'static Source,
    start: JobStart,
    parity: RecordParity,
}

impl JobKind {
    pub const fn new(name: &'static str, source: &'static Source, start: JobStart) -> Self {
        Self {
            name,
            source,
            start,
            parity: RecordParity::Exact,
        }
    }

    #[must_use]
    pub const fn kept_alike(mut self, alike: fn(&[u8], &[u8]) -> bool) -> Self {
        self.parity = RecordParity::Within(alike);
        self
    }

    #[doc(hidden)]
    pub fn name(&self) -> &'static str {
        self.name
    }

    #[doc(hidden)]
    pub fn source(&self) -> &'static Source {
        self.source
    }

    #[doc(hidden)]
    pub fn parity(&self) -> RecordParity {
        self.parity
    }

    #[doc(hidden)]
    pub fn start(&self, input: &[u8], cx: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
        (self.start)(input, cx)
    }

    pub fn undecoded(&self, error: impl std::fmt::Display) -> Error {
        Error::new(ErrorCode::InvalidRecord, format!("$.jobs.{}", self.name))
            .got(format!("its output does not decode: {error}"))
    }
}

impl From<&'static JobKind> for JobRef {
    fn from(kind: &'static JobKind) -> Self {
        JobRef::named(kind.name, kind.source.hash())
    }
}

pub trait Job: Send {
    fn step(&mut self, budget: u64, cx: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error>;
}

const HELD_VALUES: usize = 8;

type HeldValues = Vec<([u8; 32], Arc<dyn Any + Send + Sync>)>;

#[derive(Default)]
pub struct JobMemo {
    held: HashMap<TypeId, HeldValues>,
}

impl JobMemo {
    fn find<T: Send + Sync + 'static>(&self, key: [u8; 32]) -> Option<Arc<T>> {
        let held = self.held.get(&TypeId::of::<T>())?;
        let (_, value) = held.iter().find(|(at, _)| *at == key)?;
        Arc::clone(value).downcast::<T>().ok()
    }

    fn hold<T: Send + Sync + 'static>(&mut self, key: [u8; 32], value: Arc<T>) {
        let held = self.held.entry(TypeId::of::<T>()).or_default();
        if held.len() >= HELD_VALUES {
            held.remove(0);
        }
        held.push((key, value as Arc<dyn Any + Send + Sync>));
    }
}

pub struct JobCx<'c> {
    list: &'c JobList,
    shares: &'c dyn Shares,
    memo: RefCell<&'c mut JobMemo>,
}

impl<'c> JobCx<'c> {
    #[doc(hidden)]
    pub fn new(list: &'c JobList, shares: &'c dyn Shares, memo: &'c mut JobMemo) -> Self {
        Self {
            list,
            shares,
            memo: RefCell::new(memo),
        }
    }
}

impl JobCx<'_> {
    pub fn start(&mut self, kind: &JobRef, input: &[u8]) -> Result<Box<dyn Job>, Error> {
        let listed = self
            .list
            .find(kind.name(), &kind.source())
            .and_then(|listed| listed.downcast_ref::<JobKind>())
            .ok_or_else(|| {
                Error::new(ErrorCode::JobUnsupported, format!("$.jobs.{}", kind.name()))
                    .got(kind.name())
            })?;
        listed.start(input, self)
    }

    pub fn share(&self, key: RecordKey) -> Option<&[u8]> {
        self.shares.share(key.0)
    }

    pub fn members<L: List>(&self) -> impl Iterator<Item = &L::Member> + '_ {
        self.list.members::<L>()
    }

    pub fn held<T: Send + Sync + 'static>(
        &self,
        key: [u8; 32],
        make: impl FnOnce() -> Result<T, Error>,
    ) -> Result<Arc<T>, Error> {
        let found = self.memo.borrow().find::<T>(key);
        if let Some(found) = found {
            return Ok(found);
        }
        let made = Arc::new(make()?);
        self.memo.borrow_mut().hold(key, Arc::clone(&made));
        Ok(made)
    }
}

#[doc(hidden)]
pub fn job_list(kinds: &[&'static JobKind]) -> Result<JobList, Error> {
    let core: [&'static JobKind; 1] = [&MESH_LODS];
    JobList::new(
        core.into_iter()
            .chain(kinds.iter().copied())
            .map(|kind| (kind.name, kind.source.hash(), kind as JobKindRef)),
    )
}

#[doc(hidden)]
pub fn job_list_with(
    kinds: &[&'static JobKind],
    members: Vec<JobMember>,
) -> Result<JobList, Error> {
    job_list(kinds).map(|list| list.with_members(members))
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, serde::Deserialize,
)]
pub struct RecordKey([u8; 32]);

impl RecordKey {
    #[doc(hidden)]
    pub fn new(source: &[u8; 32], kind: &str, identity: &[u8]) -> Self {
        record_key(source, kind, identity)
    }

    #[doc(hidden)]
    pub fn of(kind: &JobKind, input: &[u8]) -> Self {
        key_of(kind, input)
    }

    #[doc(hidden)]
    pub const fn from_hash(hash: [u8; 32]) -> Self {
        Self(hash)
    }

    #[doc(hidden)]
    pub fn shared(bytes: &[u8]) -> Self {
        share_key(bytes)
    }

    #[doc(hidden)]
    pub fn shared_tagged(tag: u8, bytes: &[u8]) -> Self {
        Self(
            Sha256::new()
                .chain_update([tag])
                .chain_update(bytes)
                .finalize()
                .into(),
        )
    }

    #[doc(hidden)]
    pub fn hash(&self) -> [u8; 32] {
        self.0
    }

    #[doc(hidden)]
    pub fn hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    pub(crate) fn identity(&self) -> u128 {
        let mut head = [0_u8; 16];
        head.copy_from_slice(&self.0[..16]);
        u128::from_le_bytes(head)
    }
}

pub(crate) fn record_key(source: &[u8; 32], kind: &str, identity: &[u8]) -> RecordKey {
    let mut hash = Sha256::new();
    hash.update(source);
    hash.update(kind.as_bytes());
    hash.update([0]);
    hash.update(identity);
    RecordKey(hash.finalize().into())
}

fn key_of(kind: &JobKind, input: &[u8]) -> RecordKey {
    record_key(&kind.source.hash(), kind.name, input)
}

fn run_key(kind: &JobKind, input: &[u8]) -> RecordKey {
    record_key(
        &kind.source.hash(),
        kind.name,
        &content_hash(input).to_le_bytes(),
    )
}

fn share_key(bytes: &[u8]) -> RecordKey {
    RecordKey(Sha256::digest(bytes).into())
}

#[derive(SystemParam)]
pub struct Jobs<'w, 's> {
    orders: Deferred<'s, JobOrders>,
    listed: Res<'w, JobListed>,
    records: Res<'w, Records>,
    exchange: Option<Res<'w, JobExchange>>,
}

impl Jobs<'_, '_> {
    pub fn admit(&mut self, priority: u32) -> bool {
        let exchange = self.exchange.as_deref();
        let (workers, running) = exchange.map_or((0, 0), |exchange| {
            let admitted = exchange.admitted.load(Ordering::Relaxed);
            (exchange.workers, exchange.running + admitted)
        });
        let admitted = JobBudget::of(workers, running).take(priority);
        if admitted {
            if let Some(exchange) = exchange {
                exchange.admitted.fetch_add(1, Ordering::Relaxed);
            }
            self.orders.admitted = true;
        }
        admitted
    }

    pub fn start<T>(
        &mut self,
        kind: &'static JobKind,
        input: Vec<u8>,
        priority: u32,
        decode: impl FnOnce(Arc<[u8]>) -> Result<T, Error> + Send + Sync + 'static,
    ) -> Background<T> {
        let admitted = std::mem::take(&mut self.orders.admitted);
        self.make(kind, input, priority, admitted, decode)
    }

    pub fn fetch<T>(
        &mut self,
        kind: &'static JobKind,
        input: &[u8],
        priority: u32,
        decode: impl FnOnce(Arc<[u8]>) -> Result<T, Error> + Send + Sync + 'static,
    ) -> Background<T> {
        self.make(kind, input.to_vec(), priority, false, decode)
    }

    fn make<T>(
        &mut self,
        kind: &'static JobKind,
        input: Vec<u8>,
        priority: u32,
        admitted: bool,
        decode: impl FnOnce(Arc<[u8]>) -> Result<T, Error> + Send + Sync + 'static,
    ) -> Background<T> {
        let key = self.listed_key(kind, &input);
        let interest = self.records.requests.ask_later(key);
        self.orders.orders.push(JobOrder::Start {
            key: key.0,
            kind: JobRef::from(kind),
            input,
            priority,
            admitted,
        });
        Background::asked(interest, decode)
    }

    pub fn run<T>(
        &mut self,
        kind: &'static JobKind,
        input: Vec<u8>,
        priority: u32,
        decode: impl FnOnce(Arc<[u8]>) -> Result<T, Error> + Send + Sync + 'static,
    ) -> Background<T> {
        self.listed(kind);
        let key = run_key(kind, &input);
        let interest = self.records.requests.ask_later(key);
        self.orders.orders.push(JobOrder::Run {
            key: key.0,
            kind: JobRef::from(kind),
            input,
            priority,
        });
        Background::asked(interest, decode)
    }

    pub fn look<T>(
        &mut self,
        kind: &'static JobKind,
        input: &[u8],
        priority: u32,
        decode: impl FnOnce(Arc<[u8]>) -> Result<T, Error> + Send + Sync + 'static,
    ) -> Background<T> {
        let key = self.listed_key(kind, input);
        let interest = self.records.requests.ask_later(key);
        self.orders.orders.push(JobOrder::Look {
            key: key.0,
            kind: Some(JobRef::from(kind)),
            priority,
        });
        Background::asked(interest, decode)
    }

    pub fn share(&mut self, bytes: Arc<[u8]>) -> RecordKey {
        let key = share_key(&bytes);
        self.orders
            .orders
            .push(JobOrder::Share { key: key.0, bytes });
        key
    }

    pub fn share_as(&mut self, key: RecordKey, bytes: Arc<[u8]>) {
        self.orders
            .orders
            .push(JobOrder::Share { key: key.0, bytes });
    }

    pub fn unshare(&mut self, key: RecordKey) {
        self.orders.orders.push(JobOrder::Unshare { key: key.0 });
    }

    pub fn keep(&mut self, kind: &'static JobKind, input: &[u8], bytes: Vec<u8>) {
        let key = self.listed_key(kind, input);
        self.orders.orders.push(JobOrder::Keep {
            key: key.0,
            kind: Some(JobRef::from(kind)),
            bytes,
            parity: kind.parity,
        });
    }

    fn listed(&self, kind: &JobKind) {
        debug_assert!(
            self.listed.listed(kind.name, &kind.source.hash()),
            "job_kind_unlisted {}",
            kind.name
        );
    }

    fn listed_key(&self, kind: &JobKind, input: &[u8]) -> RecordKey {
        self.listed(kind);
        key_of(kind, input)
    }
}

pub(crate) enum Work {
    Waiting {
        batch: JobBatch,
        admitted: bool,
        message: JobMessage,
    },
    Sent {
        kind: Option<JobRef>,
        asked: f64,
        then: Option<JobRequest>,
    },
}

pub(crate) type RecordSet = Requests<RecordKey, Arc<[u8]>, Work>;

#[derive(Resource, Default, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[component(immutable)]
pub struct Records {
    pub(crate) requests: RecordSet,
}

impl Records {
    pub fn get(&self, interest: &Interest<RecordKey>) -> Option<Result<Arc<[u8]>, Error>> {
        match self.requests.of(interest)? {
            Request::Asked(_) => None,
            Request::Ready(bytes) => Some(Ok(Arc::clone(bytes))),
            Request::Failed(error) => Some(Err(error.clone())),
            Request::Missing => Some(Err(missing_record("$.records"))),
        }
    }

    pub fn request(&self, interest: &Interest<RecordKey>) -> Option<&Request<Arc<[u8]>>> {
        self.requests.of(interest)
    }

    pub fn census(&self) -> Census {
        self.requests.census()
    }

    pub fn failures(&self) -> impl Iterator<Item = (&RecordKey, &Error)> + '_ {
        self.requests.failures()
    }
}

#[doc(hidden)]
pub fn look_record<T>(
    world: &mut World,
    kind: Option<JobRef>,
    (key, priority): (RecordKey, u32),
    decode: impl FnOnce(Arc<[u8]>) -> Result<T, Error> + Send + Sync + 'static,
) -> Background<T> {
    let interest = world.resource::<Records>().requests.ask_later(key);
    world
        .get_resource_or_init::<engine_registry::JobQueue>()
        .orders
        .push((
            JobBatch { owner: 0, run: 0 },
            JobOrder::Look {
                key: key.0,
                kind,
                priority,
            },
        ));
    Background::asked(interest, decode)
}

fn missing_record(path: &str) -> Error {
    Error::new(ErrorCode::InvalidRecord, path).got(MISSING_RECORD)
}

pub fn record_missing(error: &Error) -> bool {
    error.code == ErrorCode::InvalidRecord
        && error
            .detail
            .got
            .as_ref()
            .and_then(serde_json::Value::as_str)
            == Some(MISSING_RECORD)
}

static LODBUILD: Source =
    Source::generated("engine_lodbuild", false, crate::sources::LODBUILD, &[]);

pub(crate) const MESH_LODS_NAME: &str = "engine::mesh_lods";

#[doc(hidden)]
pub static MESH_LODS: JobKind = JobKind::new(MESH_LODS_NAME, &LODBUILD, start_mesh_lods);

struct Whole {
    input: Vec<u8>,
    run: fn(&[u8]) -> Result<Vec<u8>, Error>,
}

impl Job for Whole {
    fn step(&mut self, _: u64, _: &mut JobCx<'_>) -> Result<Option<Vec<u8>>, Error> {
        (self.run)(&self.input).map(Some)
    }
}

fn start_mesh_lods(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Ok(Box::new(Whole {
        input: input.to_vec(),
        run: engine_lodbuild::run,
    }))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(pub u64);

pub const MISSING_RECORD: &str = "no store holds the record";

#[derive(Clone, Debug, PartialEq)]
pub struct JobRequest {
    pub id: JobId,
    pub kind: JobRef,
    pub input: Vec<u8>,
    pub priority: u32,
    pub record: Option<RecordKey>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordFetch {
    pub id: JobId,
    pub kind: Option<JobRef>,
    pub record: RecordKey,
    pub priority: u32,
}

#[derive(Debug)]
pub enum JobMessage {
    Cancel(JobId),
    Share(JobShare),
    Run(JobRequest),
    Fetch(RecordFetch),
    Keep(RecordKey, Option<JobRef>, Vec<u8>, RecordParity),
}

impl JobMessage {
    fn asked(&self) -> Option<(JobId, Option<&JobRef>, u32)> {
        match self {
            Self::Run(request) => Some((request.id, Some(&request.kind), request.priority)),
            Self::Fetch(fetch) => Some((fetch.id, fetch.kind.as_ref(), fetch.priority)),
            Self::Cancel(_) | Self::Share(_) | Self::Keep(..) => None,
        }
    }
}

pub const URGENT_JOBS: u8 = 0;
pub const GAMEPLAY_JOBS: u8 = 1;
pub const VIEW_JOBS: u8 = 2;
pub const IDLE_JOBS: u8 = 3;

pub const fn job_priority(class: u8, place: u32) -> u32 {
    ((class as u32) << 24) | if place > 0xff_ffff { 0xff_ffff } else { place }
}

pub const fn job_class(priority: u32) -> u8 {
    (priority >> 24) as u8
}

pub fn content_hash(bytes: &[u8]) -> u128 {
    const C1: u64 = 0x87c3_7b91_1142_53d5;
    const C2: u64 = 0x4cf5_ad43_2745_937f;
    let mix = |mut k: u64| {
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        k ^ (k >> 33)
    };
    let word = |bytes: &[u8]| {
        let mut word = [0_u8; 8];
        word[..bytes.len()].copy_from_slice(bytes);
        u64::from_le_bytes(word)
    };
    let (mut h1, mut h2) = (0_u64, 0_u64);
    let (blocks, tail) = bytes.as_chunks::<16>();
    for block in blocks {
        let (k1, k2) = (word(&block[..8]), word(&block[8..]));
        h1 ^= k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 = h1
            .rotate_left(27)
            .wrapping_add(h2)
            .wrapping_mul(5)
            .wrapping_add(0x52dc_e729);
        h2 ^= k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 = h2
            .rotate_left(31)
            .wrapping_add(h1)
            .wrapping_mul(5)
            .wrapping_add(0x3849_5ab5);
    }
    if tail.len() > 8 {
        h2 ^= word(&tail[8..])
            .wrapping_mul(C2)
            .rotate_left(33)
            .wrapping_mul(C1);
    }
    if !tail.is_empty() {
        h1 ^= word(&tail[..tail.len().min(8)])
            .wrapping_mul(C1)
            .rotate_left(31)
            .wrapping_mul(C2);
    }
    let len = bytes.len() as u64;
    h1 ^= len;
    h2 ^= len;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = mix(h1);
    h2 = mix(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    (u128::from(h2) << 64) | u128::from(h1)
}

#[derive(Clone, Debug, PartialEq)]
pub enum JobShare {
    Clear,
    Set { key: RecordKey, bytes: Arc<[u8]> },
    Remove { key: RecordKey },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct JobStats {
    pub workers: u32,
    pub running: u32,
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub shares: u32,
    pub shared_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_bytes: u64,
    pub cache_mismatches: u64,
    pub held: u32,
    pub in_place: u64,
}

struct Share {
    key: RecordKey,
    bytes: Arc<[u8]>,
    shared: bool,
    reads: u32,
}

struct Shared<'a>(&'a [Share]);

impl Shares for Shared<'_> {
    fn share(&self, key: [u8; 32]) -> Option<&[u8]> {
        self.0
            .iter()
            .find(|share| share.key.0 == key)
            .map(|share| &*share.bytes)
    }
}

#[doc(hidden)]
#[derive(Resource)]
pub struct JobExchange {
    workers: u32,
    hosted: bool,
    verifying: bool,
    list: Arc<JobList>,
    sent: Vec<JobMessage>,
    queue: Vec<RecordKey>,
    running: usize,
    admitted: AtomicUsize,
    cache: [u64; 4],
    held: u32,
    shared: Vec<Share>,
    memo: JobMemo,
    stats: JobStats,
    last_failure: Option<String>,
}

impl Default for JobExchange {
    fn default() -> Self {
        Self {
            workers: 0,
            hosted: false,
            verifying: false,
            list: Arc::new(job_list(&[]).expect("core_job_kinds_distinct")),
            sent: vec![JobMessage::Share(JobShare::Clear)],
            queue: Vec::new(),
            running: 0,
            admitted: AtomicUsize::new(0),
            cache: [0; 4],
            held: 0,
            shared: Vec::new(),
            memo: JobMemo::default(),
            stats: JobStats::default(),
            last_failure: None,
        }
    }
}

impl JobExchange {
    pub fn workers(&self) -> u32 {
        self.workers
    }

    pub fn running(&self) -> usize {
        self.running
    }

    pub fn verifying(&self) -> bool {
        self.verifying
    }

    pub fn set_verifying(&mut self, verifying: bool) {
        self.verifying = verifying;
    }

    pub fn hosted(&self) -> bool {
        self.hosted
    }

    pub fn set_workers(&mut self, workers: u32) {
        self.workers = workers;
        self.hosted = true;
    }

    pub fn list(&self) -> &JobList {
        &self.list
    }

    pub fn shared_list(&self) -> Arc<JobList> {
        Arc::clone(&self.list)
    }

    pub(crate) fn set_list(&mut self, list: JobList) {
        self.list = Arc::new(list);
        self.memo = JobMemo::default();
    }

    pub fn keep(&mut self, record: RecordKey, kind: Option<JobRef>, bytes: Vec<u8>) {
        self.keep_with(record, kind, bytes, RecordParity::Exact);
    }

    fn keep_with(
        &mut self,
        record: RecordKey,
        kind: Option<JobRef>,
        bytes: Vec<u8>,
        parity: RecordParity,
    ) {
        if self.hosted {
            self.sent
                .push(JobMessage::Keep(record, kind, bytes, parity));
        }
    }

    pub fn take(&mut self) -> Vec<JobMessage> {
        std::mem::take(&mut self.sent)
    }

    pub fn share(&mut self, key: RecordKey, bytes: Arc<[u8]>) {
        match self.shared.iter_mut().find(|share| share.key == key) {
            Some(share) => share.shared = true,
            None => self.add_share(key, bytes, (true, 0)),
        }
    }

    pub fn unshare(&mut self, key: RecordKey) {
        if let Some(share) = self.shared.iter_mut().find(|share| share.key == key) {
            share.shared = false;
        }
        self.drop_share(key);
    }

    pub(crate) fn hold_read(&mut self, key: RecordKey, bytes: Arc<[u8]>) {
        match self.shared.iter_mut().find(|share| share.key == key) {
            Some(share) => share.reads += 1,
            None => self.add_share(key, bytes, (false, 1)),
        }
    }

    pub(crate) fn release_read(&mut self, key: RecordKey) {
        if let Some(share) = self.shared.iter_mut().find(|share| share.key == key) {
            share.reads = share.reads.saturating_sub(1);
        }
        self.drop_share(key);
    }

    fn add_share(&mut self, key: RecordKey, bytes: Arc<[u8]>, (shared, reads): (bool, u32)) {
        self.shared.push(Share {
            key,
            bytes: Arc::clone(&bytes),
            shared,
            reads,
        });
        self.sent
            .push(JobMessage::Share(JobShare::Set { key, bytes }));
    }

    fn drop_share(&mut self, key: RecordKey) {
        let before = self.shared.len();
        self.shared
            .retain(|share| share.key != key || share.shared || share.reads > 0);
        if self.shared.len() < before {
            self.sent.push(JobMessage::Share(JobShare::Remove { key }));
        }
    }

    pub fn run_here(&mut self, kind: &JobRef, input: &[u8]) -> Result<Vec<u8>, String> {
        self.stats.in_place += 1;
        let shares = Shared(&self.shared);
        let mut cx = JobCx::new(&self.list, &shares, &mut self.memo);
        let mut job = cx.start(kind, input).map_err(|error| error.to_string())?;
        loop {
            if let Some(output) = job
                .step(u64::MAX, &mut cx)
                .map_err(|error| error.to_string())?
            {
                return Ok(output);
            }
        }
    }

    pub fn report_cache(&mut self, hits: u64, misses: u64, bytes: u64, mismatches: u64) {
        self.cache = [hits, misses, bytes, mismatches];
    }

    pub fn report_held(&mut self, held: u32) {
        self.held = held;
    }

    pub fn last_failure(&self) -> Option<&str> {
        self.last_failure.as_deref()
    }

    pub fn stats(&self) -> JobStats {
        JobStats {
            workers: self.workers,
            running: self.running as u32,
            shares: self.shared.len() as u32,
            shared_bytes: self
                .shared
                .iter()
                .map(|share| share.bytes.len() as u64)
                .sum(),
            cache_hits: self.cache[0],
            cache_misses: self.cache[1],
            cache_bytes: self.cache[2],
            cache_mismatches: self.cache[3],
            held: self.held,
            ..self.stats
        }
    }

    fn settle(&mut self, orders: Vec<(JobBatch, JobOrder)>, records: &mut RecordSet, now: f64) {
        for (key, released, work) in records.settle_work(now) {
            self.abandon(key, &released, work);
        }
        let mut firsts: Vec<JobBatch> = Vec::new();
        for (batch, order) in orders {
            if !firsts.iter().any(|first| first.owner == batch.owner) {
                firsts.push(batch);
            }
            match order {
                JobOrder::Start {
                    key,
                    kind,
                    input,
                    priority,
                    admitted,
                } => self.want(records, RecordKey(key), (batch, admitted), |id| {
                    JobMessage::Run(JobRequest {
                        id,
                        kind,
                        input,
                        priority,
                        record: Some(RecordKey(key)),
                    })
                }),
                JobOrder::Run {
                    key,
                    kind,
                    input,
                    priority,
                } => self.want(records, RecordKey(key), (batch, false), |id| {
                    JobMessage::Run(JobRequest {
                        id,
                        kind,
                        input,
                        priority,
                        record: None,
                    })
                }),
                JobOrder::Look {
                    key,
                    kind,
                    priority,
                } => self.want(records, RecordKey(key), (batch, false), |id| {
                    JobMessage::Fetch(RecordFetch {
                        id,
                        kind,
                        record: RecordKey(key),
                        priority,
                    })
                }),
                JobOrder::Share { key, bytes } => self.share(RecordKey(key), bytes),
                JobOrder::Unshare { key } => self.unshare(RecordKey(key)),
                JobOrder::Keep {
                    key,
                    kind,
                    bytes,
                    parity,
                } => {
                    let key = RecordKey(key);
                    if let Some(ticket) = records.ticket(&key) {
                        let sent = matches!(records.work(&key), Some(Work::Sent { .. }));
                        records.ready(&key, ticket, Arc::from(bytes.as_slice()));
                        self.queue.retain(|queued| *queued != key);
                        if sent {
                            self.cancel(JobId(ticket.number()));
                        }
                    }
                    self.keep_with(key, kind, bytes, parity);
                }
            }
        }
        self.admit(records, &firsts);
    }

    fn want(
        &mut self,
        records: &mut RecordSet,
        key: RecordKey,
        (batch, admitted): (JobBatch, bool),
        message: impl FnOnce(JobId) -> JobMessage,
    ) {
        let Some(ticket) = records.ticket(&key) else {
            return;
        };
        let message = message(JobId(ticket.number()));
        let made = matches!(message, JobMessage::Run(_));
        match records.work_mut(&key) {
            None => {
                records.set_work(
                    &key,
                    Work::Waiting {
                        batch,
                        admitted,
                        message,
                    },
                );
                self.queue.push(key);
            }
            Some(Work::Waiting {
                message: waiting, ..
            }) => {
                if made && matches!(waiting, JobMessage::Fetch(_)) {
                    *waiting = message;
                }
            }
            Some(Work::Sent { then, .. }) => {
                if let JobMessage::Run(request) = message {
                    then.get_or_insert(request);
                }
            }
        }
    }

    fn admit(&mut self, records: &mut RecordSet, firsts: &[JobBatch]) {
        *self.admitted.get_mut() = 0;
        self.running = records
            .works()
            .filter(|(_, work)| matches!(work, Work::Sent { .. }))
            .count();
        let mut budgets: Vec<(JobBatch, JobBudget)> = Vec::new();
        for key in std::mem::take(&mut self.queue) {
            let begun = match records.work_mut(&key) {
                Some(Work::Waiting { admitted: true, .. }) => true,
                Some(Work::Waiting { batch, message, .. }) => {
                    let held = if batch.run == LEFT_OVER {
                        firsts
                            .iter()
                            .copied()
                            .find(|first| first.owner == batch.owner)
                            .unwrap_or(*batch)
                    } else {
                        *batch
                    };
                    let slot = match budgets.iter().position(|(kept, _)| *kept == held) {
                        Some(slot) => slot,
                        None => {
                            budgets.push((held, JobBudget::of(self.workers, self.running)));
                            budgets.len() - 1
                        }
                    };
                    let priority = message.asked().map_or(0, |(_, _, priority)| priority);
                    let taken = budgets[slot].1.take(priority);
                    if !taken {
                        *batch = JobBatch {
                            run: LEFT_OVER,
                            ..held
                        };
                    }
                    taken
                }
                Some(Work::Sent { .. }) | None => continue,
            };
            if begun {
                self.begin(records, key);
            } else {
                self.queue.push(key);
            }
        }
    }

    fn begin(&mut self, records: &mut RecordSet, key: RecordKey) {
        let Some(work) = records.work_mut(&key) else {
            return;
        };
        let kind = match &*work {
            Work::Waiting { message, .. } => message.asked().and_then(|(_, kind, _)| kind.cloned()),
            Work::Sent { .. } => return,
        };
        let sent = Work::Sent {
            kind,
            asked: crate::profile::now_ms(),
            then: None,
        };
        let Work::Waiting { message, .. } = std::mem::replace(work, sent) else {
            return;
        };
        self.send(records, message);
    }

    fn send(&mut self, records: &mut RecordSet, message: JobMessage) {
        self.stats.submitted += 1;
        if self.hosted {
            self.running += 1;
            self.sent.push(message);
            return;
        }
        let (id, output) = match message {
            JobMessage::Run(request) => (
                request.id,
                self.run_here(&request.kind, &request.input).map(Arc::from),
            ),
            JobMessage::Fetch(fetch) => (fetch.id, Err(MISSING_RECORD.to_owned())),
            JobMessage::Cancel(_) | JobMessage::Share(_) | JobMessage::Keep(..) => return,
        };
        self.running += 1;
        self.land(records, id, output);
    }

    pub(crate) fn land(
        &mut self,
        records: &mut RecordSet,
        id: JobId,
        output: Result<Arc<[u8]>, String>,
    ) {
        let ticket = Ticket::numbered(id.0);
        let Some(&key) = records.key_of(ticket) else {
            return;
        };
        let Some(Work::Sent { kind, asked, then }) = records.work_mut(&key) else {
            return;
        };
        let (name, asked, then) = (
            kind.as_ref()
                .map_or("records", |kind| kind.name())
                .to_owned(),
            *asked,
            then.take(),
        );
        self.running = self.running.saturating_sub(1);
        let missing = matches!(&output, Err(message) if message == MISSING_RECORD);
        if let (true, Some(request)) = (missing, then) {
            if let Some(Work::Sent { kind, .. }) = records.work_mut(&key) {
                *kind = Some(request.kind.clone());
            }
            self.send(records, JobMessage::Run(request));
            return;
        }
        crate::profile::waited(&name, asked);
        match output {
            Ok(bytes) => {
                self.stats.completed += 1;
                records.ready(&key, ticket, bytes);
            }
            Err(_) if missing => {
                records.miss(&key, ticket);
            }
            Err(message) => {
                self.stats.failed += 1;
                self.last_failure = Some(message.clone());
                let error =
                    Error::new(ErrorCode::InvalidRecord, format!("$.jobs.{name}")).got(message);
                records.fail(&key, ticket, error);
            }
        }
    }

    fn abandon(&mut self, key: RecordKey, released: &Request<Arc<[u8]>>, work: Option<Work>) {
        match (work, released.asked()) {
            (Some(Work::Waiting { .. }), _) => self.queue.retain(|queued| *queued != key),
            (Some(Work::Sent { .. }), Some(ticket)) => self.cancel(JobId(ticket.number())),
            (Some(Work::Sent { .. }) | None, _) => {}
        }
    }

    fn cancel(&mut self, id: JobId) {
        self.stats.cancelled += 1;
        self.running = self.running.saturating_sub(1);
        match self
            .sent
            .iter()
            .position(|message| message.asked().is_some_and(|(sent, ..)| sent == id))
        {
            Some(at) => {
                self.sent.remove(at);
            }
            None => self.sent.push(JobMessage::Cancel(id)),
        }
    }
}

pub(crate) fn exchange_orders(world: &mut World) {
    let orders = match world.get_resource_mut::<engine_registry::JobQueue>() {
        Some(mut queue) if !queue.orders.is_empty() => std::mem::take(&mut queue.orders),
        Some(_) | None => Vec::new(),
    };
    let idle = orders.is_empty()
        && world
            .get_resource::<JobExchange>()
            .is_none_or(|exchange| exchange.queue.is_empty())
        && world
            .get_resource::<Records>()
            .is_none_or(|records| !records.requests.changed());
    if idle {
        return;
    }
    let now = world
        .get_resource::<crate::FrameTime>()
        .map_or(0.0, |time| time.real_elapsed);
    world.resource_scope(|world, mut exchange: Mut<JobExchange>| {
        crate::modify(world, |records: &mut Records| {
            exchange.settle(orders, &mut records.requests, now);
        });
    });
}

pub(crate) fn land_job(world: &mut World, id: JobId, output: Result<Arc<[u8]>, String>) {
    world.resource_scope(|world, mut exchange: Mut<JobExchange>| {
        crate::modify(world, |records: &mut Records| {
            exchange.land(&mut records.requests, id, output);
        });
    });
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JobFailures {
    pub count: u64,
    pub last: Option<(u64, String)>,
}

impl JobFailures {
    pub fn record(&mut self, frame: u64, error: String) {
        self.count += 1;
        self.last = Some((frame, error));
    }
}

const LEFT_OVER: u64 = u64::MAX;

struct JobBudget {
    slots: usize,
    urgent: usize,
}

impl JobBudget {
    fn of(workers: u32, running: usize) -> Self {
        let workers = workers.max(1) as usize;
        Self {
            slots: (workers * 2).saturating_sub(running),
            urgent: (workers * 3).saturating_sub(running),
        }
    }

    fn take(&mut self, priority: u32) -> bool {
        let free = if job_class(priority) == URGENT_JOBS {
            self.urgent
        } else {
            self.slots
        };
        if free == 0 {
            return false;
        }
        self.slots = self.slots.saturating_sub(1);
        self.urgent = self.urgent.saturating_sub(1);
        true
    }
}
