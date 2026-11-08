use crate::collections::HashMap;
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

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
    Census, Error, ErrorCode, Interest, JobRef, List, Request, Requests, Source, Ticket,
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
    let core: [&'static JobKind; 2] = [&TEXTURE_COOK, &MESH_LODS];
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
            (exchange.workers, exchange.running() + admitted)
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

    pub fn start(
        &mut self,
        kind: &'static JobKind,
        input: Vec<u8>,
        priority: u32,
    ) -> Interest<RecordKey> {
        let key = self.listed_key(kind, &input);
        let admitted = std::mem::take(&mut self.orders.admitted);
        let interest = self.records.requests.ask_later(key);
        self.orders.orders.push(JobOrder::Start {
            key: key.0,
            kind: kind.name,
            source: kind.source.hash(),
            input,
            priority,
            admitted,
        });
        interest
    }

    pub fn fetch(
        &mut self,
        kind: &'static JobKind,
        input: &[u8],
        priority: u32,
    ) -> Interest<RecordKey> {
        let key = self.listed_key(kind, input);
        let interest = self.records.requests.ask_later(key);
        self.orders.orders.push(JobOrder::Fetch {
            key: key.0,
            kind: kind.name,
            source: kind.source.hash(),
            input: input.to_vec(),
            priority,
        });
        interest
    }

    pub fn look(
        &mut self,
        kind: &'static JobKind,
        input: &[u8],
        priority: u32,
    ) -> Interest<RecordKey> {
        let key = self.listed_key(kind, input);
        let interest = self.records.requests.ask_later(key);
        self.orders.orders.push(JobOrder::Look {
            key: key.0,
            kind: kind.name,
            source: kind.source.hash(),
            priority,
        });
        interest
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

    pub fn share_record(&mut self, interest: &Interest<RecordKey>) -> RecordKey {
        let key = *interest.key();
        self.orders
            .orders
            .push(JobOrder::ShareRecord { key: key.0 });
        key
    }

    pub fn unshare(&mut self, key: RecordKey) {
        self.orders.orders.push(JobOrder::Unshare { key: key.0 });
    }

    pub fn keep(&mut self, kind: &'static JobKind, input: &[u8], bytes: Vec<u8>) {
        let key = self.listed_key(kind, input);
        self.orders.orders.push(JobOrder::Keep {
            key: key.0,
            kind: kind.name,
            source: kind.source.hash(),
            bytes,
            parity: kind.parity,
        });
    }

    fn listed_key(&self, kind: &JobKind, input: &[u8]) -> RecordKey {
        debug_assert!(
            self.listed.listed(kind.name, &kind.source.hash()),
            "job_kind_unlisted {}",
            kind.name
        );
        key_of(kind, input)
    }
}

#[derive(Resource, Default, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[component(immutable)]
pub struct Records {
    requests: Requests<RecordKey, Arc<[u8]>>,
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

    pub(crate) fn arrive(&mut self, key: RecordKey, record: Result<Arc<[u8]>, Error>) {
        let Some(ticket) = self.requests.ticket(&key) else {
            return;
        };
        match record {
            Ok(bytes) => self.requests.ready(&key, ticket, bytes),
            Err(error) => self.requests.fail(&key, ticket, error),
        };
    }
}

fn missing_record(path: &str) -> Error {
    Error::new(ErrorCode::InvalidRecord, path).got(MISSING_RECORD)
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[component(immutable)]
pub struct CookMode {
    pub cooking: bool,
}

static TEXTURES: Source =
    Source::generated("engine_textures", false, crate::sources::TEXTURES, &[]);

static LODBUILD: Source =
    Source::generated("engine_lodbuild", false, crate::sources::LODBUILD, &[]);

pub(crate) const TEXTURE_COOK_NAME: &str = "engine::texture_cook";

pub(crate) const MESH_LODS_NAME: &str = "engine::mesh_lods";

#[doc(hidden)]
pub static TEXTURE_COOK: JobKind = JobKind::new(TEXTURE_COOK_NAME, &TEXTURES, start_texture_cook);

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

fn start_texture_cook(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Ok(Box::new(Whole {
        input: input.to_vec(),
        run: engine_textures::cook_job,
    }))
}

fn start_mesh_lods(input: &[u8], _: &mut JobCx<'_>) -> Result<Box<dyn Job>, Error> {
    Ok(Box::new(Whole {
        input: input.to_vec(),
        run: engine_lodbuild::run,
    }))
}

static SUBMITTED: AtomicU64 = AtomicU64::new(0);

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

#[doc(hidden)]
pub type KeptRecord = (RecordKey, Option<JobRef>, Vec<u8>, RecordParity);

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Want {
    Make,
    Fetch,
    FetchOrMake,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Waiting { batch: JobBatch, admitted: bool },
    Fetching(JobId),
    Making(JobId),
}

struct Flight {
    key: RecordKey,
    ticket: Ticket,
    kind: JobRef,
    input: Vec<u8>,
    priority: u32,
    want: Want,
    stage: Stage,
    asked: f64,
}

struct Shared<'a>(&'a [(RecordKey, Arc<[u8]>)]);

impl Shares for Shared<'_> {
    fn share(&self, key: [u8; 32]) -> Option<&[u8]> {
        self.0
            .iter()
            .find(|(shared, _)| shared.0 == key)
            .map(|(_, bytes)| &**bytes)
    }
}

fn message(error: &Error) -> String {
    match error.detail.got.as_ref().and_then(serde_json::Value::as_str) {
        Some(message) => message.to_owned(),
        None => error.to_string(),
    }
}

#[doc(hidden)]
#[derive(Resource)]
pub struct JobExchange {
    workers: u32,
    hosted: bool,
    verifying: bool,
    list: Arc<JobList>,
    requests: Vec<JobRequest>,
    fetches: Vec<RecordFetch>,
    keeps: Vec<KeptRecord>,
    shares: Vec<JobShare>,
    cancels: Vec<JobId>,
    jobs: HashMap<JobId, Request<Arc<[u8]>>>,
    work: Vec<Flight>,
    admitted: AtomicUsize,
    cache: [u64; 4],
    held: u32,
    shared: Vec<(RecordKey, Arc<[u8]>)>,
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
            requests: Vec::new(),
            fetches: Vec::new(),
            keeps: Vec::new(),
            shares: vec![JobShare::Clear],
            cancels: Vec::new(),
            jobs: HashMap::default(),
            work: Vec::new(),
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
        self.jobs
            .values()
            .filter(|job| job.asked().is_some())
            .count()
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

    pub fn submit(&mut self, kind: JobRef, input: Vec<u8>, priority: u32) -> JobId {
        self.submit_with(kind, input, priority, None)
    }

    pub fn submit_record(
        &mut self,
        kind: JobRef,
        input: Vec<u8>,
        priority: u32,
        record: RecordKey,
    ) -> JobId {
        self.submit_with(kind, input, priority, Some(record))
    }

    fn submit_with(
        &mut self,
        kind: JobRef,
        input: Vec<u8>,
        priority: u32,
        record: Option<RecordKey>,
    ) -> JobId {
        let id = self.ask();
        if !self.hosted {
            let output = self.run_here(&kind, &input);
            self.finish(id, output.map(Arc::from));
            return id;
        }
        self.requests.push(JobRequest {
            id,
            kind,
            input,
            priority,
            record,
        });
        id
    }

    pub fn fetch(&mut self, kind: Option<JobRef>, record: RecordKey, priority: u32) -> JobId {
        let id = self.ask();
        if !self.hosted {
            self.finish(id, Err(MISSING_RECORD.to_owned()));
            return id;
        }
        self.fetches.push(RecordFetch {
            id,
            kind,
            record,
            priority,
        });
        id
    }

    fn ask(&mut self) -> JobId {
        let id = next_id();
        self.stats.submitted += 1;
        self.jobs.insert(id, Request::Asked(Ticket::numbered(id.0)));
        id
    }

    pub fn keep(&mut self, record: RecordKey, kind: Option<JobRef>, bytes: Vec<u8>) {
        self.keep_with(record, kind, bytes, RecordParity::Exact);
    }

    pub fn keep_with(
        &mut self,
        record: RecordKey,
        kind: Option<JobRef>,
        bytes: Vec<u8>,
        parity: RecordParity,
    ) {
        if self.hosted {
            self.keeps.push((record, kind, bytes, parity));
        }
    }

    pub fn take(&mut self, id: JobId) -> Option<Result<Arc<[u8]>, String>> {
        match self.ended(id)? {
            Request::Ready(bytes) => Some(Ok(bytes)),
            Request::Failed(error) => Some(Err(message(&error))),
            Request::Missing => Some(Err(MISSING_RECORD.to_owned())),
            Request::Asked(_) => None,
        }
    }

    fn ended(&mut self, id: JobId) -> Option<Request<Arc<[u8]>>> {
        if self.jobs.get(&id)?.asked().is_some() {
            return None;
        }
        self.jobs.remove(&id)
    }

    pub fn cancel(&mut self, id: JobId) {
        let Some(Request::Asked(_)) = self.jobs.remove(&id) else {
            return;
        };
        self.stats.cancelled += 1;
        if let Some(index) = self.requests.iter().position(|request| request.id == id) {
            self.requests.remove(index);
        } else if let Some(index) = self.fetches.iter().position(|fetch| fetch.id == id) {
            self.fetches.remove(index);
        } else {
            self.cancels.push(id);
        }
    }

    pub fn share(&mut self, key: RecordKey, bytes: Arc<[u8]>) {
        if self.shared.iter().any(|(shared, _)| *shared == key) {
            return;
        }
        self.shared.push((key, Arc::clone(&bytes)));
        self.shares.push(JobShare::Set { key, bytes });
    }

    pub fn unshare(&mut self, key: RecordKey) {
        let before = self.shared.len();
        self.shared.retain(|(shared, _)| *shared != key);
        if self.shared.len() < before {
            self.shares.push(JobShare::Remove { key });
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

    pub fn take_shares(&mut self, out: &mut Vec<JobShare>) {
        out.append(&mut self.shares);
    }

    pub fn take_requests(&mut self, out: &mut Vec<JobRequest>) {
        out.append(&mut self.requests);
    }

    pub fn take_fetches(&mut self, out: &mut Vec<RecordFetch>) {
        out.append(&mut self.fetches);
    }

    pub fn take_keeps(&mut self, out: &mut Vec<KeptRecord>) {
        out.append(&mut self.keeps);
    }

    pub fn take_cancels(&mut self, out: &mut Vec<JobId>) {
        out.append(&mut self.cancels);
    }

    pub fn complete(&mut self, id: JobId, output: Result<Arc<[u8]>, String>) {
        if self.jobs.get(&id).is_some_and(|job| job.asked().is_some()) {
            self.finish(id, output);
        }
    }

    pub fn report_cache(&mut self, hits: u64, misses: u64, bytes: u64, mismatches: u64) {
        self.cache = [hits, misses, bytes, mismatches];
    }

    pub fn report_held(&mut self, held: u32) {
        self.held = held;
    }

    fn finish(&mut self, id: JobId, output: Result<Arc<[u8]>, String>) {
        let ended = match output {
            Ok(bytes) => {
                self.stats.completed += 1;
                Request::Ready(bytes)
            }
            Err(message) if message == MISSING_RECORD => Request::Missing,
            Err(message) => {
                self.stats.failed += 1;
                self.last_failure = Some(message.clone());
                Request::Failed(Error::new(ErrorCode::InvalidRecord, "$.jobs").got(message))
            }
        };
        self.jobs.insert(id, ended);
    }

    pub fn last_failure(&self) -> Option<&str> {
        self.last_failure.as_deref()
    }

    pub fn stats(&self) -> JobStats {
        JobStats {
            workers: self.workers,
            running: self.running() as u32,
            shares: self.shared.len() as u32,
            shared_bytes: self
                .shared
                .iter()
                .map(|(_, bytes)| bytes.len() as u64)
                .sum(),
            cache_hits: self.cache[0],
            cache_misses: self.cache[1],
            cache_bytes: self.cache[2],
            cache_mismatches: self.cache[3],
            held: self.held,
            ..self.stats
        }
    }

    fn settle(
        &mut self,
        orders: Vec<(JobBatch, JobOrder)>,
        records: &mut Requests<RecordKey, Arc<[u8]>>,
        now: f64,
    ) {
        for (key, released) in records.settle(now) {
            if released.asked().is_some() {
                self.abandon(key);
            }
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
                    source,
                    input,
                    priority,
                    admitted,
                } => self.want(
                    records,
                    RecordKey(key),
                    (JobRef::named(kind, source), input, priority),
                    (Want::Make, Stage::Waiting { batch, admitted }),
                ),
                JobOrder::Fetch {
                    key,
                    kind,
                    source,
                    input,
                    priority,
                } => self.want(
                    records,
                    RecordKey(key),
                    (JobRef::named(kind, source), input, priority),
                    (
                        Want::FetchOrMake,
                        Stage::Waiting {
                            batch,
                            admitted: false,
                        },
                    ),
                ),
                JobOrder::Look {
                    key,
                    kind,
                    source,
                    priority,
                } => self.want(
                    records,
                    RecordKey(key),
                    (JobRef::named(kind, source), Vec::new(), priority),
                    (
                        Want::Fetch,
                        Stage::Waiting {
                            batch,
                            admitted: false,
                        },
                    ),
                ),
                JobOrder::Share { key, bytes } => self.share(RecordKey(key), bytes),
                JobOrder::ShareRecord { key } => {
                    if let Some(bytes) = records.get(&RecordKey(key)).and_then(Request::ready) {
                        let bytes = Arc::clone(bytes);
                        self.share(RecordKey(key), bytes);
                    }
                }
                JobOrder::Unshare { key } => self.unshare(RecordKey(key)),
                JobOrder::Keep {
                    key,
                    kind,
                    source,
                    bytes,
                    parity,
                } => {
                    let key = RecordKey(key);
                    if let Some(ticket) = records.ticket(&key) {
                        records.ready(&key, ticket, Arc::from(bytes.as_slice()));
                        self.abandon(key);
                    }
                    self.keep_with(key, Some(JobRef::named(kind, source)), bytes, parity);
                }
            }
        }
        self.admit(&firsts);
        self.land(records);
    }

    fn want(
        &mut self,
        records: &Requests<RecordKey, Arc<[u8]>>,
        key: RecordKey,
        (kind, input, priority): (JobRef, Vec<u8>, u32),
        (want, stage): (Want, Stage),
    ) {
        let Some(ticket) = records.ticket(&key) else {
            return;
        };
        if self.work.iter().any(|flight| flight.key == key) {
            return;
        }
        self.work.push(Flight {
            key,
            ticket,
            kind,
            input,
            priority,
            want,
            stage,
            asked: crate::profile::now_ms(),
        });
    }

    fn admit(&mut self, firsts: &[JobBatch]) {
        *self.admitted.get_mut() = 0;
        let mut budgets: Vec<(JobBatch, JobBudget)> = Vec::new();
        for at in 0..self.work.len() {
            let Stage::Waiting { batch, admitted } = self.work[at].stage else {
                continue;
            };
            if !admitted {
                let batch = if batch.run == LEFT_OVER {
                    firsts
                        .iter()
                        .copied()
                        .find(|first| first.owner == batch.owner)
                        .unwrap_or(batch)
                } else {
                    batch
                };
                let slot = match budgets.iter().position(|(held, _)| *held == batch) {
                    Some(slot) => slot,
                    None => {
                        budgets.push((batch, JobBudget::of(self.workers, self.running())));
                        budgets.len() - 1
                    }
                };
                if !budgets[slot].1.take(self.work[at].priority) {
                    self.work[at].stage = Stage::Waiting {
                        batch: JobBatch {
                            run: LEFT_OVER,
                            ..batch
                        },
                        admitted: false,
                    };
                    continue;
                }
            }
            self.begin(at);
        }
    }

    fn begin(&mut self, at: usize) {
        let (kind, key, priority, want) = {
            let flight = &self.work[at];
            (flight.kind.clone(), flight.key, flight.priority, flight.want)
        };
        let stage = match want {
            Want::Fetch | Want::FetchOrMake => {
                Stage::Fetching(self.fetch(Some(kind), key, priority))
            }
            Want::Make => {
                let input = std::mem::take(&mut self.work[at].input);
                Stage::Making(self.submit_record(kind, input, priority, key))
            }
        };
        let flight = &mut self.work[at];
        flight.stage = stage;
        flight.asked = crate::profile::now_ms();
    }

    fn abandon(&mut self, key: RecordKey) {
        let Some(at) = self.work.iter().position(|flight| flight.key == key) else {
            return;
        };
        let flight = self.work.remove(at);
        match flight.stage {
            Stage::Fetching(job) | Stage::Making(job) => self.cancel(job),
            Stage::Waiting { .. } => {}
        }
    }

    fn land(&mut self, records: &mut Requests<RecordKey, Arc<[u8]>>) {
        let mut at = 0;
        while at < self.work.len() {
            let (Stage::Fetching(job) | Stage::Making(job)) = self.work[at].stage else {
                at += 1;
                continue;
            };
            let Some(ended) = self.ended(job) else {
                at += 1;
                continue;
            };
            let flight = &mut self.work[at];
            let fetched = matches!(flight.stage, Stage::Fetching(_));
            if fetched && flight.want == Want::FetchOrMake && ended.ready().is_none() {
                let input = std::mem::take(&mut flight.input);
                let (kind, priority, key) = (flight.kind.clone(), flight.priority, flight.key);
                let job = self.submit_record(kind, input, priority, key);
                self.work[at].stage = Stage::Making(job);
                continue;
            }
            let flight = self.work.remove(at);
            crate::profile::waited(flight.kind.name(), flight.asked);
            match ended {
                Request::Ready(bytes) => {
                    records.ready(&flight.key, flight.ticket, bytes);
                }
                Request::Missing => {
                    records.miss(&flight.key, flight.ticket);
                }
                Request::Failed(error) => {
                    let error = Error::new(
                        ErrorCode::InvalidRecord,
                        format!("$.jobs.{}", flight.kind.name()),
                    )
                    .got(message(&error));
                    records.fail(&flight.key, flight.ticket, error);
                }
                Request::Asked(_) => {}
            }
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
            .is_none_or(|exchange| exchange.work.is_empty())
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

fn next_id() -> JobId {
    JobId(SUBMITTED.fetch_add(1, Ordering::Relaxed) + 1)
}

pub fn job_output<T, B: AsRef<[u8]>, F: std::fmt::Display, E: std::fmt::Display>(
    kind: &str,
    output: Result<B, F>,
    decode: impl FnOnce(&[u8]) -> Result<T, E>,
) -> Result<T, String> {
    let bytes = output.map_err(|error| format!("{kind}: {error}"))?;
    decode(bytes.as_ref()).map_err(|error| format!("{kind}: its output does not decode: {error}"))
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
