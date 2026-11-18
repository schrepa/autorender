use std::cell::{Cell, RefCell};
use std::collections::hash_map::Entry;
use std::fmt;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use engine_math::collections::HashMap;
use serde::{Serialize, Serializer};

use crate::readiness::{Awaits, CORE_ROW};
use crate::{Error, Readiness};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Ticket(u64);

static TICKETS: AtomicU64 = AtomicU64::new(0);

impl Ticket {
    #[doc(hidden)]
    pub const fn numbered(number: u64) -> Self {
        Self(number)
    }

    pub fn number(self) -> u64 {
        self.0
    }

    fn next() -> Self {
        Self(TICKETS.fetch_add(1, Ordering::Relaxed) + 1)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct Progress {
    pub done: u64,
    pub total: u64,
}

impl Progress {
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        Self {
            done: self.done.saturating_add(other.done),
            total: self.total.saturating_add(other.total),
        }
    }

    fn less(self, other: Self) -> Self {
        Self {
            done: self.done.saturating_sub(other.done),
            total: self.total.saturating_sub(other.total),
        }
    }

    pub fn left(self) -> u64 {
        self.total.saturating_sub(self.done)
    }

    const fn asked() -> Self {
        Self { done: 0, total: 1 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Request<V> {
    Asked(Ticket),
    Ready(V),
    Failed(Error),
    Missing,
}

impl<V> Request<V> {
    pub fn ready(&self) -> Option<&V> {
        match self {
            Self::Ready(value) => Some(value),
            Self::Asked(_) | Self::Failed(_) | Self::Missing => None,
        }
    }

    pub fn asked(&self) -> Option<Ticket> {
        match self {
            Self::Asked(ticket) => Some(*ticket),
            Self::Ready(_) | Self::Failed(_) | Self::Missing => None,
        }
    }
}

enum Change<K> {
    Ask(K, engine_profile::Carried),
    Release(K),
}

type Changes<K> = Arc<Mutex<Vec<Change<K>>>>;

fn lock<T>(held: &Mutex<T>) -> MutexGuard<'_, T> {
    held.lock().unwrap_or_else(PoisonError::into_inner)
}

#[must_use]
pub struct Interest<K: Clone> {
    key: K,
    changes: Changes<K>,
}

impl<K: Clone> Interest<K> {
    pub fn key(&self) -> &K {
        &self.key
    }
}

impl<K: Clone> Clone for Interest<K> {
    fn clone(&self) -> Self {
        lock(&self.changes).push(Change::Ask(self.key.clone(), engine_profile::carry()));
        Self {
            key: self.key.clone(),
            changes: Arc::clone(&self.changes),
        }
    }
}

impl<K: Clone> Drop for Interest<K> {
    fn drop(&mut self) {
        lock(&self.changes).push(Change::Release(self.key.clone()));
    }
}

impl<K: Clone + fmt::Debug> fmt::Debug for Interest<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Interest").field(&self.key).finish()
    }
}

#[derive(Default)]
struct Tally {
    loading: Progress,
    settling: u32,
    failed: Vec<(u64, Error)>,
    joined: bool,
    owner: &'static str,
    awaits: Awaits,
}

type Tallied = Arc<Mutex<Tally>>;

#[doc(hidden)]
#[derive(Clone, Default)]
pub struct Board(Arc<Mutex<Vec<Weak<Mutex<Tally>>>>>);

thread_local! {
    static BOARD: RefCell<Option<Board>> = const { RefCell::new(None) };
    static OWNER: Cell<&'static str> = const { Cell::new(CORE_ROW) };
}

struct Owner(&'static str);

impl Drop for Owner {
    fn drop(&mut self) {
        let previous = self.0;
        let _ = OWNER.try_with(|owner| owner.set(previous));
    }
}

fn owner() -> &'static str {
    OWNER.try_with(Cell::get).unwrap_or(CORE_ROW)
}

struct Restore(Option<Board>);

impl Drop for Restore {
    fn drop(&mut self) {
        let previous = self.0.take();
        let _ = BOARD.try_with(|board| *board.borrow_mut() = previous);
    }
}

impl Board {
    pub fn scope<R>(&self, run: impl FnOnce() -> R) -> R {
        let previous = BOARD
            .try_with(|board| board.borrow_mut().replace(self.clone()))
            .ok()
            .flatten();
        let _restore = Restore(previous);
        run()
    }

    pub fn owned_by<R>(brick: &'static str, run: impl FnOnce() -> R) -> R {
        let previous = OWNER
            .try_with(|owner| owner.replace(brick))
            .unwrap_or(CORE_ROW);
        let _restore = Owner(previous);
        run()
    }

    pub fn add_to(&self, readiness: &mut Readiness, viewed: bool) {
        lock(&self.0).retain(|member| {
            let Some(tally) = member.upgrade() else {
                return false;
            };
            let tally = lock(&tally);
            readiness.load(tally.owner, tally.loading);
            match tally.awaits {
                Awaits::View if !viewed => readiness.await_view(tally.owner, tally.settling),
                Awaits::Finished => readiness.finish(tally.owner, tally.settling),
                Awaits::Frames => readiness.await_frames(tally.owner, tally.settling),
                Awaits::Work | Awaits::View => readiness.settle(tally.owner, tally.settling),
            }
            for (_, error) in &tally.failed {
                readiness.fail(tally.owner, error.clone());
            }
            true
        });
    }
}

impl fmt::Debug for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Board").field(&lock(&self.0).len()).finish()
    }
}

fn joined() -> Tallied {
    let tally = Arc::new(Mutex::new(Tally::default()));
    join(&tally);
    tally
}

fn join(tally: &Tallied) {
    if lock(tally).joined {
        return;
    }
    let _ = BOARD.try_with(|board| {
        if let Some(board) = board.borrow().as_ref() {
            lock(&board.0).push(Arc::downgrade(tally));
            let mut joined = lock(tally);
            joined.joined = true;
            joined.owner = owner();
        }
    });
}

pub struct Settling(Tallied);

impl Default for Settling {
    fn default() -> Self {
        Self(joined())
    }
}

impl Settling {
    pub fn on_view() -> Self {
        let settling = Self::default();
        lock(&settling.0).awaits = Awaits::View;
        settling
    }

    pub fn on_finished() -> Self {
        let settling = Self::default();
        lock(&settling.0).awaits = Awaits::Finished;
        settling
    }

    pub fn on_frames() -> Self {
        let settling = Self::default();
        lock(&settling.0).awaits = Awaits::Frames;
        settling
    }

    pub fn left(&self, left: u32) {
        join(&self.0);
        lock(&self.0).settling = left;
    }

    pub fn failed(&self, failed: Option<Error>) {
        join(&self.0);
        lock(&self.0).failed = failed.into_iter().map(|error| (0, error)).collect();
    }
}

impl fmt::Debug for Settling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Settling")
            .field(&lock(&self.0).settling)
            .finish()
    }
}

struct Held<V, A> {
    request: Request<V>,
    interests: u32,
    since: f64,
    serial: u64,
    work: Option<A>,
    asker: engine_profile::Carried,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Census {
    pub asked: u32,
    pub ready: u32,
    pub failed: u32,
    pub missing: u32,
    pub interests: u64,
    pub loading: Progress,
    pub oldest_asked_seconds: f64,
}

pub struct Requests<K: Clone, V, A = ()> {
    entries: HashMap<K, Held<V, A>>,
    asked: HashMap<Ticket, K>,
    changes: Changes<K>,
    tally: Tallied,
    serials: u64,
    now: f64,
}

impl<K: Clone, V, A> Default for Requests<K, V, A> {
    fn default() -> Self {
        Self {
            entries: HashMap::default(),
            asked: HashMap::default(),
            changes: Arc::new(Mutex::new(Vec::new())),
            tally: joined(),
            serials: 0,
            now: 0.0,
        }
    }
}

impl<K: Clone, V, A> fmt::Debug for Requests<K, V, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Requests")
            .field("entries", &self.entries.len())
            .finish()
    }
}

fn left_asked<V, A>(tally: &mut Tally, held: &Held<V, A>) {
    match &held.request {
        Request::Asked(_) => tally.loading = tally.loading.less(Progress::asked()),
        Request::Failed(_) => tally.failed.retain(|(serial, _)| *serial != held.serial),
        Request::Ready(_) | Request::Missing => {}
    }
}

impl<K: Clone + Eq + Hash, V, A> Requests<K, V, A> {
    pub fn ask(&mut self, key: K) -> Interest<K> {
        self.count(key.clone(), engine_profile::carry());
        Interest {
            key,
            changes: Arc::clone(&self.changes),
        }
    }

    pub fn ask_later(&self, key: K) -> Interest<K> {
        lock(&self.changes).push(Change::Ask(key.clone(), engine_profile::carry()));
        Interest {
            key,
            changes: Arc::clone(&self.changes),
        }
    }

    pub fn ask_again(&mut self, key: &K) -> Option<Ticket> {
        let held = self.entries.get_mut(key)?;
        if let Request::Asked(ticket) = held.request {
            return Some(ticket);
        }
        let ticket = Ticket::next();
        let mut tally = lock(&self.tally);
        left_asked(&mut tally, held);
        tally.loading = tally.loading.and(Progress::asked());
        held.request = Request::Asked(ticket);
        held.since = self.now;
        held.work = None;
        self.asked.insert(ticket, key.clone());
        Some(ticket)
    }

    pub fn cause(&self, key: &K) -> engine_profile::Cause {
        self.entries
            .get(key)
            .map_or(engine_profile::Cause(0), |held| held.asker.cause())
    }

    fn count(&mut self, key: K, asker: engine_profile::Carried) {
        match self.entries.entry(key) {
            Entry::Occupied(mut held) => {
                held.get_mut().interests = held.get().interests.saturating_add(1);
            }
            Entry::Vacant(free) => {
                let ticket = Ticket::next();
                self.serials += 1;
                self.asked.insert(ticket, free.key().clone());
                let mut tally = lock(&self.tally);
                tally.loading = tally.loading.and(Progress::asked());
                free.insert(Held {
                    request: Request::Asked(ticket),
                    interests: 1,
                    since: self.now,
                    serial: self.serials,
                    work: None,
                    asker,
                });
            }
        }
    }

    pub fn settle(&mut self, now: f64) -> Vec<(K, Request<V>)> {
        self.settle_work(now)
            .into_iter()
            .map(|(key, request, _)| (key, request))
            .collect()
    }

    pub fn settle_work(&mut self, now: f64) -> Vec<(K, Request<V>, Option<A>)> {
        join(&self.tally);
        self.now = now;
        let queued = std::mem::take(&mut *lock(&self.changes));
        let mut released = Vec::new();
        for change in queued {
            match change {
                Change::Ask(key, asker) => self.count(key, asker),
                Change::Release(key) => {
                    let Some(held) = self.entries.get_mut(&key) else {
                        continue;
                    };
                    held.interests = held.interests.saturating_sub(1);
                    if held.interests == 0
                        && let Some(held) = self.entries.remove(&key)
                    {
                        left_asked(&mut lock(&self.tally), &held);
                        if let Some(ticket) = held.request.asked() {
                            self.asked.remove(&ticket);
                        }
                        released.push((key, held.request, held.work));
                    }
                }
            }
        }
        released
    }

    pub fn changed(&self) -> bool {
        !lock(&self.changes).is_empty()
    }

    pub fn get(&self, key: &K) -> Option<&Request<V>> {
        self.entries.get(key).map(|held| &held.request)
    }

    pub fn of(&self, interest: &Interest<K>) -> Option<&Request<V>> {
        self.get(&interest.key)
    }

    pub fn ticket(&self, key: &K) -> Option<Ticket> {
        self.get(key).and_then(Request::asked)
    }

    pub fn key_of(&self, ticket: Ticket) -> Option<&K> {
        self.asked.get(&ticket)
    }

    pub fn interests(&self, key: &K) -> u32 {
        self.entries.get(key).map_or(0, |held| held.interests)
    }

    pub fn work(&self, key: &K) -> Option<&A> {
        self.entries.get(key)?.work.as_ref()
    }

    pub fn work_mut(&mut self, key: &K) -> Option<&mut A> {
        self.entries.get_mut(key)?.work.as_mut()
    }

    pub fn set_work(&mut self, key: &K, work: A) -> bool {
        match self.entries.get_mut(key) {
            Some(held) if held.request.asked().is_some() => {
                held.work = Some(work);
                true
            }
            Some(_) | None => false,
        }
    }

    pub fn works(&self) -> impl Iterator<Item = (&K, &A)> + '_ {
        self.entries
            .iter()
            .filter_map(|(key, held)| Some((key, held.work.as_ref()?)))
    }

    pub fn ready(&mut self, key: &K, ticket: Ticket, value: V) -> bool {
        self.end(key, ticket, Request::Ready(value))
    }

    pub fn fail(&mut self, key: &K, ticket: Ticket, error: Error) -> bool {
        self.end(key, ticket, Request::Failed(error))
    }

    pub fn miss(&mut self, key: &K, ticket: Ticket) -> bool {
        self.end(key, ticket, Request::Missing)
    }

    fn end(&mut self, key: &K, ticket: Ticket, end: Request<V>) -> bool {
        let Some(held) = self.entries.get_mut(key) else {
            return false;
        };
        if held.request.asked() != Some(ticket) {
            return false;
        }
        let mut tally = lock(&self.tally);
        left_asked(&mut tally, held);
        if let Request::Failed(error) = &end {
            tally.failed.push((held.serial, error.clone()));
        }
        self.asked.remove(&ticket);
        held.request = end;
        held.work = None;
        true
    }

    pub fn asked(&self) -> impl Iterator<Item = (&K, Ticket)> + '_ {
        self.entries
            .iter()
            .filter_map(|(key, held)| Some((key, held.request.asked()?)))
    }

    pub fn failures(&self) -> impl Iterator<Item = (&K, &Error)> + '_ {
        self.entries
            .iter()
            .filter_map(|(key, held)| match &held.request {
                Request::Failed(error) => Some((key, error)),
                Request::Asked(_) | Request::Ready(_) | Request::Missing => None,
            })
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &Request<V>)> + '_ {
        self.entries.iter().map(|(key, held)| (key, &held.request))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn census(&self) -> Census {
        let mut census = Census::default();
        for held in self.entries.values() {
            census.interests += u64::from(held.interests);
            match &held.request {
                Request::Asked(_) => {
                    census.asked += 1;
                    census.loading = census.loading.and(Progress::asked());
                    census.oldest_asked_seconds =
                        census.oldest_asked_seconds.max(self.now - held.since);
                }
                Request::Ready(_) => census.ready += 1,
                Request::Failed(_) => census.failed += 1,
                Request::Missing => census.missing += 1,
            }
        }
        census
    }
}

impl<K: Clone + Eq + Hash, V, A> Serialize for Requests<K, V, A> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.census().serialize(serializer)
    }
}

#[cfg(feature = "schema")]
impl<K: Clone, V, A> schemars::JsonSchema for Requests<K, V, A> {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Census".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Census::json_schema(generator)
    }
}

impl<K: Clone + Serialize> Serialize for Interest<K> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.key.serialize(serializer)
    }
}

#[cfg(feature = "schema")]
impl<K: Clone + schemars::JsonSchema> schemars::JsonSchema for Interest<K> {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> std::borrow::Cow<'static, str> {
        K::schema_name()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        K::json_schema(generator)
    }
}
