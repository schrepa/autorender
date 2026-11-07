use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;
use std::hash::Hash;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde::{Serialize, Serializer};

use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Ticket(u64);

impl Ticket {
    #[doc(hidden)]
    pub const fn numbered(number: u64) -> Self {
        Self(number)
    }

    pub fn number(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Progress {
    pub done: u64,
    pub total: u64,
}

impl Progress {
    pub const fn of(done: u64, total: u64) -> Self {
        Self { done, total }
    }

    #[must_use]
    pub fn and(self, other: Self) -> Self {
        Self {
            done: self.done.saturating_add(other.done),
            total: self.total.saturating_add(other.total),
        }
    }

    pub fn left(self) -> u64 {
        self.total.saturating_sub(self.done)
    }

    fn asked(progress: Option<Self>) -> Self {
        let progress = progress.unwrap_or_default();
        let total = progress.total.max(1);
        Self {
            done: progress.done.min(total - 1),
            total,
        }
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
    Ask(K),
    Release(K),
}

type Changes<K> = Arc<Mutex<Vec<Change<K>>>>;

fn changes<K>(held: &Mutex<Vec<Change<K>>>) -> MutexGuard<'_, Vec<Change<K>>> {
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
        changes(&self.changes).push(Change::Ask(self.key.clone()));
        Self {
            key: self.key.clone(),
            changes: Arc::clone(&self.changes),
        }
    }
}

impl<K: Clone> Drop for Interest<K> {
    fn drop(&mut self) {
        changes(&self.changes).push(Change::Release(self.key.clone()));
    }
}

impl<K: Clone + fmt::Debug> fmt::Debug for Interest<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Interest").field(&self.key).finish()
    }
}

struct Held<V> {
    request: Request<V>,
    interests: u32,
    progress: Option<Progress>,
    since: f64,
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

pub struct Requests<K: Clone, V> {
    entries: HashMap<K, Held<V>>,
    changes: Changes<K>,
    tickets: u64,
    now: f64,
}

impl<K: Clone, V> Default for Requests<K, V> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            changes: Arc::new(Mutex::new(Vec::new())),
            tickets: 0,
            now: 0.0,
        }
    }
}

impl<K: Clone, V> fmt::Debug for Requests<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Requests")
            .field("entries", &self.entries.len())
            .finish()
    }
}

impl<K: Clone + Eq + Hash, V> Requests<K, V> {
    pub fn ask(&mut self, key: K) -> Interest<K> {
        self.count(key.clone());
        Interest {
            key,
            changes: Arc::clone(&self.changes),
        }
    }

    pub fn ask_later(&self, key: K) -> Interest<K> {
        changes(&self.changes).push(Change::Ask(key.clone()));
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
        self.tickets += 1;
        let ticket = Ticket(self.tickets);
        held.request = Request::Asked(ticket);
        held.progress = None;
        held.since = self.now;
        Some(ticket)
    }

    fn count(&mut self, key: K) {
        match self.entries.entry(key) {
            Entry::Occupied(mut held) => {
                held.get_mut().interests = held.get().interests.saturating_add(1);
            }
            Entry::Vacant(free) => {
                self.tickets += 1;
                free.insert(Held {
                    request: Request::Asked(Ticket(self.tickets)),
                    interests: 1,
                    progress: None,
                    since: self.now,
                });
            }
        }
    }

    pub fn settle(&mut self, now: f64) -> Vec<(K, Request<V>)> {
        self.now = now;
        let queued = std::mem::take(&mut *changes(&self.changes));
        let mut released = Vec::new();
        for change in queued {
            match change {
                Change::Ask(key) => self.count(key),
                Change::Release(key) => {
                    let Some(held) = self.entries.get_mut(&key) else {
                        continue;
                    };
                    held.interests = held.interests.saturating_sub(1);
                    if held.interests == 0
                        && let Some(held) = self.entries.remove(&key)
                    {
                        released.push((key, held.request));
                    }
                }
            }
        }
        released
    }

    pub fn changed(&self) -> bool {
        !changes(&self.changes).is_empty()
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

    pub fn interests(&self, key: &K) -> u32 {
        self.entries.get(key).map_or(0, |held| held.interests)
    }

    pub fn progress_of(&self, key: &K) -> Option<Progress> {
        self.entries.get(key)?.progress
    }

    pub fn progress(&mut self, key: &K, progress: Progress) {
        if let Some(held) = self.entries.get_mut(key)
            && matches!(held.request, Request::Asked(_))
        {
            held.progress = Some(progress);
        }
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
        held.request = end;
        held.progress = None;
        true
    }

    pub fn asked(&self) -> impl Iterator<Item = (&K, Ticket)> + '_ {
        self.entries
            .iter()
            .filter_map(|(key, held)| Some((key, held.request.asked()?)))
    }

    pub fn failures(&self) -> impl Iterator<Item = (&K, &Error)> + '_ {
        self.entries.iter().filter_map(|(key, held)| match &held.request {
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
                    census.loading = census.loading.and(Progress::asked(held.progress));
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

impl<K: Clone + Eq + Hash, V> Serialize for Requests<K, V> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.census().serialize(serializer)
    }
}

#[cfg(feature = "schema")]
impl<K: Clone, V> schemars::JsonSchema for Requests<K, V> {
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
