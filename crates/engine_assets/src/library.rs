use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use bevy_ecs::prelude::Resource;
use engine_codec::{Error, ErrorCode, Expected};
use engine_math::collections::{Cursor, Feed, Since};
use serde::Serialize;

static NEXT_LOAD: AtomicU32 = AtomicU32::new(1);

fn next_load() -> u32 {
    NEXT_LOAD.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssetId {
    slot: u32,
    generation: u32,
}

impl AssetId {
    fn slot(self) -> usize {
        self.slot as usize
    }
}

pub struct Asset<T> {
    id: AssetId,
    of: PhantomData<fn() -> T>,
}

impl<T> Asset<T> {
    fn at(id: AssetId) -> Self {
        Self {
            id,
            of: PhantomData,
        }
    }

    pub fn id(self) -> AssetId {
        self.id
    }

    pub fn slot(self) -> usize {
        self.id.slot()
    }
}

impl<T> Clone for Asset<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Asset<T> {}

impl<T> PartialEq for Asset<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for Asset<T> {}

impl<T> PartialOrd for Asset<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Asset<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl<T> std::hash::Hash for Asset<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T> std::fmt::Debug for Asset<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("Asset").field(&self.id).finish()
    }
}

impl<T> From<Asset<T>> for AssetId {
    fn from(asset: Asset<T>) -> Self {
        asset.id
    }
}

impl<T> engine_settings::Setting for Asset<T> {}

impl<T: 'static> serde::Serialize for Asset<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let name = SCOPE.with(|scope| match scope.borrow().as_ref() {
            Some(Scope::Named(names)) => names.name_of(TypeId::of::<T>(), self.id),
            Some(Scope::Collect { names, .. }) => names.name_of(TypeId::of::<T>(), self.id),
            None => Err(outside()),
        });
        let name =
            name.map_err(|error| <S::Error as serde::ser::Error>::custom(error.to_string()))?;
        serializer.serialize_str(&name)
    }
}

impl<'de, T: 'static> serde::Deserialize<'de> for Asset<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = <String as serde::Deserialize>::deserialize(deserializer)?;
        let of = TypeId::of::<T>();
        let id = SCOPE.with(|scope| match scope.borrow_mut().as_mut() {
            Some(Scope::Named(names)) => names.find(of, &name),
            Some(Scope::Collect {
                names,
                folder,
                read,
            }) => {
                let id = names.take(of, &name, folder)?;
                read.push(id.slot);
                Ok(id)
            }
            None => Err(outside()),
        });
        id.map(Self::at)
            .map_err(|error| <D::Error as serde::de::Error>::custom(error.to_string()))
    }
}

impl<T: 'static> schemars::JsonSchema for Asset<T> {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Asset".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "minLength": 1,
        })
    }
}

fn outside() -> Error {
    Error::new(ErrorCode::UnknownReference, "$").expected(Expected::Live)
}

type Value = Arc<dyn Any + Send + Sync>;
type Parse = Arc<dyn Fn(Arc<[u8]>, &str) -> Result<Value, Error> + Send + Sync>;

#[derive(Clone)]
pub struct AssetKind {
    name: &'static str,
    of: TypeId,
    parse: Parse,
    declared: bool,
}

impl AssetKind {
    pub fn of<T: Send + Sync + 'static>(
        name: &'static str,
        parse: fn(Arc<[u8]>, &str) -> Result<T, Error>,
    ) -> Self {
        Self::new(name, parse, false)
    }

    pub fn declared<T: Send + Sync + 'static>(
        name: &'static str,
        parse: fn(Arc<[u8]>, &str) -> Result<T, Error>,
    ) -> Self {
        Self::new(name, parse, true)
    }

    fn new<T: Send + Sync + 'static>(
        name: &'static str,
        parse: fn(Arc<[u8]>, &str) -> Result<T, Error>,
        declared: bool,
    ) -> Self {
        Self {
            name,
            of: TypeId::of::<T>(),
            parse: Arc::new(move |bytes, path| {
                parse(bytes, path).map(|value| Arc::new(value) as Value)
            }),
            declared,
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }
}

#[derive(Clone, Default)]
struct Names {
    generation: u32,
    slots: Vec<(TypeId, String)>,
    index: BTreeMap<TypeId, BTreeMap<String, u32>>,
    uris: BTreeSet<TypeId>,
}

impl Names {
    fn id(&self, slot: u32) -> AssetId {
        AssetId {
            slot,
            generation: self.generation,
        }
    }

    fn slot_of(&self, of: TypeId, name: &str) -> Option<u32> {
        self.index.get(&of)?.get(name).copied()
    }

    fn name_of(&self, of: TypeId, id: AssetId) -> Result<String, Error> {
        self.slots
            .get(id.slot())
            .filter(|(known, _)| id.generation == self.generation && *known == of)
            .map(|(_, name)| name.clone())
            .ok_or_else(outside)
    }

    fn find(&self, of: TypeId, name: &str) -> Result<AssetId, Error> {
        if let Some(slot) = self.slot_of(of, name) {
            return Ok(self.id(slot));
        }
        if !self.uris.contains(&of) && !engine_settings::is_id(name) {
            return Err(Error::new(ErrorCode::InvalidId, "$")
                .expected(Expected::Id)
                .at_most(engine_settings::MAX_ID_BYTES as u64)
                .got(name));
        }
        Err(Error::new(ErrorCode::UnknownReference, "$")
            .got(name)
            .variants(
                self.slots
                    .iter()
                    .filter(|(known, _)| *known == of)
                    .map(|(_, known)| known.clone()),
            ))
    }

    fn take(&mut self, of: TypeId, name: &str, folder: &str) -> Result<AssetId, Error> {
        if !self.uris.contains(&of) {
            return self.find(of, name);
        }
        let uri = format!("{folder}{name}");
        engine_settings::valid_uri(&uri, "$")?;
        let slot = self.slot(of, uri);
        Ok(self.id(slot))
    }

    fn slot(&mut self, of: TypeId, name: String) -> u32 {
        if let Some(slot) = self.slot_of(of, &name) {
            return slot;
        }
        let slot = self.slots.len() as u32;
        self.slots.push((of, name.clone()));
        self.index.entry(of).or_default().insert(name, slot);
        slot
    }
}

fn unsupported(name: &str) -> Error {
    Error::new(ErrorCode::UnknownReference, "$")
        .expected(Expected::Supported)
        .got(name)
}

enum Scope {
    Named(Arc<Names>),
    Collect {
        names: Names,
        folder: String,
        read: Vec<u32>,
    },
}

thread_local! {
    static SCOPE: RefCell<Option<Scope>> = const { RefCell::new(None) };
}

#[doc(hidden)]
#[derive(Clone)]
pub struct AssetNames(Arc<Names>);

impl AssetNames {
    pub fn scope<R>(&self, run: impl FnOnce() -> R) -> R {
        within(Scope::Named(Arc::clone(&self.0)), run).0
    }
}

struct Restore(Option<Scope>);

impl Drop for Restore {
    fn drop(&mut self) {
        let previous = self.0.take();
        SCOPE.with(|scope| *scope.borrow_mut() = previous);
    }
}

fn within<R>(scope: Scope, run: impl FnOnce() -> R) -> (R, Option<Scope>) {
    let restore = Restore(SCOPE.with(|held| held.borrow_mut().replace(scope)));
    let result = run();
    let ours = SCOPE.with(|held| held.borrow_mut().take());
    drop(restore);
    (result, ours)
}

enum State {
    Waiting,
    Ready(Value),
    Refused(Error),
}

const UNREAD: usize = usize::MAX;

struct Entry {
    file: Option<String>,
    needs: Vec<AssetId>,
    state: State,
    revision: u64,
    group: usize,
    bytes: u64,
}

impl Entry {
    fn waiting(file: Option<String>) -> Self {
        Self {
            file,
            needs: Vec::new(),
            state: State::Waiting,
            revision: 0,
            group: UNREAD,
            bytes: 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Landing {
    Waiting,
    Compiling,
    Ready,
}

struct Group {
    name: String,
    landing: Landing,
}

struct Membership {
    joined: Vec<usize>,
    filed: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "dev", derive(schemars::JsonSchema))]
pub struct GroupReport {
    pub name: String,
    pub files: u64,
    pub files_in: u64,
    pub bytes_in: u64,
    pub stage: GroupStage,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "dev", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum GroupStage {
    Arriving,
    Compiling,
    Ready,
    Failed(Error),
}

const ONE_GROUP: &str = "all";

#[derive(Resource)]
pub struct Assets {
    names: Arc<Names>,
    entries: Vec<Entry>,
    kinds: Vec<AssetKind>,
    groups: Vec<Group>,
    membership: OnceLock<Membership>,
    revision: u64,
    changes: Feed<AssetId>,
}

impl Default for Assets {
    fn default() -> Self {
        Self {
            names: Arc::new(Names {
                generation: next_load(),
                ..Names::default()
            }),
            entries: Vec::new(),
            kinds: Vec::new(),
            groups: vec![Group {
                name: ONE_GROUP.to_owned(),
                landing: Landing::Waiting,
            }],
            membership: OnceLock::new(),
            revision: 0,
            changes: Feed::new(1),
        }
    }
}

impl Assets {
    fn entry(&self, id: AssetId) -> Option<&Entry> {
        self.entries
            .get(id.slot())
            .filter(|_| id.generation == self.names.generation)
    }

    fn typed(&self, id: AssetId, of: TypeId) -> Option<&Entry> {
        self.entry(id)
            .filter(|_| self.names.slots[id.slot()].0 == of)
    }

    fn bump(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    fn note(&mut self, slot: usize) -> u64 {
        self.bump();
        self.changes.resize(self.entries.len());
        self.changes.push(self.names.id(slot as u32));
        self.revision
    }

    pub fn get<T: 'static>(&self, asset: Asset<T>) -> Result<&T, Error> {
        let entry = self
            .typed(asset.id, TypeId::of::<T>())
            .ok_or_else(outside)?;
        match &entry.state {
            State::Ready(value) => value.downcast_ref::<T>().ok_or_else(outside),
            State::Waiting => Err(Error::new(ErrorCode::MissingFile, "$")
                .expected(Expected::Present)
                .got(&self.names.slots[asset.slot()].1)),
            State::Refused(error) => Err(error.clone()),
        }
    }

    pub fn ready(&self, id: impl Into<AssetId>) -> bool {
        self.entry(id.into()).is_some_and(|entry| {
            matches!(entry.state, State::Ready(_))
                && entry.needs.iter().all(|&need| self.ready(need))
        })
    }

    pub fn refusal(&self, id: impl Into<AssetId>) -> Option<&Error> {
        let entry = self.entry(id.into())?;
        match &entry.state {
            State::Refused(error) => Some(error),
            State::Waiting | State::Ready(_) => {
                entry.needs.iter().find_map(|&need| self.refusal(need))
            }
        }
    }

    pub fn revision_of(&self, id: impl Into<AssetId>) -> Option<u64> {
        self.entry(id.into()).map(|entry| entry.revision)
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn changed(&self, cursor: &mut Cursor) -> Since<'_, AssetId> {
        self.changes.since(cursor)
    }

    pub fn asset<T: 'static>(&self, id: AssetId) -> Option<Asset<T>> {
        self.typed(id, TypeId::of::<T>()).map(|_| Asset::at(id))
    }

    pub fn named<T: 'static>(&self, name: &str) -> Option<Asset<T>> {
        self.names
            .slot_of(TypeId::of::<T>(), name)
            .map(|slot| Asset::at(self.names.id(slot)))
    }

    pub fn name(&self, id: impl Into<AssetId>) -> Option<&str> {
        let id = id.into();
        self.entry(id)
            .map(|_| self.names.slots[id.slot()].1.as_str())
    }

    pub fn uri(&self, id: impl Into<AssetId>) -> Option<&str> {
        self.entry(id.into())?.file.as_deref()
    }

    pub fn each<T: 'static>(&self) -> impl Iterator<Item = (Asset<T>, &T)> + '_ {
        let of = TypeId::of::<T>();
        self.entries
            .iter()
            .zip(&self.names.slots)
            .enumerate()
            .filter(move |(_, (_, (known, _)))| *known == of)
            .filter_map(|(slot, (entry, _))| match &entry.state {
                State::Ready(value) => value
                    .downcast_ref::<T>()
                    .map(|value| (Asset::at(self.names.id(slot as u32)), value)),
                State::Waiting | State::Refused(_) => None,
            })
    }

    pub fn ids(&self) -> impl Iterator<Item = AssetId> + '_ {
        (0..self.entries.len() as u32).map(|slot| self.names.id(slot))
    }

    #[doc(hidden)]
    pub fn kind(&mut self, kind: AssetKind) -> Result<(), Error> {
        if self.kinds.iter().any(|known| known.of == kind.of) {
            return Err(
                Error::new(ErrorCode::NameTwice, format!("$.asset_kinds.{}", kind.name))
                    .got(kind.name),
            );
        }
        if !kind.declared {
            Arc::make_mut(&mut self.names).uris.insert(kind.of);
        }
        self.kinds.push(kind);
        Ok(())
    }

    fn moved(&mut self) {
        self.membership.take();
    }

    #[doc(hidden)]
    #[must_use]
    pub fn fresh(&self) -> Self {
        Self {
            names: Arc::new(Names {
                generation: next_load(),
                uris: self.names.uris.clone(),
                ..Names::default()
            }),
            entries: Vec::new(),
            kinds: self.kinds.clone(),
            groups: self
                .groups
                .iter()
                .map(|group| Group {
                    name: group.name.clone(),
                    landing: Landing::Waiting,
                })
                .collect(),
            membership: OnceLock::new(),
            revision: self.revision.wrapping_add(1),
            changes: Feed::new(1),
        }
    }

    #[doc(hidden)]
    pub fn declare<T: 'static>(
        &mut self,
        name: &str,
        file: Option<&str>,
    ) -> Result<Asset<T>, Error> {
        let of = TypeId::of::<T>();
        if self.names.slot_of(of, name).is_some() {
            return Err(Error::new(ErrorCode::NameTwice, "$").got(name));
        }
        let kind = self.kinds.iter().find(|kind| kind.of == of);
        if kind.is_some_and(|kind| !kind.declared) || (file.is_some() && kind.is_none()) {
            return Err(unsupported(std::any::type_name::<T>()));
        }
        let slot = Arc::make_mut(&mut self.names).slot(of, name.to_owned());
        self.entries.push(Entry::waiting(file.map(str::to_owned)));
        self.moved();
        self.note(slot as usize);
        Ok(Asset::at(self.names.id(slot)))
    }

    #[doc(hidden)]
    pub fn make<T: Send + Sync + 'static>(
        &mut self,
        asset: Asset<T>,
        value: T,
        needs: Vec<AssetId>,
    ) -> Result<(), Error> {
        if self.typed(asset.id, TypeId::of::<T>()).is_none() {
            return Err(outside());
        }
        let revision = self.note(asset.slot());
        let entry = &mut self.entries[asset.slot()];
        entry.state = State::Ready(Arc::new(value));
        entry.needs = needs;
        entry.revision = revision;
        self.moved();
        Ok(())
    }

    #[doc(hidden)]
    pub fn collect<R>(&mut self, group: usize, folder: &str, run: impl FnOnce() -> R) -> R {
        let names = std::mem::take(Arc::make_mut(&mut self.names));
        let scope = Scope::Collect {
            names,
            folder: folder.to_owned(),
            read: Vec::new(),
        };
        let (result, back) = within(scope, run);
        let Some(Scope::Collect { names, read, .. }) = back else {
            return result;
        };
        let known = self.entries.len();
        for (_, uri) in &names.slots[self.entries.len()..] {
            self.entries.push(Entry::waiting(Some(uri.clone())));
        }
        *Arc::make_mut(&mut self.names) = names;
        for slot in read {
            let entry = &mut self.entries[slot as usize];
            entry.group = entry.group.min(group);
        }
        self.moved();
        for slot in known..self.entries.len() {
            self.note(slot);
        }
        result
    }

    #[doc(hidden)]
    pub fn scope<R>(&self, run: impl FnOnce() -> R) -> R {
        self.names().scope(run)
    }

    #[doc(hidden)]
    pub fn names(&self) -> AssetNames {
        AssetNames(Arc::clone(&self.names))
    }

    fn membership(&self) -> &Membership {
        self.membership.get_or_init(|| {
            let joined = self.joined();
            let mut filed: Vec<(usize, usize)> = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.file.is_some())
                .map(|(slot, _)| (joined[slot], slot))
                .collect();
            filed.sort_unstable();
            let mut seen = BTreeSet::new();
            filed.retain(|&(_, slot)| seen.insert(self.entries[slot].file.as_deref()));
            Membership { joined, filed }
        })
    }

    fn joined(&self) -> Vec<usize> {
        let last = self.groups.len() - 1;
        let mut joined: Vec<usize> = self
            .entries
            .iter()
            .map(|entry| entry.group.min(last))
            .collect();
        let mut moved = true;
        while moved {
            moved = false;
            for (slot, entry) in self.entries.iter().enumerate() {
                let group = joined[slot];
                for need in &entry.needs {
                    if self.entry(*need).is_some() && joined[need.slot()] > group {
                        joined[need.slot()] = group;
                        moved = true;
                    }
                }
            }
        }
        joined
    }

    #[doc(hidden)]
    pub fn files(&self) -> Vec<(&str, &str)> {
        self.membership()
            .filed
            .iter()
            .filter_map(|&(group, slot)| {
                let entry = &self.entries[slot];
                let uri = entry
                    .file
                    .as_deref()
                    .filter(|_| matches!(entry.state, State::Waiting))?;
                Some((uri, self.groups[group].name.as_str()))
            })
            .collect()
    }

    #[doc(hidden)]
    pub fn file(&mut self, uri: &str, read: Result<Arc<[u8]>, Error>) -> Result<(), Error> {
        let taking: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.file.as_deref() == Some(uri) && matches!(entry.state, State::Waiting)
            })
            .map(|(slot, _)| slot)
            .collect();
        if taking.is_empty() {
            return Err(Error::new(ErrorCode::UnknownReference, "$.uri").got(uri));
        }
        for slot in taking {
            let of = self.names.slots[slot].0;
            let kind = self.kinds.iter().find(|kind| kind.of == of);
            let state = match (&read, kind) {
                (Err(error), _) => State::Refused(error.clone()),
                (Ok(bytes), Some(kind)) => match (kind.parse)(Arc::clone(bytes), uri) {
                    Ok(value) => State::Ready(value),
                    Err(error) => State::Refused(error),
                },
                (Ok(_), None) => State::Refused(unsupported(uri)),
            };
            let revision = self.note(slot);
            let entry = &mut self.entries[slot];
            entry.state = state;
            entry.revision = revision;
            entry.bytes = read.as_ref().map_or(0, |bytes| bytes.len() as u64);
        }
        Ok(())
    }

    #[doc(hidden)]
    pub fn name_groups(&mut self, names: Vec<String>) {
        let names = if names.is_empty() {
            vec![ONE_GROUP.to_owned()]
        } else {
            names
        };
        self.groups = names
            .into_iter()
            .map(|name| Group {
                name,
                landing: Landing::Waiting,
            })
            .collect();
        self.moved();
        self.bump();
    }

    pub fn groups(&self) -> Vec<GroupReport> {
        (0..self.groups.len()).map(|at| self.report(at)).collect()
    }

    pub fn group(&self, name: &str) -> Option<GroupReport> {
        let at = self.groups.iter().position(|group| group.name == name)?;
        Some(self.report(at))
    }

    fn report(&self, at: usize) -> GroupReport {
        let Membership { joined, filed } = self.membership();
        let group = &self.groups[at];
        let files = filed
            .iter()
            .filter(|(of, _)| *of == at)
            .map(|&(_, slot)| &self.entries[slot]);
        let arrived = files
            .clone()
            .filter(|entry| !matches!(entry.state, State::Waiting));
        let stage = match group.landing {
            Landing::Compiling => GroupStage::Compiling,
            Landing::Ready => GroupStage::Ready,
            Landing::Waiting => self
                .entries
                .iter()
                .zip(joined)
                .filter(|(_, of)| **of == at)
                .find_map(|(entry, _)| match &entry.state {
                    State::Refused(error) => Some(GroupStage::Failed(error.clone())),
                    State::Waiting | State::Ready(_) => None,
                })
                .unwrap_or(GroupStage::Arriving),
        };
        GroupReport {
            name: group.name.clone(),
            files: files.count() as u64,
            files_in: arrived.clone().count() as u64,
            bytes_in: arrived.map(|entry| entry.bytes).sum(),
            stage,
        }
    }

    #[doc(hidden)]
    pub fn arrived(&self, group: usize) -> bool {
        self.membership()
            .joined
            .iter()
            .enumerate()
            .filter(|(_, of)| **of == group)
            .all(|(slot, _)| self.ready(self.names.id(slot as u32)))
    }

    #[doc(hidden)]
    pub fn landed(&mut self, group: usize) {
        if let Some(group) = self.groups.get_mut(group) {
            group.landing = Landing::Compiling;
            self.revision = self.revision.wrapping_add(1);
        }
    }

    #[doc(hidden)]
    pub fn compiling(&self) -> bool {
        self.groups
            .iter()
            .any(|group| group.landing == Landing::Compiling)
    }

    #[doc(hidden)]
    pub fn compiled(&mut self) {
        for group in &mut self.groups {
            if group.landing == Landing::Compiling {
                group.landing = Landing::Ready;
            }
        }
        self.bump();
    }
}
