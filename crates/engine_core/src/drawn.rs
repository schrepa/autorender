use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use bevy_ecs::change_detection::Mut;
use bevy_ecs::entity::Entity;
use bevy_ecs::lifecycle::{Insert, Remove};
use bevy_ecs::observer::On;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::ResMut;
use bevy_ecs::world::World;
use engine_assets::{Asset, AssetLibrary, attributes_wgsl};
use engine_place::Changes;
use engine_registry::Registry;

use crate::{
    AssetResidency, DeviceLimits, DrawKind, Drawn, Error, ErrorCode, Expected,
    GeometryHandle, GlobalTransforms, GpuStruct, Interest, KindDecl, Layers, MaterialHandle,
    PassMask, Refusals, Request, Transform, gpu_struct, settings,
};

pub const DRAWN_PART: &str = "drawn";

settings! {
    #[section("drawn")]
    pub struct DrawnSettings {
        pub instance_floats: u32 = 16 => range(0, 1024),
    }
}

const SLOTS: &str = "fn drawn_slot(instance: u32) -> u32 {
    return drawn_order[instance];
}
fn drawn_tint(slot: u32) -> vec4<f32> {
    return drawn_values[drawn_records[slot].values];
}
fn drawn_emission(slot: u32) -> vec3<f32> {
    return drawn_values[drawn_records[slot].values + 1u].xyz;
}
fn drawn_floats(slot: u32, quad: u32) -> vec4<f32> {
    return drawn_values[drawn_records[slot].values + 2u + quad];
}
fn drawn_float_count(slot: u32) -> u32 {
    return drawn_records[slot].floats;
}
";

gpu_struct! {
    pub struct DrawnRecord {
        pub node: u32,
        pub values: u32,
        pub floats: u32,
    }
}

pub fn drawn_wgsl() -> String {
    let mut text = String::from(DrawnRecord::WGSL);
    text.push_str(&attributes_wgsl());
    text.push_str(SLOTS);
    text
}

const NO_BATCH: u32 = u32::MAX;
const NO_SLOT: u32 = u32::MAX;
const VALUE_HEAD: usize = 2;

fn stride_of(floats: u32) -> usize {
    VALUE_HEAD + floats.div_ceil(4) as usize
}

#[derive(Clone, Debug)]
pub enum DrawnState {
    Drawn { slot: u32 },
    Hidden,
    Unplaced,
    Waiting,
    Refused(Error),
}

#[derive(Clone, Debug, PartialEq)]
pub struct DrawnBatch {
    pub kind: DrawKind,
    pub order: i16,
    pub geometry: Option<GeometryHandle>,
    pub material: Option<MaterialHandle>,
    pub passes: Option<PassMask>,
    pub layers: Layers,
    pub instances: Range<u32>,
    #[cfg(any(feature = "dev", feature = "profiling"))]
    pub cause: u32,
}

type BatchKey = (
    usize,
    i16,
    Option<GeometryHandle>,
    Option<MaterialHandle>,
    Option<u32>,
    u32,
);

struct Held {
    state: DrawnState,
    drawn: Drawn,
    uses: Vec<Interest<Asset>>,
}

#[derive(Clone, Copy)]
struct Home {
    batch: u32,
    at: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Seen {
    residency: u64,
    library: (u64, u64),
    stride: usize,
    room: u64,
}

struct Context<'a> {
    globals: &'a GlobalTransforms,
    residency: &'a AssetResidency,
    library: &'a AssetLibrary,
    floats: usize,
    room: u64,
}

#[derive(Resource)]
pub struct DrawnTable {
    kinds: Vec<KindDecl>,
    held: BTreeMap<Entity, Held>,
    pending: BTreeSet<Entity>,
    homes: Vec<Home>,
    records: Vec<DrawnRecord>,
    values: Vec<[f32; 4]>,
    free: Vec<u32>,
    order: Vec<u32>,
    batches: Vec<DrawnBatch>,
    keys: Vec<BatchKey>,
    rooms: Vec<u32>,
    keyed: BTreeMap<BatchKey, u32>,
    sorted: Vec<Vec<u32>>,
    spare_batches: Vec<u32>,
    spare_regions: Vec<Vec<u32>>,
    slot_changes: Changes,
    order_changes: Changes,
    touched: Vec<Entity>,
    verdicts: Vec<(Entity, Result<(), Error>)>,
    seen: Option<Seen>,
    stride: usize,
}

impl Default for DrawnTable {
    fn default() -> Self {
        Self {
            kinds: Vec::new(),
            held: BTreeMap::new(),
            pending: BTreeSet::new(),
            homes: Vec::new(),
            records: Vec::new(),
            values: Vec::new(),
            free: Vec::new(),
            order: Vec::new(),
            batches: Vec::new(),
            keys: Vec::new(),
            rooms: Vec::new(),
            keyed: BTreeMap::new(),
            sorted: Vec::new(),
            spare_batches: Vec::new(),
            spare_regions: Vec::new(),
            slot_changes: Changes::new(1),
            order_changes: Changes::new(1),
            touched: Vec::new(),
            verdicts: Vec::new(),
            seen: None,
            stride: stride_of(DrawnSettings::default().instance_floats),
        }
    }
}

impl DrawnTable {
    pub fn kinds(&self) -> &[KindDecl] {
        &self.kinds
    }

    fn kind_at(&self, kind: DrawKind) -> Option<usize> {
        self.kinds.iter().position(|decl| decl.kind == kind)
    }


    pub fn state(&self, entity: Entity) -> Option<&DrawnState> {
        self.held.get(&entity).map(|held| &held.state)
    }

    pub fn slots(&self) -> usize {
        self.records.len()
    }

    pub fn records(&self) -> &[DrawnRecord] {
        &self.records
    }

    pub fn values(&self) -> &[[f32; 4]] {
        &self.values
    }

    pub fn value_vec4s(&self) -> usize {
        self.stride
    }

    pub fn order(&self) -> &[u32] {
        &self.order
    }

    pub fn slot_changes(&self) -> &Changes {
        &self.slot_changes
    }

    pub fn order_changes(&self) -> &Changes {
        &self.order_changes
    }

    pub fn batches(&self, kind: DrawKind) -> impl Iterator<Item = &DrawnBatch> + '_ {
        self.kind_at(kind)
            .and_then(|at| self.sorted.get(at))
            .into_iter()
            .flatten()
            .map(|&batch| &self.batches[batch as usize])
    }

    fn settle(&mut self, world: &World, settings: &DrawnSettings) {
        let (Some(globals), Some(residency), Some(library)) = (
            world.get_resource::<GlobalTransforms>(),
            world.get_resource::<AssetResidency>(),
            world.get_resource::<AssetLibrary>(),
        ) else {
            return;
        };
        self.slot_changes.settle();
        self.order_changes.settle();
        let stride = stride_of(settings.instance_floats);
        let room = world
            .get_resource::<DeviceLimits>()
            .map_or(0, |limits| limits.storage_room);
        let seen = Seen {
            residency: residency.revision(),
            library: library.revision(),
            stride,
            room,
        };
        let mut touched = std::mem::take(&mut self.touched);
        if stride != self.stride {
            self.stride = stride;
            self.values = vec![[0.0; 4]; self.records.len() * stride];
        }
        match self.seen.replace(seen) {
            Some(known) if known == seen => {}
            Some(known)
                if (known.stride, known.room, known.library.0)
                    == (stride, room, seen.library.0) =>
            {
                touched.extend(self.pending.iter().copied());
                if known.library != seen.library {
                    let before = known.library.1;
                    touched.extend(
                        self.held
                            .iter()
                            .filter(|(_, kept)| stale(&kept.drawn, library, before))
                            .map(|(&entity, _)| entity),
                    );
                }
            }
            Some(_) | None => touched.extend(self.held.keys().copied()),
        }
        let placed = globals.placed().read();
        if placed.all {
            touched.extend(self.held.keys().copied());
        } else {
            touched.extend(
                placed
                    .listed
                    .iter()
                    .filter_map(|&node| globals.entity_at(node))
                    .filter(|entity| self.held.contains_key(entity)),
            );
        }
        globals.placed().take(0, placed.upto);
        touched.sort_unstable();
        touched.dedup();
        let cx = Context {
            globals,
            residency,
            library,
            floats: settings.instance_floats as usize,
            room,
        };
        for &entity in &touched {
            match world.get::<Drawn>(entity) {
                Some(drawn) => under(world, entity, || self.refresh(entity, drawn, &cx)),
                None => self.forget(entity),
            }
        }
        touched.clear();
        self.touched = touched;
    }

    fn take_held(&mut self, entity: Entity, drawn: &Drawn, residency: &AssetResidency) -> Held {
        match self.held.remove(&entity) {
            Some(mut kept) => {
                if (kept.drawn.geometry, kept.drawn.material) != (drawn.geometry, drawn.material) {
                    kept.uses = hold(residency, drawn);
                    kept.drawn.geometry = drawn.geometry;
                    kept.drawn.material = drawn.material;
                }
                kept
            }
            None => Held {
                state: DrawnState::Waiting,
                drawn: drawn.clone(),
                uses: hold(residency, drawn),
            },
        }
    }

    fn refresh(&mut self, entity: Entity, drawn: &Drawn, cx: &Context<'_>) {
        let mut kept = self.take_held(entity, drawn, cx.residency);
        let slot = match kept.state {
            DrawnState::Drawn { slot } => Some(slot),
            DrawnState::Hidden
            | DrawnState::Unplaced
            | DrawnState::Waiting
            | DrawnState::Refused(_) => None,
        };
        let state = match (self.refusal(entity, drawn, cx), slot) {
            (None, Some(slot)) => self.place(slot, entity, drawn, cx),
            (None, None) => {
                let slot = self.take();
                if self.beyond(slot, cx.room) {
                    self.release(slot);
                    DrawnState::Refused(
                        Error::new(ErrorCode::BindingLimit, "$.drawn.instance_floats")
                            .at_most(cx.room)
                            .got((u64::from(slot) + 1) * self.slot_bytes()),
                    )
                } else {
                    self.place(slot, entity, drawn, cx)
                }
            }
            (Some(state), Some(slot)) => {
                self.release(slot);
                state
            }
            (Some(state), None) => state,
        };
        let now = refused_of(&state);
        if refused_of(&kept.state) != now {
            self.verdicts
                .push((entity, now.cloned().map_or(Ok(()), Err)));
        }
        match state {
            DrawnState::Waiting | DrawnState::Refused(_) => self.pending.insert(entity),
            DrawnState::Drawn { .. } | DrawnState::Hidden | DrawnState::Unplaced => {
                self.pending.remove(&entity)
            }
        };
        kept.state = state;
        kept.drawn.clone_from(drawn);
        self.held.insert(entity, kept);
    }

    fn place(&mut self, slot: u32, entity: Entity, drawn: &Drawn, cx: &Context<'_>) -> DrawnState {
        self.write(slot, entity, drawn, cx);
        if let Some(kind) = self.kind_at(drawn.kind) {
            let key = (
                kind,
                drawn.order,
                drawn.geometry,
                drawn.material,
                drawn.passes.map(PassMask::word),
                drawn.layers.0,
            );
            self.file(slot, key);
        }
        DrawnState::Drawn { slot }
    }

    fn slot_bytes(&self) -> u64 {
        (size_of::<DrawnRecord>() as u64).max((self.stride * size_of::<[f32; 4]>()) as u64)
    }

    fn beyond(&self, slot: u32, room: u64) -> bool {
        room != 0 && (u64::from(slot) + 1) * self.slot_bytes() > room
    }

    fn refusal(&self, entity: Entity, drawn: &Drawn, cx: &Context<'_>) -> Option<DrawnState> {
        let Some(decl) = self.kind_at(drawn.kind).map(|at| &self.kinds[at]) else {
            return Some(DrawnState::Refused(
                Error::new(ErrorCode::InvalidRenderable, "$.drawn.kind")
                    .got(drawn.kind.name())
                    .variants(self.kinds.iter().map(|decl| decl.kind.name())),
            ));
        };
        if drawn.values.floats.len() > cx.floats {
            return Some(DrawnState::Refused(
                Error::new(ErrorCode::OutOfRange, "$.drawn.values.floats")
                    .at_most(cx.floats as u64)
                    .got(drawn.values.floats.len()),
            ));
        }
        if let Some(own) = drawn.projection
            && !(own.vertical_fov > 0.0
                && own.vertical_fov < std::f32::consts::PI
                && own.depth_scale > 0.0
                && own.depth_scale <= 1.0)
        {
            return Some(DrawnState::Refused(
                Error::new(ErrorCode::OutOfRange, "$.drawn.projection")
                    .got([own.vertical_fov, own.depth_scale]),
            ));
        }
        if drawn.geometry.is_none()
            && (decl.needs != engine_assets::Attributes::NONE || !decl.streams.is_empty())
        {
            return Some(DrawnState::Refused(
                Error::new(ErrorCode::InvalidRenderable, "$.drawn.mesh")
                    .expected(Expected::Present)
                    .got(drawn.kind.name()),
            ));
        }
        if !cx.globals.drawn(entity) {
            return Some(DrawnState::Hidden);
        }
        if cx
            .globals
            .get(entity)
            .and_then(|global| global.matrix())
            .is_none()
        {
            return Some(DrawnState::Unplaced);
        }
        let assets = [
            drawn
                .geometry
                .map(|geometry| (Asset::Mesh(geometry), "$.drawn.mesh")),
            drawn
                .material
                .map(|material| (Asset::Material(material), "$.drawn.material")),
        ];
        for (asset, path) in assets.into_iter().flatten() {
            if let Some(error) = cx.residency.refusal(asset, path) {
                return Some(DrawnState::Refused(error));
            }
            if !matches!(cx.residency.state(asset), Some(Request::Ready(_))) {
                return Some(DrawnState::Waiting);
            }
        }
        if let Some(material) = drawn.material {
            let live = cx.library.material(material).is_some_and(|material| {
                material
                    .textures()
                    .all(|texture| cx.library.texture(texture).is_some())
            });
            if !live {
                return Some(DrawnState::Waiting);
            }
        }
        let geometry = drawn.geometry?;
        let Some(mesh) = cx.library.mesh(geometry) else {
            return Some(DrawnState::Waiting);
        };
        if let Err(error) = mesh.attributes().check(decl.needs, "$.drawn.mesh") {
            return Some(DrawnState::Refused(error));
        }
        decl.streams
            .iter()
            .find(|&&shape| mesh.stream(shape).is_none())
            .map(|missing| {
                let has: Vec<&str> = mesh.streams().iter().map(|stream| stream.name()).collect();
                DrawnState::Refused(
                    Error::new(
                        ErrorCode::InvalidRenderable,
                        format!("$.drawn.mesh.streams.{}", missing.name),
                    )
                    .variants(decl.streams.iter().map(|shape| shape.name))
                    .got(has),
                )
            })
    }

    fn take(&mut self) -> u32 {
        if let Some(slot) = self.free.pop() {
            return slot;
        }
        self.homes.push(Home {
            batch: NO_BATCH,
            at: 0,
        });
        self.records.push(DrawnRecord::default());
        self.values
            .extend(std::iter::repeat_n([0.0; 4], self.stride));
        (self.records.len() - 1) as u32
    }

    fn release(&mut self, slot: u32) {
        self.unfile(slot);
        self.records[slot as usize] = DrawnRecord::default();
        self.free.push(slot);
    }

    fn forget(&mut self, entity: Entity) {
        let Some(gone) = self.held.remove(&entity) else {
            return;
        };
        self.pending.remove(&entity);
        if refused_of(&gone.state).is_some() {
            self.verdicts.push((entity, Ok(())));
        }
        if let DrawnState::Drawn { slot } = gone.state {
            self.release(slot);
        }
    }

    fn write(&mut self, slot: u32, entity: Entity, drawn: &Drawn, cx: &Context<'_>) {
        let at = slot as usize;
        let record = DrawnRecord {
            node: cx
                .globals
                .node_index(entity)
                .map_or(u32::MAX, |index| index as u32),
            values: (at * self.stride) as u32,
            floats: drawn.values.floats.len() as u32,
        };
        let mut changed = self.records[at] != record;
        let [red, green, blue] = drawn.values.emission;
        let floats = &drawn.values.floats;
        for index in 0..self.stride {
            let quad = match index {
                0 => drawn.values.tint,
                1 => [red, green, blue, 0.0],
                _ => std::array::from_fn(|lane| {
                    floats
                        .get((index - VALUE_HEAD) * 4 + lane)
                        .copied()
                        .unwrap_or(0.0)
                }),
            };
            let held = &mut self.values[at * self.stride + index];
            if held.map(f32::to_bits) != quad.map(f32::to_bits) {
                *held = quad;
                changed = true;
            }
        }
        if changed {
            self.records[at] = record;
            self.slot_changes.mark(slot);
        }
    }

    fn file(&mut self, slot: u32, key: BatchKey) {
        let home = self.homes[slot as usize];
        if home.batch != NO_BATCH {
            if self.keys[home.batch as usize] == key {
                return;
            }
            self.unfile(slot);
        }
        let batch = match self.keyed.get(&key) {
            Some(&batch) => batch,
            None => self.open(key),
        };
        let at = batch as usize;
        if self.batches[at].instances.len() as u32 == self.rooms[at] {
            self.grow(batch);
        }
        let end = self.batches[at].instances.end;
        self.order[end as usize] = slot;
        self.order_changes.mark(end);
        self.homes[slot as usize] = Home {
            batch,
            at: end - self.batches[at].instances.start,
        };
        self.batches[at].instances.end += 1;
    }

    fn unfile(&mut self, slot: u32) {
        let Home { batch, at } = std::mem::replace(
            &mut self.homes[slot as usize],
            Home {
                batch: NO_BATCH,
                at: 0,
            },
        );
        if batch == NO_BATCH {
            return;
        }
        let instances = self.batches[batch as usize].instances.clone();
        let (hole, last) = (instances.start + at, instances.end - 1);
        if hole != last {
            let moved = self.order[last as usize];
            self.order[hole as usize] = moved;
            self.homes[moved as usize].at = at;
            self.order_changes.mark(hole);
        }
        self.order[last as usize] = NO_SLOT;
        self.order_changes.mark(last);
        self.batches[batch as usize].instances.end = last;
        if last == instances.start {
            self.close(batch);
        }
    }

    fn open(&mut self, key: BatchKey) -> u32 {
        let start = self.region(1);
        let (kind, order, geometry, material, passes, layers) = key;
        let made = DrawnBatch {
            kind: self.kinds[kind].kind,
            order,
            geometry,
            material,
            passes: passes.map(PassMask),
            layers: Layers(layers),
            instances: start..start,
            #[cfg(any(feature = "dev", feature = "profiling"))]
            cause: engine_profile::cause().0,
        };
        let batch = match self.spare_batches.pop() {
            Some(batch) => {
                let at = batch as usize;
                self.batches[at] = made;
                self.keys[at] = key;
                self.rooms[at] = 1;
                batch
            }
            None => {
                self.batches.push(made);
                self.keys.push(key);
                self.rooms.push(1);
                (self.batches.len() - 1) as u32
            }
        };
        self.keyed.insert(key, batch);
        let keys = &self.keys;
        let sorted = &mut self.sorted[kind];
        let at = sorted.partition_point(|&known| keys[known as usize] < key);
        sorted.insert(at, batch);
        batch
    }

    fn close(&mut self, batch: u32) {
        let at = batch as usize;
        let key = self.keys[at];
        self.keyed.remove(&key);
        let keys = &self.keys;
        let sorted = &mut self.sorted[key.0];
        let found = sorted.partition_point(|&known| keys[known as usize] < key);
        sorted.remove(found);
        let start = self.batches[at].instances.start;
        self.spare_regions[self.rooms[at].trailing_zeros() as usize].push(start);
        self.spare_batches.push(batch);
    }

    fn grow(&mut self, batch: u32) {
        let at = batch as usize;
        let room = self.rooms[at] * 2;
        let start = self.region(room);
        let instances = self.batches[at].instances.clone();
        let (from, len) = (instances.start as usize, instances.len());
        self.order
            .copy_within(from..from + len, start as usize);
        self.order[from..from + len].fill(NO_SLOT);
        for index in start..start + len as u32 {
            self.order_changes.mark(index);
        }
        self.spare_regions[self.rooms[at].trailing_zeros() as usize].push(instances.start);
        self.rooms[at] = room;
        self.batches[at].instances = start..start + len as u32;
    }

    fn region(&mut self, room: u32) -> u32 {
        let class = room.trailing_zeros() as usize;
        if self.spare_regions.len() <= class {
            self.spare_regions.resize_with(class + 1, Vec::new);
        }
        if let Some(start) = self.spare_regions[class].pop() {
            return start;
        }
        let start = self.order.len();
        self.order.resize(start + room as usize, NO_SLOT);
        start as u32
    }
}

fn stale(drawn: &Drawn, library: &AssetLibrary, before: u64) -> bool {
    let moved = |revision: Option<u64>| revision.is_none_or(|revision| revision > before);
    drawn
        .geometry
        .is_some_and(|geometry| moved(library.revision_of(geometry)))
        || drawn.material.is_some_and(|handle| {
            moved(library.revision_of(handle))
                || library.material(handle).is_some_and(|material| {
                    material
                        .textures()
                        .any(|texture| moved(library.revision_of(texture)))
                })
        })
}

#[cfg(any(feature = "dev", feature = "profiling"))]
fn under<R>(world: &World, entity: Entity, work: impl FnOnce() -> R) -> R {
    let cause = world.get::<crate::Caused>(entity).map_or(0, |caused| caused.0);
    engine_profile::caused(engine_profile::Cause(cause), work)
}

#[cfg(not(any(feature = "dev", feature = "profiling")))]
fn under<R>(_: &World, _: Entity, work: impl FnOnce() -> R) -> R {
    work()
}

fn refused_of(state: &DrawnState) -> Option<&Error> {
    match state {
        DrawnState::Refused(reason) => Some(reason),
        DrawnState::Drawn { .. }
        | DrawnState::Hidden
        | DrawnState::Unplaced
        | DrawnState::Waiting => None,
    }
}

fn hold(residency: &AssetResidency, drawn: &Drawn) -> Vec<Interest<Asset>> {
    drawn
        .geometry
        .map(|geometry| residency.hold(geometry))
        .into_iter()
        .chain(drawn.material.map(|material| residency.hold(material)))
        .collect()
}

fn touched(insert: On<Insert, Drawn>, mut table: ResMut<DrawnTable>) {
    table.touched.push(insert.entity);
}

fn dropped(remove: On<Remove, Drawn>, mut table: ResMut<DrawnTable>) {
    table.touched.push(remove.entity);
}

pub(crate) fn install(world: &mut World) {
    world.register_required_components::<Drawn, Transform>();
    world.init_resource::<DrawnTable>();
    world.add_observer(touched);
    world.add_observer(dropped);
}

pub(crate) fn install_kinds(world: &mut World) {
    let kinds: Vec<(&'static str, KindDecl)> = world
        .get_resource::<Registry>()
        .into_iter()
        .flat_map(|registry| registry.iter())
        .flat_map(|(_, record)| {
            record
                .render_entries::<KindDecl>()
                .map(move |decl| (record.package, *decl))
        })
        .collect();
    let mut problems = Vec::new();
    let mut listed: Vec<KindDecl> = Vec::with_capacity(kinds.len());
    for (brick, decl) in kinds {
        if listed.iter().any(|known| known.kind == decl.kind) {
            problems.push(
                Error::new(
                    ErrorCode::NameTwice,
                    format!("$.draw_kinds.{}", decl.kind.name()),
                )
                .got(brick),
            );
            continue;
        }
        listed.push(decl);
    }
    let mut table = world.resource_mut::<DrawnTable>();
    table.sorted.resize_with(listed.len(), Vec::new);
    table.kinds = listed;
    table.seen = None;
    if let Some(mut registry) = world.get_resource_mut::<Registry>() {
        for problem in problems {
            registry.report(problem);
        }
    }
}

pub(crate) fn ask(world: &mut World) {
    if !world.contains_resource::<DrawnTable>() {
        return;
    }
    world.resource_scope(|world, mut table: Mut<DrawnTable>| {
        let Some(residency) = world.get_resource::<AssetResidency>() else {
            return;
        };
        let touched = std::mem::take(&mut table.touched);
        for &entity in &touched {
            if let Some(drawn) = world.get::<Drawn>(entity) {
                let kept = under(world, entity, || table.take_held(entity, drawn, residency));
                table.held.insert(entity, kept);
            }
        }
        table.touched = touched;
    });
}

pub(crate) fn resolve(world: &mut World) {
    let settings = world
        .get_resource::<DrawnSettings>()
        .cloned()
        .unwrap_or_default();
    if !world.contains_resource::<DrawnTable>() {
        return;
    }
    let verdicts = world.resource_scope(|world, mut table: Mut<DrawnTable>| {
        table.settle(world, &settings);
        std::mem::take(&mut table.verdicts)
    });
    if let Some(mut refusals) = world.get_resource_mut::<Refusals>() {
        for (entity, verdict) in verdicts {
            refusals.judge(entity, DRAWN_PART, verdict);
        }
    }
    crate::refusals::prune(world);
}

#[cfg(any(feature = "dev", feature = "profiling"))]
#[derive(serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

#[cfg(any(feature = "dev", feature = "profiling"))]
fn table_of<'w>(
    reach: &crate::Reach<'w>,
    given: &serde_json::Value,
) -> Result<Option<&'w DrawnTable>, Error> {
    let NoArgs {} = crate::parse_value_at(given.clone(), "$.args")?;
    Ok(reach.world.get_resource::<DrawnTable>())
}

#[cfg(any(feature = "dev", feature = "profiling"))]
impl DrawnTable {
    fn objects(&self) -> serde_json::Value {
        let mut rows: Vec<(u32, serde_json::Value)> = self
            .held
            .iter()
            .filter_map(|(entity, kept)| match kept.state {
                DrawnState::Drawn { slot } => Some((slot, *entity)),
                DrawnState::Hidden
                | DrawnState::Unplaced
                | DrawnState::Waiting
                | DrawnState::Refused(_) => None,
            })
            .map(|(slot, entity)| {
                let record = self.records[slot as usize];
                let batch = self.homes[slot as usize].batch;
                let first = record.values as usize + VALUE_HEAD;
                let row = serde_json::json!({
                    "slot": slot,
                    "entity": entity.to_string(),
                    "node": (record.node != u32::MAX).then_some(record.node),
                    "kind": self.batches.get(batch as usize).map(|batch| batch.kind.name()),
                    "batch": (batch != NO_BATCH).then_some(batch),
                    "values": [record.values as usize, record.values as usize + self.stride],
                    "floats": [first, first + record.floats.div_ceil(4) as usize],
                });
                (slot, row)
            })
            .collect();
        rows.sort_unstable_by_key(|&(slot, _)| slot);
        serde_json::Value::Array(rows.into_iter().map(|(_, row)| row).collect())
    }

    fn batch_listing(&self, passes: Option<&crate::ScenePasses>) -> serde_json::Value {
        let named = |mask: PassMask| -> Vec<&'static str> {
            passes
                .into_iter()
                .flat_map(crate::ScenePasses::listed)
                .filter(|listed| listed.bit.is_some_and(|bit| mask.has(bit)))
                .map(|listed| listed.name)
                .collect()
        };
        let mut kinds = serde_json::Map::new();
        for (decl, sorted) in self.kinds.iter().zip(&self.sorted) {
            let batches = sorted
                .iter()
                .map(|&batch| {
                    let made = &self.batches[batch as usize];
                    serde_json::json!({
                        "batch": batch,
                        "order": made.order,
                        "geometry": made.geometry.map(GeometryHandle::slot),
                        "material": made.material.map(MaterialHandle::slot),
                        "passes": made.passes.map(named),
                        "layers": made.layers.0,
                        "instances": [made.instances.start, made.instances.end],
                        "room": self.rooms[batch as usize],
                        "cause": made.cause,
                    })
                })
                .collect();
            kinds.insert(decl.kind.name().to_owned(), serde_json::Value::Array(batches));
        }
        serde_json::Value::Object(kinds)
    }

    fn slot_counts(&self) -> serde_json::Value {
        serde_json::json!({
            "used": self.records.len() - self.free.len(),
            "capacity": self.records.len(),
            "free": self.free.len(),
            "batches": self.keyed.len(),
            "order": self.order.len(),
            "value_vec4s": self.stride,
        })
    }
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) fn object_names(world: &mut World) {
    use crate::{Kind, NameEntry, register};
    let shape = engine_place::schema_of::<NoArgs>();
    let entries = [
        NameEntry::read("objects", Kind::State, |reach, given| {
            Ok(table_of(reach, given)?.map_or(serde_json::json!([]), DrawnTable::objects))
        }),
        NameEntry::read("objects.batches", Kind::State, |reach, given| {
            let passes = reach.world.get_resource::<crate::ScenePasses>();
            Ok(table_of(reach, given)?
                .map_or(serde_json::json!({}), |table| table.batch_listing(passes)))
        }),
        NameEntry::read("objects.slots", Kind::State, |reach, given| {
            Ok(table_of(reach, given)?
                .map_or_else(|| DrawnTable::default().slot_counts(), DrawnTable::slot_counts))
        }),
    ];
    for entry in entries {
        register(world, entry.args(shape.clone(), serde_json::json!({})));
    }
}
