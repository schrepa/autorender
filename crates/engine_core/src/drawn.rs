use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::fmt::Write as _;
use std::ops::Range;

use bevy_ecs::change_detection::{DetectChanges, MAX_CHANGE_AGE, Mut, Ref, Tick};
use bevy_ecs::entity::Entity;
use bevy_ecs::lifecycle::{Insert, Remove};
use bevy_ecs::observer::On;
use bevy_ecs::query::QueryState;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::ResMut;
use bevy_ecs::world::World;
use engine_assets::{Asset, AssetLibrary, Material, MaterialBinding, SurfaceSlot, attributes_wgsl};
use engine_registry::Registry;

use crate::math::sin_cos;
use crate::{
    AssetResidency, Cursor, DeviceLimits, DrawKind, Drawn, DrawnBounds, Error, ErrorCode, Expected,
    Feed, GeometryHandle, GlobalTransforms, GpuBits, GpuStruct, InstanceValues, Interest, KindDecl,
    MaterialHandle, PassMask, PositionReach, Refusals, Request, Revision, ScenePasses, Since,
    Transform, gpu_struct, settings,
};

pub const DRAWN_PART: &str = "drawn";

settings! {
    #[section("drawn")]
    pub struct DrawnSettings {
        pub instance_floats: u32 = 16 => range(0, 1024),
        pub changes: u32 = 16384 => at_least(1),
    }
}

#[crate::gpu_bits(crate = crate)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawnFlags {
    pub own_projection: bool,
    pub unbounded: bool,
}

impl DrawnFlags {
    pub fn of(record: &DrawnRecord) -> Self {
        Self::unpack(record.flags).unwrap_or_default()
    }
}

const CLIP: &str = "fn drawn_value_vec4(record: DrawnRecord, index: u32) -> u32 {
    return record.values + index / 4u;
}
fn drawn_clip(record: DrawnRecord, clip: vec4<f32>, view_focal: f32, center: vec2<f32>) -> vec4<f32> {
    if !drawn_flags_own_projection(record.flags) {
        return clip;
    }
    let across = (clip.xy - center) * (record.projection.x / view_focal) + center;
    return vec4<f32>(across, mix(clip.w, clip.z, record.projection.y), clip.w);
}
";

gpu_struct! {
    pub struct DrawnRecord {
        pub tint: [f32; 4],
        pub emission: [f32; 4],
        pub projection: [f32; 4],
        pub bounds_min: [f32; 4],
        pub bounds_max: [f32; 4],
        pub node: u32,
        pub geometry: u32,
        pub material: u32,
        pub passes: u32,
        pub layers: u32,
        pub flags: u32,
        pub generation: u32,
        pub order: i32,
        pub values: u32,
        pub floats: u32,
        pub kind: u32,
        pub mobility: u32,
    }
}

pub fn drawn_wgsl() -> String {
    let mut text = String::from(DrawnRecord::WGSL);
    text.push_str(<DrawnFlags as GpuBits>::WGSL);
    text.push_str(&attributes_wgsl());
    text.push_str(CLIP);
    text
}

fn vec4s(floats: u32) -> usize {
    floats.div_ceil(4) as usize
}

#[derive(Clone, Debug)]
pub enum DrawnState {
    Drawn { slot: u32 },
    Hidden,
    Unplaced,
    Waiting,
    Refused(Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanState {
    Stale,
    Built { generation: u64 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct DrawnBatch {
    pub kind: DrawKind,
    pub geometry: Option<GeometryHandle>,
    pub material: Option<MaterialHandle>,
    pub order: i16,
    pub instances: Range<u32>,
}

type BatchKey = (i16, DrawKind, Option<usize>, Option<usize>);

type Shape = (i16, DrawKind, Option<GeometryHandle>, Option<MaterialHandle>);

type Keyed = Option<(BatchKey, Option<MaterialHandle>)>;

type Edits = QueryState<(Entity, Ref<'static, Drawn>, Option<Ref<'static, DrawnBounds>>)>;

struct Held {
    state: DrawnState,
    drawn: Drawn,
    uses: Vec<Interest<Asset>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Seen {
    residency: u64,
    library: (u64, u64),
    topology: Revision,
    words: u64,
    stride: usize,
    room: u64,
}

struct Context<'a> {
    globals: &'a GlobalTransforms,
    residency: &'a AssetResidency,
    library: &'a AssetLibrary,
    defaults: &'a [PassMask],
    floats: usize,
    room: u64,
}

#[derive(Resource)]
pub struct DrawnTable {
    kinds: Vec<KindDecl>,
    reaches: Vec<PositionReach>,
    held: BTreeMap<Entity, Held>,
    owners: Vec<Option<Entity>>,
    generations: Vec<u32>,
    records: Vec<DrawnRecord>,
    values: Vec<[f32; 4]>,
    free: Vec<u32>,
    touched: Vec<Entity>,
    verdicts: Vec<(Entity, Result<(), Error>)>,
    changes: Feed<u32>,
    plan: PlanState,
    generation: u64,
    ordered: Vec<u32>,
    batches: Vec<DrawnBatch>,
    seen: Option<Seen>,
    stride: usize,
    shapes: Vec<Option<Shape>>,
    keys: Vec<Keyed>,
    canonical: crate::collections::HashMap<MaterialBinding, MaterialHandle>,
    resolved: Vec<Option<(MaterialHandle, Option<MaterialHandle>)>>,
    drifted: bool,
    defaults: Vec<PassMask>,
    floats: usize,
    loose: bool,
    ticked: Tick,
    clock: Tick,
    age: u64,
    edits: Option<Edits>,
}

impl Default for DrawnTable {
    fn default() -> Self {
        let settings = DrawnSettings::default();
        Self {
            kinds: Vec::new(),
            reaches: Vec::new(),
            held: BTreeMap::new(),
            owners: Vec::new(),
            generations: Vec::new(),
            records: Vec::new(),
            values: Vec::new(),
            free: Vec::new(),
            touched: Vec::new(),
            verdicts: Vec::new(),
            changes: Feed::new(settings.changes as usize),
            plan: PlanState::Stale,
            generation: 0,
            ordered: Vec::new(),
            batches: Vec::new(),
            seen: None,
            stride: vec4s(settings.instance_floats),
            shapes: Vec::new(),
            keys: Vec::new(),
            canonical: crate::collections::HashMap::default(),
            resolved: Vec::new(),
            drifted: false,
            defaults: Vec::new(),
            floats: settings.instance_floats as usize,
            loose: true,
            ticked: Tick::default(),
            clock: Tick::default(),
            age: 0,
            edits: None,
        }
    }
}

impl DrawnTable {
    pub fn kinds(&self) -> &[KindDecl] {
        &self.kinds
    }

    pub fn kind(&self, kind: DrawKind) -> Option<&KindDecl> {
        self.kinds.iter().find(|decl| decl.kind == kind)
    }

    pub fn kind_id(&self, kind: DrawKind) -> Option<u32> {
        self.kinds
            .iter()
            .position(|decl| decl.kind == kind)
            .map(|at| at as u32 + 1)
    }

    pub fn kind_of(&self, id: u32) -> Option<DrawKind> {
        let at = (id as usize).checked_sub(1)?;
        self.kinds.get(at).map(|decl| decl.kind)
    }

    pub fn wgsl(&self) -> String {
        let mut text = String::new();
        for decl in &self.kinds {
            let name = engine_place::wgsl_name(decl.kind.name());
            if let Some(id) = self.kind_id(decl.kind) {
                let _ = writeln!(text, "const KIND_{name}: u32 = {id}u;");
            }
        }
        text
    }

    pub fn reach(&self, material: &Material) -> f32 {
        material
            .slots()
            .iter()
            .find(|(slot, _)| *slot == SurfaceSlot::Position.name())
            .and_then(|(_, module)| {
                self.reaches
                    .iter()
                    .find(|reach| reach.module == module.as_str())
            })
            .map_or(0.0, |reach| reach.reach)
    }

    pub fn state(&self, entity: Entity) -> Option<&DrawnState> {
        self.held.get(&entity).map(|held| &held.state)
    }

    pub(crate) fn drawn_count(&self) -> usize {
        self.owners.len() - self.free.len()
    }

    pub fn slot(&self, entity: Entity) -> Option<u32> {
        match self.state(entity) {
            Some(DrawnState::Drawn { slot }) => Some(*slot),
            Some(
                DrawnState::Hidden
                | DrawnState::Unplaced
                | DrawnState::Waiting
                | DrawnState::Refused(_),
            )
            | None => None,
        }
    }

    pub fn owner(&self, slot: u32) -> Option<Entity> {
        self.owners.get(slot as usize).copied().flatten()
    }

    pub fn drawn(&self, slot: u32) -> Option<&Drawn> {
        self.owner(slot)
            .and_then(|entity| self.held.get(&entity))
            .map(|held| &held.drawn)
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

    pub fn changes_since(&self, cursor: &mut Cursor) -> Since<'_, u32> {
        self.changes.since(cursor)
    }

    pub fn plan(&self) -> PlanState {
        self.plan
    }

    pub fn ordered(&self) -> &[u32] {
        &self.ordered
    }

    pub fn batches(&self) -> &[DrawnBatch] {
        &self.batches
    }

    pub fn rows(&self, kind: DrawKind) -> impl Iterator<Item = (u32, Entity, &Drawn)> + '_ {
        self.batches
            .iter()
            .filter(move |batch| batch.kind == kind)
            .flat_map(move |batch| {
                &self.ordered[batch.instances.start as usize..batch.instances.end as usize]
            })
            .filter_map(move |&slot| {
                let entity = self.owner(slot)?;
                Some((slot, entity, &self.held.get(&entity)?.drawn))
            })
    }

    fn settle(&mut self, world: &World, settings: &DrawnSettings) {
        let (Some(globals), Some(residency), Some(library)) = (
            world.get_resource::<GlobalTransforms>(),
            world.get_resource::<AssetResidency>(),
            world.get_resource::<AssetLibrary>(),
        ) else {
            return;
        };
        let now = world.read_change_tick();
        self.age = self
            .age
            .saturating_add(u64::from(now.get().wrapping_sub(self.clock.get())));
        self.clock = now;
        self.changes.resize(settings.changes as usize);
        let stride = vec4s(settings.instance_floats);
        if stride != self.stride {
            self.stride = stride;
            self.values = vec![[0.0; 4]; self.owners.len() * stride];
        }
        let room = world
            .get_resource::<DeviceLimits>()
            .map_or(0, |limits| limits.storage_room);
        let seen = Seen {
            residency: residency.revision(),
            library: library.revision(),
            topology: globals.topology_revision(),
            words: globals.words_revision(),
            stride,
            room,
        };
        let floats = settings.instance_floats as usize;
        let passes = world.get_resource::<ScenePasses>();
        let mut defaults = std::mem::take(&mut self.defaults);
        self.loose |= floats != self.floats || defaults.len() != self.kinds.len();
        defaults.resize(self.kinds.len(), PassMask::NONE);
        for (held, decl) in defaults.iter_mut().zip(&self.kinds) {
            let next = passes.map_or(PassMask::NONE, |passes| passes.defaults(decl.kind));
            self.loose |= *held != next;
            *held = next;
        }
        self.floats = floats;
        let known = self.seen;
        let mut touched = std::mem::take(&mut self.touched);
        let mut held = std::mem::take(&mut self.held);
        if known != Some(seen) {
            self.seen = Some(seen);
            let narrow = known.filter(|known| {
                !self.loose
                    && known.library.0 == seen.library.0
                    && seen.library.1 != u64::MAX
                    && (known.stride, known.room) == (stride, room)
                    && self.age < u64::from(MAX_CHANGE_AGE)
            });
            if let (Some(known), Some(edits)) = (narrow, self.edits.as_mut()) {
                let since = self.ticked;
                let edited = |tick: Tick| tick == since || tick.is_newer_than(since, now);
                edits.update_archetypes(world);
                for (entity, drawn, bounds) in edits.iter_manual(world) {
                    if (edited(drawn.last_changed())
                        || bounds.is_some_and(|bounds| edited(bounds.last_changed())))
                        && held.contains_key(&entity)
                    {
                        touched.push(entity);
                    }
                }
                let before = (known.library != seen.library).then_some(known.library.1);
                touched.extend(
                    held.iter()
                        .filter(|&(&entity, kept)| {
                            self.moved(entity, kept, (globals, library), before)
                        })
                        .map(|(&entity, _)| entity),
                );
            } else {
                touched.extend(held.keys().copied());
            }
            self.loose = false;
            self.ticked = now;
            self.age = 0;
        }
        touched.sort_unstable();
        touched.dedup();
        let cx = Context {
            globals,
            residency,
            library,
            defaults: &defaults,
            floats,
            room,
        };
        for &entity in &touched {
            match world.get::<Drawn>(entity) {
                Some(drawn) => {
                    let bounds = world.get::<DrawnBounds>(entity);
                    self.refresh(&mut held, entity, (drawn, bounds), &cx);
                }
                None => self.forget(&mut held, entity),
            }
        }
        touched.clear();
        self.touched = touched;
        self.held = held;
        self.defaults = defaults;
        if known.is_none_or(|known| known.library != seen.library)
            && self.plan != PlanState::Stale
        {
            let agrees = match known {
                Some(known) if known.library.0 == seen.library.0 && !self.drifted => {
                    self.agrees(library, known.library.1)
                }
                Some(_) | None => false,
            };
            if agrees {
                self.generation += 1;
                self.plan = PlanState::Built {
                    generation: self.generation,
                };
            } else {
                self.plan = PlanState::Stale;
            }
        }
        if self.plan == PlanState::Stale {
            self.rebuild(library);
        }
    }

    fn moved(
        &self,
        entity: Entity,
        kept: &Held,
        (globals, library): (&GlobalTransforms, &AssetLibrary),
        before: Option<u64>,
    ) -> bool {
        let slot = match kept.state {
            DrawnState::Drawn { slot } => slot,
            DrawnState::Hidden => return globals.drawn(entity),
            DrawnState::Unplaced => {
                return !globals.drawn(entity)
                    || globals
                        .get(entity)
                        .and_then(|global| global.matrix())
                        .is_some();
            }
            DrawnState::Waiting | DrawnState::Refused(_) => return true,
        };
        let record = &self.records[slot as usize];
        let node = record.node as usize;
        let placed = globals.entity_at(record.node) == Some(entity)
            && globals.word_at(node).is_some_and(|word| {
                word.drawn() && word.mobility as u32 + 1 == record.mobility
            })
            && globals
                .links()
                .get(node)
                .is_some_and(|link| link.has_global());
        let stale = |revision: Option<u64>| {
            before.is_some_and(|before| revision.is_none_or(|revision| revision > before))
        };
        !placed
            || PartialEq::ne(record, record)
            || kept
                .drawn
                .geometry
                .is_some_and(|geometry| stale(library.revision_of(geometry)))
            || kept.drawn.material.is_some_and(|handle| {
                stale(library.revision_of(handle))
                    || library.material(handle).is_some_and(|material| {
                        material
                            .textures()
                            .any(|texture| stale(library.revision_of(texture)))
                    })
            })
    }

    fn refresh(
        &mut self,
        held: &mut BTreeMap<Entity, Held>,
        entity: Entity,
        (drawn, bounds): (&Drawn, Option<&DrawnBounds>),
        cx: &Context<'_>,
    ) {
        let kept = match held.entry(entity) {
            Entry::Occupied(kept) => kept.into_mut(),
            Entry::Vacant(free) => free.insert(Held {
                state: DrawnState::Waiting,
                drawn: drawn.clone(),
                uses: hold(cx.residency, drawn),
            }),
        };
        let slot = match kept.state {
            DrawnState::Drawn { slot } => Some(slot),
            DrawnState::Hidden
            | DrawnState::Unplaced
            | DrawnState::Waiting
            | DrawnState::Refused(_) => None,
        };
        let (was, shaped) = (shape(&kept.drawn), shape(drawn));
        if slot.is_some() && was != shaped {
            if batch_key(was) == batch_key(shaped) {
                self.drifted = true;
            } else {
                self.plan = PlanState::Stale;
            }
        }
        if (kept.drawn.geometry, kept.drawn.material) != (drawn.geometry, drawn.material) {
            let uses = hold(cx.residency, drawn);
            drop(std::mem::replace(&mut kept.uses, uses));
        }
        let state = match (self.refusal(entity, drawn, cx), slot) {
            (None, Some(slot)) => {
                self.write(slot, entity, (drawn, bounds), cx);
                DrawnState::Drawn { slot }
            }
            (None, None) => {
                let slot = self.take(entity);
                if self.beyond(slot, cx.room) {
                    self.release(slot);
                    DrawnState::Refused(
                        Error::new(ErrorCode::BindingLimit, "$.drawn.instance_floats")
                            .at_most(cx.room)
                            .got((u64::from(slot) + 1) * self.slot_bytes()),
                    )
                } else {
                    self.write(slot, entity, (drawn, bounds), cx);
                    DrawnState::Drawn { slot }
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
        if let DrawnState::Drawn { slot } = state {
            self.shapes[slot as usize] = Some(shaped);
        }
        kept.state = state;
        keep(&mut kept.drawn, drawn);
    }

    fn slot_bytes(&self) -> u64 {
        (size_of::<DrawnRecord>() as u64).max((self.stride * size_of::<[f32; 4]>()) as u64)
    }

    fn beyond(&self, slot: u32, room: u64) -> bool {
        room != 0 && (u64::from(slot) + 1) * self.slot_bytes() > room
    }

    fn refusal(&self, entity: Entity, drawn: &Drawn, cx: &Context<'_>) -> Option<DrawnState> {
        let Some(decl) = self.kind(drawn.kind) else {
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

    fn take(&mut self, entity: Entity) -> u32 {
        self.plan = PlanState::Stale;
        if let Some(slot) = self.free.pop() {
            self.owners[slot as usize] = Some(entity);
            return slot;
        }
        self.owners.push(Some(entity));
        self.generations.push(0);
        self.records.push(DrawnRecord::default());
        self.values
            .extend(std::iter::repeat_n([0.0; 4], self.stride));
        self.shapes.push(None);
        (self.owners.len() - 1) as u32
    }

    fn release(&mut self, slot: u32) {
        let at = slot as usize;
        self.owners[at] = None;
        self.generations[at] = self.generations[at].wrapping_add(1);
        self.records[at] = DrawnRecord::default();
        self.shapes[at] = None;
        self.free.push(slot);
        self.changes.push(slot);
        self.plan = PlanState::Stale;
    }

    fn forget(&mut self, held: &mut BTreeMap<Entity, Held>, entity: Entity) {
        let Some(gone) = held.remove(&entity) else {
            return;
        };
        if refused_of(&gone.state).is_some() {
            self.verdicts.push((entity, Ok(())));
        }
        if let DrawnState::Drawn { slot } = gone.state {
            self.release(slot);
        }
    }

    fn write(
        &mut self,
        slot: u32,
        entity: Entity,
        (drawn, supplied): (&Drawn, Option<&DrawnBounds>),
        cx: &Context<'_>,
    ) {
        let at = slot as usize;
        let bounds = supplied.map(|bounds| (bounds.min, bounds.max)).or_else(|| {
            drawn
                .geometry
                .and_then(|geometry| cx.library.mesh(geometry))
                .map(|mesh| {
                    let bounds = mesh.bounds();
                    (bounds.min().to_array(), bounds.max().to_array())
                })
        });
        let reach = drawn
            .material
            .and_then(|material| cx.library.material(material))
            .map_or(0.0, |material| self.reach(material));
        let ([low_x, low_y, low_z], [high_x, high_y, high_z]) =
            bounds.map_or(([0.0; 3], [0.0; 3]), |(low, high)| {
                (
                    low.map(|value| value - reach),
                    high.map(|value| value + reach),
                )
            });
        let kind = self.kinds.iter().position(|decl| decl.kind == drawn.kind);
        let passes = drawn
            .passes
            .unwrap_or_else(|| kind.map_or(PassMask::NONE, |at| cx.defaults[at]));
        let projection = drawn.projection.map_or([1.0, 1.0, 0.0, 0.0], |own| {
            let (sine, cosine) = sin_cos(own.vertical_fov * 0.5);
            [cosine / sine, own.depth_scale, 0.0, 0.0]
        });
        let flags = DrawnFlags {
            own_projection: drawn.projection.is_some(),
            unbounded: bounds.is_none(),
        };
        let [red, green, blue] = drawn.values.emission;
        let record = DrawnRecord {
            tint: drawn.values.tint,
            emission: [red, green, blue, 0.0],
            projection,
            bounds_min: [low_x, low_y, low_z, 0.0],
            bounds_max: [high_x, high_y, high_z, 0.0],
            node: cx
                .globals
                .node_index(entity)
                .map_or(u32::MAX, |index| index as u32),
            geometry: drawn
                .geometry
                .map_or(u32::MAX, |geometry| geometry.slot() as u32),
            material: drawn
                .material
                .map_or(u32::MAX, |material| material.slot() as u32),
            passes: passes.word(),
            layers: drawn.layers.0,
            flags: flags.pack(),
            generation: self.generations[at],
            order: i32::from(drawn.order),
            values: (at * self.stride) as u32,
            floats: drawn.values.floats.len() as u32,
            kind: kind.map_or(0, |at| at as u32 + 1),
            mobility: cx
                .globals
                .mobility(entity)
                .map_or(0, |mobility| mobility as u32 + 1),
        };
        let mut changed = self.records[at] != record;
        let floats = &drawn.values.floats;
        for index in 0..self.stride {
            let quad: [f32; 4] =
                std::array::from_fn(|lane| floats.get(index * 4 + lane).copied().unwrap_or(0.0));
            let held = &mut self.values[at * self.stride + index];
            if held.map(f32::to_bits) != quad.map(f32::to_bits) {
                *held = quad;
                changed = true;
            }
        }
        if changed {
            self.records[at] = record;
            self.changes.push(slot);
        }
    }

    fn agrees(&mut self, library: &AssetLibrary, before: u64) -> bool {
        let Self {
            shapes,
            keys,
            canonical,
            resolved,
            ..
        } = self;
        if shapes.len() != keys.len() {
            return false;
        }
        resolved.clear();
        for (shape, key) in shapes.iter().zip(keys.iter()) {
            let (shape, kept) = match (shape, key) {
                (Some(shape), Some((_, kept))) => (shape, *kept),
                (None, None) => continue,
                (Some(_), None) | (None, Some(_)) => return false,
            };
            let Some(handle) = shape.3 else {
                continue;
            };
            if library
                .revision_of(handle)
                .is_some_and(|revision| revision <= before)
            {
                continue;
            }
            let at = handle.slot();
            if resolved.len() <= at {
                resolved.resize(at + 1, None);
            }
            if let Some((known, checked)) = resolved[at]
                && known == handle
            {
                if checked != kept {
                    return false;
                }
                continue;
            }
            let binding = library.material(handle).and_then(Material::binding);
            let same = match (kept, binding) {
                (None, None) => true,
                (Some(kept), Some(binding)) => canonical.get(&binding) == Some(&kept),
                (Some(_), None) | (None, Some(_)) => false,
            };
            if !same {
                return false;
            }
            resolved[at] = Some((handle, kept));
        }
        true
    }

    fn rebuild(&mut self, library: &AssetLibrary) {
        let Self {
            owners,
            ordered,
            batches,
            shapes,
            keys,
            canonical,
            resolved,
            ..
        } = self;
        canonical.clear();
        resolved.clear();
        keys.clear();
        keys.extend(shapes.iter().map(|shape| {
            let (order, kind, geometry, material) = (*shape)?;
            let material = material.and_then(|handle| {
                let at = handle.slot();
                if resolved.len() <= at {
                    resolved.resize(at + 1, None);
                }
                if let Some((known, found)) = resolved[at]
                    && known == handle
                {
                    return found;
                }
                let found = library
                    .material(handle)
                    .and_then(Material::binding)
                    .map(|binding| *canonical.entry(binding).or_insert(handle));
                resolved[at] = Some((handle, found));
                found
            });
            Some((batch_key((order, kind, geometry, material)), material))
        }));
        ordered.clear();
        ordered.extend(
            owners
                .iter()
                .enumerate()
                .filter(|(_, owner)| owner.is_some())
                .map(|(slot, _)| slot as u32),
        );
        ordered.sort_unstable_by_key(|&slot| {
            (keys[slot as usize].as_ref().map(|(key, _)| *key), slot)
        });
        batches.clear();
        let mut start = 0;
        for at in 1..=ordered.len() {
            let first = ordered[start] as usize;
            let same = at < ordered.len()
                && keys[ordered[at] as usize].as_ref().map(|(key, _)| key)
                    == keys[first].as_ref().map(|(key, _)| key);
            if same {
                continue;
            }
            if let (Some((order, kind, geometry, _)), Some((_, material))) =
                (shapes[first], &keys[first])
            {
                batches.push(DrawnBatch {
                    kind,
                    geometry,
                    material: *material,
                    order,
                    instances: start as u32..at as u32,
                });
            }
            start = at;
        }
        self.drifted = false;
        self.generation += 1;
        self.plan = PlanState::Built {
            generation: self.generation,
        };
    }
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

fn shape(drawn: &Drawn) -> Shape {
    (drawn.order, drawn.kind, drawn.geometry, drawn.material)
}

fn batch_key((order, kind, geometry, material): Shape) -> BatchKey {
    (
        order,
        kind,
        geometry.map(GeometryHandle::slot),
        material.map(MaterialHandle::slot),
    )
}

fn keep(held: &mut Drawn, drawn: &Drawn) {
    let Drawn {
        kind,
        geometry,
        material,
        passes,
        layers,
        order,
        projection,
        values:
            InstanceValues {
                tint,
                emission,
                floats,
            },
    } = drawn;
    held.kind = *kind;
    held.geometry = *geometry;
    held.material = *material;
    held.passes = *passes;
    held.layers = *layers;
    held.order = *order;
    held.projection = *projection;
    held.values.tint = *tint;
    held.values.emission = *emission;
    held.values.floats.clone_from(floats);
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

fn bounded(insert: On<Insert, DrawnBounds>, mut table: ResMut<DrawnTable>) {
    table.touched.push(insert.entity);
}

fn unbounded(remove: On<Remove, DrawnBounds>, mut table: ResMut<DrawnTable>) {
    table.touched.push(remove.entity);
}

pub(crate) fn install(world: &mut World) {
    world.register_required_components::<Drawn, Transform>();
    world.init_resource::<DrawnTable>();
    world.init_resource::<crate::DrawnAreas>();
    world.add_observer(touched);
    world.add_observer(dropped);
    world.add_observer(bounded);
    world.add_observer(unbounded);
    let edits = Edits::new(world);
    world.resource_mut::<DrawnTable>().edits = Some(edits);
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
    let reaches: Vec<PositionReach> = world
        .get_resource::<Registry>()
        .into_iter()
        .flat_map(|registry| registry.iter())
        .flat_map(|(_, record)| record.render_entries::<PositionReach>().copied())
        .collect();
    let mut table = world.resource_mut::<DrawnTable>();
    table.kinds = listed;
    table.reaches = reaches;
    table.seen = None;
    if let Some(mut registry) = world.get_resource_mut::<Registry>() {
        for problem in problems {
            registry.report(problem);
        }
    }
}

pub(crate) fn resolve(world: &mut World) {
    let settings = world
        .get_resource::<DrawnSettings>()
        .cloned()
        .unwrap_or_default();
    if let Some(mut areas) = world.get_resource_mut::<crate::DrawnAreas>() {
        areas.resize(settings.changes as usize);
    }
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
