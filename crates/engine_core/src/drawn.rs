use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::ops::Range;

use bevy_ecs::change_detection::Mut;
use bevy_ecs::entity::Entity;
use bevy_ecs::lifecycle::{Insert, Remove};
use bevy_ecs::observer::On;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::ResMut;
use bevy_ecs::world::World;
use engine_assets::{Asset, AssetLibrary, Material, MaterialBinding, SurfaceSlot, attributes_wgsl};
use engine_registry::Registry;

use crate::math::sin_cos;
use crate::{
    AssetResidency, Cursor, DeviceLimits, DrawKind, Drawn, DrawnBounds, Error, ErrorCode,
    Expected, Feed, GlobalTransforms, GpuBits, GpuStruct, Interest, KindDecl, MaterialHandle,
    MeshHandle, PassMask, PositionReach, Refusals, Request, Revision, ScenePasses, Since, Transform,
    gpu_struct, settings,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawnCensus {
    pub drawn: u32,
    pub hidden: u32,
    pub unplaced: u32,
    pub waiting: u32,
    pub refused: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanState {
    Stale,
    Built { generation: u64 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct DrawnBatch {
    pub kind: DrawKind,
    pub geometry: Option<MeshHandle>,
    pub material: Option<MaterialHandle>,
    pub order: i16,
    pub instances: Range<u32>,
}

type BatchKey = (i16, DrawKind, Option<usize>, Option<usize>);

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
    passes: Option<&'a ScenePasses>,
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

    pub fn states(&self) -> impl Iterator<Item = (Entity, &DrawnState)> + '_ {
        self.held.iter().map(|(&entity, held)| (entity, &held.state))
    }

    pub fn census(&self, kind: DrawKind) -> DrawnCensus {
        let mut census = DrawnCensus::default();
        for held in self.held.values().filter(|held| held.drawn.kind == kind) {
            let count = match held.state {
                DrawnState::Drawn { .. } => &mut census.drawn,
                DrawnState::Hidden => &mut census.hidden,
                DrawnState::Unplaced => &mut census.unplaced,
                DrawnState::Waiting => &mut census.waiting,
                DrawnState::Refused(_) => &mut census.refused,
            };
            *count += 1;
        }
        census
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
        let mut touched = std::mem::take(&mut self.touched);
        if self.seen.is_none_or(|known| known.library != seen.library) {
            self.plan = PlanState::Stale;
        }
        if self.seen != Some(seen) {
            self.seen = Some(seen);
            touched.extend(self.held.keys().copied());
        }
        touched.sort_unstable();
        touched.dedup();
        let cx = Context {
            globals,
            residency,
            library,
            passes: world.get_resource::<ScenePasses>(),
            floats: settings.instance_floats as usize,
            room,
        };
        for entity in touched {
            match world.get::<Drawn>(entity) {
                Some(drawn) => {
                    self.refresh(entity, (drawn, world.get::<DrawnBounds>(entity)), &cx);
                }
                None => self.forget(entity),
            }
        }
        if self.plan == PlanState::Stale {
            self.rebuild(library);
        }
    }

    fn refresh(
        &mut self,
        entity: Entity,
        (drawn, bounds): (&Drawn, Option<&DrawnBounds>),
        cx: &Context<'_>,
    ) {
        let previous = self.held.remove(&entity);
        let was = previous
            .as_ref()
            .and_then(|held| refused_of(&held.state))
            .cloned();
        let (slot, uses) = match previous {
            Some(held) => {
                let slot = match held.state {
                    DrawnState::Drawn { slot } => Some(slot),
                    DrawnState::Hidden
                    | DrawnState::Unplaced
                    | DrawnState::Waiting
                    | DrawnState::Refused(_) => None,
                };
                if slot.is_some() && batch_key(&held.drawn) != batch_key(drawn) {
                    self.plan = PlanState::Stale;
                }
                let uses = if (held.drawn.geometry, held.drawn.material)
                    == (drawn.geometry, drawn.material)
                {
                    held.uses
                } else {
                    hold(cx.residency, drawn)
                };
                (slot, uses)
            }
            None => (None, hold(cx.residency, drawn)),
        };
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
        if was.as_ref() != now {
            self.verdicts
                .push((entity, now.cloned().map_or(Ok(()), Err)));
        }
        self.held.insert(
            entity,
            Held {
                state,
                drawn: drawn.clone(),
                uses,
            },
        );
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
        if drawn.geometry.is_none() && decl.needs != engine_assets::Attributes::NONE {
            return Some(DrawnState::Refused(
                Error::new(ErrorCode::InvalidRenderable, "$.drawn.geometry")
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
                .map(|geometry| (Asset::Mesh(geometry), "$.drawn.geometry")),
            drawn
                .material
                .map(|material| (Asset::Material(material), "$.drawn.material")),
        ];
        for (asset, path) in assets.into_iter().flatten() {
            match cx.residency.state(asset) {
                Some(Request::Ready(_)) => {}
                Some(Request::Failed(error)) => return Some(DrawnState::Refused(error.clone())),
                Some(Request::Missing) => {
                    return Some(DrawnState::Refused(Error::new(
                        ErrorCode::UnknownReference,
                        path,
                    )));
                }
                Some(Request::Asked(_)) | None => return Some(DrawnState::Waiting),
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
        mesh.attributes()
            .check(decl.needs, "$.drawn.geometry")
            .err()
            .map(DrawnState::Refused)
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
        (self.owners.len() - 1) as u32
    }

    fn release(&mut self, slot: u32) {
        let at = slot as usize;
        self.owners[at] = None;
        self.generations[at] = self.generations[at].wrapping_add(1);
        self.records[at] = DrawnRecord::default();
        self.free.push(slot);
        self.changes.push(slot);
        self.plan = PlanState::Stale;
    }

    fn forget(&mut self, entity: Entity) {
        let Some(held) = self.held.remove(&entity) else {
            return;
        };
        if refused_of(&held.state).is_some() {
            self.verdicts.push((entity, Ok(())));
        }
        if let DrawnState::Drawn { slot } = held.state {
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
        let ([low_x, low_y, low_z], [high_x, high_y, high_z]) = bounds.map_or(
            ([0.0; 3], [0.0; 3]),
            |(low, high)| (low.map(|value| value - reach), high.map(|value| value + reach)),
        );
        let passes = drawn.passes.unwrap_or_else(|| {
            cx.passes
                .map_or(PassMask::NONE, |passes| passes.defaults(drawn.kind))
        });
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
            kind: self.kind_id(drawn.kind).unwrap_or(0),
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

    fn rebuild(&mut self, library: &AssetLibrary) {
        let mut canonical: crate::collections::HashMap<MaterialBinding, MaterialHandle> =
            crate::collections::HashMap::default();
        let keys: Vec<Option<(BatchKey, Option<MaterialHandle>)>> = self
            .owners
            .iter()
            .map(|owner| {
                let held = owner.and_then(|entity| self.held.get(&entity))?;
                let material = held.drawn.material.and_then(|handle| {
                    let binding = library.material(handle)?.binding()?;
                    Some(*canonical.entry(binding).or_insert(handle))
                });
                let (order, kind, geometry, _) = batch_key(&held.drawn);
                Some((
                    (order, kind, geometry, material.map(|material| material.slot())),
                    material,
                ))
            })
            .collect();
        self.ordered.clear();
        self.ordered.extend(
            self.owners
                .iter()
                .enumerate()
                .filter(|(_, owner)| owner.is_some())
                .map(|(slot, _)| slot as u32),
        );
        self.ordered.sort_unstable_by_key(|&slot| {
            (keys[slot as usize].as_ref().map(|(key, _)| *key), slot)
        });
        self.batches.clear();
        let mut start = 0;
        for at in 1..=self.ordered.len() {
            let first = self.ordered[start] as usize;
            let same = at < self.ordered.len()
                && keys[self.ordered[at] as usize].as_ref().map(|(key, _)| key)
                    == keys[first].as_ref().map(|(key, _)| key);
            if same {
                continue;
            }
            let found = self.owners[first]
                .and_then(|entity| self.held.get(&entity))
                .zip(keys[first].as_ref());
            if let Some((held, &(_, material))) = found {
                self.batches.push(DrawnBatch {
                    kind: held.drawn.kind,
                    geometry: held.drawn.geometry,
                    material,
                    order: held.drawn.order,
                    instances: start as u32..at as u32,
                });
            }
            start = at;
        }
        self.generation += 1;
        self.plan = PlanState::Built {
            generation: self.generation,
        };
    }
}

fn refused_of(state: &DrawnState) -> Option<&Error> {
    match state {
        DrawnState::Refused(reason) => Some(reason),
        DrawnState::Drawn { .. } | DrawnState::Hidden | DrawnState::Unplaced | DrawnState::Waiting => {
            None
        }
    }
}

fn batch_key(drawn: &Drawn) -> BatchKey {
    (
        drawn.order,
        drawn.kind,
        drawn.geometry.map(|geometry| geometry.slot()),
        drawn.material.map(|material| material.slot()),
    )
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
}

pub(crate) fn install_kinds(world: &mut World) {
    let mut kinds: Vec<(&'static str, KindDecl)> = world
        .get_resource::<Registry>()
        .into_iter()
        .flat_map(|registry| registry.iter())
        .flat_map(|(_, record)| {
            record
                .render_entries::<KindDecl>()
                .map(move |decl| (record.package, *decl))
        })
        .collect();
    kinds.sort_by_key(|(brick, decl)| (decl.kind, *brick));
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
