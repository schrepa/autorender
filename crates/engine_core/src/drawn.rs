use std::collections::BTreeMap;
use std::ops::Range;

use bevy_ecs::entity::Entity;
use bevy_ecs::lifecycle::{Insert, Remove};
use bevy_ecs::observer::On;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::ResMut;
use bevy_ecs::change_detection::Mut;
use bevy_ecs::world::World;
use engine_assets::{Asset, AssetLibrary, attributes_wgsl};
use engine_registry::Registry;

use crate::math::sin_cos;
use crate::{
    AssetResidency, Cursor, DrawKind, Drawn, Error, ErrorCode, Feed, GlobalTransforms, GpuStruct,
    Interest, KindDecl, PassMask, Request, Revision, ScenePasses, Since, gpu_struct, settings,
};

settings! {
    #[section("drawn")]
    pub struct DrawnSettings {
        pub instance_floats: u32 = 16 => range(0, 1024),
        pub changes: u32 = 16384 => at_least(1),
    }
}

crate::gpu_consts! {
    pub const DRAWN_RECEIVES_SHADOWS: u32 = 1;
    pub const DRAWN_OWN_PROJECTION: u32 = 2;
}

const CLIP: &str = "fn drawn_value_index(slot: u32, index: u32) -> u32 {
    return slot * DRAWN_VALUE_VEC4S + index / 4u;
}
fn drawn_clip(record: DrawnRecord, clip: vec4<f32>, view_focal: f32, center: vec2<f32>) -> vec4<f32> {
    if (record.flags & DRAWN_OWN_PROJECTION) == 0u {
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
    }
}

pub fn drawn_wgsl(settings: &DrawnSettings) -> String {
    let mut text = String::from(DrawnRecord::WGSL);
    text.push_str(WGSL);
    text.push_str(&attributes_wgsl());
    text.push_str(&format!(
        "const DRAWN_VALUE_VEC4S: u32 = {}u;\n",
        vec4s(settings.instance_floats)
    ));
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
    pub geometry: crate::GeometryHandle,
    pub material: crate::MaterialHandle,
    pub order: i16,
    pub instances: Range<u32>,
}

struct Held {
    state: DrawnState,
    drawn: Drawn,
    uses: [Interest<Asset>; 2],
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Seen {
    residency: u64,
    topology: Revision,
    words: u64,
    stride: usize,
}

struct Context<'a> {
    globals: &'a GlobalTransforms,
    residency: &'a AssetResidency,
    library: &'a AssetLibrary,
    passes: Option<&'a ScenePasses>,
    floats: usize,
}

#[derive(Resource)]
pub struct DrawnTable {
    kinds: Vec<KindDecl>,
    held: BTreeMap<Entity, Held>,
    owners: Vec<Option<Entity>>,
    generations: Vec<u32>,
    records: Vec<DrawnRecord>,
    values: Vec<[f32; 4]>,
    free: Vec<u32>,
    touched: Vec<Entity>,
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
            held: BTreeMap::new(),
            owners: Vec::new(),
            generations: Vec::new(),
            records: Vec::new(),
            values: Vec::new(),
            free: Vec::new(),
            touched: Vec::new(),
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

    pub fn state(&self, entity: Entity) -> Option<&DrawnState> {
        self.held.get(&entity).map(|held| &held.state)
    }

    pub fn states(&self) -> impl Iterator<Item = (Entity, &DrawnState)> + '_ {
        self.held.iter().map(|(&entity, held)| (entity, &held.state))
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

    pub fn slot_generation(&self, slot: u32) -> Option<u32> {
        self.generations.get(slot as usize).copied()
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
        let seen = Seen {
            residency: residency.revision(),
            topology: globals.topology_revision(),
            words: globals.words_revision(),
            stride,
        };
        let mut touched = std::mem::take(&mut self.touched);
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
        };
        for entity in touched {
            match world.get::<Drawn>(entity) {
                Some(drawn) => self.refresh(entity, drawn, &cx),
                None => self.forget(entity),
            }
        }
        if self.plan == PlanState::Stale {
            self.rebuild();
        }
    }

    fn refresh(&mut self, entity: Entity, drawn: &Drawn, cx: &Context<'_>) {
        let previous = self.held.remove(&entity);
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
        let state = match (self.refusal(drawn, cx), slot) {
            (None, Some(slot)) => {
                self.write(slot, drawn, cx);
                DrawnState::Drawn { slot }
            }
            (None, None) => {
                let slot = self.take(entity);
                self.write(slot, drawn, cx);
                DrawnState::Drawn { slot }
            }
            (Some(state), Some(slot)) => {
                self.release(slot);
                state
            }
            (Some(state), None) => state,
        };
        self.held.insert(
            entity,
            Held {
                state,
                drawn: drawn.clone(),
                uses,
            },
        );
    }

    fn refusal(&self, drawn: &Drawn, cx: &Context<'_>) -> Option<DrawnState> {
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
        if !cx.globals.drawn(drawn.node) {
            return Some(DrawnState::Hidden);
        }
        if cx
            .globals
            .get(drawn.node)
            .and_then(|global| global.matrix())
            .is_none()
        {
            return Some(DrawnState::Unplaced);
        }
        for (asset, path) in [
            (Asset::Mesh(drawn.geometry), "$.drawn.geometry"),
            (Asset::Material(drawn.material), "$.drawn.material"),
        ] {
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
        let Some(mesh) = cx.library.mesh(drawn.geometry) else {
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
        if let Some(Held {
            state: DrawnState::Drawn { slot },
            ..
        }) = self.held.remove(&entity)
        {
            self.release(slot);
        }
    }

    fn write(&mut self, slot: u32, drawn: &Drawn, cx: &Context<'_>) {
        let at = slot as usize;
        let (min, max) = cx
            .library
            .mesh(drawn.geometry)
            .map_or(([0.0; 3], [0.0; 3]), |mesh| {
                let bounds = mesh.bounds();
                (bounds.min().to_array(), bounds.max().to_array())
            });
        let passes = drawn.passes.unwrap_or_else(|| {
            cx.passes
                .map_or(PassMask::NONE, |passes| passes.defaults(drawn.kind))
        });
        let projection = drawn.projection.map_or([1.0, 1.0, 0.0, 0.0], |own| {
            let (sine, cosine) = sin_cos(own.vertical_fov * 0.5);
            [cosine / sine, own.depth_scale, 0.0, 0.0]
        });
        let mut flags = 0;
        if drawn.receives_shadows {
            flags |= DRAWN_RECEIVES_SHADOWS;
        }
        if drawn.projection.is_some() {
            flags |= DRAWN_OWN_PROJECTION;
        }
        let [red, green, blue] = drawn.values.emission;
        let record = DrawnRecord {
            tint: drawn.values.tint,
            emission: [red, green, blue, 0.0],
            projection,
            bounds_min: [min[0], min[1], min[2], 0.0],
            bounds_max: [max[0], max[1], max[2], 0.0],
            node: cx
                .globals
                .node_index(drawn.node)
                .map_or(u32::MAX, |index| index as u32),
            geometry: drawn.geometry.slot() as u32,
            material: drawn.material.slot() as u32,
            passes: passes.word(),
            layers: drawn.layers.0,
            flags,
            generation: self.generations[at],
            order: i32::from(drawn.order),
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

    fn rebuild(&mut self) {
        self.ordered.clear();
        self.ordered.extend(
            self.owners
                .iter()
                .enumerate()
                .filter(|(_, owner)| owner.is_some())
                .map(|(slot, _)| slot as u32),
        );
        let keys: Vec<Option<(i16, DrawKind, usize, usize)>> = self
            .owners
            .iter()
            .map(|owner| {
                owner
                    .and_then(|entity| self.held.get(&entity))
                    .map(|held| batch_key(&held.drawn))
            })
            .collect();
        self.ordered
            .sort_unstable_by_key(|&slot| (keys[slot as usize], slot));
        self.batches.clear();
        let mut start = 0;
        for at in 1..=self.ordered.len() {
            let first = self.ordered[start] as usize;
            if at == self.ordered.len() || keys[self.ordered[at] as usize] != keys[first] {
                let Some(held) = self.owners[first].and_then(|entity| self.held.get(&entity))
                else {
                    start = at;
                    continue;
                };
                self.batches.push(DrawnBatch {
                    kind: held.drawn.kind,
                    geometry: held.drawn.geometry,
                    material: held.drawn.material,
                    order: held.drawn.order,
                    instances: start as u32..at as u32,
                });
                start = at;
            }
        }
        self.generation += 1;
        self.plan = PlanState::Built {
            generation: self.generation,
        };
    }
}

fn batch_key(drawn: &Drawn) -> (i16, DrawKind, usize, usize) {
    (
        drawn.order,
        drawn.kind,
        drawn.geometry.slot(),
        drawn.material.slot(),
    )
}

fn hold(residency: &AssetResidency, drawn: &Drawn) -> [Interest<Asset>; 2] {
    [
        residency.hold(drawn.geometry),
        residency.hold(drawn.material),
    ]
}

fn touched(insert: On<Insert, Drawn>, mut table: ResMut<DrawnTable>) {
    table.touched.push(insert.entity);
}

fn dropped(remove: On<Remove, Drawn>, mut table: ResMut<DrawnTable>) {
    table.touched.push(remove.entity);
}

pub(crate) fn install(world: &mut World) {
    world.init_resource::<DrawnTable>();
    world.add_observer(touched);
    world.add_observer(dropped);
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
    world.resource_mut::<DrawnTable>().kinds = listed;
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
    if !world.contains_resource::<DrawnTable>() {
        return;
    }
    world.resource_scope(|world, mut table: Mut<DrawnTable>| table.settle(world, &settings));
}
