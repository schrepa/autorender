use engine::bevy_ecs::prelude::*;
use engine_core::collections::HashMap;
use engine_core::glam::{Quat, Vec3};
use engine_core::{
    Camera, Drawn, Engine, GeometryHandle, GlobalTransforms, Listing, Material, MaterialHandle,
    Mesh, MeshVertex, Mover, MoverId, Operation, Pace, Parent, PureSession, Schedule, Transform,
    make_fresh, set_parent, set_parents,
};
use engine_meshes::{MESH, MeshesBrick, MeshesListing};

use serde_json::{Value, json};

use super::random::Random;

pub const SEED: u64 = 0x5eed_c0de;
const ENTITIES: usize = 2000;
const TICKS: u64 = 200;
const CHECK_EVERY: u64 = 4;
const EDITS_BEFORE_TICK: usize = 24;
const EDITS_AFTER_TICK: usize = 6;
const SHOWN: usize = 8;
const MOVER: Mover = Mover {
    by: MoverId::Game,
    pace: Pace::Tick,
};

struct Assets {
    meshes: Vec<GeometryHandle>,
    materials: Vec<MaterialHandle>,
}

impl Assets {
    fn add(engine: &mut Engine) -> Result<Self, String> {
        let mut library = engine
            .app
            .world_mut()
            .resource_mut::<engine_core::AssetLibrary>();
        let meshes = [Mesh::cube(), Mesh::plane(), octahedron()?]
            .into_iter()
            .map(|mesh| library.insert_mesh(mesh).map_err(|error| error.to_string()))
            .collect::<Result<_, _>>()?;
        let materials = [
            [0.8, 0.3, 0.2, 1.0],
            [0.2, 0.6, 0.9, 1.0],
            [0.9, 0.9, 0.9, 1.0],
        ]
        .into_iter()
        .map(|color| {
            Material::new(color)
                .and_then(|material| library.insert_material(material))
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
        Ok(Self { meshes, materials })
    }
}

fn octahedron() -> Result<Mesh, String> {
    let mut vertices = Vec::with_capacity(24);
    for y in [-1.0, 1.0] {
        for (a, b) in [
            (Vec3::X, Vec3::Z),
            (Vec3::Z, Vec3::NEG_X),
            (Vec3::NEG_X, Vec3::NEG_Z),
            (Vec3::NEG_Z, Vec3::X),
        ] {
            let tip = Vec3::Y * y;
            let (a, b) = if y > 0.0 { (b, a) } else { (a, b) };
            let normal = (a - tip).cross(b - tip).normalize().to_array();
            for (position, uv) in [tip, a, b]
                .into_iter()
                .zip([[0.5, 0.0], [0.0, 1.0], [1.0, 1.0]])
            {
                vertices.push(MeshVertex {
                    position: (position * 0.5).to_array(),
                    normal,
                    uv,
                });
            }
        }
    }
    let indices = (0..vertices.len() as u32).collect();
    Mesh::new(vertices, indices).map_err(|error| error.to_string())
}

#[derive(Clone, Copy, Debug)]
struct Authored {
    transform: Transform,
    mesh: Option<(usize, usize)>,
    tint: [f32; 4],
    moved: bool,
    drift: Option<Drift>,
    parent: Option<Entity>,
}

#[derive(Component, Clone, Copy, Debug)]
struct Drift {
    velocity: Vec3,
    spin: f32,
}

fn drift(mut drifting: Query<(&mut Transform, &Drift)>) {
    for (mut transform, drift) in &mut drifting {
        transform.translation += drift.velocity;
        transform.rotation = (transform.rotation * Quat::from_rotation_y(drift.spin)).normalize();
    }
}

#[derive(Clone, Copy, Debug)]
enum Edit {
    Spawn(Authored),
    Despawn(Entity),
    Move(Entity, Transform),
    Parent(Entity, Option<Entity>),
    Mesh(Entity, Option<(usize, usize)>),
    Tint(Entity, Option<[f32; 4]>),
    Moved(Entity, bool),
    Drift(Entity, Option<Drift>),
    Teleport(Entity),
}

fn engine() -> Result<(Engine, Assets), String> {
    let mut engine = engine::engine(|list| {
        list.with::<MeshesBrick>()
            .places(engine_places::PLACES)
            .places(engine_places_rendering::PLACES)
            .places(engine_places_simulation::PLACES)
            .places(engine_places_world::PLACES)
            .places(engine_open_world_kit::PLACES)
    })
    .map_err(|error| error.to_string())?;
    engine.set_frame(super::FRAME);
    engine.app.add_systems(
        Schedule::FixedSimulation.label(),
        drift.ambiguous_with_all(),
    );
    let assets = Assets::add(&mut engine)?;
    Ok((engine, assets))
}

fn drifting(random: &mut Random) -> Drift {
    Drift {
        velocity: Vec3::new(
            random.range(-0.5, 0.5),
            random.range(-0.2, 0.2),
            random.range(-0.5, 0.5),
        ),
        spin: random.range(-0.2, 0.2),
    }
}

fn transform(random: &mut Random) -> Transform {
    let axis = Vec3::new(
        random.range(-1.0, 1.0),
        random.range(-1.0, 1.0),
        random.range(-1.0, 1.0),
    )
    .try_normalize()
    .unwrap_or(Vec3::Y);
    Transform {
        translation: Vec3::new(
            random.range(-60.0, 60.0),
            random.range(-10.0, 30.0),
            random.range(-60.0, 60.0),
        ),
        rotation: Quat::from_axis_angle(axis, random.range(-3.0, 3.0)),
        scale: Vec3::splat(random.range(0.5, 2.0)),
    }
}

fn color(random: &mut Random) -> [f32; 4] {
    [
        random.range(0.0, 1.0),
        random.range(0.0, 1.0),
        random.range(0.0, 1.0),
        1.0,
    ]
}

fn authored(random: &mut Random, parent: Option<Entity>) -> Authored {
    Authored {
        transform: transform(random),
        mesh: random
            .chance(70)
            .then(|| (random.below(3), random.below(3))),
        tint: if random.chance(20) {
            color(random)
        } else {
            [1.0; 4]
        },
        moved: random.chance(50),
        drift: random.chance(40).then(|| drifting(random)),
        parent,
    }
}

fn entities(world: &mut World) -> Vec<Entity> {
    world
        .query_filtered::<Entity, (With<Transform>, Without<Camera>)>()
        .iter(world)
        .collect()
}

fn read(world: &World, assets: &Assets, entity: Entity) -> Authored {
    let drawn = world.get::<Drawn>(entity);
    let mesh = drawn
        .and_then(|drawn| drawn.geometry.zip(drawn.material))
        .map(|(mesh, material)| {
            (
                index(&assets.meshes, mesh),
                index(&assets.materials, material),
            )
        });
    Authored {
        transform: *world
            .get::<Transform>(entity)
            .expect("entity_has_transform"),
        mesh,
        tint: drawn.map_or([1.0; 4], |drawn| drawn.values.tint),
        moved: world.get::<Mover>(entity).is_some(),
        drift: world.get::<Drift>(entity).copied(),
        parent: world.get::<Parent>(entity).map(Parent::parent),
    }
}

fn index<T: PartialEq>(handles: &[T], handle: T) -> usize {
    handles
        .iter()
        .position(|held| *held == handle)
        .expect("handle_owned_by_smoke")
}

fn spawn(world: &mut World, assets: &Assets, authored: &Authored) -> Entity {
    let mut entity = world.spawn(authored.transform);
    if let Some((mesh, material)) = authored.mesh {
        let mut drawn = Drawn::new(MESH, assets.meshes[mesh], assets.materials[material]);
        drawn.values.tint = authored.tint;
        entity.insert(drawn);
    }
    if authored.moved {
        entity.insert(MOVER);
    }
    if let Some(drift) = authored.drift {
        entity.insert(drift);
    }
    entity.id()
}

fn subtree(world: &mut World, entity: Entity) -> Vec<Entity> {
    let mut children: HashMap<Entity, Vec<Entity>> = HashMap::default();
    for (child, parent) in world.query::<(Entity, &Parent)>().iter(world) {
        children.entry(parent.parent()).or_default().push(child);
    }
    let mut found = vec![entity];
    let mut next = 0;
    while let Some(&parent) = found.get(next) {
        found.extend(children.get(&parent).into_iter().flatten().copied());
        next += 1;
    }
    found
}

fn apply(
    world: &mut World,
    assets: &Assets,
    edit: &Edit,
    map: &dyn Fn(Entity) -> Entity,
) -> Option<Entity> {
    match *edit {
        Edit::Spawn(authored) => {
            let entity = spawn(world, assets, &authored);
            if let Some(parent) = authored.parent {
                let _ = set_parent(world, entity, Some(map(parent)));
            }
            return Some(entity);
        }
        Edit::Despawn(entity) => {
            let entity = map(entity);
            if world.get_entity(entity).is_ok() {
                for entity in subtree(world, entity) {
                    world.despawn(entity);
                }
            }
        }
        Edit::Move(entity, transform) => {
            if let Some(mut held) = world.get_mut::<Transform>(map(entity)) {
                *held = transform;
            }
        }
        Edit::Parent(entity, parent) => {
            let _ = set_parent(world, map(entity), parent.map(map));
        }
        Edit::Mesh(entity, mesh) => {
            let entity = map(entity);
            let tint = world
                .get::<Drawn>(entity)
                .map_or([1.0; 4], |drawn| drawn.values.tint);
            let Ok(mut held) = world.get_entity_mut(entity) else {
                return None;
            };
            match mesh {
                Some((mesh, material)) => {
                    let mut drawn =
                        Drawn::new(MESH, assets.meshes[mesh], assets.materials[material]);
                    drawn.values.tint = tint;
                    held.insert(drawn);
                }
                None => {
                    held.remove::<Drawn>();
                }
            }
        }
        Edit::Tint(entity, tint) => {
            if let Some(mut drawn) = world.get_mut::<Drawn>(map(entity)) {
                drawn.values.tint = tint.unwrap_or([1.0; 4]);
            }
        }
        Edit::Moved(entity, on) => {
            let Ok(mut held) = world.get_entity_mut(map(entity)) else {
                return None;
            };
            if on {
                held.insert(MOVER);
            } else {
                held.remove::<Mover>();
            }
        }
        Edit::Drift(entity, drift) => {
            let Ok(mut held) = world.get_entity_mut(map(entity)) else {
                return None;
            };
            match drift {
                Some(drift) => {
                    held.insert(drift);
                }
                None => {
                    held.remove::<Drift>();
                }
            }
        }
        Edit::Teleport(entity) => {
            make_fresh(world, &[map(entity)]);
        }
    }
    None
}

fn edits(random: &mut Random, entities: &[Entity], count: usize) -> Vec<Edit> {
    let any = |random: &mut Random| entities[random.below(entities.len())];
    (0..count)
        .map(|_| match random.below(100) {
            0..=29 => Edit::Move(any(random), transform(random)),
            30..=41 => {
                let parent = random.chance(70).then(|| any(random));
                Edit::Parent(any(random), parent)
            }
            42..=51 => {
                let parent = random.chance(50).then(|| any(random));
                Edit::Spawn(authored(random, parent))
            }
            52..=55 => Edit::Despawn(any(random)),
            56..=65 => {
                let mesh = random
                    .chance(80)
                    .then(|| (random.below(3), random.below(3)));
                Edit::Mesh(any(random), mesh)
            }
            66..=79 => {
                let tint = random.chance(60).then(|| color(random));
                Edit::Tint(any(random), tint)
            }
            80..=87 => Edit::Moved(any(random), random.chance(50)),
            88..=93 => {
                let drift = random.chance(70).then(|| drifting(random));
                Edit::Drift(any(random), drift)
            }
            _ => Edit::Teleport(any(random)),
        })
        .collect()
}

fn rebuild(
    living: &mut Engine,
    living_assets: &Assets,
) -> Result<(Engine, Assets, HashMap<Entity, Entity>), String> {
    let (mut fresh, assets) = engine()?;
    let world = living.app.world_mut();
    let held: Vec<(Entity, Authored)> = entities(world)
        .into_iter()
        .map(|entity| (entity, read(world, living_assets, entity)))
        .collect();
    let target = fresh.app.world_mut();
    let mut map = HashMap::with_capacity_and_hasher(held.len(), Default::default());
    for (entity, authored) in &held {
        map.insert(*entity, spawn(target, &assets, authored));
    }
    let parents: Vec<(Entity, Option<Entity>)> = held
        .iter()
        .filter_map(|(entity, authored)| {
            authored
                .parent
                .map(|parent| (map[entity], Some(map[&parent])))
        })
        .collect();
    set_parents(target, &parents).map_err(|error| format!("{error:?}"))?;
    Ok((fresh, assets, map))
}

struct Frame {
    globals: HashMap<Entity, Option<[u32; 16]>>,
    instances: Vec<String>,
    counts: [usize; 2],
}

fn bits(matrix: engine_core::glam::Mat4) -> [u32; 16] {
    matrix.to_cols_array().map(f32::to_bits)
}

fn frame(engine: &mut Engine) -> Result<Frame, String> {
    let world = engine
        .extract_frame()
        .map_err(|error| format!("extract_frame: {error}"))?
        .world();
    let globals = world
        .resource::<GlobalTransforms>()
        .iter()
        .map(|(entity, global)| (entity, global.matrix().map(bits)))
        .collect();
    let retained = engine::host::call::<MeshesListing>(engine.app.world_mut(), MeshesListing {})
        .map_err(|error| format!("{}: {error}", MeshesListing::NAME))?;
    let mut instances: Vec<String> = retained
        .instances
        .iter()
        .map(|instance| serde_json::to_string(instance).unwrap_or_default())
        .collect();
    instances.sort_unstable();
    Ok(Frame {
        globals,
        counts: [instances.len(), retained.missing as usize],
        instances,
    })
}

fn readable(bits: &Option<[u32; 16]>) -> String {
    bits.map_or_else(
        || "invalid".to_owned(),
        |bits| format!("{:?}", bits.map(f32::from_bits)),
    )
}

fn differences(living: &Frame, fresh: &Frame, map: &HashMap<Entity, Entity>) -> Vec<Value> {
    let mut found = Vec::new();
    if living.globals.len() != fresh.globals.len() {
        found.push(json!({
            "part": "world_matrix",
            "check": "entity_count",
            "living": living.globals.len(),
            "fresh": fresh.globals.len(),
        }));
    }
    for (entity, value) in &living.globals {
        let other = map.get(entity).and_then(|entity| fresh.globals.get(entity));
        if other != Some(value) {
            let other = other.map_or_else(|| "none".to_owned(), readable);
            found.push(json!({
                "part": "world_matrix",
                "entity": format!("{entity:?}"),
                "living": readable(value),
                "fresh": other,
            }));
        }
    }
    if living.counts != fresh.counts {
        found.push(json!({
            "part": "snapshot_counts",
            "fields": ["instances", "missing"],
            "living": living.counts,
            "fresh": fresh.counts,
        }));
    }
    if living.instances != fresh.instances {
        let differ = living
            .instances
            .iter()
            .zip(&fresh.instances)
            .find(|(a, b)| a != b);
        found.push(json!({
            "part": "snapshot_instances",
            "living": differ.map(|(living, _)| living),
            "fresh": differ.map(|(_, fresh)| fresh),
        }));
    }
    found
}

type Fresh = Option<(Engine, Assets, HashMap<Entity, Entity>)>;

fn apply_both(
    living: &mut Engine,
    assets: &Assets,
    fresh: &mut Fresh,
    edits: &[Edit],
    log: &mut Vec<Value>,
    (phase, tick): (&str, u64),
) {
    for edit in edits {
        log.push(json!({"phase": phase, "tick": tick, "edit": format!("{edit:?}")}));
        let spawned = apply(living.app.world_mut(), assets, edit, &|entity| entity);
        if let Some((engine, assets, map)) = fresh.as_mut() {
            let lookup = |entity| map.get(&entity).copied().unwrap_or(entity);
            let made = apply(engine.app.world_mut(), assets, edit, &lookup);
            if let (Some(living), Some(made)) = (spawned, made) {
                map.insert(living, made);
            }
        }
    }
}

fn edits_of(random: &mut Random, engine: &mut Engine, count: usize) -> Vec<Edit> {
    let held = entities(engine.app.world_mut());
    if held.is_empty() {
        Vec::new()
    } else {
        edits(random, &held, count)
    }
}

pub fn run(seed: u64, session: &mut dyn PureSession) -> Result<(), String> {
    let mut random = Random::new(seed);
    let (mut living, assets) = engine()?;
    {
        let world = living.app.world_mut();
        let mut spawned: Vec<Entity> = Vec::with_capacity(ENTITIES);
        for _ in 0..ENTITIES {
            let parent = (!spawned.is_empty() && random.chance(40))
                .then(|| spawned[random.below(spawned.len())]);
            let authored = authored(&mut random, parent);
            let entity = spawn(world, &assets, &authored);
            if let Some(parent) = parent {
                set_parent(world, entity, Some(parent)).map_err(|error| format!("{error:?}"))?;
            }
            spawned.push(entity);
        }
    }
    let mut log = Vec::new();
    let mut checks = 0;
    let mut mismatches = 0;
    let mut since = 0;
    for tick in 1..=TICKS {
        let mut fresh: Fresh = if tick % CHECK_EVERY == 0 {
            Some(rebuild(&mut living, &assets)?)
        } else {
            None
        };
        let before = edits_of(&mut random, &mut living, EDITS_BEFORE_TICK);
        apply_both(
            &mut living,
            &assets,
            &mut fresh,
            &before,
            &mut log,
            ("before", tick),
        );
        living.step_fixed(1).map_err(|error| error.to_string())?;
        if let Some((engine, _, _)) = fresh.as_mut() {
            engine.step_fixed(1).map_err(|error| error.to_string())?;
        }
        let after = edits_of(&mut random, &mut living, EDITS_AFTER_TICK);
        apply_both(
            &mut living,
            &assets,
            &mut fresh,
            &after,
            &mut log,
            ("after", tick),
        );
        let Some((mut engine, _, map)) = fresh else {
            living
                .extract_frame()
                .map_err(|error| format!("extract_frame: {error}"))?;
            continue;
        };
        checks += 1;
        let living_frame = frame(&mut living)?;
        let fresh_frame = frame(&mut engine)?;
        let found = differences(&living_frame, &fresh_frame, &map);
        if !found.is_empty() {
            mismatches += 1;
            let shown: Vec<_> = found.iter().take(SHOWN).cloned().collect();
            let broken = json!({
                "check": "caches_match_fresh",
                "seed": seed,
                "tick": tick,
                "places": found.len(),
                "differences": shown,
                "edits_since_tick": since,
                "edits": log,
            });
            session.check(false, &|| broken.to_string());
        }
        log.clear();
        since = tick;
    }
    let world = living.app.world_mut();
    session.record("caches checks", json!(checks));
    session.record("caches mismatches", json!(mismatches));
    session.record("caches entities", entities(world).len().into());
    Ok(())
}
