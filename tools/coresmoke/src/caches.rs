//! The incremental caches against fresh engines. One engine lives through
//! seeded random edits (spawns, despawns of whole subtrees, moves,
//! reparenting, teleports, meshes, cubes, tints, interpolation and drifting
//! switched on and off) before and after each fixed tick, in which drifting
//! entities move, and a frame at a random fraction after each tick. Every
//! `CHECK_EVERY` ticks a fresh engine is built from its state before the
//! tick, both apply the tick's edits and run it, and their world matrices,
//! interpolated poses and render snapshots at fractions 0, 0.37 and 1 must
//! agree to the bit: what the caches kept up incrementally is what
//! computing from scratch gives.

use engine::collections::HashMap;
use std::fmt::Write as _;

use engine_core::bevy_ecs::prelude::*;
use engine_core::glam::{Quat, Vec3};
use engine_core::{
    Cube, Engine, EngineConfig, FixedSimulation, GlobalTransforms, InterpolateTransform, Material,
    MaterialHandle, Mesh, Mesh3d, MeshHandle, Parent, PresentationMode, PresentedTransforms,
    RenderSnapshot, Tint, Transform, reset_interpolation, set_parent, set_parents,
};

use crate::Report;
use crate::random::Random;

/// The seed of the edits, unless `--seed` names another.
pub const SEED: u64 = 0x5eed_c0de;
const ENTITIES: usize = 2000;
const TICKS: u64 = 200;
const CHECK_EVERY: u64 = 4;
const EDITS_BEFORE_TICK: usize = 24;
const EDITS_AFTER_TICK: usize = 6;
const FRACTIONS: [f32; 3] = [0.0, 0.37, 1.0];
const ASPECT: f32 = 16.0 / 9.0;
/// Mismatches a failed check lists at most.
const SHOWN: usize = 8;

/// The meshes and materials both engines hold, in the same slots.
struct Assets {
    meshes: Vec<MeshHandle>,
    materials: Vec<MaterialHandle>,
}

impl Assets {
    fn add(engine: &mut Engine) -> Result<Self, String> {
        let mut library = engine
            .app
            .world_mut()
            .resource_mut::<engine_core::AssetLibrary>();
        let meshes = [Mesh::cube(), Mesh::plane(), Mesh::octahedron()]
            .into_iter()
            .map(|mesh| library.insert_mesh(mesh).map_err(|error| error.to_string()))
            .collect::<Result<_, _>>()?;
        // Two opaque materials and a transparent one.
        let materials = [
            ([0.8, 0.3, 0.2, 1.0], 1.0),
            ([0.2, 0.6, 0.9, 1.0], 1.0),
            ([0.9, 0.9, 0.9, 1.0], 0.5),
        ]
        .into_iter()
        .map(|(color, opacity)| {
            Material::new(color)
                .and_then(|material| material.with_opacity(opacity))
                .and_then(|material| library.insert_material(material))
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
        Ok(Self { meshes, materials })
    }
}

/// What an entity is authored with: everything a fresh engine needs to
/// hold it the same way.
#[derive(Clone, Copy, Debug)]
struct Authored {
    transform: Transform,
    /// Its mesh and material, by their indices in [`Assets`].
    mesh: Option<(usize, usize)>,
    cube: Option<[f32; 4]>,
    tint: Option<[f32; 4]>,
    interpolate: bool,
    drift: Option<Drift>,
    parent: Option<Entity>,
}

/// Moves its entity every fixed tick: what interpolation presents between
/// ticks.
#[derive(Component, Clone, Copy, Debug)]
struct Drift {
    /// Meters a tick.
    velocity: Vec3,
    /// Radians a tick about the entity's own y.
    spin: f32,
}

fn drift(mut drifting: Query<(&mut Transform, &Drift)>) {
    for (mut transform, drift) in &mut drifting {
        transform.translation += drift.velocity;
        transform.rotation = (transform.rotation * Quat::from_rotation_y(drift.spin)).normalize();
    }
}

/// One edit of the world, naming the living engine's entities.
#[derive(Clone, Copy, Debug)]
enum Edit {
    Spawn(Authored),
    /// The entity and its descendants.
    Despawn(Entity),
    Move(Entity, Transform),
    Parent(Entity, Option<Entity>),
    Mesh(Entity, Option<(usize, usize)>),
    Cube(Entity, Option<[f32; 4]>),
    Tint(Entity, Option<[f32; 4]>),
    Interpolate(Entity, bool),
    Drift(Entity, Option<Drift>),
    Teleport(Entity),
}

fn engine() -> Result<(Engine, Assets), String> {
    let mut engine = Engine::new(EngineConfig::default());
    engine.app.add_systems(FixedSimulation, drift);
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
        cube: random.chance(20).then(|| color(random)),
        tint: random.chance(20).then(|| color(random)),
        interpolate: random.chance(50),
        drift: random.chance(40).then(|| drifting(random)),
        parent,
    }
}

/// The entities the engine holds with transforms, in its own order.
fn entities(world: &mut World) -> Vec<Entity> {
    world
        .query_filtered::<Entity, With<Transform>>()
        .iter(world)
        .collect()
}

/// What `world` holds of `entity`, its mesh and material by their indices
/// in `assets`, the world's own.
fn read(world: &World, assets: &Assets, entity: Entity) -> Authored {
    let mesh = world.get::<Mesh3d>(entity).map(|mesh| {
        (
            index(&assets.meshes, mesh.mesh),
            index(&assets.materials, mesh.material),
        )
    });
    Authored {
        transform: *world
            .get::<Transform>(entity)
            .expect("entities read hold a transform"),
        mesh,
        cube: world.get::<Cube>(entity).map(|cube| cube.color),
        tint: world.get::<Tint>(entity).map(|tint| tint.0),
        interpolate: world.get::<InterpolateTransform>(entity).is_some(),
        drift: world.get::<Drift>(entity).copied(),
        parent: world.get::<Parent>(entity).map(Parent::parent),
    }
}

/// Where `handle` is in `handles`: an asset the smoke made.
fn index<T: PartialEq>(handles: &[T], handle: T) -> usize {
    handles
        .iter()
        .position(|held| *held == handle)
        .expect("entities draw the smoke's own meshes and materials")
}

/// Spawns `authored` without its parent, which callers set.
fn spawn(world: &mut World, assets: &Assets, authored: &Authored) -> Entity {
    let mut entity = world.spawn(authored.transform);
    if let Some((mesh, material)) = authored.mesh {
        entity.insert(Mesh3d {
            mesh: assets.meshes[mesh],
            material: assets.materials[material],
        });
    }
    if let Some(color) = authored.cube {
        entity.insert(Cube { color });
    }
    if let Some(tint) = authored.tint {
        entity.insert(Tint(tint));
    }
    if authored.interpolate {
        entity.insert(InterpolateTransform);
    }
    if let Some(drift) = authored.drift {
        entity.insert(drift);
    }
    entity.id()
}

/// `entity` and its descendants, parents first.
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

/// Applies `edit` to `world`, its entities named through `map`; the entity
/// a spawn made.
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
                // A parent that would close a cycle, or nest too deep, is
                // refused alike in both engines.
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
            let Ok(mut held) = world.get_entity_mut(map(entity)) else {
                return None;
            };
            match mesh {
                Some((mesh, material)) => {
                    held.insert(Mesh3d {
                        mesh: assets.meshes[mesh],
                        material: assets.materials[material],
                    });
                }
                None => {
                    held.remove::<Mesh3d>();
                }
            }
        }
        Edit::Cube(entity, cube) => {
            let Ok(mut held) = world.get_entity_mut(map(entity)) else {
                return None;
            };
            match cube {
                Some(color) => {
                    held.insert(Cube { color });
                }
                None => {
                    held.remove::<Cube>();
                }
            }
        }
        Edit::Tint(entity, tint) => {
            let Ok(mut held) = world.get_entity_mut(map(entity)) else {
                return None;
            };
            match tint {
                Some(tint) => {
                    held.insert(Tint(tint));
                }
                None => {
                    held.remove::<Tint>();
                }
            }
        }
        Edit::Interpolate(entity, on) => {
            let Ok(mut held) = world.get_entity_mut(map(entity)) else {
                return None;
            };
            if on {
                held.insert(InterpolateTransform);
            } else {
                held.remove::<InterpolateTransform>();
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
            reset_interpolation(world, map(entity));
        }
    }
    None
}

/// `count` edits of the living engine's `entities`.
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
            66..=71 => {
                let cube = random.chance(60).then(|| color(random));
                Edit::Cube(any(random), cube)
            }
            72..=79 => {
                let tint = random.chance(60).then(|| color(random));
                Edit::Tint(any(random), tint)
            }
            80..=87 => Edit::Interpolate(any(random), random.chance(50)),
            88..=93 => {
                let drift = random.chance(70).then(|| drifting(random));
                Edit::Drift(any(random), drift)
            }
            _ => Edit::Teleport(any(random)),
        })
        .collect()
}

/// Builds a fresh engine holding what `living` holds now; the living
/// engine's entities mapped to the fresh one's.
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

/// One engine's frame at `alpha`: each entity's world and presented
/// matrices, and the snapshot's instances, sorted, with its counts.
struct Frame {
    globals: HashMap<Entity, Option<[u32; 16]>>,
    presented: HashMap<Entity, Option<[u32; 16]>>,
    instances: Vec<String>,
    counts: [usize; 4],
}

fn bits(matrix: engine_core::glam::Mat4) -> [u32; 16] {
    matrix.to_cols_array().map(f32::to_bits)
}

fn frame(engine: &mut Engine, alpha: f32, snapshot: &mut RenderSnapshot) -> Result<Frame, String> {
    engine
        .extract_with_presentation(ASPECT, snapshot, PresentationMode::Interpolated { alpha })
        .map_err(|error| format!("{error:?}"))?;
    let world = engine.app.world();
    let globals = world
        .resource::<GlobalTransforms>()
        .iter()
        .map(|(entity, global)| (entity, global.matrix().map(bits)))
        .collect();
    let presented = world
        .resource::<PresentedTransforms>()
        .iter()
        .map(|(entity, pose)| (entity, pose.matrix().map(bits)))
        .collect();
    let mut instances: Vec<String> = snapshot
        .instances
        .iter()
        .map(|instance| {
            format!(
                "{} {:?} {:?} {:?} {:?} {:?}",
                instance.mesh.slot(),
                instance.material.map(MaterialHandle::slot),
                bits(instance.model),
                instance.color.map(f32::to_bits),
                instance.surface,
                instance.skin,
            )
        })
        .collect();
    instances.sort_unstable();
    Ok(Frame {
        globals,
        presented,
        instances,
        counts: [
            snapshot.instances.len(),
            snapshot.culled_instances,
            snapshot.dropped_instances,
            snapshot.missing_assets,
        ],
    })
}

/// A matrix's bits as its numbers, or that it is invalid.
fn readable(bits: &Option<[u32; 16]>) -> String {
    bits.map_or_else(
        || "invalid".to_owned(),
        |bits| format!("{:?}", bits.map(f32::from_bits)),
    )
}

/// Where the living engine's `frame` and the fresh one's (its entities
/// named through `map`) disagree.
fn differences(
    living: &Frame,
    fresh: &Frame,
    map: &HashMap<Entity, Entity>,
    alpha: f32,
) -> Vec<String> {
    let mut found = Vec::new();
    for (what, living, fresh) in [
        ("world matrix", &living.globals, &fresh.globals),
        ("presented matrix", &living.presented, &fresh.presented),
    ] {
        if living.len() != fresh.len() {
            found.push(format!(
                "{what}s at {alpha}: {} entities against a fresh engine's {}",
                living.len(),
                fresh.len()
            ));
        }
        for (entity, value) in living {
            let other = map.get(entity).and_then(|entity| fresh.get(entity));
            if other != Some(value) {
                let other = other.map_or_else(|| "none".to_owned(), readable);
                found.push(format!(
                    "{what} of {entity:?} at {alpha}: {}, a fresh engine's {other}",
                    readable(value)
                ));
            }
        }
    }
    if living.counts != fresh.counts {
        found.push(format!(
            "snapshot counts (instances, culled, dropped, missing) at {alpha}: {:?}, a fresh \
             engine's {:?}",
            living.counts, fresh.counts
        ));
    }
    if living.instances != fresh.instances {
        let differ = living
            .instances
            .iter()
            .zip(&fresh.instances)
            .find(|(a, b)| a != b);
        found.push(format!(
            "snapshot instances at {alpha} differ, first {differ:?}"
        ));
    }
    found
}

/// The living engine, and the fresh one built at a check with its
/// entities mapped from the living one's.
type Fresh = Option<(Engine, Assets, HashMap<Entity, Entity>)>;

/// Applies `edits` to the living engine and the fresh one, if any, logging
/// each with `when`.
fn apply_both(
    living: &mut Engine,
    assets: &Assets,
    fresh: &mut Fresh,
    edits: &[Edit],
    log: &mut String,
    when: &str,
) {
    for edit in edits {
        let _ = writeln!(log, "{when}: {edit:?}");
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

/// A tick's edits: `count` of them, of the entities the engine holds.
fn edits_of(random: &mut Random, engine: &mut Engine, count: usize) -> Vec<Edit> {
    let held = entities(engine.app.world_mut());
    if held.is_empty() {
        Vec::new()
    } else {
        edits(random, &held, count)
    }
}

pub fn run(seed: u64, report: &mut Report) -> Result<(), String> {
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
    let mut snapshot = RenderSnapshot::default();
    let mut fresh_snapshot = RenderSnapshot::default();
    let mut log = String::new();
    let mut checks = 0;
    let mut mismatches = 0;
    let mut since = 0;
    for tick in 1..=TICKS {
        let mut fresh: Fresh = if tick % CHECK_EVERY == 0 {
            Some(rebuild(&mut living, &assets)?)
        } else {
            None
        };
        // Edits before the tick, which it simulates from; then between it
        // and the frame, which presents them.
        let before = edits_of(&mut random, &mut living, EDITS_BEFORE_TICK);
        apply_both(
            &mut living,
            &assets,
            &mut fresh,
            &before,
            &mut log,
            &format!("before tick {tick}"),
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
            &format!("after tick {tick}"),
        );
        let Some((mut engine, _, map)) = fresh else {
            let alpha = random.range(0.0, 1.0);
            frame(&mut living, alpha, &mut snapshot)?;
            continue;
        };
        checks += 1;
        let mut found = Vec::new();
        for alpha in FRACTIONS {
            let living_frame = frame(&mut living, alpha, &mut snapshot)?;
            let fresh_frame = frame(&mut engine, alpha, &mut fresh_snapshot)?;
            found.extend(differences(&living_frame, &fresh_frame, &map, alpha));
        }
        if !found.is_empty() {
            mismatches += 1;
            let shown: Vec<_> = found.iter().take(SHOWN).cloned().collect();
            report.broke(format!(
                "caches: seed {seed}, tick {tick}: the living engine disagrees with a fresh \
                 one in {} place(s): {}; edits since tick {since}:\n{log}",
                found.len(),
                shown.join("; ")
            ));
        }
        log.clear();
        since = tick;
    }
    let world = living.app.world_mut();
    report.record("caches checks", checks);
    report.record("caches mismatches", mismatches);
    report.record("caches entities", entities(world).len());
    Ok(())
}
