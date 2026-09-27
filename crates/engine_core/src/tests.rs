use crate::prelude::*;

#[test]
fn optional_fixed_boundaries_apply_commands_before_gameplay_and_history_capture() {
    let mut engine = engine_at(8.0, 3);
    let parent = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let child = engine
        .app
        .world_mut()
        .spawn(Transform {
            translation: Vec3::X,
            ..Default::default()
        })
        .id();
    set_parent(engine.app.world_mut(), child, Some(parent)).unwrap();
    engine
        .app
        .add_systems(FixedPreSimulation, move |mut commands: Commands| {
            commands.entity(parent).insert(Transform {
                translation: Vec3::X * 2.0,
                ..Default::default()
            });
        });
    engine
        .app
        .add_systems(FixedSimulation, move |world: &mut World| {
            assert_eq!(
                world
                    .resource::<GlobalTransforms>()
                    .get(child)
                    .unwrap()
                    .translation()
                    .unwrap()
                    .x,
                3.0
            );
            edit_transform(world, parent, |transform| {
                transform.translation.x = 4.0;
            })
            .unwrap();
        });
    engine.app.add_systems(
        FixedPostSimulation,
        move |globals: Res<GlobalTransforms>, mut commands: Commands| {
            assert_eq!(globals.get(child).unwrap().translation().unwrap().x, 5.0);
            commands.entity(parent).insert(Transform {
                translation: Vec3::X * 6.0,
                ..Default::default()
            });
        },
    );
    engine.advance(0.125);
    assert_eq!(
        engine
            .app
            .world()
            .resource::<GlobalTransforms>()
            .get(child)
            .unwrap()
            .translation()
            .unwrap()
            .x,
        7.0
    );
    let mut snapshot = RenderSnapshot::default();
    engine
        .extract_with_presentation(
            1.0,
            &mut snapshot,
            PresentationMode::Interpolated { alpha: 0.5 },
        )
        .unwrap();
    assert_eq!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get(parent)
            .unwrap()
            .matrix()
            .unwrap()
            .w_axis
            .x,
        3.0
    );
    assert_eq!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get(child)
            .unwrap()
            .matrix()
            .unwrap()
            .w_axis
            .x,
        4.0
    );
}

#[test]
fn empty_fixed_extension_schedules_add_no_transform_boundaries() {
    #[derive(Resource, Default)]
    struct Propagations(u32);
    let mut engine = engine_at(8.0, 3);
    engine.app.init_resource::<Propagations>().add_systems(
        TransformPropagation,
        |mut count: ResMut<Propagations>| {
            count.0 += 1;
        },
    );
    engine.advance(0.375);
    // Before frame + one per fixed tick + after presentation, as before the
    // optional collision extension schedules were introduced.
    assert_eq!(engine.app.world().resource::<Propagations>().0, 5);
}

fn engine_at(hz: f64, max_fixed_steps: u32) -> Engine {
    Engine::new(EngineConfig {
        fixed_hz: hz,
        max_fixed_steps,
        ..EngineConfig::default()
    })
}

#[test]
fn catch_up_discards_whole_ticks_but_preserves_fraction() {
    let mut engine = engine_at(8.0, 3);
    let first = engine.advance(0.0625);
    assert_eq!(first.fixed_steps, 0);
    assert_eq!(first.interpolation, 0.5);

    let stalled = engine.advance(1.0);
    assert_eq!(stalled.fixed_steps, 3);
    assert_eq!(stalled.dropped_seconds, 0.625);
    assert_eq!(stalled.interpolation, 0.5);
    let time = engine.app.world().resource::<FixedTime>();
    assert_eq!(time.tick, 3);
    assert_eq!(time.elapsed_seconds, 0.375);

    let resumed = engine.advance(0.0625);
    assert_eq!(resumed.fixed_steps, 1);
    assert_eq!(resumed.interpolation, 0.0);
    assert_eq!(resumed.dropped_seconds, 0.0);
}

#[test]
fn decimal_tick_frequency_does_not_leave_a_phantom_tick() {
    let mut engine = engine_at(10.0, 5);
    let stalled = engine.advance(1.0);
    assert_eq!(stalled.fixed_steps, 5);
    assert_eq!(stalled.dropped_seconds, 0.5);
    assert_eq!(stalled.interpolation, 0.0);
    assert_eq!(engine.advance(0.0).fixed_steps, 0);
}

#[test]
fn invalid_deltas_do_not_advance_or_poison_the_clock() {
    let mut engine = engine_at(8.0, 3);
    engine.advance(0.0625);
    for delta in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -10.0] {
        let report = engine.advance(delta);
        assert_eq!(report.fixed_steps, 0);
        assert_eq!(report.interpolation, 0.5);
        assert_eq!(report.dropped_seconds, 0.0);
        assert_eq!(
            engine.app.world().resource::<FrameTime>().delta_seconds,
            0.0
        );
    }
    assert_eq!(engine.advance(0.0625).fixed_steps, 1);
    let extreme = engine.advance(f64::MAX);
    assert_eq!(extreme.fixed_steps, 3);
    assert!(extreme.dropped_seconds.is_finite());
    assert!((0.0..1.0).contains(&extreme.interpolation));
}

#[test]
fn suspension_resets_fraction_without_rewinding_simulation() {
    let mut engine = engine_at(8.0, 3);
    engine.advance(0.1875);
    engine.reset_clock();
    assert_eq!(engine.app.world().resource::<FixedTime>().tick, 1);
    assert_eq!(engine.advance(0.0625).fixed_steps, 0);
    assert_eq!(engine.advance(0.0625).fixed_steps, 1);
}

#[derive(Component)]
struct Spawned;

#[derive(Resource, Default)]
struct Observations {
    fixed_counts: Vec<usize>,
    presentation_count: usize,
}

fn spawn_during_tick(
    mut commands: Commands,
    spawned: Query<&Spawned>,
    mut seen: ResMut<Observations>,
) {
    seen.fixed_counts.push(spawned.iter().count());
    commands.spawn((Spawned, Transform::default(), Cube { color: [1.0; 4] }));
}

fn observe_presentation(spawned: Query<&Spawned>, mut seen: ResMut<Observations>) {
    seen.presentation_count = spawned.iter().count();
}

#[test]
fn deferred_spawns_are_visible_on_next_tick_and_in_presentation() {
    let mut engine = engine_at(8.0, 3);
    engine
        .app
        .init_resource::<Observations>()
        .add_systems(FixedSimulation, spawn_during_tick)
        .add_systems(Presentation, observe_presentation);
    engine.advance(0.25);
    let seen = engine.app.world().resource::<Observations>();
    assert_eq!(seen.fixed_counts, vec![0, 1]);
    assert_eq!(seen.presentation_count, 2);
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.instances.len(), 2);
}

#[derive(Resource, Default)]
struct PresentedFrames(Vec<(u64, f32)>);

#[test]
fn presentation_runs_once_even_without_fixed_work_and_caps_delta() {
    let mut engine = engine_at(8.0, 3);
    engine.app.init_resource::<PresentedFrames>().add_systems(
        Presentation,
        |frame: Res<FrameTime>, mut presented: ResMut<PresentedFrames>| {
            presented.0.push((frame.frame, frame.delta_seconds));
        },
    );
    engine.advance(0.0);
    engine.advance(5.0);
    assert_eq!(
        engine.app.world().resource::<PresentedFrames>().0,
        vec![(1, 0.0), (2, 0.25)]
    );
    assert_eq!(
        engine.app.world().resource::<FrameTime>().elapsed_seconds,
        0.25
    );
}

#[test]
fn extraction_bounds_count_and_reuses_owned_storage() {
    let mut engine = Engine::new(EngineConfig {
        max_render_instances: 2,
        ..EngineConfig::default()
    });
    let mut entities = Vec::new();
    for x in 0..3 {
        entities.push(
            engine
                .app
                .world_mut()
                .spawn((
                    Transform {
                        translation: Vec3::new(x as f32, 0.0, 0.0),
                        ..Transform::default()
                    },
                    Cube { color: [1.0; 4] },
                ))
                .id(),
        );
    }
    let mut snapshot = RenderSnapshot::default();
    engine.extract(16.0 / 9.0, &mut snapshot);
    assert_eq!(snapshot.instances.len(), 2);
    assert_eq!(snapshot.dropped_instances, 1);
    assert!(snapshot.view_projection.is_finite());
    let storage = snapshot.instances.as_ptr();
    let first_model = snapshot.instances[0].model;
    engine.app.world_mut().entity_mut(entities[0]).despawn();
    // The previous snapshot remains an independent, usable copy.
    assert_eq!(snapshot.instances[0].model, first_model);
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.instances.len(), 2);
    assert_eq!(snapshot.dropped_instances, 0);
    assert_eq!(snapshot.instances.as_ptr(), storage);
}

#[test]
fn zero_render_budget_counts_overflow_without_allocating() {
    let mut engine = Engine::new(EngineConfig {
        max_render_instances: 0,
        ..EngineConfig::default()
    });
    engine
        .app
        .world_mut()
        .spawn((Transform::default(), Cube { color: [1.0; 4] }));
    let mut snapshot = RenderSnapshot::default();
    engine.extract(0.0, &mut snapshot);
    assert!(snapshot.instances.is_empty());
    assert_eq!(snapshot.instances.capacity(), 0);
    assert_eq!(snapshot.dropped_instances, 1);
}

#[test]
fn camera_maps_near_and_far_planes_to_webgpu_depth() {
    let camera = Camera {
        position: Vec3::ZERO,
        target: Vec3::NEG_Z,
        near: 0.5,
        far: 10.0,
        ..Camera::default()
    };
    let projection = camera.view_projection(1.0);
    assert!(projection.project_point3(Vec3::new(0.0, 0.0, -0.5)).z.abs() < 1.0e-6);
    assert!((projection.project_point3(Vec3::new(0.0, 0.0, -10.0)).z - 1.0).abs() < 1.0e-6);
}

#[test]
fn mixed_mesh_extraction_deduplicates_sources_and_mesh3d_overrides_cube() {
    let mut engine = Engine::default();
    let (plane, octahedron, material) = {
        let mut assets = engine.app.world_mut().resource_mut::<AssetLibrary>();
        (
            assets.insert_mesh(Mesh::plane()).unwrap(),
            assets.insert_mesh(Mesh::octahedron()).unwrap(),
            assets
                .insert_material(Material::new([0.1, 0.3, 0.8, 1.0]).unwrap())
                .unwrap(),
        )
    };
    engine
        .app
        .world_mut()
        .spawn((Transform::default(), Cube { color: [1.0; 4] }));
    engine.app.world_mut().spawn((
        Transform::default(),
        Mesh3d {
            mesh: plane,
            material,
        },
    ));
    engine.app.world_mut().spawn((
        Transform::default(),
        Mesh3d {
            mesh: plane,
            material,
        },
    ));
    engine.app.world_mut().spawn((
        Transform::default(),
        Mesh3d {
            mesh: octahedron,
            material,
        },
        Cube { color: [1.0; 4] },
    ));
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.instances.len(), 4);
    assert_eq!(snapshot.meshes.len(), 3);
    assert_eq!(
        snapshot
            .instances
            .iter()
            .filter(|instance| instance.mesh == plane)
            .count(),
        2
    );
    assert_eq!(
        snapshot
            .instances
            .iter()
            .find(|instance| instance.mesh == octahedron)
            .unwrap()
            .color,
        [0.1, 0.3, 0.8, 1.0]
    );
    let assets = engine.app.world().resource::<AssetLibrary>();
    assert_eq!(std::sync::Arc::strong_count(assets.mesh(plane).unwrap()), 2);
    assert_eq!(snapshot.missing_assets, 0);
    let mesh_storage = snapshot.meshes.as_ptr();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.meshes.as_ptr(), mesh_storage);
    assert_eq!(
        std::sync::Arc::strong_count(
            engine
                .app
                .world()
                .resource::<AssetLibrary>()
                .mesh(plane)
                .unwrap()
        ),
        2
    );
}

#[test]
fn old_snapshot_retains_geometry_after_asset_removal_and_new_snapshot_counts_missing() {
    let mut engine = Engine::default();
    let (mesh, material) = {
        let mut assets = engine.app.world_mut().resource_mut::<AssetLibrary>();
        (
            assets.insert_mesh(Mesh::plane()).unwrap(),
            assets
                .insert_material(Material::new([1.0; 4]).unwrap())
                .unwrap(),
        )
    };
    engine
        .app
        .world_mut()
        .spawn((Transform::default(), Mesh3d { mesh, material }));
    let mut old = RenderSnapshot::default();
    engine.extract(1.0, &mut old);
    engine
        .app
        .world_mut()
        .resource_mut::<AssetLibrary>()
        .remove_mesh(mesh);
    let mut next = RenderSnapshot::default();
    engine.extract(1.0, &mut next);
    assert_eq!(old.instances.len(), 1);
    assert_eq!(old.meshes[0].mesh.indices().len(), 6);
    assert!(next.instances.is_empty());
    assert!(next.meshes.is_empty());
    assert_eq!(next.missing_assets, 1);
    assert_eq!(next.dropped_instances, 0);
}

#[test]
fn textures_are_retained_once_and_missing_dependencies_skip_whole_entities() {
    let mut engine = Engine::default();
    let (mesh, texture, material) = {
        let mut assets = engine.app.world_mut().resource_mut::<AssetLibrary>();
        let mesh = assets.insert_mesh(Mesh::plane()).unwrap();
        let texture = assets
            .insert_texture(Texture::new(1, 1, vec![200, 100, 50, 255]).unwrap())
            .unwrap();
        let material = assets
            .insert_material(Material::textured([1.0; 4], texture).unwrap())
            .unwrap();
        (mesh, texture, material)
    };
    for _ in 0..3 {
        engine
            .app
            .world_mut()
            .spawn((Transform::default(), Mesh3d { mesh, material }));
    }
    let mut old = RenderSnapshot::default();
    engine.extract(1.0, &mut old);
    assert_eq!(old.instances.len(), 3);
    assert_eq!(old.textures.len(), 1);
    assert!(
        old.instances
            .iter()
            .all(|instance| instance.texture == Some(texture))
    );
    assert_eq!(
        std::sync::Arc::strong_count(
            engine
                .app
                .world()
                .resource::<AssetLibrary>()
                .texture(texture)
                .unwrap()
        ),
        2
    );
    let storage = old.textures.as_ptr();
    engine.extract(1.0, &mut old);
    assert_eq!(old.textures.as_ptr(), storage);
    engine
        .app
        .world_mut()
        .resource_mut::<AssetLibrary>()
        .remove_texture(texture);
    let mut next = RenderSnapshot::default();
    engine.extract(1.0, &mut next);
    assert_eq!(old.textures[0].texture.pixels(), &[200, 100, 50, 255]);
    assert_eq!(old.instances.len(), 3);
    assert!(next.instances.is_empty());
    assert!(next.textures.is_empty());
    assert!(next.meshes.is_empty());
    assert_eq!(next.missing_assets, 3);
}

#[test]
fn extraction_overflow_does_not_retain_unused_textures() {
    let mut engine = Engine::new(EngineConfig {
        max_render_instances: 0,
        ..EngineConfig::default()
    });
    let (mesh, material) = {
        let mut assets = engine.app.world_mut().resource_mut::<AssetLibrary>();
        let mesh = assets.insert_mesh(Mesh::plane()).unwrap();
        let texture = assets
            .insert_texture(Texture::new(1, 1, vec![255; 4]).unwrap())
            .unwrap();
        let material = assets
            .insert_material(Material::textured([1.0; 4], texture).unwrap())
            .unwrap();
        (mesh, material)
    };
    engine
        .app
        .world_mut()
        .spawn((Transform::default(), Mesh3d { mesh, material }));
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.dropped_instances, 1);
    assert!(snapshot.textures.is_empty());
    assert_eq!(snapshot.textures.capacity(), 0);
}

#[test]
fn invalid_mesh3d_does_not_fall_back_to_cube_and_overflow_counts_valid_only() {
    let mut engine = Engine::new(EngineConfig {
        max_render_instances: 1,
        ..EngineConfig::default()
    });
    let (mesh, material) = {
        let mut assets = engine.app.world_mut().resource_mut::<AssetLibrary>();
        (
            assets.insert_mesh(Mesh::plane()).unwrap(),
            assets
                .insert_material(Material::new([1.0; 4]).unwrap())
                .unwrap(),
        )
    };
    engine.app.world_mut().spawn((
        Transform::default(),
        Mesh3d { mesh, material },
        Cube { color: [1.0; 4] },
    ));
    engine
        .app
        .world_mut()
        .resource_mut::<AssetLibrary>()
        .remove_material(material);
    for _ in 0..2 {
        engine
            .app
            .world_mut()
            .spawn((Transform::default(), Cube { color: [1.0; 4] }));
    }
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.instances.len(), 1);
    assert_eq!(snapshot.meshes.len(), 1);
    assert_eq!(snapshot.missing_assets, 1);
    assert_eq!(snapshot.dropped_instances, 1);
}

#[cfg(feature = "dev")]
mod development {
    use super::*;

    #[test]
    fn exact_steps_bypass_catch_up_limit_and_discard_only_clock_remainder() {
        let mut engine = engine_at(8.0, 1);
        assert_eq!(engine.advance(0.0625).interpolation, 0.5);

        let report = engine.step_fixed(120).unwrap();
        assert_eq!(report.fixed_steps, 120);
        assert_eq!(report.dropped_seconds, 0.0);
        assert_eq!(report.interpolation, 0.0);
        let fixed = engine.app.world().resource::<FixedTime>();
        assert_eq!(fixed.tick, 120);
        assert_eq!(fixed.elapsed_seconds, 15.0);
        let frame = engine.app.world().resource::<FrameTime>();
        assert_eq!(frame.frame, 2);
        assert_eq!(frame.delta_seconds, 0.25);
        assert_eq!(frame.elapsed_seconds, 0.3125);

        let resumed = engine.advance(0.0625);
        assert_eq!(resumed.fixed_steps, 0);
        assert_eq!(resumed.interpolation, 0.5);
        assert_eq!(engine.advance(0.0625).fixed_steps, 1);
    }

    #[derive(Resource, Default)]
    struct ActionObservations(Vec<(bool, bool, bool)>);

    #[test]
    fn exact_steps_preserve_pending_actions_and_consume_edges_on_first_tick_only() {
        let mut engine = engine_at(8.0, 1);
        engine
            .app
            .init_resource::<ActionObservations>()
            .add_systems(
                FixedSimulation,
                |input: Res<ActionInput>, mut seen: ResMut<ActionObservations>| {
                    seen.0.push((
                        input.pressed(ActionId::JUMP),
                        input.just_pressed(ActionId::JUMP),
                        input.just_released(ActionId::JUMP),
                    ));
                },
            );
        engine.app.world_mut().resource_mut::<ActionInput>().ingest(
            0,
            ActionId::JUMP.mask(),
            ActionId::JUMP.mask(),
        );
        engine.advance(0.0625);
        engine.step_fixed(3).unwrap();
        assert_eq!(
            engine.app.world().resource::<ActionObservations>().0,
            vec![
                (false, true, true),
                (false, false, false),
                (false, false, false)
            ]
        );

        engine
            .app
            .world_mut()
            .resource_mut::<ActionInput>()
            .ingest(ActionId::JUMP.mask(), 0, 0);
        engine.step_fixed(2).unwrap();
        assert_eq!(
            &engine.app.world().resource::<ActionObservations>().0[3..],
            &[(true, true, false), (true, false, false)]
        );
    }

    #[test]
    fn invalid_steps_preserve_clock_and_inputs_without_running_schedules() {
        let mut engine = engine_at(8.0, 1);
        engine
            .app
            .init_resource::<ActionObservations>()
            .add_systems(
                FixedSimulation,
                |input: Res<ActionInput>, mut seen: ResMut<ActionObservations>| {
                    seen.0.push((
                        input.pressed(ActionId::JUMP),
                        input.just_pressed(ActionId::JUMP),
                        input.just_released(ActionId::JUMP),
                    ));
                },
            );
        engine.app.init_resource::<PresentedFrames>().add_systems(
            Presentation,
            |frame: Res<FrameTime>, mut presented: ResMut<PresentedFrames>| {
                presented.0.push((frame.frame, frame.delta_seconds));
            },
        );
        engine.advance(0.0625);
        engine
            .app
            .world_mut()
            .resource_mut::<ActionInput>()
            .ingest(ActionId::JUMP.mask(), 0, 0);
        for ticks in [0, 121, u32::MAX] {
            assert_eq!(
                engine.step_fixed(ticks).unwrap_err(),
                DevStepError { ticks }
            );
        }
        let frame = engine.app.world().resource::<FrameTime>();
        assert_eq!(frame.frame, 1);
        assert_eq!(frame.delta_seconds, 0.0625);
        assert_eq!(frame.elapsed_seconds, 0.0625);
        assert_eq!(engine.app.world().resource::<FixedTime>().tick, 0);
        assert!(
            engine
                .app
                .world()
                .resource::<ActionInput>()
                .pressed(ActionId::JUMP)
        );
        assert_eq!(
            engine.app.world().resource::<PresentedFrames>().0,
            vec![(1, 0.0625)]
        );
        assert_eq!(engine.advance(0.0625).fixed_steps, 1);
        assert_eq!(
            engine.app.world().resource::<ActionObservations>().0,
            vec![(true, true, false)]
        );
    }

    #[test]
    fn invalid_first_step_does_not_finalize_plugins() {
        #[derive(Resource)]
        struct Finished;
        struct FinishPlugin;
        impl Plugin for FinishPlugin {
            fn build(&self, _app: &mut App) {}
            fn finish(&self, app: &mut App) {
                app.insert_resource(Finished);
            }
        }
        let mut engine = Engine::default();
        engine.app.add_plugins(FinishPlugin);
        assert!(engine.step_fixed(0).is_err());
        assert!(!engine.app.world().contains_resource::<Finished>());
        engine.step_fixed(1).unwrap();
        assert!(engine.app.world().contains_resource::<Finished>());
    }

    #[test]
    fn exact_steps_apply_commands_each_tick_and_present_once_per_request() {
        let mut engine = engine_at(8.0, 1);
        engine
            .app
            .init_resource::<Observations>()
            .init_resource::<PresentedFrames>()
            .add_systems(FixedSimulation, spawn_during_tick)
            .add_systems(Presentation, observe_presentation)
            .add_systems(
                Presentation,
                |frame: Res<FrameTime>, mut presented: ResMut<PresentedFrames>| {
                    presented.0.push((frame.frame, frame.delta_seconds));
                },
            );
        engine.step_fixed(2).unwrap();
        let seen = engine.app.world().resource::<Observations>();
        assert_eq!(seen.fixed_counts, vec![0, 1]);
        assert_eq!(seen.presentation_count, 2);
        engine.step_fixed(1).unwrap();
        assert_eq!(
            engine.app.world().resource::<PresentedFrames>().0,
            vec![(1, 0.25), (2, 0.125)]
        );
        assert_eq!(
            engine
                .app
                .world()
                .resource::<Observations>()
                .presentation_count,
            3
        );
    }
}

use crate::{NodeInfo, NodePose};

#[derive(Component)]
struct Drift(Vec3);

fn drift(mut objects: Query<(&Drift, &mut Transform)>) {
    for (drift, mut transform) in &mut objects {
        transform.translation += drift.0;
        transform.rotation = Quat::from_rotation_y(0.1) * transform.rotation;
    }
}

/// A static parent with drifting interpolated children and a drifting root.
fn drifting_world() -> (Engine, Vec<Entity>) {
    let mut engine = Engine::new(EngineConfig::default());
    engine.app.add_systems(FixedSimulation, drift);
    let world = engine.app.world_mut();
    let parent = world.spawn(Transform::default()).id();
    let mut entities = vec![parent];
    for index in 0..6 {
        let child = world
            .spawn((
                Transform {
                    translation: Vec3::X * index as f32,
                    ..Default::default()
                },
                Drift(Vec3::new(0.1, 0.0, 0.05 * index as f32)),
                InterpolateTransform,
                Cube { color: [1.0; 4] },
            ))
            .id();
        entities.push(child);
    }
    let root = world
        .spawn((
            Transform::default(),
            Drift(Vec3::Y * 0.2),
            InterpolateTransform,
            Cube { color: [1.0; 4] },
        ))
        .id();
    entities.push(root);
    let links: Vec<_> = entities[1..7]
        .iter()
        .map(|&child| (child, Some(parent)))
        .collect();
    set_parents(engine.app.world_mut(), &links).unwrap();
    (engine, entities)
}

fn authoritative(engine: &Engine, entities: &[Entity]) -> (u64, Vec<[u32; 16]>) {
    let world = engine.app.world();
    let globals = world.resource::<GlobalTransforms>();
    let matrices = entities
        .iter()
        .map(|&entity| {
            let matrix = globals.get(entity).unwrap().matrix().unwrap();
            matrix.to_cols_array().map(f32::to_bits)
        })
        .collect();
    (world.resource::<FixedTime>().tick, matrices)
}

type GatheredInputs = (u32, u64, bool, Vec<NodePose>, Vec<NodePose>, Vec<NodeInfo>);

fn gathered(engine: &Engine) -> GatheredInputs {
    let view = engine
        .app
        .world()
        .resource::<PresentedTransforms>()
        .gpu_presentation(engine.app.world().resource::<GlobalTransforms>());
    (
        view.alpha.to_bits(),
        view.sequence,
        view.shift,
        view.current.to_vec(),
        view.previous.to_vec(),
        view.info.to_vec(),
    )
}

#[test]
fn running_ticks_ahead_splits_them_across_frames_without_changing_results() {
    let (mut reference, entities) = drifting_world();
    let (mut pipelined, _) = drifting_world();
    let delta = 1.0 / 120.0;
    let (mut begun, mut completed, mut finished_when_due) = (0, 0, 0);
    for _ in 0..48 {
        let expected = reference.advance(delta);
        reference
            .extract_gpu_frame(1.0, expected.interpolation)
            .unwrap();
        let in_flight = pipelined.tick_in_flight();
        let report = pipelined.advance(delta);
        finished_when_due += usize::from(in_flight && !pipelined.tick_in_flight());
        assert_eq!(report.interpolation, expected.interpolation);
        pipelined
            .extract_gpu_frame(1.0, report.interpolation)
            .unwrap();
        // Every presented frame shows the same state and GPU inputs. An
        // in-flight tick has advanced fixed time; its poses publish when it
        // completes.
        let (tick, matrices) = authoritative(&pipelined, &entities);
        let (expected_tick, expected_matrices) = authoritative(&reference, &entities);
        assert_eq!(matrices, expected_matrices);
        assert_eq!(tick, expected_tick + u64::from(pipelined.tick_in_flight()));
        assert_eq!(gathered(&pipelined), gathered(&reference));
        // Each frame runs at most one stage after submitting.
        let in_flight = pipelined.tick_in_flight();
        if pipelined.run_ahead(delta) {
            if in_flight {
                assert!(!pipelined.tick_in_flight());
                completed += 1;
            } else {
                assert!(pipelined.tick_in_flight());
                begun += 1;
            }
        }
    }
    // A tick starts two frames before it is due and completes the frame
    // before; the due frame only presents it. Only the first tick, started
    // one frame before it was due, finishes when due.
    assert!(
        begun >= 22 && completed >= 22,
        "{begun} begun, {completed} completed"
    );
    assert!(
        finished_when_due <= 1,
        "{finished_when_due} finished when due"
    );
    // Pipelined gathers still follow single quiet ticks.
    assert!(gathered(&pipelined).2);
}

#[test]
fn ticks_completed_ahead_stay_hidden_from_frames_before_they_are_due() {
    let (mut reference, entities) = drifting_world();
    let (mut pipelined, _) = drifting_world();
    let delta = 1.0 / 120.0;
    let mut early_frames = 0;
    for frame in 0..40 {
        // Every fourth frame arrives early, after a tick completed ahead.
        let step = if frame % 4 == 3 { delta * 0.1 } else { delta };
        let expected = reference.advance(step);
        reference
            .extract_gpu_frame(1.0, expected.interpolation)
            .unwrap();
        let report = pipelined.advance(step);
        assert_eq!(report.interpolation, expected.interpolation);
        pipelined
            .extract_gpu_frame(1.0, report.interpolation)
            .unwrap();
        // A tick completed before it is due is not presented early: its
        // inputs, gathered when it completed, are held, so a consumer keeps
        // presenting the inputs it applied at this frame's fraction. Due
        // frames present exactly the reference's inputs.
        let hidden = !pipelined.tick_in_flight()
            && authoritative(&pipelined, &entities).0 != authoritative(&reference, &entities).0;
        let (actual, expected) = (gathered(&pipelined), gathered(&reference));
        let held = pipelined
            .app
            .world()
            .resource::<PresentedTransforms>()
            .gpu_presentation(pipelined.app.world().resource::<GlobalTransforms>())
            .hold;
        assert_eq!(held, hidden);
        if hidden {
            assert_eq!(actual.0, expected.0);
            // One gather ahead of the reference, which gathers when it is due.
            assert_eq!(actual.1, expected.1 + 1);
        } else {
            assert_eq!(actual, expected);
        }
        early_frames += usize::from(hidden && step < delta);
        assert!(!pipelined.prepare_presented_transforms() || !hidden);
        pipelined.run_ahead(delta);
    }
    assert!(early_frames >= 5, "{early_frames} early frames");
}

#[test]
fn explicitly_completed_ticks_stay_hidden_from_gpu_frames_until_due_or_revealed() {
    let (mut engine, _) = drifting_world();
    let delta = 1.0 / 120.0;
    // Advance until the next tick's first stage runs after a frame.
    for _ in 0..64 {
        let report = engine.advance(delta);
        engine.extract_gpu_frame(1.0, report.interpolation).unwrap();
        if engine.run_ahead(delta) && engine.tick_in_flight() {
            break;
        }
    }
    assert!(engine.tick_in_flight());
    let (_, sequence, ..) = gathered(&engine);
    // A development command completes it; ordinary frames still show the
    // previous interval until it is due.
    engine.complete_pending_tick();
    let report = engine.advance(0.0);
    engine.extract_gpu_frame(1.0, report.interpolation).unwrap();
    assert_eq!(gathered(&engine).1, sequence);
    assert!(!engine.prepare_presented_transforms());
    // An explicit preview presents the completed tick.
    engine.reveal_ticks_ahead();
    engine.extract_gpu_frame(1.0, 0.5).unwrap();
    assert_eq!(gathered(&engine).1, sequence + 1);
}

#[test]
fn frames_during_an_in_flight_tick_present_the_last_gathered_inputs() {
    let (mut engine, entities) = drifting_world();
    let delta = 1.0 / 120.0;
    for _ in 0..4 {
        let report = engine.advance(delta);
        engine.extract_gpu_frame(1.0, report.interpolation).unwrap();
    }
    let before = authoritative(&engine, &entities);
    let (_, sequence, ..) = gathered(&engine);
    // A tick predicted due by a longer frame than actually follows.
    assert!(engine.run_ahead(2.0 * delta));
    assert!(engine.tick_in_flight());
    engine.extract_gpu_frame(1.0, 0.75).unwrap();
    let (alpha, still, ..) = gathered(&engine);
    assert_eq!((alpha, still), (0.75_f32.to_bits(), sequence));
    assert!(engine.presentation().gpu_pending());
    assert!(!engine.prepare_presented_transforms());
    // Authoritative globals still describe the last completed tick.
    let globals_unchanged = {
        let (_, matrices) = authoritative(&engine, &entities);
        matrices == before.1
    };
    assert!(globals_unchanged);
    // Exact extraction completes the tick first; the clock later counts it.
    let tick = engine.app.world().resource::<FixedTime>().tick;
    engine
        .extract_frame(1.0, PresentationMode::Current)
        .unwrap();
    assert!(!engine.tick_in_flight());
    assert_ne!(authoritative(&engine, &entities).1, before.1);
    for _ in 0..2 {
        engine.advance(delta);
    }
    assert_eq!(
        engine.app.world().resource::<FixedTime>().tick,
        tick,
        "the completed tick is not repeated when it becomes due"
    );
    engine.advance(2.0 * delta);
    assert_eq!(engine.app.world().resource::<FixedTime>().tick, tick + 1);
}

#[test]
fn frames_carry_the_atmosphere_and_the_eye() {
    let mut engine = Engine::default();
    let world = engine.app.world_mut();
    world.resource_mut::<Camera>().position = Vec3::new(1.0, 2.0, 3.0);
    *world.resource_mut::<Atmosphere>() = Atmosphere::night();
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.5, &mut snapshot);
    assert_eq!(snapshot.camera_position, Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(snapshot.atmosphere, Atmosphere::night());
    let frame = engine
        .extract_frame(1.5, PresentationMode::Current)
        .unwrap();
    assert_eq!(frame.snapshot().atmosphere, Atmosphere::night());
    assert_eq!(frame.snapshot().camera_position, Vec3::new(1.0, 2.0, 3.0));
    // Unusable lighting and a non-finite eye fall back to defaults.
    let world = engine.app.world_mut();
    world.resource_mut::<Atmosphere>().fog_max = 2.0;
    world.resource_mut::<Camera>().position = Vec3::NAN;
    engine.extract(1.5, &mut snapshot);
    assert_eq!(snapshot.atmosphere, Atmosphere::default());
    assert_eq!(snapshot.camera_position, Camera::default().position);
}

#[test]
fn frames_carry_sanitized_effects_and_presentation_time() {
    let mut engine = Engine::default();
    let shadows = Shadows {
        bounds: Some((Vec3::splat(-10.0), Vec3::splat(10.0))),
        resolution: 2048,
        ..Shadows::default()
    };
    let world = engine.app.world_mut();
    *world.resource_mut::<Shadows>() = shadows;
    world.resource_mut::<AmbientOcclusion>().intensity = 3.0;
    world.resource_mut::<Bloom>().intensity = 0.0;
    engine.advance(0.5);
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.5, &mut snapshot);
    assert_eq!(snapshot.shadows, shadows);
    assert_eq!(snapshot.ambient_occlusion.intensity, 1.0);
    assert!(!snapshot.bloom.enabled());
    assert_eq!(
        snapshot.elapsed_seconds, 0.25,
        "presentation time is capped per frame"
    );
    let frame = engine
        .extract_frame(1.5, PresentationMode::Current)
        .unwrap();
    assert_eq!(frame.snapshot().shadows, shadows);
    assert_eq!(frame.snapshot().elapsed_seconds, 0.25);
    // Removing an effect's resource removes the effect.
    let world = engine.app.world_mut();
    world.remove_resource::<Shadows>();
    world.remove_resource::<AmbientOcclusion>();
    world.remove_resource::<Bloom>();
    world.remove_resource::<Atmosphere>();
    engine.extract(1.5, &mut snapshot);
    assert!(!snapshot.shadows.enabled);
    assert!(!snapshot.ambient_occlusion.enabled());
    assert!(!snapshot.bloom.enabled());
    assert_eq!(snapshot.atmosphere, Atmosphere::default());
}

#[test]
fn surfaces_reach_instances_and_their_changes_advance_color_stamps() {
    let mut engine = Engine::default();
    let world = engine.app.world_mut();
    let mut assets = world.resource_mut::<AssetLibrary>();
    let mesh = assets.insert_mesh(Mesh::cube()).unwrap();
    let glass = Surface {
        roughness: 0.1,
        ..Surface::MATTE
    };
    let material = assets
        .insert_material(
            Material::new([1.0; 4])
                .unwrap()
                .with_surface(glass)
                .unwrap(),
        )
        .unwrap();
    let lamp = assets
        .insert_material(
            Material::new([1.0; 4])
                .unwrap()
                .with_surface(Surface {
                    emission: 4.0,
                    ..Surface::MATTE
                })
                .unwrap(),
        )
        .unwrap();
    let entity = world
        .spawn((Transform::default(), Mesh3d { mesh, material }))
        .id();
    world.spawn((Transform::default(), Cube { color: [1.0; 4] }));
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.5, &mut snapshot);
    let surfaces: Vec<Surface> = snapshot.instances.iter().map(|i| i.surface).collect();
    assert!(surfaces.contains(&glass) && surfaces.contains(&Surface::MATTE));
    let (revision, stamp) = {
        let frame = engine
            .extract_frame(1.5, PresentationMode::Current)
            .unwrap();
        let slot = frame
            .visible_indices()
            .iter()
            .copied()
            .find(|&slot| frame.snapshot().instances[slot].surface == glass)
            .unwrap();
        (frame.color_revision(), frame.color_changes()[slot])
    };
    // Same color, new surface: the color channel re-uploads.
    engine.app.world_mut().entity_mut(entity).insert(Mesh3d {
        mesh,
        material: lamp,
    });
    let frame = engine
        .extract_frame(1.5, PresentationMode::Current)
        .unwrap();
    assert!(frame.color_revision() > revision);
    let slot = frame
        .visible_indices()
        .iter()
        .copied()
        .find(|&slot| frame.snapshot().instances[slot].surface.emission == 4.0)
        .unwrap();
    assert!(frame.color_changes()[slot] > stamp);
}

#[test]
fn cpu_frames_list_culled_slots_as_shadow_casters() {
    let mut engine = Engine::default();
    let world = engine.app.world_mut();
    for x in [0.0, 10_000.0] {
        world.spawn((
            Transform {
                translation: Vec3::new(x, 0.0, 0.0),
                ..Transform::default()
            },
            Cube { color: [1.0; 4] },
        ));
    }
    let (visible, casters, revision) = {
        let frame = engine
            .extract_frame(1.5, PresentationMode::Current)
            .unwrap();
        (
            frame.visible_indices().len(),
            frame.casters().to_vec(),
            frame.caster_revision(),
        )
    };
    assert_eq!(visible, 1, "the far cube is outside the frustum");
    assert_eq!(casters.len(), 2, "culled slots still cast shadows");
    assert!(casters.is_sorted());
    // Camera motion alone changes visibility, not casters.
    engine.app.world_mut().resource_mut::<Camera>().position.x += 1.0;
    let frame = engine
        .extract_frame(1.5, PresentationMode::Current)
        .unwrap();
    assert_eq!(frame.casters(), casters);
    assert_eq!(frame.caster_revision(), revision);
}
