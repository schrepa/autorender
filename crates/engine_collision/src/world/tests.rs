use super::*;
use crate::{CollisionPlugin, CollisionSystems};
use engine_core::prelude::*;

fn engine() -> Engine {
    let mut engine = Engine::new(EngineConfig {
        fixed_hz: 8.0,
        frustum_culling: false,
        ..Default::default()
    });
    engine.app.add_plugins(CollisionPlugin);
    engine
}

fn spawn(engine: &mut Engine, collider: Collider, transform: Transform) -> Entity {
    engine.app.world_mut().spawn((collider, transform)).id()
}

fn sync_engine(engine: &mut Engine) -> CollisionReport {
    refresh_collision_world(engine.app.world_mut()).unwrap()
}
fn unit_box() -> Collider {
    Collider::cuboid(Vec3::splat(0.5)).unwrap()
}
fn ray(origin: Vec3, direction: Vec3, distance: f32) -> CollisionRay {
    CollisionRay {
        origin,
        direction,
        max_distance: distance,
    }
}

#[test]
fn scheduled_collision_declares_pose_resource_reads_without_exclusive_or_deferred_access() {
    use bevy_ecs::system::{IntoSystem, System};

    let mut engine = engine();
    let world = engine.app.world_mut();
    let mut sync_system = IntoSystem::into_system(sync);
    let sync_access = sync_system.initialize(world);
    let sync_access = sync_access.combined_access();
    assert!(!sync_system.is_exclusive());
    assert!(!sync_system.has_deferred());
    let components = [
        world.component_id::<Collider>().unwrap(),
        world.component_id::<Transform>().unwrap(),
        world.component_id::<GlobalTransforms>().unwrap(),
    ];
    for component in components {
        assert!(sync_access.has_read(component));
        assert!(!sync_access.has_write(component));
    }
    let collisions = world.component_id::<CollisionWorld>().unwrap();
    let events = world.component_id::<SensorEvents>().unwrap();
    assert!(sync_access.has_write(collisions));
    assert!(!sync_access.has_write(events));

    let mut detect_system = IntoSystem::into_system(detect);
    let detect_access = detect_system.initialize(world);
    let detect_access = detect_access.combined_access();
    assert!(!detect_system.is_exclusive());
    assert!(!detect_system.has_deferred());
    for component in components {
        assert!(!detect_access.has_read(component));
        assert!(!detect_access.has_write(component));
    }
    assert!(detect_access.has_write(collisions));
    assert!(detect_access.has_write(events));
}

#[test]
fn direct_refresh_metadata_survives_cache_replacement_and_new_archetypes_without_events() {
    #[derive(Component)]
    struct NewArchetype;

    let mut engine = engine();
    let sensor = spawn(
        &mut engine,
        unit_box().with_sensor(true),
        Transform::default(),
    );
    let solid = spawn(&mut engine, unit_box(), Transform::default());
    engine.advance(0.125);
    sync_engine(&mut engine);
    let world = engine.app.world_mut();
    assert!(world.contains_resource::<DirectSyncState>());
    let sequence = world.resource::<SensorEvents>().sequence();
    let frame = world.resource::<FrameTime>().frame;
    let tick = world.resource::<FixedTime>().tick;
    world.remove_resource::<CollisionWorld>();
    assert_eq!(
        refresh_collision_world(world).unwrap_err(),
        CollisionError::MissingPlugin
    );
    world.init_resource::<CollisionWorld>();
    world.despawn(solid);
    let replacement = world
        .spawn((
            unit_box(),
            Transform {
                translation: Vec3::X * 4.0,
                ..Default::default()
            },
            NewArchetype,
        ))
        .id();
    world.resource_mut::<CollisionConfig>().max_colliders = 1;
    let overflow = refresh_collision_world(world).unwrap();
    assert_eq!(overflow.colliders, 2);
    assert_eq!(overflow.overflow_colliders, 1);
    assert!(!overflow.complete);
    world.resource_mut::<CollisionConfig>().max_colliders = 2;
    let report = refresh_collision_world(world).unwrap();
    assert!(report.complete);
    assert_eq!(report.active_colliders, 2);
    assert_eq!((report.frame, report.tick), (frame, tick));
    let hit = world
        .resource::<CollisionWorld>()
        .raycast(ray(Vec3::X * 3.0, Vec3::X, 5.0), QueryFilter::default())
        .unwrap()
        .unwrap();
    assert_eq!(hit.entity, replacement);
    assert_eq!(hit.distance, 0.5);
    assert_eq!(world.resource::<FrameTime>().frame, frame);
    assert_eq!(world.resource::<FixedTime>().tick, tick);
    let events = world.resource::<SensorEvents>();
    assert_eq!(events.sequence(), sequence);
    assert_eq!(events.committed_tick(), Some(tick));
    assert_eq!(
        events.events(),
        &[SensorEvent {
            pair: SensorPair::new(sensor, solid),
            kind: SensorEventKind::Enter,
        }]
    );
}

#[test]
fn closest_and_all_rays_report_world_distance_sorted_hits_and_start_inside() {
    let mut engine = engine();
    let far = spawn(
        &mut engine,
        unit_box(),
        Transform {
            translation: Vec3::X * 5.0,
            ..Default::default()
        },
    );
    let near = spawn(
        &mut engine,
        unit_box(),
        Transform {
            translation: Vec3::X * 2.0,
            ..Default::default()
        },
    );
    sync_engine(&mut engine);
    let collision = engine.app.world().resource::<CollisionWorld>();
    let ray = ray(Vec3::ZERO, Vec3::X * 10.0, 10.0);
    let hit = collision
        .raycast(ray, QueryFilter::default())
        .unwrap()
        .unwrap();
    assert_eq!(hit.entity, near);
    assert_eq!(hit.distance, 1.5);
    assert_eq!(hit.normal, -Vec3::X);
    let mut hits = Vec::new();
    collision
        .raycast_all(ray, QueryFilter::default(), &mut hits, 2)
        .unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.entity).collect::<Vec<_>>(),
        [near, far]
    );
    let inside = collision
        .raycast(
            CollisionRay {
                origin: Vec3::X * 2.0,
                ..ray
            },
            QueryFilter::default(),
        )
        .unwrap()
        .unwrap();
    assert!(inside.started_inside);
    assert_eq!(inside.distance, 0.0);
    assert_eq!(inside.normal, Vec3::ZERO);
    assert_eq!(
        collision.raycast_all(ray, QueryFilter::default(), &mut hits, 1),
        Err(CollisionError::ResultBudget)
    );
    assert!(hits.is_empty());
}

#[test]
fn floor_touch_blocks_downward_and_allows_tangent_or_separating_motion() {
    let mut engine = engine();
    let floor = spawn(
        &mut engine,
        Collider::cuboid(Vec3::new(10.0, 0.5, 10.0)).unwrap(),
        Transform {
            translation: -Vec3::Y * 0.5,
            ..Default::default()
        },
    );
    sync_engine(&mut engine);
    let mut collision = engine.app.world_mut().resource_mut::<CollisionWorld>();
    let shape = unit_box().shape();
    let matrix = Mat4::from_translation(Vec3::Y * 0.5);
    let options = ShapeCastOptions {
        stop_at_penetration: false,
        target_distance: 0.0,
    };
    let hit = collision
        .cast_shape(shape, matrix, -Vec3::Y, QueryFilter::default(), options)
        .unwrap()
        .unwrap();
    assert_eq!(hit.entity, floor);
    assert!(hit.started_overlapping);
    assert_eq!(hit.fraction, 0.0);
    assert!(hit.normal.abs_diff_eq(Vec3::Y, 1.0e-4), "{hit:?}");
    assert!(
        collision
            .cast_shape(shape, matrix, Vec3::X, QueryFilter::default(), options)
            .unwrap()
            .is_none()
    );
    assert!(
        collision
            .cast_shape(shape, matrix, Vec3::Y, QueryFilter::default(), options)
            .unwrap()
            .is_none()
    );
    let blocking = collision
        .cast_shape(
            shape,
            matrix,
            Vec3::X,
            QueryFilter::default(),
            ShapeCastOptions::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(blocking.fraction, 0.0);
}

#[test]
fn repeated_casts_keep_scaled_player_on_a_large_floor() {
    let mut engine = engine();
    spawn(
        &mut engine,
        Collider::cuboid(Vec3::new(20.0, 0.5, 20.0)).unwrap(),
        Transform {
            translation: -Vec3::Y * 0.5,
            ..Default::default()
        },
    );
    sync_engine(&mut engine);
    let mut collision = engine.app.world_mut().resource_mut::<CollisionWorld>();
    let mut player = Transform {
        translation: Vec3::Y * 0.6,
        scale: Vec3::splat(1.2),
        ..Default::default()
    };
    let options = ShapeCastOptions {
        stop_at_penetration: false,
        target_distance: 0.0,
    };
    for tick in 0..36 {
        for displacement in [-Vec3::Y * (20.0 / (60.0 * 60.0)), -Vec3::Z * (6.5 / 60.0)] {
            let hit = collision
                .cast_shape(
                    unit_box().shape(),
                    player.matrix(),
                    displacement,
                    QueryFilter::default(),
                    options,
                )
                .unwrap();
            player.translation += displacement * hit.map_or(1.0, |hit| hit.fraction);
        }
        assert!(
            (player.translation.y - 0.6).abs() < 1.0e-5,
            "tick{tick} player={player:?}"
        );
    }
    assert!((player.translation.z + 3.9).abs() < 1.0e-4);
}

#[test]
fn swept_cuboid_and_sphere_stop_at_wall_without_tunneling() {
    let mut engine = engine();
    let wall = spawn(
        &mut engine,
        unit_box(),
        Transform {
            translation: Vec3::X * 5.0,
            ..Default::default()
        },
    );
    sync_engine(&mut engine);
    let mut collision = engine.app.world_mut().resource_mut::<CollisionWorld>();
    for shape in [unit_box().shape(), Collider::sphere(0.5).unwrap().shape()] {
        let hit = collision
            .cast_shape(
                shape,
                Mat4::IDENTITY,
                Vec3::X * 10.0,
                QueryFilter::default(),
                ShapeCastOptions::default(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(hit.entity, wall);
        assert!((hit.fraction - 0.4).abs() < 1.0e-4, "{hit:?}");
        assert!(hit.normal.abs_diff_eq(-Vec3::X, 1.0e-3), "{hit:?}");
        assert!(!hit.started_overlapping);
    }
}

#[test]
fn affine_sphere_is_an_ellipsoid_and_sheared_box_uses_full_hierarchy() {
    let mut engine = engine();
    let parent = engine
        .app
        .world_mut()
        .spawn(Transform {
            scale: Vec3::new(3.0, 1.0, 2.0),
            ..Default::default()
        })
        .id();
    let local = Transform {
        rotation: Quat::from_rotation_z(0.7),
        translation: Vec3::Y * 3.0,
        ..Default::default()
    };
    let child = spawn(&mut engine, unit_box(), local);
    set_parent(engine.app.world_mut(), child, Some(parent)).unwrap();
    let sphere = spawn(
        &mut engine,
        Collider::sphere(1.0).unwrap(),
        Transform {
            scale: Vec3::new(2.0, 1.0, 0.5),
            ..Default::default()
        },
    );
    sync_engine(&mut engine);
    let matrix = engine
        .app
        .world()
        .resource::<GlobalTransforms>()
        .get(child)
        .unwrap()
        .matrix()
        .unwrap();
    let world_point = matrix.transform_point3(Vec3::new(0.49, 0.49, 0.0));
    let mut found = Vec::new();
    let collision = engine.app.world().resource::<CollisionWorld>();
    collision
        .overlap(
            ColliderShape::Sphere { radius: 0.01 },
            Mat4::from_translation(world_point),
            QueryFilter::default(),
            &mut found,
            10,
        )
        .unwrap();
    assert_eq!(found, [child]);
    let hit = collision
        .raycast(
            ray(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 10.0),
            QueryFilter::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(hit.entity, sphere);
    assert!((hit.distance - 3.0).abs() < 1.0e-5);
    assert!(
        collision
            .raycast(
                ray(Vec3::new(-5.0, 1.1, 0.0), Vec3::X, 10.0),
                QueryFilter::default()
            )
            .unwrap()
            .is_none()
    );
}

#[test]
fn plugin_absence_is_explicit_and_collider_is_independent_data() {
    let mut engine = Engine::default();
    engine
        .app
        .world_mut()
        .spawn((Transform::default(), unit_box()));
    assert_eq!(
        refresh_collision_world(engine.app.world_mut()).unwrap_err(),
        CollisionError::MissingPlugin
    );
    engine.advance(0.125);
    assert!(!engine.app.world().contains_resource::<CollisionWorld>());
}

#[test]
fn layer_masks_are_symmetric_and_sensor_filter_and_exclusion_apply() {
    let mut engine = engine();
    let entity = spawn(
        &mut engine,
        unit_box()
            .with_layers(CollisionLayers::new(2, 4))
            .with_sensor(true),
        Transform::default(),
    );
    sync_engine(&mut engine);
    let collision = engine.app.world().resource::<CollisionWorld>();
    let ray = ray(-Vec3::X * 3.0, Vec3::X, 6.0);
    assert!(
        collision
            .raycast(ray, QueryFilter::default())
            .unwrap()
            .is_none()
    );
    let filter = QueryFilter {
        layers: CollisionLayers::new(4, 2),
        ..Default::default()
    };
    assert_eq!(
        collision.raycast(ray, filter).unwrap().unwrap().entity,
        entity
    );
    for filter in [
        QueryFilter {
            exclude: Some(entity),
            ..filter
        },
        QueryFilter {
            sensors: SensorFilter::Exclude,
            ..filter
        },
        QueryFilter {
            layers: CollisionLayers::new(8, 2),
            ..filter
        },
    ] {
        assert!(collision.raycast(ray, filter).unwrap().is_none());
    }
}

#[test]
fn sensor_transitions_commit_only_at_fixed_boundaries_and_refresh_is_read_only() {
    let mut engine = engine();
    let sensor = spawn(
        &mut engine,
        unit_box().with_sensor(true),
        Transform::default(),
    );
    let solid = spawn(&mut engine, unit_box(), Transform::default());
    sync_engine(&mut engine);
    assert!(
        engine
            .app
            .world()
            .resource::<SensorEvents>()
            .tick()
            .is_none()
    );
    engine.advance(0.125);
    let events = engine.app.world().resource::<SensorEvents>();
    assert_eq!(
        events.events(),
        &[SensorEvent {
            pair: SensorPair::new(sensor, solid),
            kind: SensorEventKind::Enter
        }]
    );
    assert_eq!(events.committed_tick(), Some(1));
    let sequence = events.sequence();
    edit_transform(engine.app.world_mut(), solid, |transform| {
        transform.translation.x = 5.0;
    })
    .unwrap();
    let report = sync_engine(&mut engine);
    assert_eq!(report.tick, 1);
    engine.advance(0.0);
    engine.extract(1.0, &mut RenderSnapshot::default());
    let events = engine.app.world().resource::<SensorEvents>();
    assert_eq!(events.sequence(), sequence);
    assert_eq!(events.events()[0].kind, SensorEventKind::Enter);
    engine.advance(0.125);
    let events = engine.app.world().resource::<SensorEvents>();
    assert_eq!(
        events.events(),
        &[SensorEvent {
            pair: SensorPair::new(sensor, solid),
            kind: SensorEventKind::Exit
        }]
    );
    assert!(events.pairs().is_empty());
    engine.advance(0.125);
    assert!(
        engine
            .app
            .world()
            .resource::<SensorEvents>()
            .events()
            .is_empty()
    );
}

#[test]
fn post_consumers_see_each_fixed_tick_and_deferred_despawn_has_one_exit() {
    #[derive(Resource, Default)]
    struct Seen(Vec<(u64, SensorEventKind)>);
    let mut engine = engine();
    engine.app.init_resource::<Seen>();
    let sensor = spawn(
        &mut engine,
        unit_box().with_sensor(true),
        Transform::default(),
    );
    let solid = spawn(&mut engine, unit_box(), Transform::default());
    engine.app.add_systems(
        FixedPostSimulation,
        (move |events: Res<SensorEvents>, mut seen: ResMut<Seen>, mut commands: Commands| {
            for event in events.events() {
                seen.0.push((events.tick().unwrap(), event.kind));
                if event.kind == SensorEventKind::Enter {
                    commands.entity(solid).despawn();
                }
            }
        })
        .after(CollisionSystems::Detect),
    );
    engine.advance(0.375);
    assert_eq!(
        engine.app.world().resource::<Seen>().0,
        [(1, SensorEventKind::Enter), (2, SensorEventKind::Exit)]
    );
    assert!(engine.app.world().entities().contains(sensor));
    assert!(!engine.app.world().entities().contains(solid));
    assert!(
        engine
            .app
            .world()
            .resource::<SensorEvents>()
            .pairs()
            .is_empty()
    );
}

#[test]
fn sensor_pairs_are_unique_and_do_not_scan_unrelated_solid_pairs() {
    let mut engine = engine();
    for _ in 0..100 {
        spawn(&mut engine, unit_box(), Transform::default());
    }
    engine.advance(0.125);
    assert_eq!(
        engine
            .app
            .world()
            .resource::<SensorEvents>()
            .candidate_pairs(),
        0
    );
    for _ in 0..2 {
        spawn(
            &mut engine,
            unit_box().with_sensor(true),
            Transform::default(),
        );
    }
    engine.advance(0.125);
    let events = engine.app.world().resource::<SensorEvents>();
    assert_eq!(events.pairs().len(), 201);
    assert_eq!(events.events().len(), 201);
    assert_eq!(events.candidate_pairs(), 201);
}

#[test]
fn pair_or_candidate_exhaustion_preserves_complete_history_without_false_exits() {
    for pair_budget in [true, false] {
        let mut engine = engine();
        let sensor = spawn(
            &mut engine,
            unit_box().with_sensor(true),
            Transform::default(),
        );
        let old = spawn(&mut engine, unit_box(), Transform::default());
        engine.advance(0.125);
        let new = spawn(&mut engine, unit_box(), Transform::default());
        {
            let mut config = engine.app.world_mut().resource_mut::<CollisionConfig>();
            if pair_budget {
                config.max_sensor_pairs = 1;
            } else {
                config.max_candidate_pairs = 1;
            }
        }
        engine.advance(0.125);
        let events = engine.app.world().resource::<SensorEvents>();
        assert!(!events.complete());
        assert_eq!(
            events.error(),
            Some(if pair_budget {
                CollisionError::PairBudget
            } else {
                CollisionError::CandidateBudget
            })
        );
        assert_eq!(events.tick(), Some(2));
        assert_eq!(events.committed_tick(), Some(1));
        assert_eq!(events.pairs(), &[SensorPair::new(sensor, old)]);
        assert!(events.events().is_empty());
        engine.app.world_mut().despawn(old);
        engine.advance(0.125);
        let events = engine.app.world().resource::<SensorEvents>();
        assert!(events.complete());
        assert_eq!(events.pairs(), &[SensorPair::new(sensor, new)]);
        assert_eq!(events.events().len(), 2);
        assert!(
            events
                .events()
                .iter()
                .any(|event| event.kind == SensorEventKind::Exit)
        );
        assert!(
            events
                .events()
                .iter()
                .any(|event| event.kind == SensorEventKind::Enter)
        );
    }
}

#[test]
fn query_candidate_budget_errors_instead_of_claiming_partial_nearest_results() {
    let mut engine = engine();
    for _ in 0..3 {
        spawn(&mut engine, unit_box(), Transform::default());
    }
    engine
        .app
        .world_mut()
        .resource_mut::<CollisionConfig>()
        .max_query_candidates = 1;
    sync_engine(&mut engine);
    let mut collision = engine.app.world_mut().resource_mut::<CollisionWorld>();
    let ray = ray(-Vec3::X * 2.0, Vec3::X, 4.0);
    assert_eq!(
        collision.raycast(ray, QueryFilter::default()).unwrap_err(),
        CollisionError::CandidateBudget
    );
    let mut rays = Vec::new();
    assert_eq!(
        collision.raycast_all(ray, QueryFilter::default(), &mut rays, 5),
        Err(CollisionError::CandidateBudget)
    );
    assert!(rays.is_empty());
    let mut overlaps = Vec::new();
    assert_eq!(
        collision.overlap(
            unit_box().shape(),
            Mat4::IDENTITY,
            QueryFilter::default(),
            &mut overlaps,
            5
        ),
        Err(CollisionError::CandidateBudget)
    );
    assert!(overlaps.is_empty());
    assert_eq!(
        collision
            .cast_shape(
                unit_box().shape(),
                Mat4::from_translation(-Vec3::X * 2.0),
                Vec3::X * 4.0,
                QueryFilter::default(),
                ShapeCastOptions::default()
            )
            .unwrap_err(),
        CollisionError::CandidateBudget
    );
}

#[test]
fn invalid_and_overflow_colliders_fail_queries_explicitly_and_recover() {
    let mut engine = engine();
    let a = spawn(&mut engine, unit_box(), Transform::default());
    let b = spawn(&mut engine, unit_box(), Transform::default());
    engine
        .app
        .world_mut()
        .resource_mut::<CollisionConfig>()
        .max_colliders = 1;
    let report = sync_engine(&mut engine);
    assert_eq!(
        (
            report.colliders,
            report.active_colliders,
            report.overflow_colliders
        ),
        (2, 1, 1)
    );
    assert!(!report.complete);
    assert_eq!(
        engine
            .app
            .world()
            .resource::<CollisionWorld>()
            .raycast(ray(-Vec3::X, Vec3::X, 2.0), QueryFilter::default())
            .unwrap_err(),
        CollisionError::IncompleteWorld
    );
    engine
        .app
        .world_mut()
        .resource_mut::<CollisionConfig>()
        .max_colliders = 2;
    edit_transform(engine.app.world_mut(), a, |transform| {
        transform.translation.x = 2.0e6;
    })
    .unwrap();
    let report = sync_engine(&mut engine);
    assert_eq!((report.active_colliders, report.invalid_colliders), (1, 1));
    engine.app.world_mut().entity_mut(a).remove::<Transform>();
    assert_eq!(sync_engine(&mut engine).invalid_colliders, 1);
    engine.app.world_mut().despawn(a);
    assert!(sync_engine(&mut engine).complete);
    assert_eq!(
        engine
            .app
            .world()
            .resource::<CollisionWorld>()
            .collider_status(b),
        Some(ColliderStatus::Active)
    );
}

#[test]
fn generation_reuse_does_not_inherit_sensor_identity_and_filter_change_exits() {
    let mut engine = engine();
    let sensor = spawn(
        &mut engine,
        unit_box().with_sensor(true),
        Transform::default(),
    );
    let old = spawn(&mut engine, unit_box(), Transform::default());
    engine.advance(0.125);
    let new = engine.app.world_mut().despawn_no_free(old).unwrap();
    engine
        .app
        .world_mut()
        .spawn_at(new, (unit_box(), Transform::default()))
        .unwrap();
    engine.advance(0.125);
    let events = engine.app.world().resource::<SensorEvents>();
    assert_eq!(events.events().len(), 2);
    assert_eq!(events.pairs(), &[SensorPair::new(sensor, new)]);
    assert!(
        !events
            .pairs()
            .iter()
            .any(|pair| pair.first == old || pair.second == old)
    );
    engine
        .app
        .world_mut()
        .entity_mut(new)
        .insert(unit_box().with_layers(CollisionLayers::new(0, 0)));
    engine.advance(0.125);
    let events = engine.app.world().resource::<SensorEvents>();
    assert_eq!(events.events().len(), 1);
    assert_eq!(events.events()[0].kind, SensorEventKind::Exit);
}

#[test]
fn unchanged_cache_retains_revision_storage_and_despawn_releases_live_ids() {
    let mut engine = engine();
    let entities: Vec<_> = (0..1000)
        .map(|index| {
            spawn(
                &mut engine,
                unit_box(),
                Transform {
                    translation: Vec3::X * (index as f32 * 2.0),
                    ..Default::default()
                },
            )
        })
        .collect();
    let first = sync_engine(&mut engine);
    let collision = engine.app.world().resource::<CollisionWorld>();
    let pointers = (
        collision.inputs.as_ptr(),
        collision.entries.as_ptr(),
        collision.bounds.as_ptr(),
    );
    assert_eq!(first.bvh_nodes, 1999);
    engine.advance(0.125);
    let second = sync_engine(&mut engine);
    assert_eq!(first.revision, second.revision);
    assert_eq!(first.cache_bytes, second.cache_bytes);
    let collision = engine.app.world().resource::<CollisionWorld>();
    assert_eq!(
        pointers,
        (
            collision.inputs.as_ptr(),
            collision.entries.as_ptr(),
            collision.bounds.as_ptr()
        )
    );
    for entity in entities {
        engine.app.world_mut().despawn(entity);
    }
    let empty = sync_engine(&mut engine);
    assert_eq!(empty.active_colliders, 0);
    assert_eq!(empty.bvh_nodes, 0);
    assert_eq!(first.cache_bytes, empty.cache_bytes);
    let collision = engine.app.world().resource::<CollisionWorld>();
    assert!(
        collision.inputs.is_empty()
            && collision.entries.is_empty()
            && collision.statuses.is_empty()
    );
}

#[test]
fn broadphase_sparse_queries_do_not_visit_every_collider() {
    let mut engine = engine();
    for index in 0..10_000 {
        spawn(
            &mut engine,
            unit_box(),
            Transform {
                translation: Vec3::new((index % 100) as f32 * 3.0, 0.0, (index / 100) as f32 * 3.0),
                ..Default::default()
            },
        );
    }
    sync_engine(&mut engine);
    let collision = engine.app.world().resource::<CollisionWorld>();
    let bounds = Bounds {
        min: Vec3::splat(-0.25),
        max: Vec3::splat(0.25),
    };
    let tests = std::cell::Cell::new(0);
    let hits: Vec<_> = collision
        .bvh
        .query(|candidate| {
            tests.set(tests.get() + 1);
            candidate.overlaps(bounds)
        })
        .collect();
    assert_eq!(hits.len(), 1);
    assert!(tests.get() < 100, "{} BVH nodes visited", tests.get());
    let mut overlaps = Vec::new();
    collision
        .overlap(
            unit_box().shape(),
            Mat4::IDENTITY,
            QueryFilter::default(),
            &mut overlaps,
            1,
        )
        .unwrap();
    assert_eq!(overlaps.len(), 1);
}

#[test]
fn affine_box_cast_retains_shear_reflection_and_world_normal() {
    let mut engine = engine();
    let parent = engine
        .app
        .world_mut()
        .spawn(Transform {
            scale: Vec3::new(-2.0, 1.0, 3.0),
            ..Default::default()
        })
        .id();
    let target = spawn(
        &mut engine,
        unit_box(),
        Transform {
            rotation: Quat::from_rotation_z(0.6),
            ..Default::default()
        },
    );
    set_parent(engine.app.world_mut(), target, Some(parent)).unwrap();
    sync_engine(&mut engine);
    let matrix = engine
        .app
        .world()
        .resource::<GlobalTransforms>()
        .get(target)
        .unwrap()
        .matrix()
        .unwrap();
    let axis = matrix.x_axis.truncate();
    let mut query_matrix = matrix;
    query_matrix.w_axis = (-axis * 4.0).extend(1.0);
    let hit = engine
        .app
        .world_mut()
        .resource_mut::<CollisionWorld>()
        .cast_shape(
            unit_box().shape(),
            query_matrix,
            axis * 8.0,
            QueryFilter::default(),
            ShapeCastOptions::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(hit.entity, target);
    assert!((hit.fraction - 0.375).abs() < 1.0e-6, "{hit:?}");
    let expected_normal = -matrix
        .inverse()
        .transpose()
        .transform_vector3(Vec3::X)
        .normalize();
    assert!(hit.normal.abs_diff_eq(expected_normal, 1.0e-5), "{hit:?}");
    let point_local = matrix.inverse().transform_point3(hit.point);
    assert!(point_local.abs().max_element() <= 0.50001);
    assert!((point_local.abs().max_element() - 0.5).abs() < 1.0e-5);
}

#[test]
fn numeric_domain_and_query_inputs_reject_invalid_or_unsafe_values() {
    for extent in [0.0, -1.0, f32::NAN, f32::INFINITY, 1.0e-6, 1.0e5] {
        assert_eq!(
            Collider::sphere(extent).unwrap_err(),
            CollisionError::InvalidShape
        );
        assert_eq!(
            Collider::cuboid(Vec3::new(1.0, extent, 1.0)).unwrap_err(),
            CollisionError::InvalidShape
        );
    }
    for radius in [MIN_COLLIDER_EXTENT, MAX_COLLIDER_EXTENT] {
        assert!(
            validate_collider_transform(Collider::sphere(radius).unwrap().shape(), Mat4::IDENTITY)
                .is_ok()
        );
    }
    let shape = unit_box().shape();
    for matrix in [
        Mat4::from_translation(Vec3::X * 2.0e6),
        Mat4::from_scale(Vec3::splat(1.0e6)),
        Mat4::from_scale(Vec3::splat(1.0e-6)),
        Mat4::from_scale(Vec3::new(1.0, 1.0, 1.0e-4)),
    ] {
        assert_eq!(
            validate_collider_transform(shape, matrix),
            Err(CollisionError::InvalidTransform)
        );
    }
    for invalid in [
        ray(Vec3::ZERO, Vec3::ZERO, 1.0),
        ray(Vec3::ZERO, Vec3::X, f32::INFINITY),
        ray(Vec3::ZERO, Vec3::X, -1.0),
        ray(Vec3::splat(f32::NAN), Vec3::X, 1.0),
        ray(Vec3::X * 1.0e6, Vec3::X, 1.0),
    ] {
        assert_eq!(invalid.validate(), Err(CollisionError::InvalidRay));
    }
    assert_eq!(
        validate_shape_cast(
            shape,
            Mat4::IDENTITY,
            Vec3::splat(f32::NAN),
            ShapeCastOptions::default()
        ),
        Err(CollisionError::InvalidCast)
    );
    assert_eq!(
        ShapeCastOptions {
            stop_at_penetration: true,
            target_distance: -0.1
        }
        .validate(),
        Err(CollisionError::InvalidCast)
    );
    let mut engine = engine();
    engine
        .app
        .world_mut()
        .resource_mut::<CollisionConfig>()
        .max_colliders = MAX_COLLISION_NODES + 1;
    let report = sync_engine(&mut engine);
    assert!(!report.complete);
    assert_eq!(
        engine.app.world().resource::<CollisionWorld>().last_error(),
        Some(CollisionError::InvalidConfig)
    );
}

#[test]
fn collision_uses_authoritative_pose_and_refresh_preserves_presented_pose() {
    let mut engine = engine();
    let entity = spawn(&mut engine, unit_box(), Transform::default());
    engine
        .app
        .world_mut()
        .entity_mut(entity)
        .insert(InterpolateTransform);
    engine
        .app
        .add_systems(FixedSimulation, move |world: &mut World| {
            edit_transform(world, entity, |transform| transform.translation.x += 4.0).unwrap();
        });
    engine.advance(0.125);
    engine
        .extract_with_presentation(
            1.0,
            &mut RenderSnapshot::default(),
            PresentationMode::Interpolated { alpha: 0.5 },
        )
        .unwrap();
    let prepared = engine
        .app
        .world()
        .resource::<PresentedTransforms>()
        .get(entity)
        .unwrap()
        .matrix();
    assert_eq!(prepared.unwrap().w_axis.x, 2.0);
    let collision = engine.app.world().resource::<CollisionWorld>();
    let hit = collision
        .raycast(ray(Vec3::ZERO, Vec3::X, 10.0), QueryFilter::default())
        .unwrap()
        .unwrap();
    assert_eq!(hit.distance, 3.5);
    edit_transform(engine.app.world_mut(), entity, |transform| {
        transform.translation.x = 8.0;
    })
    .unwrap();
    sync_engine(&mut engine);
    assert_eq!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get(entity)
            .unwrap()
            .matrix(),
        prepared
    );
    assert_eq!(engine.app.world().resource::<FixedTime>().tick, 1);
    assert_eq!(
        engine
            .app
            .world()
            .resource::<CollisionWorld>()
            .raycast(ray(Vec3::ZERO, Vec3::X, 10.0), QueryFilter::default())
            .unwrap()
            .unwrap()
            .distance,
        7.5
    );
}

#[test]
fn rotated_affine_box_sat_agrees_with_independent_parry_queries_away_from_boundaries() {
    use parry3d::{math::Pose, query::intersection_test, shape::Cuboid};
    let a = AffineShape::new(unit_box().shape(), Mat4::IDENTITY).unwrap();
    let mut state = 0x12345678_u32;
    let mut random = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 8) as f32 / (1_u32 << 24) as f32
    };
    for _ in 0..1000 {
        let translation = Vec3::new(
            random() * 4.0 - 2.0,
            random() * 4.0 - 2.0,
            random() * 4.0 - 2.0,
        );
        let rotation = Quat::from_euler(
            glam::EulerRot::XYZ,
            random() * 3.0,
            random() * 3.0,
            random() * 3.0,
        );
        let scale = Vec3::new(random() + 0.5, random() + 0.5, random() + 0.5);
        let b = AffineShape::new(
            unit_box().shape(),
            Mat4::from_scale_rotation_translation(scale, rotation, translation),
        )
        .unwrap();
        let reference = intersection_test(
            &Pose::IDENTITY,
            &Cuboid::new(Vec3::splat(0.5)),
            &Pose::from_parts(translation, rotation),
            &Cuboid::new(scale * 0.5),
        )
        .unwrap();
        assert_eq!(
            intersects(&a, &b).unwrap(),
            reference,
            "translation={translation:?}, rotation={rotation:?}, scale={scale:?}"
        );
    }
}

#[test]
fn refitted_trees_answer_queries_like_trees_built_from_scratch() {
    let mut state = 0x9e37_79b9_u32;
    let mut random = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0
    };
    let count = 160;
    let colliders: Vec<Collider> = (0..count)
        .map(|index| {
            if index % 3 == 0 {
                Collider::sphere(0.6).unwrap()
            } else {
                unit_box()
            }
            .with_sensor(index % 7 == 0)
        })
        .collect();
    let mut poses: Vec<Transform> = (0..count)
        .map(|_| Transform {
            translation: Vec3::new(random(), random(), random()) * 20.0,
            rotation: Quat::from_rotation_y(random() * 3.0),
            scale: Vec3::new(1.0 + random().abs(), 1.0, 1.0 + random().abs()),
        })
        .collect();
    let mut moving = engine();
    let entities: Vec<Entity> = (0..count)
        .map(|index| spawn(&mut moving, colliders[index], poses[index]))
        .collect();
    sync_engine(&mut moving);
    let index_of = |world: &World, entity: Entity| {
        world
            .iter_entities()
            .filter(|entity| entity.contains::<Collider>())
            .position(|candidate| candidate.id() == entity)
            .unwrap()
    };
    for step in 0..12 {
        // One large jump degrades the refitted tree past its limit.
        let scale = if step == 8 { 40.0 } else { 0.4 };
        for (&entity, pose) in entities.iter().zip(&mut poses) {
            pose.translation += Vec3::new(random(), random(), random()) * scale;
            *moving.app.world_mut().get_mut::<Transform>(entity).unwrap() = *pose;
        }
        let refits = moving.app.world().resource::<CollisionWorld>().bvh.refits;
        sync_engine(&mut moving);
        let refitted = moving.app.world().resource::<CollisionWorld>().bvh.refits > refits;
        assert_eq!(refitted, step != 8, "step {step}");
        let mut fresh = engine();
        for (collider, pose) in colliders.iter().zip(&poses) {
            spawn(&mut fresh, *collider, *pose);
        }
        sync_engine(&mut fresh);
        for _ in 0..40 {
            let origin = Vec3::new(random(), random(), random()) * 30.0;
            let direction = Vec3::new(random(), random(), random()) + Vec3::splat(1.0e-3);
            let probe = ray(origin, direction, 60.0);
            let mut hits = [Vec::new(), Vec::new()];
            let mut overlaps = [Vec::new(), Vec::new()];
            let mut nearest = [None, None];
            let mut casts = [None, None];
            for (slot, engine) in [&mut moving, &mut fresh].into_iter().enumerate() {
                let world = engine.app.world_mut();
                let mut collisions = world.remove_resource::<CollisionWorld>().unwrap();
                nearest[slot] = collisions
                    .raycast(probe, QueryFilter::default())
                    .unwrap()
                    .map(|hit| {
                        (
                            index_of(world, hit.entity),
                            hit.distance.to_bits(),
                            hit.normal,
                        )
                    });
                let mut all = Vec::new();
                collisions
                    .raycast_all(probe, QueryFilter::default(), &mut all, 256)
                    .unwrap();
                hits[slot] = all
                    .iter()
                    .map(|hit| (index_of(world, hit.entity), hit.distance.to_bits()))
                    .collect();
                let mut entities = Vec::new();
                collisions
                    .overlap(
                        ColliderShape::Cuboid {
                            half_extents: Vec3::splat(4.0),
                        },
                        Mat4::from_translation(origin * 0.5),
                        QueryFilter::default(),
                        &mut entities,
                        256,
                    )
                    .unwrap();
                let mut indices: Vec<_> = entities.iter().map(|&e| index_of(world, e)).collect();
                indices.sort_unstable();
                overlaps[slot] = indices;
                casts[slot] = collisions
                    .cast_shape(
                        ColliderShape::Sphere { radius: 0.5 },
                        Mat4::from_translation(origin),
                        direction.normalize() * 30.0,
                        QueryFilter::default(),
                        ShapeCastOptions::default(),
                    )
                    .unwrap()
                    .map(|hit| {
                        (
                            index_of(world, hit.entity),
                            hit.fraction.to_bits(),
                            hit.normal,
                        )
                    });
                world.insert_resource(collisions);
            }
            assert_eq!(nearest[0], nearest[1]);
            assert_eq!(hits[0], hits[1]);
            assert_eq!(overlaps[0], overlaps[1]);
            assert_eq!(casts[0], casts[1]);
        }
        assert_eq!(
            moving
                .app
                .world()
                .resource::<CollisionWorld>()
                .report()
                .bvh_nodes,
            fresh
                .app
                .world()
                .resource::<CollisionWorld>()
                .report()
                .bvh_nodes
        );
    }
}
