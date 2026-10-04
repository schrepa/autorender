use super::*;
use crate::prelude::*;

fn spawn(world: &mut World, translation: Vec3) -> Entity {
    world
        .spawn(Transform {
            translation,
            ..Transform::default()
        })
        .id()
}

fn matrix(world: &World, entity: Entity) -> Mat4 {
    world
        .resource::<GlobalTransforms>()
        .get(entity)
        .unwrap()
        .matrix()
        .unwrap()
}

#[test]
fn composed_rotation_and_nonuniform_scale_retain_shear() {
    let mut world = World::new();
    let parent_local = Transform {
        translation: Vec3::new(4.0, 3.0, -2.0),
        rotation: Quat::from_rotation_y(0.4),
        scale: Vec3::new(3.0, 1.0, 2.0),
    };
    let child_local = Transform {
        translation: Vec3::new(1.0, 2.0, 3.0),
        rotation: Quat::from_rotation_z(0.7),
        scale: Vec3::new(1.0, 0.5, 2.0),
    };
    let parent = world.spawn(parent_local).id();
    let child = world.spawn(child_local).id();
    let grandchild = spawn(&mut world, Vec3::X);
    set_parents(
        &mut world,
        &[(grandchild, Some(child)), (child, Some(parent))],
    )
    .unwrap();
    assert_eq!(propagate(&mut world).invalid_transforms, 0);
    let expected = parent_local.matrix() * child_local.matrix();
    assert!(matrix(&world, child).abs_diff_eq(expected, 1.0e-6));
    assert!(
        matrix(&world, grandchild).abs_diff_eq(expected * Mat4::from_translation(Vec3::X), 1.0e-6)
    );
    // Orthogonal TRS decomposition would lose this composed shear.
    assert!(
        expected
            .x_axis
            .truncate()
            .dot(expected.y_axis.truncate())
            .abs()
            > 0.5
    );
}

#[test]
fn reparent_and_detach_preserve_local_values() {
    let mut world = World::new();
    let parent = spawn(&mut world, Vec3::new(5.0, 0.0, 0.0));
    let child = spawn(&mut world, Vec3::Y);
    set_parent(&mut world, child, Some(parent)).unwrap();
    propagate(&mut world);
    assert_eq!(
        matrix(&world, child).w_axis.truncate(),
        Vec3::new(5.0, 1.0, 0.0)
    );
    set_parent(&mut world, child, None).unwrap();
    propagate(&mut world);
    assert_eq!(world.get::<Transform>(child).unwrap().translation, Vec3::Y);
    assert_eq!(matrix(&world, child).w_axis.truncate(), Vec3::Y);
    assert!(world.get::<Parent>(child).is_none());
}

#[test]
fn rejected_cycles_and_missing_targets_do_not_change_relationships_or_globals() {
    let mut world = World::new();
    let root = spawn(&mut world, Vec3::X);
    let child = spawn(&mut world, Vec3::Y);
    set_parent(&mut world, child, Some(root)).unwrap();
    propagate(&mut world);
    let before = matrix(&world, child);
    assert!(matches!(
        set_parent(&mut world, root, Some(child)),
        Err(HierarchyError::Cycle(_))
    ));
    assert!(world.get::<Parent>(root).is_none());
    assert_eq!(world.get::<Parent>(child).unwrap().parent(), root);
    assert_eq!(matrix(&world, child), before);
    assert!(matches!(
        set_parent(&mut world, root, Some(root)),
        Err(HierarchyError::Cycle(_))
    ));
    let no_transform = world.spawn_empty().id();
    assert_eq!(
        set_parent(&mut world, child, Some(no_transform)),
        Err(HierarchyError::MissingTransform(no_transform))
    );
    world.despawn(no_transform);
    assert_eq!(
        set_parent(&mut world, child, Some(no_transform)),
        Err(HierarchyError::MissingEntity(no_transform))
    );
    assert_eq!(world.get::<Parent>(child).unwrap().parent(), root);
}

#[test]
fn whole_batch_rejects_cycles_and_duplicates_before_installing_any_link() {
    let mut world = World::new();
    let first = spawn(&mut world, Vec3::ZERO);
    let second = spawn(&mut world, Vec3::ZERO);
    let third = spawn(&mut world, Vec3::ZERO);
    assert!(matches!(
        set_parents(
            &mut world,
            &[
                (first, Some(second)),
                (second, Some(third)),
                (third, Some(first))
            ]
        ),
        Err(HierarchyError::Cycle(_))
    ));
    for entity in [first, second, third] {
        assert!(world.get::<Parent>(entity).is_none());
    }
    assert_eq!(
        set_parents(&mut world, &[(first, Some(second)), (first, None)]),
        Err(HierarchyError::DuplicateChild(first))
    );
    assert!(world.get::<Parent>(first).is_none());
}

#[test]
fn depth_limit_includes_existing_descendants_when_moving_a_subtree() {
    let mut world = World::new();
    let chain: Vec<_> = (0..=MAX_HIERARCHY_DEPTH)
        .map(|_| spawn(&mut world, Vec3::Y))
        .collect();
    let links: Vec<_> = chain
        .windows(2)
        .map(|pair| (pair[1], Some(pair[0])))
        .collect();
    set_parents(&mut world, &links).unwrap();
    assert_eq!(propagate(&mut world).invalid_transforms, 0);
    let extra = spawn(&mut world, Vec3::ZERO);
    assert_eq!(
        set_parent(&mut world, chain[0], Some(extra)),
        Err(HierarchyError::DepthExceeded)
    );
    assert!(world.get::<Parent>(chain[0]).is_none());
    assert_eq!(
        set_parent(&mut world, extra, Some(*chain.last().unwrap())),
        Err(HierarchyError::DepthExceeded)
    );
    assert!(world.get::<Parent>(extra).is_none());
    // Moving the existing root and detaching its subtree in one batch is valid.
    set_parents(&mut world, &[(chain[0], Some(extra)), (chain[1], None)]).unwrap();
    assert_eq!(propagate(&mut world).invalid_transforms, 0);
}

#[test]
fn despawned_parent_does_not_own_children_or_bind_to_a_reused_entity_slot() {
    let mut world = World::new();
    let parent = spawn(&mut world, Vec3::X);
    let child = spawn(&mut world, Vec3::Y);
    set_parent(&mut world, child, Some(parent)).unwrap();
    propagate(&mut world);
    let replacement = world.despawn_no_free(parent).unwrap();
    world
        .spawn_at(
            replacement,
            Transform {
                translation: Vec3::new(99.0, 0.0, 0.0),
                ..Transform::default()
            },
        )
        .unwrap();
    assert_eq!(parent.index(), replacement.index());
    assert_ne!(parent, replacement);
    let report = propagate(&mut world);
    assert_eq!(report.invalid_transforms, 1);
    assert!(world.entities().contains(child));
    assert!(
        world
            .resource::<GlobalTransforms>()
            .get(child)
            .unwrap()
            .matrix()
            .is_none()
    );
    set_parent(&mut world, child, None).unwrap();
    assert_eq!(propagate(&mut world).invalid_transforms, 0);
    assert_eq!(matrix(&world, child).w_axis.truncate(), Vec3::Y);
}

#[test]
fn removing_a_local_transform_invalidates_its_derived_value_and_descendants() {
    let mut world = World::new();
    let parent = spawn(&mut world, Vec3::X);
    let child = spawn(&mut world, Vec3::Y);
    set_parent(&mut world, child, Some(parent)).unwrap();
    propagate(&mut world);
    world.entity_mut(parent).remove::<Transform>();
    assert_eq!(propagate(&mut world).invalid_transforms, 1);
    assert!(world.resource::<GlobalTransforms>().get(parent).is_none());
    assert!(
        world
            .resource::<GlobalTransforms>()
            .get(child)
            .unwrap()
            .matrix()
            .is_none()
    );
}

#[test]
fn invalid_runtime_graphs_skip_only_the_affected_renderables_and_can_be_repaired() {
    let mut engine = Engine::default();
    let world = engine.app.world_mut();
    let healthy = spawn(world, Vec3::X);
    let first = spawn(world, Vec3::Y);
    let second = spawn(world, Vec3::Z);
    for entity in [healthy, first, second] {
        world.entity_mut(entity).insert(Cube { color: [1.0; 4] });
    }
    // Private test construction simulates unchecked copying of Parent values.
    world.entity_mut(first).insert(Parent(second));
    world.entity_mut(second).insert(Parent(first));
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.instances.len(), 1);
    assert_eq!(snapshot.invalid_transforms, 2);
    assert_eq!(snapshot.missing_assets, 0);
    // Unrelated invalid branches do not reject a healthy attachment.
    let next = spawn(engine.app.world_mut(), Vec3::ZERO);
    set_parent(engine.app.world_mut(), next, Some(healthy)).unwrap();
    set_parent(engine.app.world_mut(), first, None).unwrap();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.instances.len(), 3);
    assert_eq!(snapshot.invalid_transforms, 0);
}

#[test]
fn invalid_runtime_depth_leaves_shallower_ancestors_valid() {
    let mut world = World::new();
    // Spawn deepest first to exercise an unresolved chain longer than the limit.
    let chain: Vec<_> = (0..MAX_HIERARCHY_DEPTH + 3)
        .map(|_| spawn(&mut world, Vec3::Y))
        .collect();
    for pair in chain.windows(2) {
        world.entity_mut(pair[0]).insert(Parent(pair[1]));
    }
    let report = propagate(&mut world);
    assert_eq!(report.invalid_transforms, 2);
    assert!(
        world
            .resource::<GlobalTransforms>()
            .get(chain[0])
            .unwrap()
            .matrix()
            .is_none()
    );
    assert!(
        world
            .resource::<GlobalTransforms>()
            .get(chain[1])
            .unwrap()
            .matrix()
            .is_none()
    );
    assert!(
        world
            .resource::<GlobalTransforms>()
            .get(chain[2])
            .unwrap()
            .matrix()
            .is_some()
    );
}

#[test]
fn invalid_matrices_and_rotation_do_not_poison_healthy_branches() {
    let mut world = World::new();
    let healthy = spawn(&mut world, Vec3::X);
    world.spawn(Transform {
        scale: Vec3::ZERO,
        ..Transform::default()
    });
    world.spawn(Transform {
        translation: Vec3::splat(f32::NAN),
        ..Transform::default()
    });
    world.spawn(Transform {
        rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
        ..Transform::default()
    });
    let huge = world
        .spawn(Transform {
            scale: Vec3::splat(1.0e10),
            ..Transform::default()
        })
        .id();
    let overflow = world
        .spawn(Transform {
            scale: Vec3::splat(1.0e10),
            ..Transform::default()
        })
        .id();
    set_parent(&mut world, overflow, Some(huge)).unwrap();
    assert_eq!(propagate(&mut world).invalid_transforms, 4);
    assert_eq!(matrix(&world, healthy).w_axis.truncate(), Vec3::X);
    assert!(!valid_global_matrix(Mat4::from_scale(Vec3::splat(1.0e-30))));
    assert!(!valid_global_matrix(Mat4::from_scale(Vec3::splat(1.0e30))));
}

#[test]
fn paused_extraction_observes_direct_edits_without_advancing_clocks() {
    let mut engine = Engine::default();
    let parent = spawn(engine.app.world_mut(), Vec3::X);
    let child = spawn(engine.app.world_mut(), Vec3::Y);
    engine
        .app
        .world_mut()
        .entity_mut(child)
        .insert(Cube { color: [1.0; 4] });
    set_parent(engine.app.world_mut(), child, Some(parent)).unwrap();
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.0, &mut snapshot);
    edit_transform(engine.app.world_mut(), parent, |transform| {
        transform.translation = Vec3::Z;
    })
    .unwrap();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(
        snapshot.instances[0].model.w_axis.truncate(),
        Vec3::new(0.0, 1.0, 1.0)
    );
    assert_eq!(engine.app.world().resource::<FixedTime>().tick, 0);
    assert_eq!(engine.app.world().resource::<FrameTime>().frame, 0);
}

#[derive(Resource, Default)]
struct Seen(Vec<Vec3>);

#[test]
fn globals_refresh_between_fixed_ticks_and_after_presentation() {
    let mut engine = Engine::new(EngineConfig {
        fixed_hz: 8.0,
        ..EngineConfig::default()
    });
    let parent = spawn(engine.app.world_mut(), Vec3::ZERO);
    let child = spawn(engine.app.world_mut(), Vec3::Y);
    set_parent(engine.app.world_mut(), child, Some(parent)).unwrap();
    engine
        .app
        .init_resource::<Seen>()
        .add_systems(FixedSimulation, move |world: &mut World| {
            let observed = world
                .resource::<GlobalTransforms>()
                .get(child)
                .unwrap()
                .translation()
                .unwrap();
            world.resource_mut::<Seen>().0.push(observed);
            world.get_mut::<Transform>(parent).unwrap().translation.x += 1.0;
        });
    engine.app.add_systems(
        Presentation,
        move |mut transforms: Query<&mut Transform>| {
            transforms.get_mut(parent).unwrap().translation.z = 7.0;
        },
    );
    engine.advance(0.25);
    assert_eq!(
        engine.app.world().resource::<Seen>().0,
        vec![Vec3::Y, Vec3::new(1.0, 1.0, 0.0)]
    );
    assert_eq!(
        matrix(engine.app.world(), child).w_axis.truncate(),
        Vec3::new(2.0, 1.0, 7.0)
    );
}

#[cfg(feature = "dev")]
#[test]
fn exact_development_ticks_refresh_globals_before_observation() {
    let mut engine = Engine::default();
    let parent = spawn(engine.app.world_mut(), Vec3::ZERO);
    let child = spawn(engine.app.world_mut(), Vec3::Y);
    set_parent(engine.app.world_mut(), child, Some(parent)).unwrap();
    engine.app.add_systems(
        FixedSimulation,
        move |mut transforms: Query<&mut Transform>| {
            transforms.get_mut(parent).unwrap().translation.x += 1.0;
        },
    );
    engine.step_fixed(3).unwrap();
    assert_eq!(
        matrix(engine.app.world(), child).w_axis.truncate(),
        Vec3::new(3.0, 1.0, 0.0)
    );
}

#[test]
fn propagation_reuses_storage_and_releases_despawned_entity_references() {
    let mut world = World::new();
    let entities: Vec<_> = (0..128).map(|_| spawn(&mut world, Vec3::Y)).collect();
    let links: Vec<_> = entities[..64]
        .windows(2)
        .map(|pair| (pair[1], Some(pair[0])))
        .collect();
    set_parents(&mut world, &links).unwrap();
    propagate(&mut world);
    let storage = |scratch: &HierarchyScratch| {
        [
            scratch.sources.as_ptr() as usize,
            scratch.locals.as_ptr() as usize,
            scratch.history_hints.as_ptr() as usize,
            scratch.links.as_ptr() as usize,
            scratch.globals.as_ptr() as usize,
            scratch.revisions.as_ptr() as usize,
        ]
    };
    let lengths = |scratch: &HierarchyScratch| {
        [
            scratch.sources.len(),
            scratch.locals.len(),
            scratch.history_hints.len(),
            scratch.links.len(),
            scratch.globals.len(),
            scratch.revisions.len(),
        ]
    };
    let (arrays, indices, path) = {
        let scratch = world.resource::<HierarchyScratch>();
        (
            storage(scratch),
            scratch.indices.capacity(),
            scratch.path.as_ptr(),
        )
    };
    propagate(&mut world);
    let scratch = world.resource::<HierarchyScratch>();
    assert_eq!(storage(scratch), arrays);
    assert_eq!(lengths(scratch), [128; 6]);
    assert_eq!(scratch.indices.capacity(), indices);
    assert_eq!(scratch.path.as_ptr(), path);
    for entity in entities {
        world.despawn(entity);
    }
    assert_eq!(propagate(&mut world), HierarchyReport::default());
    let scratch = world.resource::<HierarchyScratch>();
    assert_eq!(lengths(scratch), [0; 6]);
    assert!(scratch.indices.is_empty());
    assert!(scratch.path.is_empty());
    assert_eq!(storage(scratch), arrays);
}

#[test]
fn unchanged_boundaries_retain_published_pose_revisions() {
    let mut world = World::new();
    let parent = spawn(&mut world, Vec3::X);
    let child = spawn(&mut world, Vec3::Y);
    set_parent(&mut world, child, Some(parent)).unwrap();
    propagate(&mut world);
    world.clear_trackers();
    propagate(&mut world);
    assert_eq!(changed_globals(&mut world).len(), 0);
    edit_transform(&mut world, parent, |transform| {
        transform.translation = Vec3::Z;
    })
    .unwrap();
    propagate(&mut world);
    assert_eq!(
        matrix(&world, child).w_axis.truncate(),
        Vec3::new(0.0, 1.0, 1.0)
    );
}

#[test]
fn mutable_edits_survive_tracker_cleanup_and_archetype_moves() {
    #[derive(Component)]
    struct Unrelated;

    let mut world = World::new();
    let entities: Vec<_> = (0..16).map(|_| spawn(&mut world, Vec3::ZERO)).collect();
    propagate(&mut world);
    let a = entities[0];
    let b = entities[1];
    for x in 1..=5 {
        world.get_mut::<Transform>(a).unwrap().translation.x = x as f32;
    }
    world.get_mut::<Transform>(b).unwrap().translation.y = 2.0;
    world.get_mut::<Transform>(a).unwrap().translation.x = 6.0;
    // The engine watermark is independent of ordinary Bevy tracker cleanup.
    world.clear_trackers();
    propagate(&mut world);
    assert_eq!(matrix(&world, a).w_axis.truncate(), Vec3::X * 6.0);
    assert_eq!(matrix(&world, b).w_axis.truncate(), Vec3::Y * 2.0);
    let mut changed = vec![a, b];
    changed.sort_unstable();
    assert_eq!(changed_globals(&mut world), changed);
    // Moving between archetypes changes ECS query order without changing poses.
    world.entity_mut(a).insert(Unrelated);
    world.clear_trackers();
    propagate(&mut world);
    assert!(changed_globals(&mut world).is_empty());
    let mut poses = world.query::<&mut Transform>();
    // No caller-maintained tick or schedule separates these direct query edits.
    for z in [3.0, 4.0] {
        poses.get_mut(&mut world, a).unwrap().translation.z = z;
        propagate(&mut world);
        assert_eq!(matrix(&world, a).w_axis.truncate(), Vec3::new(6.0, 0.0, z));
        assert_eq!(matrix(&world, b).w_axis.truncate(), Vec3::Y * 2.0);
        for &entity in &entities[2..] {
            assert_eq!(matrix(&world, entity).w_axis.truncate(), Vec3::ZERO);
        }
    }
}

#[test]
fn aged_scan_window_recovers_an_unobserved_clamped_pose() {
    let mut world = World::new();
    let parent = spawn(&mut world, Vec3::X);
    let child = spawn(&mut world, Vec3::Y);
    let unrelated = spawn(&mut world, Vec3::Z);
    set_parent(&mut world, child, Some(parent)).unwrap();
    propagate(&mut world);
    world.clear_trackers();

    let now = world.read_change_tick();
    let aged = Tick::new(now.get().wrapping_sub(MAX_CHANGE_AGE));
    {
        let mut pose = world.get_mut::<Transform>(parent).unwrap();
        pose.translation = Vec3::X * 4.0;
        // Emulate Bevy clamping an old, unobserved edit to the same age as the
        // previous scan. Manual tick editing is only a test fixture here, not a
        // supported way for callers to suppress transform change detection.
        pose.set_last_changed(aged);
    }
    world.resource_mut::<HierarchyScratch>().last_scan = Some(aged);
    assert!(!aged.is_newer_than(aged, now));
    assert!(!world.resource::<tracking::Invalidation>().full_scan);

    assert_eq!(propagate(&mut world).invalid_transforms, 0);
    assert_eq!(matrix(&world, parent).w_axis.truncate(), Vec3::X * 4.0);
    assert_eq!(
        matrix(&world, child).w_axis.truncate(),
        Vec3::new(4.0, 1.0, 0.0)
    );
    assert_eq!(matrix(&world, unrelated).w_axis.truncate(), Vec3::Z);
    let mut expected = vec![parent, child];
    expected.sort_unstable();
    assert_eq!(changed_globals(&mut world), expected);
    world.clear_trackers();
    propagate(&mut world);
    assert!(changed_globals(&mut world).is_empty());
}

#[test]
fn scan_watermark_detects_mutation_across_tick_wraparound() {
    let mut world = World::new();
    let parent = spawn(&mut world, Vec3::ZERO);
    let child = spawn(&mut world, Vec3::Y);
    set_parent(&mut world, child, Some(parent)).unwrap();
    propagate(&mut world);
    let now = world.read_change_tick();
    let before_wrap = Tick::new(u32::MAX - 1);
    assert!(now.get() < before_wrap.get());
    assert!(now.get().wrapping_sub(before_wrap.get()) < MAX_CHANGE_AGE);
    assert!(now.is_newer_than(before_wrap, now));
    world.resource_mut::<HierarchyScratch>().last_scan = Some(before_wrap);
    assert!(!world.resource::<tracking::Invalidation>().full_scan);

    world.get_mut::<Transform>(parent).unwrap().translation.x = 5.0;
    world.clear_trackers();
    assert_eq!(propagate(&mut world).invalid_transforms, 0);
    assert_eq!(
        matrix(&world, child).w_axis.truncate(),
        Vec3::new(5.0, 1.0, 0.0)
    );
    // The newly committed window also admits another immediate ordinary edit.
    world.get_mut::<Transform>(parent).unwrap().translation.x = 6.0;
    propagate(&mut world);
    assert_eq!(
        matrix(&world, child).w_axis.truncate(),
        Vec3::new(6.0, 1.0, 0.0)
    );
}

#[test]
fn explicit_insert_observers_run_but_in_place_propagation_emits_no_lifecycle_events() {
    #[derive(Resource, Default)]
    struct Observed {
        inserts: usize,
        discards: usize,
    }

    let mut world = World::new();
    let parent = spawn(&mut world, Vec3::ZERO);
    let child = spawn(&mut world, Vec3::Y);
    set_parent(&mut world, child, Some(parent)).unwrap();
    propagate(&mut world);
    world.init_resource::<Observed>();
    world.add_observer(
        move |event: On<Insert, Transform>,
              mut observed: ResMut<Observed>,
              mut commands: Commands| {
            if event.entity != parent {
                return;
            }
            observed.inserts += 1;
            if observed.inserts != 1 {
                return;
            }
            commands.entity(parent).insert(Transform {
                translation: Vec3::X * 4.0,
                ..Default::default()
            });
            commands.entity(child).insert(Transform {
                translation: Vec3::Y * 3.0,
                ..Default::default()
            });
        },
    );
    world.add_observer(
        move |event: On<Discard, Transform>, mut observed: ResMut<Observed>| {
            if event.entity == parent {
                observed.discards += 1;
            }
        },
    );
    world.get_mut::<Transform>(parent).unwrap().translation.x = 2.0;
    assert_eq!(propagate(&mut world).invalid_transforms, 0);
    assert_eq!(matrix(&world, parent).w_axis.x, 2.0);
    assert_eq!(world.resource::<Observed>().inserts, 0);
    assert_eq!(world.resource::<Observed>().discards, 0);

    // Explicit insertion retains Bevy's lifecycle events and deferred commands.
    world.entity_mut(parent).insert(Transform::default());
    world.flush();
    assert_eq!(propagate(&mut world).invalid_transforms, 0);
    assert_eq!(world.resource::<Observed>().inserts, 2);
    assert_eq!(world.resource::<Observed>().discards, 2);
    assert_eq!(matrix(&world, parent).w_axis.truncate(), Vec3::X * 4.0);
    assert_eq!(
        matrix(&world, child).w_axis.truncate(),
        Vec3::new(4.0, 3.0, 0.0)
    );
}

#[test]
fn mutable_pose_edits_remain_visible_after_caught_user_panics() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let mut world = World::new();
    let parent = spawn(&mut world, Vec3::ZERO);
    let child = spawn(&mut world, Vec3::Y);
    set_parent(&mut world, child, Some(parent)).unwrap();
    propagate(&mut world);
    world.add_observer(
        move |event: On<Discard, Transform>, mut failed: Local<bool>| {
            if event.entity == parent && !*failed {
                *failed = true;
                panic!("interrupt a local replacement before insertion");
            }
        },
    );
    world.get_mut::<Transform>(parent).unwrap().translation.x = 4.0;
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.entity_mut(parent).insert(Transform::default());
        }))
        .is_err()
    );
    assert_eq!(propagate(&mut world).invalid_transforms, 0);
    assert_eq!(matrix(&world, parent).w_axis.truncate(), Vec3::X * 4.0);
    assert_eq!(
        matrix(&world, child).w_axis.truncate(),
        Vec3::new(4.0, 1.0, 0.0)
    );

    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.get_mut::<Transform>(parent).unwrap().translation.x = 7.0;
            panic!("interrupt a direct mutable edit after its value changed");
        }))
        .is_err()
    );
    assert_eq!(propagate(&mut world).invalid_transforms, 0);
    assert_eq!(matrix(&world, parent).w_axis.x, 7.0);
    assert_eq!(
        matrix(&world, child).w_axis.truncate(),
        Vec3::new(7.0, 1.0, 0.0)
    );

    // The typed modify API also marks mutation before invoking its closure.
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = world.modify_component(parent, |pose: &mut Transform| {
                pose.translation.x = 9.0;
                panic!("interrupt a modify closure after its value changed");
            });
        }))
        .is_err()
    );
    assert_eq!(propagate(&mut world).invalid_transforms, 0);
    assert_eq!(matrix(&world, parent).w_axis.truncate(), Vec3::X * 9.0);
    assert_eq!(
        matrix(&world, child).w_axis.truncate(),
        Vec3::new(9.0, 1.0, 0.0)
    );
}

#[test]
fn explicit_insert_observer_commands_preserve_removed_targets() {
    let mut world = World::new();
    let trigger = spawn(&mut world, Vec3::ZERO);
    let removed = spawn(&mut world, Vec3::Y);
    let parent = spawn(&mut world, Vec3::X);
    let child = spawn(&mut world, Vec3::Z);
    set_parent(&mut world, child, Some(parent)).unwrap();
    propagate(&mut world);
    world.add_observer(
        move |event: On<Insert, Transform>, mut commands: Commands| {
            if event.entity == trigger {
                commands.entity(removed).remove::<Transform>();
                commands.entity(parent).insert(Transform {
                    translation: Vec3::X * 7.0,
                    ..Default::default()
                });
            }
        },
    );
    world.entity_mut(trigger).insert(Transform::default());
    assert_eq!(propagate(&mut world).invalid_transforms, 0);
    assert!(world.resource::<GlobalTransforms>().get(removed).is_none());
    assert_eq!(
        matrix(&world, child).w_axis.truncate(),
        Vec3::new(7.0, 0.0, 1.0)
    );
    propagate(&mut world);
    assert!(changed_globals(&mut world).is_empty());
}

#[test]
fn indexed_reads_validate_entity_generations_and_topology_changes() {
    let mut world = World::new();
    let first = spawn(&mut world, Vec3::X);
    let second = spawn(&mut world, Vec3::Y);
    propagate(&mut world);
    let mut hint = TransformIndex::default();
    let first_pose = world
        .resource::<GlobalTransforms>()
        .get_indexed(first, &mut hint)
        .unwrap();
    assert_eq!(first_pose.translation(), Some(Vec3::X));
    assert_eq!(
        world
            .resource::<GlobalTransforms>()
            .get_indexed(second, &mut hint)
            .unwrap()
            .translation(),
        Some(Vec3::Y)
    );
    let replacement = world.despawn_no_free(first).unwrap();
    world
        .spawn_at(
            replacement,
            Transform {
                translation: Vec3::Z,
                ..Default::default()
            },
        )
        .unwrap();
    propagate(&mut world);
    assert!(
        world
            .resource::<GlobalTransforms>()
            .get_indexed(first, &mut hint)
            .is_none()
    );
    assert_eq!(
        world
            .resource::<GlobalTransforms>()
            .get_indexed(replacement, &mut hint)
            .unwrap()
            .translation(),
        Some(Vec3::Z)
    );
    assert_eq!(first_pose.translation(), Some(Vec3::X));
}

#[test]
fn canonical_revisions_survive_noops_and_structural_values_are_preserved() {
    let mut world = World::new();
    let entity = spawn(&mut world, Vec3::X);
    propagate(&mut world);
    let original = world.resource::<GlobalTransforms>().get(entity).unwrap();
    world.get_mut::<Transform>(entity).unwrap().translation = Vec3::X;
    world.clear_trackers();
    propagate(&mut world);
    assert_eq!(
        world
            .resource::<GlobalTransforms>()
            .get(entity)
            .unwrap()
            .revision(),
        original.revision()
    );
    spawn(&mut world, Vec3::Z);
    propagate(&mut world);
    assert_eq!(
        world
            .resource::<GlobalTransforms>()
            .get(entity)
            .unwrap()
            .matrix(),
        original.matrix()
    );
    world.get_mut::<Transform>(entity).unwrap().translation = Vec3::Y;
    propagate(&mut world);
    assert_ne!(
        world
            .resource::<GlobalTransforms>()
            .get(entity)
            .unwrap()
            .revision(),
        original.revision()
    );
    assert_eq!(original.translation(), Some(Vec3::X));
}

#[test]
fn finite_matrix_and_local_bit_comparisons_match_exact_bits() {
    let mut state = 0x853c_49e6_748f_ea9b_u64;
    let mut random = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
    };
    let specials = [
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1.0e-45,
        3.0e38,
    ];
    for case in 0..20_000 {
        let mut left = Mat4::from_cols_array(&std::array::from_fn(|_| random() * 100.0));
        let mut right = left;
        let lane = case % 16;
        match case % 4 {
            0 => {}
            1 => {
                right.col_mut(lane / 4)[lane % 4] =
                    f32::from_bits(left.col(lane / 4)[lane % 4].to_bits() ^ 1)
            }
            2 => {
                let value = specials[case % specials.len()];
                left.col_mut(lane / 4)[lane % 4] = value;
                right.col_mut(lane / 4)[lane % 4] = -value;
            }
            _ => right.col_mut(lane / 4)[lane % 4] = specials[case % specials.len()],
        }
        assert_eq!(
            super::same_finite_matrix_bits(left, right),
            super::same_matrix_bits(left, right),
            "{left:?} {right:?}"
        );
        let pose = |matrix: Mat4| Transform {
            translation: matrix.x_axis.truncate(),
            rotation: Quat::from_vec4(matrix.y_axis),
            scale: matrix.z_axis.truncate(),
        };
        let (a, b) = (pose(left), pose(right));
        let expected = [a.translation.to_array(), a.scale.to_array()]
            .concat()
            .iter()
            .chain(&a.rotation.to_array())
            .map(|value| value.to_bits())
            .eq([b.translation.to_array(), b.scale.to_array()]
                .concat()
                .iter()
                .chain(&b.rotation.to_array())
                .map(|value| value.to_bits()));
        assert_eq!(super::same_local_bits(&a, &b), expected);
    }
    // Identical NaN payloads are equal local bits; different payloads are not.
    let nan = Transform {
        translation: Vec3::splat(f32::NAN),
        ..Transform::default()
    };
    assert!(super::same_local_bits(&nan, &nan));
    let mut other = nan;
    other.translation.x = f32::from_bits(f32::NAN.to_bits() | 1);
    assert!(!super::same_local_bits(&nan, &other));
}

#[test]
fn vector_pose_comparison_matches_lane_bits() {
    let specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(f32::NAN.to_bits() | 1),
    ];
    let mut state = 0x2545_f491_u32;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for case in 0..200_000 {
        let mut left: NodePose = std::array::from_fn(|_| (random() % 7) as f32 - 3.0);
        let mut right = left;
        // One to three lanes differ, repeat or swap special values.
        for _ in 0..=case % 3 {
            let lane = random() as usize % 10;
            let special = specials[random() as usize % specials.len()];
            match random() % 3 {
                0 => left[lane] = special,
                1 => (left[lane], right[lane]) = (special, special),
                _ => right[lane] = special,
            }
        }
        let expected = left
            .iter()
            .zip(&right)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert_eq!(
            super::same_pose_bits(&left, &right),
            expected,
            "{left:?} {right:?}"
        );
    }
}

#[test]
fn change_window_matches_bevy_stamp_comparison_across_wraparound() {
    let mut state = 0x9e37_79b9_u32;
    let mut random = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let edges = [
        0,
        1,
        2,
        MAX_CHANGE_AGE - 1,
        MAX_CHANGE_AGE,
        MAX_CHANGE_AGE + 1,
        u32::MAX / 2,
        u32::MAX - 1,
        u32::MAX,
    ];
    let mut changed = 0;
    for case in 0..300_000_usize {
        let this_run = if case % 3 == 0 {
            edges[case % edges.len()]
        } else {
            random()
        };
        let window_age = if case % 5 == 0 {
            edges[(case / 5) % edges.len()]
        } else {
            random()
        };
        let stamp_age = if case % 7 == 0 {
            edges[(case / 7) % edges.len()]
        } else if case % 2 == 0 {
            random() % window_age.max(1).saturating_mul(2)
        } else {
            random()
        };
        let (last_run, this_run) = (
            Tick::new(this_run.wrapping_sub(window_age)),
            Tick::new(this_run),
        );
        let tick = Tick::new(this_run.get().wrapping_sub(stamp_age));
        let window = ChangeWindow::new(last_run, this_run);
        let expected = tick.is_newer_than(last_run, this_run);
        assert_eq!(
            window.contains(tick),
            expected,
            "{tick:?} {last_run:?} {this_run:?}"
        );
        assert_eq!(
            window.any(&[Tick::new(this_run.get()), tick][1..]),
            expected
        );
        changed += usize::from(expected);
    }
    assert!(changed > 30_000 && changed < 270_000, "{changed}");
}

#[test]
fn whole_slice_change_test_finds_a_change_in_blocks_and_the_tail() {
    let this_run = Tick::new(1_000);
    let window = ChangeWindow::new(Tick::new(900), this_run);
    let (old, new) = (Tick::new(10), Tick::new(950));
    assert!(!window.contains(old) && window.contains(new));
    // Two full blocks and a partial one.
    let len = 2 * STAMP_BLOCK + 5;
    assert!(!window.any(&vec![old; len]));
    assert!(!window.any(&[]));
    for position in 0..len {
        let mut stamps = vec![old; len];
        stamps[position] = new;
        assert!(window.any(&stamps), "{position}");
        assert!(window.any(&stamps[..=position]), "{position}");
        assert!(!window.any(&stamps[..position]), "{position}");
    }
}

/// Independent recursive reference using the scalar local matrix.
fn reference_global(world: &World, entity: Entity) -> Option<Mat4> {
    let local = *world.get::<Transform>(entity)?;
    let parent = match world.get::<Parent>(entity) {
        Some(parent) => reference_global(world, parent.parent())?,
        None => Mat4::IDENTITY,
    };
    let matrix = parent * super::local_matrix(&local)?;
    valid_global_matrix(matrix).then_some(matrix)
}

#[test]
fn dense_blocks_and_sparse_seeds_match_scalar_reference_bits() {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut random = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
    };
    let pose = |random: &mut dyn FnMut() -> f32| Transform {
        translation: Vec3::new(random(), random(), random()) * 50.0,
        rotation: Quat::from_euler(
            glam::EulerRot::YXZ,
            random() * 3.0,
            random() * 3.0,
            random() * 3.0,
        ),
        scale: Vec3::new(random(), random(), random()).signum()
            * (Vec3::new(random(), random(), random()).abs() * 2.8 + 0.2),
    };
    let mut world = World::new();
    let mut entities = Vec::new();
    let mut links = Vec::new();
    // 81 nodes: several parent/child dependencies share four-lane blocks and
    // the final block is partial.
    for _ in 0..3 {
        let root = world.spawn(pose(&mut random)).id();
        entities.push(root);
        for child in 0..21 {
            let entity = world.spawn(pose(&mut random)).id();
            links.push((entity, Some(root)));
            entities.push(entity);
            if child % 5 == 0 {
                let grandchild = world.spawn(pose(&mut random)).id();
                links.push((grandchild, Some(entity)));
                entities.push(grandchild);
            }
        }
    }
    set_parents(&mut world, &links).unwrap();
    let check = |world: &mut World| {
        super::propagate_transforms(world);
        let store = world.resource::<GlobalTransforms>();
        let mut invalid = 0;
        for &entity in &entities {
            let actual = store.get(entity).unwrap().matrix();
            let expected = reference_global(world, entity);
            invalid += usize::from(expected.is_none());
            assert_eq!(
                actual.map(|matrix| matrix.to_cols_array().map(f32::to_bits)),
                expected.map(|matrix| matrix.to_cols_array().map(f32::to_bits)),
                "{entity:?}"
            );
        }
        invalid
    };
    // Invalid locals inside dense blocks invalidate only their own branches.
    world
        .get_mut::<Transform>(entities[5])
        .unwrap()
        .translation
        .x = f32::NAN;
    world.get_mut::<Transform>(entities[30]).unwrap().rotation =
        Quat::from_xyzw(0.0, 0.0, 0.0, 0.9);
    assert!(check(&mut world) >= 2);
    for round in 0..6 {
        let moved: Vec<Entity> = match round % 3 {
            0 => entities.clone(),                              // dense
            1 => vec![entities[7], entities[40], entities[80]], // sparse
            _ => vec![entities[0], entities[28]],               // parents only
        };
        for entity in moved {
            let next = pose(&mut random);
            *world.get_mut::<Transform>(entity).unwrap() = next;
        }
        if round == 3 {
            world
                .get_mut::<Transform>(entities[5])
                .unwrap()
                .translation
                .x = 1.0;
        }
        check(&mut world);
    }
}

#[derive(Resource, Default)]
struct RevisionBaseline(HashMap<Entity, u64>);

fn propagate(world: &mut World) -> HierarchyReport {
    let previous = world
        .get_resource::<GlobalTransforms>()
        .map_or_else(HashMap::new, |poses| {
            poses
                .iter()
                .map(|(entity, pose)| (entity, pose.revision()))
                .collect()
        });
    world.insert_resource(RevisionBaseline(previous));
    super::propagate_transforms(world)
}

fn changed_globals(world: &mut World) -> Vec<Entity> {
    let current: HashMap<_, _> = world
        .resource::<GlobalTransforms>()
        .iter()
        .map(|(entity, pose)| (entity, pose.revision()))
        .collect();
    let mut entities: Vec<_> = current
        .iter()
        .filter_map(|(&entity, &revision)| {
            (world
                .get_resource::<RevisionBaseline>()
                .and_then(|old| old.0.get(&entity))
                .copied()
                != Some(revision))
            .then_some(entity)
        })
        .collect();
    world.insert_resource(RevisionBaseline(current));
    entities.sort_unstable();
    entities
}

#[test]
fn leaf_motion_preserves_other_globals_and_the_cached_topology() {
    let mut world = World::new();
    let leaf = spawn(&mut world, Vec3::Y);
    let sibling = spawn(&mut world, Vec3::Z);
    let parent = spawn(&mut world, Vec3::X);
    let unrelated = spawn(&mut world, Vec3::splat(9.0));
    set_parents(&mut world, &[(leaf, Some(parent)), (sibling, Some(parent))]).unwrap();
    propagate(&mut world);
    let unchanged: Vec<_> = [parent, sibling, unrelated]
        .map(|entity| (entity, matrix(&world, entity)))
        .into();
    // Interpolation indexes this exact parent-before-child cache. Local edits
    // must update its matrices without replacing its topology or storage.
    let (links, globals, order_storage, topology, order) = {
        let scratch = world.resource::<HierarchyScratch>();
        (
            scratch.links.as_ptr(),
            scratch.globals.as_ptr(),
            scratch.order.as_ptr(),
            scratch
                .links
                .iter()
                .zip(&scratch.sources)
                .map(|(link, source)| (source.entity, link.parent()))
                .collect::<Vec<_>>(),
            scratch.order.clone(),
        )
    };
    for translation in [Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, 3.0, 1.0)] {
        world.clear_trackers();
        // Negating a quaternion changes its stored pose bits but not its matrix.
        // The independently dirty leaf still needs evaluation beneath it.
        let parent_rotation = world.get::<Transform>(parent).unwrap().rotation;
        edit_transform(&mut world, parent, |transform| {
            transform.rotation = -parent_rotation;
        })
        .unwrap();
        edit_transform(&mut world, leaf, |transform| {
            transform.translation = translation;
        })
        .unwrap();
        assert_eq!(propagate(&mut world).invalid_transforms, 0);
        assert_eq!(changed_globals(&mut world), vec![leaf]);
        assert_eq!(
            matrix(&world, leaf).w_axis.truncate(),
            Vec3::X + translation
        );
        for &(entity, before) in &unchanged {
            assert_eq!(matrix(&world, entity), before);
        }
        let scratch = world.resource::<HierarchyScratch>();
        assert_eq!(scratch.links.as_ptr(), links);
        assert_eq!(scratch.globals.as_ptr(), globals);
        assert_eq!(scratch.order.as_ptr(), order_storage);
        assert_eq!(scratch.order, order);
        for (index, &(entity, parent_index)) in topology.iter().enumerate() {
            assert_eq!(scratch.indices[&entity], index);
            assert_eq!(scratch.sources[index].entity, entity);
            assert_eq!(scratch.links[index].parent(), parent_index);
            assert_eq!(scratch.global(index), Some(matrix(&world, entity)));
        }
    }
}

#[test]
fn moving_an_ancestor_updates_its_descendants_without_rewriting_a_sibling_branch() {
    #[derive(Resource)]
    struct RecordedGlobals(Vec<Entity>);

    let mut engine = Engine::new(EngineConfig {
        fixed_hz: 8.0,
        ..Default::default()
    });
    let world = engine.app.world_mut();
    // ECS iteration order differs from dependency order.
    let grandchild = spawn(world, Vec3::Z);
    let child = spawn(world, Vec3::Y);
    let ancestor = spawn(world, Vec3::X);
    let sibling = spawn(world, Vec3::new(-4.0, 0.0, 0.0));
    let root = spawn(world, Vec3::splat(2.0));
    world.entity_mut(ancestor).insert(InterpolateTransform);
    set_parents(
        world,
        &[
            (grandchild, Some(child)),
            (child, Some(ancestor)),
            (ancestor, Some(root)),
            (sibling, Some(root)),
        ],
    )
    .unwrap();
    propagate(world);
    let root_before = matrix(world, root);
    let sibling_before = matrix(world, sibling);
    world.clear_trackers();
    let replacement = Transform {
        translation: Vec3::new(5.0, 0.0, -2.0),
        rotation: Quat::from_rotation_z(0.5),
        scale: Vec3::new(2.0, 1.0, 3.0),
    };
    engine.app.add_systems(
        FixedSimulation,
        move |mut transforms: Query<&mut Transform>| {
            *transforms.get_mut(ancestor).unwrap() = replacement;
        },
    );
    // Record change notifications before Engine ends the frame and clears them.
    engine.app.add_systems(Presentation, |world: &mut World| {
        let changed = changed_globals(world);
        world.insert_resource(RecordedGlobals(changed));
    });
    // A new observer initially sees component creation. Prime its change tick
    // without advancing gameplay before measuring the moving branch.
    assert_eq!(engine.advance(0.0).fixed_steps, 0);
    assert_eq!(engine.advance(0.125).fixed_steps, 1);
    let world = engine.app.world();
    let mut expected_changes = vec![ancestor, child, grandchild];
    expected_changes.sort_unstable();
    assert_eq!(world.resource::<RecordedGlobals>().0, expected_changes);
    let expected = root_before * replacement.matrix();
    assert!(matrix(world, ancestor).abs_diff_eq(expected, 1.0e-6));
    assert!(matrix(world, child).abs_diff_eq(expected * Mat4::from_translation(Vec3::Y), 1.0e-6));
    assert!(matrix(world, grandchild).abs_diff_eq(
        expected * Mat4::from_translation(Vec3::Y) * Mat4::from_translation(Vec3::Z),
        1.0e-6
    ));
    assert_eq!(matrix(world, root), root_before);
    assert_eq!(matrix(world, sibling), sibling_before);
    let authoritative = [ancestor, child, grandchild].map(|entity| (entity, matrix(world, entity)));

    engine
        .extract_with_presentation(
            1.0,
            &mut RenderSnapshot::default(),
            PresentationMode::Interpolated { alpha: 0.5 },
        )
        .unwrap();
    let world = engine.app.world_mut();
    assert!(changed_globals(world).is_empty());
    let halfway = root_before
        * Transform {
            translation: Vec3::X.lerp(replacement.translation, 0.5),
            rotation: Quat::IDENTITY.slerp(replacement.rotation, 0.5),
            scale: Vec3::ONE.lerp(replacement.scale, 0.5),
        }
        .matrix();
    for (entity, local) in [
        (ancestor, Mat4::IDENTITY),
        (child, Mat4::from_translation(Vec3::Y)),
        (grandchild, Mat4::from_translation(Vec3::Y + Vec3::Z)),
    ] {
        assert!(
            world
                .resource::<PresentedTransforms>()
                .get(entity)
                .unwrap()
                .matrix()
                .unwrap()
                .abs_diff_eq(halfway * local, 1.0e-6)
        );
    }
    for (entity, before) in authoritative {
        assert_eq!(matrix(world, entity), before);
    }
}

#[test]
fn restoring_a_removed_ancestor_transform_repairs_the_cached_descendant_branch() {
    let mut world = World::new();
    let parent = spawn(&mut world, Vec3::X);
    let child = spawn(&mut world, Vec3::Y);
    let grandchild = spawn(&mut world, Vec3::Z);
    let unrelated = spawn(&mut world, Vec3::splat(7.0));
    set_parents(
        &mut world,
        &[(child, Some(parent)), (grandchild, Some(child))],
    )
    .unwrap();
    propagate(&mut world);
    let unrelated_before = matrix(&world, unrelated);
    world.entity_mut(parent).remove::<Transform>();
    assert_eq!(propagate(&mut world).invalid_transforms, 2);
    for entity in [parent, child, grandchild] {
        assert!(
            world
                .resource::<GlobalTransforms>()
                .get(entity)
                .and_then(|pose| pose.matrix())
                .is_none()
        );
    }
    world.clear_trackers();
    world.entity_mut(parent).insert(Transform {
        translation: Vec3::new(10.0, 0.0, 0.0),
        ..Transform::default()
    });
    assert_eq!(propagate(&mut world).invalid_transforms, 0);
    assert_eq!(
        matrix(&world, grandchild).w_axis.truncate(),
        Vec3::new(10.0, 1.0, 1.0)
    );
    assert_eq!(matrix(&world, unrelated), unrelated_before);
    let mut expected_changes = vec![parent, child, grandchild];
    expected_changes.sort_unstable();
    // Structural reconstruction may reissue unchanged output revisions.
    for entity in expected_changes {
        assert!(
            world
                .resource::<GlobalTransforms>()
                .get(entity)
                .unwrap()
                .matrix()
                .is_some()
        );
    }
}

#[test]
fn invalid_ancestor_pose_and_repair_propagate_without_relationship_changes() {
    let mut world = World::new();
    let parent = spawn(&mut world, Vec3::X);
    let child = spawn(&mut world, Vec3::Y);
    let grandchild = spawn(&mut world, Vec3::Z);
    let unrelated = spawn(&mut world, Vec3::splat(7.0));
    set_parents(
        &mut world,
        &[(child, Some(parent)), (grandchild, Some(child))],
    )
    .unwrap();
    propagate(&mut world);
    for scale in [Vec3::ZERO, Vec3::ONE] {
        world.clear_trackers();
        edit_transform(&mut world, parent, |transform| {
            transform.scale = scale;
        })
        .unwrap();
        let invalid = scale == Vec3::ZERO;
        assert_eq!(
            propagate(&mut world).invalid_transforms,
            if invalid { 3 } else { 0 }
        );
        for entity in [parent, child, grandchild] {
            assert_eq!(
                world
                    .resource::<GlobalTransforms>()
                    .get(entity)
                    .unwrap()
                    .matrix()
                    .is_none(),
                invalid
            );
        }
        assert_eq!(
            matrix(&world, unrelated).w_axis.truncate(),
            Vec3::splat(7.0)
        );
        let mut expected_changes = vec![parent, child, grandchild];
        expected_changes.sort_unstable();
        assert_eq!(changed_globals(&mut world), expected_changes);
        if invalid {
            world.clear_trackers();
            // Both poses change while the ancestor remains singular. Its
            // unchanged None result must not discard the descendant's edit.
            edit_transform(&mut world, parent, |transform| {
                transform.translation = Vec3::X * 2.0;
            })
            .unwrap();
            edit_transform(&mut world, grandchild, |transform| {
                transform.translation = Vec3::Z * 2.0;
            })
            .unwrap();
            assert_eq!(propagate(&mut world).invalid_transforms, 3);
            assert!(changed_globals(&mut world).is_empty());
        }
    }
    assert_eq!(
        matrix(&world, grandchild).w_axis.truncate(),
        Vec3::new(2.0, 1.0, 2.0)
    );
}

#[test]
fn repeated_invalid_nan_inputs_do_not_rewrite_the_world_at_every_boundary() {
    let mut world = World::new();
    let healthy = spawn(&mut world, Vec3::Y);
    let invalid = world
        .spawn(Transform {
            translation: Vec3::new(f32::from_bits(0x7fc0_0042), 0.0, 0.0),
            ..Transform::default()
        })
        .id();
    assert_eq!(propagate(&mut world).invalid_transforms, 1);
    world.clear_trackers();
    for _ in 0..4 {
        assert_eq!(propagate(&mut world).invalid_transforms, 1);
        assert!(
            world
                .resource::<GlobalTransforms>()
                .get(invalid)
                .unwrap()
                .matrix()
                .is_none()
        );
        assert_eq!(matrix(&world, healthy).w_axis.truncate(), Vec3::Y);
    }
    assert_eq!(changed_globals(&mut world).len(), 0);
    // A repair must still invalidate the cached failed result immediately.
    edit_transform(&mut world, invalid, |transform| {
        transform.translation = Vec3::X;
    })
    .unwrap();
    assert_eq!(propagate(&mut world).invalid_transforms, 0);
    assert_eq!(matrix(&world, invalid).w_axis.truncate(), Vec3::X);
}

#[test]
fn normal_conditioning_rejects_extreme_anisotropy_but_admits_extreme_uniform_composition() {
    for scale in [1.0e-4, 1.0e4] {
        let mut world = World::new();
        let nodes: Vec<_> = (0..3)
            .map(|_| {
                world
                    .spawn(Transform {
                        scale: Vec3::splat(scale),
                        rotation: Quat::from_rotation_y(0.3),
                        ..Transform::default()
                    })
                    .id()
            })
            .collect();
        set_parents(
            &mut world,
            &[(nodes[1], Some(nodes[0])), (nodes[2], Some(nodes[1]))],
        )
        .unwrap();
        assert_eq!(propagate(&mut world).invalid_transforms, 0);
        assert!(valid_global_matrix(matrix(&world, nodes[2])));
    }
    assert!(valid_global_matrix(Mat4::from_scale(Vec3::new(
        1.0, 1.0, 9_000.0
    ))));
    assert!(!valid_global_matrix(Mat4::from_scale(Vec3::new(
        0.0001, 10_000.0, 1.0
    ))));
    let almost_singular = Mat4::from_cols(
        Vec3::X.extend(0.0),
        Vec3::new(1.0, 1.0e-8, 0.0).extend(0.0),
        Vec3::Z.extend(0.0),
        Vec3::ZERO.extend(1.0),
    );
    assert!(almost_singular.inverse().is_finite());
    assert!(!valid_global_matrix(almost_singular));
}

#[test]
fn transform_budget_invalidates_overflow_and_recovers_after_removal() {
    const LIMIT: usize = 64;
    let mut world = World::new();
    world.insert_resource(TransformNodeLimit(LIMIT));
    for _ in 0..LIMIT + 1 {
        world.spawn(Transform::default());
    }
    assert_eq!(
        set_parents(&mut world, &[]),
        Err(HierarchyError::CapacityExceeded)
    );
    let report = propagate(&mut world);
    assert_eq!(report.transforms, LIMIT + 1);
    assert_eq!(report.invalid_transforms, 1);
    assert_eq!(report.overflow_transforms, 1);
    let removed = world
        .query_filtered::<Entity, With<Transform>>()
        .iter(&world)
        .next()
        .unwrap();
    world.despawn(removed);
    let report = propagate(&mut world);
    assert_eq!(report.invalid_transforms, 0);
    assert_eq!(report.transforms, LIMIT);
    assert_eq!(world.resource::<HierarchyScratch>().links.len(), LIMIT);
}

#[test]
fn deferred_leaves_read_eager_bits_validity_and_new_revisions() {
    let mut world = World::new();
    let parent_local = Transform {
        translation: Vec3::new(3.0, 1.0, -2.0),
        rotation: Quat::from_rotation_y(0.6),
        scale: Vec3::new(1.2, 0.9, 0.8),
    };
    let parent = world.spawn(parent_local).id();
    // A certified tall box, a box too flat for the certificate (the eager
    // gate or fallback decides), a non-unit rotation (invalid) and a scale
    // beyond the coefficient bound, plus a root leaf.
    let mut leaves = [
        Transform {
            translation: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_rotation_z(0.3),
            scale: Vec3::new(0.7, 8.6, 0.7),
        },
        Transform {
            rotation: Quat::from_rotation_x(1.0),
            scale: Vec3::new(1.0, 1.0, 1.0e-3),
            ..Transform::default()
        },
        Transform {
            rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 1.1),
            ..Transform::default()
        },
        Transform {
            scale: Vec3::splat(1.0e7),
            ..Transform::default()
        },
        Transform {
            translation: Vec3::new(-4.0, 0.5, 2.0),
            rotation: Quat::from_rotation_x(0.2),
            scale: Vec3::new(2.0, 1.0, 0.5),
        },
    ];
    let entities: Vec<_> = leaves
        .iter()
        .map(|&local| world.spawn(local).id())
        .collect();
    let changes: Vec<_> = entities[..4]
        .iter()
        .map(|&child| (child, Some(parent)))
        .collect();
    set_parents(&mut world, &changes).unwrap();
    let bits = |matrix: Mat4| matrix.to_cols_array().map(f32::to_bits);
    let mut previous_revisions = Vec::new();
    for round in 0..3 {
        if round > 0 {
            // Moved leaves take the scan's fused path; the first round
            // evaluated them after the topology rebuild.
            for (entity, local) in entities.iter().zip(&mut leaves) {
                local.translation += Vec3::new(0.25, -0.5, 0.125);
                *world.get_mut::<Transform>(*entity).unwrap() = *local;
            }
        }
        propagate_transforms(&mut world);
        let store = world.resource::<GlobalTransforms>();
        let mut revisions = Vec::new();
        for (index, (&entity, local)) in entities.iter().zip(&leaves).enumerate() {
            let composed = if index < 4 {
                parent_local.matrix() * local.matrix()
            } else {
                local.matrix()
            };
            let near_unit = (local.rotation.length_squared() - 1.0).abs() <= 1.0e-3;
            let expected = (near_unit && valid_global_matrix(composed)).then_some(composed);
            let (value, revision) = store
                .get_indexed_with_revision(entity, &mut TransformIndex::default())
                .unwrap();
            assert_eq!(value.matrix().map(bits), expected.map(bits), "leaf {index}");
            revisions.push(revision);
        }
        let state = |entity: Entity| store.links[store.indices[&entity]].global;
        assert!(state(entities[0]) == Global::Deferred);
        assert!(state(entities[4]) == Global::Deferred);
        assert!(state(entities[2]) == Global::Invalid);
        assert!(state(entities[3]) != Global::Deferred);
        assert!(state(parent) == Global::Stored);
        // Every changed deferred leaf publishes a new revision.
        if round > 0 {
            assert_ne!(revisions[0], previous_revisions[0]);
            assert_ne!(revisions[4], previous_revisions[4]);
        }
        previous_revisions = revisions;
    }
}
