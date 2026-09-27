use super::*;
use crate::prelude::*;

fn engine() -> Engine {
    Engine::new(EngineConfig {
        fixed_hz: 8.0,
        frustum_culling: false,
        ..Default::default()
    })
}

fn present(engine: &mut Engine, alpha: f32) -> RenderSnapshot {
    let mut snapshot = RenderSnapshot::default();
    engine
        .extract_with_presentation(1.0, &mut snapshot, PresentationMode::Interpolated { alpha })
        .unwrap();
    snapshot
}

fn displayed(engine: &Engine, entity: Entity) -> Mat4 {
    engine
        .app
        .world()
        .resource::<PresentedTransforms>()
        .get(entity)
        .unwrap()
        .matrix()
        .unwrap()
}

fn current(engine: &Engine, entity: Entity) -> Mat4 {
    engine
        .app
        .world()
        .resource::<GlobalTransforms>()
        .get(entity)
        .unwrap()
        .matrix()
        .unwrap()
}

fn assert_matrix(actual: Mat4, expected: Mat4) {
    assert!(
        actual.abs_diff_eq(expected, 1.0e-5),
        "actual={actual:?}, expected={expected:?}"
    );
}

fn move_x(
    transforms: Query<(Entity, &Transform), With<InterpolateTransform>>,
    mut edits: TransformEdits,
) {
    for (entity, transform) in &transforms {
        edits
            .edit(entity, transform, |pose| pose.translation.x += 1.0)
            .unwrap();
    }
}

#[test]
fn static_current_presentation_reuses_revisions_and_retains_last_prepared_pose() {
    let mut engine = engine();
    let parent = engine.app.world_mut().spawn(Transform::default()).id();
    let child = engine
        .app
        .world_mut()
        .spawn(Transform {
            translation: Vec3::X,
            ..Default::default()
        })
        .id();
    set_parent(engine.app.world_mut(), child, Some(parent)).unwrap();
    let mut snapshot = RenderSnapshot::default();
    assert!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get(child)
            .is_none()
    );
    engine.extract(1.0, &mut snapshot);
    let original = engine
        .app
        .world()
        .resource::<PresentedTransforms>()
        .get(child)
        .unwrap();
    engine.app.world_mut().clear_trackers();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get(child)
            .unwrap()
            .revision(),
        original.revision()
    );
    engine
        .app
        .world_mut()
        .get_mut::<Transform>(parent)
        .unwrap()
        .translation = Vec3::Y;
    engine.refresh_transforms();
    assert_eq!(current(&engine, child).w_axis.truncate(), Vec3::X + Vec3::Y);
    assert_eq!(displayed(&engine, child), original.matrix().unwrap());
    engine.extract(1.0, &mut snapshot);
    assert_eq!(
        displayed(&engine, child).w_axis.truncate(),
        Vec3::X + Vec3::Y
    );
    assert_ne!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get(child)
            .unwrap()
            .revision(),
        original.revision()
    );
}

#[test]
fn stable_history_updates_in_place_and_preserves_the_fixed_pair() {
    let mut engine = engine();
    let entity = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    present(&mut engine, 0.5);
    let allocation = engine
        .app
        .world()
        .resource::<InterpolationState>()
        .history
        .as_ptr();
    begin_tick(engine.app.world_mut());
    assert_eq!(
        engine
            .app
            .world()
            .resource::<InterpolationState>()
            .history
            .as_ptr(),
        allocation
    );
    edit_transform(engine.app.world_mut(), entity, |transform| {
        transform.translation.x = 8.0;
    })
    .unwrap();
    engine.refresh_transforms();
    end_tick(engine.app.world_mut());
    assert_eq!(
        engine
            .app
            .world()
            .resource::<InterpolationState>()
            .history
            .as_ptr(),
        allocation
    );
    for (alpha, expected) in [(0.0, 0.0), (0.25, 2.0), (0.75, 6.0), (1.0, 8.0)] {
        present(&mut engine, alpha);
        assert_eq!(displayed(&engine, entity).w_axis.x, expected);
        assert_eq!(current(&engine, entity).w_axis.x, 8.0);
    }
}

#[test]
fn blended_ancestor_owns_descendant_work_even_when_spawned_after_it() {
    let mut engine = engine();
    let child = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let parent = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let leaf = engine
        .app
        .world_mut()
        .spawn(Transform {
            translation: Vec3::Z,
            ..Default::default()
        })
        .id();
    let unrelated = engine
        .app
        .world_mut()
        .spawn(Transform {
            translation: Vec3::X * 20.0,
            ..Default::default()
        })
        .id();
    set_parents(
        engine.app.world_mut(),
        &[(child, Some(parent)), (leaf, Some(child))],
    )
    .unwrap();
    present(&mut engine, 0.5);
    let unrelated_revision = engine
        .app
        .world()
        .resource::<PresentedTransforms>()
        .get(unrelated)
        .unwrap()
        .revision();
    begin_tick(engine.app.world_mut());
    edit_transform(engine.app.world_mut(), parent, |transform| {
        transform.translation = Vec3::X * 4.0;
    })
    .unwrap();
    edit_transform(engine.app.world_mut(), child, |transform| {
        transform.translation = Vec3::Y * 6.0;
    })
    .unwrap();
    engine.refresh_transforms();
    end_tick(engine.app.world_mut());
    engine.app.world_mut().clear_trackers();
    present(&mut engine, 0.5);
    assert_eq!(
        displayed(&engine, leaf).w_axis.truncate(),
        Vec3::new(2.0, 3.0, 1.0)
    );
    assert_eq!(
        current(&engine, leaf).w_axis.truncate(),
        Vec3::new(4.0, 6.0, 1.0)
    );
    assert_eq!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get(unrelated)
            .unwrap()
            .revision(),
        unrelated_revision
    );
    reset_interpolation(engine.app.world_mut(), child);
    present(&mut engine, 0.5);
    // Snapping the child's local pair still inherits its displayed parent.
    assert_eq!(
        displayed(&engine, leaf).w_axis.truncate(),
        Vec3::new(2.0, 6.0, 1.0)
    );
}

#[test]
fn dense_and_sparse_presentation_agree_through_resets_and_unmarked_descendants() {
    let mut engine = engine();
    let parent = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let child = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let leaf = engine
        .app
        .world_mut()
        .spawn(Transform {
            translation: Vec3::Z,
            ..Default::default()
        })
        .id();
    let snapped = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let bridge = engine.app.world_mut().spawn(Transform::default()).id();
    let tip = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    set_parents(
        engine.app.world_mut(),
        &[
            (child, Some(parent)),
            (leaf, Some(child)),
            (bridge, Some(snapped)),
            (tip, Some(bridge)),
        ],
    )
    .unwrap();
    present(&mut engine, 0.5);
    begin_tick(engine.app.world_mut());
    for (entity, translation) in [
        (parent, Vec3::X * 4.0),
        (child, Vec3::Y * 6.0),
        (tip, Vec3::Y * 8.0),
    ] {
        edit_transform(engine.app.world_mut(), entity, |pose| {
            pose.translation = translation;
        })
        .unwrap();
    }
    engine.refresh_transforms();
    end_tick(engine.app.world_mut());
    reset_interpolation(engine.app.world_mut(), child);
    // Observation snaps this marker, but the tip still owns a valid blend pair.
    // Its local blend must be suppressed across the unmarked bridge.
    edit_transform(engine.app.world_mut(), snapped, |pose| {
        pose.translation = Vec3::X * 20.0;
    })
    .unwrap();
    let entities = [parent, child, leaf, snapped, bridge, tip];
    let alphas = [0.0, 0.25, 0.75];
    let dense: Vec<_> = alphas
        .map(|alpha| {
            present(&mut engine, alpha);
            entities.map(|entity| displayed(&engine, entity))
        })
        .into();
    assert_eq!(dense[1][2].w_axis.truncate(), Vec3::new(1.0, 6.0, 1.0));
    assert_eq!(dense[1][5].w_axis.truncate(), Vec3::new(20.0, 8.0, 0.0));
    assert!(
        engine
            .app
            .world()
            .resource::<InterpolationState>()
            .blended_locals
            .is_empty()
    );

    // Unrelated static nodes select sparse traversal without altering this
    // branch's interval. Compare exact matrices, not approximate poses.
    let padding: Vec<_> = (0..64)
        .map(|_| engine.app.world_mut().spawn(Transform::default()).id())
        .collect();
    for (alpha, expected) in alphas.into_iter().zip(&dense) {
        present(&mut engine, alpha);
        assert_eq!(&entities.map(|entity| displayed(&engine, entity)), expected);
        assert!(
            !engine
                .app
                .world()
                .resource::<InterpolationState>()
                .blended_locals
                .is_empty()
        );
    }
    // Returning to dense must replace sparse flags and revalidate source hints
    // after the admission/order change, including the inactive snap bridge.
    for entity in padding {
        engine.app.world_mut().despawn(entity);
    }
    for (alpha, expected) in alphas.into_iter().zip(dense) {
        present(&mut engine, alpha);
        assert_eq!(entities.map(|entity| displayed(&engine, entity)), expected);
        let state = engine.app.world().resource::<InterpolationState>();
        assert!(state.dense_display);
        assert!(state.touched_display.is_empty());
        assert!(state.blended_locals.is_empty());
    }
}

#[test]
fn cached_history_indices_revalidate_entities_after_topology_reordering() {
    #[derive(Component)]
    struct Reordered;
    let mut engine = engine();
    let a = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let b = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    engine.refresh_transforms();
    begin_tick(engine.app.world_mut());
    for (entity, x) in [(a, 4.0), (b, 40.0)] {
        edit_transform(engine.app.world_mut(), entity, |pose| {
            pose.translation.x = x
        })
        .unwrap();
    }
    engine.refresh_transforms();
    end_tick(engine.app.world_mut());
    present(&mut engine, 0.25);
    let expected = [displayed(&engine, a), displayed(&engine, b)];
    let previous_index = engine.app.world().resource::<HierarchyScratch>().indices[&a];
    engine.app.world_mut().entity_mut(a).insert(Reordered);
    // A structural edit rebuilds topology after the unrelated archetype move.
    engine.app.world_mut().spawn(Transform::default());
    engine.refresh_transforms();
    assert_ne!(
        engine.app.world().resource::<HierarchyScratch>().indices[&a],
        previous_index
    );
    assert_eq!([displayed(&engine, a), displayed(&engine, b)], expected);
    present(&mut engine, 0.25);
    assert_eq!([displayed(&engine, a), displayed(&engine, b)], expected);
    let world = engine.app.world();
    let hierarchy = world.resource::<HierarchyScratch>();
    let state = world.resource::<InterpolationState>();
    // Every record and its node form the mutually validated hint pair.
    for (record, history) in state.history.iter().enumerate() {
        let node = state.history_nodes[record] as usize;
        assert_eq!(hierarchy.sources[node].entity, history.entity);
        assert_eq!(hierarchy.history_hints[node] as usize, record);
    }
}

#[test]
fn history_reordering_keeps_an_already_captured_prefix_and_new_entities_snap() {
    #[derive(Component)]
    struct Reordered;
    let mut engine = engine();
    let a = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let b = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let c = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    present(&mut engine, 0.5);
    begin_tick(engine.app.world_mut());
    for (entity, x) in [(a, 10.0), (b, 20.0), (c, 30.0)] {
        edit_transform(engine.app.world_mut(), entity, |transform| {
            transform.translation.x = x;
        })
        .unwrap();
    }
    // Move the middle entity to a new archetype: the unchanged first row must
    // consume End only once while the remaining records change iteration order.
    engine.app.world_mut().entity_mut(b).insert(Reordered);
    engine.refresh_transforms();
    end_tick(engine.app.world_mut());
    present(&mut engine, 0.5);
    for (entity, x) in [(a, 5.0), (b, 10.0), (c, 15.0)] {
        assert_eq!(displayed(&engine, entity).w_axis.x, x);
    }
    engine.app.world_mut().despawn(b);
    // Production refreshes authoritative state before its O(1) Begin boundary.
    engine.refresh_transforms();
    begin_tick(engine.app.world_mut());
    assert!(
        !engine
            .app
            .world()
            .resource::<InterpolationState>()
            .indices
            .contains_key(&b)
    );
    edit_transform(engine.app.world_mut(), a, |transform| {
        transform.translation.x = 12.0;
    })
    .unwrap();
    edit_transform(engine.app.world_mut(), c, |transform| {
        transform.translation.x = 34.0;
    })
    .unwrap();
    let added = engine
        .app
        .world_mut()
        .spawn((
            Transform {
                translation: Vec3::X * 99.0,
                ..Default::default()
            },
            InterpolateTransform,
        ))
        .id();
    engine.refresh_transforms();
    end_tick(engine.app.world_mut());
    present(&mut engine, 0.5);
    for (entity, x) in [(a, 11.0), (c, 32.0), (added, 99.0)] {
        assert_eq!(displayed(&engine, entity).w_axis.x, x);
    }
}

#[test]
fn lazy_intervals_freeze_once_and_expire_after_an_unchanged_tick() {
    let mut engine = engine();
    let entity = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    engine.refresh_transforms();
    begin_tick(engine.app.world_mut());
    for x in [2.0, 8.0] {
        edit_transform(engine.app.world_mut(), entity, |pose| {
            pose.translation.x = x
        })
        .unwrap();
        engine.refresh_transforms();
    }
    end_tick(engine.app.world_mut());
    present(&mut engine, 0.25);
    assert_eq!(displayed(&engine, entity).w_axis.x, 2.0);

    begin_tick(engine.app.world_mut());
    end_tick(engine.app.world_mut());
    present(&mut engine, 0.25);
    assert_eq!(displayed(&engine, entity).w_axis.x, 8.0);

    begin_tick(engine.app.world_mut());
    edit_transform(engine.app.world_mut(), entity, |pose| {
        pose.translation.x = 12.0
    })
    .unwrap();
    engine.refresh_transforms();
    assert_eq!(reset_interpolation(engine.app.world_mut(), entity), 1);
    edit_transform(engine.app.world_mut(), entity, |pose| {
        pose.translation.x = 16.0
    })
    .unwrap();
    engine.refresh_transforms();
    end_tick(engine.app.world_mut());
    present(&mut engine, 0.25);
    assert_eq!(displayed(&engine, entity).w_axis.x, 16.0);

    begin_tick(engine.app.world_mut());
    edit_transform(engine.app.world_mut(), entity, |pose| {
        pose.translation.x = 20.0
    })
    .unwrap();
    engine.refresh_transforms();
    end_tick(engine.app.world_mut());
    present(&mut engine, 0.25);
    assert_eq!(displayed(&engine, entity).w_axis.x, 17.0);
}

#[test]
fn shared_current_pose_preserves_the_previous_endpoint_across_a_mid_tick_rebuild() {
    let mut engine = engine();
    let entity = engine
        .app
        .world_mut()
        .spawn((
            Transform {
                translation: Vec3::X * 3.0,
                ..Default::default()
            },
            InterpolateTransform,
        ))
        .id();
    engine.refresh_transforms();
    begin_tick(engine.app.world_mut());
    edit_transform(engine.app.world_mut(), entity, |pose| {
        pose.translation.x = 7.0
    })
    .unwrap();
    // Admission forces reconciliation against the old authoritative source
    // cache before it is replaced. A later refresh must not freeze 7 as Begin.
    engine.app.world_mut().spawn(Transform::default());
    engine.refresh_transforms();
    edit_transform(engine.app.world_mut(), entity, |pose| {
        pose.translation.x = 11.0
    })
    .unwrap();
    engine.refresh_transforms();
    end_tick(engine.app.world_mut());
    present(&mut engine, 0.25);
    assert_eq!(displayed(&engine, entity).w_axis.x, 5.0);
    assert_eq!(current(&engine, entity).w_axis.x, 11.0);
}

#[test]
fn lazy_validity_uses_fixed_endpoints_and_does_not_resurrect_an_invalid_interval() {
    let mut engine = engine();
    let parent = engine.app.world_mut().spawn(Transform::default()).id();
    let entity = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    set_parent(engine.app.world_mut(), entity, Some(parent)).unwrap();
    engine.refresh_transforms();
    begin_tick(engine.app.world_mut());
    edit_transform(engine.app.world_mut(), entity, |pose| {
        pose.translation.x = 8.0;
    })
    .unwrap();
    edit_transform(engine.app.world_mut(), parent, |pose| pose.scale.x = 0.0).unwrap();
    engine.refresh_transforms();
    edit_transform(engine.app.world_mut(), parent, |pose| pose.scale.x = 1.0).unwrap();
    engine.refresh_transforms();
    end_tick(engine.app.world_mut());
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, entity).w_axis.x, 4.0);

    begin_tick(engine.app.world_mut());
    edit_transform(engine.app.world_mut(), entity, |pose| {
        pose.translation.x = 12.0;
    })
    .unwrap();
    edit_transform(engine.app.world_mut(), parent, |pose| pose.scale.x = 0.0).unwrap();
    engine.refresh_transforms();
    end_tick(engine.app.world_mut());
    edit_transform(engine.app.world_mut(), parent, |pose| pose.scale.x = 1.0).unwrap();
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, entity).w_axis.x, 12.0);

    begin_tick(engine.app.world_mut());
    edit_transform(engine.app.world_mut(), entity, |pose| {
        pose.translation.x = 16.0
    })
    .unwrap();
    engine.refresh_transforms();
    end_tick(engine.app.world_mut());
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, entity).w_axis.x, 14.0);

    // No source event on the marked child during either invalid tick. A later
    // inherited repair must still start from current, not resurrect 12 -> 16.
    begin_tick(engine.app.world_mut());
    edit_transform(engine.app.world_mut(), parent, |pose| pose.scale.x = 0.0).unwrap();
    engine.refresh_transforms();
    end_tick(engine.app.world_mut());
    begin_tick(engine.app.world_mut());
    end_tick(engine.app.world_mut());
    edit_transform(engine.app.world_mut(), parent, |pose| pose.scale.x = 1.0).unwrap();
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, entity).w_axis.x, 16.0);
}

#[test]
fn inactive_scratch_matrices_cannot_leak_after_a_branch_reset() {
    let mut engine = engine();
    let stopped = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let moving = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    engine.app.add_systems(FixedSimulation, move_x);
    engine.advance(0.125);
    present(&mut engine, 0.25);
    assert_eq!(displayed(&engine, stopped).w_axis.x, 0.25);
    reset_interpolation(engine.app.world_mut(), stopped);
    present(&mut engine, 0.75);
    assert_eq!(displayed(&engine, stopped).w_axis.x, 1.0);
    assert_eq!(displayed(&engine, moving).w_axis.x, 0.75);
    edit_transform(engine.app.world_mut(), stopped, |transform| {
        transform.translation.x = 9.0;
    })
    .unwrap();
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, stopped).w_axis.x, 9.0);
    assert_eq!(displayed(&engine, moving).w_axis.x, 0.5);
}

#[test]
fn presented_identity_survives_archetype_moves_and_entity_generation_reuse() {
    #[derive(Component)]
    struct Extra;
    let mut engine = engine();
    let moving = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let stationary = engine
        .app
        .world_mut()
        .spawn(Transform {
            translation: Vec3::Y * 9.0,
            ..Default::default()
        })
        .id();
    engine.app.add_systems(FixedSimulation, move_x);
    engine.advance(0.125);
    present(&mut engine, 0.5);
    let mut hint = TransformIndex::default();
    assert_eq!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get_indexed(moving, &mut hint)
            .unwrap()
            .matrix()
            .unwrap()
            .w_axis
            .x,
        0.5
    );
    engine.app.world_mut().entity_mut(moving).insert(Extra);
    let replacement = engine.app.world_mut().despawn_no_free(stationary).unwrap();
    engine
        .app
        .world_mut()
        .spawn_at(
            replacement,
            Transform {
                translation: Vec3::Z * 3.0,
                ..Default::default()
            },
        )
        .unwrap();
    engine.refresh_transforms();
    // Until extraction, published identity and values still describe the old frame.
    assert_eq!(displayed(&engine, stationary).w_axis.y, 9.0);
    assert!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get(replacement)
            .is_none()
    );
    present(&mut engine, 0.25);
    assert_eq!(displayed(&engine, moving).w_axis.x, 0.25);
    assert!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get_indexed(stationary, &mut hint)
            .is_none()
    );
    assert_eq!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get_indexed(replacement, &mut hint)
            .unwrap()
            .matrix()
            .unwrap()
            .w_axis
            .z,
        3.0
    );
    assert_eq!(current(&engine, moving).w_axis.x, 1.0);
    engine.extract(1.0, &mut RenderSnapshot::default());
    assert_eq!(displayed(&engine, moving).w_axis.x, 1.0);
    present(&mut engine, 0.75);
    assert_eq!(displayed(&engine, moving).w_axis.x, 0.75);
}

#[test]
fn first_tick_retains_initial_pose_and_exact_extract_preserves_history() {
    let mut engine = engine();
    let entity = engine
        .app
        .world_mut()
        .spawn((
            Transform::default(),
            InterpolateTransform,
            Cube { color: [1.0; 4] },
        ))
        .id();
    engine.app.add_systems(FixedSimulation, move_x);
    assert!(engine.presentation().frame().is_none());
    assert!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get(entity)
            .and_then(|pose| pose.matrix())
            .is_none()
    );
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, entity).w_axis.x, 0.0);
    engine.advance(0.125);
    for (alpha, expected) in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)] {
        let snapshot = present(&mut engine, alpha);
        assert_eq!(snapshot.instances[0].model.w_axis.x, expected);
        assert_eq!(current(&engine, entity).w_axis.x, 1.0);
        assert_eq!(
            engine
                .app
                .world()
                .get::<Transform>(entity)
                .unwrap()
                .translation
                .x,
            1.0
        );
    }
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.instances[0].model.w_axis.x, 1.0);
    assert_eq!(engine.presentation().mode(), PresentationMode::Current);
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, entity).w_axis.x, 0.5);
}

#[test]
fn zero_and_multiple_ticks_keep_only_latest_fixed_interval() {
    let mut engine = engine();
    let entity = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    engine.app.add_systems(FixedSimulation, move_x);
    let report = engine.advance(0.4375);
    assert_eq!(report.fixed_steps, 3);
    assert_eq!(report.interpolation, 0.5);
    present(&mut engine, report.interpolation);
    assert_eq!(displayed(&engine, entity).w_axis.x, 2.5);
    let report = engine.advance(0.03125);
    assert_eq!(report.fixed_steps, 0);
    assert_eq!(report.interpolation, 0.75);
    present(&mut engine, report.interpolation);
    assert_eq!(displayed(&engine, entity).w_axis.x, 2.75);
    assert_eq!(current(&engine, entity).w_axis.x, 3.0);
}

#[test]
fn shortest_rotation_arc_and_local_scale_preserve_hierarchical_shear() {
    let mut engine = engine();
    let initial = Transform {
        rotation: Quat::from_rotation_z(170_f32.to_radians()),
        scale: Vec3::new(2.0, 1.0, 1.0),
        ..Default::default()
    };
    let parent = engine
        .app
        .world_mut()
        .spawn((initial, InterpolateTransform))
        .id();
    let child_local = Transform {
        translation: Vec3::X,
        rotation: Quat::from_rotation_y(0.6),
        ..Default::default()
    };
    let child = engine.app.world_mut().spawn(child_local).id();
    set_parent(engine.app.world_mut(), child, Some(parent)).unwrap();
    engine
        .app
        .add_systems(FixedSimulation, move |world: &mut World| {
            edit_transform(world, parent, |transform| {
                transform.rotation = Quat::from_rotation_z(-170_f32.to_radians());
                transform.scale = Vec3::new(4.0, 1.0, 1.0);
            })
            .unwrap();
        });
    engine.advance(0.125);
    present(&mut engine, 0.5);
    let expected = Transform {
        rotation: Quat::from_rotation_z(std::f32::consts::PI),
        scale: Vec3::new(3.0, 1.0, 1.0),
        ..Default::default()
    }
    .matrix();
    assert_matrix(displayed(&engine, parent), expected);
    assert_matrix(displayed(&engine, child), expected * child_local.matrix());
    assert!(!displayed(&engine, child).abs_diff_eq(current(&engine, child), 0.01));
}

#[test]
fn small_angle_rotation_stays_within_the_reference_budget_and_preserves_large_arcs() {
    fn normalized(q: [f64; 4]) -> [f64; 4] {
        let length = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        q.map(|x| x / length)
    }
    fn reference(previous: Quat, current: Quat, alpha: f32) -> [f64; 4] {
        let previous = normalized(previous.to_array().map(f64::from));
        let mut current = normalized(current.to_array().map(f64::from));
        let mut dot: f64 = previous.iter().zip(current).map(|(a, b)| a * b).sum();
        if dot < 0.0 {
            current = current.map(|x| -x);
            dot = -dot;
        }
        let theta = dot.min(1.0).acos();
        if theta < 1.0e-12 {
            return previous;
        }
        let alpha = f64::from(alpha);
        let left = ((1.0 - alpha) * theta).sin() / theta.sin();
        let right = (alpha * theta).sin() / theta.sin();
        normalized(std::array::from_fn(|i| {
            previous[i] * left + current[i] * right
        }))
    }
    fn angular_error(actual: Quat, expected: [f64; 4]) -> f64 {
        let mut actual = normalized(actual.to_array().map(f64::from));
        if actual.iter().zip(expected).map(|(a, b)| a * b).sum::<f64>() < 0.0 {
            actual = actual.map(|x| -x);
        }
        let difference = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let sum = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a + b).powi(2))
            .sum::<f64>()
            .sqrt();
        4.0 * difference.atan2(sum)
    }

    let threshold_angle = 2.0 * LINEAR_ROTATION_DOT_THRESHOLD.acos();
    let bases = [
        Quat::IDENTITY,
        Quat::from_rotation_x(1.3),
        Quat::from_rotation_y(-2.4) * Quat::from_rotation_z(0.7),
        Quat::from_axis_angle(Vec3::new(1.0, -2.0, 3.0).normalize(), 3.0),
    ];
    let axes = [Vec3::X, Vec3::Y, Vec3::new(2.0, 3.0, -1.0).normalize()];
    let angles = [
        0.0,
        1.0e-5,
        0.02,
        threshold_angle - 2.0e-5,
        threshold_angle - 2.0e-7,
        threshold_angle,
        threshold_angle + 2.0e-7,
        threshold_angle + 2.0e-5,
        0.7,
    ];
    let mut small_cases = 0;
    let mut large_cases = 0;
    for base in bases {
        for axis in axes {
            for angle in angles {
                for (previous_length, current_length) in
                    [(1.0, 1.0), (0.9996, 1.0004), (1.0004, 0.9996)]
                {
                    for (previous_sign, current_sign) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)] {
                        let previous = base * (previous_length * previous_sign);
                        let current = (base * Quat::from_axis_angle(axis, angle))
                            * (current_length * current_sign);
                        assert!((previous.length_squared() - 1.0).abs() <= 1.0e-3);
                        assert!((current.length_squared() - 1.0).abs() <= 1.0e-3);
                        let small = previous.normalize().dot(current.normalize()).abs()
                            >= LINEAR_ROTATION_DOT_THRESHOLD;
                        for alpha in [
                            0.0,
                            0.001,
                            0.125,
                            0.211_334_5,
                            0.5,
                            0.788_665_5,
                            0.875,
                            0.999,
                            1.0,
                        ] {
                            let actual = blend_rotation(previous, current, alpha);
                            assert!(actual.is_finite());
                            if small {
                                small_cases += 1;
                                let error =
                                    angular_error(actual, reference(previous, current, alpha));
                                assert!(
                                    error <= 1.0e-5,
                                    "angular error {error} at angle {angle}, alpha {alpha}"
                                );
                            } else {
                                large_cases += 1;
                                let legacy = previous
                                    .normalize()
                                    .slerp(current.normalize(), alpha)
                                    .normalize();
                                assert_eq!(
                                    actual.to_array().map(f32::to_bits),
                                    legacy.to_array().map(f32::to_bits)
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(small_cases > 0 && large_cases > 0);
}

#[test]
fn small_angle_policy_keeps_exact_previous_and_current_endpoints() {
    let previous = Transform {
        rotation: Quat::from_rotation_x(1.1) * 1.0004,
        ..Default::default()
    };
    let next = Transform {
        rotation: (previous.rotation.normalize() * Quat::from_rotation_y(0.03)) * 0.9996,
        ..Default::default()
    };
    assert!(same_local_bits(&blend(&previous, &next, 0.0), &previous));
    let mut engine = engine();
    let entity = engine
        .app
        .world_mut()
        .spawn((previous, InterpolateTransform))
        .id();
    engine.refresh_transforms();
    begin_tick(engine.app.world_mut());
    edit_transform(engine.app.world_mut(), entity, |pose| *pose = next).unwrap();
    engine.refresh_transforms();
    end_tick(engine.app.world_mut());
    present(&mut engine, 0.0);
    assert_eq!(displayed(&engine, entity), previous.matrix());
    present(&mut engine, 1.0);
    assert_eq!(displayed(&engine, entity), current(&engine, entity));
}

#[test]
fn presentation_and_paused_writes_snap_the_affected_branch() {
    let mut engine = engine();
    let parent = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let child = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    set_parent(engine.app.world_mut(), child, Some(parent)).unwrap();
    engine.app.add_systems(FixedSimulation, move_x);
    engine
        .app
        .add_systems(Presentation, move |world: &mut World| {
            edit_transform(world, parent, |transform| {
                transform.translation.x = 20.0;
            })
            .unwrap();
        });
    engine.advance(0.125);
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, parent).w_axis.x, 20.0);
    assert_eq!(displayed(&engine, child).w_axis.x, 21.0);
    edit_transform(engine.app.world_mut(), child, |transform| {
        transform.translation.x = 30.0;
    })
    .unwrap();
    present(&mut engine, 0.0);
    assert_eq!(displayed(&engine, child).w_axis.x, 50.0);
}

#[test]
fn mid_tick_reset_suppresses_pending_interval_then_interpolation_resumes() {
    let mut engine = engine();
    let parent = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let child = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    set_parent(engine.app.world_mut(), child, Some(parent)).unwrap();
    engine
        .app
        .add_systems(FixedSimulation, move |world: &mut World| {
            for entity in [parent, child] {
                edit_transform(world, entity, |transform| {
                    transform.translation.x += 1.0;
                })
                .unwrap();
            }
            if world.resource::<FixedTime>().tick == 1 {
                edit_transform(world, parent, |transform| {
                    transform.translation.x = 100.0;
                })
                .unwrap();
                assert_eq!(reset_interpolation(world, parent), 2);
            }
        });
    engine.advance(0.125);
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, parent).w_axis.x, 100.0);
    assert_eq!(displayed(&engine, child).w_axis.x, 101.0);
    engine.advance(0.125);
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, parent).w_axis.x, 100.5);
    assert_eq!(displayed(&engine, child).w_axis.x, 102.0);
    engine.reset_clock();
    present(&mut engine, 0.0);
    assert_eq!(displayed(&engine, child).w_axis.x, 103.0);
}

#[test]
fn scale_sign_flip_snaps_all_affected_local_histories() {
    let mut engine = engine();
    let parent = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let child = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    set_parent(engine.app.world_mut(), child, Some(parent)).unwrap();
    engine
        .app
        .add_systems(FixedSimulation, move |world: &mut World| {
            edit_transform(world, parent, |transform| {
                transform.scale.x = -1.0;
            })
            .unwrap();
            edit_transform(world, child, |transform| {
                transform.translation.x = 2.0;
            })
            .unwrap();
        });
    engine.advance(0.125);
    for alpha in [0.0, 0.5, 1.0] {
        present(&mut engine, alpha);
        assert_matrix(displayed(&engine, parent), current(&engine, parent));
        assert_eq!(displayed(&engine, child).w_axis.x, -2.0);
    }
}

#[test]
fn reparenting_unmarked_ancestor_resets_tracked_descendants() {
    let mut engine = engine();
    let target = engine
        .app
        .world_mut()
        .spawn(Transform {
            translation: Vec3::X * 10.0,
            ..Default::default()
        })
        .id();
    let group = engine.app.world_mut().spawn(Transform::default()).id();
    let child = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    set_parent(engine.app.world_mut(), child, Some(group)).unwrap();
    engine
        .app
        .add_systems(FixedSimulation, move |world: &mut World| {
            edit_transform(world, child, |transform| {
                transform.translation.x = 4.0;
            })
            .unwrap();
            set_parent(world, group, Some(target)).unwrap();
        });
    engine.advance(0.125);
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, child).w_axis.x, 14.0);
    assert_eq!(
        engine
            .app
            .world()
            .get::<Transform>(child)
            .unwrap()
            .translation
            .x,
        4.0
    );
}

#[test]
fn marker_replacement_and_mid_tick_spawn_start_with_current_pose() {
    let mut engine = engine();
    let entity = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    engine.app.add_systems(
        FixedSimulation,
        (move_x, |mut commands: Commands| {
            commands.spawn((
                Transform {
                    translation: Vec3::X * 20.0,
                    ..Default::default()
                },
                InterpolateTransform,
            ));
        })
            .chain(),
    );
    engine.advance(0.125);
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, entity).w_axis.x, 0.5);
    let spawned = engine
        .app
        .world_mut()
        .query::<(Entity, &Transform)>()
        .iter(engine.app.world())
        .find(|(_, transform)| transform.translation.x == 20.0)
        .unwrap()
        .0;
    assert_eq!(displayed(&engine, spawned).w_axis.x, 20.0);
    engine
        .app
        .world_mut()
        .entity_mut(entity)
        .remove::<InterpolateTransform>();
    engine
        .app
        .world_mut()
        .entity_mut(entity)
        .insert(InterpolateTransform);
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, entity).w_axis.x, 1.0);
}

#[test]
fn invalid_authoritative_pose_or_orphan_is_never_resurrected() {
    let mut engine = engine();
    let parent = engine.app.world_mut().spawn(Transform::default()).id();
    let child = engine
        .app
        .world_mut()
        .spawn((
            Transform::default(),
            InterpolateTransform,
            Cube { color: [1.0; 4] },
        ))
        .id();
    set_parent(engine.app.world_mut(), child, Some(parent)).unwrap();
    engine.app.add_systems(FixedSimulation, move_x);
    engine.advance(0.125);
    present(&mut engine, 0.5);
    edit_transform(engine.app.world_mut(), child, |transform| {
        transform.translation.x = f32::NAN;
    })
    .unwrap();
    let snapshot = present(&mut engine, 0.0);
    assert_eq!(snapshot.invalid_transforms, 1);
    assert!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get(child)
            .unwrap()
            .matrix()
            .is_none()
    );
    edit_transform(engine.app.world_mut(), child, |transform| {
        transform.translation.x = 3.0;
    })
    .unwrap();
    present(&mut engine, 0.0);
    assert_eq!(displayed(&engine, child).w_axis.x, 3.0);
    engine.app.world_mut().despawn(parent);
    let snapshot = present(&mut engine, 0.0);
    assert_eq!(snapshot.invalid_transforms, 1);
    assert!(snapshot.instances.is_empty());
}

#[test]
fn culling_and_drawing_share_displayed_matrices() {
    let mut engine = engine();
    engine.set_frustum_culling(true);
    *engine.app.world_mut().resource_mut::<Camera>() = Camera {
        position: Vec3::ZERO,
        target: -Vec3::Z,
        ..Default::default()
    };
    let entity = engine
        .app
        .world_mut()
        .spawn((
            Transform {
                translation: -Vec3::Z * 5.0,
                ..Default::default()
            },
            InterpolateTransform,
            Cube { color: [1.0; 4] },
        ))
        .id();
    engine
        .app
        .add_systems(FixedSimulation, move |world: &mut World| {
            edit_transform(world, entity, |transform| {
                transform.translation.x = 100.0;
            })
            .unwrap();
        });
    engine.advance(0.125);
    let previous = present(&mut engine, 0.0);
    assert_eq!(
        (previous.visible_instances, previous.culled_instances),
        (1, 0)
    );
    assert_eq!(previous.instances[0].model.w_axis.x, 0.0);
    let current = present(&mut engine, 1.0);
    assert_eq!(
        (current.visible_instances, current.culled_instances),
        (0, 1)
    );
    assert_eq!(
        engine
            .app
            .world()
            .get::<Transform>(entity)
            .unwrap()
            .translation
            .x,
        100.0
    );
}

#[test]
fn invalid_alpha_is_atomic_and_observation_preserves_last_extraction_identity() {
    let mut engine = engine();
    let entity = engine
        .app
        .world_mut()
        .spawn((
            Transform::default(),
            InterpolateTransform,
            Cube { color: [1.0; 4] },
        ))
        .id();
    engine.app.add_systems(FixedSimulation, move_x);
    engine.advance(0.125);
    let mut snapshot = present(&mut engine, 0.5);
    assert_eq!(
        (engine.presentation().frame(), engine.presentation().tick()),
        (Some(1), Some(1))
    );
    engine.advance(0.125);
    engine.refresh_transforms();
    assert_eq!(displayed(&engine, entity).w_axis.x, 0.5);
    assert_eq!(
        (engine.presentation().frame(), engine.presentation().tick()),
        (Some(1), Some(1))
    );
    edit_transform(engine.app.world_mut(), entity, |transform| {
        transform.translation.x = 40.0;
    })
    .unwrap();
    for alpha in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        assert!(
            engine
                .extract_with_presentation(
                    1.0,
                    &mut snapshot,
                    PresentationMode::Interpolated { alpha }
                )
                .is_err()
        );
        assert_eq!(snapshot.instances[0].model.w_axis.x, 0.5);
        assert_eq!(current(&engine, entity).w_axis.x, 2.0);
        assert_eq!(displayed(&engine, entity).w_axis.x, 0.5);
        assert_eq!(
            engine.presentation().mode(),
            PresentationMode::Interpolated { alpha: 0.5 }
        );
    }
}

#[test]
fn extreme_opposite_translations_blend_without_subtraction_overflow() {
    let mut engine = engine();
    let entity = engine
        .app
        .world_mut()
        .spawn((
            Transform {
                translation: Vec3::X * 2.0e38,
                ..Default::default()
            },
            InterpolateTransform,
        ))
        .id();
    engine
        .app
        .add_systems(FixedSimulation, move |world: &mut World| {
            edit_transform(world, entity, |transform| {
                transform.translation.x = -2.0e38;
            })
            .unwrap();
        });
    engine.advance(0.125);
    assert!(current(&engine, entity).is_finite());
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, entity).w_axis.x, 0.0);
}

#[test]
fn no_opt_in_uses_no_history_or_display_scratch_and_reports_component_payload() {
    let mut engine = engine();
    engine.app.world_mut().spawn(Transform::default());
    engine.advance(0.125);
    present(&mut engine, 0.5);
    assert_eq!(engine.presentation().history_nodes(), 0);
    assert_eq!(engine.presentation().history_capacity(), 0);
    assert_eq!(engine.presentation().history_bytes(), 0);
    assert_eq!(engine.presentation().presentation_scratch_bytes(), 0);
    assert_eq!(engine.presentation().presented_nodes(), 1);
    assert_eq!(engine.presentation().presented_component_bytes(), 0);
}

#[test]
fn unsafe_intermediate_composition_falls_back_to_current_for_whole_branch() {
    let mut engine = engine();
    let parent = engine
        .app
        .world_mut()
        .spawn(Transform {
            scale: Vec3::new(1000.0, 1.0, 1.0),
            ..Default::default()
        })
        .id();
    let child = engine
        .app
        .world_mut()
        .spawn((
            Transform {
                scale: Vec3::new(0.001, 1.0, 1.0),
                ..Default::default()
            },
            InterpolateTransform,
        ))
        .id();
    let grandchild = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    set_parents(
        engine.app.world_mut(),
        &[(child, Some(parent)), (grandchild, Some(child))],
    )
    .unwrap();
    engine.refresh_transforms();
    assert!(valid_global_matrix(current(&engine, child)));
    engine
        .app
        .add_systems(FixedSimulation, move |world: &mut World| {
            edit_transform(world, child, |transform| {
                transform.rotation = Quat::from_rotation_z(std::f32::consts::PI);
            })
            .unwrap();
            edit_transform(world, grandchild, |transform| {
                transform.translation.x = 10.0;
            })
            .unwrap();
        });
    engine.advance(0.125);
    assert!(valid_global_matrix(current(&engine, child)));
    let intermediate = current(&engine, parent)
        * Transform {
            rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            scale: Vec3::new(0.001, 1.0, 1.0),
            ..Default::default()
        }
        .matrix();
    assert!(!valid_global_matrix(intermediate));
    present(&mut engine, 0.5);
    assert_matrix(displayed(&engine, child), current(&engine, child));
    assert_matrix(displayed(&engine, grandchild), current(&engine, grandchild));
}

#[test]
fn replacement_local_and_reused_entity_slot_do_not_inherit_old_history() {
    let mut engine = engine();
    let entity = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    engine.app.add_systems(FixedSimulation, move_x);
    engine.advance(0.125);
    present(&mut engine, 0.5);
    let local = *engine.app.world().get::<Transform>(entity).unwrap();
    engine
        .app
        .world_mut()
        .entity_mut(entity)
        .remove::<Transform>();
    engine.app.world_mut().entity_mut(entity).insert(local);
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, entity).w_axis.x, 1.0);
    engine.advance(0.125);
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, entity).w_axis.x, 1.5);
    let replacement = engine.app.world_mut().despawn_no_free(entity).unwrap();
    engine
        .app
        .world_mut()
        .spawn_at(
            replacement,
            (
                Transform {
                    translation: Vec3::X * 30.0,
                    ..Default::default()
                },
                InterpolateTransform,
            ),
        )
        .unwrap();
    assert_eq!(entity.index(), replacement.index());
    assert_ne!(entity, replacement);
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, replacement).w_axis.x, 30.0);
    let state = engine.app.world().resource::<InterpolationState>();
    assert_eq!(state.history.len(), 1);
    assert!(!state.indices.contains_key(&entity));
}

#[test]
fn equal_value_modify_component_starts_a_new_transform_lifetime() {
    let mut engine = engine();
    let entity = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    engine.app.add_systems(FixedSimulation, move_x);
    engine.advance(0.125);
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, entity).w_axis.x, 0.5);
    // Bevy resets Added for modify_component even when the pose bits match.
    assert_eq!(
        engine
            .app
            .world_mut()
            .modify_component(entity, |_: &mut Transform| {})
            .unwrap(),
        Some(())
    );
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, entity).w_axis.x, 1.0);
    engine.advance(0.125);
    present(&mut engine, 0.5);
    assert_eq!(displayed(&engine, entity).w_axis.x, 1.5);
}

#[test]
fn history_and_display_storage_are_bounded_reused_and_pruned_after_despawn() {
    const LIMIT: usize = 64;
    let mut engine = engine();
    engine
        .app
        .world_mut()
        .insert_resource(TransformNodeLimit(LIMIT));
    let entities: Vec<_> = (0..=LIMIT)
        .map(|_| {
            engine
                .app
                .world_mut()
                .spawn((Transform::default(), InterpolateTransform))
                .id()
        })
        .collect();
    engine.advance(0.125);
    present(&mut engine, 0.5);
    assert_eq!(engine.presentation().history_nodes(), LIMIT);
    assert_eq!(engine.presentation().presented_nodes(), LIMIT);
    let history_capacity = engine.presentation().history_capacity();
    let history_bytes = engine.presentation().history_bytes();
    let scratch_bytes = engine.presentation().presentation_scratch_bytes();
    assert!(history_capacity <= 2 * LIMIT);
    assert_eq!(
        history_bytes,
        history_capacity
            * (size_of::<History>()
                + size_of::<Interval>()
                + size_of::<NodePose>()
                + size_of::<u32>())
    );
    assert!(
        scratch_bytes
            <= LIMIT * (size_of::<Display>() + size_of::<Transform>() + 4 * size_of::<usize>())
    );
    assert_eq!(engine.presentation().presented_component_bytes(), 0);
    let world = engine.app.world_mut();
    assert_eq!(world.resource::<GlobalTransforms>().len(), LIMIT);
    assert_eq!(world.resource::<PresentedTransforms>().len(), LIMIT);
    assert_eq!(
        entities
            .iter()
            .filter(|&&entity| world.resource::<GlobalTransforms>().get(entity).is_none())
            .count(),
        1
    );
    let state = world.resource::<InterpolationState>();
    let allocation = (
        state.history.as_ptr(),
        state.next_history.as_ptr(),
        state.display.as_ptr(),
    );
    present(&mut engine, 0.25);
    present(&mut engine, 0.75);
    let state = engine.app.world().resource::<InterpolationState>();
    assert_eq!(
        allocation,
        (
            state.history.as_ptr(),
            state.next_history.as_ptr(),
            state.display.as_ptr()
        )
    );
    assert_eq!(engine.presentation().history_bytes(), history_bytes);
    assert_eq!(
        engine.presentation().presentation_scratch_bytes(),
        scratch_bytes
    );
    for entity in entities {
        engine.app.world_mut().despawn(entity);
    }
    present(&mut engine, 0.5);
    let state = engine.app.world().resource::<InterpolationState>();
    assert!(
        state.history.is_empty()
            && state.next_history.is_empty()
            && state.indices.is_empty()
            && state.display.is_empty()
    );
    assert_eq!(engine.presentation().history_nodes(), 0);
    assert_eq!(engine.presentation().presented_nodes(), 0);
    assert_eq!(engine.presentation().presented_component_bytes(), 0);
    assert_eq!(engine.presentation().history_bytes(), history_bytes);
    assert_eq!(
        engine.presentation().presentation_scratch_bytes(),
        scratch_bytes
    );
}
