use super::*;
use crate::{Engine, EngineConfig, Material, Mesh, PresentationMode, RenderSnapshot, Transform};
use glam::Vec3;

fn engine() -> Engine {
    let mut engine = Engine::new(EngineConfig::default());
    engine.set_frustum_culling(false);
    engine
}

type InstanceFingerprint = (MeshHandle, Option<MaterialHandle>, [u32; 16], [u32; 4]);

fn fingerprint(frame: RenderFrame<'_>) -> Vec<InstanceFingerprint> {
    frame
        .visible_indices()
        .iter()
        .map(|&index| {
            let record = &frame.snapshot().instances[index];
            (
                record.mesh,
                record.material,
                frame.model(index).to_cols_array().map(f32::to_bits),
                record.color.map(f32::to_bits),
            )
        })
        .collect()
}

#[test]
fn retained_changes_keep_structure_stable_and_split_pose_from_color() {
    let mut engine = engine();
    let entity = engine
        .app
        .world_mut()
        .spawn((Transform::default(), Cube { color: [1.0; 4] }))
        .id();
    let first = engine
        .extract_frame(1.0, PresentationMode::Current)
        .unwrap();
    let slot = first.visible_indices()[0];
    let revisions = (
        first.structure_revision(),
        first.pose_revision(),
        first.color_revision(),
    );
    let stable = engine
        .extract_frame(1.0, PresentationMode::Current)
        .unwrap();
    assert_eq!(
        (
            stable.structure_revision(),
            stable.pose_revision(),
            stable.color_revision()
        ),
        revisions
    );

    engine
        .app
        .world_mut()
        .get_mut::<Transform>(entity)
        .unwrap()
        .translation = Vec3::X;
    let moved = engine
        .extract_frame(1.0, PresentationMode::Current)
        .unwrap();
    assert_eq!(moved.visible_indices(), &[slot]);
    assert_eq!(moved.structure_revision(), revisions.0);
    assert_ne!(moved.pose_revision(), revisions.1);
    assert_eq!(moved.color_revision(), revisions.2);
    assert_eq!(moved.model(slot).w_axis.x, 1.0);
    let pose_revision = moved.pose_revision();

    engine
        .app
        .world_mut()
        .get_mut::<Cube>(entity)
        .unwrap()
        .color = [0.5, 0.25, 1.0, 1.0];
    let colored = engine
        .extract_frame(1.0, PresentationMode::Current)
        .unwrap();
    assert_eq!(colored.structure_revision(), revisions.0);
    assert_eq!(colored.pose_revision(), pose_revision);
    assert_ne!(colored.color_revision(), revisions.2);
    assert_eq!(
        colored.snapshot().instances[slot].color,
        [0.5, 0.25, 1.0, 1.0]
    );
}

#[test]
fn retained_asset_invalidation_and_entity_reuse_match_owned_extraction() {
    let mut engine = engine();
    let (mesh, material) = {
        let mut assets = engine.app.world_mut().resource_mut::<AssetLibrary>();
        (
            assets.insert_mesh(Mesh::cube()).unwrap(),
            assets
                .insert_material(Material::new([0.2, 0.3, 0.4, 1.0]).unwrap())
                .unwrap(),
        )
    };
    let first = engine
        .app
        .world_mut()
        .spawn((Transform::default(), Mesh3d { mesh, material }))
        .id();
    let second = engine
        .app
        .world_mut()
        .spawn((
            Transform {
                translation: Vec3::X,
                ..Default::default()
            },
            Cube { color: [1.0; 4] },
        ))
        .id();
    let before = engine
        .extract_frame(1.0, PresentationMode::Current)
        .unwrap();
    let stream = before.identity();
    let mut owned = RenderSnapshot::default();
    engine.extract(1.0, &mut owned);
    assert_eq!(owned.instances.len(), 2);

    // Library mutation invalidates the material cache even through a bypassed
    // ECS change tick: invalidation belongs to AssetLibrary's own contract.
    engine
        .app
        .world_mut()
        .resource_mut::<AssetLibrary>()
        .bypass_change_detection()
        .remove_material(material);
    let missing = engine
        .extract_frame(1.0, PresentationMode::Current)
        .unwrap();
    assert_eq!(missing.visible_indices().len(), 1);
    assert_eq!(missing.snapshot().missing_assets, 1);
    assert_eq!(owned.instances.len(), 2); // independently owned export survives
    engine.app.world_mut().despawn(first);
    engine.app.world_mut().despawn(second);
    assert!(
        engine
            .extract_frame(1.0, PresentationMode::Current)
            .unwrap()
            .visible_indices()
            .is_empty()
    );
    let replacement = engine
        .app
        .world_mut()
        .spawn((
            Transform {
                translation: Vec3::Y,
                ..Default::default()
            },
            Cube { color: [0.5; 4] },
        ))
        .id();
    assert_ne!(replacement, second);
    let retained = engine
        .extract_frame(1.0, PresentationMode::Current)
        .unwrap();
    assert_eq!(retained.identity(), stream);
    let expected = fingerprint(retained);
    engine.extract(1.0, &mut owned);
    let actual: Vec<_> = owned
        .instances
        .iter()
        .map(|record| {
            (
                record.mesh,
                record.material,
                record.model.to_cols_array().map(f32::to_bits),
                record.color.map(f32::to_bits),
            )
        })
        .collect();
    assert_eq!(expected, actual);
}

#[test]
fn frame_models_follow_prepared_fractional_poses_and_reindexed_slots() {
    use crate::{FixedSimulation, InterpolateTransform, set_parent};
    let mut engine = engine();
    engine.app.add_systems(
        FixedSimulation,
        |mut roots: Query<&mut Transform, With<InterpolateTransform>>| {
            for mut transform in &mut roots {
                transform.translation.x += 4.0;
            }
        },
    );
    let root = engine
        .app
        .world_mut()
        .spawn((Transform::default(), InterpolateTransform))
        .id();
    let child = engine
        .app
        .world_mut()
        .spawn((
            Transform {
                translation: Vec3::Y,
                ..Default::default()
            },
            Cube { color: [1.0; 4] },
        ))
        .id();
    set_parent(engine.app.world_mut(), child, Some(root)).unwrap();
    for _ in 0..2 {
        engine.advance(1.0 / 60.0);
    }
    let x = engine
        .app
        .world()
        .get::<Transform>(root)
        .unwrap()
        .translation
        .x;
    // Upload rows: the transposed top three rows of the model matrix, exact.
    let rows = |matrix: Mat4| {
        let columns = [matrix.x_axis, matrix.y_axis, matrix.z_axis, matrix.w_axis];
        let rows: PoseRows = [0, 1, 2].map(|row| columns.map(|column| column[row]));
        rows.map(|row| row.map(f32::to_bits))
    };
    let mut owned = RenderSnapshot::default();
    for alpha in [0.25, 0.5, 1.0] {
        let mode = PresentationMode::Interpolated { alpha };
        let frame = engine.extract_frame(1.0, mode).unwrap();
        let slot = frame.visible_indices()[0];
        let model = frame.model(slot);
        assert_eq!(
            frame.pose_rows()[slot].map(|row| row.map(f32::to_bits)),
            rows(model)
        );
        let presented = engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get(child)
            .unwrap()
            .matrix()
            .unwrap();
        assert_eq!(model, presented);
        assert!((model.w_axis.x - (x - 4.0 * (1.0 - alpha))).abs() < 1.0e-5);
        engine
            .extract_with_presentation(1.0, &mut owned, mode)
            .unwrap();
        assert_eq!(owned.instances[0].model, model);
    }
    // A structural rebuild can reorder store indices; active slots follow it.
    let first = engine.app.world_mut().spawn(Transform::default()).id();
    engine.app.world_mut().despawn(first);
    set_parent(engine.app.world_mut(), child, None).unwrap();
    let frame = engine
        .extract_frame(1.0, PresentationMode::Current)
        .unwrap();
    let slot = frame.visible_indices()[0];
    assert_eq!(frame.model(slot), Mat4::from_translation(Vec3::Y));
    assert_eq!(
        frame.pose_rows()[slot].map(|row| row.map(f32::to_bits)),
        rows(Mat4::from_translation(Vec3::Y))
    );
    assert_eq!(frame.pose_rows().len(), frame.snapshot().instances.len());
    // Out-of-range or never-mapped slots stay finite and are never submitted.
    assert_eq!(frame.model(usize::MAX), Mat4::IDENTITY);
}

#[test]
fn retained_visibility_camera_budget_and_rejected_fraction_preserve_contracts() {
    let mut engine = Engine::new(EngineConfig {
        max_render_instances: 1,
        ..Default::default()
    });
    engine
        .app
        .world_mut()
        .spawn((Transform::default(), Cube { color: [1.0; 4] }));
    engine.app.world_mut().spawn((
        Transform {
            translation: Vec3::X * 10_000.0,
            ..Default::default()
        },
        Cube { color: [1.0; 4] },
    ));
    let culled = engine
        .extract_frame(1.0, PresentationMode::Current)
        .unwrap();
    assert_eq!(culled.visible_indices().len(), 1);
    assert_eq!(culled.snapshot().culled_instances, 1);
    assert_eq!(culled.snapshot().meshes.len(), 1);
    let sequence = culled.sequence();
    assert!(
        engine
            .extract_frame(1.0, PresentationMode::Interpolated { alpha: f32::NAN })
            .is_err()
    );
    engine.set_frustum_culling(false);
    let uncull = engine
        .extract_frame(1.0, PresentationMode::Current)
        .unwrap();
    assert_eq!(uncull.sequence(), sequence + 1);
    assert_eq!(uncull.snapshot().visible_instances, 2);
    assert_eq!(uncull.snapshot().dropped_instances, 1);
    assert_eq!(uncull.visible_indices().len(), 1);
}

/// Candidate slots of a GPU-presented frame as `(entity, color)` in order.
fn gpu_candidates(engine: &mut Engine) -> (Vec<(Entity, [f32; 4])>, u64, u64) {
    let frame = engine.extract_gpu_frame(1.0, 0.5).unwrap();
    assert!(frame.gpu().is_some());
    let slots = frame.visible_indices().to_vec();
    let revisions = (frame.structure_revision(), frame.source_revision());
    let nodes: Vec<u32> = slots.iter().map(|&slot| frame.slot_nodes()[slot]).collect();
    let colors: Vec<[f32; 4]> = slots
        .iter()
        .map(|&slot| frame.snapshot().instances[slot].color)
        .collect();
    let hierarchy = engine.app.world().resource::<GlobalTransforms>();
    let entities = nodes
        .iter()
        .map(|&node| hierarchy.sources[node as usize].entity)
        .zip(colors)
        .collect();
    (entities, revisions.0, revisions.1)
}

#[test]
fn gpu_frames_list_every_candidate_and_reuse_unchanged_structure() {
    let mut engine = Engine::new(EngineConfig::default());
    assert!(engine.frustum_culling());
    let cube = |color| Cube {
        color: [color, 1.0, 1.0, 1.0],
    };
    let world = engine.app.world_mut();
    // One object far behind the camera: the CPU would cull it, the GPU decides.
    let near = world.spawn((Transform::default(), cube(0.1))).id();
    let behind = world
        .spawn((
            Transform {
                translation: Vec3::new(0.0, 0.0, 500.0),
                ..Default::default()
            },
            cube(0.2),
        ))
        .id();
    let (candidates, structure, sources) = gpu_candidates(&mut engine);
    assert_eq!(
        candidates,
        vec![(near, [0.1, 1.0, 1.0, 1.0]), (behind, [0.2, 1.0, 1.0, 1.0])]
    );
    {
        let frame = engine.extract_gpu_frame(1.0, 0.25).unwrap();
        assert!(frame.culling());
        assert_eq!(
            frame.instance_budget(),
            EngineConfig::default().max_render_instances
        );
        assert_eq!(frame.gpu().unwrap().alpha, 0.25);
        // Unchanged renderables reuse the structure without a per-renderable pass.
        assert_eq!(
            (frame.structure_revision(), frame.source_revision()),
            (structure, sources)
        );
    }
    // Source edits, spawns and despawns are still observed.
    engine
        .app
        .world_mut()
        .get_mut::<Cube>(behind)
        .unwrap()
        .color = [0.3, 1.0, 1.0, 1.0];
    let (candidates, ..) = gpu_candidates(&mut engine);
    assert_eq!(candidates[1], (behind, [0.3, 1.0, 1.0, 1.0]));
    let added = engine
        .app
        .world_mut()
        .spawn((Transform::default(), cube(0.4)))
        .id();
    let (candidates, ..) = gpu_candidates(&mut engine);
    assert_eq!(candidates.len(), 3);
    assert_eq!(candidates[2].0, added);
    engine.app.world_mut().despawn(near);
    let (candidates, ..) = gpu_candidates(&mut engine);
    // Query order after the table's swap-remove; membership is what matters.
    let mut entities: Vec<_> = candidates.iter().map(|(entity, _)| *entity).collect();
    entities.sort();
    let mut expected = vec![behind, added];
    expected.sort();
    assert_eq!(entities, expected);
}

#[test]
fn gpu_frame_counters_partition_candidates_and_list_invalid_nodes() {
    let mut engine = Engine::new(EngineConfig::default());
    let world = engine.app.world_mut();
    let valid = world
        .spawn((Transform::default(), Cube { color: [0.1; 4] }))
        .id();
    // Zero scale is admitted to the hierarchy without a valid world matrix.
    let invalid = world
        .spawn((
            Transform {
                scale: Vec3::ZERO,
                ..Default::default()
            },
            Cube { color: [0.2; 4] },
        ))
        .id();
    {
        let frame = engine.extract_gpu_frame(1.0, 0.5).unwrap();
        let snapshot = frame.snapshot();
        assert_eq!(snapshot.render_candidates, 2);
        assert_eq!(snapshot.invalid_transforms, 1);
        // Valid listed candidates, before GPU culling and budget.
        assert_eq!(snapshot.visible_instances, 1);
        assert_eq!(
            snapshot.culled_instances + snapshot.missing_assets + snapshot.dropped_instances,
            0
        );
    }
    // Validity can change without a structural rebuild, so invalid nodes stay
    // listed and the GPU hides them.
    let (candidates, ..) = gpu_candidates(&mut engine);
    let entities: Vec<Entity> = candidates.iter().map(|(entity, _)| *entity).collect();
    assert_eq!(entities, vec![valid, invalid]);
}

#[test]
fn cpu_frames_after_gpu_frames_recompute_culling_and_pose_rows() {
    use crate::{FixedSimulation, InterpolateTransform};
    let build = || {
        let mut engine = Engine::new(EngineConfig::default());
        engine.app.add_systems(
            FixedSimulation,
            |mut moving: Query<&mut Transform, With<InterpolateTransform>>| {
                for mut transform in &mut moving {
                    transform.translation.z += 40.0;
                }
            },
        );
        let world = engine.app.world_mut();
        for index in 0..6 {
            let mut entity = world.spawn((
                Transform {
                    translation: Vec3::new(index as f32, 0.0, -10.0),
                    ..Default::default()
                },
                Cube { color: [1.0; 4] },
            ));
            if index % 2 == 0 {
                entity.insert(InterpolateTransform);
            }
        }
        engine
    };
    let mut reference = build();
    let mut mixed = build();
    // One CPU frame, then motion presented only on the GPU: moving objects
    // leave the view without any CPU culling or pose rows being prepared.
    mixed.extract_frame(1.0, PresentationMode::Current).unwrap();
    for _ in 0..3 {
        reference.advance(1.0 / 60.0);
        mixed.advance(1.0 / 60.0);
        mixed.extract_gpu_frame(1.0, 0.5).unwrap();
    }
    let expected = {
        let frame = reference
            .extract_frame(1.0, PresentationMode::Current)
            .unwrap();
        (
            frame.visible_indices().to_vec(),
            frame.snapshot().culled_instances,
            frame
                .visible_indices()
                .iter()
                .map(|&slot| (pose_rows(frame.model(slot)), frame.pose_rows()[slot]))
                .collect::<Vec<_>>(),
        )
    };
    let frame = mixed.extract_frame(1.0, PresentationMode::Current).unwrap();
    assert!(frame.gpu().is_none());
    assert_eq!(frame.visible_indices(), expected.0.as_slice());
    assert_eq!(frame.snapshot().culled_instances, expected.1);
    assert!(expected.1 > 0, "moving objects left the view");
    for (&slot, (rows, reference_rows)) in frame.visible_indices().iter().zip(&expected.2) {
        assert_eq!(frame.pose_rows()[slot], *rows);
        assert_eq!(rows, reference_rows);
    }
}

#[test]
fn tints_multiply_colors_and_their_insertion_change_and_removal_reach_every_frame() {
    let mut engine = Engine::new(EngineConfig::default());
    let world = engine.app.world_mut();
    let plain = world
        .spawn((
            Transform::default(),
            Cube {
                color: [0.5, 0.5, 0.5, 1.0],
            },
        ))
        .id();
    let tinted = world
        .spawn((
            Transform::default(),
            Cube {
                color: [0.5, 0.5, 0.5, 1.0],
            },
            // Factors beyond [0, 1] read as the nearest bound, NaN as 1.
            Tint([0.5, 2.0, f32::NAN, 1.0]),
        ))
        .id();
    let color = |engine: &mut Engine, entity| {
        let (candidates, ..) = gpu_candidates(engine);
        candidates
            .into_iter()
            .find(|(candidate, _)| *candidate == entity)
            .unwrap()
            .1
    };
    assert_eq!(color(&mut engine, plain), [0.5, 0.5, 0.5, 1.0]);
    assert_eq!(color(&mut engine, tinted), [0.25, 0.5, 0.5, 1.0]);
    // Changing, inserting and removing tints all reach GPU frames, whose
    // unchanged structure skips the per-renderable pass.
    engine.app.world_mut().get_mut::<Tint>(tinted).unwrap().0 = [1.0, 0.0, 1.0, 1.0];
    assert_eq!(color(&mut engine, tinted), [0.5, 0.0, 0.5, 1.0]);
    engine
        .app
        .world_mut()
        .entity_mut(plain)
        .insert(Tint([0.0, 1.0, 1.0, 0.5]));
    assert_eq!(color(&mut engine, plain), [0.0, 0.5, 0.5, 0.5]);
    engine.app.world_mut().entity_mut(tinted).remove::<Tint>();
    assert_eq!(color(&mut engine, tinted), [0.5, 0.5, 0.5, 1.0]);
    // CPU frames too.
    engine.app.world_mut().entity_mut(plain).remove::<Tint>();
    let frame = engine
        .extract_frame(1.0, PresentationMode::Current)
        .unwrap();
    assert!(
        frame
            .snapshot()
            .instances
            .iter()
            .all(|instance| instance.color == [0.5, 0.5, 0.5, 1.0])
    );
    // And public snapshots.
    engine
        .app
        .world_mut()
        .entity_mut(plain)
        .insert(Tint([0.5, 1.0, 1.0, 1.0]));
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.0, &mut snapshot);
    let mut colors: Vec<_> = snapshot
        .instances
        .iter()
        .map(|instance| instance.color)
        .collect();
    colors.sort_by(|a, b| a[0].total_cmp(&b[0]));
    assert_eq!(colors, [[0.25, 0.5, 0.5, 1.0], [0.5, 0.5, 0.5, 1.0]]);
}
