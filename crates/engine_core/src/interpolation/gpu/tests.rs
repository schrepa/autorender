use super::*;
use crate::prelude::*;
use glam::{Vec3, Vec4};

#[derive(Component)]
struct Spin {
    step: Vec3,
    turn: f32,
}

fn animate(mut objects: Query<(&Spin, &mut Transform)>) {
    for (spin, mut transform) in &mut objects {
        transform.translation += spin.step;
        transform.rotation = Quat::from_rotation_y(spin.turn) * transform.rotation;
    }
}

fn engine() -> Engine {
    let mut engine = Engine::new(EngineConfig {
        fixed_hz: 8.0,
        frustum_culling: false,
        ..Default::default()
    });
    engine.app.add_systems(FixedSimulation, animate);
    engine
}

fn spawn(
    engine: &mut Engine,
    translation: Vec3,
    spin: Option<(Vec3, f32)>,
    marked: bool,
) -> Entity {
    let world = engine.app.world_mut();
    let mut entity = world.spawn(Transform {
        translation,
        rotation: Quat::from_rotation_z(0.3),
        scale: Vec3::new(1.5, 0.75, 1.0),
    });
    if let Some((step, turn)) = spin {
        entity.insert(Spin { step, turn });
    }
    if marked {
        entity.insert(InterpolateTransform);
    }
    entity.id()
}

fn rows_matrix(rows: PoseRows) -> Mat4 {
    Mat4::from_cols(
        Vec4::from(rows[0]),
        Vec4::from(rows[1]),
        Vec4::from(rows[2]),
        Vec4::W,
    )
    .transpose()
}

fn unpack(pose: &NodePose) -> Transform {
    let [tx, ty, tz, sx, rx, ry, rz, rw, sy, sz] = *pose;
    Transform {
        translation: Vec3::new(tx, ty, tz),
        rotation: Quat::from_xyzw(rx, ry, rz, rw),
        scale: Vec3::new(sx, sy, sz),
    }
}

/// A consumer applying gathers as the renderer does: everything on a new
/// stream identity; otherwise the previous-pose copy when `shift` follows the
/// gather it applied last, then each record stamped since its last upload.
#[derive(Default)]
struct SimulatedGpu {
    identity: u64,
    uploaded: Option<u64>,
    blend_ok: bool,
    current: Vec<NodePose>,
    previous: Vec<NodePose>,
    info: Vec<NodeInfo>,
    authority: Vec<PoseRows>,
}

impl SimulatedGpu {
    /// Returns the number of previous poses uploaded explicitly.
    fn apply(&mut self, view: &GpuPresentation<'_>) -> usize {
        let since = if self.identity == view.identity {
            self.uploaded
        } else {
            None
        };
        // Held gathers apply at their due frame; this consumer keeps its inputs.
        if since == Some(view.sequence) || (view.hold && since.is_some()) {
            return 0;
        }
        let Some(since) = since else {
            self.identity = view.identity;
            self.current = view.current.to_vec();
            self.previous = view.previous.to_vec();
            self.info = view.info.to_vec();
            self.authority = view.authority.to_vec();
            self.blend_ok = true;
            self.uploaded = Some(view.sequence);
            return usize::MAX;
        };
        if view.shift_sequence > since {
            if view.shift && since + 1 == view.sequence {
                self.previous.clone_from(&self.current);
                self.blend_ok = true;
            } else {
                self.blend_ok = false;
            }
        }
        fn copy<T: Copy>(target: &mut [T], source: &[T], stamps: &[u64], since: u64) -> usize {
            let mut copied = 0;
            for (index, &stamp) in stamps.iter().enumerate() {
                if stamp > since {
                    target[index] = source[index];
                    copied += 1;
                }
            }
            copied
        }
        // One gather behind, the published counts bound every stream's
        // stamped records, so the renderer may skip or copy streams whole.
        if since + 1 == view.sequence {
            let stamps = [
                view.current_stamps,
                view.previous_stamps,
                view.info_stamps,
                view.authority_stamps,
            ];
            for (stamps, &bound) in stamps.iter().zip(&view.stamped) {
                let stamped = stamps.iter().filter(|&&stamp| stamp > since).count();
                assert!(stamped <= bound, "{stamped} stamped records > {bound}");
            }
        }
        copy(&mut self.current, view.current, view.current_stamps, since);
        let previous = copy(
            &mut self.previous,
            view.previous,
            view.previous_stamps,
            since,
        );
        copy(&mut self.info, view.info, view.info_stamps, since);
        copy(
            &mut self.authority,
            view.authority,
            view.authority_stamps,
            since,
        );
        self.uploaded = Some(view.sequence);
        previous
    }
}

/// The GPU algorithm over uploaded records only (decision 014): inactive valid
/// nodes display their authority rows; active nodes compose their (blended)
/// local with the parent's composed or authoritative pose, validate, and fall
/// back to the current-pose composition for themselves and their subtree.
/// Uses the CPU blend, so results are comparable bit for bit.
fn mirror(gpu: &SimulatedGpu, alpha: f32, order: &[usize]) -> Vec<Option<Mat4>> {
    assert!(gpu.blend_ok, "the simulated consumer applied every gather");
    let count = gpu.info.len();
    let mut displayed = vec![None; count];
    let mut current = vec![Mat4::IDENTITY; count];
    let mut fallback = vec![false; count];
    for &index in order {
        let [parent, flags] = gpu.info[index];
        if flags & NODE_VALID == 0 {
            continue;
        }
        if flags & NODE_ACTIVE == 0 {
            let matrix = rows_matrix(gpu.authority[index]);
            displayed[index] = Some(matrix);
            current[index] = matrix;
            continue;
        }
        let (base_display, base_current, parent_fallback) = if flags & NODE_PARENT_ACTIVE != 0 {
            let parent = parent as usize;
            (
                displayed[parent].unwrap(),
                current[parent],
                fallback[parent],
            )
        } else if parent == NO_PARENT {
            (Mat4::IDENTITY, Mat4::IDENTITY, false)
        } else {
            let matrix = rows_matrix(gpu.authority[parent as usize]);
            (matrix, matrix, false)
        };
        let (previous, now) = (unpack(&gpu.previous[index]), unpack(&gpu.current[index]));
        let current_matrix = base_current * now.matrix();
        current[index] = current_matrix;
        let blended = (!parent_fallback).then(|| {
            let local = if flags & NODE_BLEND != 0 && alpha < 1.0 {
                blend(&previous, &now, alpha)
            } else {
                now
            };
            base_display * local.matrix()
        });
        displayed[index] = Some(
            match blended.filter(|matrix| valid_global_matrix(*matrix)) {
                Some(matrix) => matrix,
                None => {
                    fallback[index] = true;
                    current_matrix
                }
            },
        );
    }
    displayed
}

fn rows_bits(matrix: Mat4) -> [f32; 12] {
    // Top three rows; `==` equates the signed zeros an authority row cannot keep.
    let rows = pose_rows(matrix);
    std::array::from_fn(|index| rows[index / 4][index % 4])
}

/// CPU-prepares `alpha`, then gathers GPU inputs for the same state and checks
/// the mirror reproduces every admitted node's presented matrix and validity.
fn assert_gpu_inputs_match(engine: &mut Engine, gpu: &mut SimulatedGpu, alpha: f32) -> usize {
    engine
        .extract_frame(1.0, PresentationMode::Interpolated { alpha })
        .unwrap();
    let expected: Vec<(Entity, Option<Mat4>)> = {
        let world = engine.app.world();
        let hierarchy = world.resource::<GlobalTransforms>();
        let presented = world.resource::<PresentedTransforms>();
        hierarchy
            .sources
            .iter()
            .map(|source| {
                (
                    source.entity,
                    presented.get(source.entity).and_then(|pose| pose.matrix()),
                )
            })
            .collect()
    };
    engine.extract_gpu_frame(1.0, alpha).unwrap();
    let world = engine.app.world();
    let hierarchy = world.resource::<GlobalTransforms>();
    let state = world.resource::<PresentedTransforms>();
    assert!(world.resource::<PresentationState>().gpu_pending());
    let view = state.gpu_presentation(hierarchy);
    assert_eq!(view.alpha, alpha);
    // Parents are listed shallowest first, each after its own active parent.
    let mut seen = vec![false; view.info.len()];
    for &parent in view.parents {
        let [grandparent, flags] = view.info[parent as usize];
        assert!(flags & NODE_ACTIVE != 0);
        if flags & NODE_PARENT_ACTIVE != 0 {
            assert!(
                seen[grandparent as usize],
                "parent composed before its parent"
            );
        }
        seen[parent as usize] = true;
    }
    assert_eq!(
        view.levels.last().copied().unwrap_or(0) as usize,
        view.parents.len()
    );
    gpu.apply(&view);
    let actual = mirror(gpu, alpha, &hierarchy.order);
    for (index, (entity, expected)) in expected.iter().enumerate() {
        assert_eq!(hierarchy.sources[index].entity, *entity);
        assert_eq!(
            actual[index].map(rows_bits),
            expected.map(rows_bits),
            "node {index} ({entity:?}) at alpha {alpha}"
        );
    }
    view.info
        .iter()
        .filter(|info| info[1] & NODE_ACTIVE != 0)
        .count()
}

fn build_scene(engine: &mut Engine, extra_roots: usize, extra_static: usize) -> [Entity; 4] {
    let tick_step = Vec3::new(0.25, 0.0, 0.125);
    let root_a = spawn(engine, Vec3::ZERO, Some((tick_step, 0.2)), true);
    let child_a1 = spawn(engine, Vec3::X, Some((Vec3::Y * 0.5, -0.1)), true);
    let child_a2 = spawn(engine, Vec3::Z, None, false);
    let grandchild = spawn(engine, Vec3::Y, Some((Vec3::X * 0.1, 0.01)), true);
    // Static root with interpolated leaves, as in the stress workload.
    let root_b = spawn(engine, Vec3::splat(5.0), None, false);
    let leaves: Vec<Entity> = (0..4)
        .map(|index| {
            spawn(
                engine,
                Vec3::X * index as f32,
                Some((tick_step, 0.05)),
                true,
            )
        })
        .collect();
    // Large per-tick turns take the spherical interpolation path.
    let spinner = spawn(engine, Vec3::NEG_X * 3.0, Some((Vec3::ZERO, 1.3)), true);
    let snapped = spawn(engine, Vec3::NEG_Z * 4.0, Some((tick_step, 0.3)), true);
    let snapped_child = spawn(engine, Vec3::X, Some((tick_step, 0.2)), true);
    let plain = spawn(
        engine,
        Vec3::new(2.0, 1.0, -2.0),
        Some((tick_step, 0.4)),
        false,
    );
    let orphan_parent = spawn(engine, Vec3::ONE, None, false);
    let orphan = spawn(engine, Vec3::ONE, Some((tick_step, 0.1)), true);
    for index in 0..extra_roots {
        spawn(
            engine,
            Vec3::Y * index as f32,
            Some((tick_step, 0.02)),
            true,
        );
    }
    for index in 0..extra_static {
        spawn(engine, Vec3::Z * index as f32, None, false);
    }
    let mut links = vec![
        (child_a1, Some(root_a)),
        (child_a2, Some(root_a)),
        (grandchild, Some(child_a1)),
        (snapped_child, Some(snapped)),
        (orphan, Some(orphan_parent)),
    ];
    links.extend(leaves.iter().map(|&leaf| (leaf, Some(root_b))));
    set_parents(engine.app.world_mut(), &links).unwrap();
    let _ = (spinner, plain);
    [root_a, snapped, orphan_parent, orphan]
}

fn run(engine: &mut Engine, ticks: usize) {
    for _ in 0..ticks {
        engine.advance(0.125);
    }
}

#[test]
fn gathered_inputs_reproduce_dense_and_sparse_cpu_presentation() {
    for (extra_roots, extra_static) in [(40, 0), (0, 120)] {
        let mut engine = engine();
        let [_, snapped, orphan_parent, _] = build_scene(&mut engine, extra_roots, extra_static);
        let mut gpu = SimulatedGpu::default();
        run(&mut engine, 3);
        for alpha in [0.0, 0.25, 0.5, 0.999, 1.0] {
            assert!(assert_gpu_inputs_match(&mut engine, &mut gpu, alpha) > 0);
        }
        // Single quiet ticks keep previous poses on the GPU.
        for _ in 0..2 {
            run(&mut engine, 1);
            for alpha in [0.2, 0.9] {
                assert_gpu_inputs_match(&mut engine, &mut gpu, alpha);
            }
        }
        // A reset snaps the subtree for the pending interval; a despawned
        // parent leaves its child invalid.
        engine.app.world_mut().despawn(orphan_parent);
        run(&mut engine, 1);
        reset_interpolation(engine.app.world_mut(), snapped);
        for alpha in [0.0, 0.4, 0.8] {
            assert_gpu_inputs_match(&mut engine, &mut gpu, alpha);
        }
        // Two ticks between gathers publish explicit previous poses.
        run(&mut engine, 2);
        for alpha in [0.3, 0.7] {
            assert_gpu_inputs_match(&mut engine, &mut gpu, alpha);
        }
        // An edit refreshed between frames also needs explicit previous poses.
        run(&mut engine, 1);
        assert_gpu_inputs_match(&mut engine, &mut gpu, 0.5);
        edit_transform(engine.app.world_mut(), snapped, |pose| {
            pose.translation.y += 3.0;
        })
        .unwrap();
        engine.refresh_transforms();
        run(&mut engine, 1);
        for alpha in [0.1, 0.6] {
            assert_gpu_inputs_match(&mut engine, &mut gpu, alpha);
        }
    }
}

#[test]
fn gathering_runs_only_after_inputs_change_and_stamps_changed_records() {
    let mut engine = engine();
    build_scene(&mut engine, 8, 8);
    run(&mut engine, 2);
    engine.extract_gpu_frame(1.0, 0.5).unwrap();
    let (identity, sequence) = {
        let view = engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .gpu_presentation(engine.app.world().resource::<GlobalTransforms>());
        (view.identity, view.sequence)
    };
    // Another fraction of the same interval reuses every gathered record.
    engine.extract_gpu_frame(1.0, 0.75).unwrap();
    {
        let view = engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .gpu_presentation(engine.app.world().resource::<GlobalTransforms>());
        assert_eq!(
            (view.identity, view.sequence, view.alpha),
            (identity, sequence, 0.75)
        );
    }
    run(&mut engine, 1);
    engine.extract_gpu_frame(1.0, 0.5).unwrap();
    let world = engine.app.world();
    let view = world
        .resource::<PresentedTransforms>()
        .gpu_presentation(world.resource::<GlobalTransforms>());
    assert_eq!(view.identity, identity);
    assert_eq!(view.sequence, sequence + 1);
    // One quiet tick: moving nodes rewrote only current poses; the consumer
    // copies the previous ones. Static authority rows did not change.
    assert!(view.shift);
    assert_eq!(view.shift_sequence, view.sequence);
    assert!(view.current_stamps.contains(&view.sequence));
    assert!(!view.previous_stamps.contains(&view.sequence));
    let hierarchy = world.resource::<GlobalTransforms>();
    for (index, &[_, flags]) in view.info.iter().enumerate() {
        if flags & NODE_VALID != 0 && flags & NODE_ACTIVE == 0 {
            let rows = rows_matrix(view.authority[index]);
            assert_eq!(
                rows_bits(rows),
                rows_bits(hierarchy.global(index).unwrap()),
                "authority row {index} is current"
            );
        }
    }
    // Topology rebuilds reassign node indices under a new stream identity.
    spawn(&mut engine, Vec3::ONE, None, false);
    engine.extract_gpu_frame(1.0, 0.5).unwrap();
    let view = engine
        .app
        .world()
        .resource::<PresentedTransforms>()
        .gpu_presentation(engine.app.world().resource::<GlobalTransforms>());
    assert_ne!(view.identity, identity);
}

#[test]
fn previous_poses_upload_only_when_the_consumer_copy_cannot_supply_them() {
    let mut engine = engine();
    let [root, ..] = build_scene(&mut engine, 6, 2);
    run(&mut engine, 3);
    // (shift, explicit previous poses) of one gather after `ticks` ticks.
    let gather = |engine: &mut Engine, ticks: usize| {
        run(engine, ticks);
        engine.extract_gpu_frame(1.0, 0.5).unwrap();
        let view = engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .gpu_presentation(engine.app.world().resource::<GlobalTransforms>());
        let explicit = view
            .previous_stamps
            .iter()
            .filter(|&&stamp| stamp == view.sequence)
            .count();
        (view.shift, explicit)
    };
    let (shift, explicit) = gather(&mut engine, 0);
    assert!(
        !shift && explicit > 0,
        "a new stream sends every previous pose"
    );
    assert_eq!(
        gather(&mut engine, 1),
        (true, 0),
        "one quiet tick copies them"
    );
    let (shift, explicit) = gather(&mut engine, 2);
    assert!(!shift && explicit > 0, "a skipped tick needs them again");
    assert_eq!(gather(&mut engine, 1), (true, 0));
    // An edit refreshed before the tick moves a local outside the interval.
    edit_transform(engine.app.world_mut(), root, |pose| pose.scale.x += 0.5).unwrap();
    engine.refresh_transforms();
    let (shift, explicit) = gather(&mut engine, 1);
    assert!(!shift && explicit > 0);
    assert_eq!(gather(&mut engine, 1), (true, 0));
}

#[test]
fn preparing_a_gpu_frame_on_demand_matches_cpu_extraction() {
    let mut engine = engine();
    build_scene(&mut engine, 12, 3);
    run(&mut engine, 3);
    engine
        .extract_frame(1.0, PresentationMode::Interpolated { alpha: 0.6 })
        .unwrap();
    let expected: Vec<(Entity, Option<Mat4>, u64)> = engine
        .app
        .world()
        .resource::<PresentedTransforms>()
        .iter()
        .map(|(entity, pose)| (entity, pose.matrix(), pose.revision()))
        .collect();
    let state = engine.presentation();
    let (frame, tick) = (state.frame(), state.tick());
    engine.extract_gpu_frame(1.0, 0.6).unwrap();
    assert!(engine.presentation().gpu_pending());
    assert!(engine.prepare_presented_transforms());
    assert!(
        !engine.prepare_presented_transforms(),
        "already CPU-prepared"
    );
    let state = engine.presentation();
    assert!(!state.gpu_pending());
    assert_eq!((state.frame(), state.tick()), (frame, tick));
    assert_eq!(state.mode(), PresentationMode::Interpolated { alpha: 0.6 });
    let presented = engine.app.world().resource::<PresentedTransforms>();
    for (entity, matrix, revision) in expected {
        let pose = presented.get(entity).unwrap();
        assert_eq!(pose.matrix(), matrix);
        // Identical values keep their output revisions.
        assert_eq!(pose.revision(), revision);
    }
}

#[test]
fn preparing_a_gpu_frame_is_refused_after_its_inputs_change() {
    let mut engine = engine();
    build_scene(&mut engine, 12, 3);
    run(&mut engine, 3);
    engine.extract_gpu_frame(1.0, 0.4).unwrap();
    let before: Vec<_> = engine
        .app
        .world()
        .resource::<PresentedTransforms>()
        .iter()
        .map(|(entity, pose)| (entity, pose.matrix()))
        .collect();
    // A tick moves the inputs past the GPU-presented frame.
    run(&mut engine, 1);
    assert!(!engine.prepare_presented_transforms());
    assert!(engine.presentation().gpu_pending());
    let after: Vec<_> = engine
        .app
        .world()
        .resource::<PresentedTransforms>()
        .iter()
        .map(|(entity, pose)| (entity, pose.matrix()))
        .collect();
    assert_eq!(before, after, "a refused preparation changes nothing");
    // A CPU extraction clears the pending state.
    engine
        .extract_frame(1.0, PresentationMode::Interpolated { alpha: 0.4 })
        .unwrap();
    assert!(!engine.presentation().gpu_pending());
}

#[test]
fn a_late_frame_superseding_a_gather_made_ahead_keeps_blending() {
    let (mut reference, mut pipelined) = (engine(), engine());
    build_scene(&mut reference, 4, 4);
    build_scene(&mut pipelined, 4, 4);
    let mut gpu = SimulatedGpu::default();
    // Two frames per 8 Hz tick, as 120 Hz frames per 60 Hz tick.
    let delta = 1.0 / 16.0;
    let (mut late_frames, mut held_frames) = (0, 0);
    let mut late = false;
    for frame in 0..48 {
        // After a tick completed ahead, make the next frame late: its tick
        // is due together with the next one, superseding the held gather.
        let step = if late { 3.0 * delta } else { delta };
        late_frames += usize::from(late);
        let expected = reference.advance(step);
        reference
            .extract_frame(
                1.0,
                PresentationMode::Interpolated {
                    alpha: expected.interpolation,
                },
            )
            .unwrap();
        let report = pipelined.advance(step);
        assert_eq!(report.interpolation, expected.interpolation);
        pipelined
            .extract_gpu_frame(1.0, report.interpolation)
            .unwrap();
        let world = pipelined.app.world();
        let hierarchy = world.resource::<GlobalTransforms>();
        let view = world
            .resource::<PresentedTransforms>()
            .gpu_presentation(hierarchy);
        held_frames += usize::from(view.hold);
        gpu.apply(&view);
        // The consumer never misses a copy, and displays what the CPU
        // reference presents at the same fraction.
        let displayed = mirror(&gpu, view.alpha, &hierarchy.order);
        let presented = reference.app.world().resource::<PresentedTransforms>();
        for (index, source) in hierarchy.sources.iter().enumerate() {
            assert_eq!(
                displayed[index].map(rows_bits),
                presented
                    .get(source.entity)
                    .and_then(|pose| pose.matrix())
                    .map(rows_bits),
                "node {index} at frame {frame}"
            );
        }
        pipelined.run_ahead(delta);
        late = !late && late_frames < 4 && pipelined.tick_completed_ahead();
    }
    assert!(late_frames >= 3, "{late_frames} late frames");
    let _ = held_frames;
}
