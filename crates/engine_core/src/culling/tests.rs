use super::*;
use crate::prelude::*;
use glam::DMat4;

fn camera() -> Camera {
    Camera {
        position: Vec3::ZERO,
        target: Vec3::NEG_Z,
        vertical_fov_radians: std::f32::consts::FRAC_PI_2,
        near: 1.0,
        far: 10.0,
    }
}

fn engine(max_render_instances: usize) -> Engine {
    let mut engine = Engine::new(EngineConfig {
        max_render_instances,
        ..EngineConfig::default()
    });
    engine.app.world_mut().insert_resource(camera());
    engine
}

fn cube(engine: &mut Engine, position: Vec3) -> Entity {
    engine
        .app
        .world_mut()
        .spawn((
            Transform {
                translation: position,
                ..Transform::default()
            },
            Cube { color: [1.0; 4] },
        ))
        .id()
}

fn bounds(min: Vec3, max: Vec3) -> Aabb {
    Mesh::new(
        vec![
            MeshVertex {
                position: min.to_array(),
                normal: [0.0, 1.0, 0.0],
                uv: [0.0; 2],
            },
            MeshVertex {
                position: max.to_array(),
                normal: [0.0, 1.0, 0.0],
                uv: [0.0; 2],
            },
            MeshVertex {
                position: min.to_array(),
                normal: [0.0, 1.0, 0.0],
                uv: [0.0; 2],
            },
        ],
        vec![0, 1, 2],
    )
    .unwrap()
    .bounds()
}

#[test]
fn all_six_planes_reject_only_wholly_outside_boxes() {
    let frustum = Frustum::new(camera().view_projection(1.0)).unwrap();
    let bounds = Mesh::cube().bounds();
    assert!(!frustum.outside(Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)), bounds));
    for position in [
        Vec3::new(-7.0, 0.0, -5.0),
        Vec3::new(7.0, 0.0, -5.0),
        Vec3::new(0.0, -7.0, -5.0),
        Vec3::new(0.0, 7.0, -5.0),
        Vec3::new(0.0, 0.0, -0.25),
        Vec3::new(0.0, 0.0, -11.0),
        Vec3::new(0.0, 0.0, 2.0),
    ] {
        assert!(
            frustum.outside(Mat4::from_translation(position), bounds),
            "{position:?}"
        );
    }
    for position in [
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, -10.0),
        Vec3::new(5.0, 0.0, -5.0),
    ] {
        assert!(
            !frustum.outside(Mat4::from_translation(position), bounds),
            "{position:?}"
        );
    }
    assert!(!frustum.outside(Mat4::from_scale(Vec3::splat(100.0)), bounds));
}

#[test]
fn depth_is_zero_to_one_and_grazing_boxes_get_a_conservative_margin() {
    let frustum = Frustum::new(Mat4::IDENTITY).unwrap();
    let bounds = Mesh::cube().bounds();
    assert!(frustum.outside(Mat4::from_translation(Vec3::new(0.0, 0.0, -0.75)), bounds));
    assert!(!frustum.outside(Mat4::IDENTITY, bounds));
    assert!(frustum.outside(Mat4::from_translation(Vec3::new(0.0, 0.0, 1.6)), bounds));
    for x in [1.5, 1.500_001] {
        assert!(!frustum.outside(Mat4::from_translation(Vec3::new(x, 0.0, 0.5)), bounds));
    }
    assert!(frustum.outside(Mat4::from_translation(Vec3::new(1.51, 0.0, 0.5)), bounds));
}

#[test]
fn extreme_or_overflowing_arithmetic_fails_open() {
    let frustum = Frustum::new(Mat4::IDENTITY).unwrap();
    let cube = Mesh::cube().bounds();
    let extreme = bounds(Vec3::splat(-f32::MAX), Vec3::splat(f32::MAX));
    assert!(!frustum.outside(Mat4::IDENTITY, extreme));
    assert!(!frustum.outside(Mat4::from_translation(Vec3::splat(1.0e20)), cube));
    assert!(Frustum::new(Mat4::from_scale(Vec3::splat(1.0e20))).is_none());
    assert!(Frustum::new(Mat4::from_cols_array(&[f32::NAN; 16])).is_none());
    let projection = Frustum::new(Mat4::from_scale(Vec3::splat(1.0e15))).unwrap();
    let large = bounds(Vec3::splat(1.0e15), Vec3::splat(1.0e15));
    assert!(!projection.outside(Mat4::from_scale(Vec3::splat(1.0e12)), large));
}

#[test]
fn transformed_plane_support_preserves_rotated_scaled_sheared_and_reflected_boxes() {
    let mut engine = engine(10);
    let parent = engine
        .app
        .world_mut()
        .spawn(Transform {
            translation: Vec3::new(0.0, 0.0, -5.0),
            rotation: Quat::from_rotation_y(0.3),
            scale: Vec3::new(2.0, 0.5, 1.0),
        })
        .id();
    let child = engine
        .app
        .world_mut()
        .spawn((
            Transform {
                rotation: Quat::from_rotation_z(0.4),
                scale: Vec3::new(-1.0, 1.0, 1.0),
                ..Transform::default()
            },
            Cube { color: [1.0; 4] },
        ))
        .id();
    set_parent(engine.app.world_mut(), child, Some(parent)).unwrap();
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.instances.len(), 1);
    let model = snapshot.instances[0].model;
    assert!(model.x_axis.truncate().dot(model.y_axis.truncate()).abs() > 0.1);
    edit_transform(engine.app.world_mut(), parent, |transform| {
        transform.translation.x = 100.0;
    })
    .unwrap();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.instances.len(), 0);
    assert_eq!(snapshot.culled_instances, 1);
    assert_eq!(snapshot.invalid_transforms, 0);
    edit_transform(engine.app.world_mut(), parent, |transform| {
        transform.translation.x = 0.0;
    })
    .unwrap();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.instances.len(), 1);
}

#[test]
fn runtime_disable_gives_the_same_world_without_culling_or_clock_changes() {
    let mut engine = engine(10);
    cube(&mut engine, Vec3::new(0.0, 0.0, -5.0));
    cube(&mut engine, Vec3::new(100.0, 0.0, -5.0));
    assert!(engine.frustum_culling());
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(
        (
            snapshot.render_candidates,
            snapshot.visible_instances,
            snapshot.culled_instances
        ),
        (2, 1, 1)
    );
    engine.set_frustum_culling(false);
    assert!(!engine.frustum_culling());
    engine.extract(1.0, &mut snapshot);
    assert_eq!(
        (
            snapshot.render_candidates,
            snapshot.visible_instances,
            snapshot.culled_instances
        ),
        (2, 2, 0)
    );
    assert_eq!(snapshot.instances.len(), 2);
    assert_eq!(engine.app.world().resource::<FrameTime>().frame, 0);
    assert_eq!(engine.app.world().resource::<FixedTime>().tick, 0);
    engine.set_frustum_culling(true);
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.instances.len(), 1);
}

#[test]
fn counters_partition_candidates_and_culled_sources_survive_without_charging_budget() {
    let mut engine = engine(0);
    let (
        hidden_mesh,
        hidden_texture,
        hidden_material,
        dropped_mesh,
        dropped_texture,
        dropped_material,
        missing_material,
    ) = {
        let mut assets = engine.app.world_mut().resource_mut::<AssetLibrary>();
        let hidden_mesh = assets.insert_mesh(Mesh::octahedron()).unwrap();
        let hidden_texture = assets
            .insert_texture(Texture::new(1, 1, vec![255; 4]).unwrap())
            .unwrap();
        let hidden_material = assets
            .insert_material(Material::textured([1.0; 4], hidden_texture).unwrap())
            .unwrap();
        let dropped_mesh = assets.insert_mesh(Mesh::plane()).unwrap();
        let dropped_texture = assets
            .insert_texture(Texture::new(1, 1, vec![127, 127, 127, 255]).unwrap())
            .unwrap();
        let dropped_material = assets
            .insert_material(Material::textured([1.0; 4], dropped_texture).unwrap())
            .unwrap();
        let missing_material = assets
            .insert_material(Material::new([1.0; 4]).unwrap())
            .unwrap();
        assets.remove_material(missing_material);
        (
            hidden_mesh,
            hidden_texture,
            hidden_material,
            dropped_mesh,
            dropped_texture,
            dropped_material,
            missing_material,
        )
    };
    cube(&mut engine, Vec3::new(0.0, 0.0, -5.0));
    for (position, mesh, material) in [
        (Vec3::new(0.0, 0.0, -5.0), dropped_mesh, dropped_material),
        (Vec3::new(100.0, 0.0, -5.0), hidden_mesh, hidden_material),
        (Vec3::new(100.0, 0.0, -5.0), hidden_mesh, missing_material),
    ] {
        engine.app.world_mut().spawn((
            Transform {
                translation: position,
                ..Transform::default()
            },
            Mesh3d { mesh, material },
        ));
    }
    let invalid = cube(&mut engine, Vec3::new(100.0, 0.0, -5.0));
    edit_transform(engine.app.world_mut(), invalid, |transform| {
        transform.scale = Vec3::ZERO;
    })
    .unwrap();
    engine.app.world_mut().spawn(Cube { color: [1.0; 4] });
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.render_candidates, 6);
    assert_eq!(snapshot.invalid_transforms, 2);
    assert_eq!(snapshot.missing_assets, 1);
    assert_eq!(snapshot.culled_instances, 1);
    assert_eq!(snapshot.visible_instances, 2);
    assert_eq!(snapshot.dropped_instances, 2);
    assert!(snapshot.instances.is_empty());
    assert_eq!(snapshot.meshes.len(), 1);
    assert_eq!(snapshot.meshes[0].handle, hidden_mesh);
    assert_eq!(snapshot.textures.len(), 1);
    assert_eq!(snapshot.textures[0].handle, hidden_texture);
    assert!(
        !snapshot
            .textures
            .iter()
            .any(|source| source.handle == dropped_texture)
    );
    assert_eq!(
        snapshot.render_candidates,
        snapshot.invalid_transforms
            + snapshot.missing_assets
            + snapshot.culled_instances
            + snapshot.visible_instances
    );
    assert_eq!(
        snapshot.instances.len() + snapshot.dropped_instances,
        snapshot.visible_instances
    );
    let retained = std::sync::Arc::clone(&snapshot.meshes[0].mesh);
    engine
        .app
        .world_mut()
        .resource_mut::<AssetLibrary>()
        .remove_mesh(hidden_mesh);
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.culled_instances, 0);
    assert_eq!(snapshot.missing_assets, 2);
    assert!(snapshot.meshes.is_empty() && snapshot.textures.is_empty());
    assert_eq!(retained.bounds(), Mesh::octahedron().bounds());
}

#[test]
fn culled_instances_do_not_use_slots_needed_by_visible_instances() {
    let mut engine = engine(1);
    cube(&mut engine, Vec3::new(100.0, 0.0, -5.0));
    cube(&mut engine, Vec3::new(0.0, 0.0, -5.0));
    cube(&mut engine, Vec3::new(0.0, 1.0, -5.0));
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.render_candidates, 3);
    assert_eq!(snapshot.culled_instances, 1);
    assert_eq!(snapshot.visible_instances, 2);
    assert_eq!(snapshot.instances.len(), 1);
    assert_eq!(snapshot.dropped_instances, 1);
}

/// Double-precision clip distances of all eight corners, per plane.
fn reference_distances(view_projection: Mat4, model: Mat4, bounds: Aabb) -> Vec<[f64; 6]> {
    let vp = DMat4::from_cols_array(&view_projection.to_cols_array().map(f64::from));
    let model = DMat4::from_cols_array(&model.to_cols_array().map(f64::from));
    let mut corners = Vec::with_capacity(8);
    for x in [bounds.min().x, bounds.max().x] {
        for y in [bounds.min().y, bounds.max().y] {
            for z in [bounds.min().z, bounds.max().z] {
                let clip = vp * (model * Vec3::new(x, y, z).extend(1.0).as_dvec4());
                corners.push([
                    clip.w + clip.x,
                    clip.w - clip.x,
                    clip.w + clip.y,
                    clip.w - clip.y,
                    clip.z,
                    clip.w - clip.z,
                ]);
            }
        }
    }
    corners
}

fn reference_outside(view_projection: Mat4, model: Mat4, bounds: Aabb) -> bool {
    let corners = reference_distances(view_projection, model, bounds);
    (0..6).any(|plane| corners.iter().all(|corner| corner[plane] < 0.0))
}

fn reference_inside(view_projection: Mat4, model: Mat4, bounds: Aabb) -> bool {
    reference_distances(view_projection, model, bounds)
        .iter()
        .all(|corner| corner.iter().all(|distance| *distance > 0.0))
}

fn seeded(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as f32 / 16_777_216.0
    }
}

/// Interior acceptance must be proven, and must never change a decision.
fn check_classification(view_projection: Mat4, model: Mat4, bounds: Aabb) -> Classification {
    let frustum = Frustum::new(view_projection).unwrap();
    let classification = frustum.classify(model, bounds);
    assert_eq!(
        classification.outside(),
        frustum.support_outside(model, bounds),
        "{model:?} {bounds:?}"
    );
    if classification == Classification::Interior {
        assert!(
            reference_inside(view_projection, model, bounds),
            "unproven interior acceptance for {model:?} {bounds:?}"
        );
    }
    classification
}

#[test]
fn interior_acceptance_is_proven_and_never_changes_support_decisions() {
    let mut random = seeded(0x3c6e_f372_fe94_f82b);
    let bounds = bounds(Vec3::new(-0.2, -1.0, 0.1), Vec3::new(1.4, 0.3, 1.8));
    let mut counts = [0usize; 3];
    for case in 0..20_000 {
        let camera = Camera {
            position: Vec3::new(3.0, 2.0, 4.0) * (1.0 + random() * 30.0),
            target: Vec3::new(random() - 0.5, random() - 0.5, random() - 0.5) * 20.0,
            near: 0.05 + random(),
            far: 80.0 + random() * 400.0,
            vertical_fov_radians: 0.3 + random() * 2.0,
        };
        let vp = camera.view_projection(0.4 + random() * 2.0);
        let position = Vec3::new(
            random() * 240.0 - 120.0,
            random() * 160.0 - 80.0,
            random() * 240.0 - 160.0,
        );
        let scale = Vec3::new(
            (0.02 + random() * 8.0) * if case % 3 == 0 { -1.0 } else { 1.0 },
            0.02 + random() * 5.0,
            0.02 + random() * 8.0,
        );
        let rotation = Quat::from_euler(
            glam::EulerRot::XYZ,
            random() * 6.0,
            random() * 6.0,
            random() * 6.0,
        );
        // A rotated child of a nonuniformly scaled parent introduces shear.
        let model = Mat4::from_scale_rotation_translation(scale, rotation, position)
            * Mat4::from_quat(Quat::from_rotation_z(random() * 6.0));
        let classification = check_classification(vp, model, bounds);
        counts[classification as usize] += 1;
    }
    let [interior, retained, outside] = counts;
    assert!(
        interior > 2_000 && retained > 500 && outside > 2_000,
        "{counts:?}"
    );
}

#[test]
fn interior_acceptance_covers_ordinary_views_and_grazing_boxes_take_support_test() {
    let frustum = Frustum::new(Mat4::IDENTITY).unwrap();
    let unit = Mesh::cube().bounds();
    let inside = Mat4::from_scale_rotation_translation(
        Vec3::splat(0.5),
        Quat::from_rotation_y(0.3),
        Vec3::new(0.0, 0.0, 0.5),
    );
    assert_eq!(frustum.classify(inside, unit), Classification::Interior);
    // Touching x = w, the near plane, or intersecting it keeps the old decision.
    for translation in [
        Vec3::new(0.5, 0.0, 0.5),
        Vec3::new(0.0, 0.0, 0.5 - f32::EPSILON),
        Vec3::new(0.0, 0.0, 0.5),
        Vec3::new(0.0, 0.3, 0.9),
    ] {
        let model = Mat4::from_translation(translation);
        assert_eq!(
            frustum.classify(model, unit),
            Classification::Retained,
            "{translation:?}"
        );
    }
    assert_eq!(
        frustum.classify(Mat4::from_translation(Vec3::new(1.6, 0.0, 0.5)), unit),
        Classification::Outside
    );
    // Perspective views accept interior content, including just past near.
    let vp = camera().view_projection(1.0);
    for position in [
        Vec3::new(0.0, 0.0, -5.0),
        Vec3::new(1.0, -1.0, -8.0),
        Vec3::new(0.0, 0.0, -1.6),
    ] {
        assert_eq!(
            check_classification(vp, Mat4::from_translation(position), unit),
            Classification::Interior,
            "{position:?}"
        );
    }
    let mut engine = engine(10);
    cube(&mut engine, Vec3::new(0.0, 0.0, -5.0));
    cube(&mut engine, Vec3::new(0.0, 0.0, -0.9));
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(
        (snapshot.instances.len(), snapshot.culled_instances),
        (2, 0)
    );
}

#[test]
fn interior_acceptance_rejects_non_affine_and_guarded_extremes() {
    let frustum = Frustum::new(Mat4::IDENTITY).unwrap();
    let cube = Mesh::cube().bounds();
    let projective = Mat4::perspective_rh(1.0, 1.0, 0.1, 10.0);
    assert!(!frustum.contains(projective, cube));
    let mut row =
        Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5)) * Mat4::from_scale(Vec3::splat(0.5));
    assert!(frustum.contains(row, cube));
    row.x_axis.w = f32::MIN_POSITIVE;
    assert!(!frustum.contains(row, cube));
    assert!(!frustum.contains(
        Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5)),
        bounds(Vec3::splat(-f32::MAX), Vec3::splat(f32::MAX))
    ));
    let mut nonfinite = Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5));
    nonfinite.x_axis.y = f32::NAN;
    assert!(!frustum.contains(nonfinite, cube));
    nonfinite.x_axis.y = f32::INFINITY;
    assert!(!frustum.contains(nonfinite, cube));

    // The coefficient guard, not the geometry, decides these two boxes.
    let vp = Mat4::from_scale(Vec3::splat(1.0e-16));
    let far = Mat4::from_translation(Vec3::new(0.0, 0.0, 5.0e15));
    for (half, expected) in [
        (0.9e15, Classification::Interior),
        (1.1e15, Classification::Retained),
    ] {
        let large = bounds(Vec3::splat(-half), Vec3::splat(half));
        assert!(reference_inside(vp, far, large));
        assert_eq!(check_classification(vp, far, large), expected, "{half}");
    }
    // Product bounds near f32::MAX fail open instead of trusting overflow.
    let huge = Mat4::from_scale_rotation_translation(
        Vec3::new(1.0, 1.0, 1.0),
        Quat::IDENTITY,
        Vec3::new(0.0, 0.0, f32::MAX / 4.0),
    );
    let deep = Frustum::new(Mat4::from_scale(Vec3::new(1.0, 1.0, 1.0e-38))).unwrap();
    assert!(!deep.contains(huge, cube));
}

#[test]
fn stress_overview_objects_mostly_take_the_interior_path() {
    // The Stress Lab overview camera and 90,000-object field geometry.
    let span = 300.0 * 2.4;
    let camera = Camera {
        position: Vec3::new(span * 0.46, span * 0.62, span * 0.8),
        target: Vec3::ZERO,
        far: 5000.0,
        vertical_fov_radians: 55.0_f32.to_radians(),
        ..Camera::default()
    };
    let vp = camera.view_projection(1812.0 / 843.0);
    let cube = Mesh::cube().bounds();
    let mut random = seeded(0x5851_f42d_4c95_7f2d);
    let mut counts = [0usize; 3];
    for _ in 0..4_000 {
        let model = Mat4::from_scale_rotation_translation(
            Vec3::new(0.8, 1.0 + random() * 7.6, 0.8),
            Quat::from_rotation_y(random() * 6.0) * Quat::from_rotation_z(random() * 0.24 - 0.12),
            Vec3::new(
                random() * 720.0 - 360.0,
                random() * 8.0,
                random() * 720.0 - 360.0,
            ),
        );
        counts[check_classification(vp, model, cube) as usize] += 1;
    }
    assert!(
        counts[0] * 10 > counts.iter().sum::<usize>() * 9,
        "{counts:?}"
    );
}

#[test]
fn support_test_has_no_false_rejections_against_independent_clip_corner_reference() {
    let bounds = bounds(Vec3::new(-0.2, -1.0, 0.1), Vec3::new(1.4, 0.3, 1.8));
    let mut state = 0x6a09_e667_f3bc_c909u64;
    let mut random = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as f32 / 16_777_216.0
    };
    let mut rejected = 0;
    let mut visible = 0;
    for _ in 0..5000 {
        let camera = Camera {
            position: Vec3::new(3.0, 2.0, 4.0),
            target: Vec3::ZERO,
            near: 0.2,
            far: 80.0,
            ..camera()
        };
        let vp = camera.view_projection(0.4 + random() * 2.0);
        let position = Vec3::new(
            random() * 120.0 - 60.0,
            random() * 80.0 - 40.0,
            random() * 120.0 - 80.0,
        );
        let model = Mat4::from_scale_rotation_translation(
            Vec3::new(
                -(0.1 + random() * 8.0),
                0.1 + random() * 5.0,
                0.1 + random() * 8.0,
            ),
            Quat::from_euler(
                glam::EulerRot::XYZ,
                random() * 6.0,
                random() * 6.0,
                random() * 6.0,
            ),
            position,
        ) * Mat4::from_quat(Quat::from_rotation_z(random() * 6.0));
        let culled = Frustum::new(vp).unwrap().outside(model, bounds);
        if culled {
            rejected += 1;
            assert!(
                reference_outside(vp, model, bounds),
                "false rejection for {model:?}"
            );
        } else {
            visible += 1;
        }
    }
    assert!(rejected > 1000 && visible > 1000, "{rejected} / {visible}");
}
