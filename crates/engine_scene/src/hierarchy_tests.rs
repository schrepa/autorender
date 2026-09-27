use engine_core::{
    AssetBudget, AssetLibrary, Camera, GlobalTransforms, Mesh3d, Parent, Transform,
    bevy_ecs::world::World,
    edit_transform,
    glam::{Mat4, Vec3},
    propagate_transforms, set_parent,
};

use crate::{SceneDocument, SceneEntity, SceneObject};

const SOURCE: &str = r#"{
  "version": 3, "id": "garden", "revision": 1,
  "materials": [{"id":"stone","color":[1,1,1,1],"texture":"pixel"}],
  "textures": [{"id":"pixel","width":1,"height":1,"rgba8":"/////w=="}],
  "meshes": [{"id":"cube","source":{"kind":"cube"}}],
  "prefabs": [{"id":"lantern","entities":[
    {"id":"body","mesh":"cube","material":"stone","tags":["lamp"]},
    {"id":"cap","parent":"body","mesh":"cube","material":"stone","transform":{"translation":[0,1,0]}}
  ]}],
  "entities": [{"id":"west","prefab":"lantern","transform":{"translation":[3,0,0]}}]
}"#;

fn document() -> SceneDocument {
    // One opaque RGBA pixel. Kept explicit to make texture sharing measurable.
    SceneDocument::from_json(SOURCE).unwrap()
}

fn world() -> World {
    let mut world = World::new();
    world.insert_resource(AssetLibrary::new(AssetBudget::default()));
    world.insert_resource(Camera::default());
    world
}

fn node(id: &str, parent: Option<&str>) -> SceneObject {
    serde_json::from_value(serde_json::json!({"id": id, "parent": parent})).unwrap()
}

#[test]
fn expansion_preserves_local_identity_and_shares_all_assets_across_placements() {
    let mut document = document();
    let mut world = world();
    let once = document.instantiate(&mut world).unwrap();
    let bytes = world.resource::<AssetLibrary>().resident_bytes();
    once.despawn(&mut world);
    document.entities = (0..64)
        .map(|index| {
            let mut root = document.entities[0].clone();
            root.id = format!("lamp-{index}");
            root
        })
        .collect();
    let instance = document.instantiate(&mut world).unwrap();
    assert_eq!(instance.entity_count(), 192);
    let assets = world.resource::<AssetLibrary>();
    assert_eq!(assets.mesh_count(), 1);
    assert_eq!(assets.material_count(), 1);
    assert_eq!(assets.texture_count(), 1);
    assert_eq!(assets.resident_bytes(), bytes);
    let first = world
        .get::<Mesh3d>(instance.entity("lamp-0.body").unwrap())
        .unwrap();
    let last = world
        .get::<Mesh3d>(instance.entity("lamp-63.cap").unwrap())
        .unwrap();
    assert_eq!(first.mesh, last.mesh);
    assert_eq!(first.material, last.material);
    let cap = instance.entity("lamp-0.cap").unwrap();
    let metadata = world.get::<SceneEntity>(cap).unwrap();
    assert_eq!(metadata.authored_parent.as_deref(), Some("lamp-0.body"));
    assert_eq!(metadata.prefab_id.as_deref(), Some("lantern"));
    assert_eq!(metadata.prefab_instance.as_deref(), Some("lamp-0"));
    assert_eq!(metadata.prefab_local_id.as_deref(), Some("cap"));
    assert_eq!(
        world.get::<Parent>(cap).unwrap().parent(),
        instance.entity("lamp-0.body").unwrap()
    );
    assert!(
        world
            .get::<Mesh3d>(instance.entity("lamp-0").unwrap())
            .is_none()
    );
    let root = instance.entity("lamp-0").unwrap();
    edit_transform(&mut world, root, |transform| transform.translation.x = 9.0).unwrap();
    propagate_transforms(&mut world);
    assert_eq!(
        world
            .resource::<GlobalTransforms>()
            .get(cap)
            .unwrap()
            .matrix()
            .unwrap()
            .transform_point3(Vec3::ZERO),
        Vec3::new(9.0, 1.0, 0.0)
    );
    instance.despawn(&mut world);
    assert_eq!(world.resource::<AssetLibrary>().resident_bytes(), 0);
}

#[test]
fn parent_validation_handles_order_missing_self_cycles_and_depth() {
    let mut document = document();
    document.entities = vec![node("child", Some("root")), node("root", None)];
    assert!(document.validate().is_ok());
    document.entities[0].parent = Some("missing".into());
    assert_eq!(document.validate().unwrap_err().code, "unknown_parent");
    document.entities[0].parent = Some("child".into());
    assert_eq!(document.validate().unwrap_err().code, "hierarchy_cycle");
    document.entities[0].parent = Some("root".into());
    document.entities[1].parent = Some("child".into());
    assert_eq!(document.validate().unwrap_err().code, "hierarchy_cycle");
    document.entities = (0..65)
        .rev()
        .map(|index| {
            node(
                &format!("n{index}"),
                (index > 0).then(|| format!("n{}", index - 1)).as_deref(),
            )
        })
        .collect();
    assert!(document.validate().is_ok());
    document.entities.push(node("too-deep", Some("n64")));
    assert_eq!(document.validate().unwrap_err().code, "hierarchy_depth");
}

#[test]
fn prefabs_validate_unused_definitions_and_keep_parent_references_local() {
    let mut document = document();
    document.entities.clear();
    document.prefabs[0].entities[0].parent = Some("west".into());
    let error = document.validate().unwrap_err();
    assert_eq!(error.code, "unknown_parent");
    assert_eq!(error.path, "$.prefabs[0].entities[0].parent");
    document.prefabs[0].entities[0].parent = Some("cap".into());
    assert_eq!(document.validate().unwrap_err().code, "hierarchy_cycle");
    document.prefabs[0].entities[0].parent = None;
    document.prefabs[0].entities[0].prefab = Some("lantern".into());
    assert_eq!(document.validate().unwrap_err().code, "nested_prefab");
}

#[test]
fn expansion_rejects_collisions_and_long_qualified_ids_without_a_count_budget() {
    let mut document = document();
    document.entities.push(node("west.body", None));
    assert_eq!(document.validate().unwrap_err().code, "duplicate_id");
    document.entities.pop();
    document.entities[0].id = "x".repeat(62);
    assert_eq!(document.validate().unwrap_err().code, "invalid_id");
    document.entities[0].id = "west".into();
    let root = document.entities[0].clone();
    document.entities = (0..3334)
        .map(|index| {
            let mut root = root.clone();
            root.id = format!("i{index}");
            root
        })
        .collect();
    // Placements expand without a count budget (decision 029).
    assert_eq!(document.expanded_entities().unwrap().len(), 3334 * 3);
}

#[test]
fn version_three_and_renderable_pairs_are_explicit() {
    for version in [1, 2] {
        let mut document = document();
        document.textures.clear();
        document.materials[0].texture = None;
        document.version = version;
        assert_eq!(document.validate().unwrap_err().code, "version_required");
        document.prefabs.clear();
        document.entities = vec![node("root", None)];
        assert_eq!(document.validate().unwrap_err().code, "version_required");
        document.entities[0].mesh = Some("cube".into());
        document.entities[0].material = Some("stone".into());
        assert!(document.validate().is_ok());
        document.entities[0].parent = Some("root".into());
        assert_eq!(document.validate().unwrap_err().code, "version_required");
    }
    let mut document = document();
    document.entities[0].mesh = Some("cube".into());
    assert_eq!(document.validate().unwrap_err().code, "invalid_renderable");
    document.entities[0].material = Some("stone".into());
    assert_eq!(document.validate().unwrap_err().code, "invalid_renderable");
    document.entities[0].prefab = None;
    assert!(document.validate().is_ok());
}

#[test]
fn full_parent_matrices_preserve_shear() {
    let mut document = document();
    document.entities[0].transform.scale = [2.0, 3.0, 4.0];
    document.prefabs[0].entities[0].transform.rotation_degrees = [0.0, 0.0, 45.0];
    let expected = document.entities[0].transform.transform();
    let expected = Mat4::from_scale_rotation_translation(
        expected.scale,
        expected.rotation,
        expected.translation,
    ) * Mat4::from_rotation_z(45f32.to_radians());
    let mut world = world();
    let instance = document.instantiate(&mut world).unwrap();
    let matrix = world
        .resource::<GlobalTransforms>()
        .get(instance.entity("west.body").unwrap())
        .unwrap()
        .matrix()
        .unwrap();
    assert!(matrix.abs_diff_eq(expected, 0.000_001));
    assert!(matrix.x_axis.truncate().dot(matrix.y_axis.truncate()).abs() > 1.0);
}

#[test]
fn composed_overflow_and_underflow_fail_before_world_mutation() {
    for scale in [0.0001, 10_000.0] {
        let mut document = document();
        document.entities = (0..5)
            .map(|index| {
                let mut node = node(
                    &format!("n{index}"),
                    (index > 0).then(|| format!("n{}", index - 1)).as_deref(),
                );
                node.transform.scale = [scale; 3];
                node
            })
            .collect();
        let mut world = world();
        let camera = world.resource::<Camera>().position;
        let count = world.entities().len();
        assert_eq!(
            document.instantiate(&mut world).unwrap_err().code,
            "invalid_world_transform"
        );
        assert_eq!(world.entities().len(), count);
        assert_eq!(world.resource::<AssetLibrary>().resident_bytes(), 0);
        assert_eq!(world.resource::<Camera>().position, camera);
    }
}

#[test]
fn deleting_a_parent_and_disposing_scene_never_cascades_outside_ownership() {
    let mut world = world();
    let instance = document().instantiate(&mut world).unwrap();
    let outside = world.spawn(Transform::default()).id();
    let root = instance.entity("west").unwrap();
    set_parent(&mut world, outside, Some(root)).unwrap();
    world.despawn(instance.entity("west.body").unwrap());
    propagate_transforms(&mut world);
    assert!(
        world
            .resource::<GlobalTransforms>()
            .get(instance.entity("west.cap").unwrap())
            .unwrap()
            .matrix()
            .is_none()
    );
    instance.despawn(&mut world);
    propagate_transforms(&mut world);
    assert!(world.get_entity(outside).is_ok());
    assert!(
        world
            .resource::<GlobalTransforms>()
            .get(outside)
            .unwrap()
            .matrix()
            .is_none()
    );
    assert_eq!(world.resource::<AssetLibrary>().resident_bytes(), 0);
}

#[test]
fn hierarchy_capacity_failure_rolls_back_new_entities_assets_and_camera() {
    // A world at its configured transform-node limit.
    const LIMIT: usize = 64;
    let mut world = world();
    world.insert_resource(engine_core::TransformNodeLimit(LIMIT));
    world.spawn_batch((0..LIMIT).map(|_| (Transform::default(),)));
    let camera = world.resource::<Camera>().position;
    let count = world.entities().len();
    let mut document = document();
    document.camera.position = [1.0, 2.0, 3.0];
    let error = document.instantiate(&mut world).unwrap_err();
    assert_eq!(error.code, "hierarchy_install_failed");
    assert_eq!(world.entities().len(), count);
    assert_eq!(world.query::<&SceneEntity>().iter(&world).count(), 0);
    assert_eq!(world.resource::<AssetLibrary>().resident_bytes(), 0);
    assert_eq!(world.resource::<Camera>().position, camera);
}

#[test]
fn expanded_hierarchy_validation_includes_instance_roots_and_external_attachments() {
    let mut document = document();
    document.entities.push(node("attachment", Some("west.cap")));
    let expanded = document.expanded_entities().unwrap();
    assert_eq!(expanded.len(), 4);
    assert!(expanded.iter().all(|entity| entity.prefab.is_none()));
    document.entities[0].parent = Some("attachment".into());
    assert_eq!(document.validate().unwrap_err().code, "hierarchy_cycle");
    document.entities[0].parent = None;
    document.prefabs[0].entities = (0..65)
        .map(|index| {
            node(
                &format!("n{index}"),
                (index > 0).then(|| format!("n{}", index - 1)).as_deref(),
            )
        })
        .collect();
    document.entities.pop();
    assert_eq!(document.validate().unwrap_err().code, "hierarchy_depth");
    document.prefabs[0].entities.pop();
    assert!(document.validate().is_ok());
}
