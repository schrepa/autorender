use engine_core::{
    AssetBudget, AssetLibrary, Camera, Mesh, Mesh3d, Transform, bevy_ecs::world::World,
};

use crate::{MeshSource, SceneDocument, SceneEntity, SceneMesh};

const SOURCE: &str = r#"{
  "version": 1,
  "id": "test-scene",
  "revision": 4,
  "materials": [{"id": "teal", "color": [0.1, 0.8, 0.6, 1.0]}],
  "meshes": [{"id": "box", "source": {"kind": "cube"}}],
  "entities": [{
    "id": "player", "name": "Player", "mesh": "box", "material": "teal",
    "transform": {"translation": [0.0, 1.0, 0.0], "rotation_degrees": [0.0, 90.0, 0.0]},
    "tags": ["spin"]
  }]
}"#;

fn document() -> SceneDocument {
    SceneDocument::from_json(SOURCE).unwrap()
}

fn world(budget: AssetBudget) -> World {
    let mut world = World::new();
    world.insert_resource(AssetLibrary::new(budget));
    world.insert_resource(Camera::default());
    world
}

#[test]
fn canonical_roundtrip_preserves_authored_identity_and_defaults() {
    let document = document();
    let source = document.to_json().unwrap();
    let restored = SceneDocument::from_json(&source).unwrap();
    assert_eq!(restored.to_json().unwrap(), source);
    assert_eq!(restored.id, "test-scene");
    assert_eq!(restored.revision, 4);
    assert_eq!(restored.entities[0].id, "player");
    assert_eq!(restored.entities[0].transform.scale, [1.0; 3]);
    assert!(!source.contains("generation"));
    assert!(!source.contains("handle"));
}

#[test]
fn rejects_future_versions_and_unknown_fields() {
    let error = SceneDocument::from_json(&SOURCE.replace(
        "\"version\": 1",
        &format!("\"version\": {}", crate::SCENE_VERSION + 1),
    ))
    .unwrap_err();
    assert_eq!(error.code, "unsupported_version");
    assert_eq!(error.path, "$.version");
    for malformed in [
        SOURCE.replace("\"revision\": 4", "\"revision\": 4, \"typo\": true"),
        SOURCE.replace("\"kind\": \"cube\"", "\"kind\": \"cube\", \"typo\": true"),
        SOURCE.replace("\"translation\"", "\"translatoin\""),
    ] {
        assert_eq!(
            SceneDocument::from_json(&malformed).unwrap_err().code,
            "invalid_json"
        );
    }
}

#[test]
fn rejects_duplicate_ids_and_missing_references_with_paths() {
    let mut document = document();
    document.entities.push(document.entities[0].clone());
    let error = document.validate().unwrap_err();
    assert_eq!(error.code, "duplicate_id");
    assert_eq!(error.path, "$.entities[1].id");
    document.entities.pop();
    document.entities[0].mesh = Some("missing".into());
    let error = document.validate().unwrap_err();
    assert_eq!(error.code, "unknown_reference");
    assert_eq!(error.path, "$.entities[0].mesh");
    document.entities[0].mesh = Some("box".into());
    document.materials.push(document.materials[0].clone());
    assert_eq!(document.validate().unwrap_err().path, "$.materials[1].id");
    document.materials.pop();
    document.meshes.push(document.meshes[0].clone());
    assert_eq!(document.validate().unwrap_err().path, "$.meshes[1].id");
}

#[test]
fn rejects_nonfinite_singular_and_overflowing_transforms() {
    for value in [0.0, -1.0, 0.000_001, f32::NAN, f32::INFINITY, f32::MAX] {
        let mut document = document();
        document.entities[0].transform.scale[1] = value;
        let error = document.validate().unwrap_err();
        assert_eq!(error.code, "invalid_scale");
        assert_eq!(error.path, "$.entities[0].transform.scale[1]");
    }
    let mut document = document();
    document.entities[0].transform.translation[0] = f32::NAN;
    assert_eq!(document.validate().unwrap_err().code, "invalid_number");
    document.entities[0].transform.translation[0] = 0.0;
    document.entities[0].transform.rotation_degrees[2] = f32::INFINITY;
    assert_eq!(document.validate().unwrap_err().code, "invalid_number");
}

#[test]
fn rejects_bad_lens_and_coincident_camera() {
    let mut document = document();
    document.camera.fov_degrees = 180.0;
    assert_eq!(
        document.validate().unwrap_err().path,
        "$.camera.fov_degrees"
    );
    document.camera.fov_degrees = 55.0;
    document.camera.near = 0.0;
    assert_eq!(document.validate().unwrap_err().path, "$.camera.near");
    document.camera.near = 0.1;
    document.camera.far = 0.05;
    assert_eq!(document.validate().unwrap_err().path, "$.camera.far");
    document.camera.far = 500.0;
    document.camera.target = document.camera.position;
    assert_eq!(document.validate().unwrap_err().path, "$.camera.target");
}

#[test]
fn bounds_input_names_and_game_tags() {
    assert_eq!(
        SceneDocument::from_json(&" ".repeat(crate::document::MAX_SOURCE_BYTES + 1))
            .unwrap_err()
            .code,
        "source_too_large"
    );
    let mut document = document();
    document.id = "invalid id".into();
    assert_eq!(document.validate().unwrap_err().code, "invalid_id");
    document.id = "x".repeat(65);
    assert_eq!(document.validate().unwrap_err().code, "invalid_id");
    document.id = "scene".into();
    document.entities[0].name = "line\nbreak".into();
    assert_eq!(document.validate().unwrap_err().code, "invalid_name");
    document.entities[0].name.clear();
    document.entities[0].tags.push("spin".into());
    assert_eq!(document.validate().unwrap_err().code, "duplicate_tag");
    // Any number of distinct tags.
    document.entities[0].tags = (0..64).map(|index| format!("tag-{index}")).collect();
    assert!(document.validate().is_ok());
}

#[test]
fn rejects_invalid_inline_mesh_before_world_mutation() {
    let mut document = document();
    document.meshes.push(SceneMesh {
        id: "broken".into(),
        source: MeshSource::Inline {
            vertices: vec![crate::SceneVertex {
                position: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
                uv: [0.0; 2],
                color: None,
                uv1: None,
                joints: None,
                weights: None,
            }],
            indices: vec![0, 1, 2],
        },
    });
    let mut world = world(AssetBudget::default());
    let error = document.instantiate(&mut world).unwrap_err();
    assert_eq!(error.code, "invalid_mesh");
    assert_eq!(error.path, "$.meshes[1].source");
    assert_eq!(world.resource::<AssetLibrary>().mesh_count(), 0);
    assert_eq!(world.query::<&SceneEntity>().iter(&world).count(), 0);
}

#[test]
fn requires_explicit_asset_owner() {
    let error = document().instantiate(&mut World::new()).unwrap_err();
    assert_eq!(error.code, "missing_asset_library");
}

#[test]
fn material_capacity_failure_rolls_back_meshes_and_keeps_existing_resources() {
    let mut world = world(AssetBudget {
        max_materials: 0,
        ..AssetBudget::default()
    });
    let existing = world
        .resource_mut::<AssetLibrary>()
        .insert_mesh(Mesh::plane())
        .unwrap();
    let original_bytes = world.resource::<AssetLibrary>().resident_bytes();
    let original_camera = *world.resource::<Camera>();
    let mut document = document();
    document.camera.position = [3.0, 4.0, 5.0];
    let error = document.instantiate(&mut world).unwrap_err();
    assert_eq!(error.code, "asset_budget_exceeded");
    assert_eq!(error.path, "$.materials[0]");
    let assets = world.resource::<AssetLibrary>();
    assert_eq!(assets.mesh_count(), 1);
    assert_eq!(assets.material_count(), 0);
    assert_eq!(assets.resident_bytes(), original_bytes);
    assert!(assets.mesh(existing).is_some());
    assert_eq!(world.query::<&SceneEntity>().iter(&world).count(), 0);
    assert_eq!(
        world.resource::<Camera>().position,
        original_camera.position
    );
}

#[test]
fn partial_mesh_insertion_rolls_back_on_capacity_failure() {
    let mut world = world(AssetBudget {
        max_meshes: 1,
        ..AssetBudget::default()
    });
    let mut document = document();
    document.meshes.push(SceneMesh {
        id: "ground".into(),
        source: MeshSource::Plane {},
    });
    let error = document.instantiate(&mut world).unwrap_err();
    assert_eq!(error.path, "$.meshes[1]");
    assert_eq!(world.resource::<AssetLibrary>().mesh_count(), 0);
    assert_eq!(world.resource::<AssetLibrary>().resident_bytes(), 0);
    assert_eq!(world.query::<&SceneEntity>().iter(&world).count(), 0);
}

#[test]
fn instantiation_sets_authored_transform_and_identity() {
    let mut world = world(AssetBudget::default());
    let document = document();
    let instance = document.instantiate(&mut world).unwrap();
    assert_eq!(instance.id(), "test-scene");
    assert_eq!(instance.revision(), 4);
    assert_eq!(instance.entity_count(), 1);
    assert!(instance.entity("nonexistent").is_none());
    let entity = instance.entity("player").unwrap();
    let identity = world.get::<SceneEntity>(entity).unwrap();
    assert_eq!(identity.authored_id, "player");
    assert_eq!(identity.tags, ["spin"]);
    let transform = world.get::<Transform>(entity).unwrap();
    assert_eq!(transform.translation.to_array(), [0.0, 1.0, 0.0]);
    let rotated_x = transform.rotation * engine_core::glam::Vec3::X;
    assert!((rotated_x - engine_core::glam::Vec3::NEG_Z).length() < 0.000_001);
    instance.despawn(&mut world);
}

#[test]
fn explicit_disposal_preserves_unrelated_entities_and_other_instances() {
    let mut world = world(AssetBudget::default());
    let unrelated = world.spawn(Transform::default()).id();
    let first = document().instantiate(&mut world).unwrap();
    let second = document().instantiate(&mut world).unwrap();
    let first_entity = first.entity("player").unwrap();
    let second_entity = second.entity("player").unwrap();
    let stale_mesh = world.get::<Mesh3d>(first_entity).unwrap().mesh;
    assert_ne!(first_entity, second_entity);
    first.despawn(&mut world);
    assert!(world.get_entity(first_entity).is_err());
    assert!(world.get_entity(second_entity).is_ok());
    assert!(world.get_entity(unrelated).is_ok());
    let assets = world.resource::<AssetLibrary>();
    assert_eq!(assets.mesh_count(), 1);
    assert_eq!(assets.material_count(), 1);
    assert!(assets.mesh(stale_mesh).is_none());
    second.despawn(&mut world);
    assert_eq!(world.resource::<AssetLibrary>().resident_bytes(), 0);
    assert_eq!(world.query::<&Transform>().iter(&world).count(), 1);
    assert_eq!(world.query::<&SceneEntity>().iter(&world).count(), 0);
}

#[test]
fn failed_reload_preserves_old_instance_and_camera() {
    let mut world = world(AssetBudget {
        max_meshes: 1,
        ..AssetBudget::default()
    });
    let old = document().instantiate(&mut world).unwrap();
    let old_entity = old.entity("player").unwrap();
    let old_camera = *world.resource::<Camera>();
    let mut replacement = document();
    replacement.revision += 1;
    replacement.camera.position = [7.0, 8.0, 9.0];
    assert!(replacement.instantiate(&mut world).is_err());
    assert!(world.get_entity(old_entity).is_ok());
    assert_eq!(world.resource::<AssetLibrary>().mesh_count(), 1);
    assert_eq!(world.resource::<AssetLibrary>().material_count(), 1);
    assert_eq!(world.resource::<Camera>().position, old_camera.position);
    assert_eq!(old.revision(), 4);
    old.despawn(&mut world);
}

#[test]
fn disposal_tolerates_entities_removed_by_gameplay() {
    let mut world = world(AssetBudget::default());
    let instance = document().instantiate(&mut world).unwrap();
    world.despawn(instance.entity("player").unwrap());
    let unrelated = world.spawn(Transform::default()).id();
    instance.despawn(&mut world);
    assert!(world.get_entity(unrelated).is_ok());
    assert_eq!(world.resource::<AssetLibrary>().mesh_count(), 0);
}

#[test]
fn errors_have_stable_serializable_structure() {
    let mut document = document();
    document.version = crate::SCENE_VERSION + 1;
    let value = serde_json::to_value(document.validate().unwrap_err()).unwrap();
    assert_eq!(value["code"], "unsupported_version");
    assert_eq!(value["path"], "$.version");
    assert!(
        value["message"]
            .as_str()
            .unwrap()
            .contains("expected version 1")
    );
}

fn textured_document() -> SceneDocument {
    let mut document = document();
    document.version = crate::SCENE_VERSION;
    document.textures.push(crate::SceneTexture::inline(
        "tiles",
        2,
        1,
        &[255, 80, 20, 255, 40, 180, 240, 255],
    ));
    document.materials[0].texture = Some("tiles".into());
    document
}

#[test]
fn authored_texture_roundtrip_resolves_dependency_and_disposes_owned_pixels() {
    let document = textured_document();
    let document = SceneDocument::from_json(&document.to_json().unwrap()).unwrap();
    let mut world = world(AssetBudget::default());
    let instance = document.instantiate(&mut world).unwrap();
    let (id, texture_handle) = instance.textures().next().unwrap();
    assert_eq!(id, "tiles");
    let mesh = world
        .get::<Mesh3d>(instance.entity("player").unwrap())
        .unwrap();
    let assets = world.resource::<AssetLibrary>();
    assert_eq!(
        assets.material(mesh.material).unwrap().texture(),
        Some(texture_handle)
    );
    assert_eq!(assets.texture_bytes(), 8);
    assert_eq!(
        assets.texture(texture_handle).unwrap().pixels()[..4],
        [255, 80, 20, 255]
    );
    instance.despawn(&mut world);
    let assets = world.resource::<AssetLibrary>();
    assert_eq!(assets.texture_count(), 0);
    assert_eq!(assets.resident_bytes(), 0);
    assert!(assets.texture(texture_handle).is_none());
}

#[test]
fn legacy_scenes_stay_readable_and_new_texture_semantics_require_version_two() {
    let legacy = document();
    assert_eq!(legacy.version, 1);
    assert!(!legacy.to_json().unwrap().contains("textures"));
    let mut textured = textured_document();
    textured.version = 1;
    assert_eq!(textured.validate().unwrap_err().code, "version_required");
    textured.version = crate::SCENE_VERSION;
    assert!(textured.validate().is_ok());
}

#[test]
fn material_surfaces_require_version_five_and_reach_textured_materials() {
    let mut document = textured_document();
    document.materials[0].surface = Some(crate::SceneSurface {
        roughness: 0.1,
        emission: 3.0,
        ..crate::SceneSurface::default()
    });
    document.version = 4;
    let error = document.validate().unwrap_err();
    assert_eq!(error.code, "version_required");
    assert_eq!(error.path, "$.materials[0].surface");
    document.version = crate::SCENE_VERSION;
    let source = document.to_json().unwrap();
    assert!(source.contains("\"emission\": 3.0"));
    let document = SceneDocument::from_json(&source).unwrap();
    let mut world = world(AssetBudget::default());
    let instance = document.instantiate(&mut world).unwrap();
    let mesh = world
        .get::<Mesh3d>(instance.entity("player").unwrap())
        .unwrap();
    let material = *world
        .resource::<AssetLibrary>()
        .material(mesh.material)
        .unwrap();
    assert!(material.texture().is_some());
    assert_eq!(
        material.surface(),
        engine_core::Surface {
            roughness: 0.1,
            emission: 3.0,
            ..engine_core::Surface::MATTE
        }
    );
    let mut invalid = document.clone();
    invalid.materials[0].surface.as_mut().unwrap().metalness = 2.0;
    let error = invalid.validate().unwrap_err();
    assert_eq!(error.code, "invalid_material");
    assert_eq!(error.path, "$.materials[0].surface");
    // Omitted fields keep the matte defaults.
    let partial: crate::SceneSurface = serde_json::from_str(r#"{"ripples": 1}"#).unwrap();
    assert_eq!(partial.roughness, engine_core::Surface::MATTE.roughness);
    assert_eq!(partial.ripples, 1.0);
}

#[test]
fn version_five_environments_own_the_look_and_validate_every_field() {
    use engine_core::{AmbientOcclusion, Atmosphere, Bloom, Shadows};
    let source = r#"{
        "atmosphere": {"preset": "night", "exposure": 1.1},
        "shadows": {"enabled": false, "bounds": {"min": [-5, 0, -5], "max": [5, 9, 5]}},
        "bloom": {"intensity": 0}
    }"#;
    let environment: crate::SceneEnvironment = serde_json::from_str(source).unwrap();
    let mut document = document();
    document.environment = Some(environment);
    document.version = 4;
    let error = document.validate().unwrap_err();
    assert_eq!(error.code, "version_required");
    assert_eq!(error.path, "$.environment");
    document.version = crate::SCENE_VERSION;
    let document = SceneDocument::from_json(&document.to_json().unwrap()).unwrap();
    let mut world = world(AssetBudget::default());
    world.insert_resource(AmbientOcclusion {
        intensity: 0.1,
        radius: 9.0,
    });
    let instance = document.instantiate(&mut world).unwrap();
    let atmosphere = *world.resource::<Atmosphere>();
    assert_eq!(atmosphere.exposure, 1.1);
    assert_eq!(atmosphere.zenith, Atmosphere::night().zenith);
    let shadows = *world.resource::<Shadows>();
    assert!(!shadows.enabled);
    assert_eq!(shadows.bounds.unwrap().1.y, 9.0);
    assert!(!world.resource::<Bloom>().enabled());
    // Omitted sections take the defaults: the document owns the whole look.
    assert_eq!(
        *world.resource::<AmbientOcclusion>(),
        AmbientOcclusion::default()
    );
    // The current resources, as a section each, reproduce themselves.
    let current = crate::SceneEnvironment::current(&world);
    let resolved = current.resolve("$").unwrap();
    assert_eq!(resolved.atmosphere, Some(atmosphere));
    assert_eq!(resolved.shadows, Some(shadows));
    instance.despawn(&mut world);
    // Disposal keeps the environment for the next scene to replace.
    assert_eq!(*world.resource::<Atmosphere>(), atmosphere);

    for (json, path) in [
        (
            r#"{"shadows":{"resolution":3000}}"#,
            "$.environment.shadows.resolution",
        ),
        (
            r#"{"shadows":{"near_distance":900}}"#,
            "$.environment.shadows.near_distance",
        ),
        (
            r#"{"shadows":{"bounds":{"min":[1,0,0],"max":[0,1,1]}}}"#,
            "$.environment.shadows.bounds",
        ),
        (
            r#"{"atmosphere":{"fog_max":1.5}}"#,
            "$.environment.atmosphere.fog_max",
        ),
        (
            r#"{"atmosphere":{"sun_direction":[0,0,0]}}"#,
            "$.environment.atmosphere.sun_direction",
        ),
        (
            r#"{"atmosphere":{"horizon":[1,-1,1]}}"#,
            "$.environment.atmosphere.horizon",
        ),
        (
            r#"{"ambient_occlusion":{"radius":0}}"#,
            "$.environment.ambient_occlusion.radius",
        ),
        (
            r#"{"bloom":{"intensity":5}}"#,
            "$.environment.bloom.intensity",
        ),
    ] {
        let mut invalid = self::document();
        invalid.version = crate::SCENE_VERSION;
        invalid.environment = Some(serde_json::from_str(json).unwrap());
        let error = invalid.validate().unwrap_err();
        assert_eq!(error.code, "invalid_environment", "{json}");
        assert_eq!(error.path, path, "{json}");
    }
    assert!(serde_json::from_str::<crate::SceneEnvironment>(r#"{"fog":1}"#).is_err());

    // Without an environment the game's own settings stay.
    let mut plain = self::document();
    plain.version = crate::SCENE_VERSION;
    let mut world = self::world(AssetBudget::default());
    world.insert_resource(Atmosphere::dusk());
    let instance = plain.instantiate(&mut world).unwrap();
    assert_eq!(*world.resource::<Atmosphere>(), Atmosphere::dusk());
    instance.despawn(&mut world);
}

#[test]
fn large_documents_have_no_count_budget() {
    let mut document = document();
    let entity = document.entities.pop().unwrap();
    for index in 0..20_000 {
        let mut entity = entity.clone();
        entity.id = format!("object-{index}");
        document.entities.push(entity);
    }
    let source = document.to_json().unwrap();
    assert!(source.len() > 1024 * 1024);
    assert_eq!(
        SceneDocument::from_json(&source).unwrap().entities.len(),
        20_000
    );
}

#[test]
fn rejects_texture_reference_encoding_dimensions_and_alpha_before_mutation() {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    let mut broken = textured_document();
    broken.materials[0].texture = Some("missing".into());
    assert_eq!(
        broken.validate().unwrap_err().path,
        "$.materials[0].texture"
    );
    let mut cases = Vec::new();
    let mut value = textured_document();
    value.textures[0].rgba8 = Some("!".repeat(12));
    cases.push(value);
    let mut value = textured_document();
    value.textures[0].width = Some(0);
    cases.push(value);
    let mut value = textured_document();
    value.textures[0].width = Some(u32::MAX);
    cases.push(value);
    // Before version 8 texels are opaque.
    let mut value = textured_document();
    value.version = 7;
    value.textures[0].rgba8 = Some(STANDARD.encode([0u8; 8]));
    cases.push(value);
    let mut value = textured_document();
    value.textures[0].rgba8 = Some(STANDARD.encode([255u8; 4]));
    cases.push(value);
    // Both inline pixels and a file, or neither.
    let mut value = textured_document();
    value.textures[0].uri = Some("tiles.png".into());
    cases.push(value);
    for value in cases {
        let mut world = world(AssetBudget::default());
        let entity_count = world.entities().len();
        assert_eq!(
            value.instantiate(&mut world).unwrap_err().code,
            "invalid_texture"
        );
        assert_eq!(world.resource::<AssetLibrary>().resident_bytes(), 0);
        assert_eq!(world.entities().len(), entity_count);
    }
    // From version 8 texel alpha is coverage (decision 031).
    let mut value = textured_document();
    value.textures[0].rgba8 = Some(STANDARD.encode([0u8; 8]));
    let mut world = world(AssetBudget::default());
    let instance = value.instantiate(&mut world).unwrap();
    let (_, texture) = instance.textures().next().unwrap();
    assert_eq!(
        world
            .resource::<AssetLibrary>()
            .texture(texture)
            .unwrap()
            .pixels(),
        [0; 8]
    );
}

#[test]
fn version_seven_files_resolve_from_the_host_and_earlier_versions_refuse_them() {
    use crate::{SceneFileKind, SceneFiles, encode_mesh_buffer};
    use engine_core::MeshVertex;
    let mut document = textured_document();
    document.textures[0] = crate::SceneTexture::file("tiles", "textures/tiles.png");
    document.meshes.push(crate::SceneMesh {
        id: "tri".into(),
        source: MeshSource::Buffer {
            uri: "meshes/tri.bin".into(),
        },
    });
    // Structure validates without the files, which the document lists.
    let document = SceneDocument::from_json(&document.to_json().unwrap()).unwrap();
    let files = document.files();
    assert_eq!(
        files
            .iter()
            .map(|file| (file.uri.as_str(), file.kind))
            .collect::<Vec<_>>(),
        [
            ("textures/tiles.png", SceneFileKind::Image),
            ("meshes/tri.bin", SceneFileKind::Buffer),
        ]
    );
    let mut world = world(AssetBudget::default());
    let missing = document.instantiate(&mut world).unwrap_err();
    assert_eq!(
        (missing.code, missing.path.as_str()),
        ("missing_file", "$.textures[0].uri")
    );
    let mut supplied = SceneFiles::new();
    supplied.add_image(
        "textures/tiles.png",
        2,
        1,
        vec![255, 80, 20, 255, 40, 180, 240, 255],
    );
    let vertex = |position: [f32; 3]| MeshVertex {
        position,
        normal: [0.0, 0.0, 1.0],
        uv: [0.0; 2],
    };
    supplied.add_buffer(
        "meshes/tri.bin",
        encode_mesh_buffer(
            &[
                vertex([0.0; 3]),
                vertex([1.0, 0.0, 0.0]),
                vertex([0.0, 1.0, 0.0]),
            ],
            &[0, 1, 2],
        ),
    );
    document.validate_with(&supplied).unwrap();
    let instance = document.instantiate_with(&mut world, &supplied).unwrap();
    assert_eq!(world.resource::<AssetLibrary>().texture_count(), 1);
    instance.despawn(&mut world);
    // A truncated buffer and paths leaving the folder are refused.
    let mut broken = supplied.clone();
    broken.add_buffer("meshes/tri.bin", b"EMB1".to_vec());
    assert_eq!(
        document.validate_with(&broken).unwrap_err().code,
        "invalid_mesh"
    );
    for uri in ["../secret.png", "/abs.png", "a//b.png", "https://x.y/z.png"] {
        let mut outside = document.clone();
        outside.textures[0].uri = Some(uri.into());
        assert_eq!(outside.validate().unwrap_err().code, "invalid_uri", "{uri}");
    }
    let mut old = document.clone();
    old.version = 6;
    assert_eq!(
        (
            old.validate().unwrap_err().code,
            old.validate().unwrap_err().path
        ),
        ("version_required", "$.textures[0].uri".to_owned())
    );
}

#[test]
fn texture_admission_and_late_material_failure_roll_back_new_assets() {
    for budget in [
        AssetBudget {
            max_textures: 0,
            ..AssetBudget::default()
        },
        AssetBudget {
            max_materials: 0,
            ..AssetBudget::default()
        },
    ] {
        let mut world = world(budget);
        let existing = world
            .resource_mut::<AssetLibrary>()
            .insert_mesh(Mesh::plane())
            .unwrap();
        let bytes = world.resource::<AssetLibrary>().resident_bytes();
        let camera = world.resource::<Camera>().position;
        let error = textured_document().instantiate(&mut world).unwrap_err();
        assert_eq!(error.code, "asset_budget_exceeded");
        let assets = world.resource::<AssetLibrary>();
        assert_eq!(assets.resident_bytes(), bytes);
        assert_eq!(assets.texture_count(), 0);
        assert_eq!(assets.material_count(), 0);
        assert!(assets.mesh(existing).is_some());
        assert_eq!(world.resource::<Camera>().position, camera);
    }
}

#[test]
fn failed_textured_replacement_preserves_old_texture_and_entities() {
    let mut world = world(AssetBudget {
        max_textures: 1,
        ..AssetBudget::default()
    });
    let old = textured_document().instantiate(&mut world).unwrap();
    let texture = old.textures().next().unwrap().1;
    let entity = old.entity("player").unwrap();
    let bytes = world.resource::<AssetLibrary>().resident_bytes();
    assert!(textured_document().instantiate(&mut world).is_err());
    assert!(world.get_entity(entity).is_ok());
    assert!(world.resource::<AssetLibrary>().texture(texture).is_some());
    assert_eq!(world.resource::<AssetLibrary>().resident_bytes(), bytes);
    old.despawn(&mut world);
}

#[cfg(feature = "schema")]
#[test]
fn generated_schema_rejects_unknown_properties_and_describes_source_variants() {
    let schema: serde_json::Value = serde_json::from_str(&SceneDocument::schema_json()).unwrap();
    assert_eq!(schema["additionalProperties"], false);
    assert!(schema["properties"]["version"].is_object());
    let text = schema.to_string();
    assert!(text.contains("octahedron"));
    assert!(text.contains("inline"));
}

/// Every bound the schema gives an environment section's field passes that
/// field's `validate`: the schema's ranges (attributes) agree with the rules.
#[cfg(feature = "schema")]
#[test]
fn environment_schema_bounds_pass_validation() {
    use engine_core::{
        AmbientOcclusion, Bloom, Decals, InvalidSetting, LevelOfDetail, LocalLights, Particles,
        ReflectionProbes, Reflections, Refraction, Shadows, SunShafts, VirtualGeometry,
    };
    use serde::{Serialize, de::DeserializeOwned};

    fn bounds<T: Default + Serialize + DeserializeOwned>(
        definitions: &serde_json::Value,
        name: &str,
        validate: fn(&T) -> Result<(), InvalidSetting>,
        failures: &mut Vec<String>,
    ) -> usize {
        let properties = definitions[name]["properties"]
            .as_object()
            .unwrap_or_else(|| panic!("{name} has properties"));
        let mut checked = 0;
        for (field, schema) in properties {
            for bound in ["minimum", "maximum"] {
                let Some(value) = schema.get(bound) else {
                    continue;
                };
                let mut section = serde_json::to_value(T::default()).unwrap();
                section[field.as_str()] = value.clone();
                let settings: T = serde_json::from_value(section).unwrap();
                checked += 1;
                // Another field's rule may refuse the combination; this
                // field's own rule must not.
                if let Err(invalid) = validate(&settings)
                    && invalid.field == field.as_str()
                {
                    failures.push(format!("{name}.{field} {bound} {value}: {invalid}"));
                }
            }
        }
        checked
    }

    let schema: serde_json::Value = serde_json::from_str(&SceneDocument::schema_json()).unwrap();
    let definitions = &schema["$defs"];
    let mut failures = Vec::new();
    let checked = bounds(
        definitions,
        "SceneShadows",
        Shadows::validate,
        &mut failures,
    ) + bounds(
        definitions,
        "SceneAmbientOcclusion",
        AmbientOcclusion::validate,
        &mut failures,
    ) + bounds(definitions, "SceneBloom", Bloom::validate, &mut failures)
        + bounds(
            definitions,
            "SceneLocalLights",
            LocalLights::validate,
            &mut failures,
        )
        + bounds(
            definitions,
            "SceneSunShafts",
            SunShafts::validate,
            &mut failures,
        )
        + bounds(
            definitions,
            "SceneReflections",
            Reflections::validate,
            &mut failures,
        )
        + bounds(
            definitions,
            "SceneRefraction",
            Refraction::validate,
            &mut failures,
        )
        + bounds(
            definitions,
            "SceneParticles",
            Particles::validate,
            &mut failures,
        )
        + bounds(definitions, "SceneDecals", Decals::validate, &mut failures)
        + bounds(
            definitions,
            "SceneReflectionProbes",
            ReflectionProbes::validate,
            &mut failures,
        )
        + bounds(
            definitions,
            "SceneVirtualGeometry",
            VirtualGeometry::validate,
            &mut failures,
        )
        + bounds(
            definitions,
            "SceneLevelOfDetail",
            LevelOfDetail::validate,
            &mut failures,
        );
    assert!(checked >= 30, "only {checked} bounds found");
    assert!(
        failures.is_empty(),
        "{}",
        failures.join(
            "
"
        )
    );
}

#[test]
fn version_six_lights_particles_opacity_and_effects_instantiate_and_validate() {
    use engine_core::{
        LocalLight, LocalLights, ParticleBlend, ParticleEmitter, Particles, Reflections, SunShafts,
    };
    let mut source: serde_json::Value = serde_json::from_str(SOURCE).unwrap();
    source["materials"][0]["opacity"] = serde_json::json!(0.4);
    source["entities"][0]["light"] = serde_json::json!({
        "color": [1.0, 0.5, 0.25], "intensity": 30, "range": 9, "offset": [0, 2, 0],
        "spot": {"direction": [0, -2, 0], "inner_angle_degrees": 20, "outer_angle_degrees": 35}
    });
    source["entities"][0]["particles"] = serde_json::json!({
        "rate": 10, "lifetime": 3, "blend": "additive", "spread_degrees": 90
    });
    source["environment"] = serde_json::json!({
        "local_lights": {"max_lights": 64, "distance": 90},
        "sun_shafts": {"intensity": 2},
        "particles": {"max_particles": 500}
    });
    // Every version 6 field is refused by earlier versions, with its path.
    for (version, path) in [
        (5, "$.environment.local_lights"),
        (4, "$.environment.local_lights"),
    ] {
        source["version"] = serde_json::json!(version);
        let error = SceneDocument::from_json(&source.to_string()).unwrap_err();
        assert_eq!(
            (error.code, error.path.as_str()),
            ("version_required", path)
        );
    }
    let environment = source["environment"].take();
    source["version"] = serde_json::json!(5);
    let error = SceneDocument::from_json(&source.to_string()).unwrap_err();
    assert_eq!(error.path, "$.materials[0].opacity");
    source["environment"] = environment;
    source["version"] = serde_json::json!(crate::SCENE_VERSION);
    let document = SceneDocument::from_json(&source.to_string()).unwrap();
    let document = SceneDocument::from_json(&document.to_json().unwrap()).unwrap();
    let mut world = world(AssetBudget::default());
    world.insert_resource(Reflections::default());
    let instance = document.instantiate(&mut world).unwrap();
    let player = instance.entity("player").unwrap();
    let light = *world.get::<LocalLight>(player).unwrap();
    assert_eq!(light.offset, engine_core::glam::Vec3::new(0.0, 2.0, 0.0));
    let spot = light.spot.unwrap();
    assert!((spot.outer_angle - 35.0_f32.to_radians()).abs() < 1.0e-6);
    let emitter = *world.get::<ParticleEmitter>(player).unwrap();
    assert_eq!(emitter.blend, ParticleBlend::Additive);
    assert!((emitter.spread - std::f32::consts::FRAC_PI_2).abs() < 1.0e-6);
    // The material is transparent.
    let mesh = world.get::<Mesh3d>(player).unwrap().material;
    assert!(
        world
            .resource::<AssetLibrary>()
            .material(mesh)
            .unwrap()
            .is_transparent()
    );
    // The environment owns the look: named sections set, opt-in effects
    // omitted from it removed, the others at their defaults.
    assert_eq!(world.resource::<LocalLights>().max_lights, Some(64));
    assert_eq!(world.resource::<LocalLights>().distance, 90.0);
    // Cluster lists have no budget unless a scene sets one.
    assert_eq!(world.resource::<LocalLights>().lights_per_cluster, None);
    let budgeted: LocalLights = serde_json::from_str(r#"{"lights_per_cluster": 32}"#).unwrap();
    assert_eq!(budgeted.lights_per_cluster, Some(32));
    assert!(budgeted.validate().is_ok());
    let empty: LocalLights = serde_json::from_str(r#"{"lights_per_cluster": 0}"#).unwrap();
    assert_eq!(empty.validate().unwrap_err().field, "lights_per_cluster");
    assert_eq!(world.resource::<SunShafts>().intensity, 2.0);
    assert_eq!(world.resource::<Particles>().max_particles, Some(500));
    assert!(world.get_resource::<Reflections>().is_none());
    // Inspection round trips: the current look resolves to itself, with
    // missing reflections read as off.
    let current = crate::SceneEnvironment::current(&world);
    let resolved = current.resolve("$").unwrap();
    assert_eq!(resolved.sun_shafts, Some(*world.resource::<SunShafts>()));
    assert!(!resolved.reflections.unwrap().enabled());
    assert_eq!(crate::SceneLight::from(&light).runtime("$").unwrap(), light);
    assert_eq!(
        crate::SceneEmitter::from(&emitter).runtime("$").unwrap(),
        emitter
    );
    instance.despawn(&mut world);

    for (field, value, path) in [
        (
            "light",
            serde_json::json!({"range": 0}),
            "$.entities[0].light.range",
        ),
        (
            "light",
            serde_json::json!({"spot": {"inner_angle_degrees": 50, "outer_angle_degrees": 40}}),
            "$.entities[0].light.spot.outer_angle_degrees",
        ),
        (
            "particles",
            serde_json::json!({"rate": 0}),
            "$.entities[0].particles.rate",
        ),
        (
            "particles",
            serde_json::json!({"color": [[2, 0, 0, 1], [0, 0, 0, 0]]}),
            "$.entities[0].particles.color",
        ),
    ] {
        let mut invalid = source.clone();
        invalid["entities"][0][field] = value;
        let error = SceneDocument::from_json(&invalid.to_string()).unwrap_err();
        assert_eq!(error.path, path, "{field}");
    }
    let mut invalid = source.clone();
    invalid["environment"]["sun_shafts"] = serde_json::json!({"anisotropy": 1});
    let error = SceneDocument::from_json(&invalid.to_string()).unwrap_err();
    assert_eq!(error.path, "$.environment.sun_shafts.anisotropy");
}

#[test]
fn version_six_shadow_cascades_validate_and_earlier_versions_refuse_them() {
    use engine_core::Shadows;
    let mut source: serde_json::Value = serde_json::from_str(SOURCE).unwrap();
    source["environment"] = serde_json::json!({"shadows": {"cascades": 2}});
    source["version"] = serde_json::json!(5);
    let error = SceneDocument::from_json(&source.to_string()).unwrap_err();
    assert_eq!(
        (error.code, error.path.as_str()),
        ("version_required", "$.environment.shadows.cascades")
    );
    source["version"] = serde_json::json!(crate::SCENE_VERSION);
    let document = SceneDocument::from_json(&source.to_string()).unwrap();
    let mut world = world(AssetBudget::default());
    let instance = document.instantiate(&mut world).unwrap();
    assert_eq!(world.resource::<Shadows>().cascades, 2);
    // Inspection reproduces them; an omitted count takes the default.
    let current = crate::SceneEnvironment::current(&world);
    assert_eq!(current.shadows.unwrap().cascades, 2);
    instance.despawn(&mut world);
    source["environment"]["shadows"] = serde_json::json!({"enabled": true});
    let document = SceneDocument::from_json(&source.to_string()).unwrap();
    let instance = document.instantiate(&mut world).unwrap();
    assert_eq!(
        world.resource::<Shadows>().cascades,
        Shadows::default().cascades
    );
    instance.despawn(&mut world);
    for cascades in [0, 5] {
        source["environment"]["shadows"] = serde_json::json!({"cascades": cascades});
        let error = SceneDocument::from_json(&source.to_string()).unwrap_err();
        assert_eq!(
            (error.code, error.path.as_str()),
            ("invalid_environment", "$.environment.shadows.cascades")
        );
    }
    // The default count needs no version 6, a null one reads as omitted,
    // and scenes write it only when it differs.
    source["version"] = serde_json::json!(5);
    for cascades in [serde_json::json!(3), serde_json::Value::Null] {
        source["environment"]["shadows"] = serde_json::json!({"cascades": cascades});
        let document = SceneDocument::from_json(&source.to_string()).unwrap();
        assert!(!document.to_json().unwrap().contains("cascades"));
    }
}

#[test]
fn version_six_grading_overrides_its_preset_and_earlier_versions_refuse_it() {
    use engine_core::{Atmosphere, glam::Vec3};
    let mut source: serde_json::Value = serde_json::from_str(SOURCE).unwrap();
    source["environment"] = serde_json::json!({"atmosphere": {
        "preset": "dusk", "vibrance": 0.6, "contrast": 1.3,
        "shadow_tint": [0.7, 0.8, 1.5], "highlight_tint": [1.2, 1.0, 0.8]
    }});
    for (version, path) in [
        (5, "$.environment.atmosphere.contrast"),
        (4, "$.environment.atmosphere.contrast"),
    ] {
        source["version"] = serde_json::json!(version);
        let error = SceneDocument::from_json(&source.to_string()).unwrap_err();
        assert_eq!(
            (error.code, error.path.as_str()),
            ("version_required", path)
        );
    }
    source["version"] = serde_json::json!(crate::SCENE_VERSION);
    let document = SceneDocument::from_json(&source.to_string()).unwrap();
    let mut world = world(AssetBudget::default());
    let instance = document.instantiate(&mut world).unwrap();
    let atmosphere = *world.resource::<Atmosphere>();
    assert_eq!(
        atmosphere,
        Atmosphere {
            contrast: 1.3,
            vibrance: 0.6,
            shadow_tint: Vec3::new(0.7, 0.8, 1.5),
            highlight_tint: Vec3::new(1.2, 1.0, 0.8),
            ..Atmosphere::dusk()
        }
    );
    // Inspection reproduces the grade.
    let current = crate::SceneEnvironment::current(&world);
    assert_eq!(current.resolve("$").unwrap().atmosphere, Some(atmosphere));
    instance.despawn(&mut world);
    for (field, value) in [
        ("contrast", serde_json::json!(-0.5)),
        ("vibrance", serde_json::json!(-1.0)),
        ("highlight_tint", serde_json::json!([1.0, -0.5, 1.0])),
    ] {
        let mut invalid = source.clone();
        invalid["environment"]["atmosphere"][field] = value;
        let error = SceneDocument::from_json(&invalid.to_string()).unwrap_err();
        assert_eq!(
            (error.code, error.path.as_str()),
            (
                "invalid_environment",
                format!("$.environment.atmosphere.{field}").as_str()
            ),
            "{field}"
        );
    }
}

#[test]
fn version_eight_gltf_maps_instantiate_with_samplers_transforms_and_parameters() {
    let mut document = textured_document();
    let source = r#"{
      "id": "tiles-pbr", "color": [1.0, 1.0, 1.0, 1.0],
      "texture": {"id": "tiles", "wrap_u": "mirrored_repeat", "mag_filter": "nearest",
        "mipmap_filter": "none", "offset": [0.5, 0.0], "rotation": 0.25, "scale": [4.0, 2.0]},
      "metallic_roughness_texture": "tiles",
      "normal_texture": {"id": "tiles", "scale": [4.0, 4.0]}, "normal_scale": 0.5,
      "occlusion_texture": "tiles", "occlusion_strength": 0.75,
      "emissive": [2.0, 1.0, 0.0], "emissive_texture": "tiles",
      "opacity": 0.8, "alpha_mode": "mask", "alpha_cutoff": 0.3, "double_sided": true
    }"#;
    document
        .materials
        .push(serde_json::from_str(source).unwrap());
    document.entities[0].material = Some("tiles-pbr".into());
    let document = SceneDocument::from_json(&document.to_json().unwrap()).unwrap();
    let mut world = world(AssetBudget::default());
    let instance = document.instantiate(&mut world).unwrap();
    let mesh = world
        .get::<Mesh3d>(instance.entity("player").unwrap())
        .unwrap();
    let assets = world.resource::<AssetLibrary>();
    let material = assets.material(mesh.material).unwrap();
    let base = material.base_color_texture().unwrap();
    assert_eq!(base.sampler.wrap_u, engine_core::Wrap::MirroredRepeat);
    assert_eq!(base.sampler.wrap_v, engine_core::Wrap::Repeat);
    assert_eq!(base.sampler.mag_filter, engine_core::Filter::Nearest);
    assert_eq!(base.sampler.mipmap_filter, None);
    assert_eq!(
        base.transform,
        engine_core::UvTransform {
            offset: [0.5, 0.0],
            rotation: 0.25,
            scale: [4.0, 2.0]
        }
    );
    // Plain IDs take glTF's defaults from version 8.
    assert_eq!(
        material.metallic_roughness_texture().unwrap().sampler,
        engine_core::Sampler::REPEAT
    );
    assert_eq!(material.normal_scale(), 0.5);
    assert_eq!(material.occlusion_strength(), 0.75);
    assert_eq!(material.emissive(), [2.0, 1.0, 0.0]);
    assert!(material.emissive_texture().is_some());
    assert_eq!(
        material.alpha_mode(),
        engine_core::AlphaMode::Mask { cutoff: 0.3 }
    );
    assert_eq!(material.opacity(), 0.8);
    assert!(material.is_double_sided() && !material.is_transparent());
    assert_eq!(material.textures().count(), 5);
}

#[test]
fn version_eight_fields_need_version_eight_and_valid_references() {
    let base = textured_document();
    let reject = |edit: &dyn Fn(&mut crate::SceneMaterial), version: u32| {
        let mut document = base.clone();
        document.version = version;
        edit(&mut document.materials[0]);
        let error = document.validate().unwrap_err();
        (error.code, error.path)
    };
    assert_eq!(
        reject(&|material| material.double_sided = Some(true), 7),
        ("version_required", "$.materials[0].double_sided".into())
    );
    assert_eq!(
        reject(
            &|material| {
                material.texture = Some(crate::SceneMap::Map(crate::SceneTextureMap {
                    id: "tiles".into(),
                    wrap_u: None,
                    wrap_v: None,
                    mag_filter: None,
                    min_filter: None,
                    mipmap_filter: None,
                    offset: None,
                    rotation: None,
                    scale: Some([2.0, 2.0]),
                    tex_coord: None,
                }))
            },
            7
        ),
        ("version_required", "$.materials[0].texture".into())
    );
    assert_eq!(
        reject(
            &|material| material.normal_texture = Some("missing".into()),
            8
        ),
        ("unknown_reference", "$.materials[0].normal_texture".into())
    );
    assert_eq!(
        reject(&|material| material.alpha_cutoff = Some(0.5), 8),
        ("invalid_material", "$.materials[0].alpha_cutoff".into())
    );
    assert_eq!(
        reject(
            &|material| {
                material.opacity = Some(0.5);
                material.alpha_mode = Some(crate::SceneAlphaMode::Opaque);
            },
            8
        ),
        ("invalid_material", "$.materials[0].alpha_mode".into())
    );
    assert_eq!(
        reject(&|material| material.occlusion_strength = Some(2.0), 8),
        (
            "invalid_material",
            "$.materials[0].occlusion_strength".into()
        )
    );
    assert_eq!(
        reject(
            &|material| material.emissive = Some([f32::NAN, 0.0, 0.0]),
            8
        ),
        ("invalid_material", "$.materials[0].emissive".into())
    );
    // Glass (decision 038).
    assert_eq!(
        reject(&|material| material.transmission = Some(1.0), 7),
        ("version_required", "$.materials[0].transmission".into())
    );
    assert_eq!(
        reject(&|material| material.ior = Some(1.3), 8),
        ("invalid_material", "$.materials[0].ior".into())
    );
    assert_eq!(
        reject(
            &|material| {
                material.transmission = Some(0.5);
                material.ior = Some(0.5);
            },
            8
        ),
        ("invalid_material", "$.materials[0].transmission".into())
    );
}

#[test]
fn version_eight_glass_and_refraction_instantiate_opt_in() {
    use engine_core::{Refraction, Transmission};
    let mut document = textured_document();
    document.materials[0].transmission = Some(0.9);
    document.materials[0].ior = Some(1.33);
    document.materials[0].thickness = Some(0.25);
    let mut world = world(AssetBudget::default());
    world.insert_resource(Refraction::default());
    let instance = document.instantiate(&mut world).unwrap();
    let player = instance.entity("player").unwrap();
    let handle = world.get::<Mesh3d>(player).unwrap().material;
    let material = *world.resource::<AssetLibrary>().material(handle).unwrap();
    assert_eq!(
        material.transmission(),
        Transmission {
            factor: 0.9,
            ior: 1.33,
            thickness: 0.25
        }
    );
    assert!(material.is_transparent());
    // A scene environment owns refraction: omitted, it is removed.
    let mut source = serde_json::to_value(&document).unwrap();
    source["environment"] = serde_json::json!({});
    let bare = SceneDocument::from_json(&source.to_string()).unwrap();
    let mut world = world_with_refraction();
    let _ = bare.instantiate(&mut world).unwrap();
    assert!(world.get_resource::<Refraction>().is_none());
    source["environment"] = serde_json::json!({"refraction": {"strength": 2, "blur": 4}});
    let refracting = SceneDocument::from_json(&source.to_string()).unwrap();
    let mut world = world_with_refraction();
    let _ = refracting.instantiate(&mut world).unwrap();
    assert_eq!(
        *world.resource::<Refraction>(),
        Refraction {
            strength: 2.0,
            blur: 4.0
        }
    );
    // Earlier versions refuse the section.
    source["version"] = serde_json::json!(7);
    source["materials"][0]
        .as_object_mut()
        .unwrap()
        .retain(|key, _| !["transmission", "ior", "thickness"].contains(&key.as_str()));
    let error = SceneDocument::from_json(&source.to_string()).unwrap_err();
    assert_eq!(
        (error.code, error.path.as_str()),
        ("version_required", "$.environment.refraction")
    );
}

fn world_with_refraction() -> World {
    let mut world = world(AssetBudget::default());
    world.insert_resource(engine_core::Refraction::default());
    world
}

#[test]
fn texture_ids_clamp_before_version_eight_and_upgrading_keeps_that_meaning() {
    let mut document = textured_document();
    document.version = 7;
    let sampler = |document: &SceneDocument| {
        let mut world = world(AssetBudget::default());
        let instance = document.instantiate(&mut world).unwrap();
        let mesh = world
            .get::<Mesh3d>(instance.entity("player").unwrap())
            .unwrap();
        world
            .resource::<AssetLibrary>()
            .material(mesh.material)
            .unwrap()
            .base_color_texture()
            .unwrap()
            .sampler
    };
    assert_eq!(sampler(&document), engine_core::Sampler::CLAMP);
    let mut upgraded = document.clone();
    upgraded.upgrade();
    assert_eq!(upgraded.version, crate::SCENE_VERSION);
    assert_eq!(sampler(&upgraded), engine_core::Sampler::CLAMP);
    // The same plain ID repeats in a version 8 document.
    document.version = 8;
    assert_eq!(sampler(&document), engine_core::Sampler::REPEAT);
}

#[test]
fn version_eight_tints_attach_to_their_objects_and_validate() {
    let mut document = textured_document();
    document.entities[0].tint = Some([1.0, 0.5, 0.25, 1.0]);
    let document = SceneDocument::from_json(&document.to_json().unwrap()).unwrap();
    let mut world = world(AssetBudget::default());
    let instance = document.instantiate(&mut world).unwrap();
    assert_eq!(
        world.get::<engine_core::Tint>(instance.entity("player").unwrap()),
        Some(&engine_core::Tint([1.0, 0.5, 0.25, 1.0]))
    );
    let reject = |edit: &dyn Fn(&mut SceneDocument)| {
        let mut document = document.clone();
        edit(&mut document);
        let error = document.validate().unwrap_err();
        (error.code, error.path)
    };
    assert_eq!(
        reject(&|document| document.version = 7),
        ("version_required", "$.entities[0].tint".into())
    );
    assert_eq!(
        reject(&|document| document.entities[0].tint = Some([1.5, 1.0, 1.0, 1.0])),
        ("invalid_tint", "$.entities[0].tint".into())
    );
    assert_eq!(
        reject(&|document| document.entities[0].tint = Some([f32::NAN, 1.0, 1.0, 1.0])),
        ("invalid_tint", "$.entities[0].tint".into())
    );
}

#[test]
fn version_eight_vertex_colors_and_second_coordinates_load_inline_and_from_buffers() {
    use crate::{MeshStreams, SceneFiles, encode_mesh_buffer_with};
    let vertex = |position: [f32; 3]| crate::SceneVertex {
        position,
        normal: [0.0, 0.0, 1.0],
        uv: [position[0], position[1]],
        color: Some([1.0, 0.5, 0.25, 1.0]),
        uv1: Some([0.5, 0.5]),
        joints: None,
        weights: None,
    };
    let vertices = vec![
        vertex([0.0; 3]),
        vertex([1.0, 0.0, 0.0]),
        vertex([0.0, 1.0, 0.0]),
    ];
    let mut document = textured_document();
    document.meshes.push(SceneMesh {
        id: "inline-colors".into(),
        source: MeshSource::Inline {
            vertices: vertices.clone(),
            indices: vec![0, 1, 2],
        },
    });
    document.meshes.push(SceneMesh {
        id: "buffer-colors".into(),
        source: MeshSource::Buffer {
            uri: "meshes/colors.bin".into(),
        },
    });
    for mesh in ["inline-colors", "buffer-colors"] {
        let mut entity = document.entities[0].clone();
        entity.id = mesh.into();
        entity.mesh = Some(mesh.into());
        entity.tags.clear();
        document.entities.push(entity);
    }
    let colors: Vec<[f32; 4]> = vertices
        .iter()
        .map(|vertex| vertex.color.unwrap())
        .collect();
    let uv1: Vec<[f32; 2]> = vertices.iter().map(|vertex| vertex.uv1.unwrap()).collect();
    let bytes = encode_mesh_buffer_with(
        &vertices,
        &[0, 1, 2],
        MeshStreams {
            colors: Some(&colors),
            uv1: Some(&uv1),
            ..MeshStreams::default()
        },
    );
    assert_eq!(&bytes[..4], b"EMB2");
    let mut files = SceneFiles::new();
    files.add_buffer("meshes/colors.bin", bytes);
    let document = SceneDocument::from_json(&document.to_json().unwrap()).unwrap();
    let mut world = world(AssetBudget::default());
    let instance = document.instantiate_with(&mut world, &files).unwrap();
    for id in ["inline-colors", "buffer-colors"] {
        let mesh = world
            .get::<Mesh3d>(instance.entity(id).unwrap())
            .unwrap()
            .mesh;
        let assets = world.resource::<AssetLibrary>();
        let mesh = assets.mesh(mesh).unwrap();
        assert_eq!(mesh.colors().unwrap(), &colors[..]);
        assert_eq!(mesh.uv1().unwrap(), &uv1[..]);
    }
    // Mixed vertices and earlier versions refuse them.
    let mut mixed = document.clone();
    if let MeshSource::Inline { vertices, .. } = &mut mixed.meshes[1].source {
        vertices[1].color = None;
    }
    let error = mixed.validate().unwrap_err();
    assert_eq!(
        (error.code, error.path),
        ("invalid_mesh", "$.meshes[1].source.vertices".into())
    );
    let mut earlier = document.clone();
    earlier.version = 7;
    let error = earlier.validate().unwrap_err();
    assert_eq!(
        (error.code, error.path),
        ("version_required", "$.meshes[1].source.vertices".into())
    );
    // A version 7 document may not read EMB2 extras either.
    let mut earlier = document.clone();
    earlier.version = 7;
    earlier.meshes.remove(1);
    earlier
        .entities
        .retain(|entity| entity.id != "inline-colors");
    let error = earlier.validate_with(&files).unwrap_err();
    assert_eq!(error.code, "version_required");
}

#[test]
fn compressed_image_files_load_as_block_compressed_textures_from_version_eight() {
    use crate::{SceneFiles, SceneTexture};
    use engine_core::TextureFormat;
    let mut document = textured_document();
    document.version = 8;
    document.textures = vec![SceneTexture::file("tiles", "images/checker.ktx2")];
    let mut files = SceneFiles::new();
    files.add_compressed_image(
        "images/checker.ktx2",
        TextureFormat::Bc7,
        8,
        8,
        vec![vec![7; 64], vec![8; 16], vec![9; 16], vec![10; 16]],
    );
    let mut world = world(AssetBudget::default());
    let instance = document.instantiate_with(&mut world, &files).unwrap();
    let (_, handle) = instance.textures().next().unwrap();
    let texture = world
        .resource::<AssetLibrary>()
        .texture(handle)
        .unwrap()
        .clone();
    assert_eq!(texture.format(), TextureFormat::Bc7);
    assert_eq!(texture.levels().len(), 4);
    // Before version 8 compressed images are refused; so are broken ones.
    document.version = 7;
    let error = document.validate_with(&files).unwrap_err();
    assert_eq!(
        (error.code, error.path.as_str()),
        ("version_required", "$.textures[0].uri")
    );
    document.version = 8;
    files.add_compressed_image(
        "images/checker.ktx2",
        TextureFormat::Bc7,
        8,
        8,
        vec![vec![7; 63]],
    );
    assert_eq!(
        document.validate_with(&files).unwrap_err().code,
        "invalid_texture"
    );
}

#[test]
fn version_eight_decals_instantiate_validate_and_earlier_versions_refuse_them() {
    use engine_core::{Decal, DecalBlend, Decals};
    let source = serde_json::json!({
        "version": 8,
        "id": "decal-scene",
        "revision": 1,
        "materials": [
            {"id": "teal", "color": [0.1, 0.8, 0.6, 1.0]},
            {"id": "grime", "color": [0.4, 0.3, 0.2, 1.0], "opacity": 0.8}
        ],
        "meshes": [{"id": "box", "source": {"kind": "cube"}}],
        "entities": [
            {"id": "floor", "mesh": "box", "material": "teal"},
            {"id": "stain", "decal": {"material": "grime", "blend": "multiply", "opacity": 0.5,
                "fade_degrees": 90.0},
                "transform": {"scale": [2.0, 0.5, 2.0]}},
            {"id": "mark", "decal": {"material": "teal"}}
        ],
        "environment": {"decals": {"enabled": true, "max_decals": 8}}
    });
    let document = SceneDocument::from_json(&source.to_string()).unwrap();
    let mut world = world(AssetBudget::default());
    let instance = document.instantiate(&mut world).unwrap();
    let grime = instance
        .materials()
        .find(|(id, _)| *id == "grime")
        .map(|(_, handle)| handle)
        .unwrap();
    let stain = *world
        .get::<Decal>(instance.entity("stain").unwrap())
        .unwrap();
    assert_eq!(stain.material, grime);
    assert_eq!(stain.blend, DecalBlend::Multiply);
    assert_eq!(stain.opacity, 0.5);
    assert_eq!(stain.fade_angle, std::f32::consts::FRAC_PI_2);
    // Omitted fields take the defaults.
    let mark = *world
        .get::<Decal>(instance.entity("mark").unwrap())
        .unwrap();
    assert_eq!(mark.blend, DecalBlend::Alpha);
    assert_eq!(mark.opacity, 1.0);
    assert!((mark.fade_angle - 60.0_f32.to_radians()).abs() < 1.0e-6);
    assert_eq!(
        *world.resource::<Decals>(),
        Decals {
            enabled: true,
            max_decals: Some(8)
        }
    );
    // Decals round-trip as authored.
    let saved = serde_json::to_value(&document).unwrap();
    assert_eq!(
        saved["entities"][1]["decal"],
        source["entities"][1]["decal"]
    );
    // An environment without the section draws every decal.
    let mut bare = source.clone();
    bare["environment"] = serde_json::json!({});
    let mut world = self::world(AssetBudget::default());
    let _ = SceneDocument::from_json(&bare.to_string())
        .unwrap()
        .instantiate(&mut world)
        .unwrap();
    assert_eq!(*world.resource::<Decals>(), Decals::default());
    let refused = |edit: &dyn Fn(&mut serde_json::Value)| {
        let mut source = source.clone();
        edit(&mut source);
        let error = SceneDocument::from_json(&source.to_string()).unwrap_err();
        (error.code, error.path)
    };
    assert_eq!(
        refused(&|source| source["entities"][1]["decal"]["material"] = "missing".into()),
        ("unknown_reference", "$.entities[1].decal.material".into())
    );
    assert_eq!(
        refused(&|source| source["entities"][1]["decal"]["opacity"] = 2.into()),
        ("invalid_decal", "$.entities[1].decal.opacity".into())
    );
    assert_eq!(
        refused(&|source| source["entities"][1]["decal"]["fade_degrees"] = 91.into()),
        ("invalid_decal", "$.entities[1].decal.fade_degrees".into())
    );
    assert_eq!(
        refused(&|source| source["environment"]["decals"]["max_decals"] = 0.into()),
        (
            "invalid_environment",
            "$.environment.decals.max_decals".into()
        )
    );
    // Earlier versions refuse decals and their settings.
    assert_eq!(
        refused(&|source| source["version"] = 7.into()),
        ("version_required", "$.entities[1].decal".into())
    );
    assert_eq!(
        refused(&|source| {
            source["version"] = 7.into();
            source["entities"].as_array_mut().unwrap().truncate(1);
        }),
        ("version_required", "$.environment.decals".into())
    );
}

#[test]
fn version_eight_reflection_probes_instantiate_validate_and_earlier_versions_refuse_them() {
    use crate::{SceneEnvironment, SceneReflectionProbe};
    use engine_core::{ReflectionProbe, ReflectionProbes, glam::Vec3};
    let source = serde_json::json!({
        "version": 8,
        "id": "probe-scene",
        "revision": 1,
        "materials": [{"id": "teal", "color": [0.1, 0.8, 0.6, 1.0]}],
        "meshes": [{"id": "box", "source": {"kind": "cube"}}],
        "prefabs": [{"id": "kiosk", "entities": [
            {"id": "booth", "reflection_probe": {"extent": [1.0, 1.5, 1.0]}}
        ]}],
        "entities": [
            {"id": "floor", "mesh": "box", "material": "teal"},
            {"id": "hall", "reflection_probe": {"extent": [8.0, 3.0, 6.0], "blend": 2.0},
                "transform": {"translation": [0.0, 3.0, 0.0], "scale": [2.0, 1.0, 1.0]}},
            {"id": "corner", "prefab": "kiosk", "transform": {"translation": [5.0, 0.0, 5.0]}}
        ],
        "environment": {"reflection_probes": {"enabled": true, "resolution": 256,
            "max_probes": 4, "faces_per_frame": 2}}
    });
    let document = SceneDocument::from_json(&source.to_string()).unwrap();
    let mut world = world(AssetBudget::default());
    let instance = document.instantiate(&mut world).unwrap();
    // The box keeps world axes: the entity's scale leaves its extent.
    let hall = *world
        .get::<ReflectionProbe>(instance.entity("hall").unwrap())
        .unwrap();
    assert_eq!(
        hall,
        ReflectionProbe::new(Vec3::new(8.0, 3.0, 6.0)).with_blend(2.0)
    );
    // Placements carry their prefab's probes; an omitted blend takes 1.
    let booth = *world
        .get::<ReflectionProbe>(instance.entity("corner.booth").unwrap())
        .unwrap();
    assert_eq!(booth, ReflectionProbe::new(Vec3::new(1.0, 1.5, 1.0)));
    assert_eq!(booth.blend, 1.0);
    assert!(
        world
            .get::<ReflectionProbe>(instance.entity("floor").unwrap())
            .is_none()
    );
    assert_eq!(
        *world.resource::<ReflectionProbes>(),
        ReflectionProbes {
            enabled: true,
            resolution: 256,
            max_probes: Some(4),
            faces_per_frame: 2
        }
    );
    // Probes and their settings round-trip as authored, and inspection
    // reproduces them.
    let saved = serde_json::to_value(&document).unwrap();
    assert_eq!(
        saved["entities"][1]["reflection_probe"],
        source["entities"][1]["reflection_probe"]
    );
    assert_eq!(saved["environment"], source["environment"]);
    assert_eq!(
        SceneReflectionProbe::from(&hall).runtime("$").unwrap(),
        hall
    );
    assert_eq!(
        SceneEnvironment::current(&world)
            .resolve("$")
            .unwrap()
            .reflection_probes,
        Some(*world.resource::<ReflectionProbes>())
    );
    // An environment without the section restores the engine's default.
    let mut bare = source.clone();
    bare["environment"] = serde_json::json!({});
    let mut world = self::world(AssetBudget::default());
    world.insert_resource(ReflectionProbes {
        enabled: false,
        ..ReflectionProbes::default()
    });
    let _ = SceneDocument::from_json(&bare.to_string())
        .unwrap()
        .instantiate(&mut world)
        .unwrap();
    assert_eq!(
        *world.resource::<ReflectionProbes>(),
        ReflectionProbes::default()
    );
    // A missing resource reads as off.
    world.remove_resource::<ReflectionProbes>();
    assert!(
        !SceneEnvironment::current(&world)
            .reflection_probes
            .unwrap()
            .enabled
    );
    let refused = |edit: &dyn Fn(&mut serde_json::Value)| {
        let mut source = source.clone();
        edit(&mut source);
        let error = SceneDocument::from_json(&source.to_string()).unwrap_err();
        (error.code, error.path)
    };
    for extent in [[0.0, 3.0, 6.0], [8.0, -1.0, 6.0]] {
        assert_eq!(
            refused(&|source| {
                source["entities"][1]["reflection_probe"]["extent"] = serde_json::json!(extent)
            }),
            (
                "invalid_reflection_probe",
                "$.entities[1].reflection_probe.extent".into()
            )
        );
    }
    assert_eq!(
        refused(&|source| {
            source["prefabs"][0]["entities"][0]["reflection_probe"]["blend"] =
                serde_json::json!(-0.5)
        }),
        (
            "invalid_reflection_probe",
            "$.prefabs[0].entities[0].reflection_probe.blend".into()
        )
    );
    assert_eq!(
        refused(&|source| source["entities"][1]["reflection_probe"]["radius"] = 2.into()).0,
        "invalid_json"
    );
    // Values JSON cannot carry are refused alike.
    for (probe, field) in [
        (
            SceneReflectionProbe::new([f32::INFINITY, 1.0, 1.0]),
            "extent",
        ),
        (
            SceneReflectionProbe {
                blend: f32::NAN,
                ..SceneReflectionProbe::new([1.0; 3])
            },
            "blend",
        ),
    ] {
        let mut invalid = document.clone();
        invalid.entities[1].reflection_probe = Some(probe);
        assert_eq!(
            invalid.validate().unwrap_err().path,
            format!("$.entities[1].reflection_probe.{field}")
        );
    }
    for (field, value) in [
        ("resolution", 100),
        ("resolution", 8),
        ("resolution", 2048),
        ("max_probes", 0),
        ("faces_per_frame", 0),
    ] {
        assert_eq!(
            refused(&|source| source["environment"]["reflection_probes"][field] = value.into()),
            (
                "invalid_environment",
                format!("$.environment.reflection_probes.{field}")
            ),
            "{field} {value}"
        );
    }
    // Earlier versions refuse probes, in placements and definitions alike,
    // and their settings.
    assert_eq!(
        refused(&|source| source["version"] = 7.into()),
        ("version_required", "$.entities[1].reflection_probe".into())
    );
    assert_eq!(
        refused(&|source| {
            source["version"] = 7.into();
            source["entities"].as_array_mut().unwrap().truncate(1);
        }),
        (
            "version_required",
            "$.prefabs[0].entities[0].reflection_probe".into()
        )
    );
    assert_eq!(
        refused(&|source| {
            source["version"] = 7.into();
            source["entities"].as_array_mut().unwrap().truncate(1);
            source.as_object_mut().unwrap().remove("prefabs");
        }),
        ("version_required", "$.environment.reflection_probes".into())
    );
}

/// The smallest valid virtual mesh file (`EVG1`, decision 044): one level
/// of one group holding one cluster of one triangle.
fn one_triangle_virtual_mesh() -> Vec<u8> {
    let bits = f32::to_bits;
    let mut words = vec![0u32; 32];
    words[0] = u32::from_le_bytes(*b"EVG1");
    words[1] = 2;
    // Levels, nodes, groups, clusters, pages, triangles (and source
    // triangles) and depth.
    words[3..11].copy_from_slice(&[1, 1, 1, 1, 1, 1, 1, 1]);
    // Positions quantized over the unit cube; bounds.
    for axis in 0..3 {
        words[15 + axis] = bits(1.0 / 65535.0);
        words[25 + axis] = bits(1.0);
    }
    // One pinned page (the group's error exceeds the pin error 0), ending
    // the prefix: the header, node, group, cluster, page table and page.
    words[28] = 1;
    words[29] = bits(0.0);
    words[30] = 32 + 8 + 8 + 12 + 2 + 13;
    let sphere = [bits(0.5), bits(0.5), bits(0.0), bits(0.8)];
    // The level's root: a leaf holding group 0, never simplified further.
    words.extend(sphere);
    words.extend([bits(f32::MAX), 0, 0, 0]);
    // Group 0: cluster 0, level 0.
    words.extend(sphere);
    words.extend([bits(f32::MAX), 0, 1, 0]);
    // Cluster 0: no cone, original geometry, group 0, page 0, vertices at
    // word 0 and triangles at word 12, 3 vertices and 1 triangle.
    words.extend(sphere);
    words.extend([u32::from_le_bytes([0, 0, 0, 127]), u32::MAX, 0, 0, 0, 12]);
    words.extend([3 | (1 << 16), 0]);
    // The page table, then the page.
    words.extend([0, 13]);
    for [x, y] in [[0u32, 0], [65535, 0], [0, 65535]] {
        words.extend([x | (y << 16), 0, 0, 0]);
    }
    words.push(u32::from_le_bytes([0, 1, 2, 0]));
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

#[test]
fn version_eight_virtual_meshes_load_from_files_and_earlier_versions_refuse_them() {
    use crate::{SceneFileKind, SceneFiles};
    let mut source: serde_json::Value = serde_json::from_str(SOURCE).unwrap();
    source["version"] = 8.into();
    source["meshes"][0]["source"] =
        serde_json::json!({"kind": "virtual", "uri": "meshes/rock.evg"});
    let document = SceneDocument::from_json(&source.to_string()).unwrap();
    // Structure validates without the file, which the document lists as a
    // virtual mesh.
    document.validate().unwrap();
    assert_eq!(
        document
            .files()
            .iter()
            .map(|file| (file.uri.as_str(), file.kind))
            .collect::<Vec<_>>(),
        [("meshes/rock.evg", SceneFileKind::VirtualMesh)]
    );
    let mut world = world(AssetBudget::default());
    let missing = document.instantiate(&mut world).unwrap_err();
    assert_eq!(
        (missing.code, missing.path.as_str()),
        ("missing_file", "$.meshes[0].source.uri")
    );
    // Hosts supply the file's prefix and URL; a whole file as a buffer
    // works too.
    let mut buffered = SceneFiles::new();
    buffered.add_buffer("meshes/rock.evg", one_triangle_virtual_mesh());
    document.validate_with(&buffered).unwrap();
    let mut files = SceneFiles::new();
    files.add_virtual_mesh(
        "meshes/rock.evg",
        one_triangle_virtual_mesh(),
        Some("https://example.test/meshes/rock.evg"),
    );
    let instance = document.instantiate_with(&mut world, &files).unwrap();
    let player = instance.entity("player").unwrap();
    let handle = world.get::<Mesh3d>(player).unwrap().mesh;
    let assets = world.resource::<AssetLibrary>();
    let mesh = assets.mesh(handle).unwrap();
    let geometry = mesh
        .virtual_geometry()
        .expect("the mesh keeps its hierarchy");
    assert_eq!((geometry.cluster_count(), mesh.indices().len()), (1, 3));
    assert_eq!(
        geometry.source(),
        Some("https://example.test/meshes/rock.evg")
    );
    instance.despawn(&mut world);
    // Broken files, empty fallbacks and earlier versions are refused.
    let mut broken = files.clone();
    broken.add_virtual_mesh("meshes/rock.evg", b"EVG1".to_vec(), None);
    let error = document.validate_with(&broken).unwrap_err();
    assert_eq!(
        (error.code, error.path.as_str()),
        ("invalid_mesh", "$.meshes[0].source.uri")
    );
    let mut empty = source.clone();
    empty["meshes"][0]["source"]["fallback_triangles"] = 0.into();
    let error = SceneDocument::from_json(&empty.to_string()).unwrap_err();
    assert_eq!(
        (error.code, error.path.as_str()),
        ("invalid_mesh", "$.meshes[0].source.fallback_triangles")
    );
    let mut old = source.clone();
    old["version"] = 7.into();
    let error = SceneDocument::from_json(&old.to_string()).unwrap_err();
    assert_eq!(
        (error.code, error.path.as_str()),
        ("version_required", "$.meshes[0].source")
    );
}

#[test]
fn version_eight_virtual_geometry_settings_resolve_and_earlier_versions_refuse_them() {
    use engine_core::VirtualGeometry;
    let mut source: serde_json::Value = serde_json::from_str(SOURCE).unwrap();
    source["version"] = 8.into();
    source["environment"] = serde_json::json!({"virtual_geometry": {"enabled": false, "pixel_error": 3.0, "max_resident_pages": 64}});
    let document = SceneDocument::from_json(&source.to_string()).unwrap();
    let mut world = world(AssetBudget::default());
    let instance = document.instantiate(&mut world).unwrap();
    assert_eq!(
        *world.resource::<VirtualGeometry>(),
        VirtualGeometry {
            enabled: false,
            pixel_error: 3.0,
            max_resident_pages: Some(64),
            visibility_buffer: true,
        }
    );
    instance.despawn(&mut world);
    source["environment"]["virtual_geometry"]["max_resident_pages"] = 0.into();
    let error = SceneDocument::from_json(&source.to_string()).unwrap_err();
    assert_eq!(
        error.path.as_str(),
        "$.environment.virtual_geometry.max_resident_pages"
    );
    source["environment"]["virtual_geometry"]["max_resident_pages"] = 64.into();
    source["environment"]["virtual_geometry"]["pixel_error"] = 100.0.into();
    let error = SceneDocument::from_json(&source.to_string()).unwrap_err();
    assert_eq!(
        (error.code, error.path.as_str()),
        (
            "invalid_environment",
            "$.environment.virtual_geometry.pixel_error"
        )
    );
    source["environment"]["virtual_geometry"]["pixel_error"] = 3.0.into();
    source["version"] = 7.into();
    let error = SceneDocument::from_json(&source.to_string()).unwrap_err();
    assert_eq!(
        (error.code, error.path.as_str()),
        ("version_required", "$.environment.virtual_geometry")
    );
}
