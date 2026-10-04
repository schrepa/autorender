use engine_collision::{Collider, ColliderShape, validate_collider_transform};
use engine_core::{
    AssetBudget, AssetLibrary, Camera, GlobalTransforms, Mesh3d, Transform, bevy_ecs::world::World,
};
use serde_json::{Value, json};

use crate::{SceneColliderShape, SceneDocument};

fn source() -> Value {
    json!({"version":4,"id":"collision-fixture","revision":1,
    "materials":[],"meshes":[],"entities":[
        {"id":"parent","transform":{"scale":[2,1,0.5]}},
        {"id":"child","parent":"parent","transform":{"rotation_degrees":[0,35,0]},
         "collider":{"shape":{"kind":"cuboid","half_extents":[0.4,0.3,0.2]},"memberships":2,"filter":5,"sensor":true}}
    ]})
}

fn document(value: Value) -> SceneDocument {
    SceneDocument::from_json(&value.to_string()).unwrap()
}

fn world() -> World {
    let mut world = World::new();
    world.insert_resource(AssetLibrary::new(AssetBudget::default()));
    world.insert_resource(Camera::default());
    world
}

#[test]
fn collider_round_trip_defaults_and_legacy_versions_are_explicit() {
    let mut value = source();
    value["entities"] =
        json!([{"id":"sensor","collider":{"shape":{"kind":"sphere","radius":0.5}}}]);
    let scene = document(value.clone());
    let collider = scene.entities[0].collider.unwrap();
    assert_eq!(collider.memberships, 1);
    assert_eq!(collider.filter, u32::MAX);
    assert!(!collider.sensor);
    let text = scene.to_json().unwrap();
    assert_eq!(
        document(serde_json::from_str(&text).unwrap())
            .to_json()
            .unwrap(),
        text
    );
    for version in [1, 2, 3] {
        value["version"] = json!(version);
        let error = SceneDocument::from_json(&value.to_string()).unwrap_err();
        assert_eq!(
            (error.code, error.path.as_str()),
            ("version_required", "$.entities[0].collider")
        );
        let legacy = json!({"version":version,"id":"old","revision":1,"materials":[],"meshes":[],"entities":[]});
        assert!(document(legacy).entities.is_empty());
    }
    let mut unused = source();
    unused["version"] = json!(3);
    unused["entities"] = json!([]);
    unused["prefabs"] = json!([{"id":"unused","entities":[{"id":"sensor","collider":{"shape":{"kind":"sphere","radius":1}}}]}]);
    assert_eq!(
        SceneDocument::from_json(&unused.to_string())
            .unwrap_err()
            .code,
        "version_required"
    );
}

#[test]
fn shape_layers_and_unknown_fields_have_strict_admission() {
    for invalid in [
        json!({"shape":{"kind":"sphere","radius":0}}),
        json!({"shape":{"kind":"sphere","radius":10001}}),
        json!({"shape":{"kind":"cuboid","half_extents":[1,-1,1]}}),
        json!({"shape":{"kind":"cuboid","half_extents":[1,0.00001,1]}}),
    ] {
        let mut value = source();
        value["entities"][1]["collider"] = invalid;
        let error = SceneDocument::from_json(&value.to_string()).unwrap_err();
        assert_eq!(error.code, "invalid_collider");
        assert_eq!(error.path, "$.entities[1].collider.shape");
    }
    for invalid in [
        json!({"shape":{"kind":"sphere","radius":1,"typo":true}}),
        json!({"shape":{"kind":"capsule","radius":1}}),
        json!({"shape":{"kind":"sphere","radius":1},"memberships":4294967296u64}),
        json!({"shape":{"kind":"sphere","radius":1},"filter":-1}),
        json!({"shape":{"kind":"sphere","radius":1},"sensor":1}),
        json!({"shape":{"kind":"sphere","radius":1},"offset":[0,1,0]}),
    ] {
        let mut value = source();
        value["entities"][1]["collider"] = invalid;
        assert_eq!(
            SceneDocument::from_json(&value.to_string())
                .unwrap_err()
                .code,
            "invalid_json"
        );
    }
    let mut scene = document(source());
    scene.entities[1].collider.as_mut().unwrap().shape =
        SceneColliderShape::Sphere { radius: f32::NAN };
    assert_eq!(scene.validate().unwrap_err().code, "invalid_collider");
    scene.entities[1].collider.as_mut().unwrap().shape = SceneColliderShape::Sphere { radius: 1.0 };
    scene.entities[1].collider.as_mut().unwrap().memberships = 0;
    scene.entities[1].collider.as_mut().unwrap().filter = 0;
    scene.validate().unwrap();
}

#[test]
fn affine_colliders_on_nonrendering_prefab_nodes_instantiate_and_dispose() {
    let mut value = source();
    let mut child = value["entities"][1].clone();
    child.as_object_mut().unwrap().remove("parent");
    value["prefabs"] = json!([{"id":"shape","entities":[child]}]);
    value["entities"] = json!([
        {"id":"west","prefab":"shape","transform":{"translation":[-2,0,0],"scale":[2,1,0.5]},"collider":{"shape":{"kind":"sphere","radius":0.2}}},
        {"id":"east","prefab":"shape","transform":{"translation":[2,0,0]}}
    ]);
    let scene = document(value);
    let mut world = world();
    let outside = world
        .spawn((Transform::default(), Collider::sphere(0.5).unwrap()))
        .id();
    let instance = scene.instantiate(&mut world).unwrap();
    assert_eq!(instance.entity_count(), 4);
    let child = instance.entity("west.child").unwrap();
    assert!(world.get::<Mesh3d>(child).is_none());
    let collider = *world.get::<Collider>(child).unwrap();
    assert!(collider.is_sensor());
    assert_eq!(
        (collider.layers().memberships, collider.layers().filter),
        (2, 5)
    );
    let matrix = world
        .resource::<GlobalTransforms>()
        .get(child)
        .unwrap()
        .matrix()
        .unwrap();
    assert!(
        matrix.x_axis.truncate().dot(matrix.z_axis.truncate()).abs() > 0.1,
        "Fixture must have inherited shear."
    );
    validate_collider_transform(collider.shape(), matrix).unwrap();
    let root = instance.entity("west").unwrap();
    assert_eq!(
        world.get::<Collider>(root).unwrap().shape(),
        ColliderShape::Sphere { radius: 0.2 }
    );
    assert_eq!(world.resource::<AssetLibrary>().resident_bytes(), 0);
    instance.despawn(&mut world);
    assert!(world.get_entity(child).is_err());
    assert!(world.get::<Collider>(outside).is_some());
}

#[test]
fn composed_collision_domain_rejects_before_world_mutation() {
    let good = document(source());
    let mut world = world();
    let live = good.instantiate(&mut world).unwrap();
    let player = live.entity("child").unwrap();
    let before_camera = *world.resource::<Camera>();
    let before_count = world.entities().len();
    let before_matrix = world
        .resource::<GlobalTransforms>()
        .get(player)
        .unwrap()
        .matrix();
    let mut invalid = good.clone();
    invalid.entities[0].transform.translation = [1_000_000.0, 0.0, 0.0];
    invalid.entities[1].transform.translation = [1.0, 0.0, 0.0];
    let error = invalid.instantiate(&mut world).unwrap_err();
    assert_eq!(error.code, "invalid_collider_transform");
    assert_eq!(error.path, "$.entities[1].collider");
    assert_eq!(world.entities().len(), before_count);
    assert_eq!(
        world
            .resource::<GlobalTransforms>()
            .get(player)
            .unwrap()
            .matrix(),
        before_matrix
    );
    assert_eq!(world.resource::<Camera>().position, before_camera.position);
    assert_eq!(world.resource::<AssetLibrary>().resident_bytes(), 0);
}

#[test]
fn unused_definitions_and_placed_scales_use_the_same_collision_bounds() {
    let mut value = source();
    value["entities"] = json!([]);
    value["prefabs"] = json!([{"id":"shape","entities":[{"id":"sphere","collider":{"shape":{"kind":"sphere","radius":2}}}]}]);
    let good = document(value.clone());
    let mut invalid = good.clone();
    invalid.prefabs[0].entities[0].transform.scale = [10000.0; 3];
    assert_eq!(
        invalid.validate().unwrap_err().code,
        "invalid_collider_transform"
    );
    value["entities"] =
        json!([{"id":"large","prefab":"shape","transform":{"scale":[10000,10000,10000]}}]);
    let error = SceneDocument::from_json(&value.to_string()).unwrap_err();
    assert_eq!(error.code, "invalid_collider_transform");
    assert_eq!(error.path, "$.prefabs[0].entities[0].collider");
}

#[cfg(feature = "schema")]
#[test]
fn schema_exposes_collider_shape_masks_and_version_four() {
    let schema: Value = serde_json::from_str(&SceneDocument::schema_json()).unwrap();
    assert_eq!(
        schema["properties"]["version"]["maximum"],
        crate::SCENE_VERSION
    );
    let collider = &schema["$defs"]["SceneCollider"];
    assert_eq!(collider["additionalProperties"], false);
    assert_eq!(collider["properties"]["memberships"]["default"], 1);
    assert_eq!(collider["properties"]["filter"]["default"], u32::MAX);
    assert_eq!(collider["properties"]["sensor"]["default"], false);
    assert!(schema["$defs"]["SceneObject"]["properties"]["collider"].is_object());
    assert_eq!(
        schema["$defs"]["SceneColliderShape"]["oneOf"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}
