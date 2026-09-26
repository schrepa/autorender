use engine_collision::{Collider, CollisionConfig, CollisionPlugin, CollisionWorld, SensorEvents};
use engine_core::{
    ActionId, ActionInput, Engine, FixedSimulation, FixedTime, FrameTime, InterpolateTransform,
    PresentationMode, PresentedTransforms, RenderSnapshot, Transform, TransformEdits,
    bevy_ecs::{self, prelude::*},
    edit_transform,
    glam::Vec3,
};
use engine_scene::{SceneDocument, SceneInstance};
use serde_json::{Value, json};

use crate::DevTools;

const SOURCE: &str = r#"{
  "version":1,"id":"collision-test","revision":1,
  "materials":[{"id":"white","color":[1,1,1,1]}],
  "meshes":[{"id":"box","source":{"kind":"cube"}}],
  "entities":[
    {"id":"target","mesh":"box","material":"white"},
    {"id":"sensor","mesh":"box","material":"white","transform":{"translation":[0,0,4]}}
  ]
}"#;

fn fixture() -> (Engine, SceneInstance) {
    let mut engine = Engine::default();
    engine.app.add_plugins(CollisionPlugin);
    let scene = SceneDocument::from_json(SOURCE)
        .unwrap()
        .instantiate(engine.app.world_mut())
        .unwrap();
    engine
        .app
        .world_mut()
        .entity_mut(scene.entity("target").unwrap())
        .insert(Collider::cuboid(Vec3::ONE).unwrap());
    engine
        .app
        .world_mut()
        .entity_mut(scene.entity("sensor").unwrap())
        .insert(Collider::sphere(1.0).unwrap().with_sensor(true));
    (engine, scene)
}

fn execute(engine: &mut Engine, scene: Option<&SceneInstance>, mut request: Value) -> Value {
    request["version"] = json!(1);
    serde_json::from_str(&DevTools::default().execute(engine, scene, &request.to_string())).unwrap()
}

fn ray() -> Value {
    json!({"op":"raycast","origin":[0,0,10],"direction":[0,0,-4],"max_distance":20})
}

#[test]
fn spatial_queries_are_discoverable_and_return_complete_authoritative_hits() {
    let (mut engine, scene) = fixture();
    let description = execute(
        &mut engine,
        Some(&scene),
        json!({"op":"describe","name":"protocol"}),
    );
    let variants = description["result"]["schema"]["oneOf"].as_array().unwrap();
    for op in ["raycast", "overlap", "shape_cast", "collision_events"] {
        assert!(
            variants
                .iter()
                .any(|variant| variant["properties"]["op"]["const"] == op)
        );
    }
    let first = execute(&mut engine, Some(&scene), ray());
    assert_eq!(first["ok"], true, "{first}");
    assert_eq!(first["result"]["hit"]["entity"]["id"], "sensor");
    assert_eq!(first["result"]["hit"]["distance"], 5.0);
    assert_eq!(first["result"]["complete"], true);
    assert_eq!(first["result"]["snapshot"]["complete"], true);
    assert_eq!(first["result"]["snapshot"]["active_colliders"], 2);
    let mut filtered = ray();
    filtered["filter"] = json!({"sensors":"exclude"});
    let hit = execute(&mut engine, Some(&scene), filtered);
    assert_eq!(hit["result"]["hit"]["entity"]["id"], "target");
    assert_eq!(hit["result"]["hit"]["distance"], 9.0);
    let overlap = execute(
        &mut engine,
        Some(&scene),
        json!({"op":"overlap",
        "shape":{"kind":"sphere","radius":0.5},"position":[0,0,0],"limit":10}),
    );
    assert_eq!(overlap["ok"], true, "{overlap}");
    assert_eq!(overlap["result"]["total"], 1);
    assert_eq!(overlap["result"]["hits"][0]["id"], "target");
    let cast = execute(
        &mut engine,
        Some(&scene),
        json!({"op":"shape_cast",
        "shape":{"kind":"sphere","radius":0.5},"position":[0,0,10],
        "direction":[0,0,-1],"max_distance":20,"filter":{"sensors":"exclude"}}),
    );
    assert_eq!(cast["ok"], true, "{cast}");
    assert_eq!(cast["result"]["hit"]["entity"]["id"], "target");
    assert!((cast["result"]["hit"]["distance"].as_f64().unwrap() - 8.5).abs() < 0.01);
    let collider = execute(
        &mut engine,
        Some(&scene),
        json!({"op":"inspect","entity":"sensor","component":"Collider"}),
    );
    assert_eq!(
        collider["result"]["entity"]["components"]["Collider"]["sensor"],
        true
    );
    assert_eq!(
        collider["result"]["entity"]["components"]["Collider"]["cache_status"],
        "active"
    );
    assert_eq!(
        collider["result"]["entity"]["components"]["Collider"]["world_valid"],
        true
    );
    assert_eq!(engine.app.world().resource::<FixedTime>().tick, 0);
    assert_eq!(engine.app.world().resource::<FrameTime>().frame, 0);
}

#[derive(Resource, Default)]
struct JumpEdges(u32);

#[test]
fn queries_ignore_presented_poses_and_preserve_input_events_and_extraction_identity() {
    let (mut engine, scene) = fixture();
    let target = scene.entity("target").unwrap();
    engine
        .app
        .world_mut()
        .entity_mut(target)
        .insert(InterpolateTransform);
    engine.app.init_resource::<JumpEdges>();
    engine.app.add_systems(
        FixedSimulation,
        (
            |transforms: Query<(Entity, &Transform), With<InterpolateTransform>>,
             mut edits: TransformEdits| {
                for (entity, transform) in &transforms {
                    edits
                        .edit(entity, transform, |pose| pose.translation.x += 8.0)
                        .unwrap();
                }
            },
            |input: Res<ActionInput>, mut edges: ResMut<JumpEdges>| {
                if input.just_pressed(ActionId::JUMP) {
                    edges.0 += 1;
                }
            },
        ),
    );
    engine.advance(1.0 / 60.0);
    engine
        .extract_with_presentation(
            1.0,
            &mut RenderSnapshot::default(),
            PresentationMode::Interpolated { alpha: 0.5 },
        )
        .unwrap();
    let presented = engine
        .app
        .world()
        .resource::<PresentedTransforms>()
        .get(target)
        .unwrap()
        .matrix();
    let sequence = engine.app.world().resource::<SensorEvents>().sequence();
    engine
        .app
        .world_mut()
        .resource_mut::<ActionInput>()
        .ingest(1, 1, 0);
    let mut request = ray();
    request["origin"] = json!([8, 0, 10]);
    let hit = execute(&mut engine, Some(&scene), request);
    assert_eq!(hit["result"]["hit"]["entity"]["id"], "target");
    assert_eq!(hit["observation"]["presentation"]["mode"], "interpolated");
    assert_eq!(hit["observation"]["presentation"]["alpha"], 0.5);
    assert_eq!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get(target)
            .unwrap()
            .matrix(),
        presented
    );
    assert_eq!(engine.app.world().resource::<FixedTime>().tick, 1);
    assert_eq!(engine.app.world().resource::<FrameTime>().frame, 1);
    assert_eq!(
        engine.app.world().resource::<SensorEvents>().sequence(),
        sequence
    );
    let mut displayed_ray = ray();
    displayed_ray["origin"] = json!([4, 0, 10]);
    let miss = execute(&mut engine, Some(&scene), displayed_ray);
    assert!(miss["result"]["hit"].is_null(), "{miss}");
    engine.advance(1.0 / 60.0);
    assert_eq!(engine.app.world().resource::<JumpEdges>().0, 1);
}

#[test]
fn invalid_queries_leave_cache_events_and_clocks_unchanged() {
    let (mut engine, scene) = fixture();
    assert_eq!(execute(&mut engine, Some(&scene), ray())["ok"], true);
    let revision = engine
        .app
        .world()
        .resource::<CollisionWorld>()
        .report()
        .revision;
    let sequence = engine.app.world().resource::<SensorEvents>().sequence();
    for request in [
        json!({"op":"raycast","origin":[0,0,0],"direction":[0,0,0],"max_distance":1}),
        json!({"op":"raycast","origin":[1000000.000001,0,0],"direction":[1,0,0],"max_distance":1}),
        json!({"op":"raycast","origin":[0,0,0],"direction":[1,0,0],"max_distance":-0.0000000001}),
        json!({"op":"raycast","origin":[0,0,0],"direction":[1,0,0],"max_distance":1,"filter":{"exclude":"missing"}}),
        json!({"op":"raycast","origin":[0,0,0],"direction":[1,0,0],"max_distance":1,"filter":{"extra":true}}),
        json!({"op":"raycast","origin":[0,0,0],"direction":[1,0,0],"max_distance":null}),
        json!({"op":"raycast","origin":[0,0,0],"direction":[1,0,0],"max_distance":1,"filter":null}),
        json!({"op":"raycast","origin":[0,0,0],"direction":[1,0,0],"max_distance":1,"filter":{"sensors":null}}),
        json!({"op":"raycast","origin":[0,0,0],"direction":[1,0,0],"max_distance":1,"filter":{"memberships":4294967296_u64}}),
        json!({"op":"overlap","shape":{"kind":"sphere","radius":0},"position":[0,0,0]}),
        json!({"op":"overlap","shape":{"kind":"sphere","radius":null},"position":[0,0,0]}),
        json!({"op":"overlap","shape":{"kind":"sphere","radius":1},"position":[0,0,0],"rotation":null}),
        json!({"op":"overlap","shape":{"kind":"sphere","radius":10000.000001},"position":[0,0,0]}),
        json!({"op":"overlap","shape":{"kind":"sphere","radius":1},"position":[0,0,0],"limit":101}),
        json!({"op":"shape_cast","shape":{"kind":"sphere","radius":1},"position":[0,0,0],"rotation":[0,0,0,0],"direction":[1,0,0],"max_distance":1}),
        json!({"op":"collision_events","limit":0}),
        json!({"op":"collision_events","offset":131073}),
        json!({"op":"collision_events","limit":null}),
        json!({"op":"inspect","component":"CollisionWorld","limit":0}),
    ] {
        let response = execute(&mut engine, Some(&scene), request.clone());
        assert_eq!(
            response["ok"], false,
            "request={request}, response={response}"
        );
        assert!(response["error"]["code"].is_string());
        assert_eq!(
            engine
                .app
                .world()
                .resource::<CollisionWorld>()
                .report()
                .revision,
            revision
        );
        assert_eq!(
            engine.app.world().resource::<SensorEvents>().sequence(),
            sequence
        );
        assert_eq!(engine.app.world().resource::<FixedTime>().tick, 0);
        assert_eq!(engine.app.world().resource::<FrameTime>().frame, 0);
    }
    let missing = execute(&mut Engine::default(), None, ray());
    assert_eq!(missing["ok"], false);
    assert_eq!(
        missing["error"]["code"],
        engine_collision::CollisionError::MissingPlugin.code()
    );
}

#[test]
fn overflow_never_returns_partial_hits_and_runtime_entities_remain_non_addressable() {
    let (mut engine, scene) = fixture();
    engine.app.world_mut().spawn((
        Transform {
            translation: Vec3::new(0.0, 0.0, 8.0),
            ..Transform::default()
        },
        Collider::sphere(0.5).unwrap(),
    ));
    let unknown = execute(&mut engine, Some(&scene), ray());
    assert_eq!(unknown["ok"], true, "{unknown}");
    assert!(unknown["result"]["hit"]["entity"]["id"].is_null());
    assert_eq!(unknown["result"]["hit"]["entity"]["in_active_scene"], false);
    let overlap = execute(
        &mut engine,
        Some(&scene),
        json!({"op":"overlap",
        "shape":{"kind":"sphere","radius":20},"position":[0,0,0],"limit":1}),
    );
    assert_eq!(overlap["ok"], false);
    assert!(overlap.get("result").is_none());
    assert_eq!(
        overlap["error"]["code"],
        engine_collision::CollisionError::ResultBudget.code()
    );
    engine
        .app
        .world_mut()
        .resource_mut::<CollisionConfig>()
        .max_query_candidates = 0;
    let budget = execute(&mut engine, Some(&scene), ray());
    assert_eq!(budget["ok"], false);
    assert!(budget.get("result").is_none());
}

#[test]
fn sensor_events_are_non_consuming_and_keep_removed_authored_identity() {
    let (mut engine, scene) = fixture();
    let sensor = scene.entity("sensor").unwrap();
    edit_transform(engine.app.world_mut(), sensor, |pose| {
        pose.translation = Vec3::ZERO;
    })
    .unwrap();
    engine.advance(1.0 / 60.0);
    let request = json!({"op":"collision_events","entity":"sensor","limit":1});
    let enter = execute(&mut engine, Some(&scene), request.clone());
    assert_eq!(enter["ok"], true, "{enter}");
    assert_eq!(enter["result"]["events"][0]["kind"], "enter");
    assert_eq!(enter["result"]["tick"], "1");
    let again = execute(&mut engine, Some(&scene), request.clone());
    assert_eq!(again["result"], enter["result"]);
    assert_eq!(execute(&mut engine, Some(&scene), ray())["ok"], true);
    let cache = execute(
        &mut engine,
        Some(&scene),
        json!({"op":"inspect","component":"CollisionWorld"}),
    );
    assert_eq!(cache["ok"], true, "{cache}");
    assert_eq!(
        execute(&mut engine, Some(&scene), request.clone())["result"],
        enter["result"]
    );
    engine.app.world_mut().despawn(sensor);
    engine.advance(1.0 / 60.0);
    let exit = execute(&mut engine, Some(&scene), request);
    assert_eq!(exit["result"]["events"][0]["kind"], "exit");
    let pair = &exit["result"]["events"][0];
    let removed = if pair["first"]["id"] == "sensor" {
        &pair["first"]
    } else {
        &pair["second"]
    };
    assert_eq!(removed["id"], "sensor");
    assert_eq!(removed["in_active_scene"], true);
    assert_eq!(removed["alive"], false);
}

#[test]
fn incomplete_sensor_tick_retains_complete_pairs_and_inspection_does_not_commit_events() {
    let (mut engine, scene) = fixture();
    let sensor = scene.entity("sensor").unwrap();
    edit_transform(engine.app.world_mut(), sensor, |pose| {
        pose.translation = Vec3::ZERO;
    })
    .unwrap();
    engine.advance(1.0 / 60.0);
    let events = json!({"op":"collision_events","entity":null});
    let first = execute(&mut engine, Some(&scene), events.clone());
    assert_eq!(first["result"]["active_pairs"], 1);
    engine
        .app
        .world_mut()
        .resource_mut::<CollisionConfig>()
        .max_candidate_pairs = 0;
    engine.advance(1.0 / 60.0);
    let incomplete = execute(&mut engine, Some(&scene), events.clone());
    assert_eq!(incomplete["result"]["complete"], false);
    assert_eq!(incomplete["result"]["tick"], "2");
    assert_eq!(incomplete["result"]["committed_tick"], "1");
    assert_eq!(incomplete["result"]["active_pairs"], 1);
    assert_eq!(incomplete["result"]["events"], json!([]));
    assert!(incomplete["result"]["error"]["code"].is_string());
    edit_transform(engine.app.world_mut(), sensor, |pose| {
        pose.translation = Vec3::new(0.0, 0.0, 4.0);
    })
    .unwrap();
    let inspect = execute(
        &mut engine,
        Some(&scene),
        json!({"op":"inspect","component":"CollisionWorld"}),
    );
    assert_eq!(inspect["ok"], true, "{inspect}");
    assert_eq!(
        execute(&mut engine, Some(&scene), events.clone())["result"],
        incomplete["result"]
    );
    engine
        .app
        .world_mut()
        .resource_mut::<CollisionConfig>()
        .max_candidate_pairs = 100;
    engine.advance(1.0 / 60.0);
    let recovered = execute(&mut engine, Some(&scene), events);
    assert_eq!(recovered["result"]["complete"], true);
    assert_eq!(recovered["result"]["tick"], "3");
    assert_eq!(recovered["result"]["committed_tick"], "3");
    assert_eq!(recovered["result"]["events"][0]["kind"], "exit");
}

#[test]
fn layer_filtering_checks_both_masks_and_optional_null_only_omits_selection() {
    let (mut engine, scene) = fixture();
    let target = scene.entity("target").unwrap();
    let collider = *engine.app.world().get::<Collider>(target).unwrap();
    engine
        .app
        .world_mut()
        .entity_mut(target)
        .insert(collider.with_layers(engine_collision::CollisionLayers::new(2, 4)));
    let mut request = ray();
    for (memberships, filter, hit) in [(1, 2, false), (4, 1, false), (4, 2, true)] {
        request["filter"] =
            json!({"memberships":memberships,"filter":filter,"sensors":"exclude","exclude":null});
        let response = execute(&mut engine, Some(&scene), request.clone());
        assert_eq!(response["ok"], true, "{response}");
        assert_eq!(!response["result"]["hit"].is_null(), hit);
    }
    request["filter"]["exclude"] = json!("target");
    assert!(execute(&mut engine, Some(&scene), request)["result"]["hit"].is_null());
}
