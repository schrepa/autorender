use engine_core::{
    Engine, FixedTime, FrameTime, Transform,
    bevy_ecs::{self, prelude::*},
    edit_transform,
};
use engine_scene::SceneDocument;
use serde_json::{Value, json};

use crate::{DevTools, InspectorDescriptor, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES};

const SOURCE: &str = r#"{
  "version": 1, "id": "test-scene", "revision": 4,
  "materials": [{"id": "teal", "color": [0.1, 0.8, 0.6, 1.0]}],
  "meshes": [{"id": "box", "source": {"kind": "cube"}}],
  "entities": [{"id": "player", "name": "Player", "mesh": "box", "material": "teal", "tags": ["player"]}]
}"#;

#[test]
fn benchmark_defaults_and_temporary_camera_validation_are_explicit() {
    let command: crate::BrowserControl =
        serde_json::from_value(json!({"version":1,"op":"benchmark"})).unwrap();
    assert!(matches!(
        command,
        crate::BrowserControl::Benchmark {
            warmup: 30,
            samples: 120,
            mode: crate::BenchmarkMode::Step,
            ..
        }
    ));
    for request in [
        json!({"version":1,"op":"benchmark","mode":"frames"}),
        json!({"version":1,"op":"culling","enabled":1}),
        json!({"version":1,"op":"culling","enabled":true,"extra":0}),
    ] {
        assert!(serde_json::from_value::<crate::BrowserControl>(request).is_err());
    }
    assert!(crate::validated_camera([0.0, 1.0, 10.0], [0.0; 3], 1.0, 0.1, 500.0).is_ok());
    for (position, target, fov, near, far) in [
        ([0.0; 3], [0.0; 3], 1.0, 0.1, 500.0),
        ([f32::NAN, 0.0, 0.0], [0.0; 3], 1.0, 0.1, 500.0),
        ([0.0, 1.0, 10.0], [0.0; 3], f32::INFINITY, 0.1, 500.0),
        ([0.0, 1.0, 10.0], [0.0; 3], 1.0, 500.0, 500.0),
        ([0.0, 1.0, 10.0], [0.0; 3], 1.0, 0.0, 500.0),
    ] {
        assert_eq!(
            crate::validated_camera(position, target, fov, near, far)
                .unwrap_err()
                .code,
            "invalid_camera"
        );
    }
}

fn execute(
    tools: &DevTools,
    engine: &mut Engine,
    scene: Option<&engine_scene::SceneInstance>,
    request: Value,
) -> Value {
    serde_json::from_str(&tools.execute(engine, scene, &request.to_string())).unwrap()
}

#[test]
fn presentation_controls_validate_before_rounding_and_publish_their_schema() {
    for alpha in [0.0, 0.25, 0.5, 1.0] {
        assert_eq!(
            crate::validated_presentation_alpha(alpha).unwrap(),
            alpha as f32
        );
        let request = serde_json::from_value::<crate::BrowserControl>(
            json!({"version":1,"op":"present","alpha":alpha}),
        )
        .unwrap();
        assert_eq!(request.version(), 1);
        assert!(
            matches!(request, crate::BrowserControl::Present { alpha: value, .. } if value == alpha)
        );
    }
    for alpha in [
        -0.001,
        -f64::EPSILON,
        1.0 + f64::EPSILON,
        2.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let error = crate::validated_presentation_alpha(alpha).unwrap_err();
        assert_eq!(error.code, "invalid_presentation_alpha");
        assert_eq!(error.path, "$.alpha");
    }
    for source in [
        json!({"version":1,"op":"present"}),
        json!({"version":1,"op":"present","alpha":"0.5"}),
        json!({"version":1,"op":"present","alpha":0.5,"extra":true}),
        json!({"version":1,"op":"capture","alpha":"0.5"}),
        json!({"version":1,"op":"capture","alpha":null}),
        json!({"version":1,"op":"present","alpha":null}),
    ] {
        assert!(serde_json::from_value::<crate::BrowserControl>(source).is_err());
    }
    let capture: crate::BrowserControl =
        serde_json::from_value(json!({"version":1,"op":"capture"})).unwrap();
    assert!(matches!(
        capture,
        crate::BrowserControl::Capture { alpha: None, .. }
    ));
    let capture: crate::BrowserControl =
        serde_json::from_value(json!({"version":1,"op":"capture","alpha":0.25})).unwrap();
    assert!(matches!(
        capture,
        crate::BrowserControl::Capture {
            alpha: Some(0.25),
            ..
        }
    ));
    let benchmark: crate::BrowserControl =
        serde_json::from_value(json!({"version":1,"op":"benchmark","mode":"present"})).unwrap();
    assert!(matches!(
        benchmark,
        crate::BrowserControl::Benchmark {
            mode: crate::BenchmarkMode::Present,
            ..
        }
    ));

    let metadata = execute(
        &DevTools::default(),
        &mut Engine::default(),
        None,
        json!({"version":1,"op":"describe","name":"BrowserControl"}),
    );
    let variants = metadata["result"]["schema"]["oneOf"].as_array().unwrap();
    let present = variants
        .iter()
        .find(|variant| variant["properties"]["op"]["const"] == "present")
        .unwrap();
    assert_eq!(present["properties"]["alpha"]["minimum"], 0.0);
    assert_eq!(present["properties"]["alpha"]["maximum"], 1.0);
    assert!(
        present["required"]
            .as_array()
            .unwrap()
            .contains(&json!("alpha"))
    );
    let capture = variants
        .iter()
        .find(|variant| variant["properties"]["op"]["const"] == "capture")
        .unwrap();
    assert!(
        !capture["required"]
            .as_array()
            .unwrap()
            .contains(&json!("alpha"))
    );
    assert_eq!(capture["properties"]["alpha"]["type"], "number");
    assert_eq!(capture["properties"]["alpha"]["minimum"], 0.0);
    assert_eq!(capture["properties"]["alpha"]["maximum"], 1.0);
}

#[test]
fn presented_inspection_preserves_last_extraction_and_authoritative_world() {
    use engine_core::{FixedSimulation, InterpolateTransform, PresentationMode, RenderSnapshot};

    let mut source: Value = serde_json::from_str(SOURCE).unwrap();
    source["version"] = json!(3);
    source["entities"].as_array_mut().unwrap().push(json!({
        "id":"child", "parent":"player", "mesh":"box", "material":"teal",
        "transform":{"translation":[2.0,0.0,0.0]}
    }));
    let mut engine = Engine::default();
    let scene = SceneDocument::from_json(&source.to_string())
        .unwrap()
        .instantiate(engine.app.world_mut())
        .unwrap();
    let player = scene.entity("player").unwrap();
    engine
        .app
        .world_mut()
        .entity_mut(player)
        .insert(InterpolateTransform);
    engine.app.add_systems(
        FixedSimulation,
        |mut transforms: Query<&mut Transform, With<InterpolateTransform>>| {
            for mut transform in &mut transforms {
                transform.translation.x += 4.0;
            }
        },
    );
    let tools = DevTools::default();
    let descriptor = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"describe"}),
    );
    for name in [
        "InterpolateTransform",
        "PresentedTransform",
        "PresentationState",
    ] {
        assert!(
            descriptor["result"]["types"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["name"] == name)
        );
        let metadata = execute(
            &tools,
            &mut engine,
            Some(&scene),
            json!({"version":1,"op":"describe","name":name}),
        );
        assert!(metadata["result"]["descriptor"]["schema"]["properties"].is_object());
    }
    let request =
        json!({"version":1,"op":"inspect","entity":"child","component":"PresentedTransform"});
    let unrendered = execute(&tools, &mut engine, Some(&scene), request.clone());
    assert_eq!(
        unrendered["result"]["entity"]["components"]["PresentedTransform"],
        json!({"valid":false,"matrix":null,"translation":null})
    );
    assert!(unrendered["observation"]["presentation"]["frame"].is_null());
    assert!(unrendered["observation"]["presentation"]["tick"].is_null());
    let marker = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"inspect","entity":"player","component":"InterpolateTransform"}),
    );
    assert_eq!(
        marker["result"]["entity"]["components"]["InterpolateTransform"],
        json!({"enabled":true})
    );
    let unmarked = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"inspect","entity":"child","component":"InterpolateTransform"}),
    );
    assert_eq!(unmarked["error"]["code"], "missing_component");

    assert_eq!(engine.advance(1.0 / 60.0).fixed_steps, 1);
    let mut snapshot = RenderSnapshot::default();
    engine
        .extract_with_presentation(
            1.0,
            &mut snapshot,
            PresentationMode::Interpolated { alpha: 0.25 },
        )
        .unwrap();
    let preview = execute(&tools, &mut engine, Some(&scene), request.clone());
    assert_eq!(
        preview["result"]["entity"]["components"]["PresentedTransform"]["translation"],
        json!([3.0, 0.0, 0.0])
    );
    assert_eq!(
        preview["observation"]["presentation"]["mode"],
        "interpolated"
    );
    assert_eq!(preview["observation"]["presentation"]["alpha"], 0.25);
    assert_eq!(preview["observation"]["presentation"]["frame"], "1");
    assert_eq!(preview["observation"]["presentation"]["tick"], "1");
    let state = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"inspect","component":"PresentationState"}),
    );
    assert_eq!(
        state["result"]["value"],
        preview["observation"]["presentation"]
    );

    let global_request =
        json!({"version":1,"op":"inspect","entity":"child","component":"GlobalTransform"});
    let authoritative = execute(&tools, &mut engine, Some(&scene), global_request.clone());
    assert_eq!(
        authoritative["result"]["entity"]["components"]["GlobalTransform"]["translation"],
        json!([6.0, 0.0, 0.0])
    );
    assert_eq!(authoritative["observation"], preview["observation"]);
    edit_transform(engine.app.world_mut(), player, |pose| {
        pose.translation.x = 20.0;
    })
    .unwrap();
    let edited = execute(&tools, &mut engine, Some(&scene), global_request);
    assert_eq!(
        edited["result"]["entity"]["components"]["GlobalTransform"]["translation"],
        json!([22.0, 0.0, 0.0])
    );
    let still_preview = execute(&tools, &mut engine, Some(&scene), request.clone());
    assert_eq!(still_preview["result"], preview["result"]);
    assert_eq!(still_preview["observation"], preview["observation"]);

    engine.extract(1.0, &mut snapshot);
    let current = execute(&tools, &mut engine, Some(&scene), request);
    assert_eq!(
        current["result"]["entity"]["components"]["PresentedTransform"]["translation"],
        json!([22.0, 0.0, 0.0])
    );
    assert_eq!(current["observation"]["presentation"]["mode"], "current");
    assert_eq!(current["observation"]["presentation"]["alpha"], 1.0);
    assert_eq!(
        current["observation"]["frame"],
        preview["observation"]["frame"]
    );
    assert_eq!(
        current["observation"]["tick"],
        preview["observation"]["tick"]
    );
    engine.advance(0.0);
    let status = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"status"}),
    );
    assert_eq!(status["observation"]["frame"], "2");
    assert_eq!(status["observation"]["presentation"]["frame"], "1");
    assert_eq!(status["result"]["presentation"]["frame"], "1");
    assert_eq!(status["observation"]["tick"], "1");
}

#[test]
fn inspection_materializes_gpu_presented_frames_exactly() {
    use engine_core::{FixedSimulation, InterpolateTransform, PresentationMode};

    let mut source: Value = serde_json::from_str(SOURCE).unwrap();
    source["version"] = json!(3);
    source["entities"].as_array_mut().unwrap().push(json!({
        "id":"child", "parent":"player", "mesh":"box", "material":"teal",
        "transform":{"translation":[2.0,0.0,0.0]}
    }));
    let mut engine = Engine::default();
    let scene = SceneDocument::from_json(&source.to_string())
        .unwrap()
        .instantiate(engine.app.world_mut())
        .unwrap();
    let player = scene.entity("player").unwrap();
    engine
        .app
        .world_mut()
        .entity_mut(player)
        .insert(InterpolateTransform);
    engine.app.add_systems(
        FixedSimulation,
        |mut transforms: Query<&mut Transform, With<InterpolateTransform>>| {
            for mut transform in &mut transforms {
                transform.translation.x += 4.0;
            }
        },
    );
    let tools = DevTools::default();
    assert_eq!(engine.advance(1.0 / 60.0).fixed_steps, 1);
    let request =
        json!({"version":1,"op":"inspect","entity":"child","component":"PresentedTransform"});
    engine
        .extract_frame(1.0, PresentationMode::Interpolated { alpha: 0.75 })
        .unwrap();
    let earlier = execute(&tools, &mut engine, Some(&scene), request.clone());
    engine.extract_gpu_frame(1.0, 0.25).unwrap();
    let gpu = execute(&tools, &mut engine, Some(&scene), request.clone());
    // The observation precedes materialization; the result is that frame.
    assert_eq!(gpu["observation"]["presentation"]["gpu_pending"], true);
    assert_eq!(gpu["observation"]["presentation"]["alpha"], 0.25);
    assert_eq!(
        gpu["result"]["entity"]["components"]["PresentedTransform"]["translation"],
        json!([3.0, 0.0, 0.0])
    );
    assert_ne!(gpu["result"], earlier["result"]);
    let state = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"inspect","component":"PresentationState"}),
    );
    assert_eq!(state["result"]["value"]["gpu_pending"], false);
    assert_eq!(state["result"]["value"]["alpha"], 0.25);
    engine
        .extract_frame(1.0, PresentationMode::Interpolated { alpha: 0.25 })
        .unwrap();
    let reference = execute(&tools, &mut engine, Some(&scene), request);
    assert_eq!(reference["result"], gpu["result"]);
}

#[test]
fn hierarchy_discovery_refreshes_paused_world_and_filters_current_children() {
    let mut source: Value = serde_json::from_str(SOURCE).unwrap();
    source["version"] = json!(3);
    source["prefabs"] = json!([{"id":"pair","entities":[
        {"id":"a","mesh":"box","material":"teal","transform":{"translation":[1,0,0]}},
        {"id":"b","mesh":"box","material":"teal"}
    ]}]);
    source["entities"]
        .as_array_mut()
        .unwrap()
        .push(json!({"id":"group","prefab":"pair"}));
    let mut engine = Engine::default();
    let scene = SceneDocument::from_json(&source.to_string())
        .unwrap()
        .instantiate(engine.app.world_mut())
        .unwrap();
    let tools = DevTools::default();
    let describe = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"describe"}),
    );
    for name in ["Hierarchy", "GlobalTransform"] {
        assert!(
            describe["result"]["types"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["name"] == name)
        );
    }
    let request =
        json!({"version":1,"op":"inspect","parent":"group","component":"Hierarchy","limit":1});
    let page = execute(&tools, &mut engine, Some(&scene), request.clone());
    assert_eq!(page["result"]["total"], 2);
    assert_eq!(page["result"]["next_offset"], 1);
    assert_eq!(page["result"]["entities"][0]["id"], "group.a");
    assert_eq!(
        page["result"]["entities"][0]["components"]["Hierarchy"],
        json!({
            "parent":"group","parent_scene":"test-scene","parent_alive":true,"parent_in_active_scene":true,
            "authored_parent":"group","prefab_id":"pair","prefab_instance":"group","prefab_local_id":"a"
        })
    );
    edit_transform(
        engine.app.world_mut(),
        scene.entity("group").unwrap(),
        |pose| pose.translation.x = 9.0,
    )
    .unwrap();
    let global_request =
        json!({"version":1,"op":"inspect","entity":"group.a","component":"GlobalTransform"});
    let global = execute(&tools, &mut engine, Some(&scene), global_request.clone());
    assert_eq!(
        global["result"]["entity"]["components"]["GlobalTransform"]["translation"],
        json!([10.0, 0.0, 0.0])
    );
    assert_eq!(global["observation"]["tick"], "0");
    assert_eq!(global["observation"]["frame"], "0");
    engine_core::set_parent(
        engine.app.world_mut(),
        scene.entity("group.a").unwrap(),
        Some(scene.entity("player").unwrap()),
    )
    .unwrap();
    let page = execute(&tools, &mut engine, Some(&scene), request);
    assert_eq!(page["result"]["total"], 1);
    assert_eq!(page["result"]["entities"][0]["id"], "group.b");
    engine
        .app
        .world_mut()
        .despawn(scene.entity("player").unwrap());
    let global = execute(&tools, &mut engine, Some(&scene), global_request);
    assert_eq!(
        global["result"]["entity"]["components"]["GlobalTransform"],
        json!({"valid":false,"matrix":null,"translation":null})
    );
    let hierarchy = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"inspect","entity":"group.a","component":"Hierarchy"}),
    );
    assert_eq!(
        hierarchy["result"]["entity"]["components"]["Hierarchy"]["parent_alive"],
        false
    );
    assert_eq!(
        hierarchy["result"]["entity"]["components"]["Hierarchy"]["authored_parent"],
        "group"
    );
    let other = SceneDocument::from_json(&source.to_string())
        .unwrap()
        .instantiate(engine.app.world_mut())
        .unwrap();
    engine_core::set_parent(
        engine.app.world_mut(),
        scene.entity("group.a").unwrap(),
        Some(other.entity("group").unwrap()),
    )
    .unwrap();
    let hierarchy = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"inspect","entity":"group.a","component":"Hierarchy"}),
    );
    assert_eq!(
        hierarchy["result"]["entity"]["components"]["Hierarchy"]["parent"],
        "group"
    );
    assert_eq!(
        hierarchy["result"]["entity"]["components"]["Hierarchy"]["parent_alive"],
        true
    );
    assert_eq!(
        hierarchy["result"]["entity"]["components"]["Hierarchy"]["parent_in_active_scene"],
        false
    );
    other.despawn(engine.app.world_mut());
    for (request, code) in [
        (
            json!({"entity":"group.a","parent":"group"}),
            "invalid_selector",
        ),
        (
            json!({"parent":"group","component":"Camera"}),
            "invalid_selector",
        ),
        (json!({"parent":"unknown"}), "unknown_entity"),
        (json!({"parent":"player"}), "entity_removed"),
    ] {
        let mut request = request;
        request["op"] = json!("inspect");
        request["version"] = json!(1);
        let response = execute(&tools, &mut engine, Some(&scene), request);
        assert_eq!(response["error"]["code"], code);
    }
}

#[test]
fn authored_texture_inspection_reports_readiness_without_serializing_pixels() {
    let mut source: Value = serde_json::from_str(SOURCE).unwrap();
    source["version"] = json!(2);
    source["textures"] = json!([{"id":"paint","width":1,"height":1,"rgba8":"/wAA/w=="}]);
    source["materials"][0]["texture"] = json!("paint");
    let mut engine = Engine::default();
    let scene = SceneDocument::from_json(&source.to_string())
        .unwrap()
        .instantiate(engine.app.world_mut())
        .unwrap();
    let tools = DevTools::default();
    let response = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"assets"}),
    );
    assert_eq!(response["result"]["textures"], 1);
    assert_eq!(response["result"]["texture_cpu_bytes"], 4);
    assert_eq!(response["result"]["texture_assets"][0]["id"], "paint");
    assert_eq!(
        response["result"]["texture_assets"][0]["state"],
        "cpu_ready"
    );
    assert!(!response.to_string().contains("/wAA/w=="));
    let request = json!({"version":1,"op":"inspect","entity":"player","component":"Mesh3d"});
    let response = execute(&tools, &mut engine, Some(&scene), request.clone());
    assert_eq!(
        response["result"]["entity"]["components"]["Mesh3d"]["texture_alive"],
        true
    );
    assert_eq!(
        response["result"]["entity"]["components"]["Mesh3d"]["texture_dimensions"],
        json!([1, 1])
    );
    let handle = scene.textures().next().unwrap().1;
    engine
        .app
        .world_mut()
        .resource_mut::<engine_core::AssetLibrary>()
        .remove_texture(handle);
    let response = execute(&tools, &mut engine, Some(&scene), request);
    assert_eq!(
        response["result"]["entity"]["components"]["Mesh3d"]["texture_alive"],
        false
    );
    let response = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"assets"}),
    );
    assert_eq!(response["result"]["texture_assets"][0]["state"], "missing");
    assert_eq!(engine.app.world().resource::<FixedTime>().tick, 0);
    scene.despawn(engine.app.world_mut());
}

#[test]
fn rejects_malformed_oversized_unknown_fields_and_versions() {
    let tools = DevTools::default();
    let mut engine = Engine::default();
    for (source, code) in [
        ("{".to_string(), "invalid_request"),
        (
            r#"{"version":1,"op":"unknown"}"#.to_string(),
            "invalid_request",
        ),
        (
            r#"{"version":1,"op":"status","typo":true}"#.to_string(),
            "invalid_request",
        ),
        (
            r#"{"version":1,"op":"inspect","typo":true}"#.to_string(),
            "invalid_request",
        ),
        (
            r#"{"version":2,"op":"status"}"#.to_string(),
            "unsupported_version",
        ),
        (" ".repeat(MAX_REQUEST_BYTES + 1), "request_too_large"),
    ] {
        let response: Value =
            serde_json::from_str(&tools.execute(&mut engine, None, &source)).unwrap();
        assert_eq!(response["error"]["code"], code, "{source}");
        assert_eq!(response["observation"]["frame"], "0");
    }
}

#[test]
fn inspection_preserves_u64_precision_and_does_not_advance_simulation() {
    let mut engine = Engine::default();
    let mut document = SceneDocument::from_json(SOURCE).unwrap();
    document.revision = u64::MAX;
    let scene = document.instantiate(engine.app.world_mut()).unwrap();
    engine.app.world_mut().resource_mut::<FrameTime>().frame = u64::MAX;
    engine.app.world_mut().resource_mut::<FixedTime>().tick = u64::MAX - 1;
    engine.extract(1.0, &mut engine_core::RenderSnapshot::default());
    let response = execute(
        &DevTools::default(),
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"inspect","entity":"player","component":"Transform"}),
    );
    assert_eq!(response["ok"], true);
    assert_eq!(response["observation"]["frame"], u64::MAX.to_string());
    assert_eq!(response["observation"]["tick"], (u64::MAX - 1).to_string());
    assert_eq!(
        response["observation"]["presentation"]["frame"],
        u64::MAX.to_string()
    );
    assert_eq!(
        response["observation"]["presentation"]["tick"],
        (u64::MAX - 1).to_string()
    );
    assert_eq!(
        response["observation"]["scene"]["revision"],
        u64::MAX.to_string()
    );
    assert_eq!(response["result"]["entity"]["name"], "Player");
    assert_eq!(
        response["result"]["entity"]["components"]["Transform"]["scale"],
        json!([1.0, 1.0, 1.0])
    );
    assert_eq!(engine.app.world().resource::<FrameTime>().frame, u64::MAX);
}

#[test]
fn paginates_filtered_authored_entities_in_stable_order_and_skips_removed() {
    let mut engine = Engine::default();
    let mut document = SceneDocument::from_json(SOURCE).unwrap();
    let template = document.entities.pop().unwrap();
    for index in 0..125 {
        let mut object = template.clone();
        object.id = format!("object-{index:03}");
        document.entities.push(object);
    }
    let scene = document.instantiate(engine.app.world_mut()).unwrap();
    engine
        .app
        .world_mut()
        .despawn(scene.entity("object-001").unwrap());
    engine
        .app
        .world_mut()
        .entity_mut(scene.entity("object-002").unwrap())
        .remove::<Transform>();
    let tools = DevTools::default();
    let response = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"inspect","component":"Transform","offset":1,"limit":100}),
    );
    assert_eq!(response["result"]["total"], 123);
    assert_eq!(
        response["result"]["entities"].as_array().unwrap().len(),
        100
    );
    assert_eq!(response["result"]["entities"][0]["id"], "object-003");
    assert_eq!(response["result"]["next_offset"], 101);
    let removed = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"inspect","entity":"object-001"}),
    );
    assert_eq!(removed["error"]["code"], "entity_removed");
    let invalid = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"inspect","limit":101}),
    );
    assert_eq!(invalid["error"]["code"], "invalid_page");
}

#[derive(Resource)]
struct Score(u32);

#[test]
fn registers_typed_resource_and_rejects_duplicate_or_oversized_descriptors() {
    let mut tools = DevTools::default();
    let descriptor =
        InspectorDescriptor::new("Score", "Collected score", json!({"type":"integer"}));
    tools
        .register_resource::<Score>(descriptor.clone(), |value| json!(value.0))
        .unwrap();
    assert_eq!(
        tools
            .register_resource::<Score>(descriptor, |value| json!(value.0))
            .unwrap_err()
            .code,
        "duplicate_inspector"
    );
    let descriptor = InspectorDescriptor::new("Huge", "x".repeat(513), json!({}));
    assert_eq!(
        tools
            .register_resource::<Score>(descriptor, |_| Value::Null)
            .unwrap_err()
            .code,
        "inspector_too_large"
    );
    let mut engine = Engine::default();
    engine.app.insert_resource(Score(3));
    let response = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"inspect","component":"Score"}),
    );
    assert_eq!(response["result"]["value"], 3);
    let invalid = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"inspect","entity":"player","component":"Score"}),
    );
    assert_eq!(invalid["error"]["code"], "invalid_selector");
}

#[test]
fn metadata_is_generated_from_serialized_view_and_runtime_default() {
    let mut engine = Engine::default();
    let tools = DevTools::default();
    let metadata = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"describe","name":"Transform"}),
    );
    assert_eq!(
        metadata["result"]["descriptor"]["schema"]["properties"]["rotation"]["minItems"],
        4
    );
    assert_eq!(
        metadata["result"]["descriptor"]["default"]["rotation"],
        json!([0.0, 0.0, 0.0, 1.0])
    );
    let scene_schema = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"describe","name":"SceneDocument"}),
    );
    assert_eq!(scene_schema["result"]["schema"]["title"], "SceneDocument");
}

#[test]
fn control_schema_matches_typed_defaults_and_rejects_unknown_fields() {
    let command: crate::BrowserControl =
        serde_json::from_value(json!({"version":1,"op":"input","held":1})).unwrap();
    assert_eq!(command.version(), 1);
    match command {
        crate::BrowserControl::Input {
            forward,
            right,
            up,
            yaw,
            pitch,
            held,
            pressed,
            released,
            ..
        } => {
            assert_eq!([forward, right, up, yaw, pitch], [0.0; 5]);
            assert_eq!([held, pressed, released], [1, 0, 0]);
        }
        _ => panic!("expected input"),
    }
    assert!(
        serde_json::from_value::<crate::BrowserControl>(
            json!({"version":1,"op":"step","ticks":1,"typo":true})
        )
        .is_err()
    );
    let description = execute(
        &DevTools::default(),
        &mut Engine::default(),
        None,
        json!({"version":1,"op":"describe","name":"BrowserControl"}),
    );
    let variants = description["result"]["schema"]["oneOf"].as_array().unwrap();
    let step = variants
        .iter()
        .find(|variant| variant["properties"]["op"]["const"] == "step")
        .unwrap();
    assert_eq!(step["properties"]["ticks"]["maximum"], 120);
}

#[test]
fn environment_projections_share_scene_sections_and_surfaces_are_inspectable() {
    use engine_core::{AmbientOcclusion, Atmosphere, Bloom, Shadows};
    let tools = DevTools::default();
    let mut engine = Engine::default();
    // Discovery lists every environment resource and the combined look.
    let listing = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"describe"}),
    );
    let names: Vec<&str> = listing["result"]["types"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|entry| entry["name"].as_str())
        .collect();
    for name in [
        "Atmosphere",
        "Shadows",
        "AmbientOcclusion",
        "Bloom",
        "Environment",
    ] {
        assert!(names.contains(&name), "{name}");
    }
    let scene_entry = listing["result"]["types"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == "SceneDocument")
        .unwrap();
    assert_eq!(scene_entry["source"], engine_scene::schema_path());
    let described = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"describe","name":"Shadows"}),
    );
    assert_eq!(
        described["result"]["descriptor"]["schema"]["properties"]["resolution"]["maximum"],
        4096
    );

    // Resources read in scene-section shape; the combined view resolves back.
    engine.app.world_mut().insert_resource(Atmosphere::night());
    let atmosphere = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"inspect","component":"Atmosphere"}),
    );
    let section: engine_scene::SceneAtmosphere =
        serde_json::from_value(atmosphere["result"]["value"].clone()).unwrap();
    assert_eq!(section.resolve("$").unwrap(), Atmosphere::night());
    engine.app.world_mut().remove_resource::<Bloom>();
    let environment = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"inspect","component":"Environment"}),
    );
    let look: engine_scene::SceneEnvironment =
        serde_json::from_value(environment["result"]["value"].clone()).unwrap();
    let resolved = look.resolve("$.environment").unwrap();
    assert_eq!(resolved.atmosphere, Some(Atmosphere::night()));
    assert_eq!(resolved.shadows, Some(Shadows::default()));
    assert_eq!(
        resolved.ambient_occlusion,
        Some(AmbientOcclusion::default())
    );
    assert!(
        !resolved.bloom.unwrap().enabled(),
        "a missing effect reads as off"
    );
    // A missing resource is reported, not invented.
    let missing = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"inspect","component":"Bloom"}),
    );
    assert_eq!(missing["ok"], false);

    // The environment control accepts the same sections and rejects typos.
    let control: crate::BrowserControl = serde_json::from_value(json!({
        "version":1,"op":"environment",
        "atmosphere":{"preset":"dusk"},"shadows":{"enabled":false}
    }))
    .unwrap();
    assert!(matches!(
        control,
        crate::BrowserControl::Environment {
            shadows: Some(Shadows { enabled: false, .. }),
            bloom: None,
            ..
        }
    ));
    assert!(
        serde_json::from_value::<crate::BrowserControl>(
            json!({"version":1,"op":"environment","shadow":{"enabled":false}})
        )
        .is_err()
    );
    let description = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"describe","name":"BrowserControl"}),
    );
    assert!(
        description["result"]["schema"]["oneOf"]
            .as_array()
            .unwrap()
            .iter()
            .any(|variant| variant["properties"]["op"]["const"] == "environment")
    );

    // Mesh3d reports the material surface and whether its texture masks emission.
    let mut source: Value = serde_json::from_str(SOURCE).unwrap();
    source["version"] = json!(5);
    source["materials"][0]["surface"] = json!({"roughness": 0.2, "emission": 3.0});
    let scene = SceneDocument::from_json(&source.to_string())
        .unwrap()
        .instantiate(engine.app.world_mut())
        .unwrap();
    let mesh = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"inspect","entity":"player","component":"Mesh3d"}),
    );
    let view = &mesh["result"]["entity"]["components"]["Mesh3d"];
    assert!((view["surface"]["roughness"].as_f64().unwrap() - 0.2).abs() < 1.0e-6);
    assert_eq!(view["surface"]["emission"], 3.0);
    assert_eq!(view["texture_emission_mask"], Value::Null);
    scene.despawn(engine.app.world_mut());
}

#[test]
fn rejects_scene_from_another_world_and_observes_missing_asset_readiness() {
    let mut first = Engine::default();
    let scene = SceneDocument::from_json(SOURCE)
        .unwrap()
        .instantiate(first.app.world_mut())
        .unwrap();
    let tools = DevTools::default();
    let response = execute(
        &tools,
        &mut Engine::default(),
        Some(&scene),
        json!({"version":1,"op":"status"}),
    );
    assert_eq!(response["error"]["code"], "wrong_world");
    let mesh = first
        .app
        .world()
        .get::<engine_core::Mesh3d>(scene.entity("player").unwrap())
        .unwrap()
        .mesh;
    first
        .app
        .world_mut()
        .resource_mut::<engine_core::AssetLibrary>()
        .remove_mesh(mesh);
    let response = execute(
        &tools,
        &mut first,
        Some(&scene),
        json!({"version":1,"op":"inspect","entity":"player","component":"Mesh3d"}),
    );
    assert_eq!(
        response["result"]["entity"]["components"]["Mesh3d"]["mesh_alive"],
        false
    );
    assert_eq!(
        response["result"]["entity"]["components"]["Mesh3d"]["material_alive"],
        true
    );
}

#[test]
fn bounds_extension_projection_and_entire_response() {
    let mut tools = DevTools::default();
    let mut engine = Engine::default();
    engine.app.insert_resource(Score(0));
    tools
        .register_resource::<Score>(
            InspectorDescriptor::new("Huge", "Bounded extension", json!({})),
            |_| json!("x".repeat(17000)),
        )
        .unwrap();
    let response = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"inspect","component":"Huge"}),
    );
    assert_eq!(response["error"]["code"], "inspector_too_large");
    tools
        .register_component::<Transform>(
            InspectorDescriptor::new("Long", "Per-entity view", json!({})),
            |_| json!("x".repeat(1000)),
        )
        .unwrap();
    let mut document = SceneDocument::from_json(SOURCE).unwrap();
    let template = document.entities.pop().unwrap();
    for index in 0..100 {
        let mut object = template.clone();
        object.id = format!("object-{index}");
        document.entities.push(object);
    }
    let scene = document.instantiate(engine.app.world_mut()).unwrap();
    let response = tools.execute(
        &mut engine,
        Some(&scene),
        r#"{"version":1,"op":"inspect","component":"Long","limit":100}"#,
    );
    assert!(response.len() <= MAX_RESPONSE_BYTES);
    assert_eq!(
        serde_json::from_str::<Value>(&response).unwrap()["error"]["code"],
        "response_too_large"
    );
}

#[cfg(not(target_arch = "wasm32"))]
mod authoring {
    use super::SOURCE;
    use crate::authoring::{MAX_SOURCE_BYTES, read_scene_source, save_scene, validate_scene};
    use std::{
        fs,
        path::PathBuf,
        sync::{
            Arc, Barrier,
            atomic::{AtomicU64, Ordering},
        },
    };

    static NEXT_TEST: AtomicU64 = AtomicU64::new(1);

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "engine-dev-test-{}-{}",
                std::process::id(),
                NEXT_TEST.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            fs::write(path.join("scene.json"), SOURCE).unwrap();
            Self(path)
        }
        fn source(&self) -> PathBuf {
            self.0.join("scene.json")
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn saves_exact_candidate_and_rejects_stale_revision_without_mutation() {
        let fixture = Fixture::new();
        let candidate = SOURCE.replace("\"revision\": 4", "\"revision\": 5");
        let report = save_scene(&fixture.source(), &candidate, 4).unwrap();
        assert_eq!(report.revision, "5");
        assert!(report.persisted);
        assert_eq!(fs::read_to_string(fixture.source()).unwrap(), candidate);
        let error = save_scene(&fixture.source(), &candidate, 4).unwrap_err();
        assert_eq!(error.code, "revision_conflict");
        assert_eq!(fs::read_to_string(fixture.source()).unwrap(), candidate);
        assert_eq!(fs::read_dir(&fixture.0).unwrap().count(), 1);
    }

    #[test]
    fn invalid_source_identity_revision_and_overflow_preserve_original() {
        let fixture = Fixture::new();
        let candidate = SOURCE.replace("\"revision\": 4", "\"revision\": 5");
        for (candidate, expected, code) in [
            ("{".to_string(), 4, "invalid_json"),
            (
                candidate.replace("test-scene", "another"),
                4,
                "scene_id_mismatch",
            ),
            (SOURCE.to_string(), 4, "invalid_revision"),
            (candidate, u64::MAX, "revision_exhausted"),
        ] {
            assert_eq!(
                save_scene(&fixture.source(), &candidate, expected)
                    .unwrap_err()
                    .code,
                code
            );
            assert_eq!(fs::read_to_string(fixture.source()).unwrap(), SOURCE);
        }
        let huge = " ".repeat(MAX_SOURCE_BYTES + 1);
        assert_eq!(validate_scene(&huge).unwrap_err().code, "source_too_large");
        fs::write(fixture.source(), huge).unwrap();
        assert_eq!(
            read_scene_source(&fixture.source()).unwrap_err().code,
            "source_too_large"
        );
    }

    #[test]
    fn concurrent_cooperating_writers_cannot_both_commit_same_revision() {
        let fixture = Fixture::new();
        let barrier = Arc::new(Barrier::new(2));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let path = fixture.source();
            let barrier = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                save_scene(
                    &path,
                    &SOURCE.replace("\"revision\": 4", "\"revision\": 5"),
                    4,
                )
            }));
        }
        let results: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        let error = results.into_iter().find_map(Result::err).unwrap();
        assert!(["source_locked", "revision_conflict"].contains(&error.code));
        assert_eq!(
            validate_scene(&fs::read_to_string(fixture.source()).unwrap())
                .unwrap()
                .revision,
            "5"
        );
    }
}

#[test]
fn lights_particles_and_opt_in_effects_are_discoverable_and_controllable() {
    use engine_core::{LocalLight, ParticleEmitter, Reflections, SunShafts, glam::Vec3};
    let tools = DevTools::default();
    let mut engine = Engine::default();
    let listing = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"describe"}),
    );
    let names: Vec<&str> = listing["result"]["types"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|entry| entry["name"].as_str())
        .collect();
    for name in [
        "LocalLight",
        "ParticleEmitter",
        "LocalLights",
        "SunShafts",
        "Reflections",
        "Particles",
    ] {
        assert!(names.contains(&name), "{name}");
    }
    // Opt-in effects are missing until inserted; the combined look reads
    // them as off.
    let missing = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"inspect","component":"SunShafts"}),
    );
    assert_eq!(missing["ok"], false);
    let look = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"inspect","component":"Environment"}),
    );
    assert_eq!(look["result"]["value"]["sun_shafts"]["intensity"], 0.0);
    assert_eq!(look["result"]["value"]["local_lights"]["enabled"], true);
    engine.app.insert_resource(Reflections::default());
    let reflections = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"inspect","component":"Reflections"}),
    );
    let section: Reflections =
        serde_json::from_value(reflections["result"]["value"].clone()).unwrap();
    assert_eq!(section, Reflections::default());
    assert!(section.validate().is_ok());
    let _ = SunShafts::default();

    // Entity projections are the scene fields.
    let mut source: Value = serde_json::from_str(SOURCE).unwrap();
    source["version"] = json!(6);
    source["entities"][0]["light"] = json!({"intensity": 12, "range": 7});
    source["entities"][0]["particles"] = json!({"rate": 4, "lifetime": 2});
    let scene = SceneDocument::from_json(&source.to_string())
        .unwrap()
        .instantiate(engine.app.world_mut())
        .unwrap();
    let light = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"inspect","entity":"player","component":"LocalLight"}),
    );
    let view = &light["result"]["entity"]["components"]["LocalLight"];
    assert_eq!(view["intensity"], 12.0);
    let restored: engine_scene::SceneLight = serde_json::from_value(view.clone()).unwrap();
    let player = scene.entity("player").unwrap();
    assert_eq!(
        restored.runtime("$").unwrap(),
        *engine.app.world().get::<LocalLight>(player).unwrap()
    );
    assert_eq!(
        engine.app.world().get::<LocalLight>(player).unwrap().color,
        Vec3::new(1.0, 0.85, 0.6),
        "omitted fields take the scene defaults"
    );
    let emitter = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"inspect","entity":"player","component":"ParticleEmitter"}),
    );
    assert_eq!(
        emitter["result"]["entity"]["components"]["ParticleEmitter"]["rate"],
        4.0
    );
    assert!(engine.app.world().get::<ParticleEmitter>(player).is_some());
    scene.despawn(engine.app.world_mut());

    // The environment control takes the version 6 sections too.
    let control: crate::BrowserControl = serde_json::from_value(json!({
        "version":1,"op":"environment",
        "sun_shafts":{"intensity":2},"local_lights":{"enabled":false},
        "reflections":{},"particles":{"max_particles":100}
    }))
    .unwrap();
    assert!(matches!(
        control,
        crate::BrowserControl::Environment {
            sun_shafts: Some(_),
            local_lights: Some(engine_core::LocalLights { enabled: false, .. }),
            reflections: Some(_),
            particles: Some(_),
            ..
        }
    ));
}

#[test]
fn reflection_probes_are_discoverable_inspectable_and_controllable() {
    use engine_core::{ReflectionProbe, ReflectionProbes, glam::Vec3};
    let tools = DevTools::default();
    let mut engine = Engine::default();
    let listing = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"describe"}),
    );
    let names: Vec<&str> = listing["result"]["types"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|entry| entry["name"].as_str())
        .collect();
    for name in ["ReflectionProbe", "ReflectionProbes"] {
        assert!(names.contains(&name), "{name}");
    }
    // The engine inserts the settings; they read as their scene section.
    let settings = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"inspect","component":"ReflectionProbes"}),
    );
    let section: ReflectionProbes =
        serde_json::from_value(settings["result"]["value"].clone()).unwrap();
    assert_eq!(section, ReflectionProbes::default());
    let look = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"inspect","component":"Environment"}),
    );
    assert_eq!(
        look["result"]["value"]["reflection_probes"],
        settings["result"]["value"]
    );

    // An entity's probe reads as its scene field.
    let mut source: Value = serde_json::from_str(SOURCE).unwrap();
    source["version"] = json!(8);
    source["entities"][0]["reflection_probe"] = json!({"extent": [6.0, 3.0, 4.0], "blend": 0.5});
    let scene = SceneDocument::from_json(&source.to_string())
        .unwrap()
        .instantiate(engine.app.world_mut())
        .unwrap();
    let probe = execute(
        &tools,
        &mut engine,
        Some(&scene),
        json!({"version":1,"op":"inspect","entity":"player","component":"ReflectionProbe"}),
    );
    let view = &probe["result"]["entity"]["components"]["ReflectionProbe"];
    assert_eq!(view, &source["entities"][0]["reflection_probe"]);
    let restored: engine_scene::SceneReflectionProbe =
        serde_json::from_value(view.clone()).unwrap();
    let player = scene.entity("player").unwrap();
    let component = *engine.app.world().get::<ReflectionProbe>(player).unwrap();
    assert_eq!(restored.runtime("$").unwrap(), component);
    assert_eq!(component.extent, Vec3::new(6.0, 3.0, 4.0));
    scene.despawn(engine.app.world_mut());

    // The environment control takes the section; omitted fields take the
    // defaults and typos are refused.
    let control: crate::BrowserControl = serde_json::from_value(json!({
        "version":1,"op":"environment","reflection_probes":{"max_probes":8,"faces_per_frame":2}
    }))
    .unwrap();
    let crate::BrowserControl::Environment {
        reflection_probes: Some(section),
        ..
    } = control
    else {
        panic!("expected reflection probe settings");
    };
    assert!(section.validate().is_ok());
    assert_eq!(
        section,
        ReflectionProbes {
            max_probes: Some(8),
            faces_per_frame: 2,
            ..ReflectionProbes::default()
        }
    );
    assert!(
        serde_json::from_value::<crate::BrowserControl>(
            json!({"version":1,"op":"environment","reflection_probes":{"faces":2}})
        )
        .is_err()
    );
}

#[test]
fn virtual_geometry_settings_are_discoverable_inspectable_and_controllable() {
    use engine_core::VirtualGeometry;
    let tools = DevTools::default();
    let mut engine = Engine::default();
    let listing = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"describe"}),
    );
    assert!(
        listing["result"]["types"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["name"] == "VirtualGeometry")
    );
    // The engine inserts the settings; they read as their scene section.
    let settings = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"inspect","component":"VirtualGeometry"}),
    );
    let section: VirtualGeometry =
        serde_json::from_value(settings["result"]["value"].clone()).unwrap();
    assert_eq!(section, VirtualGeometry::default());
    let look = execute(
        &tools,
        &mut engine,
        None,
        json!({"version":1,"op":"inspect","component":"Environment"}),
    );
    assert_eq!(
        look["result"]["value"]["virtual_geometry"],
        settings["result"]["value"]
    );
    // The environment control takes the section; out-of-range errors and
    // typos are refused.
    let control: crate::BrowserControl = serde_json::from_value(json!({
        "version":1,"op":"environment","virtual_geometry":{"pixel_error":3.0}
    }))
    .unwrap();
    let crate::BrowserControl::Environment {
        virtual_geometry: Some(section),
        ..
    } = control
    else {
        panic!("expected virtual geometry settings");
    };
    assert!(section.validate().is_ok());
    assert_eq!(
        section,
        VirtualGeometry {
            pixel_error: 3.0,
            ..VirtualGeometry::default()
        }
    );
    assert_eq!(
        VirtualGeometry {
            pixel_error: 0.1,
            ..Default::default()
        }
        .validate()
        .unwrap_err()
        .field,
        "pixel_error"
    );
    assert!(
        serde_json::from_value::<crate::BrowserControl>(
            json!({"version":1,"op":"environment","virtual_geometry":{"error":2}})
        )
        .is_err()
    );
}
