use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::{Entity, World};
use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use engine_core::{AuthoredId, OperationScene, Transform, set_parent, smoke_check, smoke_eq};
use glam::Vec3;
use serde_json::{Value, json};

use crate::{StreamSet, Streamed, StreamedId, StreamedItem, WorldCells, remove_streamed};

const SCENE: &str = "streaming-fixture";

struct Pieces {
    at: Vec3,
    children: u32,
}

impl Streamed for Pieces {
    fn entities(&self) -> u32 {
        1 + self.children
    }

    fn spawn(&self, world: &mut World, spawned: &mut Vec<Entity>) {
        let root = world.spawn(Transform::from_translation(self.at)).id();
        spawned.push(root);
        for child in 0..self.children {
            let part = world
                .spawn(Transform::from_xyz(0.0, 1.0 + child as f32, 0.0))
                .id();
            if set_parent(world, part, Some(root)).is_ok() {
                spawned.push(part);
            } else {
                world.despawn(part);
            }
        }
    }
}

const ITEMS: [(&str, [f32; 3], f32, u32); 5] = [
    ("crate-near", [20.0, 0.5, 0.0], 0.9, 0),
    ("crate-mid", [900.0, 0.5, 0.0], 0.9, 0),
    ("spinner", [-30.0, 30.0, 10.0], 1.6, 1),
    ("lamp-far", [1500.0, 0.0, 300.0], 12.0, 2),
    ("tower", [5000.0, 30.0, 0.0], 40.0, 0),
];

pub(crate) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    let set = stream(session.app().world_mut(), &ITEMS);
    ask(session, "control", json!({"name": "pause"}))?;
    let rendering = look(session, 0.0, 0.0)?;
    let cells = inspect_cells(session)?;
    smoke_eq!(json!(cells["items"]), json!(5), "binning_items");
    let grids: Vec<Value> = cells["grids"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|grid| json!([grid["cells"], grid["loaded"]]))
        .collect();
    smoke_eq!(
        json!(grids),
        json!([[3, 2], [1, 1], [1, 1]]),
        "binning_grids"
    );
    let streamed = &rendering["world_streaming"];
    smoke_check!(
        streamed["cells"] == 5
            && streamed["loaded"] == 4
            && streamed["pending"] == 0
            && streamed["entities"] == 7,
        "binning_stats",
        json!(streamed),
        json!({"cells": 5, "loaded": 4, "pending": 0, "entities": 7})
    );
    smoke_check!(
        !loaded(session, "crate-mid")?,
        "out_of_range_unloaded",
        json!("crate-mid"),
        json!("unloaded")
    );
    let marked = component(session, "spinner", "StreamedId")?;
    smoke_eq!(
        json!(marked["id"]),
        json!("streaming-fixture/spinner"),
        "root_carries_id"
    );
    hysteresis(session)?;
    settings(session)?;
    removals(session)?;
    let world = session.app().world_mut();
    remove_streamed(world, set);
    stream(world, &ITEMS[..1]);
    let cells = inspect_cells(session)?;
    smoke_eq!(json!(cells["items"]), json!(1), "replaced_items");
    Ok(())
}

fn hysteresis(session: &mut dyn SmokeSession) -> SmokeResult {
    look(session, 850.0, 0.0)?;
    smoke_check!(
        loaded(session, "crate-near")?,
        "hysteresis_keeps",
        json!("crate-near"),
        json!("loaded")
    );
    smoke_check!(
        loaded(session, "crate-mid")?,
        "hysteresis_loads",
        json!("crate-mid"),
        json!("loaded")
    );
    let rendering = look(session, 1000.0, 0.0)?;
    smoke_check!(
        !loaded(session, "crate-near")?,
        "hysteresis_unloads",
        json!("crate-near"),
        json!("unloaded")
    );
    smoke_check!(
        rendering["world_streaming"]["despawned"]
            .as_u64()
            .unwrap_or(0)
            > 0,
        "hysteresis_despawns",
        json!(rendering["world_streaming"]),
        json!({"despawned": 1})
    );
    let loaded_grids: Vec<Value> = inspect_cells(session)?["grids"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|grid| grid["loaded"].clone())
        .collect();
    smoke_eq!(json!(loaded_grids), json!([1, 1, 1]), "hysteresis_grids");
    Ok(())
}

fn settings(session: &mut dyn SmokeSession) -> SmokeResult {
    streaming(session, json!({"spawns_per_frame": 1}))?;
    let rendering = look(session, 0.0, 0.0)?;
    smoke_check!(
        rendering["world_streaming"]["spawned"]
            .as_u64()
            .unwrap_or(0)
            > 0
            && rendering["world_streaming"]["pending"] == 1,
        "budget_defers",
        json!(rendering["world_streaming"]),
        json!({"pending": 1})
    );
    let rendering = render(session)?;
    smoke_check!(
        rendering["world_streaming"]["pending"] == 0 && rendering["world_streaming"]["loaded"] == 4,
        "budget_catches_up",
        json!(rendering["world_streaming"]),
        json!({"pending": 0, "loaded": 4})
    );

    streaming(session, json!({"frozen": true}))?;
    let rendering = look(session, 3000.0, 3000.0)?;
    smoke_eq!(
        json!(rendering["world_streaming"]["loaded"]),
        json!(4),
        "frozen_cells_stay"
    );
    let rendering = streaming(session, json!({"frozen": false}))?;
    smoke_eq!(
        json!(rendering["world_streaming"]["loaded"]),
        json!(2),
        "unfrozen_small_cells_go"
    );
    let rendering = look(session, 0.0, 0.0)?;
    smoke_eq!(
        json!(rendering["world_streaming"]["loaded"]),
        json!(4),
        "returning_loads"
    );
    let rendering = streaming(session, json!({"enabled": false}))?;
    smoke_check!(
        rendering["world_streaming"]["loaded"] == 0
            && rendering["world_streaming"]["entities"] == 0,
        "disabled_unloads",
        json!(rendering["world_streaming"]),
        json!({"loaded": 0, "entities": 0})
    );
    let rendering = streaming(session, json!({"enabled": true}))?;
    smoke_eq!(
        json!(rendering["world_streaming"]["loaded"]),
        json!(4),
        "enabled_loads"
    );
    let refused = refuse(
        session,
        "control",
        json!({"name": "environment", "args": {"world_streaming": {"unload": 0.5}}}),
    )?;
    smoke_eq!(
        json!(refused["path"]),
        json!("$.args.world_streaming.unload"),
        "setting_refused"
    );
    Ok(())
}

fn removals(session: &mut dyn SmokeSession) -> SmokeResult {
    look(session, 0.0, 0.0)?;
    session.step(1);
    camera(session, [10.0, 60.0, 40.0], [30.0, 20.0, 0.0])?;
    session.step(1);
    let cells = inspect_cells(session)?;
    smoke_check!(
        cells["velocity"][0].as_f64().unwrap_or(0.0) > 100.0,
        "prefetch_measures_motion",
        json!(cells["velocity"]),
        json!([100, 0, 0])
    );
    smoke_check!(
        loaded(session, "crate-mid")?,
        "prefetch_loads_ahead",
        json!("crate-mid"),
        json!("loaded")
    );

    look(session, 0.0, 0.0)?;
    let removal = control(
        session,
        "world_changes",
        json!({"remove": ["streaming-fixture/crate-near"]}),
    )?;
    smoke_eq!(json!(removal["newly_removed"]), json!(1), "removal_counts");
    smoke_check!(
        !loaded(session, "crate-near")?,
        "removed_despawns",
        json!("crate-near"),
        json!("unloaded")
    );
    let changes = ask(session, "inspect", json!({"component": "WorldChanges"}))?["value"].clone();
    smoke_eq!(
        changes,
        json!({"removed": 1, "ids": ["streaming-fixture/crate-near"]}),
        "removal_inspects"
    );
    look(session, 1000.0, 0.0)?;
    look(session, 0.0, 0.0)?;
    smoke_check!(
        !loaded(session, "crate-near")?,
        "removed_stays_out",
        json!("crate-near"),
        json!("unloaded")
    );
    control(
        session,
        "world_changes",
        json!({"restore": ["streaming-fixture/crate-near"]}),
    )?;
    smoke_check!(
        loaded(session, "crate-near")?,
        "restored_spawns",
        json!("crate-near"),
        json!("loaded")
    );
    let empty = refuse(
        session,
        "control",
        json!({"name": "world_changes", "args": {"remove": [""]}}),
    )?;
    smoke_eq!(
        json!(empty["path"]),
        json!("$.args.remove[0]"),
        "empty_id_refused"
    );
    Ok(())
}

fn stream(world: &mut World, items: &[(&str, [f32; 3], f32, u32)]) -> StreamSet {
    let set = world
        .get_resource_or_insert_with(WorldCells::default)
        .add(items.iter().map(|&(id, at, reach, children)| StreamedItem {
            id: Some(Arc::from(format!("{SCENE}/{id}"))),
            position: Vec3::from_array(at),
            reach,
            content: Arc::new(Pieces {
                at: Vec3::from_array(at),
                children,
            }),
        }));
    world.insert_resource(OperationScene {
        id: SCENE.to_owned(),
        entities: Arc::new(BTreeMap::new()),
        streamed: true,
        authored,
    });
    set
}

fn authored(world: &World, entity: Entity) -> Option<AuthoredId> {
    let streamed = world.get::<StreamedId>(entity)?;
    let id = streamed.0.strip_prefix(SCENE)?.strip_prefix('/')?;
    Some(AuthoredId {
        scene: SCENE.to_owned(),
        id: id.to_owned(),
        name: id.to_owned(),
    })
}

fn loaded(session: &mut dyn SmokeSession, id: &str) -> Result<bool, SmokeFailure> {
    let reply = session.request("inspect", json!({"entity": id, "component": "Transform"}));
    let missing = reply
        .as_ref()
        .err()
        .map(|refused| refused["error"]["code"].clone());
    smoke_check!(
        missing.as_ref().is_none_or(|code| code == "unknown_entity"),
        "loaded_inspects",
        json!(missing),
        json!("unknown_entity")
    );
    Ok(reply.is_ok())
}

fn look(session: &mut dyn SmokeSession, x: f32, z: f32) -> Result<Value, SmokeFailure> {
    camera(session, [x, 60.0, z + 40.0], [x + 20.0, 20.0, z])
}

fn camera(
    session: &mut dyn SmokeSession,
    position: [f32; 3],
    target: [f32; 3],
) -> Result<Value, SmokeFailure> {
    let changed = control(
        session,
        "camera",
        json!({
            "position": position, "target": target,
            "vertical_fov_radians": std::f32::consts::FRAC_PI_3, "near": 0.5, "far": 20000,
        }),
    )?;
    Ok(changed["rendering"].clone())
}

fn render(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    Ok(control(session, "render", json!({}))?["rendering"].clone())
}

fn streaming(session: &mut dyn SmokeSession, section: Value) -> Result<Value, SmokeFailure> {
    let changed = control(session, "environment", json!({"world_streaming": section}))?;
    Ok(changed["rendering"].clone())
}

fn inspect_cells(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    Ok(ask(session, "inspect", json!({"component": "WorldCells"}))?["value"].clone())
}

fn component(
    session: &mut dyn SmokeSession,
    id: &str,
    component: &str,
) -> Result<Value, SmokeFailure> {
    let inspected = ask(
        session,
        "inspect",
        json!({"entity": id, "component": component}),
    )?;
    Ok(inspected["entity"]["components"][component].clone())
}

fn control(session: &mut dyn SmokeSession, name: &str, args: Value) -> Result<Value, SmokeFailure> {
    ask(session, "control", json!({"name": name, "args": args}))
}

fn ask(session: &mut dyn SmokeSession, verb: &str, body: Value) -> Result<Value, SmokeFailure> {
    let sent = json!({"verb": verb, "body": body});
    let reply = session.request(verb, body);
    smoke_check!(
        reply.is_ok(),
        "request_accepted",
        json!(reply.as_ref().err()),
        sent
    );
    Ok(reply
        .map(|reply| reply["result"].clone())
        .unwrap_or_default())
}

fn refuse(session: &mut dyn SmokeSession, verb: &str, body: Value) -> Result<Value, SmokeFailure> {
    let sent = json!({"verb": verb, "body": body});
    let reply = session.request(verb, body);
    smoke_check!(
        reply.is_err(),
        "request_refused",
        json!(reply.as_ref().ok()),
        sent
    );
    Ok(reply
        .err()
        .map(|reply| reply["error"].clone())
        .unwrap_or_default())
}
