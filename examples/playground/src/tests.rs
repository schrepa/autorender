use super::*;
use engine::scene::SceneFiles;
const SCENE: &str = include_str!("../../../web/public/scenes/playground.scene.json");

#[test]
fn accessory_follows_player_and_parented_physics_reload_is_rejected() {
    let mut engine = Engine::default();
    let mut game = Playground::new(&mut engine, SCENE, &SceneFiles::default()).unwrap();
    let accessory = game
        .scene_instance()
        .unwrap()
        .entity("player-accessory")
        .unwrap();
    let local = *engine.app.world().get::<Transform>(accessory).unwrap();
    let spawn = engine
        .app
        .world()
        .get::<Transform>(game.player().unwrap())
        .unwrap()
        .translation;
    engine
        .app
        .world_mut()
        .resource_mut::<InputState>()
        .move_forward = 1.0;
    engine.advance(1.0 / 60.0);
    let player = engine
        .app
        .world()
        .resource::<GlobalTransforms>()
        .get(game.player().unwrap())
        .unwrap()
        .matrix()
        .unwrap();
    let actual = engine
        .app
        .world()
        .resource::<GlobalTransforms>()
        .get(accessory)
        .unwrap()
        .matrix()
        .unwrap();
    assert!(actual.abs_diff_eq(player * local.matrix(), 0.0001));
    let mut snapshot = RenderSnapshot::default();
    engine
        .extract_with_presentation(
            1.0,
            &mut snapshot,
            PresentationMode::Interpolated { alpha: 0.5 },
        )
        .unwrap();
    let presented_player = engine
        .app
        .world()
        .resource::<PresentedTransforms>()
        .get(game.player().unwrap())
        .unwrap()
        .matrix()
        .unwrap();
    let presented_accessory = engine
        .app
        .world()
        .resource::<PresentedTransforms>()
        .get(accessory)
        .unwrap()
        .matrix()
        .unwrap();
    assert!(
        presented_player
            .w_axis
            .truncate()
            .abs_diff_eq(spawn.lerp(player.w_axis.truncate(), 0.5), 0.0001)
    );
    assert!(presented_accessory.abs_diff_eq(presented_player * local.matrix(), 0.0001));
    assert_eq!(
        engine
            .app
            .world()
            .resource::<GlobalTransforms>()
            .get(game.player().unwrap())
            .unwrap()
            .matrix(),
        Some(player)
    );
    game.reset_player(&mut engine);
    engine
        .extract_with_presentation(
            1.0,
            &mut snapshot,
            PresentationMode::Interpolated { alpha: 0.5 },
        )
        .unwrap();
    assert!(
        engine
            .app
            .world()
            .resource::<PresentedTransforms>()
            .get(game.player().unwrap())
            .unwrap()
            .matrix()
            .unwrap()
            .w_axis
            .truncate()
            .abs_diff_eq(spawn, 0.0001)
    );
    let mut document = SceneDocument::from_json(SCENE).unwrap();
    document
        .entities
        .iter_mut()
        .find(|entity| entity.id == "player")
        .unwrap()
        .parent = Some("west-lantern".into());
    let before = game.player();
    let error = game
        .reload(
            &mut engine,
            &document.to_json().unwrap(),
            &SceneFiles::default(),
        )
        .unwrap_err();
    assert_eq!(error.code, "gameplay_transform");
    assert_eq!(game.player(), before);
    assert_eq!(game.revision(), 5);
}

#[test]
fn player_lands_on_a_solid_top_instead_of_falling_through() {
    let mut engine = Engine::default();
    let game = Playground::new(&mut engine, SCENE, &SceneFiles::default()).unwrap();
    let player = game.player().unwrap();
    edit_transform(engine.app.world_mut(), player, |transform| {
        transform.translation = Vec3::new(7.0, 3.0, 4.0);
    })
    .unwrap();
    for _ in 0..90 {
        engine.advance(1.0 / 60.0);
    }
    let position = engine
        .app
        .world()
        .get::<Transform>(player)
        .unwrap()
        .translation;
    assert!((position.y - 1.8).abs() < 0.0001);
    assert_eq!(
        engine
            .app
            .world()
            .resource::<GameProgress>()
            .collision_errors,
        0
    );
}

#[test]
fn authored_game_moves_collects_and_jumps_once_during_catchup() {
    let mut engine = Engine::default();
    let game = Playground::new(&mut engine, SCENE, &SceneFiles::default()).unwrap();
    engine
        .app
        .world_mut()
        .resource_mut::<InputState>()
        .move_forward = 1.0;
    for _ in 0..30 {
        engine.advance(1.0 / 60.0);
    }
    assert!(engine.app.world().resource::<GameProgress>().collected >= 1);
    assert_eq!(
        engine
            .app
            .world()
            .resource::<GameProgress>()
            .collision_errors,
        0
    );
    let player = game.player().unwrap();
    assert!(
        engine
            .app
            .world()
            .get::<Transform>(player)
            .unwrap()
            .translation
            .z
            < 5.0
    );
    engine
        .app
        .world_mut()
        .resource_mut::<ActionInput>()
        .ingest(1, 1, 0);
    engine.advance(0.25);
    assert_eq!(engine.app.world().resource::<GameProgress>().jumps, 1);
    assert!(
        engine
            .app
            .world()
            .get::<Transform>(player)
            .unwrap()
            .translation
            .y
            > 0.6
    );
}

#[test]
fn swept_movement_stops_at_a_thin_parented_rotated_wall() {
    let mut source: serde_json::Value = serde_json::from_str(SCENE).unwrap();
    let entities = source["entities"].as_array_mut().unwrap();
    entities.push(serde_json::json!({
        "id":"wall-parent", "transform":{"translation":[0.66,0.6,7],"rotation_degrees":[0,90,0]}
    }));
    entities.push(serde_json::json!({
        "id":"thin-wall", "parent":"wall-parent", "tags":["solid"],
        "collider":{"shape":{"kind":"cuboid","half_extents":[3,2,0.01]},"memberships":1,"filter":2,"sensor":false}
    }));
    let mut engine = Engine::default();
    let game = Playground::new(&mut engine, &source.to_string(), &SceneFiles::default()).unwrap();
    engine
        .app
        .world_mut()
        .resource_mut::<InputState>()
        .move_right = 1.0;
    for _ in 0..30 {
        engine.advance(1.0 / 60.0);
    }
    let position = engine
        .app
        .world()
        .get::<Transform>(game.player().unwrap())
        .unwrap()
        .translation;
    assert!((position.x - 0.05).abs() < 0.001, "{position:?}");
    assert!((position.y - 0.6).abs() < 0.001, "{position:?}");
    assert_eq!(
        engine
            .app
            .world()
            .resource::<GameProgress>()
            .collision_errors,
        0
    );
}

#[test]
fn incomplete_collision_snapshot_stops_player_without_tunneling() {
    let mut engine = Engine::default();
    let game = Playground::new(&mut engine, SCENE, &SceneFiles::default()).unwrap();
    let player = game.player().unwrap();
    let before = engine
        .app
        .world()
        .get::<Transform>(player)
        .unwrap()
        .translation;
    engine
        .app
        .world_mut()
        .resource_mut::<engine::collision::CollisionConfig>()
        .max_colliders = 0;
    engine
        .app
        .world_mut()
        .resource_mut::<InputState>()
        .move_forward = 1.0;
    engine.advance(1.0 / 60.0);
    assert_eq!(
        engine
            .app
            .world()
            .get::<Transform>(player)
            .unwrap()
            .translation,
        before
    );
    assert!(
        engine
            .app
            .world()
            .resource::<GameProgress>()
            .collision_errors
            > 0
    );
}

#[test]
fn repeated_tangent_movement_keeps_player_on_large_floor() {
    let mut source: serde_json::Value = serde_json::from_str(SCENE).unwrap();
    let entities = source["entities"].as_array_mut().unwrap();
    entities
        .retain(|e| ["ground", "ground-collider", "player"].contains(&e["id"].as_str().unwrap()));
    entities.iter_mut().find(|e| e["id"] == "ground").unwrap()["transform"]["scale"] =
        serde_json::json!([40, 1, 40]);
    entities.iter_mut().find(|e| e["id"] == "player").unwrap()["transform"]["translation"] =
        serde_json::json!([0, 0.6, 0]);
    let mut engine = Engine::default();
    let game = Playground::new(&mut engine, &source.to_string(), &SceneFiles::default()).unwrap();
    engine
        .app
        .world_mut()
        .resource_mut::<InputState>()
        .move_forward = 1.0;
    for tick in 0..36 {
        engine.advance(1.0 / 60.0);
        let position = engine
            .app
            .world()
            .get::<Transform>(game.player().unwrap())
            .unwrap()
            .translation;
        assert!(
            (position.y - 0.6).abs() < 0.001,
            "tick {tick}: {position:?}"
        );
    }
    assert_eq!(
        engine
            .app
            .world()
            .resource::<GameProgress>()
            .collision_errors,
        0
    );
}

#[test]
fn failed_reload_preserves_game_and_repeated_reload_retires_assets() {
    let mut engine = Engine::default();
    let mut game = Playground::new(&mut engine, SCENE, &SceneFiles::default()).unwrap();
    let initial_assets = engine
        .app
        .world()
        .resource::<AssetLibrary>()
        .resident_bytes();
    let player = game.player().unwrap();
    assert!(
        game.reload(&mut engine, "{\"version\":99}", &SceneFiles::default())
            .is_err()
    );
    assert_eq!(game.player(), Some(player));
    assert!(engine.app.world().get::<Transform>(player).is_some());
    for _ in 0..10 {
        game.reload(&mut engine, SCENE, &SceneFiles::default())
            .unwrap();
    }
    assert_eq!(
        engine
            .app
            .world()
            .resource::<AssetLibrary>()
            .resident_bytes(),
        initial_assets
    );
    let mut snapshot = RenderSnapshot::default();
    engine.extract(1.0, &mut snapshot);
    assert_eq!(snapshot.meshes.len(), 6);
    assert_eq!(snapshot.textures.len(), 1);
    assert_eq!(snapshot.missing_assets, 0);
    game.unload(&mut engine);
    engine.extract(1.0, &mut snapshot);
    assert!(snapshot.instances.is_empty());
    assert!(snapshot.textures.is_empty());
    assert_eq!(
        engine
            .app
            .world()
            .resource::<AssetLibrary>()
            .texture_count(),
        0
    );
    assert_eq!(
        engine.app.world().resource::<AssetLibrary>().mesh_count(),
        1
    ); // Compatibility cube belongs to engine.
    assert_eq!(
        engine
            .app
            .world()
            .resource::<AssetLibrary>()
            .material_count(),
        0
    );
}
