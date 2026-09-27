use engine_core::{
    AnimatedProperty, AnimationChannel, AnimationPlayer, AssetBudget, AssetLibrary, Camera,
    InterpolateTransform, KeyInterpolation, Keyframes, Skin, bevy_ecs::world::World, glam,
};

use crate::{SceneDocument, SceneFiles, encode_animation_buffer};

// A two-bone rig: `hips` plays `sway` on `hips` and `arm`, whose skin
// deforms `body`, an inline triangle weighted to both bones.
const RIG: &str = r#"{
  "version": 8,
  "id": "rig",
  "revision": 1,
  "materials": [{"id": "skin", "color": [0.9, 0.7, 0.6, 1.0]}],
  "meshes": [{"id": "body", "source": {"kind": "inline", "vertices": [
    {"position": [0, 0, 0], "normal": [0, 0, 1], "joints": [0, 0, 0, 0], "weights": [1, 0, 0, 0]},
    {"position": [1, 0, 0], "normal": [0, 0, 1], "joints": [0, 1, 0, 0], "weights": [1, 1, 0, 0]},
    {"position": [0, 1, 0], "normal": [0, 0, 1], "joints": [1, 0, 0, 0], "weights": [1, 0, 0, 0]}
  ], "indices": [0, 1, 2]}}],
  "animations": [{"id": "sway", "name": "Sway", "targets": ["hips", "arm"], "channels": [
    {"target": 0, "property": "rotation", "times": [0, 1], "values": [0, 0, 0, 1, 0, 0.7071068, 0, 0.7071068]},
    {"target": 1, "property": "translation", "interpolation": "step", "times": [0, 0.5], "values": [0, 1, 0, 0, 2, 0]}
  ]}],
  "skins": [{"id": "bones", "joints": ["hips", "arm"], "inverse_binds": [
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1]
  ]}],
  "entities": [
    {"id": "hips", "animation": {"clip": "sway", "speed": 0.5, "loop": false}},
    {"id": "arm", "parent": "hips", "transform": {"translation": [0, 1, 0]}},
    {"id": "body", "mesh": "body", "material": "skin", "skin": "bones"}
  ]
}"#;

fn world() -> World {
    let mut world = World::new();
    world.insert_resource(AssetLibrary::new(AssetBudget::default()));
    world.insert_resource(Camera::default());
    world
}

#[test]
fn version_eight_players_and_skins_bind_their_entities() {
    let document = SceneDocument::from_json(RIG).unwrap();
    let source = document.to_json().unwrap();
    assert_eq!(
        SceneDocument::from_json(&source)
            .unwrap()
            .to_json()
            .unwrap(),
        source
    );
    // Defaults stay out of canonical source.
    assert!(!source.contains("\"time\""));
    let mut world = world();
    let instance = document.instantiate(&mut world).unwrap();
    let [hips, arm, body] = ["hips", "arm", "body"].map(|id| instance.entity(id).unwrap());
    let player = world.get::<AnimationPlayer>(hips).unwrap();
    assert_eq!(player.targets, [Some(hips), Some(arm)]);
    assert_eq!(
        (player.speed, player.looping, player.time),
        (0.5, false, 0.0)
    );
    assert_eq!(Some(player.clip), instance.clip("sway"));
    // Animated targets present smoothly between ticks.
    assert!(world.get::<InterpolateTransform>(hips).is_some());
    assert!(world.get::<InterpolateTransform>(arm).is_some());
    assert!(world.get::<InterpolateTransform>(body).is_none());
    let clip = world
        .resource::<AssetLibrary>()
        .clip(player.clip)
        .unwrap()
        .clone();
    assert_eq!(clip.duration(), 1.0);
    assert_eq!(clip.channels()[1].interpolation(), KeyInterpolation::Step);
    let skin = world.get::<Skin>(body).unwrap();
    assert_eq!(skin.joints, [hips, arm]);
    assert_eq!(
        skin.inverse_bind(1),
        glam::Mat4::from_translation(glam::Vec3::new(0.0, -1.0, 0.0))
    );
    // Disposal releases the clip with the other assets.
    instance.despawn(&mut world);
    assert_eq!(world.resource::<AssetLibrary>().clip_count(), 0);
}

#[test]
fn animations_and_skins_require_version_eight_and_valid_references() {
    let reject = |source: String, code: &str, path: &str| {
        let error = SceneDocument::from_json(&source).unwrap_err();
        assert_eq!(
            (error.code, error.path.as_str()),
            (code, path),
            "{}",
            error.message
        );
    };
    reject(
        RIG.replace("\"version\": 8", "\"version\": 7"),
        "version_required",
        "$.meshes[0].source.vertices",
    );
    reject(
        RIG.replace("\"clip\": \"sway\"", "\"clip\": \"missing\""),
        "unknown_reference",
        "$.entities[0].animation.clip",
    );
    reject(
        RIG.replace(
            "\"targets\": [\"hips\", \"arm\"]",
            "\"targets\": [\"hips\", \"leg\"]",
        ),
        "unknown_reference",
        "$.entities[0].animation.clip",
    );
    // A channel naming a target beyond the list.
    reject(
        RIG.replace(
            "\"targets\": [\"hips\", \"arm\"]",
            "\"targets\": [\"hips\"]",
        ),
        "invalid_animation",
        "$.animations[0]",
    );
    reject(
        RIG.replace("\"times\": [0, 1]", "\"times\": [1, 0]"),
        "invalid_animation",
        "$.animations[0].channels[0]",
    );
    // The mesh names two joints; one is too few.
    reject(
        RIG.replace("\"joints\": [\"hips\", \"arm\"]", "\"joints\": [\"hips\"]")
            .replace("[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1]", "")
            .replace(
                "[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],",
                "[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]",
            ),
        "invalid_skin",
        "$.entities[2].skin",
    );
    reject(
        RIG.replace(
            "[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1]",
            "[1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1]",
        ),
        "invalid_skin",
        "$.skins[0].inverse_binds[1]",
    );
    reject(
        RIG.replace(
            "{\"id\": \"body\", \"mesh\": \"body\", \"material\": \"skin\", \"skin\": \"bones\"}",
            "{\"id\": \"body\", \"skin\": \"bones\"}",
        ),
        "invalid_skin",
        "$.entities[2].skin",
    );
}

#[test]
fn clips_load_from_animation_buffers_and_prefab_targets_resolve_per_placement() {
    let channel = AnimationChannel::new(
        0,
        AnimatedProperty::Scale,
        KeyInterpolation::CubicSpline,
        vec![0.0, 2.0],
        Keyframes::Vectors(
            vec![
                glam::Vec3::ZERO,
                glam::Vec3::ONE,
                glam::Vec3::ZERO,
                glam::Vec3::ZERO,
                glam::Vec3::splat(2.0),
                glam::Vec3::ZERO,
            ]
            .into(),
        ),
    )
    .unwrap();
    let bytes = encode_animation_buffer(std::slice::from_ref(&channel));
    assert_eq!(&bytes[..4], b"EAN1");
    let source = r#"{
      "version": 8, "id": "lanterns", "revision": 1,
      "materials": [], "meshes": [],
      "animations": [{"id": "pulse", "targets": ["glow"], "uri": "clips/pulse.bin"}],
      "prefabs": [{"id": "lantern", "entities": [
        {"id": "glow"},
        {"id": "base", "animation": {"clip": "pulse"}}
      ]}],
      "entities": [
        {"id": "west", "prefab": "lantern"},
        {"id": "east", "prefab": "lantern", "transform": {"translation": [4, 0, 0]}}
      ]
    }"#;
    let document = SceneDocument::from_json(source).unwrap();
    assert_eq!(document.files()[0].uri, "clips/pulse.bin");
    let mut world = world();
    assert_eq!(
        document.instantiate(&mut world).unwrap_err().code,
        "missing_file"
    );
    let mut files = SceneFiles::new();
    files.add_buffer("clips/pulse.bin", bytes.clone());
    let instance = document.instantiate_with(&mut world, &files).unwrap();
    for placement in ["west", "east"] {
        let player = world
            .get::<AnimationPlayer>(instance.entity(&format!("{placement}.base")).unwrap())
            .unwrap();
        assert_eq!(
            player.targets,
            [Some(instance.entity(&format!("{placement}.glow")).unwrap())]
        );
    }
    let clip = world
        .resource::<AssetLibrary>()
        .clip(instance.clip("pulse").unwrap())
        .unwrap()
        .clone();
    assert_eq!(clip.channels(), [channel]);
    // Truncated and trailing bytes are rejected.
    for broken in [
        &bytes[..bytes.len() - 4],
        &[&bytes[..], &[0; 4]].concat()[..],
    ] {
        let mut files = SceneFiles::new();
        files.add_buffer("clips/pulse.bin", broken.to_vec());
        assert_eq!(
            document.validate_with(&files).unwrap_err().code,
            "invalid_animation"
        );
    }
}
