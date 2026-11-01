use bevy_ecs::prelude::World;
use glam::{EulerRot, Quat, Vec3};
use serde_json::{Value, json};

use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use engine_core::{
    AssetLibrary, Camera, Error, Material, Mesh, Mesh3d, Texture, Transform, set_parents,
    smoke_check, smoke_eq,
};

const SIDE: u32 = 24;
const VIEWS: [&str; 7] = [
    "kinds", "levels", "cards", "shadows", "casters", "overdraw", "normals",
];
const KINDS: [&str; 9] = [
    "mesh",
    "placed",
    "card",
    "placed_card",
    "terrain",
    "cover",
    "water",
    "falls",
    "sky",
];
const SLOTTED: [&str; 2] = ["mesh", "card"];
const QUIET_FRAMES: u32 = 13;

pub fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    let built = build(session.app().world_mut());
    smoke_check!(
        built.is_ok(),
        "fixture_builds",
        json!(built.as_ref().err()),
        json!(null)
    );
    ask(session, "control", json!({"name": "pause"}))?;
    environment(session, json!({"reflection_probes": {"enabled": false}}))?;
    let listing = ask(session, "describe", json!({}))?;
    let listed = listing["types"]
        .as_array()
        .is_some_and(|types| types.iter().any(|kind| kind["name"] == "DebugView"));
    smoke_check!(
        listed,
        "setting_discoverable",
        json!(listing["types"]),
        json!("DebugView")
    );
    let look = ask(session, "inspect", json!({"component": "Environment"}))?;
    smoke_eq!(
        json!(look["value"]["debug_view"]),
        json!({"view": "off", "shade": true}),
        "setting_starts_off"
    );
    let shaded = capture(session)?;
    smoke_check!(
        !passes(&shaded.0).contains(&"debug_view".to_owned()),
        "shaded_has_no_view_pass",
        json!(shaded.0["render_passes"]),
        json!("no debug_view")
    );
    let before = exact(session)?;
    let shown = views(session, &shaded)?;
    let off = environment(session, json!({"debug_view": {"view": "off"}}))?;
    smoke_eq!(
        json!(off["environment_changed"]),
        json!(["debug_view"]),
        "view_turns_off"
    );
    smoke_eq!(json!(shown_view(session)?), json!("off"), "view_reads_off");
    let again = capture(session)?;
    smoke_check!(
        !passes(&again.0).contains(&"debug_view".to_owned()) && again.0["pipelines_created"] == 0,
        "off_creates_nothing",
        json!({"passes": again.0["render_passes"], "created": again.0["pipelines_created"]}),
        json!({"created": 0})
    );
    smoke_eq!(exact(session)?, before.clone(), "off_is_exact");
    let unknown = refuse(
        session,
        "control",
        json!({"name": "environment", "args": {"debug_view": {"view": "everything"}}}),
    )?;
    smoke_check!(
        unknown["path"]
            .as_str()
            .is_some_and(|path| path.starts_with("$.args.debug_view")),
        "unknown_view_refused",
        json!(unknown),
        json!({"path": "$.args.debug_view"})
    );
    probes(session, &again.0)?;
    smoke_eq!(exact(session)?, before, "probe_keeps_image");
    smoke_check!(
        shown.len() == VIEWS.len(),
        "views_draw",
        json!(shown),
        json!(VIEWS)
    );
    Ok(())
}

fn views(
    session: &mut dyn SmokeSession,
    shaded: &(Value, Value),
) -> Result<Vec<Value>, SmokeFailure> {
    let mut shown = Vec::new();
    for view in VIEWS {
        let changed = environment(session, json!({"debug_view": {"view": view}}))?;
        smoke_eq!(
            json!(changed["environment_changed"]),
            json!(["debug_view"]),
            "view_changes_setting"
        );
        smoke_eq!(json!(shown_view(session)?), json!(view), "view_reads_back");
        let (rendering, image) = capture(session)?;
        smoke_check!(
            passes(&rendering).contains(&"debug_view".to_owned()),
            "view_pass_runs",
            json!({"view": view, "passes": rendering["render_passes"]}),
            json!("debug_view")
        );
        smoke_check!(
            rendering["draw_calls"].as_u64().unwrap_or(0) > 0,
            "view_draws",
            json!({"view": view, "draw_calls": rendering["draw_calls"]}),
            json!({"draw_calls": 1})
        );
        smoke_check!(
            image != shaded.1,
            "view_changes_image",
            json!({"view": view, "image": image}),
            json!({"not": shaded.1})
        );
        shown.push(json!({"view": view, "rendering": rendering, "image": image}));
    }
    let image = |view: &str| {
        shown
            .iter()
            .find(|frame| frame["view"] == view)
            .map(|frame| frame["image"].clone())
            .unwrap_or_default()
    };
    smoke_check!(
        image("levels") != image("kinds"),
        "levels_differ_from_kinds",
        json!(image("levels")),
        json!({"not": image("kinds")})
    );
    let overdraw = shown
        .iter()
        .find(|frame| frame["view"] == "overdraw")
        .map(|frame| frame["rendering"]["pipelines"].as_u64().unwrap_or(0))
        .unwrap_or(0);
    let pipelines = shaded.0["pipelines"].as_u64().unwrap_or(0);
    smoke_check!(
        overdraw > pipelines,
        "views_create_pipelines_when_shown",
        json!(overdraw),
        json!({"more_than": pipelines})
    );
    Ok(shown)
}

fn probes(session: &mut dyn SmokeSession, rendering: &Value) -> SmokeResult {
    let width = rendering["resolution"]["display_width"]
        .as_u64()
        .unwrap_or(0);
    let height = rendering["resolution"]["display_height"]
        .as_u64()
        .unwrap_or(0);
    let mut kinds = Vec::new();
    for fraction in [0.3, 0.4, 0.5, 0.6, 0.7] {
        let x = (width as f64 * fraction).floor() as u64;
        let y = (height as f64 * 0.55).floor() as u64;
        let probed = control(session, "probe", json!({"x": x, "y": y, "alpha": 0}))?;
        smoke_eq!(json!(probed["pixel"]), json!([x, y]), "probe_names_pixel");
        let kind = probed["kind"].as_str().unwrap_or_default().to_owned();
        smoke_check!(
            KINDS.contains(&kind.as_str()),
            "probe_kind",
            json!(probed),
            json!(KINDS)
        );
        smoke_check!(
            probed["shaded"]
                .as_array()
                .is_some_and(|shaded| shaded.len() == 3),
            "probe_shaded",
            json!(probed),
            json!({"shaded": [0, 0, 0]})
        );
        if kind != "sky" {
            smoke_check!(
                surface(&probed),
                "probe_surface",
                json!(probed),
                json!("surface")
            );
        }
        if SLOTTED.contains(&kind.as_str()) {
            smoke_check!(
                probed["object"].is_number() && probed["entity"].is_object(),
                "probe_object",
                json!(probed),
                json!({"object": 0, "entity": {}})
            );
        }
        kinds.push(kind);
    }
    smoke_check!(
        kinds.iter().any(|kind| SLOTTED.contains(&kind.as_str())),
        "probe_finds_entity",
        json!(kinds),
        json!(SLOTTED)
    );
    let sky = control(
        session,
        "probe",
        json!({"x": width / 2, "y": 2, "alpha": 0}),
    )?;
    smoke_check!(
        sky["kind"]
            .as_str()
            .is_some_and(|kind| KINDS.contains(&kind)),
        "probe_top",
        json!(sky),
        json!(KINDS)
    );
    let outside = refuse(
        session,
        "control",
        json!({"name": "probe", "args": {"x": width, "y": 0}}),
    )?;
    smoke_eq!(
        json!({"code": outside["code"], "path": outside["path"]}),
        json!({"code": "out_of_range", "path": "$.args.x"}),
        "probe_refuses_outside"
    );
    Ok(())
}

fn surface(probed: &Value) -> bool {
    let depth = probed["depth"].as_f64().unwrap_or(0.0);
    let world = probed["world"].as_array().is_some_and(|world| {
        world
            .iter()
            .all(|value| value.as_f64().is_some_and(f64::is_finite))
    });
    let normal = probed["normal"].as_array().map_or(0.0, |normal| {
        normal
            .iter()
            .map(|value| value.as_f64().unwrap_or(0.0).powi(2))
            .sum::<f64>()
            .sqrt()
    });
    depth > 0.0
        && depth <= 1.0
        && world
        && probed["fragments"].as_u64().unwrap_or(0) >= 1
        && (normal - 1.0).abs() < 0.01
}

fn settle(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    let mut compiling = true;
    for _ in 0..600 {
        let rendering = present(session)?;
        compiling = rendering["pipelines_compiling"] == true;
        if !compiling {
            break;
        }
    }
    smoke_check!(
        !compiling,
        "pipelines_compile",
        json!(compiling),
        json!(false)
    );
    let (mut previous, mut stable, mut rendering) = (Value::Null, 0, Value::Null);
    let mut calm = 0;
    for frame in 1..=600_u64 {
        if frame > 8 && stable >= 5 && calm >= QUIET_FRAMES {
            return Ok(rendering);
        }
        rendering = present(session)?;
        let detail = &rendering["level_of_detail"];
        let counts = json!([
            detail["lod0_instances"],
            detail["lod1_instances"],
            detail["lod2_instances"],
            detail["impostor_instances"]
        ]);
        let current = rendering["read_back_age"]
            .as_u64()
            .is_some_and(|age| age < frame);
        stable = if current && counts == previous {
            stable + 1
        } else {
            0
        };
        previous = if current { counts } else { Value::Null };
        calm = if quiet(&rendering["shadows"]) {
            calm + 1
        } else {
            0
        };
    }
    smoke_check!(
        stable >= 5,
        "read_back_settles",
        json!(rendering),
        json!({"stable": 5})
    );
    smoke_check!(
        calm >= QUIET_FRAMES,
        "shadows_settle",
        json!(rendering["shadows"]),
        json!({"quiet_frames": QUIET_FRAMES})
    );
    Ok(rendering)
}

fn quiet(shadows: &Value) -> bool {
    shadows["tile_cut"]["cut"].as_u64() == Some(0)
        && shadows["moving_cut"]["kept"].as_u64() == Some(0)
        && ["tiles_drawn", "sky_drawn"]
            .iter()
            .all(|field| shadows[*field].as_u64() == Some(0))
}

fn present(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    Ok(control(session, "present", json!({"alpha": 0}))?["rendering"].clone())
}

fn capture(session: &mut dyn SmokeSession) -> Result<(Value, Value), SmokeFailure> {
    settle(session)?;
    let captured = ask(session, "capture", json!({"alpha": 0}))?;
    let image = captured["artifact"]["sha256"].clone();
    smoke_check!(
        image.is_string(),
        "capture_keeps_image",
        json!(captured["artifact"]),
        json!("sha256")
    );
    Ok((captured["rendering"].clone(), image))
}

fn exact(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    let mut previous = Value::Null;
    for _ in 0..20 {
        let captured = ask(session, "capture", json!({"alpha": 0}))?;
        let image = captured["artifact"]["sha256"].clone();
        if image.is_string() && image == previous {
            return Ok(image);
        }
        previous = image;
    }
    smoke_check!(
        false,
        "exact_captures_agree",
        json!(previous),
        json!("two equal captures")
    );
    Ok(previous)
}

fn passes(rendering: &Value) -> Vec<String> {
    rendering["render_passes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|pass| pass.as_str().map(str::to_owned))
        .collect()
}

fn shown_view(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    Ok(ask(session, "inspect", json!({"component": "DebugView"}))?["value"]["view"].clone())
}

fn environment(session: &mut dyn SmokeSession, sections: Value) -> Result<Value, SmokeFailure> {
    control(session, "environment", sections)
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

fn checker() -> Vec<u8> {
    let size = 32_u32;
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let shade = 150 + (x * 7 + y * 13) % 48;
            let light = ((x >> 2) + (y >> 2)) % 2 == 0;
            let rgb = if light {
                [shade + 40, shade + 30, shade]
            } else {
                [shade - 80, shade - 70, shade - 50]
            };
            pixels.extend(rgb.map(|channel| u8::try_from(channel).unwrap_or(u8::MAX)));
            pixels.push(255);
        }
    }
    pixels
}

fn build(world: &mut World) -> Result<(), String> {
    let found = |error: Error| format!("{error:?}");
    let paints = [
        [0.85, 0.35, 0.2, 1.0],
        [0.25, 0.6, 0.85, 1.0],
        [0.4, 0.75, 0.3, 1.0],
        [0.9, 0.75, 0.25, 1.0],
    ];
    let mut assets = world.resource_mut::<AssetLibrary>();
    let checker = Texture::new(32, 32, checker()).map_err(found)?;
    let checker = assets.insert_texture(checker).map_err(found)?;
    let ground = Material::textured([0.55, 0.58, 0.52, 1.0], checker).map_err(found)?;
    let ground = assets.insert_material(ground).map_err(found)?;
    let mut painted = Vec::with_capacity(paints.len());
    for paint in paints {
        let material = Material::textured(paint, checker).map_err(found)?;
        painted.push(assets.insert_material(material).map_err(found)?);
    }
    let cube = assets.insert_mesh(Mesh::cube()).map_err(found)?;
    let gem = assets.insert_mesh(Mesh::octahedron()).map_err(found)?;
    world.insert_resource(Camera {
        position: Vec3::new(0.0, 30.0, SIDE as f32 * 2.6),
        target: Vec3::ZERO,
        vertical_fov_radians: 60.0_f32.to_radians(),
        near: 0.1,
        far: 1000.0,
    });
    world.spawn((
        Transform {
            translation: Vec3::new(0.0, -0.5, 0.0),
            scale: Vec3::new(SIDE as f32 * 6.0, 1.0, SIDE as f32 * 6.0),
            ..Transform::IDENTITY
        },
        Mesh3d {
            mesh: cube,
            material: ground,
        },
    ));
    let mut caps = Vec::new();
    for index in 0..SIDE * SIDE {
        let (row, column) = (index / SIDE, index % SIDE);
        let scale = 0.8 + (index % 5) as f32 * 0.3;
        let middle = (SIDE - 1) as f32 / 2.0;
        let piece = world
            .spawn((
                Transform {
                    translation: Vec3::new(
                        (column as f32 - middle) * 4.0,
                        scale / 2.0,
                        (row as f32 - middle) * 4.0,
                    ),
                    rotation: Quat::from_euler(
                        EulerRot::XYZ,
                        0.0,
                        (((index * 37) % 360) as f32).to_radians(),
                        0.0,
                    ),
                    scale: Vec3::splat(scale),
                },
                Mesh3d {
                    mesh: if index % 3 == 0 { gem } else { cube },
                    material: painted[(index % 4) as usize],
                },
            ))
            .id();
        if index % 4 == 0 {
            let cap = world
                .spawn((
                    Transform {
                        translation: Vec3::new(0.0, 0.75, 0.0),
                        scale: Vec3::splat(0.5),
                        ..Transform::IDENTITY
                    },
                    Mesh3d {
                        mesh: cube,
                        material: painted[((index + 1) % 4) as usize],
                    },
                ))
                .id();
            caps.push((cap, Some(piece)));
        }
    }
    set_parents(world, &caps).map_err(|error| format!("{error:?}"))
}
