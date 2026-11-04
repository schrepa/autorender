use std::path::Path;

use engine_core::glam::Vec3;
use engine_core::{BrickList, Camera, Engine, Projection};
use engine_registry::Registry;
use engine_scene::{SceneDocument, SceneKinds};
use engine_terrain::TileKey;
use serde_json::{Map, Value, json};

use crate::{Report, hash};

const FIXTURE: &str = "web/scripts/fixtures/terrain.scene.json";
const ASPECT: f32 = 16.0 / 9.0;
const FRAMES: u32 = 400;
const SAMPLES_PER_FRAME: u32 = 1 << 18;
const LISTING: &str = "terrain_listing";
const SECTIONS: [&str; 3] = ["terrain_stats", "water_stats", "terrain_shading_stats"];
const FIELDS: [&str; 30] = [
    "terrains",
    "resident_tiles",
    "loading_tiles",
    "missing_tiles",
    "waiting_tiles",
    "cached_tiles",
    "cached_bytes",
    "restored_tiles",
    "prefetching_tiles",
    "rewatered_tiles",
    "stale_water_tiles",
    "settled_records",
    "settling_records",
    "shaded_tiles",
    "shading_records",
    "tile_cut",
    "fallback_tiles",
    "baking",
    "lit",
    "lighting",
    "bounced",
    "bouncing",
    "colored",
    "coloring",
    "bridges",
    "tunnels",
    "falls",
    "white_water",
    "surroundings_tiles",
    "loading",
];

pub fn run(trace: Option<&Path>, report: &mut Report) -> Result<(), String> {
    let mut engine = load()?;
    let mut lines = Vec::with_capacity(FRAMES as usize);
    for frame in 0..FRAMES {
        *engine.app.world_mut().resource_mut::<Camera>() = camera(frame);
        engine.advance(1.0 / 60.0);
        let world = engine
            .extract_frame(ASPECT, 1.0)
            .map_err(|error| format!("extract_frame: {error}"))?
            .world();
        let mut line = format!("{frame}: {}", stats(world)?);
        for (surroundings, mut keys, drawn) in listing(engine.app.world_mut())? {
            keys.sort_unstable();
            line.push_str(&format!(
                " | {} tiles {} drawn {}",
                if surroundings {
                    "surroundings"
                } else {
                    "terrain"
                },
                hash(format!("{keys:?}").as_bytes()),
                drawn,
            ));
        }
        lines.push(line);
    }
    let text = lines.join("\n") + "\n";
    if let Some(path) = trace {
        if let Some(folder) = path.parent() {
            std::fs::create_dir_all(folder)
                .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
        }
        std::fs::write(path, &text)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    }
    report.record("terrain trace", hash(text.as_bytes()));
    report.record("terrain frames", FRAMES);
    report.record(
        "terrain last frame",
        lines.last().cloned().unwrap_or_default(),
    );
    Ok(())
}

fn stats(world: &bevy_ecs::world::World) -> Result<Value, String> {
    let registry = world
        .get_resource::<Registry>()
        .ok_or("the engine holds no brick registry")?;
    let sections: Vec<Value> = SECTIONS
        .iter()
        .map(|key| {
            registry
                .stats_section(key)
                .map(|entry| entry.read(world))
                .ok_or_else(|| format!("no listed brick keeps the stats section {key}"))
        })
        .collect::<Result<_, _>>()?;
    let mut fields = Map::new();
    for field in FIELDS {
        let value = sections
            .iter()
            .find_map(|section| section.get(field))
            .ok_or_else(|| format!("no terrain stats section holds {field}"))?;
        fields.insert(field.to_owned(), value.clone());
    }
    Ok(Value::Object(fields))
}

fn listing(world: &mut bevy_ecs::world::World) -> Result<Vec<(bool, Vec<TileKey>, bool)>, String> {
    let listed = engine::host::run_operation(world, LISTING, json!({}))
        .map_err(|error| format!("{LISTING}: {error}"))?;
    let unread = || format!("{LISTING} answered {listed}");
    listed["terrains"]
        .as_array()
        .ok_or_else(unread)?
        .iter()
        .map(|terrain| {
            let tiles: Vec<[u16; 3]> =
                serde_json::from_value(terrain["tiles"].clone()).map_err(|_| unread())?;
            let keys: Vec<TileKey> = tiles
                .into_iter()
                .map(|[level, x, z]| {
                    u8::try_from(level)
                        .map(|level| TileKey::new(level, x, z))
                        .map_err(|_| unread())
                })
                .collect::<Result<_, _>>()?;
            match (
                terrain["surroundings"].as_bool(),
                terrain["drawn"].as_bool(),
            ) {
                (Some(surroundings), Some(drawn)) => Ok((surroundings, keys, drawn)),
                _ => Err(unread()),
            }
        })
        .collect()
}

fn camera(frame: u32) -> Camera {
    let overview = (Vec3::new(0.0, 80.0, 140.0), Vec3::new(0.0, 30.0, 0.0));
    let across = (Vec3::new(380.0, 90.0, 380.0), Vec3::new(450.0, 40.0, 450.0));
    let corner = (Vec3::new(-420.0, 120.0, -420.0), Vec3::new(0.0, 30.0, 0.0));
    let (position, target) = match frame {
        0..120 => overview,
        120..240 => {
            let along = (frame - 120) as f32 / 120.0;
            (
                overview.0.lerp(across.0, along),
                overview.1.lerp(across.1, along),
            )
        }
        240..330 => corner,
        _ => overview,
    };
    Camera {
        position,
        target,
        near: 0.5,
        far: 4000.0,
        projection: Projection::Perspective {
            vertical_fov_radians: std::f32::consts::FRAC_PI_3,
        },
    }
}

fn load() -> Result<Engine, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("cannot locate the workspace above tools/coresmoke")?;
    let path = root.join(FIXTURE);
    let source = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {FIXTURE}: {error}"))?;
    let mut engine = engine::engine(engine_standard::everything::<BrickList>)
        .map_err(|error| error.to_string())?;
    let kinds = SceneKinds::of(engine.app.world());
    let document = SceneDocument::from_json(&source).map_err(|error| error.to_string())?;
    document
        .validate(&kinds)
        .map_err(|error| error.to_string())?;
    let files = engine_native::files::scene_files(&path, &source, &kinds)
        .map_err(|error| error.to_string())?;
    let _scene = document
        .instantiate_with(engine.app.world_mut(), &files)
        .map_err(|error| format!("{FIXTURE}: {error}"))?;
    engine::host::apply_setting(
        engine.app.world_mut(),
        "terrain.samples_per_frame",
        &json!(SAMPLES_PER_FRAME),
    )
    .map_err(|error| error.to_string())?;
    Ok(engine)
}
