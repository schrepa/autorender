use std::path::Path;

use bevy_ecs::world::World;
use engine::scene::{SceneDocument, SceneKinds};
use engine_core::glam::Vec3;
use engine_core::{
    BrickList, Camera, Engine, Operation, Projection, PureSession, Readiness, Section, Teleport,
    Transform, main_camera, place_camera, send_ask,
};
use engine_terrain::{TerrainListing, TerrainStats, TileKey};
use engine_terrain_shading::TerrainShadingStats;
use engine_water::WaterStats;
use serde_json::{Map, Value, json};

use super::hash;

const TRACE: &str = "target/pure/caches-trace.txt";
const FIXTURE: &str = "web/scripts/fixtures/terrain.scene.json";
const FRAMES: u32 = 400;
const FIELDS: [&str; 25] = [
    "terrains",
    "resident_tiles",
    "loading_tiles",
    "missing_tiles",
    "cached_tiles",
    "cached_bytes",
    "restored_tiles",
    "prefetching_tiles",
    "rewatered_tiles",
    "stale_water_tiles",
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
    "surroundings_tiles",
];

pub fn run(session: &mut dyn PureSession) -> Result<(), String> {
    let mut engine = load(session.root())?;
    let main = main_camera(engine.app.world()).ok_or("the engine holds no main view")?;
    let mut lines = Vec::with_capacity(FRAMES as usize);
    for frame in 0..FRAMES {
        let (at, cut) = camera(frame);
        let world = engine.app.world_mut();
        if cut {
            place_camera(world, main, at, lens());
        } else {
            send_ask(world, Teleport { node: main, to: at });
        }
        engine.step_fixed(1).map_err(|error| error.to_string())?;
        let world = engine
            .extract_frame()
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
    let path = session.root().join(TRACE);
    if let Some(folder) = path.parent() {
        std::fs::create_dir_all(folder)
            .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
    }
    std::fs::write(&path, &text)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    session.record("terrain trace", hash(text.as_bytes()).into());
    session.record("terrain frames", FRAMES.into());
    session.record(
        "terrain last frame",
        lines.last().cloned().unwrap_or_default().into(),
    );
    Ok(())
}

fn section<S: Section>(world: &World) -> Result<Value, String> {
    let stats = world
        .get_resource::<S>()
        .ok_or_else(|| format!("no listed brick keeps the stats section {}", S::KEY))?;
    serde_json::to_value(stats).map_err(|error| error.to_string())
}

fn stats(world: &World) -> Result<Value, String> {
    let sections = [
        section::<TerrainStats>(world)?,
        section::<WaterStats>(world)?,
        section::<TerrainShadingStats>(world)?,
    ];
    let mut fields = Map::new();
    for field in FIELDS {
        let value = sections
            .iter()
            .find_map(|section| section.get(field))
            .ok_or_else(|| format!("no terrain stats section holds {field}"))?;
        fields.insert(field.to_owned(), value.clone());
    }
    let readiness = world
        .get_resource::<Readiness>()
        .ok_or("the engine holds no readiness")?;
    fields.insert("loading".to_owned(), json!(readiness.loading));
    Ok(Value::Object(fields))
}

fn listing(world: &mut World) -> Result<Vec<(bool, Vec<TileKey>, bool)>, String> {
    let listed = engine::host::call::<TerrainListing>(world, TerrainListing {})
        .map_err(|error| format!("{}: {error}", TerrainListing::NAME))?;
    listed
        .terrains
        .into_iter()
        .map(|terrain| {
            let keys: Vec<TileKey> = terrain
                .tiles
                .into_iter()
                .map(|[level, x, z]| {
                    u8::try_from(level)
                        .map(|level| TileKey::new(level, x, z))
                        .map_err(|_| format!("{} answered level {level}", TerrainListing::NAME))
                })
                .collect::<Result<_, _>>()?;
            Ok((terrain.surroundings, keys, terrain.drawn))
        })
        .collect()
}

fn lens() -> Camera {
    Camera {
        near: 0.5,
        far: 4000.0,
        projection: Projection::Perspective {
            vertical_fov_radians: std::f32::consts::FRAC_PI_3,
        },
    }
}

fn camera(frame: u32) -> (Transform, bool) {
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
    (
        Transform::looking_at(position, target),
        matches!(frame, 0 | 240 | 330),
    )
}

fn load(root: &Path) -> Result<Engine, String> {
    let path = root.join(FIXTURE);
    let source = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {FIXTURE}: {error}"))?;
    let mut engine = engine::engine(engine_standard::everything::<BrickList>)
        .map_err(|error| error.to_string())?;
    engine.set_frame(super::FRAME);
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
    Ok(engine)
}
