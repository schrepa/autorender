use std::path::Path;

use engine_core::glam::Vec3;
use engine_core::{BrickList, Camera, Engine, Projection, RenderSnapshot};
use engine_places::ground::{GroundTiles, TerrainTiles};
use engine_registry::Registry;
use engine_scene::SceneDocument;
use serde_json::{Value, json};

use crate::{Report, hash};

const FIXTURE: &str = "web/scripts/fixtures/terrain.scene.json";
const ASPECT: f32 = 16.0 / 9.0;
const FRAMES: u32 = 400;
const SAMPLES_PER_FRAME: u32 = 1 << 18;
const STATS: &str = "terrain_stats";

pub fn run(trace: Option<&Path>, report: &mut Report) -> Result<(), String> {
    let mut engine = load()?;
    let mut snapshot = RenderSnapshot::default();
    let mut lines = Vec::with_capacity(FRAMES as usize);
    for frame in 0..FRAMES {
        *engine.app.world_mut().resource_mut::<Camera>() = camera(frame);
        engine.advance(1.0 / 60.0);
        engine.extract(ASPECT, &mut snapshot);
        let world = engine.app.world();
        let stats = world
            .get_resource::<Registry>()
            .and_then(|registry| registry.stats_section(STATS))
            .map_or(Value::Null, |entry| entry.read(world));
        let mut line = format!("{frame}: {stats}");
        let held = world.get_resource::<GroundTiles>();
        let terrains = world
            .get_resource::<TerrainTiles>()
            .map(|tiles| tiles.terrains.clone())
            .unwrap_or_default();
        for terrain in terrains.iter() {
            let mut keys: Vec<_> = terrain.tiles.iter().map(|tile| tile.key).collect();
            keys.sort_unstable();
            let drawn = held.map_or(0, |held| {
                held.held
                    .iter()
                    .filter(|tile| tile.terrain == terrain.terrain)
                    .count()
            });
            line.push_str(&format!(
                " | {} tiles {} drawn {drawn}",
                if terrain.surroundings {
                    "surroundings"
                } else {
                    "terrain"
                },
                hash(format!("{keys:?}").as_bytes()),
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
    let mut engine = engine::engine(engine_standard::standard::<BrickList>)
        .map_err(|error| error.to_string())?;
    let document = SceneDocument::from_json(&source).map_err(|error| error.to_string())?;
    let files =
        engine_native::files::scene_files(&path, &source).map_err(|error| error.to_string())?;
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
