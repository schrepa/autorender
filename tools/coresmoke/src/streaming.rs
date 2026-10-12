//! Terrain streaming as a deterministic trace: the terrain fixture scene
//! (web/scripts/fixtures/terrain.scene.json) with every engine feature,
//! its terrain streaming in place, with neither a clock nor job workers, a
//! slice of samples a frame, while the camera
//! flies a fixed path over it: it looks the terrain over, flies across it,
//! jumps to a far corner and returns. Each frame's line holds the
//! streaming counters and hashes of each terrain's resident tiles and the
//! patches it draws; equal traces between two builds mean equal streaming.

use std::path::Path;

use engine::EnginePlugins;
use engine_core::glam::Vec3;
use engine_core::{Camera, Engine, EngineConfig, RenderSnapshot, TerrainSettings};
use engine_scene::SceneDocument;

use crate::{Report, hash};

const FIXTURE: &str = "web/scripts/fixtures/terrain.scene.json";
const ASPECT: f32 = 16.0 / 9.0;
/// The path's frames: the overview, the flight across, the far corner
/// and the overview again.
const FRAMES: u32 = 400;
const SAMPLES_PER_FRAME: u32 = 1 << 18;

pub fn run(trace: Option<&Path>, report: &mut Report) -> Result<(), String> {
    let mut engine = load()?;
    let mut snapshot = RenderSnapshot::default();
    let mut lines = Vec::with_capacity(FRAMES as usize);
    for frame in 0..FRAMES {
        *engine.app.world_mut().resource_mut::<Camera>() = camera(frame);
        engine.advance(1.0 / 60.0);
        engine.extract(ASPECT, &mut snapshot);
        let terrains = &snapshot.terrain;
        let mut line = format!("{frame}: {:?}", terrains.stats);
        for terrain in &terrains.terrains {
            let mut keys: Vec<_> = terrain.tiles.iter().map(|tile| tile.key()).collect();
            keys.sort_unstable();
            line.push_str(&format!(
                " | {} tiles {} patches {}",
                if terrain.surroundings {
                    "surroundings"
                } else {
                    "terrain"
                },
                hash(format!("{keys:?}").as_bytes()),
                hash(format!("{:?}", terrain.patches).as_bytes()),
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

/// The camera at `frame` of the path, in world meters (the terrain spans
/// -512 to 512 along x and z).
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
        vertical_fov_radians: std::f32::consts::FRAC_PI_3,
        near: 0.5,
        far: 4000.0,
    }
}

/// An engine with every feature holding the fixture scene, its files read
/// from beside it as the native host reads them.
fn load() -> Result<Engine, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("cannot locate the workspace above tools/coresmoke")?;
    let path = root.join(FIXTURE);
    let source = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {FIXTURE}: {error}"))?;
    let document = SceneDocument::from_json(&source).map_err(|error| error.to_string())?;
    let files = engine_native::files::scene_files(&path, &source)
        .map_err(|error| format!("{FIXTURE}: {}", error.message))?;
    let mut engine = Engine::new(EngineConfig::default());
    engine.app.add_plugins(EnginePlugins);
    // The scene stays for the engine's life.
    let _scene = document
        .instantiate_with(engine.app.world_mut(), &files)
        .map_err(|error| format!("{FIXTURE}: {error}"))?;
    // Frames work a slice of this many samples each, without a clock:
    // enough that the views load within the path's legs.
    engine
        .app
        .world_mut()
        .resource_mut::<TerrainSettings>()
        .samples_per_frame = SAMPLES_PER_FRAME;
    Ok(engine)
}
