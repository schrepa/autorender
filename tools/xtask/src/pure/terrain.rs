//! The terrain corpus: the tracked terrain fixture's generated terrain
//! (web/scripts/fixtures/terrain.scene.json and its dome heightmap). Its
//! bakes; every tile of its pyramid and some of its surroundings'; tiles
//! under edits; its lighting; a walk; and a pack of what they made. Each
//! job runs as workers run it, whole and a slice at a time, and must give
//! the bits its code gives in place; every record must read back from the
//! pack the cook would write as it went in.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use engine_core::Terrain;
use engine_scene::{SceneDocument, SceneFileKind, SceneFiles};
use engine_terrain::jobs::{
    self, BakeJob, LightingJob, StackView, TerrainRecipe, TerrainWorker, TileJob, WalkJob,
};
use engine_terrain::lighting::{self, Ground};
use engine_terrain::{
    BiomeRules, Generator, Layer, Layers, NavOptions, Obstacle, PATCH_QUADS, PATCHES_PER_TILE,
    Pyramid, Stamp, StampOp, TILE_QUADS, Tile, TileKey, codec, walk,
};
use glam::Vec2;
use serde_json::json;

use super::{Observations, hash};

/// The fixture scene and the entity holding its terrain.
const FIXTURE: &str = "web/scripts/fixtures/terrain.scene.json";
const ENTITY: &str = "ground";
/// Work units each slice of a job gets: few enough that each takes many
/// slices, as workers slice them.
const SLICE: u64 = 4096;
/// The terrain settings' pinned levels, whose finest its lighting reads,
/// and the texels along its sky and horizon maps' sides.
const PINNED: u32 = 2;
const SKY_TEXELS: u32 = 256;
const HORIZON_TEXELS: u32 = 128;
/// Every share and job the corpus ran through its worker, in order, each
/// job's output by its hash: what `cargo xtask pure --wasm` replays in the
/// browser worker's Wasm module.
pub const JOBS: &str = "target/pure/terrain-jobs.json";

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let terrain = load(root)?;
    let authored = &terrain.layers;
    let pyramid = terrain.pyramid;
    let mut worker = Worker::default();

    // The bakes in place, and as a worker's job.
    let baked = authored.bake_all(&pyramid);
    let bytes = codec::encode(&baked);
    observed.hash("bakes", &bytes);
    let mut records = vec![(BakeJob::KIND.name(), bytes.clone())];
    observed.check(
        worker.run(
            BakeJob::KIND.name(),
            &BakeJob {
                layers: Layers::clone(authored),
                pyramid,
            }
            .input(),
        )? == bytes,
        || "the bake job differs from the bakes in place".into(),
    );
    record_bakes(&baked, observed);

    let baked = Arc::new(baked);
    let rules = terrain.surface.rules.clone();
    worker.share("stack", &jobs::share_stack(&baked))?;
    worker.share(
        "recipe",
        &jobs::share_recipe(&TerrainRecipe {
            stack: "stack".into(),
            view: None,
            pyramid,
            surface: rules.authored().clone(),
            materials: true,
        }),
    )?;
    let recipe = Recipe {
        key: "recipe",
        layers: baked.clone(),
        pyramid,
        rules: rules.clone(),
    };

    // Every tile with all its objects, the way each runs.
    let mut tiles = BTreeMap::new();
    for level in (0..pyramid.levels()).rev() {
        for key in pyramid.level_tiles(level) {
            let (tile, bytes) = recipe.tile(&mut worker, key, &[], observed)?;
            observed.hash(format!("tile {}", name(key)), &bytes);
            records.push((TileJob::KIND.name(), bytes));
            tiles.insert(key, tile);
        }
    }
    // What culling knows of every patch holds of its ground: each
    // quarter's floor lies under its samples, and every triangle's slopes
    // inside its disk.
    for (&key, tile) in &tiles {
        observed.check(culling_holds(tile), || {
            format!(
                "tile {}: its culling data claims more than its ground holds",
                name(key)
            )
        });
    }
    let objects: usize = tiles.values().map(|tile| tile.placed().len()).sum();
    observed.record("tiles", tiles.len());
    observed.record("tile objects", objects);
    observed.check(objects > 0, || "no tile holds objects".into());

    // The terrain's lighting from its finest pinned level's tiles, as
    // engines ask for it.
    let job = lighting_job(&pyramid, &tiles);
    let input = job.input();
    let bytes = worker.run(LightingJob::KIND.name(), &input)?;
    let ground = Ground {
        samples: job.samples,
        spacing: job.spacing,
        heights: &job.heights,
    };
    let made = lighting::light(&ground, job.extent.to_array(), job.sky, job.horizons);
    observed.check(codec::encode(&made) == bytes, || {
        "the lighting job differs from the lighting made in place".into()
    });
    observed.check(
        worker.sliced(LightingJob::KIND.name(), &input)? == bytes,
        || "the lighting job run a slice at a time differs from the job run whole".into(),
    );
    observed.hash("lighting", &bytes);
    let sky = made.sky_layers();
    let area = (SKY_TEXELS * SKY_TEXELS) as usize;
    let seen: u64 = sky[..area * 4]
        .chunks(4)
        .map(|texel| u64::from(texel[0]))
        .sum();
    observed.record(
        "lighting maps",
        json!({
            "sky texels": area,
            "horizon texels": HORIZON_TEXELS * HORIZON_TEXELS,
            "mean sky seen": seen as f64 / area as f64 / 255.0,
            "lowest": made.lowest,
        }),
    );
    records.push((LightingJob::KIND.name(), bytes));

    // A tile under edits, which its job carries: two pits dug in the
    // meadow, the later over the earlier.
    let edits = [
        Stamp {
            center: [600.0, 600.0],
            radius: 6.0,
            falloff: 4.0,
            op: StampOp::Lower(3.0),
        },
        Stamp {
            center: [606.0, 600.0],
            radius: 4.0,
            falloff: 2.0,
            op: StampOp::Hole,
        },
    ];
    let edited = TileKey::new(0, 2, 2);
    let (tile, bytes) = recipe.tile(&mut worker, edited, &edits, observed)?;
    observed.hash(format!("edited tile {}", name(edited)), &bytes);
    observed.check(tile.heights() != tiles[&edited].heights(), || {
        format!("the edits leave tile {} as it was", name(edited))
    });
    records.push((TileJob::KIND.name(), bytes));

    // The ground around the terrain to the horizon: the same layers seen
    // from afar, without objects.
    let surroundings = terrain
        .surroundings
        .ok_or_else(|| format!("{FIXTURE}'s {ENTITY} has no surroundings"))?;
    let (_, around, (low, high)) = surroundings
        .layout(&terrain)
        .ok_or("the surroundings have no layout")?;
    let view = StackView {
        offset: -low,
        low: Vec2::ZERO,
        high: high - low,
        blend: surroundings.blend,
    };
    let around_rules = Arc::new(rules.without_placements());
    worker.share(
        "around",
        &jobs::share_recipe(&TerrainRecipe {
            stack: "stack".into(),
            view: Some(view),
            pyramid: around,
            surface: around_rules.authored().clone(),
            materials: true,
        }),
    )?;
    let around_recipe = Recipe {
        key: "around",
        layers: Arc::new(baked.surroundings(view.offset, (view.low, view.high), view.blend)),
        pyramid: around,
        rules: around_rules,
    };
    observed.record("surroundings tiles", around.tiles(0)[0]);
    for level in (around.levels().saturating_sub(2)..around.levels()).rev() {
        for key in around.level_tiles(level) {
            let (_, bytes) = around_recipe.tile(&mut worker, key, &[], observed)?;
            observed.hash(format!("surroundings {}", name(key)), &bytes);
            records.push((TileJob::KIND.name(), bytes));
        }
    }

    // A walk over the south-western hills, around a boulder on the
    // straight line (where terrain-smoke's walk goes around a pit).
    let route = (Vec2::new(212.0, 262.0), Vec2::new(412.0, 262.0));
    let obstacles = [Obstacle {
        center: Vec2::new(312.0, 262.0),
        radius: 12.0,
    }];
    let options = NavOptions::default();
    let walked = WalkJob {
        recipe: "recipe".into(),
        stamps: Vec::new(),
        from: route.0,
        to: route.1,
        obstacles: obstacles.to_vec(),
        options,
    };
    let job = worker.run(WalkJob::KIND.name(), &walked.identity())?;
    let found = walk(&baked, &pyramid, route.0, route.1, &obstacles, &options);
    observed.check(job == codec::encode(&found), || {
        "the walk job differs from the walk in place".into()
    });
    observed.hash("walk", &job);
    let found = found.ok_or("no walk reaches over the hills")?;
    observed.check(found.length > route.1.distance(route.0), || {
        "the walk is no longer than the straight line it leaves around the boulder".into()
    });
    observed.record(
        "walk",
        json!({"points": found.points.len(), "length": found.length, "level": found.level}),
    );

    // The pack the cook writes of every record above, which reads each
    // back as it went in.
    let pack = pack(&records, observed)?;
    observed.hash("pack", &pack);

    worker.write(root)
}

/// The fixture's terrain, its heightmaps read from beside the scene.
fn load(root: &Path) -> Result<Terrain, String> {
    let path = root.join(FIXTURE);
    let source = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {FIXTURE}: {error}"))?;
    let document = SceneDocument::from_json(&source).map_err(|error| error.to_string())?;
    let folder = path.parent().ok_or("the fixture has no folder")?;
    let mut files = SceneFiles::new();
    for file in document.files() {
        if file.kind == SceneFileKind::Buffer {
            let bytes = std::fs::read(folder.join(&file.uri))
                .map_err(|error| format!("cannot read {}: {error}", file.uri))?;
            files.add_buffer(&file.uri, bytes);
        }
    }
    let (index, object) = document
        .entities
        .iter()
        .enumerate()
        .find(|(_, object)| object.id == ENTITY)
        .ok_or_else(|| format!("{FIXTURE} has no entity {ENTITY}"))?;
    object
        .terrain
        .as_ref()
        .ok_or_else(|| format!("{FIXTURE}'s {ENTITY} has no terrain"))?
        .prepare_component(
            &format!("$.entities[{index}].terrain"),
            Some(&files),
            (&document.materials, &document.meshes),
        )
        .map_err(|error| error.to_string())
}

/// The lighting job of `pyramid`'s terrain from `tiles`: its finest pinned
/// level's samples over the whole terrain, as engines gather them.
fn lighting_job(pyramid: &Pyramid, tiles: &BTreeMap<TileKey, Tile>) -> LightingJob {
    let level = pyramid.levels().saturating_sub(PINNED);
    let [tiles_x, tiles_z] = pyramid.tiles(level);
    let quads = TILE_QUADS as usize;
    let columns = tiles_x as usize * quads + 1;
    let rows = tiles_z as usize * quads + 1;
    let mut heights = vec![0.0_f32; columns * rows];
    for key in pyramid.level_tiles(level) {
        let tile = &tiles[&key];
        let first = [usize::from(key.x) * quads, usize::from(key.z) * quads];
        for z in 0..=quads {
            for x in 0..=quads {
                heights[(first[1] + z) * columns + first[0] + x] = tile.height(x as i32, z as i32);
            }
        }
    }
    LightingJob {
        samples: [columns as u32, rows as u32],
        spacing: pyramid.spacing(level),
        extent: pyramid.extent(),
        heights: heights.into_boxed_slice(),
        sky: SKY_TEXELS,
        horizons: HORIZON_TEXELS,
    }
}

/// Whether what culling knows of each of `tile`'s patches holds of its
/// ground: every triangle's slopes within its disk, each quarter's samples
/// on or above its floor.
fn culling_holds(tile: &Tile) -> bool {
    let quads = PATCH_QUADS as i32;
    let half = quads / 2;
    let inverse = 1.0 / tile.spacing();
    (0..PATCHES_PER_TILE * PATCHES_PER_TILE).all(|index| {
        let (px, pz) = (index % PATCHES_PER_TILE, index / PATCHES_PER_TILE);
        let (first_x, first_z) = (px as i32 * quads, pz as i32 * quads);
        let [center_x, center_z, radius] = tile.patch_slopes(px, pz);
        let slopes = (first_z..first_z + quads).all(|z| {
            (first_x..first_x + quads).all(|x| {
                let (a, b) = (tile.height(x, z), tile.height(x + 1, z));
                let (c, d) = (tile.height(x, z + 1), tile.height(x + 1, z + 1));
                [
                    [(b - a) * inverse, (c - a) * inverse],
                    [(d - c) * inverse, (d - b) * inverse],
                ]
                .iter()
                .all(|&[sx, sz]| {
                    let (dx, dz) = (sx - center_x, sz - center_z);
                    (dx * dx + dz * dz).sqrt() <= radius
                })
            })
        });
        let floors = tile
            .patch_floors(px, pz)
            .iter()
            .enumerate()
            .all(|(quarter, &floor)| {
                let x0 = first_x + (quarter as i32 % 2) * half;
                let z0 = first_z + (quarter as i32 / 2) * half;
                (z0..=z0 + half).all(|z| (x0..=x0 + half).all(|x| tile.height(x, z) >= floor))
            });
        slopes && floors
    })
}

/// `key` as goldens name it: level/x/z.
fn name(key: TileKey) -> String {
    format!("{}/{}/{}", key.level, key.x, key.z)
}

/// A pack of `records` (each keyed by its kind and a hash of its bytes
/// standing for its identity, under a fixed source hash, so the pack
/// changes only with what it holds and how packs are written), its records
/// compressed as the cook compresses them; each must read back from it as
/// it went in.
fn pack(records: &[(&str, Vec<u8>)], observed: &mut Observations) -> Result<Vec<u8>, String> {
    let keyed: Vec<([u8; 32], &[u8], Vec<u8>)> = records
        .iter()
        .map(|(kind, bytes)| {
            let key = engine_sources::record_key(&[0; 32], kind, &codec_identity(bytes));
            let stored =
                zstd::bulk::compress(bytes, PACK_LEVEL).map_err(|error| error.to_string())?;
            Ok((key, bytes.as_slice(), stored))
        })
        .collect::<Result<_, String>>()?;
    let pack = engine_worker::pack_bytes(&keyed, |directory| {
        zstd::bulk::compress(directory, PACK_LEVEL).unwrap_or_default()
    });
    let header = pack
        .get(..engine_worker::PACK_HEADER)
        .ok_or("the pack has no header")?;
    let (offset, stored, _) = engine_worker::pack_directory(header)?;
    let range = |offset: u64, length: u64| {
        let start = usize::try_from(offset).map_err(|error| error.to_string())?;
        let end = start + usize::try_from(length).map_err(|error| error.to_string())?;
        pack.get(start..end)
            .ok_or_else(|| "a range past the pack's end".to_owned())
    };
    let index = engine_worker::PackIndex::open(header, range(offset, stored)?)?;
    observed.record("pack records", index.len());
    for (key, bytes, _) in &keyed {
        let hex: String = key.iter().map(|byte| format!("{byte:02x}")).collect();
        let entry = index.find(&hex).ok_or("a record missing from the pack")?;
        let read = engine_worker::unpack(&entry, range(entry.offset, u64::from(entry.stored))?)?;
        observed.check(read == *bytes, || {
            format!("record {hex} reads back from the pack as other bytes")
        });
    }
    Ok(pack)
}

/// zstd's level the corpus packs at, the cook's.
const PACK_LEVEL: i32 = 19;

/// What stands for a record's identity in the corpus's pack: its bytes'
/// hash, as the records differ by what made them.
fn codec_identity(bytes: &[u8]) -> Vec<u8> {
    use sha2::Digest;
    sha2::Sha256::digest(bytes).to_vec()
}

/// What the bakes found, as readable numbers.
fn record_bakes(layers: &Layers, observed: &mut Observations) {
    for (index, layer) in layers.layers().iter().enumerate() {
        let stats = match layer {
            Layer::Hydrology(stage) => stage.baked.as_ref().map(|found| {
                let stats = &found.stats;
                json!({
                    "samples": stats.samples,
                    "channel samples": stats.channel_samples,
                    "lake samples": stats.lake_samples,
                    "max area": stats.max_area,
                    "falls": stats.falls,
                    "rivers": found.rivers().len(),
                })
            }),
            Layer::Erosion(stage) => stage.baked.as_ref().map(|found| {
                let stats = &found.stats;
                json!({
                    "samples": stats.samples,
                    "rounds": stats.rounds,
                    "deepest": stats.deepest,
                    "highest": stats.highest,
                })
            }),
            Layer::Path(stage) => stage.baked.as_ref().map(|found| {
                let stats = &found.stats;
                json!({
                    "length": stats.length,
                    "samples": stats.samples,
                    "steepest": stats.steepest,
                    "deepest cut": stats.deepest_cut,
                    "highest fill": stats.highest_fill,
                    "bridges": stats.bridges,
                    "tunnels": stats.tunnels,
                })
            }),
            _ => None,
        };
        if let Some(stats) = stats {
            observed.record(format!("bake {index}"), stats);
        }
    }
}

/// A recipe shared with the worker as `key`, and what engines generate
/// its tiles from in place.
struct Recipe {
    key: &'static str,
    layers: Arc<Layers>,
    pyramid: Pyramid,
    rules: Arc<BiomeRules>,
}

impl Recipe {
    /// Tile `key` under `stamps` (edits, in order), with every object its
    /// level places, generated by a worker's job, which must equal the tile
    /// generated in place and the job run a slice at a time: the tile and
    /// its record's bytes.
    fn tile(
        &self,
        worker: &mut Worker,
        key: TileKey,
        stamps: &[Stamp],
        observed: &mut Observations,
    ) -> Result<(Tile, Vec<u8>), String> {
        let job = TileJob {
            recipe: self.key.into(),
            key,
            stamps: stamps.to_vec(),
        };
        let input = job.identity();
        let whole = worker.run(TileJob::KIND.name(), &input)?;
        let what = format!(
            "{} tile {} under {} edits",
            self.key,
            name(key),
            stamps.len()
        );
        let layers = if stamps.is_empty() {
            self.layers.clone()
        } else {
            Arc::new(
                self.layers
                    .with_stamps(stamps)
                    .map_err(|error| error.to_string())?,
            )
        };
        let mut generator = Generator::new(layers, &self.pyramid, key)
            .with_surface(self.rules.clone())
            .with_materials(true);
        let mut unlimited = u64::MAX;
        let in_place = generator
            .advance(&mut unlimited)
            .ok_or_else(|| format!("{what} fails in place"))?;
        observed.check(codec::encode(&in_place) == whole, || {
            format!("{what}: the job differs from the tile generated in place")
        });
        observed.check(
            worker.sliced(TileJob::KIND.name(), &input)? == whole,
            || format!("{what}: the job run a slice at a time differs from the job run whole"),
        );
        let tile: Tile = codec::decode(&whole).map_err(|error| error.to_string())?;
        Ok((tile, whole))
    }
}

/// A job worker as hosts run one beside the engine, recording every share
/// and job (with its output's hash) for the replay in Wasm.
#[derive(Default)]
struct Worker {
    worker: TerrainWorker,
    next: u64,
    /// What `cargo xtask pure --wasm` replays: [`JOBS`].
    log: Vec<serde_json::Value>,
}

impl Worker {
    fn share(&mut self, key: &str, bytes: &[u8]) -> Result<(), String> {
        self.log
            .push(json!({"share": key, "bytes": STANDARD.encode(bytes)}));
        self.worker.share(key, bytes)
    }

    /// Job `kind` on `input`, run whole.
    fn run(&mut self, kind: &str, input: &[u8]) -> Result<Vec<u8>, String> {
        let output = self.worker.run(kind, input)?;
        self.log.push(json!({
            "job": kind,
            "input": STANDARD.encode(input),
            "output": hash(&output),
        }));
        Ok(output)
    }

    /// Job `kind` on `input`, run a slice at a time.
    fn sliced(&mut self, kind: &str, input: &[u8]) -> Result<Vec<u8>, String> {
        self.next += 1;
        self.worker.start(self.next, kind, input)?;
        loop {
            if let Some(output) = self.worker.step(self.next, SLICE)? {
                self.log.push(json!({
                    "job": kind,
                    "input": STANDARD.encode(input),
                    "slice": SLICE,
                    "output": hash(&output),
                }));
                return Ok(output);
            }
        }
    }

    /// Writes what it ran to [`JOBS`].
    fn write(&self, root: &Path) -> Result<(), String> {
        let path = root.join(JOBS);
        if let Some(folder) = path.parent() {
            std::fs::create_dir_all(folder)
                .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
        }
        std::fs::write(
            &path,
            serde_json::Value::Array(self.log.clone()).to_string(),
        )
        .map_err(|error| format!("cannot write {JOBS}: {error}"))
    }
}
