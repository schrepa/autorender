//! The terrain corpus: the tracked terrain fixture's generated terrain
//! (web/scripts/fixtures/terrain.scene.json and its dome heightmap). Its
//! bakes; every tile of its pyramid and some of its surroundings'; tiles
//! generated in bands with their objects placed after; objects standing far
//! away placed apart; a walk; and its `ETR1` file. Each runs as job workers
//! run it, whole and a slice at a time, and in place as engines without
//! workers run it, and every way must give the same bits.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use engine_core::{Terrain, TerrainSource};
use engine_scene::{SceneDocument, SceneFileKind, SceneFiles};
use engine_terrain::jobs::{
    self, BAKE_JOB, BAND_JOB, FAR_JOB, PLACE_JOB, StackView, TILE_JOB, TerrainRecipe,
    TerrainWorker, WALK_JOB,
};
use engine_terrain::{
    BiomeRules, FileHeader, Generator, Layer, Layers, NavOptions, Obstacle, Placing, Pyramid,
    TILE_STRIDE, TerrainBake, Tile, TileKey, codec, decode_tile, encode_file, merge_bands,
    merge_placed, walk,
};
use glam::Vec2;
use serde_json::json;

use super::{Observations, hash};

/// The fixture scene and the entity holding its terrain.
const FIXTURE: &str = "web/scripts/fixtures/terrain.scene.json";
const ENTITY: &str = "ground";
/// Work units each slice of a job or bake gets: few enough that each
/// takes many slices, as frames and workers slice them.
const SLICE: u64 = 4096;
/// The height step of the terrain file, `terrain_bake`'s default.
const STEP: f32 = 1.0 / 256.0;
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
    let TerrainSource::Generated(authored) = &terrain.source else {
        return Err(format!("{FIXTURE}'s {ENTITY} reads a terrain file"));
    };
    let pyramid = terrain.pyramid;
    let mut worker = Worker::default();

    // The bakes at once, as tools bake; in slices, as frames bake in
    // place; and as a worker's job.
    let baked = authored.bake_all(&pyramid);
    let bytes = codec::encode(&baked);
    observed.hash("bakes", &bytes);
    observed.check(
        codec::encode(&bake_in_slices(authored, &pyramid)) == bytes,
        || "the bakes in slices differ from the bakes at once".into(),
    );
    observed.check(
        worker.run(BAKE_JOB, &jobs::bake_input(authored, &pyramid))? == bytes,
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
            let tile = recipe.tile(&mut worker, key, Placing::All, observed)?;
            observed.hash(format!("tile {}", name(key)), &codec::encode(&tile));
            tiles.insert(key, tile);
        }
    }
    let objects: usize = tiles.values().map(|tile| tile.placed().len()).sum();
    observed.record("tiles", tiles.len());
    observed.record("tile objects", objects);
    observed.check(objects > 0, || "no tile holds objects".into());

    // Tiles of levels whose layers stand objects far away: their own
    // objects first, those standing far away by a job of their own after.
    for (&key, tile) in &tiles {
        if rules.places(key.level, Placing::Far) {
            let near = recipe.tile(&mut worker, key, Placing::Near, observed)?;
            let far = jobs::far_output(&worker.run(FAR_JOB, &jobs::far_input("recipe", &near))?)
                .map_err(|error| error.to_string())?;
            observed.hash(format!("far {}", name(key)), &codec::encode(&far));
            observed.record(format!("far objects {}", name(key)), far.len());
            let joined = near
                .clone()
                .with_placed(merge_placed(near.placed(), &far), Placing::All);
            observed.check(codec::encode(&joined) == codec::encode(tile), || {
                format!(
                    "tile {}: its own objects and those standing far away differ from all \
                     its objects",
                    name(key)
                )
            });
        }
    }

    // Tiles in bands of rows across workers, joined, then their objects
    // in bands of candidate rows, as urgent tiles generate: a tile of the
    // meadow's crates and balls, one holding an author's ball, and one of
    // the level whose objects all stand far away.
    for (key, placing, bands) in [
        (TileKey::new(0, 2, 2), Placing::All, 4),
        (TileKey::new(1, 0, 1), Placing::All, 3),
        (TileKey::new(2, 0, 0), Placing::Near, 2),
    ] {
        let whole = match placing {
            Placing::All => tiles[&key].clone(),
            _ => recipe.tile(&mut worker, key, placing, observed)?,
        };
        let banded = recipe.banded(&mut worker, key, placing, bands)?;
        observed.check(codec::encode(&banded) == codec::encode(&whole), || {
            format!(
                "tile {} in {bands} bands, its objects placed after, differs from the tile \
                 whole",
                name(key)
            )
        });
    }

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
            let tile = around_recipe.tile(&mut worker, key, Placing::All, observed)?;
            observed.hash(format!("surroundings {}", name(key)), &codec::encode(&tile));
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
    let job = worker.run(
        WALK_JOB,
        &jobs::walk_input("recipe", route, &obstacles, &options),
    )?;
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

    // The terrain file: every tile decodes to its heights quantized to the
    // step, with its holes.
    let file = encode_file(&pyramid, STEP, |key| tiles[&key].clone().without_placed())
        .map_err(|error| error.to_string())?;
    observed.hash("etr1", &file);
    let header = FileHeader::parse(&file).map_err(|error| error.to_string())?;
    for (&key, tile) in &tiles {
        let entry = header.tile(key).ok_or("a tile missing from the file")?;
        let start = usize::try_from(entry.offset).map_err(|error| error.to_string())?;
        let payload = file
            .get(start..start + entry.length as usize)
            .ok_or("a tile past the file's end")?;
        let decoded = decode_tile(&header, key, payload).map_err(|error| error.to_string())?;
        let quantized = tile.heights().iter().map(|&height| {
            let steps = (f64::from(height) / f64::from(STEP)).round() as i32;
            (f64::from(steps) * f64::from(STEP)) as f32
        });
        observed.check(
            decoded
                .heights()
                .iter()
                .zip(quantized)
                .all(|(read, wrote)| read.to_bits() == wrote.to_bits())
                && decoded.hole_bits() == tile.hole_bits()
                && decoded.spacing().to_bits() == tile.spacing().to_bits(),
            || {
                format!(
                    "tile {} decodes from the file to other heights than its quantized ones",
                    name(key)
                )
            },
        );
    }
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

/// `key` as goldens name it: level/x/z.
fn name(key: TileKey) -> String {
    format!("{}/{}/{}", key.level, key.x, key.z)
}

/// `layers` with every bake done in slices, as frames bake in place.
fn bake_in_slices(layers: &Layers, pyramid: &Pyramid) -> Layers {
    let mut layers = layers.clone();
    while let Some(next) = layers.unbaked() {
        let index = next.0;
        let mut bake = TerrainBake::new(&layers, next, pyramid);
        let baked = loop {
            let mut budget = SLICE;
            if let Some(baked) = bake.advance(pyramid, &mut budget) {
                break baked;
            }
        };
        layers = layers.with_baked(index, baked);
    }
    layers
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
                    "rivers": found.rivers.rivers().len(),
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
    /// Tile `key` with `placing`'s objects, generated by a worker's job,
    /// which must equal the tile generated in place, the job run a slice at
    /// a time, and the job staged (whose shape, handed out first, must be
    /// the tile's ground).
    fn tile(
        &self,
        worker: &mut Worker,
        key: TileKey,
        placing: Placing,
        observed: &mut Observations,
    ) -> Result<Tile, String> {
        let input = jobs::tile_input(self.key, key, placing, false);
        let whole = worker.run(TILE_JOB, &input)?;
        let what = format!("{} tile {} placing {placing:?}", self.key, name(key));
        let mut generator = Generator::new(self.layers.clone(), &self.pyramid, key)
            .with_surface(self.rules.clone())
            .with_placing(placing)
            .with_materials(true);
        let mut unlimited = u64::MAX;
        let in_place = generator
            .advance(&mut unlimited)
            .ok_or_else(|| format!("{what} fails in place"))?;
        observed.check(codec::encode(&in_place) == whole, || {
            format!("{what}: the job differs from the tile generated in place")
        });
        observed.check(worker.sliced(TILE_JOB, &input)? == whole, || {
            format!("{what}: the job run a slice at a time differs from the job run whole")
        });
        let (shape, staged) = worker.staged(&jobs::tile_input(self.key, key, placing, true))?;
        observed.check(staged == whole, || {
            format!("{what}: the staged job differs from the job")
        });
        let tile: Tile = codec::decode(&whole).map_err(|error| error.to_string())?;
        let shape: Option<Tile> = shape
            .map(|bytes| codec::decode(&bytes))
            .transpose()
            .map_err(|error| error.to_string())?;
        observed.check(
            shape.is_some_and(|shape| {
                shape.heights() == tile.heights() && shape.hole_bits() == tile.hole_bits()
            }),
            || format!("{what}: the staged job hands out no shape, or another ground"),
        );
        Ok(tile)
    }

    /// Tile `key` generated in `bands` bands of rows and joined, then its
    /// `placing` objects placed in as many bands of candidate rows, joined
    /// in order, as engines dispatch urgent tiles to workers.
    fn banded(
        &self,
        worker: &mut Worker,
        key: TileKey,
        placing: Placing,
        bands: usize,
    ) -> Result<Tile, String> {
        let size = (TILE_STRIDE as usize).div_ceil(bands);
        let mut parts = Vec::with_capacity(bands);
        for band in 0..bands {
            let rows = (band * size) as u32..((band + 1) * size).min(TILE_STRIDE as usize) as u32;
            let output = worker.run(BAND_JOB, &jobs::band_input(self.key, key, rows))?;
            parts.push(jobs::band_output(&output).map_err(|error| error.to_string())?);
        }
        let spacing = self.pyramid.spacing(u32::from(key.level));
        let tile = merge_bands(key, spacing, &parts).ok_or("the bands do not join")?;
        if !self.rules.places(key.level, placing) {
            return Ok(tile);
        }
        let starts: Vec<(u32, u32)> = self
            .rules
            .place_rows(key, spacing, placing)
            .iter()
            .enumerate()
            .flat_map(|(layer, &rows)| (0..rows).map(move |row| (layer as u32, row)))
            .collect();
        let bands = bands.min(starts.len()).max(1);
        let mut objects = Vec::new();
        for band in 0..bands {
            let start = if band == 0 {
                (0, 0)
            } else {
                starts[band * starts.len() / bands]
            };
            let end = (band + 1 < bands).then(|| starts[(band + 1) * starts.len() / bands]);
            let output = worker.run(
                PLACE_JOB,
                &jobs::place_input(self.key, &tile, placing, (start, end)),
            )?;
            objects.extend(jobs::far_output(&output).map_err(|error| error.to_string())?);
        }
        Ok(tile.with_placed(objects, placing))
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

    /// A staged tile job on `input`, run a slice at a time: the part it
    /// hands out first, if any, and its output.
    fn staged(&mut self, input: &[u8]) -> Result<(Option<Vec<u8>>, Vec<u8>), String> {
        self.next += 1;
        self.worker.start(self.next, TILE_JOB, input)?;
        let mut part = None;
        loop {
            let output = self.worker.step(self.next, SLICE)?;
            if part.is_none() {
                part = self.worker.take_part(self.next);
            }
            if let Some(output) = output {
                self.log.push(json!({
                    "job": TILE_JOB,
                    "input": STANDARD.encode(input),
                    "slice": SLICE,
                    "part": part.as_deref().map(hash),
                    "output": hash(&output),
                }));
                return Ok((part, output));
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
