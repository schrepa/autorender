use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use engine_codec as codec;
use engine_core::{JobRef, RecordKey};
use engine_heightfield::layer::LayerKind;
use engine_heightfield::tile::{
    BiomeRules, Generator, StackView, TerrainRecipe, Tile, TileJob, shaded_elsewhere,
};
use engine_heightfield::{
    Layers, Pyramid, Stamp, StampOp, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey,
};
use engine_navigation::{
    Gait, NavOptions, NavPath, PORTALS, PortalsJob, TilePortals, WalkData, WalkGround, WalkSearch,
};
use engine_places::ground::{GroundBridge, GroundGrid, GroundObstacle};
use engine_places::terrain::TerrainLayers;
use engine_scene::{SceneDocument, SceneKinds};
use engine_terrain::jobs::{self, BAKE, BakeJob, GROUND_GRID, GroundIdentity, TILE_BUILD};
use engine_terrain::{Terrain, TerrainScene};
use engine_terrain_shading::jobs::{
    BOUNCE, BounceJob, LIGHTING, LightingJob, MACRO_COLORS, MacroJob,
};
use engine_terrain_shading::lighting::{self, Ground};
use engine_terrain_shading::{bounce, macro_colors};
use engine_water::jobs::{PARTS, WaterJob};
use engine_water::{SettleRegion, SettleStart, WaterPart};
use glam::Vec2;
use serde_json::json;

use super::{JobLog, Observations};

const FIXTURE: &str = "web/scripts/fixtures/terrain.scene.json";
const ENTITY: &str = "ground";
const SLICE: u64 = 4096;
const WATER_CELL: f32 = 8.0;
const PINNED: u32 = 2;
const SKY_TEXELS: u32 = 256;
const HORIZON_TEXELS: u32 = 128;
const BOUNCE_PROBES: u32 = 64;
const MACRO_TEXELS: u32 = 256;
pub const JOBS: &str = "target/pure/terrain-jobs.json";

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let terrain = load(root)?;
    let authored = &terrain.layers;
    let pyramid = terrain.pyramid;
    let mut worker = JobLog::new()?;
    let kinds: Vec<&'static LayerKind> =
        worker.list().members::<TerrainLayers>().copied().collect();

    let baked = engine_terrain::bake::bake_all(authored, &pyramid);
    let bytes = jobs::share_stack(&baked).to_vec();
    observed.hash("bakes", &bytes);
    let mut records = vec![(BAKE.name(), bytes.clone())];
    observed.check(
        worker.run(
            &BAKE,
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
    let stack = worker.share(&jobs::share_stack(&baked));
    let shared = worker.share(&jobs::share_recipe(&TerrainRecipe {
        stack: stack.hash(),
        view: None,
        pyramid,
        surface: rules.authored().clone(),
        materials: true,
    }));
    let recipe = Recipe {
        name: "recipe",
        key: shared,
        layers: baked.clone(),
        pyramid,
        rules: rules.clone(),
        around: false,
    };

    let mut tiles = BTreeMap::new();
    for level in (0..pyramid.levels()).rev() {
        for key in pyramid.level_tiles(level) {
            let (tile, bytes) = recipe.tile(&mut worker, key, (&[], &kinds), observed)?;
            observed.hash(format!("tile {}", name(key)), &bytes);
            records.push((TILE_BUILD.name(), bytes));
            tiles.insert(key, tile);
        }
    }
    let objects: usize = tiles.values().map(|tile| tile.placed().len()).sum();
    observed.record("tiles", tiles.len());
    observed.record("tile objects", objects);
    observed.check(objects > 0, || "no tile holds objects".into());

    let job = lighting_job(&pyramid, shared);
    let input = codec::encode(&job);
    let bytes = worker.run(&LIGHTING, &input)?;
    let (samples, spacing, heights) = level_heights(&pyramid, &tiles);
    let ground = Ground {
        samples,
        spacing,
        heights: &heights,
    };
    let extent = pyramid.extent().to_array();
    let made = lighting::light(&ground, extent, job.sky, job.horizons);
    observed.check(codec::encode(&made) == bytes, || {
        "the lighting job differs from the lighting made in place".into()
    });
    observed.check(worker.sliced(&LIGHTING, &input, SLICE)? == bytes, || {
        "the lighting job run a slice at a time differs from the job run whole".into()
    });
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
    records.push((LIGHTING.name(), bytes));

    let surface = (baked.as_ref(), rules.as_ref());
    let transfer = bounce_transfer(&mut worker, (&job, &ground, extent), surface, observed)?;
    records.push((BOUNCE.name(), transfer));
    let colors = MacroJob {
        recipe: shared.hash(),
        level: pyramid.levels().saturating_sub(PINNED),
        size: MACRO_TEXELS,
    };
    let bytes = worker.run(&MACRO_COLORS, &codec::encode(&colors))?;
    let made = macro_colors::macro_colors(
        &baked,
        &pyramid,
        colors.level,
        colors.size,
        &rules.authored().macro_colors,
    )
    .ok_or("the macro colors' heights are invalid")?;
    observed.check(made == bytes, || {
        "the macro colors job differs from the colors made in place".into()
    });
    observed.check(
        worker.sliced(&MACRO_COLORS, &codec::encode(&colors), SLICE)? == bytes,
        || "the macro colors job run a slice at a time differs from the job run whole".into(),
    );
    observed.hash("macro colors", &bytes);
    let texels = bytes.len() / 4;
    let mean = |channel: usize| {
        let sum: u64 = bytes.chunks(4).map(|texel| u64::from(texel[channel])).sum();
        sum as f64 / texels.max(1) as f64 / 255.0
    };
    let tint = |channel: usize| mean(channel) * 255.0 / 128.0;
    observed.record(
        "macro colors",
        json!({
            "texels": texels,
            "mean tint": [tint(0), tint(1), tint(2)],
            "mean lushness": mean(3),
        }),
    );

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
    let (tile, bytes) = recipe.tile(&mut worker, edited, (&edits, &kinds), observed)?;
    observed.hash(format!("edited tile {}", name(edited)), &bytes);
    observed.check(tile.heights() != tiles[&edited].heights(), || {
        format!("the edits leave tile {} as it was", name(edited))
    });
    records.push((TILE_BUILD.name(), bytes));

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
    let around_shared = worker.share(&jobs::share_recipe(&TerrainRecipe {
        stack: stack.hash(),
        view: Some(view),
        pyramid: around,
        surface: around_rules.authored().clone(),
        materials: true,
    }));
    let around_recipe = Recipe {
        name: "around",
        key: around_shared,
        layers: Arc::new(baked.surroundings(view.offset, (view.low, view.high), view.blend)),
        pyramid: around,
        rules: around_rules,
        around: true,
    };
    observed.record("surroundings tiles", around.tiles(0)[0]);
    for level in (around.levels().saturating_sub(2)..around.levels()).rev() {
        for key in around.level_tiles(level) {
            let (_, bytes) = around_recipe.tile(&mut worker, key, (&[], &kinds), observed)?;
            observed.hash(format!("surroundings {}", name(key)), &bytes);
            records.push((TILE_BUILD.name(), bytes));
        }
    }

    let gait = NavOptions::default().gait(&pyramid);
    let mut graph = Graph::default();
    let mut bytes_held = 0;
    for key in pyramid.level_tiles(gait.level) {
        let (ground, bytes) = portals(
            &mut worker,
            (&tiles[&key], &baked, stack),
            (&pyramid, &gait),
            (&[], observed),
        )?;
        observed.hash(format!("portals {}", name(key)), &bytes);
        let held: TilePortals = codec::decode(&bytes).map_err(|error| error.to_string())?;
        graph.portals.insert(key, Arc::new(held));
        graph.grounds.insert(key, Arc::new(ground));
        bytes_held += bytes.len();
        records.push((PORTALS.name(), bytes));
    }
    let edited_layers = Arc::new(
        baked
            .with_stamps(&edits, &kinds)
            .map_err(|error| error.to_string())?,
    );
    let (_, bytes) = portals(
        &mut worker,
        (&tile, &edited_layers, stack),
        (&pyramid, &gait),
        (&edits, observed),
    )?;
    observed.hash(format!("edited portals {}", name(edited)), &bytes);
    let count: usize = graph
        .portals
        .values()
        .map(|held| held.portals().len())
        .sum();
    observed.record(
        "walk graph",
        json!({"tiles": graph.portals.len(), "portals": count, "bytes": bytes_held}),
    );
    observed.check(count > 0, || "no tile has portals".into());

    let route = (Vec2::new(212.0, 262.0), Vec2::new(412.0, 262.0));
    let (boulder, boulder_radius) = (Vec2::new(312.0, 262.0), 12.0);
    let obstacles = [GroundObstacle::Circle {
        center: boulder,
        radius: boulder_radius,
    }];
    let walked = |slice: u64| graph.walk(&pyramid, route, &obstacles, slice);
    let whole = codec::encode(&walked(u64::MAX)?);
    observed.check(whole == codec::encode(&walked(SLICE)?), || {
        "the walk a slice at a time differs from the walk whole".into()
    });
    observed.hash("walk", &whole);
    let found = walked(u64::MAX)?.ok_or("no walk reaches over the hills")?;
    observed.check(found.length > route.1.distance(route.0), || {
        "the walk is no longer than the straight line it leaves around the boulder".into()
    });
    observed.check(
        found.points.iter().all(|point| {
            Vec2::new(point[0], point[2]).distance(boulder)
                >= boulder_radius + NavOptions::default().radius - 1.0
        }),
        || "the walk crosses the boulder".into(),
    );
    observed.record(
        "walk",
        json!({"points": found.points.len(), "length": found.length, "level": found.level}),
    );

    let watered = engine_water::watered_tiles(&baked, &pyramid);
    let key = *watered
        .first()
        .ok_or("the fixture's bake waters no finest tile")?;
    let region = SettleRegion::alone(&pyramid, key, WATER_CELL);
    let mut parts = Vec::new();
    for tile in region.reached(&pyramid) {
        let input = WaterJob {
            recipe: shared.hash(),
            key: tile,
            stamps: Vec::new(),
            solids: Vec::new(),
            cell: WATER_CELL,
        }
        .input();
        let bytes = worker.run(&PARTS, &input)?;
        observed.check(worker.sliced(&PARTS, &input, SLICE)? == bytes, || {
            format!(
                "the water's part of {} run a slice at a time differs",
                name(tile)
            )
        });
        observed.hash(format!("water part {}", name(tile)), &bytes);
        parts.push(engine_water::jobs::water_output(&bytes).map_err(|error| error.to_string())?);
    }
    let parts: Vec<&WaterPart> = parts.iter().collect();
    let start = SettleStart::assemble(&baked, &pyramid, &region, &parts, None)
        .map_err(|error| error.to_string())?;
    observed.record(
        "water start",
        json!({
            "watered tiles": watered.len(),
            "tile": name(key),
            "parts": parts.len(),
            "cells": start.stored.cells,
            "stored blocks": start.stored.stored(),
            "sources": start.sources.len(),
            "datum": start.datum,
            "cook regions": engine_water::settle_regions(&baked, &pyramid, WATER_CELL).len(),
        }),
    );

    let pack = pack(&records, observed)?;
    observed.hash("pack", &pack);

    worker.write(root, JOBS)
}

fn load(root: &Path) -> Result<Terrain, String> {
    let path = root.join(FIXTURE);
    let source = std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {FIXTURE}: {error}"))?;
    let mut engine = crate::bricks::standard_engine().map_err(|error| error.to_string())?;
    let kinds = SceneKinds::of(engine.app.world());
    let document = SceneDocument::from_json(&source).map_err(|error| error.to_string())?;
    document
        .validate(&kinds)
        .map_err(|error| error.to_string())?;
    if !document.entities.iter().any(|object| object.id == ENTITY) {
        return Err(format!("{FIXTURE} has no entity {ENTITY}"));
    }
    let files = engine_native::files::scene_files(&path, &source, &kinds)
        .map_err(|error| error.to_string())?;
    let scene = document
        .instantiate_with(engine.app.world_mut(), &files)
        .map_err(|error| error.to_string())?;
    scene
        .entity(ENTITY)
        .and_then(|entity| engine.app.world().get::<TerrainScene>(entity))
        .and_then(|terrain| terrain.built.clone())
        .ok_or_else(|| format!("{FIXTURE}'s {ENTITY} has no terrain"))
}

fn lighting_job(pyramid: &Pyramid, recipe: RecordKey) -> LightingJob {
    let level = pyramid.levels().saturating_sub(PINNED);
    LightingJob {
        recipe: recipe.hash(),
        level,
        tiles: pyramid
            .level_tiles(level)
            .map(|key| (key, Vec::new()))
            .collect(),
        sky: SKY_TEXELS,
        horizons: HORIZON_TEXELS,
    }
}

fn level_heights(pyramid: &Pyramid, tiles: &BTreeMap<TileKey, Tile>) -> ([u32; 2], f32, Vec<f32>) {
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
    (
        [columns as u32, rows as u32],
        pyramid.spacing(level),
        heights,
    )
}

fn bounce_transfer(
    worker: &mut JobLog,
    (job, ground, extent): (&LightingJob, &Ground<'_>, [f32; 2]),
    (baked, rules): (&Layers, &BiomeRules),
    observed: &mut Observations,
) -> Result<Vec<u8>, String> {
    let transfer = BounceJob {
        recipe: job.recipe,
        level: job.level,
        tiles: job.tiles.clone(),
        probes: BOUNCE_PROBES,
    };
    let input = codec::encode(&transfer);
    let bytes = worker.run(&BOUNCE, &input)?;
    let made = bounce::bounce(ground, extent, (baked, rules), BOUNCE_PROBES);
    observed.check(codec::encode(&made) == bytes, || {
        "the bounce job differs from the transfer made in place".into()
    });
    observed.check(worker.sliced(&BOUNCE, &input, SLICE)? == bytes, || {
        "the bounce job run a slice at a time differs from the job run whole".into()
    });
    observed.hash("bounce", &bytes);
    let blocks = (BOUNCE_PROBES / 2) as usize * (BOUNCE_PROBES / 2) as usize;
    observed.record(
        "bounce",
        json!({
            "blocks": blocks,
            "detailed blocks": made.detailed_blocks(),
            "probes": made.probe_count(),
        }),
    );
    observed.check(made.probe_count() >= blocks, || {
        "the bounce transfer holds fewer probes than blocks".into()
    });
    Ok(bytes)
}

fn name(key: TileKey) -> String {
    format!("{}/{}/{}", key.level, key.x, key.z)
}

fn pack(records: &[(&str, Vec<u8>)], observed: &mut Observations) -> Result<Vec<u8>, String> {
    let keyed: Vec<([u8; 32], &[u8], Vec<u8>)> = records
        .iter()
        .map(|(kind, bytes)| {
            let key = RecordKey::new(&[0; 32], kind, &codec_identity(bytes)).hash();
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
    let (offset, stored, _) =
        engine_worker::pack_directory(header).map_err(|error| error.to_string())?;
    let range = |offset: u64, length: u64| {
        let start = usize::try_from(offset).map_err(|error| error.to_string())?;
        let end = start + usize::try_from(length).map_err(|error| error.to_string())?;
        pack.get(start..end)
            .ok_or_else(|| "a range past the pack's end".to_owned())
    };
    let index = engine_worker::PackIndex::open(header, range(offset, stored)?)
        .map_err(|error| error.to_string())?;
    observed.record("pack records", index.len());
    for (key, bytes, _) in &keyed {
        let hex: String = key.iter().map(|byte| format!("{byte:02x}")).collect();
        let entry = index.find(&hex).ok_or("a record missing from the pack")?;
        let read = engine_worker::unpack(&entry, range(entry.offset, u64::from(entry.stored))?)
            .map_err(|error| error.to_string())?;
        observed.check(read == *bytes, || {
            format!("record {hex} reads back from the pack as other bytes")
        });
    }
    Ok(pack)
}

const PACK_LEVEL: i32 = 19;

fn codec_identity(bytes: &[u8]) -> Vec<u8> {
    use sha2::Digest;
    sha2::Sha256::digest(bytes).to_vec()
}

fn portals(
    worker: &mut JobLog,
    (tile, layers, stack): (&Tile, &Arc<Layers>, RecordKey),
    (pyramid, gait): (&Pyramid, &Gait),
    (stamps, observed): (&[Stamp], &mut Observations),
) -> Result<(WalkGround, Vec<u8>), String> {
    let key = tile.key();
    let job = PortalsJob {
        grid: JobRef::from(&GROUND_GRID),
        source: codec::encode(&GroundIdentity {
            stack: stack.hash(),
            pyramid: *pyramid,
            edits: stamps.to_vec(),
        }),
        pyramid: *pyramid,
        key,
        gait: *gait,
    };
    let bytes = worker.run(&PORTALS, &job.input())?;
    let ground = WalkGround::new(&walk_grid(tile, layers, pyramid), walk_water(tile), pyramid);
    observed.check(
        codec::encode(&TilePortals::new(&ground, pyramid, gait)) == bytes,
        || {
            format!(
                "tile {} under {} edits: its portals' job differs from its record's portals",
                name(key),
                stamps.len()
            )
        },
    );
    Ok((ground, bytes))
}

fn walk_grid(tile: &Tile, layers: &Layers, pyramid: &Pyramid) -> GroundGrid {
    let side = TILE_SAMPLES as usize;
    let stride = TILE_STRIDE as usize;
    let stored = |x: usize, z: usize| (z + 1) * stride + x + 1;
    let key = tile.key();
    let spacing = pyramid.spacing(u32::from(key.level));
    let mut heights = vec![0.0; side * side].into_boxed_slice();
    for z in 0..side {
        for x in 0..side {
            heights[z * side + x] = tile.heights()[stored(x, z)];
        }
    }
    let first = pyramid.first_sample(key);
    let at =
        |x: i64, z: i64| glam::Vec2::new((first[0] + x) as f32, (first[1] + z) as f32) * spacing;
    let quads = i64::from(TILE_QUADS);
    let mut holes = None;
    if layers.holes_reach(at(0, 0), at(quads, quads)) {
        let mut bits = vec![0_u64; (side * side).div_ceil(64)];
        let mut any = false;
        for z in 0..side {
            for x in 0..side {
                let point = at(x as i64, z as i64);
                if layers.hole_at(point.x, point.y) {
                    let index = z * side + x;
                    bits[index / 64] |= 1 << (index % 64);
                    any = true;
                }
            }
        }
        holes = any.then(|| bits.into_boxed_slice());
    }
    GroundGrid {
        key,
        spacing,
        heights,
        holes,
        bridges: layers
            .bridges()
            .into_iter()
            .map(|bridge| GroundBridge {
                deck: bridge.deck,
                width: bridge.width,
                tunnel: bridge.tunnel,
            })
            .collect(),
        water: None,
    }
}

fn walk_water(tile: &Tile) -> Option<Box<[f32]>> {
    let side = TILE_SAMPLES as usize;
    let stride = TILE_STRIDE as usize;
    let stored = |x: usize, z: usize| (z + 1) * stride + x + 1;
    let surfaces = tile.water_surfaces()?;
    let mut water = vec![f32::NAN; side * side].into_boxed_slice();
    for z in 0..side {
        for x in 0..side {
            water[z * side + x] = surfaces[stored(x, z)];
        }
    }
    Some(water)
}

#[derive(Default)]
struct Graph {
    grounds: BTreeMap<TileKey, Arc<WalkGround>>,
    portals: BTreeMap<TileKey, Arc<TilePortals>>,
}

impl WalkData for Graph {
    fn ground(&self, key: TileKey) -> Option<Arc<WalkGround>> {
        self.grounds.get(&key).cloned()
    }

    fn portals(&self, key: TileKey) -> Option<Arc<TilePortals>> {
        self.portals.get(&key).cloned()
    }
}

impl Graph {
    fn walk(
        &self,
        pyramid: &Pyramid,
        route: (Vec2, Vec2),
        obstacles: &[GroundObstacle],
        slice: u64,
    ) -> Result<Option<NavPath>, String> {
        let mut search =
            WalkSearch::new(pyramid, route, obstacles.to_vec(), &NavOptions::default());
        loop {
            let mut budget = slice;
            if let Some(found) = search.step(&mut budget, self) {
                return Ok(found);
            }
            if !search.wants().is_empty() {
                return Err(format!(
                    "the walk waits for {:?}, which the corpus holds",
                    search.wants()
                ));
            }
        }
    }
}

fn record_bakes(layers: &Layers, observed: &mut Observations) {
    for (index, layer) in layers.layers().iter().enumerate() {
        let stats = match layer.water() {
            Some(found) => {
                let stats = &found.stats;
                Some(json!({
                    "samples": stats.samples,
                    "channel samples": stats.channel_samples,
                    "lake samples": stats.lake_samples,
                    "max area": stats.max_area,
                    "falls": stats.falls,
                    "rivers": found.rivers().len(),
                }))
            }
            None => layer.stats().and_then(|stats| match layer.kind().name {
                "erosion" => Some(json!({
                    "samples": stats["samples"],
                    "rounds": stats["rounds"],
                    "deepest": stats["deepest"],
                    "highest": stats["highest"],
                })),
                "path" => Some(json!({
                    "length": stats["length"],
                    "samples": stats["samples"],
                    "steepest": stats["steepest"],
                    "deepest cut": stats["deepest_cut"],
                    "highest fill": stats["highest_fill"],
                    "bridges": stats["bridges"],
                    "tunnels": stats["tunnels"],
                })),
                _ => None,
            }),
        };
        if let Some(stats) = stats {
            observed.record(format!("bake {index}"), stats);
        }
    }
}

struct Recipe {
    name: &'static str,
    key: RecordKey,
    layers: Arc<Layers>,
    pyramid: Pyramid,
    rules: Arc<BiomeRules>,
    around: bool,
}

impl Recipe {
    fn tile(
        &self,
        worker: &mut JobLog,
        key: TileKey,
        (stamps, kinds): (&[Stamp], &[&'static LayerKind]),
        observed: &mut Observations,
    ) -> Result<(Tile, Vec<u8>), String> {
        let job = TileJob {
            recipe: self.key.hash(),
            key,
            stamps: stamps.to_vec(),
        };
        let input = job.identity();
        let whole = worker.run(&TILE_BUILD, &input)?;
        let what = format!(
            "{} tile {} under {} edits",
            self.name,
            name(key),
            stamps.len()
        );
        let layers = if stamps.is_empty() {
            self.layers.clone()
        } else {
            Arc::new(
                self.layers
                    .with_stamps(stamps, kinds)
                    .map_err(|error| error.to_string())?,
            )
        };
        let materials = !shaded_elsewhere(key.level, self.around);
        let mut generator = Generator::new(layers, &self.pyramid, key)
            .with_surface(self.rules.clone())
            .with_materials(materials);
        let mut unlimited = u64::MAX;
        let in_place = generator
            .advance(&mut unlimited)
            .ok_or_else(|| format!("{what} fails in place"))?;
        observed.check(codec::encode(&in_place) == whole, || {
            format!("{what}: the job differs from the tile generated in place")
        });
        observed.check(worker.sliced(&TILE_BUILD, &input, SLICE)? == whole, || {
            format!("{what}: the job run a slice at a time differs from the job run whole")
        });
        let tile: Tile = codec::decode(&whole).map_err(|error| error.to_string())?;
        Ok((tile, whole))
    }
}
