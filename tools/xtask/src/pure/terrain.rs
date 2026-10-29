use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use engine_navigation::{
    Gait, NavOptions, NavPath, PortalsJob, TilePortals, WalkData, WalkGround, WalkSearch,
};
use engine_scene::{SceneDocument, SceneFileKind, SceneFiles};
use engine_terrain::bounce;
use engine_terrain::jobs::{
    self, BakeJob, BounceJob, LightingJob, MacroJob, StackView, TerrainRecipe, TileJob, WaterJob,
};
use engine_terrain::lighting::{self, Ground};
use engine_terrain::macro_colors;
use engine_terrain::{
    BiomeRules, Generator, Layers, PATCH_QUADS, PATCHES_PER_TILE, Pyramid, SettleRegion,
    SettleStart, Stamp, StampOp, TILE_QUADS, Terrain, Tile, TileKey, TileShading, WaterPart, codec,
};
use glam::Vec2;
use serde_json::json;

use super::{Observations, hash};

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
    let mut worker = Worker::default();

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
        around: false,
    };

    let mut tiles = BTreeMap::new();
    for level in (0..pyramid.levels()).rev() {
        for key in pyramid.level_tiles(level) {
            let (tile, bytes) = recipe.tile(&mut worker, key, &[], observed)?;
            observed.hash(format!("tile {}", name(key)), &bytes);
            records.push((TileJob::KIND.name(), bytes));
            tiles.insert(key, tile);
        }
    }
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

    let job = lighting_job(&pyramid);
    let input = job.input();
    let bytes = worker.run(LightingJob::KIND.name(), &input)?;
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

    let surface = (baked.as_ref(), rules.as_ref());
    let transfer = bounce_transfer(&mut worker, (&job, &ground, extent), surface, observed)?;
    records.push((BounceJob::KIND.name(), transfer));
    let colors = MacroJob {
        recipe: "recipe".into(),
        level: pyramid.levels().saturating_sub(PINNED),
        size: MACRO_TEXELS,
    };
    let bytes = worker.run(MacroJob::KIND.name(), &colors.identity())?;
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
        worker.sliced(MacroJob::KIND.name(), &colors.identity())? == bytes,
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
    let (tile, bytes) = recipe.tile(&mut worker, edited, &edits, observed)?;
    observed.hash(format!("edited tile {}", name(edited)), &bytes);
    observed.check(tile.heights() != tiles[&edited].heights(), || {
        format!("the edits leave tile {} as it was", name(edited))
    });
    records.push((TileJob::KIND.name(), bytes));

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
        around: true,
    };
    observed.record("surroundings tiles", around.tiles(0)[0]);
    for level in (around.levels().saturating_sub(2)..around.levels()).rev() {
        for key in around.level_tiles(level) {
            let (_, bytes) = around_recipe.tile(&mut worker, key, &[], observed)?;
            observed.hash(format!("surroundings {}", name(key)), &bytes);
            records.push((TileJob::KIND.name(), bytes));
        }
    }

    let gait = NavOptions::default().gait(&pyramid);
    let mut graph = Graph::default();
    let mut bytes_held = 0;
    for key in pyramid.level_tiles(gait.level) {
        let (ground, bytes) = portals(
            &mut worker,
            (&tiles[&key], &baked),
            (&pyramid, &gait),
            (&[], observed),
        )?;
        observed.hash(format!("portals {}", name(key)), &bytes);
        let held: TilePortals = codec::decode(&bytes).map_err(|error| error.to_string())?;
        graph.portals.insert(key, Arc::new(held));
        graph.grounds.insert(key, Arc::new(ground));
        bytes_held += bytes.len();
        records.push((PortalsJob::KIND, bytes));
    }
    let edited_layers = Arc::new(
        baked
            .with_stamps(&edits)
            .map_err(|error| error.to_string())?,
    );
    let (_, bytes) = portals(
        &mut worker,
        (&tile, &edited_layers),
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
    let obstacles = [engine_core::GroundObstacle {
        center: Vec2::new(312.0, 262.0),
        radius: 12.0,
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
            Vec2::new(point[0], point[2]).distance(obstacles[0].center)
                >= obstacles[0].radius + NavOptions::default().radius - 1.0
        }),
        || "the walk crosses the boulder".into(),
    );
    observed.record(
        "walk",
        json!({"points": found.points.len(), "length": found.length, "level": found.level}),
    );

    let watered = engine_terrain::watered_tiles(&baked, &pyramid);
    let key = *watered
        .first()
        .ok_or("the fixture's bake waters no finest tile")?;
    let region = SettleRegion::alone(&pyramid, key, WATER_CELL);
    let mut parts = Vec::new();
    for tile in region.reached(&pyramid) {
        let input = WaterJob {
            recipe: "recipe".into(),
            key: tile,
            stamps: Vec::new(),
            solids: Vec::new(),
            cell: WATER_CELL,
        }
        .input();
        let bytes = worker.run(WaterJob::KIND.name(), &input)?;
        observed.check(
            worker.sliced(WaterJob::KIND.name(), &input)? == bytes,
            || {
                format!(
                    "the water's part of {} run a slice at a time differs",
                    name(tile)
                )
            },
        );
        observed.hash(format!("water part {}", name(tile)), &bytes);
        parts.push(jobs::water_output(&bytes).map_err(|error| error.to_string())?);
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
            "cook regions": engine_terrain::settle_regions(&baked, &pyramid, WATER_CELL).len(),
        }),
    );

    let pack = pack(&records, observed)?;
    observed.hash("pack", &pack);

    worker.write(root)
}

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

fn lighting_job(pyramid: &Pyramid) -> LightingJob {
    let level = pyramid.levels().saturating_sub(PINNED);
    LightingJob {
        recipe: "recipe".into(),
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
    worker: &mut Worker,
    (job, ground, extent): (&LightingJob, &Ground<'_>, [f32; 2]),
    (baked, rules): (&Layers, &BiomeRules),
    observed: &mut Observations,
) -> Result<Vec<u8>, String> {
    let transfer = BounceJob {
        recipe: "recipe".into(),
        level: job.level,
        tiles: job.tiles.clone(),
        probes: BOUNCE_PROBES,
    };
    let input = transfer.input();
    let bytes = worker.run(BounceJob::KIND.name(), &input)?;
    let made = bounce::bounce(ground, extent, (baked, rules), BOUNCE_PROBES);
    observed.check(codec::encode(&made) == bytes, || {
        "the bounce job differs from the transfer made in place".into()
    });
    observed.check(
        worker.sliced(BounceJob::KIND.name(), &input)? == bytes,
        || "the bounce job run a slice at a time differs from the job run whole".into(),
    );
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

fn culling_holds(tile: &Tile) -> bool {
    let quads = PATCH_QUADS as i32;
    let half = quads / 2;
    let inverse = 1.0 / tile.spacing();
    let rounding = std::f32::consts::SQRT_2 * codec::HEIGHT_STEP * inverse;
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
                    (dx * dx + dz * dz).sqrt() <= radius + rounding
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
                (z0..=z0 + half).all(|z| {
                    (x0..=x0 + half).all(|x| tile.height(x, z) >= floor - codec::HEIGHT_STEP)
                })
            });
        slopes && floors
    })
}

fn name(key: TileKey) -> String {
    format!("{}/{}/{}", key.level, key.x, key.z)
}

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
    worker: &mut Worker,
    (tile, layers): (&Tile, &Arc<Layers>),
    (pyramid, gait): (&Pyramid, &Gait),
    (stamps, observed): (&[Stamp], &mut Observations),
) -> Result<(WalkGround, Vec<u8>), String> {
    let key = tile.key();
    let job = PortalsJob {
        source: codec::encode(&(String::from("stack"), (*pyramid, stamps.to_vec()))),
        pyramid: *pyramid,
        key,
        gait: *gait,
    };
    let bytes = worker.run(PortalsJob::KIND, &job.identity())?;
    let ground = WalkGround::new(&walk_grid(tile, layers, pyramid), pyramid);
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

fn walk_grid(tile: &Tile, layers: &Layers, pyramid: &Pyramid) -> engine_core::GroundGrid {
    let side = engine_terrain::TILE_SAMPLES as usize;
    let stride = engine_terrain::TILE_STRIDE as usize;
    let stored = |x: usize, z: usize| (z + 1) * stride + x + 1;
    let key = tile.key();
    let spacing = pyramid.spacing(u32::from(key.level));
    let surfaces = tile.water().map(engine_terrain::TileWater::surfaces);
    let mut heights = vec![0.0; side * side].into_boxed_slice();
    let mut water = surfaces.map(|_| vec![f32::NAN; side * side].into_boxed_slice());
    for z in 0..side {
        for x in 0..side {
            heights[z * side + x] = tile.heights()[stored(x, z)];
            if let (Some(water), Some(surfaces)) = (water.as_mut(), surfaces) {
                water[z * side + x] = surfaces[stored(x, z)];
            }
        }
    }
    let first = pyramid.first_sample(key);
    let at =
        |x: i64, z: i64| glam::Vec2::new((first[0] + x) as f32, (first[1] + z) as f32) * spacing;
    let quads = i64::from(engine_terrain::TILE_QUADS);
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
    engine_core::GroundGrid {
        key,
        spacing,
        heights,
        water,
        holes,
        bridges: layers
            .bridges()
            .into_iter()
            .map(|bridge| engine_core::GroundBridge {
                deck: bridge.deck,
                width: bridge.width,
                tunnel: bridge.tunnel,
            })
            .collect(),
    }
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
        obstacles: &[engine_core::GroundObstacle],
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
    key: &'static str,
    layers: Arc<Layers>,
    pyramid: Pyramid,
    rules: Arc<BiomeRules>,
    around: bool,
}

impl Recipe {
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
        let materials = !TileShading::shades(key.level, self.around);
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
        observed.check(
            worker.sliced(TileJob::KIND.name(), &input)? == whole,
            || format!("{what}: the job run a slice at a time differs from the job run whole"),
        );
        let tile: Tile = codec::decode(&whole).map_err(|error| error.to_string())?;
        Ok((tile, whole))
    }
}

#[derive(Default)]
struct Worker {
    worker: engine_worker::Worker,
    next: u64,
    log: Vec<serde_json::Value>,
}

impl Worker {
    fn share(&mut self, key: &str, bytes: &[u8]) -> Result<(), String> {
        self.log
            .push(json!({"share": key, "bytes": STANDARD.encode(bytes)}));
        self.worker.share(key, bytes)
    }

    fn run(&mut self, kind: &str, input: &[u8]) -> Result<Vec<u8>, String> {
        let output = self.worker.run(kind, input)?;
        self.log.push(json!({
            "job": kind,
            "input": STANDARD.encode(input),
            "output": hash(&output),
        }));
        Ok(output)
    }

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
