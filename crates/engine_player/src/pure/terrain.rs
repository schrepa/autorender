use std::collections::BTreeMap;
use std::sync::Arc;

use engine::BrickList;
use engine::scene::{SceneDocument, SceneKinds};
use engine_core::codec;
use engine_core::{JobKind, PureSession, RecordKey};
use engine_heightfield::layer::LayerKind;
use engine_heightfield::path::lay_decks;
use engine_heightfield::tile::{
    BiomeRules, Generator, StackView, TerrainRecipe, Tile, TileJob, shaded_elsewhere,
};
use engine_heightfield::{
    Layers, Pyramid, Stamp, StampOp, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey,
};
use engine_navigation::{
    Gait, NavOptions, NavPath, PORTALS, PortalsJob, Stepped, TilePortals, WalkData, WalkGround,
    WalkObstacle, WalkSearch,
};
use engine_open_world_kit::terrain::TerrainLayers;
use engine_terrain::jobs::{self, BAKE, BakeJob, TILE_BUILD};
use engine_terrain::{Terrain, TerrainScene};
use engine_terrain_shading::jobs::lighting::{self, Ground};
use engine_terrain_shading::jobs::{
    BOUNCE, BounceJob, LIGHTING, LightingJob, MACRO_COLORS, MacroJob, bounce, macro_colors,
};
use glam::Vec2;
use serde_json::json;

const FIXTURE: &str = "web/scripts/fixtures/terrain.scene.json";
const ENTITY: &str = "ground";
const SLICE: u64 = 4096;
const PINNED: u32 = 2;
const SKY_TEXELS: u32 = 256;
const HORIZON_TEXELS: u32 = 128;
const BOUNCE_PROBES: u32 = 64;
const MACRO_TEXELS: u32 = 256;
pub const KINDS: &[&JobKind] = &[
    &BAKE,
    &TILE_BUILD,
    &LIGHTING,
    &BOUNCE,
    &MACRO_COLORS,
    &PORTALS,
];

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub fn run(session: &mut dyn PureSession) -> Result<(), String> {
    let terrain = load(session.root())?;
    let authored = &terrain.layers;
    let pyramid = terrain.pyramid;
    let kinds: Vec<&'static LayerKind> =
        session.list().members::<TerrainLayers>().copied().collect();

    let baked = jobs::bake::bake_all(authored, &pyramid);
    let bytes = jobs::share_stack(&baked).to_vec();
    session.hash("bakes", &bytes);
    let mut records = vec![(BAKE.name(), bytes.clone())];
    let same = session.run(
        &BAKE,
        &BakeJob {
            layers: Layers::clone(authored),
            pyramid,
        }
        .input(),
    )? == bytes;
    session.check(same, &|| {
        "the bake job differs from the bakes in place".into()
    });
    record_bakes(&baked, session);

    let baked = Arc::new(baked);
    let rules = terrain.surface.rules.clone();
    let stack = session.share(&jobs::share_stack(&baked));
    let shared = session.share(&jobs::share_recipe(&TerrainRecipe {
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
            let (tile, bytes) = recipe.tile(session, key, (&[], &kinds))?;
            session.hash(&format!("tile {}", name(key)), &bytes);
            records.push((TILE_BUILD.name(), bytes));
            tiles.insert(key, tile);
        }
    }
    let objects: usize = tiles.values().map(|tile| tile.placed().len()).sum();
    session.record("tiles", tiles.len().into());
    session.record("tile objects", objects.into());
    session.check(objects > 0, &|| "no tile holds objects".into());

    let job = lighting_job(&pyramid, shared);
    let input = codec::encode(&job);
    let bytes = session.run(&LIGHTING, &input)?;
    let (samples, spacing, heights) = level_heights(&pyramid, &tiles);
    let ground = Ground {
        samples,
        spacing,
        heights: &heights,
    };
    let extent = pyramid.extent().to_array();
    let made = lighting::light(&ground, extent, job.sky, job.horizons);
    session.check(codec::encode(&made) == bytes, &|| {
        "the lighting job differs from the lighting made in place".into()
    });
    let same = session.sliced(&LIGHTING, &input, SLICE)? == bytes;
    session.check(same, &|| {
        "the lighting job run a slice at a time differs from the job run whole".into()
    });
    session.hash("lighting", &bytes);
    let sky = made.sky_layers();
    let area = (SKY_TEXELS * SKY_TEXELS) as usize;
    let seen: u64 = sky[..area * 4]
        .chunks(4)
        .map(|texel| u64::from(texel[0]))
        .sum();
    session.record(
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
    let transfer = bounce_transfer(session, (&job, &ground, extent), surface)?;
    records.push((BOUNCE.name(), transfer));
    let colors = MacroJob {
        recipe: shared.hash(),
        level: pyramid.levels().saturating_sub(PINNED),
        size: MACRO_TEXELS,
    };
    let bytes = session.run(&MACRO_COLORS, &codec::encode(&colors))?;
    let made = macro_colors::macro_colors(
        &baked,
        &pyramid,
        colors.level,
        colors.size,
        &rules.authored().macro_colors,
    )
    .ok_or("the macro colors' heights are invalid")?;
    session.check(made == bytes, &|| {
        "the macro colors job differs from the colors made in place".into()
    });
    let same = session.sliced(&MACRO_COLORS, &codec::encode(&colors), SLICE)? == bytes;
    session.check(same, &|| {
        "the macro colors job run a slice at a time differs from the job run whole".into()
    });
    session.hash("macro colors", &bytes);
    let texels = bytes.len() / 4;
    let mean = |channel: usize| {
        let sum: u64 = bytes.chunks(4).map(|texel| u64::from(texel[channel])).sum();
        sum as f64 / texels.max(1) as f64 / 255.0
    };
    let tint = |channel: usize| mean(channel) * 255.0 / 128.0;
    session.record(
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
    let (tile, bytes) = recipe.tile(session, edited, (&edits, &kinds))?;
    session.hash(&format!("edited tile {}", name(edited)), &bytes);
    session.check(tile.heights() != tiles[&edited].heights(), &|| {
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
    let around_shared = session.share(&jobs::share_recipe(&TerrainRecipe {
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
    session.record("surroundings tiles", around.tiles(0)[0].into());
    for level in (around.levels().saturating_sub(2)..around.levels()).rev() {
        for key in around.level_tiles(level) {
            let (_, bytes) = around_recipe.tile(session, key, (&[], &kinds))?;
            session.hash(&format!("surroundings {}", name(key)), &bytes);
            records.push((TILE_BUILD.name(), bytes));
        }
    }

    let gait = NavOptions::default().gait(&pyramid);
    let mut graph = Graph::default();
    let mut bytes_held = 0;
    for key in pyramid.level_tiles(gait.level) {
        let (ground, bytes) = portals(session, (&tiles[&key], &baked), (&pyramid, &gait), &[])?;
        session.hash(&format!("portals {}", name(key)), &bytes);
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
    let (_, bytes) = portals(session, (&tile, &edited_layers), (&pyramid, &gait), &edits)?;
    session.hash(&format!("edited portals {}", name(edited)), &bytes);
    let count: usize = graph
        .portals
        .values()
        .map(|held| held.portals().len())
        .sum();
    session.record(
        "walk graph",
        json!({"tiles": graph.portals.len(), "portals": count, "bytes": bytes_held}),
    );
    session.check(count > 0, &|| "no tile has portals".into());

    let route = (Vec2::new(212.0, 262.0), Vec2::new(412.0, 262.0));
    let (boulder, boulder_radius) = (Vec2::new(312.0, 262.0), 12.0);
    let obstacles = [WalkObstacle::Circle {
        center: boulder,
        radius: boulder_radius,
    }];
    let walked = |slice: u64| graph.walk(&pyramid, route, &obstacles, slice);
    let whole = codec::encode(&walked(u64::MAX)?);
    session.check(whole == codec::encode(&walked(SLICE)?), &|| {
        "the walk a slice at a time differs from the walk whole".into()
    });
    session.hash("walk", &whole);
    let found = walked(u64::MAX)?.ok_or("no walk reaches over the hills")?;
    session.check(found.length > route.1.distance(route.0), &|| {
        "the walk is no longer than the straight line it leaves around the boulder".into()
    });
    session.check(
        found.points.iter().all(|point| {
            Vec2::new(point[0], point[2]).distance(boulder)
                >= boulder_radius + NavOptions::default().radius - 1.0
        }),
        &|| "the walk crosses the boulder".into(),
    );
    session.record(
        "walk",
        json!({"points": found.points.len(), "length": found.length, "level": found.level}),
    );

    let pack = pack(&records, session)?;
    session.hash("pack", &pack);
    Ok(())
}

fn load(root: &std::path::Path) -> Result<Terrain, String> {
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
    session: &mut dyn PureSession,
    (job, ground, extent): (&LightingJob, &Ground<'_>, [f32; 2]),
    (baked, rules): (&Layers, &BiomeRules),
) -> Result<Vec<u8>, String> {
    let transfer = BounceJob {
        recipe: job.recipe,
        level: job.level,
        tiles: job.tiles.clone(),
        probes: BOUNCE_PROBES,
    };
    let input = codec::encode(&transfer);
    let bytes = session.run(&BOUNCE, &input)?;
    let made = bounce::bounce(ground, extent, (baked, rules), BOUNCE_PROBES);
    session.check(codec::encode(&made) == bytes, &|| {
        "the bounce job differs from the transfer made in place".into()
    });
    let same = session.sliced(&BOUNCE, &input, SLICE)? == bytes;
    session.check(same, &|| {
        "the bounce job run a slice at a time differs from the job run whole".into()
    });
    session.hash("bounce", &bytes);
    let blocks = (BOUNCE_PROBES / 2) as usize * (BOUNCE_PROBES / 2) as usize;
    session.record(
        "bounce",
        json!({
            "blocks": blocks,
            "detailed blocks": made.detailed_blocks(),
            "probes": made.probe_count(),
        }),
    );
    session.check(made.probe_count() >= blocks, &|| {
        "the bounce transfer holds fewer probes than blocks".into()
    });
    Ok(bytes)
}

fn name(key: TileKey) -> String {
    format!("{}/{}/{}", key.level, key.x, key.z)
}

fn pack(records: &[(&str, Vec<u8>)], session: &mut dyn PureSession) -> Result<Vec<u8>, String> {
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
    session.record("pack records", index.len().into());
    for (key, bytes, _) in &keyed {
        let hex: String = key.iter().map(|byte| format!("{byte:02x}")).collect();
        let entry = index.find(&hex).ok_or("a record missing from the pack")?;
        let read = engine_worker::unpack(&entry, range(entry.offset, u64::from(entry.stored))?)
            .map_err(|error| error.to_string())?;
        session.check(read == *bytes, &|| {
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
    session: &mut dyn PureSession,
    (tile, layers): (&Tile, &Arc<Layers>),
    (pyramid, gait): (&Pyramid, &Gait),
    stamps: &[Stamp],
) -> Result<(WalkGround, Vec<u8>), String> {
    let key = tile.key();
    let ground = walk_ground(tile, layers, pyramid);
    let job = PortalsJob {
        ground: ground.clone(),
        pyramid: *pyramid,
        gait: *gait,
    };
    let bytes = session.run(&PORTALS, &job.input())?;
    session.check(
        codec::encode(&TilePortals::new(&ground, pyramid, gait)) == bytes,
        &|| {
            format!(
                "tile {} under {} edits: its portals' job differs from its record's portals",
                name(key),
                stamps.len()
            )
        },
    );
    Ok((ground, bytes))
}

fn walk_ground(tile: &Tile, layers: &Layers, pyramid: &Pyramid) -> WalkGround {
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
    lay_decks(&layers.bridges(), first, spacing, &mut heights);
    WalkGround::new((key, spacing), heights, holes, walk_water(tile))
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
        obstacles: &[WalkObstacle],
        slice: u64,
    ) -> Result<Option<NavPath>, String> {
        let mut search =
            WalkSearch::new(pyramid, route, obstacles.to_vec(), &NavOptions::default());
        loop {
            let mut budget = slice;
            match search.step(&mut budget, self) {
                Stepped::Found(found) => return Ok(found),
                Stepped::Blocked => {
                    return Err(format!(
                        "the walk waits for {:?}, which the corpus holds",
                        search.wants()
                    ));
                }
                Stepped::Spent => {}
            }
        }
    }
}

fn record_bakes(layers: &Layers, session: &mut dyn PureSession) {
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
            session.record(&format!("bake {index}"), stats);
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
        session: &mut dyn PureSession,
        key: TileKey,
        (stamps, kinds): (&[Stamp], &[&'static LayerKind]),
    ) -> Result<(Tile, Vec<u8>), String> {
        let job = TileJob {
            recipe: self.key.hash(),
            key,
            stamps: stamps.to_vec(),
        };
        let input = job.identity();
        let whole = session.run(&TILE_BUILD, &input)?;
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
        session.check(codec::encode(&in_place) == whole, &|| {
            format!("{what}: the job differs from the tile generated in place")
        });
        let same = session.sliced(&TILE_BUILD, &input, SLICE)? == whole;
        session.check(same, &|| {
            format!("{what}: the job run a slice at a time differs from the job run whole")
        });
        let tile: Tile = codec::decode(&whole).map_err(|error| error.to_string())?;
        Ok((tile, whole))
    }
}
