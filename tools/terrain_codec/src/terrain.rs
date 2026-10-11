use std::sync::Arc;

use engine_terrain::codec::{self, Codec, Writer};
use engine_terrain::jobs::{self, TerrainRecipe, TerrainWorker};
use engine_terrain::{
    AuthoredPlacement, Biome, BiomeLayer, Biomes, Blend, ErosionLayer, ErosionStage, Generator,
    HeightImage, HeightmapLayer, HydrologyLayer, HydrologyStage, Layer, Layers, Mask, NoiseLayer,
    PATCH_QUADS, PATCHES_PER_TILE, PathLayer, PathStage, PlacementChoice, PlacementLayer, Placing,
    Pyramid, Stamp, StampOp, TILE_STRIDE, Tile, TileBand, TileKey, merge_bands,
};
use glam::{Vec2, Vec3};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn require(value: bool, why: &str) -> Result<(), String> {
    if value { Ok(()) } else { Err(why.into()) }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

fn same_tile(a: &Tile, b: &Tile) -> Result<(), String> {
    require(a.key() == b.key(), "tile identity changed")?;
    require(
        a.spacing().to_bits() == b.spacing().to_bits(),
        "tile spacing changed",
    )?;
    require(
        same_bits(a.heights(), b.heights()),
        "tile height bits changed",
    )?;
    require(a.hole_bits() == b.hole_bits(), "tile holes changed")?;
    require(a.control() == b.control(), "tile controls changed")?;
    require(a.placing() == b.placing(), "tile placing changed")?;
    require(
        codec::encode(&a.placed().to_vec()) == codec::encode(&b.placed().to_vec()),
        "placed object bits changed",
    )?;
    require(same_bits(&a.bounds(), &b.bounds()), "tile bounds changed")?;
    for z in 0..PATCHES_PER_TILE {
        for x in 0..PATCHES_PER_TILE {
            let bounds = b.patch_bounds(x, z);
            require(
                same_bits(&a.patch_bounds(x, z), &bounds),
                "patch bounds changed",
            )?;
            require(
                a.patch_has_holes(x, z) == b.patch_has_holes(x, z),
                "patch hole summary changed",
            )?;
            for row in z * PATCH_QUADS..=(z + 1) * PATCH_QUADS {
                for column in x * PATCH_QUADS..=(x + 1) * PATCH_QUADS {
                    let height = b.height(column as i32, row as i32);
                    require(
                        height >= bounds[0] && height <= bounds[1],
                        "patch bound not conservative",
                    )?;
                }
            }
            let point = Vec2::new(x as f32 * 16.0 + 0.25, z as f32 * 16.0 + 0.75);
            let left = a.sample(point).ok_or("missing serial terrain sample")?;
            let right = b.sample(point).ok_or("missing decoded terrain sample")?;
            require(
                left.height.to_bits() == right.height.to_bits()
                    && same_bits(&left.normal.to_array(), &right.normal.to_array())
                    && left.hole == right.hole,
                "height/normal/hole sample changed",
            )?;
        }
    }
    require(
        codec::encode(a) == codec::encode(b),
        "complete tile bytes changed",
    )
}

fn uncompressed_leaves() -> Result<Value, String> {
    let mut fixtures = Vec::new();
    fn fixed<T: Codec>(
        name: &str,
        value: &T,
        expected: &[u8],
        fixtures: &mut Vec<Value>,
    ) -> Result<(), String> {
        let actual = codec::encode(value);
        require(actual == expected, &format!("baseline leaf {name} changed"))?;
        let decoded: T = codec::decode(expected).map_err(|error| error.to_string())?;
        require(
            codec::encode(&decoded) == expected,
            "baseline leaf decode changed",
        )?;
        fixtures.push(json!({"name": name, "bytes": actual}));
        Ok(())
    }
    fixed(
        "u8-vector",
        &vec![0_u8, 127, 255],
        &[3, 0, 0, 0, 0, 0, 0, 0, 0, 127, 255],
        &mut fixtures,
    )?;
    fixed(
        "u16-box",
        &vec![0x1234_u16, 0xabcd].into_boxed_slice(),
        &[2, 0, 0, 0, 0, 0, 0, 0, 0x34, 0x12, 0xcd, 0xab],
        &mut fixtures,
    )?;
    fixed(
        "u32-arc",
        &Arc::<[u32]>::from(vec![0x1234_5678]),
        &[1, 0, 0, 0, 0, 0, 0, 0, 0x78, 0x56, 0x34, 0x12],
        &mut fixtures,
    )?;
    fixed(
        "u64-vector",
        &vec![0x0123_4567_89ab_cdef_u64],
        &[
            1, 0, 0, 0, 0, 0, 0, 0, 0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 1,
        ],
        &mut fixtures,
    )?;
    fixed(
        "tile-key",
        &TileKey::new(3, 0x1234, 0xabcd),
        &[3, 0x34, 0x12, 0xcd, 0xab],
        &mut fixtures,
    )?;
    fixed(
        "scalar-nan",
        &f32::from_bits(0xff80_1234),
        &[0x34, 0x12, 0x80, 0xff],
        &mut fixtures,
    )?;
    fixed(
        "fixed-float-pair",
        &[1.0_f32, -0.0],
        &[0, 0, 128, 63, 0, 0, 0, 128],
        &mut fixtures,
    )?;
    fixed(
        "vec2",
        &Vec2::new(1.0, -0.0),
        &[0, 0, 128, 63, 0, 0, 0, 128],
        &mut fixtures,
    )?;
    fixed(
        "vec3",
        &Vec3::new(1.0, -0.0, -2.0),
        &[0, 0, 128, 63, 0, 0, 0, 128, 0, 0, 0, 192],
        &mut fixtures,
    )?;
    fixed(
        "vector-of-fixed-pairs",
        &vec![[1.0_f32, -0.0]],
        &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 128],
        &mut fixtures,
    )?;
    Ok(json!(fixtures))
}

fn layers() -> Result<Layers, String> {
    let image = HeightImage::new(
        17,
        17,
        (0..289).map(|i| ((i * 73) % 65536) as u16).collect(),
    )
    .ok_or("height image")?;
    Layers::new(vec![
        Layer::Noise(NoiseLayer {
            seed: 19,
            height: 32.0,
            wavelength: 120.0,
            ..NoiseLayer::default()
        }),
        Layer::Heightmap(HeightmapLayer {
            image: Arc::new(image),
            min: Vec2::splat(80.0),
            size: Vec2::splat(80.0),
            low: 0.0,
            high: 8.0,
            falloff: 8.0,
            blend: Blend::Add,
            uri: None,
            source: None,
            attribution: None,
        }),
        Layer::Erosion(ErosionStage {
            layer: ErosionLayer {
                level: 0,
                rounds: 2,
                ..ErosionLayer::default()
            },
            baked: None,
        }),
        Layer::Hydrology(HydrologyStage {
            layer: HydrologyLayer {
                level: 0,
                channel_area: 20000.0,
                ..HydrologyLayer::default()
            },
            baked: None,
        }),
        Layer::Path(PathStage {
            layer: Arc::new(PathLayer {
                points: vec![[20.0, 20.0], [80.0, 60.0], [220.0, 220.0]],
                smoothing: 24.0,
                ..PathLayer::default()
            }),
            baked: None,
        }),
        Layer::Stamp(Stamp {
            center: [30.0, 40.0],
            radius: 4.0,
            falloff: 0.0,
            op: StampOp::Hole,
        }),
    ])
    .map_err(|error| error.to_string())
}

fn surface() -> Biomes {
    Biomes {
        materials: vec!["ground".into(), "rock".into()],
        biomes: vec![Biome {
            id: "fixture".into(),
            fitness: Mask::one(),
            layers: vec![
                BiomeLayer {
                    material: "ground".into(),
                    mask: Mask::one(),
                    opacity: 1.0,
                },
                BiomeLayer {
                    material: "rock".into(),
                    mask: Mask::Slope([15.0, 35.0]),
                    opacity: 1.0,
                },
            ],
        }],
        assets: vec!["stone".into()],
        placements: vec![PlacementLayer {
            id: "stones".into(),
            assets: vec![PlacementChoice {
                asset: "stone".into(),
                weight: 1.0,
            }],
            density: 30.0,
            footprint: 16.0,
            ..PlacementLayer::default()
        }],
        placed: vec![AuthoredPlacement {
            asset: "stone".into(),
            position: [120.0, 180.0],
            yaw: 35.0,
            ..AuthoredPlacement::default()
        }],
    }
}

pub(super) fn run() -> Result<Value, String> {
    let leaves = uncompressed_leaves()?;
    let pyramid = Pyramid::new([1, 1], 1.0).map_err(|error| error.to_string())?;
    let key = TileKey::new(0, 0, 0);
    let layers = layers()?;
    let serial_baked = layers.bake_all(&pyramid);
    require(serial_baked.unbaked().is_none(), "serial bake incomplete")?;
    let mut worker = TerrainWorker::new();
    let baked_bytes = worker.run(jobs::BAKE_JOB, &jobs::bake_input(&layers, &pyramid))?;
    let job_baked = jobs::bake_output(&baked_bytes).map_err(|error| error.to_string())?;
    require(serial_baked == job_baked, "baked stack values changed")?;
    require(
        codec::encode(&serial_baked) == baked_bytes && codec::encode(&job_baked) == baked_bytes,
        "baked stack bits changed",
    )?;
    let hydrology = job_baked.hydrology().ok_or("missing hydrology fixture")?;
    let serial_water = serial_baked.hydrology().ok_or("missing serial hydrology")?;
    require(
        same_bits(&hydrology.water.surface, &serial_water.water.surface)
            && same_bits(&hydrology.water.depth, &serial_water.water.depth),
        "water bits changed",
    )?;
    let recipe = TerrainRecipe {
        stack: "layers".into(),
        view: None,
        pyramid,
        surface: surface(),
        materials: true,
    };
    let rules = Arc::new(
        recipe
            .surface
            .compile()
            .map_err(|error| error.to_string())?,
    );
    let layers = Arc::new(serial_baked);
    worker.share("layers", &jobs::share_stack(&job_baked))?;
    worker.share("recipe", &jobs::share_recipe(&recipe))?;
    let mut generator = Generator::new(layers.clone(), &pyramid, key).with_surface(rules.clone());
    let mut budget = u64::MAX;
    let serial = generator
        .advance(&mut budget)
        .ok_or("serial tile unfinished")?;
    require(
        !serial.placed().is_empty() && serial.control().is_some() && serial.hole_bits().is_some(),
        "fixture must include objects, controls and holes",
    )?;
    let complete_bytes = worker.run(
        jobs::TILE_JOB,
        &jobs::tile_input("recipe", key, Placing::All, false),
    )?;
    let complete = jobs::tile_output(&complete_bytes).map_err(|error| error.to_string())?;
    same_tile(&serial, &complete)?;

    let mut staged_serial = Generator::new(layers.clone(), &pyramid, key)
        .with_surface(rules.clone())
        .with_staged(true);
    let mut serial_shape = None;
    loop {
        let tile = staged_serial.advance(&mut 1024);
        serial_shape = staged_serial.take_shape().or(serial_shape);
        if let Some(tile) = tile {
            same_tile(&serial, &tile)?;
            break;
        }
        require(!staged_serial.failed(), "staged serial failed")?;
    }
    let serial_shape = serial_shape.ok_or("no serial shape")?;
    worker.start(
        7,
        jobs::TILE_JOB,
        &jobs::tile_input("recipe", key, Placing::All, true),
    )?;
    let mut parts = 0;
    let mut staged_bytes = 0;
    loop {
        let done = worker.step(7, 1024)?;
        if let Some(bytes) = worker.take_part(7) {
            let part = jobs::tile_output(&bytes).map_err(|error| error.to_string())?;
            same_tile(&serial_shape, &part)?;
            parts += 1;
            staged_bytes += bytes.len();
        }
        if let Some(bytes) = done {
            same_tile(
                &serial,
                &jobs::tile_output(&bytes).map_err(|error| error.to_string())?,
            )?;
            break;
        }
    }
    require(parts == 1, "staged shape missing or repeated")?;

    let stride = TILE_STRIDE as usize;
    let partitions = [vec![0, 1, 4, 17, 128, 258, 259], (0..=stride).collect()];
    let mut band_reports = Vec::new();
    for boundaries in partitions {
        let mut bands = Vec::new();
        let mut bytes_total = 0;
        for window in boundaries.windows(2) {
            let rows = window[0]..window[1];
            let mut direct = Generator::band(layers.clone(), &pyramid, key, rows.clone())
                .ok_or("valid band refused")?
                .with_surface(rules.clone());
            let mut budget = u64::MAX;
            let _ = direct.advance(&mut budget);
            let direct = direct.take_band().ok_or("serial band unfinished")?;
            let bytes = worker.run(
                jobs::BAND_JOB,
                &jobs::band_input("recipe", key, rows.start as u32..rows.end as u32),
            )?;
            let decoded = jobs::band_output(&bytes).map_err(|error| error.to_string())?;
            require(
                same_bits(&direct.heights, &decoded.heights)
                    && direct.key == decoded.key
                    && direct.first == decoded.first
                    && direct.holes == decoded.holes
                    && direct.control == decoded.control,
                "band bits or metadata changed",
            )?;
            require(
                codec::encode(&direct) == bytes,
                "band serialized bytes changed",
            )?;
            bytes_total += bytes.len();
            bands.push(decoded);
        }
        // Merging is independent of arrival order and retains all rows.
        bands.reverse();
        let merged = merge_bands(key, 1.0, &bands).ok_or("valid bands did not merge")?;
        let reference = serial.clone().without_placed();
        same_tile(&reference, &merged)?;
        require(
            merge_bands(key, 1.0, &bands[1..]).is_none(),
            "missing rows accepted",
        )?;
        let mut invalid = bands.clone();
        invalid.push(bands[0].clone());
        require(
            merge_bands(key, 1.0, &invalid).is_none(),
            "overlapping rows accepted",
        )?;
        band_reports.push(json!({"boundaries": boundaries, "bytes": bytes_total}));
    }

    require(
        Tile::new(key, 1.0, vec![0.0; 3].into_boxed_slice(), None).is_none(),
        "invalid tile size accepted",
    )?;
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut heights = vec![0.0; stride * stride];
        heights[258] = invalid;
        require(
            Tile::new(key, 1.0, heights.clone().into_boxed_slice(), None).is_none(),
            "nonfinite tile accepted",
        )?;
        let mut out = Writer::new();
        out.put(&key)
            .put(&1.0_f32)
            .put(&heights)
            .put(&Option::<Vec<u64>>::None)
            .put(&Option::<Vec<u32>>::None)
            .put(&Option::<Vec<engine_terrain::Placed>>::None)
            .put(&Option::<Placing>::None);
        require(
            jobs::tile_output(&out.into_bytes()).is_err(),
            "wire nonfinite tile accepted",
        )?;
        let band = TileBand {
            key,
            first: 0,
            heights,
            holes: None,
            control: None,
        };
        require(
            merge_bands(key, 1.0, &[band]).is_none(),
            "nonfinite merged tile accepted",
        )?;
    }
    Ok(json!({
        "baseline_leaves": leaves, "tile_bytes": complete_bytes.len(), "tile_sha256": digest(&complete_bytes),
        "baked_bytes": baked_bytes.len(), "baked_sha256": digest(&baked_bytes), "staged_parts": parts,
        "staged_bytes": staged_bytes, "placements": serial.placed().len(), "height_words": serial.heights().len(),
        "water_words": hydrology.water.surface.len() + hydrology.water.depth.len(), "bands": band_reports,
        "invalid_tiles": 7, "patches_checked": PATCHES_PER_TILE * PATCHES_PER_TILE,
    }))
}
