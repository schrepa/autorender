//! Highlands (decision 051): an 8 km terrain the engine generates from a
//! layer stack (eroded mountains, then hand-placed stamps: a plateau
//! leveled for a hamlet, a flat valley floor and a cave mouth cut as a
//! hole), streamed around the camera, with the scene's renderables standing
//! on the plateau as the tower field does. Its ground (decision 054) is
//! meadow, dry grass, dirt, scree, rock and snow, each textured by maps
//! built offline (`cargo run -p engine_assetc --release --bin
//! terrain_materials`) and fetched by the page, or in plain colors without
//! them. Boulders lie on its meadows, stones on its scree and outcrops on
//! its rock, and spruce forests cover the lower slopes (decision 057).

use std::collections::HashMap;

use engine::prelude::*;

use crate::{StressConfig, StressFile};

/// The images the page fetches for this scene, from `stress/terrain/`:
/// each surface material's base color, normal, and occlusion, roughness,
/// metalness and height maps, which `tools/assetc`'s `terrain_materials`
/// writes.
pub const FILES: [&str; 18] = [
    "grass_albedo.ktx2",
    "grass_normal.ktx2",
    "grass_orm.ktx2",
    "dry_grass_albedo.ktx2",
    "dry_grass_normal.ktx2",
    "dry_grass_orm.ktx2",
    "dirt_albedo.ktx2",
    "dirt_normal.ktx2",
    "dirt_orm.ktx2",
    "scree_albedo.ktx2",
    "scree_normal.ktx2",
    "scree_orm.ktx2",
    "rock_albedo.ktx2",
    "rock_normal.ktx2",
    "rock_orm.ktx2",
    "snow_albedo.ktx2",
    "snow_normal.ktx2",
    "snow_orm.ktx2",
];

/// How one surface material looks: its name, which its maps' files start
/// with; its linear color and roughness without maps, and the color its
/// maps' colors are multiplied by; the meters its maps repeat over; how
/// sharply its heights win where it blends; how it lies on steep ground;
/// and the cover growing where it lies (decision 056).
struct Look {
    name: &'static str,
    color: [f32; 3],
    roughness: f32,
    tint: [f32; 3],
    tiling: f32,
    height_contrast: f32,
    projection: SurfaceProjection,
    cover: Option<Cover>,
}

const LOOKS: [Look; 6] = [
    Look {
        name: "grass",
        color: [0.105, 0.135, 0.045],
        roughness: 0.92,
        tint: [1.0, 1.0, 1.0],
        tiling: 3.0,
        height_contrast: 0.35,
        projection: SurfaceProjection::Top,
        cover: Some(Cover {
            density: 140.0,
            height: [0.1, 0.34],
            width: 0.035,
            color: [0.16, 0.23, 0.06],
            sway: 0.6,
        }),
    },
    Look {
        name: "dry_grass",
        color: [0.2, 0.16, 0.08],
        roughness: 0.92,
        tint: [0.9, 0.9, 0.9],
        tiling: 3.5,
        height_contrast: 0.35,
        projection: SurfaceProjection::Top,
        cover: Some(Cover {
            density: 70.0,
            height: [0.14, 0.46],
            width: 0.025,
            color: [0.34, 0.27, 0.12],
            sway: 0.7,
        }),
    },
    Look {
        name: "dirt",
        color: [0.21, 0.165, 0.11],
        roughness: 0.92,
        tint: [1.0, 1.0, 1.0],
        tiling: 3.0,
        height_contrast: 0.5,
        projection: SurfaceProjection::Top,
        cover: None,
    },
    Look {
        name: "scree",
        color: [0.2, 0.19, 0.18],
        roughness: 0.82,
        tint: [0.78, 0.74, 0.68],
        tiling: 4.0,
        height_contrast: 0.8,
        projection: SurfaceProjection::Top,
        cover: None,
    },
    Look {
        name: "rock",
        color: [0.25, 0.235, 0.22],
        roughness: 0.82,
        tint: [0.74, 0.7, 0.64],
        tiling: 8.0,
        height_contrast: 0.7,
        projection: SurfaceProjection::Biplanar,
        cover: None,
    },
    Look {
        name: "snow",
        color: [0.82, 0.84, 0.88],
        roughness: 0.55,
        tint: [1.0, 1.0, 1.0],
        tiling: 6.0,
        height_contrast: 0.4,
        projection: SurfaceProjection::Top,
        cover: None,
    },
];

/// Level-0 tiles along each axis, and their sample spacing: 8,192 meters.
const TILES: u32 = 32;
const SPACING: f32 = 1.0;
const SIZE: f32 = TILES as f32 * 256.0 * SPACING;
/// The plateau's center, where the hamlet stands.
const PLATEAU: [f32; 2] = [0.0, 0.0];

/// The height the hamlet's plateau is leveled to.
#[derive(Resource, Clone, Copy)]
pub(crate) struct Plateau(pub f32);

/// Spawns the terrain, its materials' maps read from `files` where the page
/// supplied them; returns the region shadows cover and the height the
/// hamlet stands at.
pub(crate) fn populate(
    world: &mut World,
    config: &StressConfig,
    hamlet: f32,
    files: &HashMap<String, StressFile>,
) -> Result<((Vec3, Vec3), f32), String> {
    let half = SIZE * 0.5;
    // Terrain-local: the terrain's first corner is at -half.
    let local = |point: [f32; 2]| [point[0] + half, point[1] + half];
    let plateau = local(PLATEAU);
    let pyramid = Pyramid::new([TILES, TILES], SPACING).map_err(|error| error.to_string())?;
    let mountains = Layer::Noise(NoiseLayer {
        seed: config.seed,
        height: 1500.0,
        wavelength: 3200.0,
        ridges: 0.5,
        warp: 0.4,
        erosion: 0.9,
        smoothing: 0.6,
        valleys: 0.7,
        roughness: 0.3,
        ..NoiseLayer::default()
    });
    // Leveled where the mountains already stand there.
    let level = Layers::new(vec![mountains.clone()])
        .map_err(|error| error.to_string())?
        .height_at(&pyramid, Vec2::from(plateau));
    world.insert_resource(Plateau(level));
    let layers = Layers::new(vec![
        mountains,
        // The hamlet's plateau, wide enough for the field's grid.
        Layer::Stamp(Stamp {
            center: plateau,
            radius: hamlet,
            falloff: 220.0,
            op: StampOp::Flatten(level),
        }),
        // A wide valley floor to the north-east, leveled low.
        Layer::Stamp(Stamp {
            center: local([1500.0, -1300.0]),
            radius: 520.0,
            falloff: 700.0,
            op: StampOp::Flatten(60.0),
        }),
        // A cave mouth in the plateau's slope.
        Layer::Stamp(Stamp {
            center: [plateau[0] + hamlet + 120.0, plateau[1]],
            radius: 7.0,
            falloff: 0.0,
            op: StampOp::Hole,
        }),
    ])
    .map_err(|error| error.to_string())?;
    let surface = surface(world, files)?;
    world.spawn(
        Terrain::generated(Vec3::new(-half, 0.0, -half), pyramid, layers).with_surface(surface),
    );
    Ok((
        (
            Vec3::new(-half, -50.0, -half),
            Vec3::new(half, 1600.0, half),
        ),
        level,
    ))
}

/// The highlands' surface: its rules over [`LOOKS`], each material textured
/// by its maps when `files` holds them.
fn surface(
    world: &mut World,
    files: &HashMap<String, StressFile>,
) -> Result<TerrainSurface, String> {
    let mut assets = world.resource_mut::<AssetLibrary>();
    let mut materials = Vec::with_capacity(LOOKS.len());
    for look in &LOOKS {
        materials.push(SurfaceMaterial {
            material: maps(&mut assets, files, look)?,
            color: look.color,
            roughness: look.roughness,
            tiling: look.tiling,
            height_contrast: look.height_contrast,
            projection: look.projection,
            cover: look.cover,
        });
    }
    // Boulders of a few shapes, in the rock's maps where the page supplied
    // them: lighter than the cliffs' and gentler in relief, as weathered
    // stones are.
    let base = Material::new([0.8, 0.77, 0.72, 1.0])
        .and_then(|material| {
            material.with_surface(Surface {
                roughness: 0.85,
                ..Surface::MATTE
            })
        })
        .map_err(|error| error.to_string())?;
    let error = |error: AssetError| error.to_string();
    let material = match materials[4]
        .material
        .and_then(|rock| assets.material(rock).copied())
    {
        Some(rock) => {
            let mut material = base;
            if let Some(map) = rock.base_color_texture() {
                material = material.with_base_color_texture(map).map_err(error)?;
            }
            if let Some(map) = rock.normal_texture() {
                material = material.with_normal_texture(map, 0.5).map_err(error)?;
            }
            if let Some(map) = rock.metallic_roughness_texture() {
                material = material
                    .with_metallic_roughness_texture(map)
                    .and_then(|material| material.with_occlusion_texture(map, 1.0))
                    .map_err(error)?;
            }
            material
        }
        None => Material::new([0.3, 0.28, 0.26, 1.0])
            .and_then(|material| material.with_surface(base.surface()))
            .map_err(error)?,
    };
    let rock = assets
        .insert_material(material)
        .map_err(|error| error.to_string())?;
    let mut placed = Vec::with_capacity(BOULDERS + SPRUCES);
    for seed in 0..BOULDERS as u32 {
        let mesh = assets
            .insert_mesh(crate::rocks::boulder(seed)?)
            .map_err(|error| error.to_string())?;
        placed.push(PlacementAsset {
            mesh,
            material: rock,
            lods: Vec::new(),
        });
    }
    // Spruces of a few shapes over one bark and needle atlas, their needles
    // cut out by its alpha.
    let atlas = Texture::with_alpha(
        crate::trees::ATLAS,
        crate::trees::ATLAS,
        crate::trees::atlas(),
    )
    .and_then(|texture| assets.insert_texture(texture))
    .map_err(error)?;
    let needles = Material::new([1.0, 1.0, 1.0, 1.0])
        .and_then(|material| {
            material.with_surface(Surface {
                roughness: 0.9,
                ..Surface::MATTE
            })
        })
        .and_then(|material| material.with_base_color_texture(TextureRef::new(atlas)))
        .and_then(|material| material.with_alpha_mode(AlphaMode::Mask { cutoff: 0.5 }))
        .and_then(|material| assets.insert_material(material))
        .map_err(error)?;
    for seed in 0..SPRUCES as u32 {
        let mut mesh = |detail| {
            crate::trees::conifer(seed, detail)
                .and_then(|mesh| assets.insert_mesh(mesh).map_err(error))
        };
        let full = mesh(0)?;
        let mut lods = Vec::with_capacity(crate::trees::CONIFER_ERRORS.len());
        for (detail, error) in crate::trees::CONIFER_ERRORS.into_iter().enumerate() {
            lods.push(PlacementLod {
                mesh: mesh(detail + 1)?,
                error,
            });
        }
        placed.push(PlacementAsset {
            mesh: full,
            material: needles,
            lods,
        });
    }
    TerrainSurface::new(&rules(), materials)
        .and_then(|surface| surface.with_assets(placed))
        .map_err(|error| error.to_string())
}

/// Boulder and spruce shapes the placement layers choose from.
const BOULDERS: usize = 5;
const SPRUCES: usize = 3;

/// `look`'s maps from `files` as an engine material tinted by its tint,
/// when the page supplied all three: base color, normal, and occlusion
/// (red), roughness (green), metalness (blue) and height (alpha) in one.
fn maps(
    assets: &mut AssetLibrary,
    files: &HashMap<String, StressFile>,
    look: &Look,
) -> Result<Option<MaterialHandle>, String> {
    let mut textures = Vec::with_capacity(3);
    for map in ["albedo", "normal", "orm"] {
        let file_name = format!("{}_{map}.ktx2", look.name);
        let Some(file) = files.get(&file_name) else {
            return Ok(None);
        };
        let Some([width, height]) = file.size else {
            return Ok(None);
        };
        let texture = Texture::with_alpha(width, height, file.bytes.clone())
            .map_err(|error| format!("{file_name}: {error}"))?;
        textures.push(
            assets
                .insert_texture(texture)
                .map_err(|error| error.to_string())?,
        );
    }
    let [red, green, blue] = look.tint;
    let material = Material::new([red, green, blue, 1.0])
        .and_then(|material| {
            material.with_surface(Surface {
                roughness: 1.0,
                ..Surface::MATTE
            })
        })
        .and_then(|material| material.with_base_color_texture(TextureRef::new(textures[0])))
        .and_then(|material| material.with_normal_texture(TextureRef::new(textures[1]), 1.0))
        .and_then(|material| material.with_metallic_roughness_texture(TextureRef::new(textures[2])))
        .and_then(|material| material.with_occlusion_texture(TextureRef::new(textures[2]), 1.0))
        .map_err(|error| error.to_string())?;
    assets
        .insert_material(material)
        .map(Some)
        .map_err(|error| error.to_string())
}

/// Where each material lies: meadow everywhere, dry grass in broad patches
/// with ragged edges, dirt where slopes thin the grass, scree on steep
/// ground below the cliffs, rock on the cliffs and snow on the peaks'
/// gentler slopes.
fn rules() -> Biomes {
    let noise = |wavelength: f32, seed: u32, octaves: u32| {
        Mask::Noise(NoiseMask {
            wavelength,
            seed,
            octaves,
        })
    };
    let scaled = |mask: Mask, by: f32| Mask::Multiply(vec![mask, Mask::Constant(by)]);
    let remap = |mask: Mask, from: f32, to: f32| {
        Mask::Remap(Remap {
            mask: Box::new(mask),
            from,
            to,
        })
    };
    let layer = |material: &str, mask: Mask| BiomeLayer {
        material: material.into(),
        mask,
        opacity: 1.0,
    };
    let boulders = |weight: f32| {
        (0..BOULDERS)
            .map(|index| PlacementChoice {
                asset: format!("boulder{index}"),
                weight,
            })
            .collect::<Vec<_>>()
    };
    Biomes {
        assets: (0..BOULDERS)
            .map(|index| format!("boulder{index}"))
            .chain((0..SPRUCES).map(|index| format!("spruce{index}")))
            .collect(),
        // Boulders in clumps over the meadows, stones strewn on scree,
        // outcrops on rock, and spruce forests in broad stands below the
        // tree line on gentle ground, thinning at their edges: placed on
        // kilometer tiles, which stay resident farther away.
        placements: vec![
            PlacementLayer {
                id: "spruces".into(),
                assets: (0..SPRUCES)
                    .map(|index| PlacementChoice {
                        asset: format!("spruce{index}"),
                        weight: 1.0,
                    })
                    .collect(),
                density: 420.0,
                footprint: 3.5,
                materials: vec!["grass".into(), "dry_grass".into(), "dirt".into()],
                mask: Mask::Multiply(vec![
                    remap(
                        Mask::Add(vec![
                            scaled(noise(320.0, 31, 3), 0.75),
                            scaled(noise(45.0, 32, 2), 0.25),
                        ]),
                        0.45,
                        0.62,
                    ),
                    Mask::Invert(Box::new(Mask::Height([560.0, 640.0]))),
                    Mask::Invert(Box::new(Mask::Slope([24.0, 34.0]))),
                ]),
                scale: [0.65, 1.15],
                align: 0.04,
                sink: 0.3,
                level: 2,
            },
            PlacementLayer {
                id: "boulders".into(),
                assets: boulders(1.0),
                density: 18.0,
                footprint: 7.0,
                materials: vec!["grass".into(), "dry_grass".into(), "dirt".into()],
                mask: remap(noise(90.0, 21, 2), 0.45, 0.75),
                scale: [0.6, 2.4],
                align: 0.3,
                sink: 0.2,
                level: 0,
            },
            PlacementLayer {
                id: "stones".into(),
                assets: boulders(1.0),
                density: 450.0,
                footprint: 1.5,
                materials: vec!["scree".into()],
                mask: Mask::one(),
                scale: [0.12, 0.45],
                align: 0.8,
                sink: 0.08,
                level: 0,
            },
            PlacementLayer {
                id: "outcrops".into(),
                assets: boulders(1.0),
                density: 25.0,
                footprint: 8.0,
                materials: vec!["rock".into()],
                mask: Mask::one(),
                scale: [2.0, 5.0],
                align: 0.6,
                sink: 0.5,
                level: 0,
            },
        ],
        materials: LOOKS.iter().map(|look| look.name.to_owned()).collect(),
        biomes: vec![Biome {
            id: "highlands".into(),
            fitness: Mask::one(),
            layers: vec![
                layer("grass", Mask::one()),
                layer(
                    "dry_grass",
                    remap(
                        Mask::Add(vec![
                            scaled(noise(160.0, 11, 3), 0.6),
                            scaled(noise(28.0, 12, 2), 0.25),
                            scaled(noise(5.0, 16, 1), 0.15),
                        ]),
                        0.48,
                        0.72,
                    ),
                ),
                layer(
                    "dirt",
                    remap(
                        Mask::Add(vec![
                            scaled(noise(60.0, 13, 2), 0.45),
                            scaled(noise(9.0, 14, 2), 0.25),
                            scaled(Mask::Slope([18.0, 34.0]), 0.5),
                        ]),
                        0.66,
                        0.8,
                    ),
                ),
                layer(
                    "scree",
                    remap(
                        Mask::Add(vec![
                            scaled(Mask::Slope([27.0, 38.0]), 0.8),
                            scaled(noise(40.0, 15, 2), 0.4),
                            scaled(noise(8.0, 17, 1), 0.15),
                        ]),
                        0.62,
                        0.8,
                    ),
                ),
                layer(
                    "rock",
                    remap(
                        Mask::Add(vec![
                            Mask::Slope([33.0, 45.0]),
                            scaled(noise(70.0, 1, 2), 0.3),
                        ]),
                        0.55,
                        0.75,
                    ),
                ),
                layer(
                    "snow",
                    Mask::Multiply(vec![
                        remap(
                            Mask::Add(vec![
                                Mask::Height([700.0, 900.0]),
                                scaled(noise(60.0, 3, 2), 0.4),
                                Mask::Constant(-0.2),
                            ]),
                            0.4,
                            0.6,
                        ),
                        Mask::Invert(Box::new(Mask::Slope([35.0, 50.0]))),
                    ]),
                ),
            ],
        }],
    }
}

/// The plateau's radius for the hamlet of `config`'s renderables, the tower
/// field's grid with a margin.
pub(crate) fn hamlet_radius(config: &StressConfig) -> f32 {
    let side = (config.renderables as f32).sqrt().ceil() * crate::scene::SPACING;
    side * 0.75 + 30.0
}

/// The ground's world height under world `(x, z)` of `world`'s terrain,
/// from its layers wherever tiles are resident or not.
fn ground(world: &World, x: f32, z: f32) -> f32 {
    let mut terrains = world
        .try_query::<&Terrain>()
        .expect("terrain is registered");
    terrains
        .iter(world)
        .find_map(|terrain| match &terrain.source {
            TerrainSource::Generated(layers) => Some(
                terrain.origin.y
                    + layers.height_at(
                        &terrain.pyramid,
                        Vec2::new(x - terrain.origin.x, z - terrain.origin.z),
                    ),
            ),
            TerrainSource::File { .. } => None,
        })
        .unwrap_or_else(|| world.resource::<Plateau>().0)
}

/// Horizontal unit directions around the plateau, every 45 degrees.
fn directions() -> impl Iterator<Item = Vec2> {
    (0..8).map(|step| {
        let (sin, cos) = math::sin_cos(step as f32 * std::f32::consts::FRAC_PI_4);
        Vec2::new(cos, sin)
    })
}

/// The direction from the plateau whose ground `reach` meters out is
/// lowest: the widest view over the valleys.
fn lowest(world: &World, reach: f32) -> Vec2 {
    directions()
        .map(|direction| {
            let point = direction * reach;
            (ground(world, point.x, point.y), direction)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map_or(Vec2::Y, |(_, direction)| direction)
}

/// A camera `distance` meters from `target` along horizontal `direction`,
/// `rise` meters up per meter out, raised until its line of sight clears
/// the ground by 30 meters.
fn clear_view(world: &World, target: Vec3, direction: Vec2, distance: f32, rise: f32) -> Vec3 {
    let mut height = distance * rise;
    for step in 1..=24 {
        let along = step as f32 / 24.0;
        let point = Vec2::new(target.x, target.z) + direction * distance * along;
        let needed = (ground(world, point.x, point.y) + 30.0 - target.y) / along;
        height = math::max(height, needed);
    }
    let point = Vec2::new(target.x, target.z) + direction * distance;
    Vec3::new(point.x, target.y + height, point.y)
}

/// The overview: high over the plateau from its lowest side, looking back
/// across the hamlet into the mountains.
pub(crate) fn overview(world: &World, config: &StressConfig) -> (Vec3, Vec3) {
    let target = Vec3::new(0.0, world.resource::<Plateau>().0, 0.0);
    let direction = lowest(world, 1200.0);
    let distance = hamlet_radius(config) + 700.0;
    (clear_view(world, target, direction, distance, 0.45), target)
}

/// Inside the plateau's edge, looking out over the lowest valley beyond it.
pub(crate) fn close_view(world: &World, config: &StressConfig) -> (Vec3, Vec3) {
    let direction = lowest(world, 1500.0);
    let edge = direction * hamlet_radius(config) * 0.85;
    let far = direction * 1500.0;
    (
        Vec3::new(edge.x, ground(world, edge.x, edge.y) + 10.0, edge.y),
        Vec3::new(far.x, ground(world, far.x, far.y) + 60.0, far.y),
    )
}

/// From high above the plateau across the whole terrain to the north.
pub(crate) fn away_view(world: &World) -> (Vec3, Vec3) {
    let height = ground(world, 0.0, 0.0) + 700.0;
    (
        Vec3::new(0.0, height, 600.0),
        Vec3::new(0.0, height - 500.0, -3000.0),
    )
}
