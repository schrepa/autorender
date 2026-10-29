mod bodies;
mod caves;
mod ground;
mod hiker;
mod rocks;
mod trees;

use engine::collections::HashMap;
use engine::host::Image;

use engine::bricks::parse_value_at;
use engine::prelude::*;
use engine::scene::SceneTerrain;
use engine::{Error, ErrorCode, Expected};
use serde_json::Value;

use crate::records;

pub(crate) const SEED: u32 = 1337;

pub type RockMaps = HashMap<String, Image>;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) const ROCK_FILES: [&str; 6] = [
    "boulder_albedo.ktx2",
    "boulder_normal.ktx2",
    "boulder_orm.ktx2",
    "cliff_albedo.ktx2",
    "cliff_normal.ktx2",
    "cliff_orm.ktx2",
];

const BOULDER_SCAN_TILING: f32 = 1.8;
const WALL_SCAN_TILING: f32 = 2.5;

const ROCK_GROUND: &str = "rock";

const GROUNDS: [&str; 9] = [
    "grass",
    "dry_grass",
    "dirt",
    ROCK_GROUND,
    "snow",
    "sand",
    "gravel",
    "cobble",
    "scree",
];

const GROUND_MAPS: [(&str, TextureUsage); 3] = [
    ("albedo", TextureUsage::Color),
    ("normal", TextureUsage::Normal),
    ("orm", TextureUsage::Data),
];

const ROCK_TINT: [f32; 3] = [0.74, 0.7, 0.64];

const ROAD: &str = "road";

const CAVE_MOUTH: &str = "cave_mouth";

const MOSS: TopLayer = TopLayer {
    color: [0.055, 0.07, 0.03],
    coverage: 0.36,
    roughness: 0.9,
};
const LICHEN: TopLayer = TopLayer {
    color: [0.11, 0.12, 0.05],
    coverage: 0.34,
    roughness: 0.85,
};

const SOIL: Grounding = Grounding { height: 0.12 };

const BOULDERS: usize = 5;
const SPRUCES: usize = 3;

pub(crate) fn provide(world: &mut World, files: &RockMaps) -> engine::Result<()> {
    let mut provided = ProvidedAssets::default();
    let boulders = (0..BOULDERS as u32)
        .map(|seed| rocks::boulder_record(world, seed))
        .collect::<engine::Result<Vec<_>>>()?;
    for (seed, &mesh) in boulders.iter().enumerate() {
        provided.provide_mesh(format!("boulder{seed}"), mesh);
    }
    provided.provide_texture("bark_atlas", trees::atlas_record(world)?);
    for seed in 0..SPRUCES as u32 {
        let (levels, shadow) = trees::spruce(world, seed)?;
        for (detail, mesh) in levels.into_iter().enumerate() {
            let name = match detail {
                0 => format!("spruce{seed}"),
                _ => format!("spruce{seed}_lod{detail}"),
            };
            provided.provide_mesh(name, mesh);
        }
        provided.provide_mesh(format!("spruce{seed}_shadow"), shadow);
    }
    let grounds = world.resource_scope(|world, mut records: Mut<Records>| {
        let mut assets = world.resource_mut::<AssetLibrary>();
        GROUNDS
            .iter()
            .map(|&name| ground_textures((&mut records, &mut assets), name))
            .collect::<engine::Result<Vec<_>>>()
    })?;
    let mut rock = Vec::new();
    for (&name, textures) in GROUNDS.iter().zip(grounds) {
        for (&(map, _), &texture) in GROUND_MAPS.iter().zip(&textures) {
            provided.provide_texture(format!("{name}_{map}"), texture);
        }
        if name == ROCK_GROUND {
            rock = textures;
        }
    }
    let mut assets = world.resource_mut::<AssetLibrary>();
    let material = match scanned(&mut assets, files, "boulder", BOULDER_SCAN_TILING, MOSS)? {
        Some(scan) => scan,
        None => in_terrain_rock(&rock, ([0.8, 0.77, 0.72], 0.85), 3.0, MOSS)?,
    };
    let boulder_rock = material
        .with_grounding(SOIL)
        .and_then(|material| assets.insert_material(material))?;
    provided.provide_material("boulder_rock", boulder_rock);
    let wall_rock = match scanned(&mut assets, files, "cliff", WALL_SCAN_TILING, LICHEN)? {
        Some(scan) => assets.insert_material(scan)?,
        None => {
            let material = in_terrain_rock(&rock, (ROCK_TINT, 0.9), 4.0, LICHEN)?;
            assets.insert_material(material)?
        }
    };
    let wall = assets
        .material(wall_rock)
        .copied()
        .map(|material| material.with_double_sided(true))
        .ok_or_else(|| {
            Error::new(ErrorCode::GameRule, "$.cave.wall").expected(Expected::Present)
        })?;
    let wall = assets.insert_material(wall)?;
    world.insert_resource(CaveRock {
        boulders,
        material: wall_rock,
        wall,
    });
    world.insert_resource(provided);
    Ok(())
}

pub(crate) fn populate(
    world: &mut World,
    document: &SceneDocument,
    bodies: u32,
) -> engine::Result<()> {
    let (path, terrain) = scene_terrain(document)?;
    let origin = Vec2::new(terrain.origin[0], terrain.origin[2]);
    let (at, road) = scene_layer(terrain, &path, ROAD)?;
    let road = parse_value_at::<Vec<[f32; 2]>>(road["points"].clone(), &format!("{at}.points"))?;
    let (at, mouth) = scene_layer(terrain, &path, CAVE_MOUTH)?;
    let center = parse_value_at::<[f32; 2]>(mouth["center"].clone(), &format!("{at}.center"))?;
    let radius = parse_value_at::<f32>(mouth["radius"].clone(), &format!("{at}.radius"))?;
    world.insert_resource(CaveMouth {
        center: Vec2::from(center) + origin,
        radius,
    });
    hiker::hiker(world, &road, origin)?;
    bodies::request(world, bodies)?;
    Ok(())
}

fn scene_terrain(document: &SceneDocument) -> engine::Result<(String, &SceneTerrain)> {
    document
        .entities
        .iter()
        .enumerate()
        .find_map(|(index, entity)| {
            let path = format!("$.entities[{index}].terrain");
            entity.terrain.as_ref().map(|terrain| (path, terrain))
        })
        .ok_or_else(|| Error::new(ErrorCode::GameRule, "$.entities").expected(Expected::Present))
}

fn scene_layer<'a>(
    terrain: &'a SceneTerrain,
    path: &str,
    id: &str,
) -> engine::Result<(String, &'a Value)> {
    terrain
        .layers
        .iter()
        .enumerate()
        .find(|(_, layer)| layer["id"] == id)
        .map(|(index, layer)| (format!("{path}.layers[{index}]"), layer))
        .ok_or_else(|| {
            Error::new(ErrorCode::GameRule, format!("{path}.layers"))
                .expected(Expected::Present)
                .got(id)
        })
}

fn ground_textures(
    (records, assets): (&mut Records, &mut AssetLibrary),
    name: &'static str,
) -> engine::Result<Vec<TextureHandle>> {
    records::textures(
        (records, assets),
        format!("highlands/{name}"),
        &GROUND_MAPS,
        move || ground::sources(name).map_err(|error| error.to_string()),
    )
}

#[derive(Resource, Clone, Copy)]
struct CaveMouth {
    center: Vec2,
    radius: f32,
}

#[derive(Resource, Clone)]
struct CaveRock {
    boulders: Vec<MeshHandle>,
    material: MaterialHandle,
    wall: MaterialHandle,
}

#[derive(Resource)]
struct CaveOpened;

const CAVE_RADIUS: [f32; 2] = [6.0, 4.5];

const CAVE_MARGIN: f32 = 1.5;

pub(crate) fn add_systems(app: &mut App) {
    app.add_systems(Presentation, (open_cave, bodies::place))
        .add_systems(FixedSimulation, hiker::hike.after(NavSystems::Walk));
}

fn open_cave(world: &mut World) {
    if world.contains_resource::<CaveOpened>() {
        return;
    }
    let (
        Some(CaveMouth {
            center: mouth,
            radius,
        }),
        Some(rock),
    ) = (
        world.get_resource::<CaveMouth>().copied(),
        world.get_resource::<CaveRock>().cloned(),
    )
    else {
        return;
    };
    let Some((terrain, layers, bake)) = ({
        let mut query = world.query::<(Entity, &Terrain)>();
        let terrains = world.resource::<Terrains>();
        query.iter(world).find_map(|(entity, terrain)| {
            Some((
                terrain.clone(),
                terrains.layers(entity)?,
                terrains.bake_record(entity)?,
            ))
        })
    }) else {
        return;
    };
    if layers.hydrology().is_none() {
        return;
    }
    let height = |point: Vec2| {
        terrain.origin.y
            + layers.height_at(
                &terrain.pyramid,
                Vec2::new(point.x - terrain.origin.x, point.y - terrain.origin.z),
            )
    };
    let gradient = Vec2::new(
        height(mouth + Vec2::X * 20.0) - height(mouth - Vec2::X * 20.0),
        height(mouth + Vec2::Y * 20.0) - height(mouth - Vec2::Y * 20.0),
    ) / 40.0;
    let slope = gradient.length();
    let uphill = if slope > 0.02 {
        gradient / slope
    } else {
        Vec2::X
    };
    let side = Vec2::new(-uphill.y, uphill.x);
    let start = mouth - uphill * (radius + 1.0);
    let center = height(start) - 0.5 + 0.55 * CAVE_RADIUS[1] * 1.15;
    let fall = math::max(0.75 - slope, 0.25);
    let at = |along: f32, across: f32, drop: f32| {
        let point = mouth + uphill * along + side * across;
        Vec3::new(point.x, center - drop, point.y)
    };
    let points = [
        at(-(radius + 1.0), 0.0, 0.0),
        at(12.0, 0.0, fall * 18.0),
        at(28.0, 6.0, fall * 18.0 + fall * 8.0),
        at(44.0, -5.0, fall * 18.0 + fall * 16.0),
        at(58.0, 0.0, fall * 18.0 + fall * 22.0),
    ];
    let facing = Quat::from_rotation_y((-uphill.x).atan2(-uphill.y));
    let framing: Vec<(Transform, MeshHandle)> = [
        (4.0, 0.0, 1.5, 10.0, 0),
        (-1.0, 10.0, -1.5, 7.0, 1),
        (-1.0, -10.0, -1.5, 7.0, 2),
    ]
    .into_iter()
    .filter_map(
        |(along, across, rise, scale, index): (f32, f32, f32, f32, usize)| {
            let point = mouth + uphill * along + side * across;
            let mesh = *rock.boulders.get(index % rock.boulders.len().max(1))?;
            Some((
                Transform {
                    translation: Vec3::new(point.x, height(point) + rise, point.y),
                    rotation: facing,
                    scale: Vec3::splat(scale),
                },
                mesh,
            ))
        },
    )
    .collect();
    let inward = Vec3::new(uphill.x, 0.0, uphill.y);
    let mut identity = bake.hash.to_vec();
    identity.extend_from_slice(&mouth.x.to_bits().to_le_bytes());
    identity.extend_from_slice(&mouth.y.to_bits().to_le_bytes());
    world.insert_resource(CaveOpened);
    let spawn = move |world: &mut World, mesh: MeshHandle, built: &Mesh| {
        spawn_cave(world, (mesh, built), &points, inward, (&rock, framing));
        Ok(())
    };
    let made = records::mesh_then(
        world,
        "highlands_cave",
        &identity,
        move || caves::cave(&points, CAVE_RADIUS, 79),
        spawn,
    );
    if let Err(error) = made {
        report_failure("highlands_cave_failed", &error);
    }
}

fn report_failure(event: &str, error: &engine::Error) {
    let record = serde_json::json!({
        "event": event,
        "error": error,
    });
    eprintln!("{record}");
}

fn spawn_cave(
    world: &mut World,
    (mesh, built): (MeshHandle, &Mesh),
    points: &[Vec3; 5],
    inward: Vec3,
    (rock, framing): (&CaveRock, Vec<(Transform, MeshHandle)>),
) {
    let (low, high) = cave_interior(built, points, inward);
    world.spawn((
        Transform {
            translation: points[0],
            ..Transform::default()
        },
        Mesh3d {
            mesh,
            material: rock.wall,
        },
    ));
    let chamber = points[4] - inward * 6.0 + Vec3::Y * 2.0;
    world.spawn((
        Transform {
            translation: chamber,
            ..Transform::default()
        },
        LocalLight::point(Vec3::new(1.0, 0.62, 0.3), 40.0, 26.0).with_region(LightRegion::Interior),
    ));
    world.spawn((
        Transform {
            translation: (low + high) * 0.5,
            ..Transform::default()
        },
        ReflectionProbe::new((high - low) * 0.5)
            .with_blend(CAVE_MARGIN)
            .with_interior(Interior { sky: 1.0, sun: 1.0 }),
    ));
    for (transform, mesh) in framing {
        world.spawn((
            transform,
            Mesh3d {
                mesh,
                material: rock.material,
            },
        ));
    }
}

fn cave_interior(mesh: &Mesh, points: &[Vec3], inward: Vec3) -> (Vec3, Vec3) {
    let bend = (points[1] - points[0]).dot(inward);
    let (low, high) = mesh
        .vertices()
        .iter()
        .map(|vertex| points[0] + Vec3::from(vertex.position))
        .filter(|&point| (point - points[0]).dot(inward) >= bend)
        .fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(low, high), point| (low.min(point), high.max(point)),
        );
    (
        low - Vec3::splat(CAVE_MARGIN),
        high + Vec3::splat(CAVE_MARGIN),
    )
}

fn scanned(
    assets: &mut AssetLibrary,
    files: &RockMaps,
    name: &str,
    tiling: f32,
    top: TopLayer,
) -> engine::Result<Option<Material>> {
    let mut textures = Vec::with_capacity(3);
    for map in ["albedo", "normal", "orm"] {
        let file_name = format!("rocks/{name}_{map}.ktx2");
        let Some(file) = files.get(&file_name) else {
            return Ok(None);
        };
        let texture = Texture::from_rgba(file.width, file.height, file.pixels.clone())
            .map_err(|error| error.within(&format!("$.rocks.{name}.{map}")))?;
        textures.push(assets.insert_texture(texture)?);
    }
    Material::new([1.0; 4])
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
        .and_then(|material| material.with_world_mapping(tiling))
        .and_then(|material| material.with_top_layer(top))
        .map(Some)
}

fn in_terrain_rock(
    textures: &[TextureHandle],
    (color, roughness): ([f32; 3], f32),
    tiling: f32,
    top: TopLayer,
) -> engine::Result<Material> {
    let [red, green, blue] = color;
    Material::new([red, green, blue, 1.0])
        .and_then(|material| {
            material.with_surface(Surface {
                roughness,
                ..Surface::MATTE
            })
        })
        .and_then(|material| material.with_base_color_texture(TextureRef::new(textures[0])))
        .and_then(|material| material.with_normal_texture(TextureRef::new(textures[1]), 0.8))
        .and_then(|material| material.with_metallic_roughness_texture(TextureRef::new(textures[2])))
        .and_then(|material| material.with_occlusion_texture(TextureRef::new(textures[2]), 1.0))
        .and_then(|material| material.with_world_mapping(tiling))
        .and_then(|material| material.with_top_layer(top))
}
