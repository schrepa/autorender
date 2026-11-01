mod ground;
mod rocks;
mod trees;

use engine::collections::HashMap;
use engine::host::Image;

use engine::bricks::parse_value_at;
use engine::prelude::*;
use engine::{Error, ErrorCode, Expected};
use engine_core::RecordExchange;
use serde_json::{Value, json};

use crate::records;
use crate::scene::set;

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

const HIKER: &str = "hiker";
const BANK_BOULDER: &str = "highlands_bank_boulder";
const LOG: &str = "highlands_log";
const BARK: &str = "highlands_bark";
const PALE: &str = "highlands_pale";
const STONE: &str = "highlands_stone";
const WALL_ROCK: &str = "highlands_wall_rock";
const CAVE_WALL: &str = "highlands_cave_wall";

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
    let grounds = world.resource_scope(|world, mut records: Mut<RecordExchange>| {
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
    provided.provide_material(WALL_ROCK, wall_rock);
    provided.provide_material(CAVE_WALL, wall);
    world.insert_resource(provided);
    Ok(())
}

pub(crate) fn populate(
    world: &mut World,
    document: &SceneDocument,
    bodies: u32,
) -> engine::Result<()> {
    let (path, terrain) = scene_terrain(document)?;
    let terrain = &terrain;
    let origin = parse_value_at::<[f32; 3]>(terrain["origin"].clone(), &format!("{path}.origin"))?;
    let origin = Vec2::new(origin[0], origin[2]);
    let (at, road) = scene_layer(terrain, &path, ROAD)?;
    let road = parse_value_at::<Vec<[f32; 2]>>(road["points"].clone(), &format!("{at}.points"))?;
    let (at, mouth) = scene_layer(terrain, &path, CAVE_MOUTH)?;
    let center = parse_value_at::<[f32; 2]>(mouth["center"].clone(), &format!("{at}.center"))?;
    let radius = parse_value_at::<f32>(mouth["radius"].clone(), &format!("{at}.radius"))?;
    let mouth = Vec2::from(center) + origin;
    set(world, "highlands.cave", json!([mouth.x, mouth.y, radius]))?;
    hiker(world, &road, origin)?;
    request_bodies(world, bodies)?;
    Ok(())
}

fn hiker(world: &mut World, road: &[[f32; 2]], origin: Vec2) -> engine::Result<()> {
    let world_point = |[x, z]: [f32; 2]| [x + origin.x, z + origin.y];
    let start = world_point(road[0]);
    let far = road
        .iter()
        .map(|&point| world_point(point))
        .find(|point| Vec2::from(*point).distance(Vec2::from(start)) > 350.0)
        .unwrap_or(world_point(road[road.len() - 1]));
    let mut assets = world.resource_mut::<AssetLibrary>();
    let mesh = assets.insert_mesh(Mesh::cube())?;
    let material = assets.insert_material(Material::new([0.62, 0.1, 0.07, 1.0])?)?;
    let offset = 0.875_f32;
    let hiker = world
        .spawn((
            Transform {
                translation: Vec3::new(start[0], 0.0, start[1]),
                scale: Vec3::new(0.45, 1.75, 0.3),
                ..Transform::default()
            },
            Mesh3d { mesh, material },
            InterpolateTransform::default(),
        ))
        .id();
    attach(world, hiker, "grounded", json!({ "offset": offset }))?;
    attach(
        world,
        hiker,
        "nav_agent",
        json!({ "destination": far, "speed": 1.6_f32, "offset": offset }),
    )?;
    attach(
        world,
        hiker,
        HIKER,
        json!({ "ends": [start, far], "toward": 1, "rest": 0.0_f32 }),
    )
}

fn attach(world: &mut World, entity: Entity, key: &str, value: Value) -> engine::Result<()> {
    let kind = engine_core::scene_component(world, key).ok_or_else(|| {
        Error::new(ErrorCode::GameRule, format!("$.{key}"))
            .expected(Expected::Present)
            .got(key)
    })?;
    kind.read(world, entity, &value)
}

fn request_bodies(world: &mut World, count: u32) -> engine::Result<()> {
    if count == 0 {
        return Ok(());
    }
    let boulder = rocks::boulder_record(world, SEED ^ 0x5eed)?;
    let mut assets = world.resource_mut::<AssetLibrary>();
    let cube = assets.insert_mesh(Mesh::cube())?;
    let bark = assets.insert_material(Material::new([0.2, 0.13, 0.08, 1.0])?)?;
    let pale = assets.insert_material(Material::new([0.42, 0.3, 0.19, 1.0])?)?;
    let stone = assets.insert_material(Material::new([0.38, 0.37, 0.35, 1.0])?)?;
    world
        .resource_mut::<ProvidedAssets>()
        .provide_mesh(BANK_BOULDER, boulder)
        .provide_mesh(LOG, cube)
        .provide_material(BARK, bark)
        .provide_material(PALE, pale)
        .provide_material(STONE, stone);
    set(world, "highlands.bodies", json!(count))
}

fn scene_terrain(document: &SceneDocument) -> engine::Result<(String, Value)> {
    document
        .entities
        .iter()
        .enumerate()
        .find_map(|(index, entity)| {
            let path = format!("$.entities[{index}].terrain");
            entity.component("terrain").map(|terrain| (path, terrain.clone()))
        })
        .ok_or_else(|| Error::new(ErrorCode::GameRule, "$.entities").expected(Expected::Present))
}

fn scene_layer<'a>(terrain: &'a Value, path: &str, id: &str) -> engine::Result<(String, &'a Value)> {
    terrain["layers"]
        .as_array()
        .into_iter()
        .flatten()
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
    (records, assets): (&mut RecordExchange, &mut AssetLibrary),
    name: &'static str,
) -> engine::Result<Vec<TextureHandle>> {
    records::textures(
        (records, assets),
        format!("highlands/{name}"),
        &GROUND_MAPS,
        move || ground::sources(name).map_err(|error| error.to_string()),
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
