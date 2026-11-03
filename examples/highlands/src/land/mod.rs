mod ground;

use engine::collections::HashMap;
use engine::host::Image;

use engine::bricks::parse_value_at;
use engine::prelude::*;
use engine::{Error, ErrorCode, Expected};
use engine_core::RecordExchange;
use serde_json::{Value, json};

use crate::records;

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

const CAVE_MOUTH: &str = "cave_mouth";

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

pub(crate) fn provide(world: &mut World, files: &RockMaps) -> engine::Result<()> {
    world.insert_resource(ProvidedAssets::default());
    control(world, json!("land"))?;
    let grounds = world.resource_scope(|world, mut records: Mut<RecordExchange>| {
        let mut assets = world.resource_mut::<AssetLibrary>();
        GROUNDS
            .iter()
            .map(|&name| ground_textures((&mut records, &mut assets), name))
            .collect::<engine::Result<Vec<_>>>()
    })?;
    let mut provided = world.resource::<ProvidedAssets>().clone();
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
    let (at, mouth) = scene_layer(terrain, &path, CAVE_MOUTH)?;
    let center = parse_value_at::<[f32; 2]>(mouth["center"].clone(), &format!("{at}.center"))?;
    let radius = parse_value_at::<f32>(mouth["radius"].clone(), &format!("{at}.radius"))?;
    let mouth = Vec2::from(center) + origin;
    control(
        world,
        json!({"cave": {
            "mouth": [mouth.x, mouth.y],
            "radius": radius,
            "rock": WALL_ROCK,
            "wall": CAVE_WALL,
        }}),
    )?;
    control(world, json!({"bodies": {"count": bodies}}))
}

pub(crate) fn control(world: &mut World, request: Value) -> engine::Result<()> {
    engine::host::run_operation(world, "highlands", request).map(drop)
}

fn scene_terrain(document: &SceneDocument) -> engine::Result<(String, Value)> {
    document
        .entities
        .iter()
        .enumerate()
        .find_map(|(index, entity)| {
            let path = format!("$.entities[{index}].terrain");
            entity
                .component("terrain")
                .map(|terrain| (path, terrain.clone()))
        })
        .ok_or_else(|| Error::new(ErrorCode::GameRule, "$.entities").expected(Expected::Present))
}

fn scene_layer<'a>(
    terrain: &'a Value,
    path: &str,
    id: &str,
) -> engine::Result<(String, &'a Value)> {
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
