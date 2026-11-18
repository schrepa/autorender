use std::ops::Range;
use std::sync::Arc;

use engine::math;
use engine::prelude::*;
use engine::{Error, ErrorCode, Expected};
use engine_core::SceneCx;
use glam::{Mat3, Quat, Vec3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::play::world::{Markers, Spawn, Zone, ZoneBox};
use crate::solids::{Block, Solids, Surface};

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Floor {
    pub half_size_m: f32,
    pub surface: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocks: Vec<BlockForm>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BlockForm {
    pub centre: [f32; 3],
    pub half_extents: [f32; 3],
    pub surface: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Cooked {
    pub collision: String,
    pub ranges: String,
    pub markers: String,
}

#[derive(Component, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArenaFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub floor: Option<Floor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooked: Option<Cooked>,
    #[serde(skip)]
    pub(super) built: Option<Arc<(Solids, Markers)>>,
}

#[derive(Deserialize)]
struct Ranges {
    meshes: Vec<RangeEntry>,
}

#[derive(Deserialize)]
struct RangeEntry {
    triangles: [u32; 2],
    #[serde(default)]
    extras: Value,
    #[serde(default)]
    mesh_extras: Value,
}

#[derive(Deserialize)]
struct MarkerFile {
    markers: Vec<Marker>,
}

#[derive(Deserialize)]
struct Marker {
    node: String,
    position: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

fn surface_named(name: &str, path: &str) -> Result<Surface, Error> {
    serde_json::from_value(Value::String(name.to_owned())).map_err(|_| {
        Error::new(ErrorCode::InvalidValue, path.to_owned())
            .got(name)
            .variants(["stone", "metal", "wood", "dirt"])
    })
}

fn json<T: serde::de::DeserializeOwned>(bytes: &[u8], path: &str) -> Result<T, Error> {
    let text = std::str::from_utf8(bytes).map_err(|_| {
        Error::new(ErrorCode::InvalidRecord, path.to_owned()).expected(Expected::Utf8)
    })?;
    engine::bricks::parse_json(text).map_err(|error| error.within(path))
}

fn floor(floor: &Floor, path: &str) -> Result<(Solids, Markers), Error> {
    let surface = surface_named(&floor.surface, &format!("{path}.floor.surface"))?;
    let h = floor.half_size_m;
    let positions = [
        Vec3::new(-h, 0.0, -h),
        Vec3::new(-h, 0.0, h),
        Vec3::new(h, 0.0, h),
        Vec3::new(h, 0.0, -h),
    ];
    let indices = [0, 1, 2, 0, 2, 3];
    let mut solids = Solids::build(&positions, &indices, &[(0..2, surface)]);
    for (index, block) in floor.blocks.iter().enumerate() {
        let at = format!("{path}.floor.blocks[{index}].surface");
        solids.add_block(Block {
            id: u32::MAX - index as u32,
            centre: Vec3::from_array(block.centre),
            half_extents: Vec3::from_array(block.half_extents),
            axes: Mat3::IDENTITY,
            surface: surface_named(&block.surface, &at)?,
        });
    }
    Ok((solids, Markers::default()))
}

fn labelled(entry: &RangeEntry) -> Option<&str> {
    [&entry.extras, &entry.mesh_extras]
        .into_iter()
        .find_map(|extras| extras.get("surface").and_then(Value::as_str))
}

fn surfaces(ranges: &Ranges, path: &str) -> Result<Vec<(Range<u32>, Surface)>, Error> {
    let mut found = Vec::new();
    for (index, entry) in ranges.meshes.iter().enumerate() {
        if let Some(name) = labelled(entry) {
            let at = format!("{path}.meshes[{index}].extras.surface");
            found.push((
                entry.triangles[0]..entry.triangles[1],
                surface_named(name, &at)?,
            ));
        }
    }
    Ok(found)
}

fn spawn(marker: &Marker) -> Spawn {
    let [x, y, z, w] = marker.rotation;
    let facing = Quat::from_xyzw(x, y, z, w) * Vec3::Z;
    Spawn {
        position: Vec3::from_array(marker.position),
        yaw_radians: math::atan2(-facing.x, -facing.z),
    }
}

fn numbered(node: &str, prefix: &str) -> Option<u32> {
    node.strip_prefix(prefix)?.parse().ok()
}

fn markers(file: &MarkerFile) -> Markers {
    let mut attack: Vec<(u32, Spawn)> = Vec::new();
    let mut defend: Vec<(u32, Spawn)> = Vec::new();
    let mut zones = Vec::new();
    let named = [
        ("zone_site_a", Zone::SiteA),
        ("zone_site_b", Zone::SiteB),
        ("zone_buy_attack", Zone::BuyAttack),
        ("zone_buy_defend", Zone::BuyDefend),
    ];
    for marker in &file.markers {
        if let Some(number) = numbered(&marker.node, "spawn_attack_") {
            attack.push((number, spawn(marker)));
        } else if let Some(number) = numbered(&marker.node, "spawn_defend_") {
            defend.push((number, spawn(marker)));
        } else if let Some(&(_, zone)) = named.iter().find(|(name, _)| *name == marker.node) {
            let [x, y, z, w] = marker.rotation;
            zones.push(ZoneBox {
                zone,
                centre: Vec3::from_array(marker.position),
                rotation: Quat::from_xyzw(x, y, z, w),
                half: Vec3::from_array(marker.scale).abs(),
            });
        }
    }
    attack.sort_by_key(|(number, _)| *number);
    defend.sort_by_key(|(number, _)| *number);
    Markers {
        attack: attack.into_iter().map(|(_, spawn)| spawn).collect(),
        defend: defend.into_iter().map(|(_, spawn)| spawn).collect(),
        zones,
    }
}

fn cooked(
    files: &Cooked,
    cx: &SceneCx<'_>,
    path: &str,
) -> Result<Option<(Solids, Markers)>, Error> {
    let at = |field: &str| format!("{path}.cooked.{field}");
    let (Some(mesh), Some(ranges), Some(marked)) = (
        cx.buffer(&files.collision, &at("collision"))?,
        cx.buffer(&files.ranges, &at("ranges"))?,
        cx.buffer(&files.markers, &at("markers"))?,
    ) else {
        return Ok(None);
    };
    let mesh = engine_assets::MeshBuffer::parse(&mesh, &at("collision"))?;
    let positions: Vec<Vec3> = mesh
        .vertices
        .iter()
        .map(|vertex| Vec3::from_array(vertex.position))
        .collect();
    let ranges: Ranges = json(&ranges, &at("ranges"))?;
    let labels = surfaces(&ranges, &at("ranges"))?;
    let marked: MarkerFile = json(&marked, &at("markers"))?;
    Ok(Some((
        Solids::build(&positions, &mesh.indices, &labels),
        markers(&marked),
    )))
}

impl ArenaFile {
    pub(super) fn checked(&self, path: &str) -> Result<(), Error> {
        match (&self.floor, &self.cooked) {
            (Some(_), None) | (None, Some(_)) => Ok(()),
            (Some(_), Some(_)) | (None, None) => {
                Err(Error::new(ErrorCode::InvalidValue, path.to_owned())
                    .variants(["floor", "cooked"]))
            }
        }
    }

    pub(super) fn files(&self) -> Vec<String> {
        self.cooked
            .iter()
            .flat_map(|files| {
                [
                    files.collision.clone(),
                    files.ranges.clone(),
                    files.markers.clone(),
                ]
            })
            .collect()
    }

    pub(super) fn prepared(mut self, cx: &SceneCx<'_>, path: &str) -> Result<Self, Error> {
        let built = match (&self.floor, &self.cooked) {
            (Some(plain), _) => Some(floor(plain, path)?),
            (None, Some(files)) => cooked(files, cx, path)?,
            (None, None) => None,
        };
        self.built = built.map(Arc::new);
        Ok(self)
    }
}
