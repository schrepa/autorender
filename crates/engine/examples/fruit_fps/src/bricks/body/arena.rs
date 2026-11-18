use std::ops::Range;
use std::sync::Arc;

use engine::math;
use engine::prelude::*;
use engine::{Error, ErrorCode, Expected};
use engine_core::SceneCx;
use glam::{Mat3, Quat, Vec3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::world::{Markers, Spawn, Zone, ZoneBox};
use crate::solids::{Block, Ground, Kind, Solids, Surface};

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
    pub records: String,
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
    node: String,
    triangles: [u32; 2],
}

#[derive(Deserialize)]
struct Layout {
    collision: Vec<Record>,
}

#[derive(Deserialize)]
struct Record {
    name: String,
    #[serde(default)]
    surface: Option<String>,
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

const DEFAULT_KIND: &str = "stone";

fn ground(name: &str, kinds: &mut Vec<(Kind, String)>) -> Ground {
    let kind = Kind::named(name);
    if !kinds.iter().any(|(known, _)| *known == kind) {
        kinds.push((kind, name.to_owned()));
    }
    Ground {
        surface: Surface::of_kind(name),
        kind,
    }
}

fn json<T: serde::de::DeserializeOwned>(bytes: &[u8], path: &str) -> Result<T, Error> {
    let text = std::str::from_utf8(bytes).map_err(|_| {
        Error::new(ErrorCode::InvalidRecord, path.to_owned()).expected(Expected::Utf8)
    })?;
    engine::bricks::parse_json(text).map_err(|error| error.within(path))
}

fn floor(floor: &Floor) -> (Solids, Markers) {
    let mut kinds = Vec::new();
    let surface = ground(&floor.surface, &mut kinds);
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
        let made = ground(&block.surface, &mut kinds);
        solids.add_block(Block {
            id: u32::MAX - index as u32,
            centre: Vec3::from_array(block.centre),
            half_extents: Vec3::from_array(block.half_extents),
            axes: Mat3::IDENTITY,
            surface: made.surface,
            kind: made.kind,
        });
    }
    let markers = Markers {
        kinds,
        ..Markers::default()
    };
    (solids, markers)
}

fn grounds(
    ranges: &Ranges,
    layout: &Layout,
    kinds: &mut Vec<(Kind, String)>,
) -> Vec<(Range<u32>, Ground)> {
    ranges
        .meshes
        .iter()
        .map(|entry| {
            let name = (layout.collision.iter())
                .find(|record| record.name == entry.node)
                .and_then(|record| record.surface.as_deref())
                .unwrap_or(DEFAULT_KIND);
            (entry.triangles[0]..entry.triangles[1], ground(name, kinds))
        })
        .collect()
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
        kinds: Vec::new(),
    }
}

fn cooked(
    files: &Cooked,
    cx: &SceneCx<'_>,
    path: &str,
) -> Result<Option<(Solids, Markers)>, Error> {
    let at = |field: &str| format!("{path}.cooked.{field}");
    let (Some(mesh), Some(ranges), Some(marked), Some(records)) = (
        cx.buffer(&files.collision, &at("collision"))?,
        cx.buffer(&files.ranges, &at("ranges"))?,
        cx.buffer(&files.markers, &at("markers"))?,
        cx.buffer(&files.records, &at("records"))?,
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
    let layout: Layout = json(&records, &at("records"))?;
    let mut kinds = Vec::new();
    let labels = grounds(&ranges, &layout, &mut kinds);
    let marked: MarkerFile = json(&marked, &at("markers"))?;
    let markers = Markers {
        kinds,
        ..markers(&marked)
    };
    Ok(Some((
        Solids::build(&positions, &mesh.indices, &labels),
        markers,
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
                    files.records.clone(),
                ]
            })
            .collect()
    }

    pub(super) fn prepared(mut self, cx: &SceneCx<'_>, path: &str) -> Result<Self, Error> {
        let built = match (&self.floor, &self.cooked) {
            (Some(plain), _) => Some(floor(plain)),
            (None, Some(files)) => cooked(files, cx, path)?,
            (None, None) => None,
        };
        self.built = built.map(Arc::new);
        Ok(self)
    }
}
