use std::ops::Range;

use engine::Error;
use engine::math;
use engine::prelude::*;
use engine_core::{Asset, Assets};
use engine_solids::{Block, Kind, Solids};
use glam::{Mat3, Quat, Vec3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::files::Triangles;
use crate::play::layout::{Collider, Layout};
use crate::play::world::{Markers, STONE, Spawn, Zone, ZoneBox, floor_or_cooked};

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
    pub collision: Asset<Triangles>,
    pub ranges: Asset<CollisionRanges>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bullet_collision: Option<Asset<Triangles>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bullet_ranges: Option<Asset<CollisionRanges>>,
    pub markers: Asset<ArenaMarkers>,
    pub records: Asset<Layout>,
}

#[derive(Component, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArenaFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub floor: Option<Floor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooked: Option<Cooked>,
}

#[derive(Deserialize)]
pub struct CollisionRanges {
    meshes: Vec<RangeEntry>,
}

#[derive(Deserialize)]
struct RangeEntry {
    node: String,
    triangles: [u32; 2],
}

#[derive(Deserialize)]
pub struct ArenaMarkers {
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

pub(super) fn stone_unknown(
    solids: &mut Solids,
    bullets: Option<&mut Solids>,
    markers: &mut Markers,
    known: &[String],
) {
    let (kept, unknown): (Vec<_>, Vec<_>) = std::mem::take(&mut markers.kinds)
        .into_iter()
        .partition(|(_, name)| known.contains(name));
    let mut bullets = bullets;
    for (kind, _) in &unknown {
        solids.relabel(*kind, STONE);
        if let Some(bullets) = bullets.as_deref_mut() {
            bullets.relabel(*kind, STONE);
        }
    }
    markers.kinds = kept;
    if !unknown.is_empty() && !markers.kinds.iter().any(|(kind, _)| *kind == STONE) {
        markers.kinds.push((STONE, DEFAULT_KIND.to_owned()));
    }
    markers.unknown_kinds = unknown.into_iter().map(|(_, name)| name).collect();
}

fn ground(name: &str, kinds: &mut Vec<(Kind, String)>) -> Kind {
    let kind = Kind::named(name);
    if !kinds.iter().any(|(known, _)| *known == kind) {
        kinds.push((kind, name.to_owned()));
    }
    kind
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
        let kind = ground(&block.surface, &mut kinds);
        solids.add_block(Block {
            id: u32::MAX - index as u32,
            centre: Vec3::from_array(block.centre),
            half_extents: Vec3::from_array(block.half_extents),
            axes: Mat3::IDENTITY,
            kind,
        });
    }
    let markers = Markers {
        kinds,
        ..Markers::default()
    };
    (solids, markers)
}

fn grounds(
    ranges: &CollisionRanges,
    colliders: &[Collider],
    kinds: &mut Vec<(Kind, String)>,
) -> Vec<(Range<u32>, Kind)> {
    ranges
        .meshes
        .iter()
        .map(|entry| {
            let name = (colliders.iter())
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

fn markers(file: &ArenaMarkers) -> Markers {
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
        unknown_kinds: Vec::new(),
    }
}

fn bullet_grounds(
    ranges: &CollisionRanges,
    kinds: &mut Vec<(Kind, String)>,
) -> Vec<(Range<u32>, Kind)> {
    ranges
        .meshes
        .iter()
        .map(|entry| {
            let name = entry.node.strip_prefix("bul_").unwrap_or(DEFAULT_KIND);
            (entry.triangles[0]..entry.triangles[1], ground(name, kinds))
        })
        .collect()
}

fn cooked(files: &Cooked, assets: &Assets) -> Option<(Solids, Option<Solids>, Markers)> {
    let mesh = assets.get(files.collision).ok()?;
    let ranges = assets.get(files.ranges).ok()?;
    let layout = assets.get(files.records).ok()?;
    let marked = assets.get(files.markers).ok()?;
    let mut kinds = Vec::new();
    let labels = grounds(ranges, &layout.collision, &mut kinds);
    let bullets = match (files.bullet_collision, files.bullet_ranges) {
        (Some(col), Some(rng)) => {
            let bmesh = assets.get(col).ok()?;
            let branges = assets.get(rng).ok()?;
            let blabels = bullet_grounds(branges, &mut kinds);
            Some(Solids::build(&bmesh.positions, &bmesh.indices, &blabels))
        }
        _ => None,
    };
    let markers = Markers {
        kinds,
        ..markers(marked)
    };
    Some((
        Solids::build(&mesh.positions, &mesh.indices, &labels),
        bullets,
        markers,
    ))
}

impl ArenaFile {
    pub(super) fn checked(&self, path: &str) -> Result<(), Error> {
        floor_or_cooked(self.floor.as_ref(), self.cooked.as_ref(), path)
    }

    pub(super) fn built(&self, assets: &Assets) -> Option<(Solids, Option<Solids>, Markers)> {
        match (&self.floor, &self.cooked) {
            (Some(plain), _) => {
                let (solids, markers) = floor(plain);
                Some((solids, None, markers))
            }
            (None, Some(files)) => cooked(files, assets),
            (None, None) => None,
        }
    }
}
