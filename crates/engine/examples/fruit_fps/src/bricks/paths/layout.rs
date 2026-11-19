use glam::Vec3;
use serde::Deserialize;

use crate::nav::{Cover, CoverClass};
use crate::play::paths::{PlaceName, ZoneArea};

#[derive(Clone, Debug, Deserialize)]
pub(super) struct Layout {
    zones: Vec<LayoutZone>,
    cover: Vec<LayoutCover>,
}

#[derive(Clone, Debug, Deserialize)]
struct LayoutZone {
    name: String,
    rect: [f32; 4],
    floor: f32,
    callout: String,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum LayoutClass {
    Full,
    Half,
}

#[derive(Clone, Debug, Deserialize)]
struct LayoutCover {
    pos: [f32; 3],
    size: [f32; 3],
    class: LayoutClass,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Map {
    pub covers: Vec<Cover>,
    pub zones: Vec<ZoneArea>,
}

fn game(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, z, -y)
}

fn zone_area(zone: &LayoutZone, clear: f32) -> ZoneArea {
    let [x0, y0, x1, y1] = zone.rect;
    let a = game(x0, y0, zone.floor);
    let b = game(x1, y1, zone.floor + clear);
    ZoneArea {
        place: PlaceName {
            name: zone.name.clone(),
            callout: zone.callout.clone(),
        },
        min: a.min(b),
        max: a.max(b),
    }
}

fn cover(cover: &LayoutCover) -> Cover {
    let [x, y, z] = cover.pos;
    let [sx, sy, sz] = cover.size;
    let half = Vec3::new(sx, sz, sy) * 0.5;
    let class = match cover.class {
        LayoutClass::Full => CoverClass::Full,
        LayoutClass::Half => CoverClass::Half,
    };
    Cover {
        center: game(x, y, z) + Vec3::Y * half.y,
        half,
        class,
    }
}

impl Layout {
    pub(super) fn map(&self, clear: f32) -> Map {
        Map {
            covers: self.cover.iter().map(cover).collect(),
            zones: self.zones.iter().map(|zone| zone_area(zone, clear)).collect(),
        }
    }
}
