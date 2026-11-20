use glam::Vec3;

use crate::bots::{LayoutClass, LayoutCover, LayoutZone, MapLayout};
use crate::play::cover::Cover;
use crate::play::numbers::PlayerNumbers;
use crate::play::paths::{PlaceName, ZoneArea};

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

fn cover(cover: &LayoutCover, player: &PlayerNumbers) -> Cover {
    let [x, y, z] = cover.pos;
    let [sx, sy, sz] = cover.size;
    let half = Vec3::new(sx, sz, sy) * 0.5;
    let eye = match cover.class {
        LayoutClass::Full => player.eye_height_m,
        LayoutClass::Half => player.crouch_eye_height_m,
    };
    Cover {
        center: game(x, y, z) + Vec3::Y * half.y,
        half,
        eye,
    }
}

pub(super) fn map(layout: &MapLayout, clear: f32, player: &PlayerNumbers) -> Map {
    Map {
        covers: layout.cover.iter().map(|each| cover(each, player)).collect(),
        zones: (layout.zones.iter())
            .map(|zone| zone_area(zone, clear))
            .collect(),
    }
}
