use glam::Vec3;
use serde::Deserialize;

use crate::nav::{CoverClass, Place, PlaceShape};
use crate::play::paths::{PlaceName, ZoneArea};

#[derive(Clone, Debug, Deserialize)]
pub(super) struct Layout {
    zones: Vec<LayoutZone>,
    chokes: Vec<LayoutChoke>,
    cover: Vec<LayoutCover>,
    zone_empties: Vec<LayoutEmpty>,
}

#[derive(Clone, Debug, Deserialize)]
struct LayoutZone {
    name: String,
    rect: [f32; 4],
    floor: f32,
    callout: String,
}

#[derive(Clone, Debug, Deserialize)]
struct LayoutChoke {
    name: String,
    zones: [String; 2],
    segment: [[f32; 2]; 2],
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum LayoutClass {
    Full,
    Half,
}

#[derive(Clone, Debug, Deserialize)]
struct LayoutCover {
    name: String,
    pos: [f32; 3],
    size: [f32; 3],
    class: LayoutClass,
}

#[derive(Clone, Debug, Deserialize)]
struct LayoutEmpty {
    name: String,
    center: [f32; 3],
    half: [f32; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Choke {
    pub name: String,
    pub zones: [String; 2],
    pub a: Vec3,
    pub b: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Site {
    pub name: String,
    pub centre: Vec3,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Map {
    pub places: Vec<Place>,
    pub zones: Vec<ZoneArea>,
    pub chokes: Vec<Choke>,
    pub sites: Vec<Site>,
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

fn cover_place(cover: &LayoutCover) -> Place {
    let [x, y, z] = cover.pos;
    let [sx, sy, sz] = cover.size;
    let half = Vec3::new(sx, sz, sy) * 0.5;
    let class = match cover.class {
        LayoutClass::Full => CoverClass::Full,
        LayoutClass::Half => CoverClass::Half,
    };
    Place {
        name: cover.name.clone(),
        shape: PlaceShape::Cover {
            center: game(x, y, z) + Vec3::Y * half.y,
            half,
            class,
        },
    }
}

impl Layout {
    pub(super) fn map(&self, clear: f32) -> Result<Map, String> {
        let mut map = Map::default();
        for empty in &self.zone_empties {
            let [cx, cy, cz] = empty.center;
            let [hx, hy, hz] = empty.half;
            let a = game(cx - hx, cy - hy, cz - hz);
            let b = game(cx + hx, cy + hy, cz + hz);
            map.places.push(Place {
                name: empty.name.clone(),
                shape: PlaceShape::Area {
                    min: a.min(b),
                    max: a.max(b),
                },
            });
            if empty.name.starts_with("zone_site_") {
                map.sites.push(Site {
                    name: empty.name.clone(),
                    centre: game(cx, cy, cz - hz),
                });
            }
        }
        for zone in &self.zones {
            let area = zone_area(zone, clear);
            map.places.push(Place {
                name: zone.name.clone(),
                shape: PlaceShape::Area {
                    min: area.min,
                    max: area.max,
                },
            });
            map.zones.push(area);
        }
        for choke in &self.chokes {
            let [[ax, ay], [bx, by]] = choke.segment;
            let floor = self
                .zones
                .iter()
                .find(|zone| zone.name == choke.zones[0])
                .map(|zone| zone.floor)
                .ok_or_else(|| {
                    format!("choke {} names unknown zone {}", choke.name, choke.zones[0])
                })?;
            let (a, b) = (game(ax, ay, floor), game(bx, by, floor));
            map.places.push(Place {
                name: choke.name.clone(),
                shape: PlaceShape::Choke { a, b },
            });
            map.chokes.push(Choke {
                name: choke.name.clone(),
                zones: choke.zones.clone(),
                a,
                b,
            });
        }
        map.places.extend(self.cover.iter().map(cover_place));
        Ok(map)
    }
}
