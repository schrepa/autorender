use glam::Vec3;
use serde::Deserialize;

pub type ZoneId = u16;

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct LayoutZone {
    pub name: String,
    pub rect: [f32; 4],
    pub floor: f32,
    pub callout: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct LayoutPortal {
    pub zones: [String; 2],
    pub segment: [[f32; 2]; 2],
    pub width: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct LayoutChoke {
    pub name: String,
    pub zones: [String; 2],
    pub segment: [[f32; 2]; 2],
    pub width: f32,
    #[serde(default, alias = "required_choke")]
    pub deliberate: bool,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct LayoutSpawn {
    pub name: String,
    pub team: String,
    pub pos: [f32; 3],
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct LayoutPlant {
    pub site: String,
    pub rect: [f32; 4],
    pub center: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutClass {
    Full,
    Half,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct LayoutCover {
    pub name: String,
    pub pos: [f32; 3],
    pub size: [f32; 3],
    pub class: LayoutClass,
    pub zone: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct LayoutCallout {
    pub name: String,
    pub zone: String,
    pub pos: [f32; 2],
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct MapLayout {
    pub zones: Vec<LayoutZone>,
    pub portals: Vec<LayoutPortal>,
    pub chokes: Vec<LayoutChoke>,
    pub spawns: Vec<LayoutSpawn>,
    pub plant_zones: Vec<LayoutPlant>,
    pub cover: Vec<LayoutCover>,
    pub callouts: Vec<LayoutCallout>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Zone {
    pub name: String,
    pub callout: String,
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    pub name: String,
    pub zones: [ZoneId; 2],
    pub a: Vec3,
    pub b: Vec3,
    pub width: f32,
    pub required: bool,
}

impl Gate {
    pub fn middle(&self) -> Vec3 {
        (self.a + self.b) * 0.5
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CoverBox {
    pub name: String,
    pub zone: ZoneId,
    pub center: Vec3,
    pub half: Vec3,
    pub full: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Site {
    pub name: String,
    pub zone: ZoneId,
    pub center: Vec3,
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Spot {
    pub name: String,
    pub zone: ZoneId,
    pub at: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Angle {
    pub at: Vec3,
    pub zone: ZoneId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlacesError {
    UnknownZone(String),
    NoZoneAt(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Places {
    pub zones: Vec<Zone>,
    pub portals: Vec<Gate>,
    pub chokes: Vec<Gate>,
    pub covers: Vec<CoverBox>,
    pub sites: Vec<Site>,
    pub attack_spawns: Vec<Vec3>,
    pub defend_spawns: Vec<Vec3>,
    pub spots: Vec<Spot>,
    pub angles: Vec<Angle>,
}

fn game_point(x: f32, y: f32, up: f32) -> Vec3 {
    Vec3::new(x, up, -y)
}

fn rect_bounds(rect: [f32; 4], floor: f32) -> (Vec3, Vec3) {
    let a = game_point(rect[0], rect[1], floor);
    let b = game_point(rect[2], rect[3], floor);
    (a.min(b), a.max(b))
}

fn inside(min: Vec3, max: Vec3, p: Vec3) -> bool {
    p.x >= min.x && p.x <= max.x && p.z >= min.z && p.z <= max.z
}

impl Places {
    pub fn from_layout(layout: &MapLayout) -> Result<Places, PlacesError> {
        let zones: Vec<Zone> = layout
            .zones
            .iter()
            .map(|z| {
                let (min, max) = rect_bounds(z.rect, z.floor);
                Zone {
                    name: z.name.clone(),
                    callout: z.callout.clone(),
                    min,
                    max,
                }
            })
            .collect();
        let mut places = Places {
            zones,
            portals: Vec::new(),
            chokes: Vec::new(),
            covers: Vec::new(),
            sites: Vec::new(),
            attack_spawns: Vec::new(),
            defend_spawns: Vec::new(),
            spots: Vec::new(),
            angles: Vec::new(),
        };
        places.read_gates(layout)?;
        places.read_items(layout)?;
        places.derive_angles();
        Ok(places)
    }

    fn named(&self, name: &str) -> Result<ZoneId, PlacesError> {
        self.zone(name)
            .ok_or_else(|| PlacesError::UnknownZone(name.to_owned()))
    }

    fn floor_at(&self, x: f32, y: f32) -> f32 {
        let p = game_point(x, y, 0.0);
        self.zone_at(p)
            .map_or(0.0, |z| self.zones[usize::from(z)].min.y)
    }

    fn read_gates(&mut self, layout: &MapLayout) -> Result<(), PlacesError> {
        for portal in &layout.portals {
            let zones = [self.named(&portal.zones[0])?, self.named(&portal.zones[1])?];
            let floor = self.zones[usize::from(zones[0])].min.y;
            self.portals.push(Gate {
                name: format!("{}-{}", portal.zones[0], portal.zones[1]),
                zones,
                a: game_point(portal.segment[0][0], portal.segment[0][1], floor),
                b: game_point(portal.segment[1][0], portal.segment[1][1], floor),
                width: portal.width,
                required: false,
            });
        }
        for choke in &layout.chokes {
            let zones = [self.named(&choke.zones[0])?, self.named(&choke.zones[1])?];
            let floor = self.zones[usize::from(zones[0])].min.y;
            self.chokes.push(Gate {
                name: choke.name.clone(),
                zones,
                a: game_point(choke.segment[0][0], choke.segment[0][1], floor),
                b: game_point(choke.segment[1][0], choke.segment[1][1], floor),
                width: choke.width,
                required: choke.deliberate,
            });
        }
        Ok(())
    }

    fn read_items(&mut self, layout: &MapLayout) -> Result<(), PlacesError> {
        for cover in &layout.cover {
            let zone = self.named(&cover.zone)?;
            let half = Vec3::new(cover.size[0], cover.size[2], cover.size[1]) * 0.5;
            let base = game_point(cover.pos[0], cover.pos[1], cover.pos[2]);
            self.covers.push(CoverBox {
                name: cover.name.clone(),
                zone,
                center: base + Vec3::new(0.0, half.y, 0.0),
                half,
                full: cover.class == LayoutClass::Full,
            });
        }
        for plant in &layout.plant_zones {
            let floor = self.floor_at(plant.center[0], plant.center[1]);
            let center = game_point(plant.center[0], plant.center[1], floor);
            let zone = self
                .zone_at(center)
                .ok_or_else(|| PlacesError::NoZoneAt(plant.site.clone()))?;
            let (min, max) = rect_bounds(plant.rect, floor);
            self.sites.push(Site {
                name: plant.site.clone(),
                zone,
                center,
                min,
                max,
            });
        }
        for spawn in &layout.spawns {
            let at = game_point(spawn.pos[0], spawn.pos[1], spawn.pos[2]);
            if spawn.team == "attack" {
                self.attack_spawns.push(at);
            } else {
                self.defend_spawns.push(at);
            }
        }
        for callout in &layout.callouts {
            let zone = self.named(&callout.zone)?;
            let floor = self.zones[usize::from(zone)].min.y;
            self.spots.push(Spot {
                name: callout.name.clone(),
                zone,
                at: game_point(callout.pos[0], callout.pos[1], floor),
            });
        }
        Ok(())
    }

    fn derive_angles(&mut self) {
        for choke in &self.chokes {
            for end in [choke.a, choke.b] {
                self.angles.push(Angle {
                    at: end,
                    zone: choke.zones[1],
                });
            }
        }
        for cover in &self.covers {
            for (sx, sz) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let at = Vec3::new(
                    cover.center.x + sx * cover.half.x,
                    cover.center.y - cover.half.y,
                    cover.center.z + sz * cover.half.z,
                );
                self.angles.push(Angle {
                    at,
                    zone: cover.zone,
                });
            }
        }
    }

    pub fn zone(&self, name: &str) -> Option<ZoneId> {
        self.zones
            .iter()
            .position(|z| z.name == name)
            .and_then(|i| ZoneId::try_from(i).ok())
    }

    pub fn zone_at(&self, p: Vec3) -> Option<ZoneId> {
        self.zones
            .iter()
            .position(|z| inside(z.min, z.max, p))
            .and_then(|i| ZoneId::try_from(i).ok())
    }

    pub fn site_at(&self, p: Vec3) -> Option<usize> {
        self.sites.iter().position(|s| inside(s.min, s.max, p))
    }

    pub fn site_zone_at(&self, p: Vec3) -> Option<usize> {
        let zone = self.zone_at(p)?;
        self.sites.iter().position(|s| s.zone == zone)
    }
}
