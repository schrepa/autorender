use crate::hydraulics::GRAVITY;
use engine_math::collections::HashMap;
use engine_math::{max, min};
use glam::Vec2;

use crate::hydraulics::{FLOODS, distances};
use crate::rivers::River;
use crate::{Bed, HydrologyBake, Layers, Rivers};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LakeSight {
    pub level: f32,
    pub area: f32,
    pub middle: [f32; 2],
    pub shore: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverSight {
    pub kind: &'static str,
    pub point: [f32; 3],
    pub direction: [f32; 2],
    pub half: f32,
    pub speed: f32,
    pub discharge: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RiverListing {
    pub name: Option<String>,
    pub given: Option<[u32; 2]>,
    pub courses: usize,
    pub length: f32,
    pub first: [f32; 2],
    pub last: [f32; 2],
    pub width: [f32; 2],
    pub depth: [f32; 2],
    pub flood: [f32; 2],
    pub given_width: bool,
    pub given_depth: bool,
    pub given_flood: bool,
    pub uncut: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlowSummary {
    pub rivers: usize,
    pub lakes: usize,
    pub points: usize,
    pub max_discharge: f32,
    pub fastest: f32,
    pub rapid_share: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct RiverCourse<'a> {
    river: &'a River,
}

impl RiverCourse<'_> {
    pub fn points(&self) -> &[[f32; 2]] {
        &self.river.points
    }

    pub fn half(&self, point: usize) -> f32 {
        self.river.half[point]
    }

    pub fn bed(&self, point: usize) -> f32 {
        self.river.bed[point]
    }

    pub fn channel(&self, point: usize) -> f32 {
        self.river.depth[point]
    }

    pub fn discharge(&self, point: usize) -> f32 {
        self.river.discharge[point]
    }

    pub fn bed_at(&self, point: usize) -> Bed {
        let river = self.river;
        Bed {
            reach: river.reach[point],
            d50: river.d50[point],
            d84: river.d84[point],
            power: river.power[point],
            flood: river.flood[point],
            carved: river.carved[point],
        }
    }

    pub fn crest(&self, point: usize) -> Option<f32> {
        self.river.crest[point].map(|crest| crest.share)
    }

    pub fn falls_at(&self, point: usize) -> bool {
        self.river.falls_at(point) || !self.river.surface[point].is_finite()
    }

    pub fn head(&self) -> bool {
        self.river.head
    }

    pub fn given(&self) -> Option<[u32; 2]> {
        self.river.given.source
    }

    pub fn cut(&self, point: usize) -> bool {
        self.river.cut[point]
    }
}

pub fn river_listing(layers: &Layers, count: usize) -> Vec<RiverListing> {
    let Some((_, hydrology)) = layers.hydrology_layer() else {
        return Vec::new();
    };
    let name = |[layer, line]: [u32; 2]| {
        layers.layers()
            .get(layer as usize)?
            .given_rivers()?
            .get(line as usize)?
            .name
            .clone()
    };
    let mut listings: Vec<RiverListing> = Vec::new();
    let mut held: Vec<(usize, usize)> = Vec::new();
    let mut by_source: HashMap<[u32; 2], usize> = HashMap::default();
    for river in hydrology.rivers.rivers() {
        let (Some(&first), Some(&last)) = (river.points.first(), river.points.last()) else {
            continue;
        };
        let given = river.given;
        let at = match given.source {
            Some(source) if by_source.contains_key(&source) => by_source[&source],
            source => {
                if let Some(source) = source {
                    by_source.insert(source, listings.len());
                }
                listings.push(RiverListing {
                    name: source.and_then(name),
                    given: source,
                    courses: 0,
                    length: 0.0,
                    first,
                    last,
                    width: [f32::INFINITY, 0.0],
                    depth: [f32::INFINITY, 0.0],
                    flood: [f32::INFINITY, 0.0],
                    given_width: given.width,
                    given_depth: given.depth,
                    given_flood: given.flood,
                    uncut: 0.0,
                });
                held.push((0, 0));
                listings.len() - 1
            }
        };
        let listing = &mut listings[at];
        listing.courses += 1;
        let points: Vec<Vec2> = river
            .points
            .iter()
            .map(|&point| Vec2::from(point))
            .collect();
        listing.length += distances(&points).last().copied().unwrap_or(0.0);
        listing.last = last;
        let widen = |range: &mut [f32; 2], value: f32| {
            range[0] = min(range[0], value);
            range[1] = max(range[1], value);
        };
        for index in 0..river.points.len() {
            widen(&mut listing.width, 2.0 * river.half[index]);
            widen(&mut listing.depth, river.depth[index]);
            widen(&mut listing.flood, FLOODS * river.discharge[index]);
        }
        held[at].0 += river.cut.iter().filter(|&&cut| !cut).count();
        held[at].1 += river.cut.len();
    }
    for (listing, &(uncut, points)) in listings.iter_mut().zip(&held) {
        listing.uncut = uncut as f32 / max(points as f32, 1.0);
    }
    listings.sort_by(|a, b| b.flood[1].total_cmp(&a.flood[1]));
    listings.truncate(count);
    listings
}

fn froude(river: &River, index: usize) -> f32 {
    let mean = 2.0 / 3.0 * max(river.depth[index], 1.0e-3);
    river.speed[index] / (GRAVITY * mean).sqrt()
}

pub fn courses(rivers: &Rivers) -> impl Iterator<Item = RiverCourse<'_>> {
    rivers.rivers.iter().map(|river| RiverCourse { river })
}

pub fn lake_sights(hydrology: &HydrologyBake, count: usize) -> Vec<LakeSight> {
    let water = &hydrology.water;
    let mut lakes: Vec<(usize, [f64; 2], Option<usize>)> =
        vec![(0, [0.0; 2], None); water.levels.len()];
    for (sample, &lake) in water.lake.iter().enumerate() {
        if lake == 0 || water.depth[sample] <= 0.0 {
            continue;
        }
        let entry = &mut lakes[usize::from(lake) - 1];
        let (x, z) = (sample % water.width, sample / water.width);
        entry.0 += 1;
        entry.1[0] += x as f64;
        entry.1[1] += z as f64;
        let beside = [(1, 0), (0, 1)].iter().any(|&(dx, dz)| {
            let (nx, nz) = (x + dx, z + dz);
            nx < water.width && nz < water.height && water.depth[nz * water.width + nx] <= 0.0
        });
        if beside && entry.2.is_none() {
            entry.2 = Some(sample);
        }
    }
    let mut order: Vec<usize> = (0..lakes.len()).collect();
    order.sort_by(|&a, &b| lakes[b].0.cmp(&lakes[a].0));
    order
        .into_iter()
        .take(count)
        .filter(|&lake| lakes[lake].0 > 0)
        .map(|lake| {
            let (samples, sum, shore) = lakes[lake];
            let at = |x: f32, z: f32| [x * water.spacing, z * water.spacing];
            let middle = at(
                (sum[0] / samples as f64) as f32,
                (sum[1] / samples as f64) as f32,
            );
            LakeSight {
                level: water.levels[lake],
                area: water.areas[lake],
                middle,
                shore: shore.map_or(middle, |sample| {
                    at((sample % water.width) as f32, (sample / water.width) as f32)
                }),
            }
        })
        .collect()
}

pub fn river_sights(hydrology: &HydrologyBake) -> Vec<RiverSight> {
    let rivers = hydrology.rivers.rivers();
    let key = |point: [f32; 2]| [point[0].to_bits(), point[1].to_bits()];
    let mut ends: HashMap<[u32; 2], u32> = HashMap::default();
    for river in rivers {
        if let Some(&end) = river.points.last() {
            *ends.entry(key(end)).or_default() += 1;
        }
    }
    let (mut largest, mut rapids, mut confluence) = (None, None, None);
    let better =
        |best: &mut Option<(f32, usize, usize)>, score: f32, river: usize, index: usize| {
            if best.is_none_or(|(held, _, _)| score > held) {
                *best = Some((score, river, index));
            }
        };
    for (number, river) in rivers.iter().enumerate() {
        for index in 0..river.points.len() {
            if !river.surface[index].is_finite() {
                continue;
            }
            better(&mut largest, river.discharge[index], number, index);
            better(
                &mut rapids,
                froude(river, index) * river.discharge[index].sqrt(),
                number,
                index,
            );
        }
        if ends
            .get(&key(river.points[0]))
            .is_some_and(|&count| count >= 2)
        {
            better(&mut confluence, river.discharge[0], number, 0);
        }
    }
    [
        ("largest", largest),
        ("rapids", rapids),
        ("confluence", confluence),
    ]
    .into_iter()
    .filter_map(|(kind, best)| {
        let (_, number, index) = best?;
        let river = &rivers[number];
        let next = (index + 1).min(river.points.len() - 1);
        let previous = index.saturating_sub(1);
        let run = Vec2::from(river.points[next]) - Vec2::from(river.points[previous]);
        let [x, z] = river.points[index];
        Some(RiverSight {
            kind,
            point: [x, river.surface[index], z],
            direction: run.normalize_or(Vec2::X).to_array(),
            half: river.half[index],
            speed: river.speed[index],
            discharge: river.discharge[index],
        })
    })
    .collect()
}

pub fn flow_summary(hydrology: &HydrologyBake) -> FlowSummary {
    let rivers = hydrology.rivers.rivers();
    let (mut points, mut rapid) = (0_usize, 0_usize);
    let (mut max_discharge, mut fastest) = (0.0_f32, 0.0_f32);
    for river in rivers {
        for index in 0..river.points.len() {
            if !river.surface[index].is_finite() {
                continue;
            }
            points += 1;
            rapid += usize::from(froude(river, index) > 1.0);
            max_discharge = max(max_discharge, river.discharge[index]);
            fastest = max(fastest, river.speed[index]);
        }
    }
    FlowSummary {
        rivers: rivers.len(),
        lakes: hydrology.water.levels.len(),
        points,
        max_discharge,
        fastest,
        rapid_share: rapid as f32 / max(points as f32, 1.0),
    }
}
