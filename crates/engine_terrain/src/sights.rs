//! What a hydrology bake's water shows those inspecting it, read here so
//! its lake grid and river lines stay the crate's own: its largest lakes,
//! where its rivers carry most, run fastest against their waves and meet
//! with the most water, how they flow, where pools stand behind what holds
//! them back, and the courses rivers run along for worlds placing things
//! in them. Water itself is read from tiles ([`crate::Tile::water_at`]).

use engine_math::collections::HashMap;
use engine_math::max;
use glam::Vec2;

use crate::hydraulics::GRAVITY;
use crate::rivers::River;
use crate::{Hydrology, Pool, RiverFlow, Rivers};

/// A lake: its level (terrain-local meters), the area its water covers
/// (square meters), and its middle and a sample of its shore (terrain-local
/// x and z).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LakeSight {
    pub level: f32,
    pub area: f32,
    pub middle: [f32; 2],
    pub shore: [f32; 2],
}

/// A river point worth looking at: what makes it one (`largest`, `rapids`
/// or `confluence`), the point (terrain-local x, surface and z), the way
/// the river runs there (a unit vector along x and z), half its channel's
/// width, its speed (meters a second), its discharge (cubic meters a
/// second) and the foam riding it (0 to 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverSight {
    pub kind: &'static str,
    pub point: [f32; 3],
    pub direction: [f32; 2],
    pub half: f32,
    pub speed: f32,
    pub discharge: f32,
    pub foam: f32,
}

/// How a bake's rivers and lakes flow: how many there are, the river
/// points holding water, the most any carries (cubic meters a second), the
/// fastest any runs (meters a second), and the shares of those points
/// running faster than their waves and carrying foam.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlowSummary {
    pub rivers: usize,
    pub lakes: usize,
    pub points: usize,
    pub max_discharge: f32,
    pub fastest: f32,
    pub rapid_share: f32,
    pub foam_share: f32,
}

/// A river's course as worlds placing things in rivers read it: its points
/// (terrain-local x and z, from upstream) and at each half its channel's
/// width, its bed and its channel's depth below its banks (meters), the
/// discharge it carries (cubic meters a second), and where it falls. Its
/// water is the tiles' to say.
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

    /// Its bed at `point`: NaN where its water falls clear of it.
    pub fn bed(&self, point: usize) -> f32 {
        self.river.surface[point] - self.river.depth[point]
    }

    pub fn channel(&self, point: usize) -> f32 {
        self.river.channel[point]
    }

    pub fn discharge(&self, point: usize) -> f32 {
        self.river.discharge[point]
    }

    /// Whether its water falls at `point`.
    pub fn falls_at(&self, point: usize) -> bool {
        self.river.falls_at(point) || !self.river.surface[point].is_finite()
    }
}

impl Rivers {
    /// Every river's course.
    pub fn courses(&self) -> impl Iterator<Item = RiverCourse<'_>> {
        self.rivers.iter().map(|river| RiverCourse { river })
    }
}

impl Hydrology {
    /// Its largest `count` lakes, by the area their water covers.
    pub fn lake_sights(&self, count: usize) -> Vec<LakeSight> {
        let water = &self.water;
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
            // A shore: a sample of the lake beside one holding none.
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
                    area: samples as f32 * water.spacing * water.spacing,
                    middle,
                    shore: shore.map_or(middle, |sample| {
                        at((sample % water.width) as f32, (sample / water.width) as f32)
                    }),
                }
            })
            .collect()
    }

    /// Where its rivers carry most, run fastest against their waves (by
    /// their Froude number and the square root of their discharge) and
    /// meet with the most water.
    pub fn river_sights(&self) -> Vec<RiverSight> {
        let rivers = self.rivers.rivers();
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
                let depth = max(river.depth[index], 1.0e-3);
                let froude = river.speed[index] / (GRAVITY * depth).sqrt();
                better(&mut largest, river.discharge[index], number, index);
                better(
                    &mut rapids,
                    froude * river.discharge[index].sqrt(),
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
                foam: river.foam[index],
            })
        })
        .collect()
    }

    /// How its rivers and lakes flow.
    pub fn flow_summary(&self) -> FlowSummary {
        let rivers = self.rivers.rivers();
        let (mut points, mut rapid, mut foamy) = (0_usize, 0_usize, 0_usize);
        let (mut max_discharge, mut fastest) = (0.0_f32, 0.0_f32);
        for river in rivers {
            for index in 0..river.points.len() {
                if !river.surface[index].is_finite() {
                    continue;
                }
                points += 1;
                let speed = river.speed[index];
                let depth = max(river.depth[index], 1.0e-3);
                rapid += usize::from(speed > (GRAVITY * depth).sqrt());
                foamy += usize::from(river.foam[index] > 0.1);
                max_discharge = max(max_discharge, river.discharge[index]);
                fastest = max(fastest, speed);
            }
        }
        let share = |count: usize| count as f32 / max(points as f32, 1.0);
        FlowSummary {
            rivers: rivers.len(),
            lakes: self.water.levels.len(),
            points,
            max_discharge,
            fastest,
            rapid_share: share(rapid),
            foam_share: share(foamy),
        }
    }
}

impl RiverFlow {
    /// Where `pool` stands (terrain-local x and z) and how far above the
    /// bake's water there it rises (meters).
    pub fn pool_place(&self, pool: &Pool) -> ([f32; 2], f32) {
        let river = &self.base().rivers()[pool.river as usize];
        let point = pool.point as usize;
        (river.points[point], pool.level - river.surface[point])
    }
}
