use std::f32::consts::TAU;

use engine_math::{clamp, max, min, sin_cos};

use crate::codec::{self, Codec, CodecError, Reader, Writer};
use crate::gpu::control_at;
use crate::lighting::{Ground, normalized, unorm};
use crate::{BiomeRules, Layers, SampleShape};

pub const LOW_PROBE: f32 = 1.5;
pub const HIGH_PROBE: f32 = 24.0;

pub const LAYERS: usize = 2;

pub const SECTORS: usize = 8;

pub const BINS: usize = 2 + 3 * SECTORS;

pub const STEEP: f32 = 0.5;
pub const LEVEL: f32 = 0.14;

pub const PROBE_BYTES: usize = LAYERS * BINS * 2;

pub const NEAREST: f32 = 0.5;
pub const DISTANCE_STEPS: f32 = 16.0;

pub const GROUND_PLANES: usize = 5;

pub const MAX_PROBES: u32 = 4096;

const AZIMUTHS: usize = 4 * SECTORS;

const REACH: f32 = 3000.0;

const ENCLOSED: f32 = 0.08;

const EDGES: [f32; 6] = [-1.0, -STEEP, -LEVEL, LEVEL, STEEP, 1.0];

#[derive(Clone, Debug, PartialEq)]
pub struct TerrainBounce {
    pub probes: u32,
    detailed: Box<[u8]>,
    records: Box<[u8]>,
    ground: Box<[u8]>,
}

impl Codec for TerrainBounce {
    fn write(&self, out: &mut Writer) {
        out.put(&self.probes);
        let blocks = (self.probes / 2) as usize;
        codec::put_planes(out, &self.detailed, blocks);
        codec::put_records(out, &self.records, PROBE_BYTES);
        codec::put_planes(out, &self.ground, blocks);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        let probes: u32 = input.get()?;
        if !(2..=MAX_PROBES).contains(&probes) || !probes.is_multiple_of(2) {
            return Err(CodecError("bounce probes of another count"));
        }
        let blocks = (probes / 2) as usize;
        let detailed = codec::get_planes(input, blocks)?;
        let records = codec::get_records(input, PROBE_BYTES)?;
        let ground = codec::get_planes(input, blocks)?;
        let held: usize = detailed
            .iter()
            .map(|&flag| if flag == 0 { 1 } else { 4 })
            .sum();
        if detailed.len() != blocks * blocks
            || detailed.iter().any(|&flag| flag > 1)
            || records.len() != held * PROBE_BYTES
            || ground.len() != GROUND_PLANES * blocks * blocks
        {
            return Err(CodecError("bounce of another size"));
        }
        Ok(Self {
            probes,
            detailed,
            records,
            ground,
        })
    }
}

impl TerrainBounce {
    pub fn texels(&self) -> u32 {
        self.probes / 2
    }

    pub fn block_words(&self) -> Vec<u32> {
        let mut first = 0_u32;
        self.detailed
            .iter()
            .map(|&flag| {
                let word = (first << 1) | u32::from(flag);
                first += if flag == 0 { 1 } else { 4 };
                word
            })
            .collect()
    }

    pub fn detailed_blocks(&self) -> usize {
        self.detailed.iter().filter(|&&flag| flag != 0).count()
    }

    pub fn probe_bytes(&self) -> &[u8] {
        &self.records
    }

    pub fn probe_count(&self) -> usize {
        self.records.len() / PROBE_BYTES
    }

    pub fn ground_layers(&self) -> Vec<u8> {
        let side = self.texels() as usize;
        let area = side * side;
        let plane = |index: usize| &self.ground[index * area..(index + 1) * area];
        let mut texels = Vec::with_capacity(8 * area);
        for at in 0..area {
            texels.extend([plane(0)[at], plane(1)[at], plane(2)[at], 0]);
        }
        for at in 0..area {
            texels.extend([plane(3)[at], plane(4)[at], 0, 0]);
        }
        texels
    }
}

pub fn bin_moments() -> [[f32; 4]; BINS] {
    let across = |sine: f32| 0.5 * (sine * (1.0 - sine * sine).sqrt() + libm::asinf(sine));
    let mut moments = [[0.0; 4]; BINS];
    for band in 0..EDGES.len() - 1 {
        let (low, high) = (EDGES[band], EDGES[band + 1]);
        let rise = 0.5 * (high * high - low * low);
        let level = across(high) - across(low);
        if band == 0 || band == EDGES.len() - 2 {
            moments[bin(band, 0)] = [0.0, TAU * rise, 0.0, TAU * (high - low)];
            continue;
        }
        let width = TAU / SECTORS as f32;
        for sector in 0..SECTORS {
            let first = (sector as f32 - 0.5) * width;
            let (sin_first, cos_first) = sin_cos(first);
            let (sin_last, cos_last) = sin_cos(first + width);
            moments[bin(band, sector)] = [
                level * (sin_last - sin_first),
                width * rise,
                level * (cos_first - cos_last),
                width * (high - low),
            ];
        }
    }
    moments
}

const fn bin(band: usize, sector: usize) -> usize {
    match band {
        0 => 0,
        1..=3 => 1 + (band - 1) * SECTORS + sector,
        _ => BINS - 1,
    }
}

#[derive(Clone, Copy, Default)]
struct View {
    filled: [f32; BINS],
    distance: [f32; BINS],
    enclosed: f32,
}

impl View {
    fn add(&mut self, sector: usize, (from, to): (f32, f32), distance: f32) {
        for band in 0..EDGES.len() - 1 {
            let overlap = min(to, EDGES[band + 1]) - max(from, EDGES[band]);
            if overlap > 0.0 {
                let at = bin(band, sector);
                self.filled[at] += overlap;
                self.distance[at] += overlap * distance;
            }
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        for band in 0..EDGES.len() - 1 {
            let caps = band == 0 || band == EDGES.len() - 2;
            let (sectors, walks) = if caps {
                (1, AZIMUTHS)
            } else {
                (SECTORS, AZIMUTHS / SECTORS)
            };
            let height = EDGES[band + 1] - EDGES[band];
            for sector in 0..sectors {
                let at = bin(band, sector);
                let share = self.filled[at] / (height * walks as f32);
                out.push(unorm(share));
                let code = if self.filled[at] > 0.0 {
                    let mean = max(self.distance[at] / self.filled[at], NEAREST);
                    let steps = libm::log2f(mean / NEAREST) * DISTANCE_STEPS;
                    (clamp(steps, 0.0, 255.0) + 0.5) as u8
                } else {
                    0
                };
                out.push(code);
            }
        }
    }
}

fn directions() -> [[f32; 2]; AZIMUTHS] {
    let width = TAU / AZIMUTHS as f32;
    let start = -0.5 * TAU / SECTORS as f32;
    let mut directions = [[0.0; 2]; AZIMUTHS];
    for (index, direction) in directions.iter_mut().enumerate() {
        let (sin, cos) = sin_cos(start + (index as f32 + 0.5) * width);
        *direction = [cos, sin];
    }
    directions
}

struct Walker<'a> {
    ground: &'a Ground<'a>,
    directions: [[f32; 2]; AZIMUTHS],
    highest: f32,
    first: f32,
    step: f32,
    reach: f32,
}

#[derive(Clone, Copy)]
struct Eye {
    height: f32,
    tangent: Option<f32>,
    horizon: f32,
}

impl Walker<'_> {
    fn views(&self, (x, z): (f32, f32)) -> [View; LAYERS] {
        let base = self.ground.height(x, z);
        let heights = [base + LOW_PROBE, base + HIGH_PROBE];
        let mut views = [View::default(); LAYERS];
        for (azimuth, &[dx, dz]) in self.directions.iter().enumerate() {
            let sector = azimuth / (AZIMUTHS / SECTORS);
            let mut eyes = heights.map(|height| Eye {
                height,
                tangent: None,
                horizon: -1.0,
            });
            let mut distance = self.first;
            loop {
                let ahead = self.ground.height(x + dx * distance, z + dz * distance);
                let mut done = true;
                for (eye, view) in eyes.iter_mut().zip(views.iter_mut()) {
                    let rise = (ahead - eye.height) / distance;
                    let tangent = match eye.tangent {
                        Some(tangent) if tangent >= rise => tangent,
                        held => {
                            let reach = if held.is_none() {
                                0.5 * distance
                            } else {
                                distance
                            };
                            let sine = rise / (1.0 + rise * rise).sqrt();
                            view.add(sector, (eye.horizon, sine), reach);
                            eye.horizon = sine;
                            rise
                        }
                    };
                    eye.tangent = Some(tangent);
                    done &= eye.height <= self.highest
                        && self.highest - eye.height <= tangent * distance;
                }
                let next = distance + max(self.step, 0.05 * distance);
                if done || next > self.reach {
                    break;
                }
                distance = next;
            }
            for (view, eye) in views.iter_mut().zip(eyes) {
                view.enclosed += max(eye.horizon, 0.0) / AZIMUTHS as f32;
            }
        }
        views
    }
}

pub fn bounce(
    ground: &Ground<'_>,
    extent: [f32; 2],
    surface: (&Layers, &BiomeRules),
    probes: u32,
) -> TerrainBounce {
    let blocks = (probes / 2) as usize;
    let spacing = [extent[0] / probes as f32, extent[1] / probes as f32];
    let walker = Walker {
        ground,
        directions: directions(),
        highest: ground
            .heights
            .iter()
            .fold(f32::NEG_INFINITY, |high, &height| max(high, height)),
        first: 0.25 * ground.spacing,
        step: 0.5 * ground.spacing,
        reach: min(REACH, 2.0 * max(extent[0], extent[1])),
    };
    let mut detailed = vec![0_u8; blocks * blocks];
    let mut records = Vec::with_capacity(blocks * blocks * PROBE_BYTES);
    let push = |views: &[View; LAYERS], records: &mut Vec<u8>| {
        for view in views {
            view.encode(records);
        }
    };
    for z in 0..blocks {
        for x in 0..blocks {
            let middle = (
                (2 * x + 1) as f32 * spacing[0],
                (2 * z + 1) as f32 * spacing[1],
            );
            let views = walker.views(middle);
            if views[0].enclosed < ENCLOSED {
                push(&views, &mut records);
                continue;
            }
            detailed[z * blocks + x] = 1;
            for corner in 0..4 {
                let column = (2 * x + corner % 2, 2 * z + corner / 2);
                let center = (
                    (column.0 as f32 + 0.5) * spacing[0],
                    (column.1 as f32 + 0.5) * spacing[1],
                );
                push(&walker.views(center), &mut records);
            }
        }
    }
    TerrainBounce {
        probes,
        detailed: detailed.into_boxed_slice(),
        records: records.into_boxed_slice(),
        ground: surface_planes(ground, extent, surface, blocks),
    }
}

fn surface_planes(
    ground: &Ground<'_>,
    extent: [f32; 2],
    (layers, rules): (&Layers, &BiomeRules),
    texels: usize,
) -> Box<[u8]> {
    let area = texels * texels;
    let mut planes = vec![0_u8; GROUND_PLANES * area];
    let step = [extent[0] / texels as f32, extent[1] / texels as f32];
    let across = max(max(step[0], step[1]), ground.spacing);
    let offset = layers.offset;
    let mut weights = Vec::new();
    for z in 0..texels {
        for x in 0..texels {
            let (px, pz) = ((x as f32 + 0.5) * step[0], (z as f32 + 0.5) * step[1]);
            let around = [-1.0, 0.0, 1.0].map(|dz| {
                [-1.0, 0.0, 1.0].map(|dx| ground.height(px + dx * across, pz + dz * across))
            });
            let shape = SampleShape::of(around, across);
            let control = control_at(
                (layers, rules),
                [px + offset.x, pz + offset.y],
                (around[1][1], shape),
                across,
                &mut weights,
            );
            let [gx, gz] = shape.gradient;
            let normal = normalized([-gx, 1.0, -gz]);
            let at = z * texels + x;
            planes[at] = control.a;
            planes[area + at] = control.b;
            planes[2 * area + at] = control.blend;
            planes[3 * area + at] = unorm(0.5 + 0.5 * normal[0]);
            planes[4 * area + at] = unorm(0.5 + 0.5 * normal[2]);
        }
    }
    planes.into_boxed_slice()
}
