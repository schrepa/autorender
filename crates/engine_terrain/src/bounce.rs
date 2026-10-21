//! A terrain's bounce light as transfer: what each probe over its ground
//! sees of that ground, made once from its heights and its surface's rules
//! (the job [`crate::jobs::BounceJob`]) and kept as a record. Nothing in it
//! depends on the sun or the sky: renderers light the ground with them every
//! frame, then gather through each probe's bins the light the ground sends
//! it, as the low-order spherical harmonics shading reads.
//!
//! Probes stand over a grid of columns across the terrain, `probes` along
//! each side, in two layers: [`LOW_PROBE`] and [`HIGH_PROBE`] meters above
//! the ground. Columns pair into blocks of two by two. Where the ground
//! around a block rises above it (valleys, hollows, the foot of slopes), so
//! that what its probes see changes quickly from one column to the next,
//! each of its columns holds probes of its own; elsewhere one probe at the
//! block's middle stands for all four. A probe's view divides into [`BINS`]
//! bins: a cap below, three bands about the horizon in [`SECTORS`]
//! directions each, and a cap above. Each bin holds the share of it the
//! ground fills and how far away that ground lies on average, where
//! renderers read the ground's light.
//!
//! The ground's surface comes along: at every texel of a grid half as fine
//! as the columns, its two strongest materials and their blend, as its rules
//! choose them, and its normal. Renderers light it there in the materials'
//! colors as they draw them.
//!
//! Views come from walks: along each of `AZIMUTHS` directions from a column,
//! the ground's rise per meter seen from each layer's probe; each time the
//! horizon walked so far rises, the sights between the old horizon and the
//! new meet the ground there. Beyond the terrain its edge carries on.

use std::f32::consts::TAU;

use engine_math::{clamp, max, min, sin_cos};

use crate::codec::{self, Codec, CodecError, Reader, Writer};
use crate::gpu::control_at;
use crate::lighting::{Ground, normalized, unorm};
use crate::{BiomeRules, Layers};

/// The probes' heights above the ground, in meters: the lower where the
/// ground's own surfaces and what stands on it read their light, the upper
/// above trees and roofs.
pub const LOW_PROBE: f32 = 1.5;
pub const HIGH_PROBE: f32 = 24.0;

/// Layers of probes over each column.
pub const LAYERS: usize = 2;

/// Directions about a probe its bands divide into: sector `k` spans an
/// eighth of a turn about `k` eighths from +x toward +z.
pub const SECTORS: usize = 8;

/// Bins of a probe's view: the cap below, the three bands of [`SECTORS`]
/// each from low to high, the cap above.
pub const BINS: usize = 2 + 3 * SECTORS;

/// The sines of the elevations bounding the bands: the caps lie beyond
/// [`STEEP`] (30 degrees) below and above, the middle band within
/// [`LEVEL`] (8 degrees) of level.
pub const STEEP: f32 = 0.5;
pub const LEVEL: f32 = 0.14;

/// Bytes a probe takes: for each layer and bin, the share of it the ground
/// fills (255 whole) and the code of its distance.
pub const PROBE_BYTES: usize = LAYERS * BINS * 2;

/// A distance code `c` stands for [`NEAREST`] times 2 to the `c` over
/// [`DISTANCE_STEPS`] meters.
pub const NEAREST: f32 = 0.5;
pub const DISTANCE_STEPS: f32 = 16.0;

/// Planes of the ground's surface, a byte per texel each: its strongest
/// material, the second, the second's share of the two, and its normal's x
/// and z from 128.
pub const GROUND_PLANES: usize = 5;

/// The most probes along a terrain's side.
pub const MAX_PROBES: u32 = 4096;

/// Directions walked from each column: four to a sector.
const AZIMUTHS: usize = 4 * SECTORS;

/// Meters walks reach at most: ground farther fills too little of a view to
/// light it.
const REACH: f32 = 3000.0;

/// The share of the sky above level the ground hides from a block's middle
/// probe (the mean over directions of the sine of the horizon's elevation)
/// from which its columns hold probes of their own.
const ENCLOSED: f32 = 0.08;

/// The sines bounding the bands, from below.
const EDGES: [f32; 6] = [-1.0, -STEEP, -LEVEL, LEVEL, STEEP, 1.0];

/// A terrain's bounce transfer: `probes` columns along each side, in blocks
/// of two by two each holding probes of its own columns or one for all four
/// (one byte per block, rows along z), every probe's bins
/// ([`PROBE_BYTES`] each, in blocks' order, a block's own columns' in rows),
/// and the ground's surface: [`GROUND_PLANES`] planes of `probes / 2`
/// texels a side, all over the terrain's extent.
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
    /// Texels along each side of the ground's surface: one per block.
    pub fn texels(&self) -> u32 {
        self.probes / 2
    }

    /// Each block's word as the GPU takes it, rows along z: the index of its
    /// first probe, times two, plus one where its columns hold probes of
    /// their own (four, in rows) rather than one for all.
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

    /// Blocks whose columns hold probes of their own.
    pub fn detailed_blocks(&self) -> usize {
        self.detailed.iter().filter(|&&flag| flag != 0).count()
    }

    /// Every probe's bytes, in blocks' order: [`PROBE_BYTES`] each, its
    /// layers in order, each bin's share then its distance's code.
    pub fn probe_bytes(&self) -> &[u8] {
        &self.records
    }

    /// Probes it holds.
    pub fn probe_count(&self) -> usize {
        self.records.len() / PROBE_BYTES
    }

    /// The ground's surface as the GPU takes it: two layers of four bytes a
    /// texel, rows along z: its strongest material, the second and the
    /// second's share (red, green, blue); then its normal's x and z, from
    /// 128 (red, green).
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

/// Each bin's moments over its solid angle, which renderers gather what it
/// holds into spherical harmonics by: the integral over it of the direction
/// (x, y and z) and its solid angle (w).
pub fn bin_moments() -> [[f32; 4]; BINS] {
    // The integral of the cosine of the elevation over its sine.
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

/// The bin of band `band` (0 the cap below, 4 the cap above) in sector
/// `sector`.
const fn bin(band: usize, sector: usize) -> usize {
    match band {
        0 => 0,
        1..=3 => 1 + (band - 1) * SECTORS + sector,
        _ => BINS - 1,
    }
}

/// What a probe's walks found of its view: per bin, the sines of the
/// ground's sights there summed over the walks' directions, and those times
/// their ground's distance; and the mean over directions of the sine of the
/// horizon's elevation above level.
#[derive(Clone, Copy, Default)]
struct View {
    filled: [f32; BINS],
    distance: [f32; BINS],
    enclosed: f32,
}

impl View {
    /// Adds the sights of a walk in sector `sector` whose sines run from
    /// `from` to `to`, meeting the ground `distance` meters away.
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

    /// Its bins as a probe's layer holds them, appended to `out`.
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

/// The directions walked from each column, x and z: `AZIMUTHS` evenly
/// around, the first four in sector 0.
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

/// What walks see of the ground around a column, read the same way from
/// every column.
struct Walker<'a> {
    ground: &'a Ground<'a>,
    directions: [[f32; 2]; AZIMUTHS],
    /// The highest ground, past which nothing farther rises.
    highest: f32,
    /// Meters to the first step, the least step and the farthest.
    first: f32,
    step: f32,
    reach: f32,
}

/// A probe on a walk: its height, and its horizon so far as the ground's
/// rise per meter and its sine; none before the first step.
#[derive(Clone, Copy)]
struct Eye {
    height: f32,
    tangent: Option<f32>,
    horizon: f32,
}

impl Walker<'_> {
    /// The views of the probes over terrain-local `(x, z)`, one per layer.
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
                            // The first step's sights meet the ground on
                            // the way to it.
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
                    // Nothing farther rises over the horizon once even the
                    // highest ground would not.
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

/// The bounce transfer of `ground` over `extent` (meters along x and z from
/// its first sample), `probes` columns along each side (even), its surface
/// chosen by `surface`'s rules over its layers.
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

/// The ground's surface planes, `texels` a side over `extent`: each texel's
/// control value as `surface`'s rules give it there, from the ground's
/// heights a texel (or a sample, whichever is wider) around it, and its
/// normal across the same.
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
            let height = ground.height(px, pz);
            let west = ground.height(px - across, pz);
            let east = ground.height(px + across, pz);
            let north = ground.height(px, pz - across);
            let south = ground.height(px, pz + across);
            let control = control_at(
                (layers, rules),
                [px + offset.x, pz + offset.y],
                [height, west, east, north, south],
                across,
                &mut weights,
            );
            let normal = normalized([west - east, 2.0 * across, north - south]);
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
