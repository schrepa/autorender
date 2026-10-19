//! Water's waves (`water_waves.wgsl`): [`WAVE_TILES`] tiles of wind-driven
//! waves that terrains' water and the sea shade with, each repeating at
//! its own length. Together their spectra are one spectrum the wind shapes
//! (Phillips's, after Tessendorf), split into bands of wave numbers, each
//! tile taking those its grid holds densely enough; they are drawn from
//! fixed noise, and drawn again when the wind or the ripples' size
//! changes. Each frame the spectra evolve (gravity and surface tension set
//! each wave's speed; frequencies are rounded so the waves loop every
//! [`LOOP_SECONDS`]) and an inverse transform turns them into the slopes of
//! the surface, whose moments (the slopes, and the square of their length)
//! fill a texture array with its mip levels. Water reads the slopes where
//! it stands and, from what the levels' averages lose of the square, the
//! roughness of what a pixel spans.

use std::f64::consts::TAU;

use engine_core::math::max;

use crate::bindings::WATER_WAVES;
use crate::gpu_layouts::constants::{WAVE_LEVEL_GROUP, WAVE_LINE_GROUP, WAVE_TEXELS, WAVE_TILES};
use crate::gpu_layouts::water::WaveFrame;
use crate::shaders::entry::water_waves;

/// Each tile's meters, as shares of the ripples' size: the first holds the
/// swell longer than the largest ripples, the second ripples down to an
/// eighth of them, the third the finer ones. Each band spans an eighth of
/// its tile's wave numbers, so its shortest waves span eight texels and
/// filter smoothly.
const TILE_SHARES: [f64; WAVE_TILES] = [32.0, 4.0, 0.5];
/// Where each band ends, in wave numbers of the next tile's grid: a tile
/// takes the waves from its first wave number to this many of the next
/// tile's, so every tile holds its band with enough wave numbers.
const BAND_END: f64 = 4.0;
/// The surface's root mean square slope at full ripples.
const FULL_SLOPE: f64 = 0.5;
/// Meters under which waves fade out: the shortest ripples tilt the
/// surface too little to see.
const SHORTEST: f64 = 0.01;
/// The calmest wind the spectrum takes (meters a second), and the share
/// of waves running against it.
const CALMEST: f64 = 1.0;
const AGAINST: f64 = 0.2;
/// Seconds after which the waves repeat, and the steps of the loop's
/// coarse phase.
const LOOP_SECONDS: f64 = 256.0;
const LOOP_STEPS: f64 = 4096.0;
/// Meters a second squared of gravity.
const GRAVITY: f64 = 9.80665;
/// Mip levels of the moments, down to one texel.
const LEVELS: u32 = WAVE_TEXELS.trailing_zeros() + 1;

/// What the spectra are drawn for: the wind's velocity (meters a second
/// along x and z) and the meters across the largest ripples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct WaveShape {
    pub(crate) wind: [f32; 2],
    pub(crate) size: f32,
}

pub(crate) struct WaterWaves {
    layout: crate::pipelines::GroupLayout,
    pipelines: Option<[wgpu::ComputePipeline; 3]>,
    uniform: wgpu::Buffer,
    spectrum: wgpu::Buffer,
    // The moments with all their levels as water samples them.
    sampled: wgpu::TextureView,
    sampler: wgpu::Sampler,
    // The rows' pass's group, the columns' and each level's.
    groups: Vec<wgpu::BindGroup>,
    shape: Option<WaveShape>,
    sizes: [f32; WAVE_TILES],
}

impl WaterWaves {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let layout = crate::pipelines::layout(device, &WATER_WAVES);
        let tiles = WAVE_TILES as u32;
        let usage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let side = (WAVE_TEXELS, WAVE_TEXELS);
        let format = wgpu::TextureFormat::Rgba16Float;
        let rows = crate::targets::texture_2d(
            device,
            "engine/water-wave-rows",
            side,
            tiles,
            1,
            format,
            usage,
        );
        let moments = crate::targets::texture_2d(
            device,
            "engine/water-waves",
            side,
            tiles,
            LEVELS,
            format,
            usage,
        );
        let spectrum = crate::buffers::storage(
            device,
            "engine/water-wave-spectrum",
            u64::from(WAVE_TEXELS * WAVE_TEXELS * tiles) * 16,
            wgpu::BufferUsages::COPY_DST,
        );
        let uniform =
            crate::buffers::uniform(device, "engine/water-waves", size_of::<WaveFrame>() as u64);
        let array = |texture: &wgpu::Texture, levels: std::ops::Range<u32>| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_mip_level: levels.start,
                mip_level_count: Some(levels.len() as u32),
                ..Default::default()
            })
        };
        // The waves repeat across their tiles; their levels and anisotropy
        // keep grazing water from shimmering.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/water-wave-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });
        let group = |label, source: &wgpu::TextureView, target: &wgpu::TextureView| {
            crate::pipelines::group(
                device,
                label,
                &WATER_WAVES,
                &layout,
                &[
                    crate::buffers::entry("waves", &uniform),
                    crate::buffers::entry("wave_spectrum", &spectrum),
                    ("wave_source", crate::pipelines::view(source)),
                    ("wave_target", crate::pipelines::view(target)),
                ],
            )
        };
        let all_rows = array(&rows, 0..1);
        let level = |index: u32| array(&moments, index..index + 1);
        // The rows' pass reads nothing: a level it does not write stands in.
        let mut groups = vec![
            group("engine/water-wave-rows", &level(1), &all_rows),
            group("engine/water-wave-columns", &all_rows, &level(0)),
        ];
        groups.extend(
            (1..LEVELS)
                .map(|index| group("engine/water-wave-level", &level(index - 1), &level(index))),
        );
        Self {
            layout,
            pipelines: None,
            uniform,
            spectrum,
            sampled: array(&moments, 0..LEVELS),
            sampler,
            groups,
            shape: None,
            sizes: [0.0; WAVE_TILES],
        }
    }

    /// The moments with their levels and the sampler repeating them, as
    /// water reads them.
    pub(crate) fn sampled(&self) -> (&wgpu::TextureView, &wgpu::Sampler) {
        (&self.sampled, &self.sampler)
    }

    /// Per meter, how many times each tile repeats (the inverse of its
    /// meters), as water's uniform holds them.
    pub(crate) fn repeats(&self) -> [f32; 4] {
        let [first, second, third] = self.sizes.map(|size| 1.0 / max(size, 1.0e-3));
        [first, second, third, 0.0]
    }

    /// Bytes of the moments with their levels, the rows' transforms and the
    /// spectra.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let texels = u64::from(WAVE_TEXELS * WAVE_TEXELS) * WAVE_TILES as u64;
        // Eight bytes a texel; the levels add a third.
        texels * 8 * 4 / 3 + texels * 8 + self.spectrum.size()
    }

    /// Prepares the waves of `shape` at simulation time `seconds` for this
    /// frame's encoding, drawing their spectra again when the shape changed
    /// and creating the pipelines the first time.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: WaveShape,
        seconds: f64,
    ) {
        if self.pipelines.is_none() {
            self.pipelines = Some(pipelines(device, &self.layout));
        }
        if self.shape != Some(shape) {
            self.shape = Some(shape);
            let size = f64::from(max(shape.size, 0.1));
            let sizes = TILE_SHARES.map(|share| size * share);
            self.sizes = sizes.map(|size| size as f32);
            let wind = shape.wind.map(f64::from);
            queue.write_buffer(
                &self.spectrum,
                0,
                bytemuck::cast_slice(&spectra(sizes, wind)),
            );
        }
        let loops = if seconds.is_finite() {
            seconds / LOOP_SECONDS
        } else {
            0.0
        };
        let fraction = loops - loops.floor();
        let coarse = (fraction * LOOP_STEPS).floor() / LOOP_STEPS;
        let frame = WaveFrame {
            tiles: self
                .sizes
                .map(|size| [(TAU / f64::from(size)) as f32, 0.0, 0.0, 0.0]),
            clock: [
                coarse as f32,
                (fraction - coarse) as f32,
                (LOOP_SECONDS / TAU) as f32,
                0.0,
            ],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&frame));
    }

    /// Evolves the spectra prepared into the moments and their levels,
    /// timed by `profiler` (`water_waves`).
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let Some([rows, columns, levels]) = &self.pipelines else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/water-waves"),
            timestamp_writes: profiler.span("water_waves").map(|span| span.compute()),
        });
        let tiles = WAVE_TILES as u32;
        pass.set_pipeline(rows);
        pass.set_bind_group(0, &self.groups[0], &[]);
        pass.dispatch_workgroups(WAVE_TEXELS, tiles, 1);
        pass.set_pipeline(columns);
        pass.set_bind_group(0, &self.groups[1], &[]);
        pass.dispatch_workgroups(WAVE_TEXELS, tiles, 1);
        pass.set_pipeline(levels);
        for (index, group) in self.groups[2..].iter().enumerate() {
            let side = (WAVE_TEXELS >> (index + 1))
                .max(1)
                .div_ceil(WAVE_LEVEL_GROUP);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(side, side, tiles);
        }
    }
}

/// The tiles' spectra, tiles of `sizes` meters (longest first) under wind
/// `wind` (meters a second along x and z): at each texel of each tile, row
/// by row, its wave number's amplitude and its opposite's (complex). Each
/// amplitude is fixed noise (two normal deviates) times the square root of
/// half Phillips's spectrum there, where the wave number lies in the tile's
/// band, scaled so the slopes' variance over all tiles is [`FULL_SLOPE`]
/// squared. Wave numbers at a grid's edge, which have no opposite there,
/// hold none.
fn spectra(sizes: [f64; WAVE_TILES], wind: [f64; 2]) -> Vec<[f32; 4]> {
    let n = WAVE_TEXELS as usize;
    let blown = wind[0].hypot(wind[1]);
    let speed = blown.max(CALMEST);
    let heading = if blown > 1.0e-3 {
        [wind[0] / blown, wind[1] / blown]
    } else {
        [1.0, 0.0]
    };
    // The longest waves the wind raises.
    let longest = speed * speed / GRAVITY;
    let mut amplitudes = vec![[0.0_f64; 2]; n * n * WAVE_TILES];
    let mut slopes = 0.0;
    for (tile, &size) in sizes.iter().enumerate() {
        let step = TAU / size;
        let band = sizes
            .get(tile + 1)
            .map_or(f64::INFINITY, |next| BAND_END * TAU / next);
        let first = if tile == 0 { 0.0 } else { BAND_END * step };
        for z in 1..n {
            for x in 1..n {
                let kx = (x as f64 - (n / 2) as f64) * step;
                let kz = (z as f64 - (n / 2) as f64) * step;
                let squared = kx * kx + kz * kz;
                let k = squared.sqrt();
                if k <= 0.0 || k < first || k >= band {
                    continue;
                }
                let along = (kx * heading[0] + kz * heading[1]) / k;
                let directional = along * along * if along < 0.0 { AGAINST } else { 1.0 };
                let power = (-1.0 / (squared * longest * longest)).exp() / (squared * squared)
                    * directional
                    * (-squared * SHORTEST * SHORTEST).exp();
                let [a, b] = normal_pair(tile as u32, x as u32, z as u32);
                let scale = (power * 0.5).sqrt();
                amplitudes[(tile * n + z) * n + x] = [a * scale, b * scale];
                slopes += squared * (a * a + b * b) * power * 0.5;
            }
        }
    }
    // Each wave number and its opposite both carry slopes.
    let gain = if slopes > 0.0 {
        FULL_SLOPE / (2.0 * slopes).sqrt()
    } else {
        0.0
    };
    (0..n * n * WAVE_TILES)
        .map(|index| {
            let (tile, x, z) = (index / (n * n), index % n, index / n % n);
            let opposite = if x == 0 || z == 0 {
                [0.0; 2]
            } else {
                amplitudes[(tile * n + n - z) * n + n - x]
            };
            let own = amplitudes[index];
            [own[0], own[1], opposite[0], opposite[1]].map(|value| (value * gain) as f32)
        })
        .collect()
}

/// Two independent standard normal deviates for texel `(x, z)` of tile
/// `tile`: hashed uniform numbers through Box and Muller's transform, the
/// same on every target.
fn normal_pair(tile: u32, x: u32, z: u32) -> [f64; 2] {
    let first = hash(tile.wrapping_mul(0x9e37_79b9) ^ hash(x ^ hash(z)));
    let second = hash(first ^ 0x85eb_ca6b);
    let unit = |bits: u32| (f64::from(bits >> 8) + 0.5) / f64::from(1_u32 << 24);
    let radius = (-2.0 * unit(first).ln()).sqrt();
    let angle = TAU * unit(second);
    [radius * angle.cos(), radius * angle.sin()]
}

/// A 32-bit integer hash (Wellons's lowbias32).
fn hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    value
}

fn pipelines(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
) -> [wgpu::ComputePipeline; 3] {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/water_waves.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::WATER_WAVES_SHADER.into()),
    });
    let pipeline_layout = crate::pipelines::pipeline_layout(
        device,
        &crate::pipeline_layouts::WATER_WAVE_PIPELINES,
        &[layout],
    );
    [
        water_waves::evolve_rows,
        water_waves::transform_columns,
        water_waves::average_level,
    ]
    .map(|entry| crate::pipelines::compute_entry(device, &pipeline_layout, &module, entry))
}

const _: () = assert!(
    WAVE_LINE_GROUP * 2 == WAVE_TEXELS,
    "an invocation takes a butterfly of a line"
);
