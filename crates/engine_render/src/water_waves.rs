//! Water's waves (`water_waves.wgsl`): [`WAVE_TILES`] tiles of waves that
//! terrains' water and the sea shade with, each repeating at its own
//! period: [`WIND_TILES`] of the wind's, each holding its own band of wave
//! lengths, then one of the current's turbulence. Their spectra, drawn from
//! fixed noise for the wind's heading, are one of the scene's static tables
//! (`engine_core::table_data::waves` makes them, with each tile's period,
//! band, height and turn), which upload as they arrive. Each frame the wind's
//! spectra evolve (gravity and surface tension set each wave's speed;
//! frequencies are rounded so the waves loop every [`LOOP_SECONDS`]) and an
//! inverse transform turns them into the slopes and heights of the surface,
//! whose moments (the slopes, the square of their length and the height)
//! fill a texture array with its mip levels. The turbulence's tile stands
//! still, its waves moving only as the current carries them, so it is
//! transformed once its spectrum arrives. Water reads the slopes where it
//! stands, each wind tile at its point turned as the tile's lattice is and
//! its slopes turned back, and, from what the levels' averages lose of the
//! square, the roughness of what a pixel spans
//! (`meshes/water_spectrum.wgsl`).

use std::cell::Cell;
use std::f64::consts::TAU;

use engine_core::table_data::waves::WaveTile;

use crate::bindings::WATER_WAVES;
use crate::gpu_layouts::constants::{
    TURBULENCE_TILE, WAVE_ANISOTROPY, WAVE_LEVEL_GROUP, WAVE_LINE_GROUP, WAVE_TEXELS, WAVE_TILES,
    WIND_TILES,
};
use crate::gpu_layouts::water::WaveFrame;
use crate::shaders::entry::water_waves;

/// Seconds after which the waves repeat, and the steps of the loop's
/// coarse phase.
const LOOP_SECONDS: f64 = 256.0;
const LOOP_STEPS: f64 = 4096.0;
/// Mip levels of the moments, down to one texel.
const LEVELS: u32 = WAVE_TEXELS.trailing_zeros() + 1;

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
    // The spectra uploaded and their tiles, and whether the turbulence's
    // tile has yet to be transformed since.
    uploaded: Option<u64>,
    tiles: [WaveTile; WAVE_TILES],
    still: Cell<bool>,
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
            anisotropy_clamp: WAVE_ANISOTROPY,
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
            uploaded: None,
            tiles: [WaveTile::default(); WAVE_TILES],
            still: Cell::new(false),
        }
    }

    /// The moments with their levels and the sampler repeating them, as
    /// water reads them.
    pub(crate) fn sampled(&self) -> (&wgpu::TextureView, &wgpu::Sampler) {
        (&self.sampled, &self.sampler)
    }

    /// Each tile as water's uniform holds it (`waves` and `turns`): how many
    /// times a meter it repeats (the inverse of its period), the longest and
    /// the shortest waves it holds (meters) and the root mean square height
    /// of its waves at unit slope variance (meters); and the cosine and sine
    /// of the angle its lattice turns by from the world's axes, from +x
    /// toward +z (xy), by which water turns each world point it reads the
    /// tile at. Before spectra arrive, the tiles hold no waves and turn by
    /// none.
    pub(crate) fn tiles(&self) -> ([[f32; 4]; WAVE_TILES], [[f32; 4]; WAVE_TILES]) {
        let waves = self.tiles.map(|tile| {
            let repeats = if tile.period > 0.0 {
                1.0 / tile.period
            } else {
                0.0
            };
            [repeats, tile.longest, tile.shortest, tile.height]
        });
        let turns = self.tiles.map(|tile| {
            let (sin, cos) = engine_core::math::sin_cos(tile.turn);
            [cos, sin, 0.0, 0.0]
        });
        (waves, turns)
    }

    /// Bytes of the moments with their levels, the rows' transforms and the
    /// spectra.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let texels = u64::from(WAVE_TEXELS * WAVE_TEXELS) * WAVE_TILES as u64;
        // Eight bytes a texel; the levels add a third.
        texels * 8 * 4 / 3 + texels * 8 + self.spectrum.size()
    }

    /// Prepares the waves at simulation time `seconds` for this frame's
    /// encoding, uploading `spectra` (the latest the scene's tables hold)
    /// unless uploaded and creating the pipelines the first time. Until
    /// spectra arrive, the surface stands still.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spectra: Option<&engine_core::StaticTable<engine_core::table_data::waves::WaveSpectra>>,
        seconds: f64,
    ) {
        if self.pipelines.is_none() {
            self.pipelines = Some(pipelines(device, &self.layout));
        }
        if let Some(spectra) = spectra
            && self.uploaded != Some(spectra.id)
        {
            self.uploaded = Some(spectra.id);
            self.tiles = spectra.table.tiles;
            self.still.set(true);
            queue.write_buffer(
                &self.spectrum,
                0,
                bytemuck::cast_slice(&spectra.table.texels),
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
            tiles: self.tiles.map(|tile| {
                let per_texel = if tile.period > 0.0 {
                    (TAU / f64::from(tile.period)) as f32
                } else {
                    0.0
                };
                let heights = if tile.height > 0.0 {
                    1.0 / tile.height
                } else {
                    0.0
                };
                [per_texel, heights, 0.0, 0.0]
            }),
            clock: [
                coarse as f32,
                (fraction - coarse) as f32,
                (LOOP_SECONDS / TAU) as f32,
                0.0,
            ],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&frame));
    }

    /// Evolves the wind's spectra prepared into the moments and their
    /// levels, and the turbulence's the first time after it arrives, timed
    /// by `profiler` (`water_waves`).
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
        // The turbulence's tile, the last, keeps the moments it was
        // transformed to.
        let tiles = if self.still.replace(false) {
            WAVE_TILES as u32
        } else {
            WIND_TILES
        };
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
const _: () = assert!(
    TURBULENCE_TILE == WIND_TILES && WAVE_TILES == WIND_TILES as usize + 1,
    "the turbulence's tile follows the wind's, the last"
);
