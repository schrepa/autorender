//! Water's waves (`water_waves.wgsl`): [`WIND_TILES`] tiles of the wind's
//! waves that terrains' water and the sea shade with, each repeating at its
//! own period and holding its own band of wave lengths. Their spectra,
//! drawn from fixed noise for the wind's heading, are one of the scene's
//! static tables (`engine_core::table_data::waves` makes them, with each
//! tile's period, band, height and turn), which upload as they arrive. Each
//! frame whose terrains' water shows the wind's waves
//! (`engine_core::Water::wind_waves`) or whose sea draws, the spectra
//! evolve (gravity and surface tension set each wave's speed; frequencies
//! are rounded so the waves loop every [`LOOP_SECONDS`]) and an inverse
//! transform turns them into the slopes and heights of the
//! surface, whose moments fill a texture array with its mip levels (LEAN
//! mapping, Olano and Baker 2010): each tile's slopes and their squares in
//! one layer, and their product, its height and the height's square in
//! another. Water reads them where it stands through the pixel's own
//! footprint, each tile at its point turned as the tile's lattice is and
//! its slopes turned back: the levels' averages give the mean of the slopes
//! the pixel spans, and what they lose of the second moments the covariance
//! of those it averages away, the roughness of what a pixel spans
//! (`meshes/water_spectrum.wgsl`). Each tile's slopes' covariance over its
//! whole repeat, which the last of its levels holds, rides water's uniform
//! for the tiles a pixel reads none of.
//!
//! Each frame whose terrains' water shows the wind's waves, the wind's tiles
//! also transform how much their waves, choppy (Tessendorf 2001), compress
//! the surface at unit gain, into their whitecaps: each texel keeps the
//! largest compression it has had, fading by e every
//! [`WATER_WHITECAP_FADE`] seconds of the waves' own clock, so a paused or
//! repeated frame repeats, in a texture array of its own whose levels
//! average as the moments' do. The rows' transforms serve both, and each
//! such frame fades a copy of the whitecaps they last left, as no pass
//! reads and writes one texture. Terrains' water reads them where it reads
//! the slopes (`meshes/water_spectrum.wgsl`'s `water_wind_whitecaps`) and
//! whitens where the sum of its gains times them passes 1, where its waves
//! have folded (`meshes/water_foam.wgsl`'s `water_folding`); the sea reads
//! none of them.

use std::cell::Cell;
use std::f64::consts::TAU;

use engine_core::table_data::waves::WaveTile;

use crate::bindings::WATER_WAVES;
use crate::gpu_layouts::constants::{
    MATERIAL_ANISOTROPY, WAVE_LEVEL_GROUP, WAVE_LINE_GROUP, WAVE_TEXELS, WIND_TILES,
};
use crate::gpu_layouts::water::WaveFrame;
use crate::shaders::entry::water_waves;

/// Seconds after which the waves repeat, and the steps of the loop's
/// coarse phase.
const LOOP_SECONDS: f64 = 256.0;
const LOOP_STEPS: f64 = 4096.0;
/// Mip levels of the moments and the whitecaps, down to one texel.
const LEVELS: u32 = WAVE_TEXELS.trailing_zeros() + 1;
/// Layers of the moments a tile takes: its slopes and their squares, and
/// their product, its height and the height's square, a layer each, the
/// second kind after every tile's first (`water_waves.wgsl`).
const MOMENT_LAYERS: u32 = 2;
/// Seconds over which whitecaps' foam fades by e: the middle of the
/// area-weighted mean decay times, 1.4 to 4.8 s, Callaghan, Deane, Stokes
/// and Ward (2012, "Observed variation in the decay time of oceanic
/// whitecap foam") measured over 552 waves breaking at sea.
const WATER_WHITECAP_FADE: f64 = 3.0;

pub(crate) struct WaterWaves {
    layout: crate::pipelines::GroupLayout,
    pipelines: Option<[wgpu::ComputePipeline; 5]>,
    uniform: wgpu::Buffer,
    spectrum: wgpu::Buffer,
    // The moments with all their levels as water samples them.
    sampled: wgpu::TextureView,
    sampler: wgpu::Sampler,
    // The rows' pass's group, the columns' and each level's.
    groups: Vec<wgpu::BindGroup>,
    // The whitecaps, the copy of their first level each frame fades, and
    // the whitecaps with all their levels as water samples them.
    whitecaps: wgpu::Texture,
    before: wgpu::Texture,
    whitecaps_sampled: wgpu::TextureView,
    // The folds' rows' pass's group, their columns' and each of the
    // whitecaps' levels'.
    folds: Vec<wgpu::BindGroup>,
    // The spectra uploaded, their tiles, and each tile's slopes' covariance
    // over its whole repeat (`slope_spreads`): none before they arrive.
    uploaded: Option<u64>,
    tiles: [WaveTile; WIND_TILES],
    spreads: [[f32; 2]; WIND_TILES],
    // The waves' clock as this frame prepares it, and as the whitecaps
    // last faded to it: none before they first do, and once new spectra
    // arrive.
    seconds: f64,
    faded: Cell<Option<f64>>,
}

impl WaterWaves {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let layout = crate::pipelines::layout(device, &WATER_WAVES);
        let tiles = WIND_TILES as u32;
        let usage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let side = (WAVE_TEXELS, WAVE_TEXELS);
        let format = wgpu::TextureFormat::Rgba16Float;
        let texture = |label, layers, levels, usage| {
            crate::targets::texture_2d(device, label, side, layers, levels, format, usage)
        };
        let rows = texture("engine/water-wave-rows", tiles, 1, usage);
        let moments = texture("engine/water-waves", MOMENT_LAYERS * tiles, LEVELS, usage);
        // The wind's tiles' whitecaps, the first level of which each frame
        // copies for the next to fade.
        let whitecaps = texture(
            "engine/water-whitecaps",
            tiles,
            LEVELS,
            usage | wgpu::TextureUsages::COPY_SRC,
        );
        let before = texture(
            "engine/water-whitecaps-before",
            tiles,
            1,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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
        // keep grazing water from shimmering, filtering as much anisotropy as
        // every read filtering a footprint resolves it by
        // (`MATERIAL_ANISOTROPY`, `scene.wgsl`'s `detail_resolved`), so far
        // water keeps across the view what the size it resolves spans.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/water-wave-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: MATERIAL_ANISOTROPY,
            ..Default::default()
        });
        // The folds' columns' pass reads the whitecaps the last frame left;
        // they stand in for the other passes, as nothing in the pass writes
        // them.
        let before_view = array(&before, 0..1);
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
                    ("wave_before", crate::pipelines::view(&before_view)),
                ],
            )
        };
        let all_rows = array(&rows, 0..1);
        let level = |index: u32| array(&moments, index..index + 1);
        let caps = |index: u32| array(&whitecaps, index..index + 1);
        // The rows' passes read nothing: a level they do not write stands in.
        let mut groups = vec![
            group("engine/water-wave-rows", &level(1), &all_rows),
            group("engine/water-wave-columns", &all_rows, &level(0)),
        ];
        groups.extend(
            (1..LEVELS)
                .map(|index| group("engine/water-wave-level", &level(index - 1), &level(index))),
        );
        // The folds take the rows' transforms once the moments' columns are
        // done with them.
        let mut folds = vec![
            group("engine/water-wave-fold-rows", &level(1), &all_rows),
            group("engine/water-wave-fold-columns", &all_rows, &caps(0)),
        ];
        folds.extend((1..LEVELS).map(|index| {
            group(
                "engine/water-whitecap-level",
                &caps(index - 1),
                &caps(index),
            )
        }));
        Self {
            layout,
            pipelines: None,
            uniform,
            spectrum,
            sampled: array(&moments, 0..LEVELS),
            sampler,
            groups,
            whitecaps_sampled: array(&whitecaps, 0..LEVELS),
            whitecaps,
            before,
            folds,
            uploaded: None,
            tiles: [WaveTile::default(); WIND_TILES],
            spreads: [[0.0; 2]; WIND_TILES],
            seconds: 0.0,
            faded: Cell::new(None),
        }
    }

    /// The moments with their levels and the sampler repeating them, as
    /// water reads them.
    pub(crate) fn sampled(&self) -> (&wgpu::TextureView, &wgpu::Sampler) {
        (&self.sampled, &self.sampler)
    }

    /// The wind's tiles' whitecaps with their levels, as water reads them
    /// with the moments' sampler.
    pub(crate) fn whitecaps(&self) -> &wgpu::TextureView {
        &self.whitecaps_sampled
    }

    /// Each tile as water's uniform holds it (`waves` and `turns`): how many
    /// times a meter it repeats (the inverse of its period), the longest and
    /// the shortest waves it holds (meters) and the root mean square height
    /// of its waves at unit slope variance (meters); and the cosine and sine
    /// of the angle its lattice turns by from the world's axes, from +x
    /// toward +z (xy), by which water turns each world point it reads the
    /// tile at, and the anisotropic part of its slopes' covariance at unit
    /// slope variance over its whole repeat, in its own axes, as the last of
    /// its levels holds it (zw, `slope_spreads`). Before spectra arrive, the
    /// tiles hold no waves, turn by none and spread alike every way.
    pub(crate) fn tiles(&self) -> ([[f32; 4]; WIND_TILES], [[f32; 4]; WIND_TILES]) {
        let waves = self.tiles.map(|tile| {
            let repeats = if tile.period > 0.0 {
                1.0 / tile.period
            } else {
                0.0
            };
            [repeats, tile.longest, tile.shortest, tile.height]
        });
        let turns = std::array::from_fn(|index| {
            let (sin, cos) = engine_core::math::sin_cos(self.tiles[index].turn);
            let [drawn_out, skewed] = self.spreads[index];
            [cos, sin, drawn_out, skewed]
        });
        (waves, turns)
    }

    /// Bytes of the moments with their levels, the rows' transforms and the
    /// spectra; and of the whitecaps with their levels and the copy each
    /// frame fades.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let texels = u64::from(WAVE_TEXELS * WAVE_TEXELS) * WIND_TILES as u64;
        // Eight bytes a texel, a layer a tile; the levels add a third. The
        // moments take MOMENT_LAYERS layers a tile with their levels, the
        // whitecaps one with theirs, and the rows and the whitecaps' copy one
        // each.
        let levels = texels * 8 * 4 / 3;
        (u64::from(MOMENT_LAYERS) + 1) * levels + 2 * texels * 8 + self.spectrum.size()
    }

    /// Prepares the waves at simulation time `seconds` for this frame's
    /// encoding, uploading `spectra` (the latest the scene's tables hold)
    /// unless uploaded and creating the pipelines the first time. Until
    /// spectra arrive, the surface stands still. The whitecaps fade by the
    /// seconds since they last did, and start anew with new spectra, whose
    /// waves folded none of them, or where the clock runs back.
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
            self.spreads = slope_spreads(&spectra.table.texels);
            self.faded.set(None);
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
        // The share of the whitecaps the last frame left that this one
        // keeps: e^(−Δt/τ), all of them while the clock stands still, none
        // where it ran back or they start anew.
        let keeps = match self.faded.get() {
            Some(faded) if seconds >= faded => (-(seconds - faded) / WATER_WHITECAP_FADE).exp(),
            _ => 0.0,
        };
        self.seconds = seconds;
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
                keeps as f32,
            ],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&frame));
    }

    /// Evolves the wind's spectra prepared into the moments and their
    /// levels, where `moments`; then, where `whitecaps`, the wind's tiles'
    /// compression into their whitecaps, fading those they last left, and
    /// their levels; timed by `profiler` (`water_waves`). Neither runs where
    /// nothing reads it, and the whitecaps fade from where they last did
    /// when they run again.
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
        [moments, whitecaps]: [bool; 2],
    ) {
        let Some([rows, columns, levels, fold_rows, fold_columns]) = &self.pipelines else {
            return;
        };
        if !moments && !whitecaps {
            return;
        }
        if whitecaps {
            encoder.copy_texture_to_texture(
                self.whitecaps.as_image_copy(),
                self.before.as_image_copy(),
                self.before.size(),
            );
            self.faded.set(Some(self.seconds));
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/water-waves"),
            timestamp_writes: profiler.span("water_waves").map(|span| span.compute()),
        });
        let tiles = WIND_TILES as u32;
        if moments {
            transform(
                &mut pass,
                [rows, columns, levels],
                &self.groups,
                (tiles, MOMENT_LAYERS * tiles),
            );
        }
        if whitecaps {
            transform(
                &mut pass,
                [fold_rows, fold_columns, levels],
                &self.folds,
                (tiles, tiles),
            );
        }
    }
}

/// Dispatches into `pass` one transform of `tiles` tiles into `layers`
/// layers: its rows' pass (`rows`), its columns' (`columns`) and each of
/// its levels' (`levels`), with `groups` in that order.
fn transform(
    pass: &mut wgpu::ComputePass<'_>,
    [rows, columns, levels]: [&wgpu::ComputePipeline; 3],
    groups: &[wgpu::BindGroup],
    (tiles, layers): (u32, u32),
) {
    pass.set_pipeline(rows);
    pass.set_bind_group(0, &groups[0], &[]);
    pass.dispatch_workgroups(WAVE_TEXELS, tiles, 1);
    pass.set_pipeline(columns);
    pass.set_bind_group(0, &groups[1], &[]);
    pass.dispatch_workgroups(WAVE_TEXELS, tiles, 1);
    pass.set_pipeline(levels);
    for (index, group) in groups[2..].iter().enumerate() {
        let side = (WAVE_TEXELS >> (index + 1))
            .max(1)
            .div_ceil(WAVE_LEVEL_GROUP);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(side, side, layers);
    }
}

/// Each tile's slopes' covariance at unit slope variance over its whole
/// repeat, in its own axes, from the spectra `texels` (at each texel of each
/// tile, row by row, its wave number's amplitude and its opposite's, as the
/// storage buffer holds them): half the difference of their variances along
/// x and z, and their covariance, each over their whole variance. Each wave
/// number's slopes vary by its power, its amplitude's and its opposite's
/// squares, times its outer product; the cross terms of a wave and its
/// opposite come and go as the two run apart, and average away. A tile
/// holding no waves spreads alike every way.
fn slope_spreads(texels: &[[f32; 4]]) -> [[f32; 2]; WIND_TILES] {
    let side = WAVE_TEXELS as usize;
    let middle = (side / 2) as f64;
    std::array::from_fn(|tile| {
        // The sums of the power times kx², kz² and kx kz, in waves a period.
        let mut sums = [0.0_f64; 3];
        let first = tile * side * side;
        for (index, texel) in texels.iter().skip(first).take(side * side).enumerate() {
            let power: f64 = texel
                .iter()
                .map(|&value| f64::from(value) * f64::from(value))
                .sum();
            let along_x = (index % side) as f64 - middle;
            let along_z = (index / side) as f64 - middle;
            sums[0] += along_x * along_x * power;
            sums[1] += along_z * along_z * power;
            sums[2] += along_x * along_z * power;
        }
        let whole = sums[0] + sums[1];
        if whole > 0.0 {
            [
                (0.5 * (sums[0] - sums[1]) / whole) as f32,
                (sums[2] / whole) as f32,
            ]
        } else {
            [0.0; 2]
        }
    })
}

fn pipelines(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
) -> [wgpu::ComputePipeline; 5] {
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
        water_waves::evolve_fold_rows,
        water_waves::transform_fold_columns,
    ]
    .map(|entry| crate::pipelines::compute_entry(device, &pipeline_layout, &module, entry))
}

const _: () = assert!(
    WAVE_LINE_GROUP * 2 == WAVE_TEXELS,
    "an invocation takes a butterfly of a line"
);
