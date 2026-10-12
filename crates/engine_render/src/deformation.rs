//! Runtime deformation's map (`deformation.wgsl`): the tracks
//! frames press around the camera, kept on the GPU between frames. Each
//! frame's stamps press in dispatches of at most [`CHUNK`], a workgroup per
//! 8 by 8 texels of a stamp's reach, every texel taking the deepest of its
//! dispatch's stamps; then their slopes and blocks. The map moves with the
//! camera without copying (texels wrap), emptying the slots it leaves, and
//! refills by one factor for the whole map: values are kept in reference
//! units, which shading scales by what is left of a track since the
//! reference time, and which a rebase scales once that falls below half.

use crate::shaders::entry::deformation;
use engine_core::math::{max, min};
use engine_core::{DeformationMap, RIM_REACH, RenderDeformation};

use crate::bindings::DEFORMATION;
use crate::cached::Cached;
use crate::gpu_layouts::constants::DEFORMATION_BLOCK as BLOCK;
use crate::gpu_layouts::constants::DEFORMATION_GROUP;

/// Stamps one dispatch presses: every texel tries each of them.
const CHUNK: usize = 64;
/// Uniform offsets of the jobs, at the WebGPU alignment.
const SLOT: u64 = 256;
/// A rebase moves the reference time to now once less than this much of a
/// track is left since it.
const REBASE: f32 = 0.5;

use crate::gpu_layouts::deformation::StampRecord;

use crate::gpu_layouts::deformation::JobUniform;

/// A recorded job: its entry point, uniform and workgroups.
#[derive(Clone, Copy, Debug)]
struct Job {
    entry: Entry,
    uniform: JobUniform,
    groups: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Entry {
    Wipe,
    Scroll,
    Rebase,
    Press,
    Slopes,
}

/// The map's shading inputs for terrain and ground cover: the map's first
/// texel (world texel indices), texels and blocks along a side (0 while
/// off); texels per meter and what is left of a track since the reference
/// time.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct DeformationShading {
    pub grid: [i32; 4],
    pub scale: [f32; 4],
}

pub(crate) struct DeformationPass {
    layout: crate::pipelines::GroupLayout,
    pipelines: Option<[wgpu::ComputePipeline; 5]>,
    // The map's words, and the texels along a side they hold (0: a
    // placeholder).
    map: wgpu::Buffer,
    resolution: u32,
    generation: u64,
    // What the map holds: where it lies, the reference time and the refill
    // its values decay with.
    placed: Option<DeformationMap>,
    reference: f64,
    refill: f32,
    // Jobs and stamps recorded since the last encoding, in order.
    jobs: Vec<Job>,
    records: Vec<StampRecord>,
    stamps: crate::buffers::Growable,
    uniforms: wgpu::Buffer,
    buffers_generation: u64,
    group: Cached<(u64, u64), wgpu::BindGroup>,
    shading: DeformationShading,
    /// Stamps this frame pressed, and the texels along the map's side (0
    /// while off).
    pub(crate) pressed: u32,
    pub(crate) texels: u32,
}

impl DeformationPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: crate::pipelines::layout(device, &DEFORMATION),
            pipelines: None,
            map: map_buffer(device, 0),
            resolution: 0,
            generation: 0,
            placed: None,
            reference: 0.0,
            refill: 0.0,
            jobs: Vec::new(),
            records: Vec::new(),
            stamps: crate::buffers::Growable::new(
                device,
                "engine/deformation-stamps",
                size_of::<StampRecord>() as u64,
                wgpu::BufferUsages::empty(),
            ),
            uniforms: crate::buffers::uniform(device, "engine/deformation-jobs", SLOT),
            buffers_generation: 0,
            group: Cached::default(),
            shading: DeformationShading::default(),
            pressed: 0,
            texels: 0,
        }
    }

    /// The map's words, and a generation advancing whenever it is another
    /// buffer.
    pub(crate) fn map(&self) -> (&wgpu::Buffer, u64) {
        (&self.map, self.generation)
    }

    /// The map's shading inputs this frame.
    pub(crate) fn shading(&self) -> DeformationShading {
        self.shading
    }

    /// Bytes of the map.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.map.size()
    }

    /// Whether jobs wait for the next encoding, which runs or drops them.
    pub(crate) fn pending(&self) -> bool {
        !self.jobs.is_empty() || !self.records.is_empty()
    }

    /// Records this frame's jobs: emptying, moving or rebasing the map as
    /// `frame` lays it at simulation time `now`, then its stamps.
    pub(crate) fn prepare(&mut self, device: &wgpu::Device, frame: &RenderDeformation, now: f64) {
        self.pressed = 0;
        let Some(map) = frame.map.and_then(|map| fit(device, map)) else {
            self.jobs.clear();
            self.records.clear();
            self.placed = None;
            self.texels = 0;
            self.shading = DeformationShading::default();
            return;
        };
        let resized = self.resolution != map.resolution;
        if resized {
            // Jobs recorded for the last map would not fit the new one.
            self.jobs.clear();
            self.records.clear();
            self.map = map_buffer(device, map.resolution);
            self.resolution = map.resolution;
            self.generation += 1;
        }
        let blocks = map.resolution / BLOCK;
        let texel_groups = map.resolution.div_ceil(DEFORMATION_GROUP);
        let uniform = |first: [i32; 2], last: [i32; 2], factor: f32| JobUniform {
            first: [first[0], first[1], last[0], last[1]],
            counts: [map.resolution, blocks, 0, 0],
            scale: [map.texel, factor, 0.0, 0.0],
        };
        let whole = [texel_groups, texel_groups, 1];
        let reset = frame.reset
            || resized
            || self
                .placed
                .is_none_or(|last| last.texel != map.texel || last.resolution != map.resolution);
        if reset {
            // The new map starts empty, its reference time now.
            if !resized {
                self.jobs.push(Job {
                    entry: Entry::Wipe,
                    uniform: uniform(map.first, map.first, 1.0),
                    groups: whole,
                });
            }
            self.reference = now;
            self.refill = map.refill;
        } else {
            if let Some(last) = self.placed.filter(|last| last.first != map.first) {
                self.jobs.push(Job {
                    entry: Entry::Scroll,
                    uniform: uniform(map.first, last.first, 1.0),
                    groups: whole,
                });
            }
            // Tracks left refilling at the refill the map's values decay
            // with, until a new one or so little is left that precision
            // would suffer: then the values take the decay and the
            // reference time moves to now.
            let left = left(now - self.reference, self.refill);
            if map.refill != self.refill || left < REBASE {
                if left < 1.0 {
                    self.jobs.push(Job {
                        entry: Entry::Rebase,
                        uniform: uniform(map.first, map.first, left),
                        groups: whole,
                    });
                }
                self.reference = now;
                self.refill = map.refill;
            }
        }
        if self.refill <= 0.0 || now < self.reference {
            self.reference = now;
        }
        self.placed = Some(map);
        self.texels = map.resolution;
        self.shading = DeformationShading {
            grid: [
                map.first[0],
                map.first[1],
                map.resolution as i32,
                blocks as i32,
            ],
            scale: [
                1.0 / map.texel,
                left(now - self.reference, self.refill),
                map.rim,
                0.0,
            ],
        };
        self.record_stamps(frame, map, uniform(map.first, map.first, 1.0));
    }

    /// Records the frame's stamps within the map, in reference units:
    /// dispatches of similar reaches together, all pressing before their
    /// slopes.
    fn record_stamps(&mut self, frame: &RenderDeformation, map: DeformationMap, base: JobUniform) {
        let size = map.resolution as i32;
        let low = map.first;
        let high = [low[0] + size - 1, low[1] + size - 1];
        let mut records: Vec<(u32, StampRecord)> = Vec::new();
        for track in &frame.tracks {
            let reach = track.radius * RIM_REACH;
            let texel = |value: f32| (value / map.texel).floor() as i32;
            let bounds = [
                texel(min(track.from[0], track.to[0]) - reach).max(low[0]),
                texel(min(track.from[1], track.to[1]) - reach).max(low[1]),
                texel(max(track.from[0], track.to[0]) + reach).min(high[0]),
                texel(max(track.from[1], track.to[1]) + reach).min(high[1]),
            ];
            if bounds[0] > bounds[2] || bounds[1] > bounds[3] {
                continue;
            }
            // Its depth where the reference time's values would hold it.
            let scale = if self.refill > 0.0 {
                ((track.seconds - self.reference) / f64::from(self.refill))
                    .min(8.0)
                    .exp() as f32
            } else {
                1.0
            };
            let depth = track.depth * scale;
            let side = (bounds[2] - bounds[0]).max(bounds[3] - bounds[1]) as u32 + 1;
            records.push((
                side,
                StampRecord {
                    ends: [track.from[0], track.from[1], track.to[0], track.to[1]],
                    shape: [track.radius, depth, track.flatness, depth * map.rim],
                    bounds,
                },
            ));
        }
        if records.is_empty() {
            return;
        }
        self.pressed = records.len() as u32;
        records.sort_unstable_by_key(|(side, _)| *side);
        let mut slopes = Vec::new();
        for chunk in records.chunks(CHUNK) {
            let start = self.records.len() as u32;
            self.records.extend(chunk.iter().map(|(_, record)| *record));
            let widest = chunk.iter().map(|(side, _)| *side).max().unwrap_or(1);
            let mut uniform = base;
            uniform.counts[2] = start;
            uniform.counts[3] = chunk.len() as u32;
            self.jobs.push(Job {
                entry: Entry::Press,
                uniform,
                groups: [
                    widest.div_ceil(DEFORMATION_GROUP),
                    widest.div_ceil(DEFORMATION_GROUP),
                    chunk.len() as u32,
                ],
            });
            // Slopes reach a texel further each way.
            let wider = (widest + 2).div_ceil(DEFORMATION_GROUP);
            slopes.push(Job {
                entry: Entry::Slopes,
                uniform,
                groups: [wider, wider, chunk.len() as u32],
            });
        }
        self.jobs.extend(slopes);
    }

    /// Runs the jobs recorded since the last encoding, timed by `profiler`
    /// (`deformation`).
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        if self.jobs.is_empty() || self.resolution == 0 {
            self.jobs.clear();
            self.records.clear();
            return;
        }
        let record_bytes = (self.records.len().max(1) * size_of::<StampRecord>()) as u64;
        if self.stamps.grow(device, record_bytes) {
            self.buffers_generation += 1;
        }
        if !self.records.is_empty() {
            queue.write_buffer(&self.stamps, 0, bytemuck::cast_slice(&self.records));
        }
        let uniform_bytes = self.jobs.len() as u64 * SLOT;
        if uniform_bytes > self.uniforms.size() {
            self.uniforms = crate::buffers::uniform(
                device,
                "engine/deformation-jobs",
                uniform_bytes.next_power_of_two(),
            );
            self.buffers_generation += 1;
        }
        let mut bytes = vec![0_u8; uniform_bytes as usize];
        for (index, job) in self.jobs.iter().enumerate() {
            let at = index * SLOT as usize;
            bytes[at..at + size_of::<JobUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&job.uniform));
        }
        queue.write_buffer(&self.uniforms, 0, &bytes);
        let pipelines = self
            .pipelines
            .get_or_insert_with(|| pipelines(device, &self.layout));
        let key = (self.generation, self.buffers_generation);
        let (layout, map, stamps, uniforms) =
            (&self.layout, &self.map, &self.stamps, &self.uniforms);
        let group = self.group.get_or_insert_with(key, || {
            crate::pipelines::group(
                device,
                "engine/deformation",
                &DEFORMATION,
                layout,
                &[
                    (
                        "job",
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: uniforms,
                            offset: 0,
                            size: wgpu::BufferSize::new(size_of::<JobUniform>() as u64),
                        }),
                    ),
                    crate::buffers::entry("stamps", stamps),
                    crate::buffers::entry("map", map),
                ],
            )
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/deformation"),
                timestamp_writes: profiler.span("deformation").map(|span| span.compute()),
            });
            for (index, job) in self.jobs.iter().enumerate() {
                let pipeline = &pipelines[job.entry as usize];
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[(index as u64 * SLOT) as u32]);
                pass.dispatch_workgroups(job.groups[0], job.groups[1], job.groups[2]);
            }
        }
        self.jobs.clear();
        self.records.clear();
    }
}

/// What is left of a track `age` seconds after it was pressed, refilling
/// over `refill` seconds (0, never).
fn left(age: f64, refill: f32) -> f32 {
    if refill > 0.0 && age > 0.0 {
        (-age / f64::from(refill)).exp() as f32
    } else {
        1.0
    }
}

/// Words of a map of `resolution` texels along a side: its blocks', its
/// tracks' and its slopes'.
fn map_words(resolution: u32) -> u64 {
    let blocks = u64::from(resolution / BLOCK);
    let texels = u64::from(resolution) * u64::from(resolution);
    blocks * blocks + 2 * texels
}

fn map_buffer(device: &wgpu::Device, resolution: u32) -> wgpu::Buffer {
    crate::buffers::storage(
        device,
        "engine/deformation-map",
        map_words(resolution) * 4,
        wgpu::BufferUsages::empty(),
    )
}

/// `map` with as many texels as the device's storage buffers hold: halved
/// (each texel twice as wide, the map as wide) while they do not, at most
/// three times; `None` beyond.
fn fit(device: &wgpu::Device, map: DeformationMap) -> Option<DeformationMap> {
    let limits = device.limits();
    let most = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let mut fitted = map;
    for _ in 0..4 {
        if map_words(fitted.resolution) * 4 <= most {
            return Some(fitted);
        }
        fitted = DeformationMap {
            first: [fitted.first[0] / 2, fitted.first[1] / 2],
            texel: fitted.texel * 2.0,
            resolution: fitted.resolution / 2,
            ..fitted
        };
    }
    None
}

fn pipelines(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
) -> [wgpu::ComputePipeline; 5] {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/deformation.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::DEFORMATION_SHADER.into()),
    });
    let pipeline_layout = crate::pipelines::pipeline_layout(
        device,
        &crate::pipeline_layouts::DEFORMATION_PIPELINES,
        &[layout],
    );
    // In `Entry`'s order.
    [
        deformation::wipe,
        deformation::scroll,
        deformation::rebase,
        deformation::press,
        deformation::slopes,
    ]
    .map(|entry| crate::pipelines::compute_entry(device, &pipeline_layout, &module, entry))
}
