//! Depth pyramids: the farthest depth of each 2x2 block of the depth buffer,
//! then of each 2x2 block of the level below, which GPU presentation's
//! occlusion test (decision 040) compares candidates' projected bounds with
//! after phase one draws; or the nearest depths, which screen-space
//! reflections (decision 025) walk after the opaque pass.
//!
//! Levels reduce in chunks, as AMD's single-pass downsampler does: one
//! dispatch writes up to six levels, each workgroup reducing a 64x64 tile of
//! the chunk's first level, the first three levels in registers and the
//! rest in workgroup memory. Devices binding fewer storage textures write
//! shorter chunks. Every texel keeps the extreme of the same block, so the
//! levels hold the same values as reducing one level at a time.

use std::fmt::Write;

/// Which depth each pyramid texel keeps of the block it covers.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reduction {
    /// The farthest, for occlusion culling.
    Farthest,
    /// The nearest, for reflection rays.
    Nearest,
}

/// Levels one chunk writes at most: a 64x64 tile of its first level down
/// to a 2x2 block.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
const CHUNK_LEVELS: u32 = 6;
/// First-level texels across a workgroup's tile.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
const TILE: u32 = 64;

/// The shader of a chunk writing `outputs` levels (1 to 6) of `reduction`
/// from the depth buffer (`from_depth`) or the level before them.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(crate) fn chunk_shader(reduction: Reduction, from_depth: bool, outputs: u32) -> String {
    debug_assert!((1..=CHUNK_LEVELS).contains(&outputs));
    // Reversed depth (decision 042): the farthest is the smallest. Depths
    // lie in [0, 1], so 2 and -1 lose to every depth.
    let (pick, identity) = match reduction {
        Reduction::Farthest => ("min", "2.0"),
        Reduction::Nearest => ("max", "-1.0"),
    };
    let (source, load) = if from_depth {
        ("texture_depth_2d", "textureLoad(source, min(p, last), 0)")
    } else {
        ("texture_2d<f32>", "textureLoad(source, min(p, last), 0).r")
    };
    let mut s = String::new();
    let _ = writeln!(s, "@group(0) @binding(0) var source: {source};");
    for level in 0..outputs {
        let _ = writeln!(
            s,
            "@group(0) @binding({}) var level{level}: texture_storage_2d<r32float, write>;",
            level + 1
        );
    }
    let _ = write!(
        s,
        "var<workgroup> tile: array<f32, 256>;

fn pick(a: f32, b: f32) -> f32 {{
    return {pick}(a, b);
}}

// The source texel at `p`, clamped to its last: texels beyond the depth
// buffer repeat its edge, which cannot change a block's extreme.
fn fetch(p: vec2<u32>) -> f32 {{
    let last = textureDimensions(source) - 1u;
    return {load};
}}

// The first level's texel `p`: the extreme of the source's 2x2 block.
fn first(p: vec2<u32>) -> f32 {{
    let base = p * 2u;
    return pick(pick(fetch(base), fetch(base + vec2<u32>(1u, 0u))),
        pick(fetch(base + vec2<u32>(0u, 1u)), fetch(base + vec2<u32>(1u, 1u))));
}}

@compute @workgroup_size(16, 16)
fn reduce(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {{
    // Each thread's 4x4 texels of the first level, 2x2 of the second and
    // one of the third.
    let origin = group.xy * 64u + local.xy * 4u;
    var third = {identity};
    for (var by = 0u; by < 2u; by++) {{
        for (var bx = 0u; bx < 2u; bx++) {{
            var second = {identity};
            for (var sy = 0u; sy < 2u; sy++) {{
                for (var sx = 0u; sx < 2u; sx++) {{
                    let p = origin + vec2<u32>(bx * 2u + sx, by * 2u + sy);
                    let value = first(p);
                    if all(p < textureDimensions(level0)) {{
                        textureStore(level0, p, vec4<f32>(value, 0.0, 0.0, 0.0));
                    }}
                    second = pick(second, value);
                }}
            }}
"
    );
    if outputs >= 2 {
        s += "            let q = origin / 2u + vec2<u32>(bx, by);
            if all(q < textureDimensions(level1)) {
                textureStore(level1, q, vec4<f32>(second, 0.0, 0.0, 0.0));
            }
";
    }
    s += "            third = pick(third, second);
        }
    }
";
    if outputs >= 3 {
        s += "    let r = group.xy * 16u + local.xy;
    if all(r < textureDimensions(level2)) {
        textureStore(level2, r, vec4<f32>(third, 0.0, 0.0, 0.0));
    }
";
    }
    // Further levels halve the workgroup's tile in workgroup memory: a
    // level's texel (x, y) is the extreme of the tile's (2x, 2y) block.
    let mut values = "third";
    for (level, (name, across)) in [(3, ("fourth", 8u32)), (4, ("fifth", 4)), (5, ("sixth", 2))] {
        if outputs <= level {
            break;
        }
        let stored = across * across * 4;
        if level > 3 {
            s += "    workgroupBarrier();\n";
        }
        let _ = write!(
            s,
            "    if lane < {stored}u {{
        tile[lane] = {values};
    }}
    workgroupBarrier();
    var {name} = {identity};
    if lane < {count}u {{
        let at = vec2<u32>(lane % {across}u, lane / {across}u);
        let i = at.y * {row}u + at.x * 2u;
        {name} = pick(pick(tile[i], tile[i + 1u]), pick(tile[i + {down}u], tile[i + {diagonal}u]));
        let p = group.xy * {across}u + at;
        if all(p < textureDimensions(level{level})) {{
            textureStore(level{level}, p, vec4<f32>({name}, 0.0, 0.0, 0.0));
        }}
    }}
",
            count = across * across,
            row = 4 * across,
            down = 2 * across,
            diagonal = 2 * across + 1,
        );
        values = name;
    }
    s += "}\n";
    s
}

#[cfg(target_arch = "wasm32")]
pub(crate) use gpu::*;

#[cfg(target_arch = "wasm32")]
mod gpu {
    use std::collections::HashMap;

    use super::*;
    use crate::gpu_data::mip_levels;

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

    /// One dispatch: its pipeline (from the depth buffer, levels written),
    /// group and workgroups across and down.
    struct Chunk {
        pipeline: (bool, u32),
        group: wgpu::BindGroup,
        workgroups: (u32, u32),
    }

    struct Levels {
        // Keeps the texture alive beside its views.
        _texture: wgpu::Texture,
        full: wgpu::TextureView,
        sizes: Vec<(u32, u32)>,
        chunks: Vec<Chunk>,
    }

    pub(crate) struct DepthPyramid {
        reduction: Reduction,
        pipelines: HashMap<(bool, u32), wgpu::ComputePipeline>,
        levels: Option<(u64, Levels)>,
        // Advances with every rebuilt texture, for the groups that bind it.
        generation: u64,
    }

    impl DepthPyramid {
        pub(crate) fn new(_device: &wgpu::Device, reduction: Reduction) -> Self {
            Self {
                reduction,
                pipelines: HashMap::new(),
                levels: None,
                generation: 0,
            }
        }

        /// Releases the levels; the next `prepare` recreates them.
        pub(crate) fn release(&mut self) {
            self.levels = None;
        }

        fn pipeline(&mut self, device: &wgpu::Device, key: (bool, u32)) {
            let reduction = self.reduction;
            self.pipelines.entry(key).or_insert_with(|| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("engine/depth-pyramid.wgsl"),
                    source: wgpu::ShaderSource::Wgsl(chunk_shader(reduction, key.0, key.1).into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("engine/depth-pyramid"),
                    layout: None,
                    module: &module,
                    entry_point: Some("reduce"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            });
        }

        /// Prepares the pyramid of the `size` depth buffer `depth` of
        /// generation `depth_generation`; returns its level count.
        pub(crate) fn prepare(
            &mut self,
            device: &wgpu::Device,
            depth: &wgpu::TextureView,
            depth_generation: u64,
            (width, height): (u32, u32),
        ) -> u32 {
            if let Some((built, levels)) = &self.levels
                && *built == depth_generation
            {
                return levels.sizes.len() as u32;
            }
            // Power-of-two levels halve exactly: an odd level's last row or
            // column would otherwise fall outside every texel of the next.
            // Texels beyond the depth buffer repeat its edge, which cannot
            // change a block's farthest or nearest depth.
            let base = (
                width.div_ceil(2).max(1).next_power_of_two(),
                height.div_ceil(2).max(1).next_power_of_two(),
            );
            let count = mip_levels(base.0, base.1);
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("engine/depth-pyramid"),
                size: wgpu::Extent3d {
                    width: base.0,
                    height: base.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let views: Vec<wgpu::TextureView> = (0..count)
                .map(|level| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        base_mip_level: level,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect();
            let sizes: Vec<(u32, u32)> = (0..count)
                .map(|level| ((base.0 >> level).max(1), (base.1 >> level).max(1)))
                .collect();
            // Every chunk binds its levels as storage textures.
            let longest = device
                .limits()
                .max_storage_textures_per_shader_stage
                .clamp(1, CHUNK_LEVELS);
            let mut chunks = Vec::new();
            let mut first = 0;
            while first < count {
                let outputs = (count - first).min(longest);
                let key = (first == 0, outputs);
                self.pipeline(device, key);
                let source = if first == 0 {
                    depth
                } else {
                    &views[first as usize - 1]
                };
                let mut entries = vec![wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                }];
                entries.extend((0..outputs).map(|level| wgpu::BindGroupEntry {
                    binding: level + 1,
                    resource: wgpu::BindingResource::TextureView(&views[(first + level) as usize]),
                }));
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("engine/depth-pyramid-chunk"),
                    layout: &self.pipelines[&key].get_bind_group_layout(0),
                    entries: &entries,
                });
                let (across, down) = sizes[first as usize];
                chunks.push(Chunk {
                    pipeline: key,
                    group,
                    workgroups: (across.div_ceil(TILE), down.div_ceil(TILE)),
                });
                first += outputs;
            }
            let full = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.levels = Some((
                depth_generation,
                Levels {
                    _texture: texture,
                    full,
                    sizes,
                    chunks,
                },
            ));
            self.generation += 1;
            count
        }

        /// Records every level's reduction, level zero from the depth
        /// buffer. With `timestamp` (a query set and index), the pass writes
        /// it when it begins.
        pub(crate) fn encode(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            timestamp: Option<(&wgpu::QuerySet, u32)>,
        ) {
            let Some((_, levels)) = &self.levels else {
                return;
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/depth-pyramid"),
                timestamp_writes: timestamp.map(|(query_set, index)| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(index),
                        end_of_pass_write_index: None,
                    }
                }),
            });
            for chunk in &levels.chunks {
                pass.set_pipeline(&self.pipelines[&chunk.pipeline]);
                pass.set_bind_group(0, &chunk.group, &[]);
                pass.dispatch_workgroups(chunk.workgroups.0, chunk.workgroups.1, 1);
            }
        }

        /// Every level, and its generation.
        pub(crate) fn view(&self) -> Option<(&wgpu::TextureView, u64)> {
            self.levels
                .as_ref()
                .map(|(_, levels)| (&levels.full, self.generation))
        }

        pub(crate) fn capacity_bytes(&self) -> u64 {
            self.levels.as_ref().map_or(0, |(_, levels)| {
                levels
                    .sizes
                    .iter()
                    .map(|&(width, height)| u64::from(width) * u64::from(height) * 4)
                    .sum()
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_shaders_are_valid_wgsl() {
        for reduction in [Reduction::Farthest, Reduction::Nearest] {
            for from_depth in [false, true] {
                for outputs in 1..=CHUNK_LEVELS {
                    let source = chunk_shader(reduction, from_depth, outputs);
                    let module = naga::front::wgsl::parse_str(&source)
                        .unwrap_or_else(|error| panic!("{error}\n{source}"));
                    naga::valid::Validator::new(
                        naga::valid::ValidationFlags::all(),
                        naga::valid::Capabilities::empty(),
                    )
                    .validate(&module)
                    .unwrap_or_else(|error| panic!("{error:?}\n{source}"));
                }
            }
        }
    }
}
