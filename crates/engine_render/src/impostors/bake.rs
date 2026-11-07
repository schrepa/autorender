use super::*;

use engine_core::{Entry, Error, Var};

use crate::install::{Bound, Program, RenderDecl, Scoped};
use crate::surface::{DrawState, MaterialVariant};

engine_core::gpu_struct! {
    pub struct ImpostorBake {
        pub sphere: [f32; 4],
        pub flags: [u32; 4],
        pub tile: [u32; 4],
    }
}

const BAKE_SAMPLES: u32 = 4;
const BAKE_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const BAKE_GROUP: u32 = 2;
const SHADE_SCALE: u32 = 4;
const SHADE_MOST: u32 = 2048;
const TILE_TEXELS: u32 = 1024;

pub(super) struct BakeTargets {
    shade: wgpu::TextureView,
    colors: [wgpu::TextureView; 4],
    depth: wgpu::TextureView,
    resolves: [wgpu::TextureView; 4],
    resolved: [wgpu::Texture; 4],
    pub(super) layer: [wgpu::Texture; 4],
    _textures: [wgpu::Texture; 6],
}

impl Drop for BakeTargets {
    fn drop(&mut self) {
        for texture in self
            ._textures
            .iter()
            .chain(&self.resolved)
            .chain(&self.layer)
        {
            crate::memory::release_any(texture);
        }
    }
}

fn tile_frames(shape: AtlasShape) -> u32 {
    (TILE_TEXELS / shape.texels).clamp(1, shape.frames)
}

fn slot(extended: bool, shade: bool, two_sided: bool) -> usize {
    usize::from(extended) | (usize::from(shade) << 1) | (usize::from(two_sided) << 2)
}

pub(crate) struct ImpostorBakes {
    pipelines: [u32; 8],
}

impl crate::GpuPart for ImpostorBakes {
    type Reads = ();
    const NAME: &'static str = "impostor_bake";

    fn install(cx: &mut crate::Install<'_, Self>) -> Result<Self, Error> {
        let module = crate::shaders::core_module(BAKE_MODULE);
        let mut pipelines = [0; 8];
        for (at, pipeline) in pipelines.iter_mut().enumerate() {
            let (extended, shade, two_sided) = (at & 1 != 0, at & 2 != 0, at & 4 != 0);
            let (vertex, fragment) = bake_entries(extended, shade);
            let mut bound = vec![(Var::generated(module, "impostor_bake"), Bound::Asset)];
            if !shade {
                bound.push((Var::generated(module, "impostor_shade"), Bound::Asset));
            }
            *pipeline = cx.core.pipeline(
                Scoped::Outside(Self::NAME),
                Program::Render(RenderDecl {
                    vertex: Entry::generated(module, vertex),
                    fragment: Some(Entry::generated(module, fragment)),
                    vertices: vec![crate::VertexLayout::Mesh; 1 + usize::from(extended)],
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        cull_mode: (!two_sided).then_some(wgpu::Face::Back),
                        ..Default::default()
                    },
                    depth: crate::DepthUse::Write {
                        bias: wgpu::DepthBiasState::default(),
                    },
                    blend: None,
                    writes: wgpu::ColorWrites::ALL,
                    targets: None,
                    samples: if shade { 1 } else { BAKE_SAMPLES },
                }),
                bound,
            )?;
        }
        Ok(Self { pipelines })
    }

    fn prepare(&mut self, (): (), _cx: &mut crate::Prepare<'_, Self>) {}
}

impl Atlas {
    pub(super) fn bake(
        &mut self,
        gpu: BakeGpu<'_, '_>,
        source: &BakeSource<'_>,
        identity: ImpostorIdentity,
        hemisphere: bool,
    ) {
        let BakeGpu {
            device,
            encoder,
            mips,
            compiler,
            readbacks,
        } = gpu;
        let encoder = mip_encoder(device, encoder);
        let shape = AtlasShape {
            hemisphere,
            ..self.shape
        };
        let targets = self
            .targets
            .get_or_insert_with(|| bake_targets(device, shape));
        draw_layer((device, &mut *encoder), compiler, targets, source, shape);
        for texture in &targets.layer {
            mips.record((device, &mut *encoder), compiler, texture, None);
        }
        self.read_back((device, encoder), readbacks, (identity, shape));
    }
}

type BakeDraw<'a> = (&'a MeshDraw<'a>, bool, &'a wgpu::BindGroup, BakeKey);

type BakeMade = (BakeKey, wgpu::RenderPipeline, wgpu::RenderPipeline);

fn bake_pipeline(
    compiler: &mut Compiler<'_>,
    pipelines: &[u32; 8],
    key: BakeKey,
    shade: bool,
) -> Option<wgpu::RenderPipeline> {
    let formats: Vec<Option<wgpu::TextureFormat>> = if shade {
        Vec::new()
    } else {
        FORMATS.map(Some).to_vec()
    };
    let variant = MaterialVariant::PLAIN.with_draw(DrawState {
        alpha_test: key.look.masked,
        ..DrawState::PLAIN
    });
    compiler
        .compile(
            pipelines[slot(key.extended, shade, key.look.two_sided)],
            (&formats, Some((BAKE_DEPTH, true))),
            (false, false),
            Some(variant),
        )
        .map(|(_, made)| made)
}

fn draw_layer(
    (device, encoder): (&wgpu::Device, &mut wgpu::CommandEncoder),
    compiler: &mut Compiler<'_>,
    targets: &BakeTargets,
    source: &BakeSource<'_>,
    shape: AtlasShape,
) {
    let draws: Vec<BakeDraw<'_>> =
        std::iter::once((&source.draw, source.extended, source.group, source.look))
            .chain(
                source
                    .parts
                    .iter()
                    .map(|part| (&part.draw, part.extended, part.group, part.look)),
            )
            .map(|(draw, extended, group, look)| (draw, extended, group, BakeKey { look, extended }))
            .collect();
    let Some(pipelines) = compiler
        .part::<ImpostorBakes>()
        .map(|bakes| bakes.pipelines)
    else {
        return;
    };
    let mut made: Vec<BakeMade> = Vec::new();
    for &(_, _, _, key) in &draws {
        if made.iter().any(|(known, _, _)| *known == key) {
            continue;
        }
        let shade = bake_pipeline(compiler, &pipelines, key, true);
        let bake = bake_pipeline(compiler, &pipelines, key, false);
        if let (Some(shade), Some(bake)) = (shade, bake) {
            made.push((key, shade, bake));
        }
    }
    let layouts = [true, false]
        .map(|shade| compiler.group_layout(pipelines[slot(false, shade, false)], BAKE_GROUP));
    let [Some(shade_layout), Some(bake_layout)] = layouts else {
        return;
    };
    let drawn = (made.as_slice(), draws.as_slice());
    let frames = shape.frames;
    {
        let group = bake_group(
            device,
            &shade_layout,
            (source, shape),
            [0, 0, frames, frames],
            None,
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/impostor-shade"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.shade,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(BAKE_GROUP, &group, &[]);
        draw_frames(&mut pass, drawn, true, frames * frames);
    }
    let side = tile_frames(shape) as usize;
    for row in (0..frames).step_by(side) {
        for column in (0..frames).step_by(side) {
            draw_tile(
                (device, &mut *encoder),
                (drawn, &bake_layout),
                targets,
                (source, shape),
                (column, row),
            );
        }
    }
}

fn draw_frames(
    pass: &mut wgpu::RenderPass<'_>,
    (made, draws): (&[BakeMade], &[BakeDraw<'_>]),
    shade: bool,
    frames: u32,
) {
    for &(mesh, extended, group, key) in draws {
        let Some((_, shading, baking)) = made.iter().find(|(known, _, _)| *known == key) else {
            continue;
        };
        pass.set_pipeline(if shade { shading } else { baking });
        pass.set_bind_group(1, group, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        if extended && let Some(extras) = mesh.extras {
            pass.set_vertex_buffer(1, extras.slice(..));
        }
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(
            mesh.first_index..mesh.first_index + mesh.index_count,
            mesh.base_vertex,
            0..frames,
        );
    }
}

fn draw_tile(
    (device, encoder): (&wgpu::Device, &mut wgpu::CommandEncoder),
    (drawn, layout): ((&[BakeMade], &[BakeDraw<'_>]), &wgpu::BindGroupLayout),
    targets: &BakeTargets,
    (source, shape): (&BakeSource<'_>, AtlasShape),
    (column, row): (u32, u32),
) {
    let side = tile_frames(shape);
    let (columns, rows) = (
        side.min(shape.frames - column),
        side.min(shape.frames - row),
    );
    let group = bake_group(
        device,
        layout,
        (source, shape),
        [column, row, columns, side],
        Some(&targets.shade),
    );
    {
        let attachments: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = targets
            .colors
            .iter()
            .zip(&targets.resolves)
            .map(|(view, resolve)| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: Some(resolve),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Discard,
                    },
                })
            })
            .collect();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/impostor-bake"),
            color_attachments: &attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(BAKE_GROUP, &group, &[]);
        draw_frames(&mut pass, drawn, false, columns * rows);
    }
    for (resolved, layer) in targets.resolved.iter().zip(&targets.layer) {
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: resolved,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: layer,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: column * shape.texels,
                    y: row * shape.texels,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: columns * shape.texels,
                height: rows * shape.texels,
                depth_or_array_layers: 1,
            },
        );
    }
}

fn bake_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    (source, shape): (&BakeSource<'_>, AtlasShape),
    tile: [u32; 4],
    shade: Option<&wgpu::TextureView>,
) -> wgpu::BindGroup {
    let baking = ImpostorRecord::new(
        source.bounds.0,
        source.bounds.1,
        0,
        ([0.0; 3], false),
        shape,
    );
    let uniform = crate::memory::buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("engine/impostor-bake"),
            contents: bytemuck::bytes_of(&ImpostorBake {
                sphere: baking.sphere,
                flags: [baking.flags, 0, 0, 0],
                tile,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    );
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: uniform.as_entire_binding(),
    }];
    entries.extend(shade.map(|shade| wgpu::BindGroupEntry {
        binding: 1,
        resource: crate::pipelines::view(shade),
    }));
    crate::pipelines::bind_group(device, "engine/impostor-bake-group", layout, &entries)
}

fn bake_texture(
    device: &wgpu::Device,
    size: u32,
    samples: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    crate::memory::texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("engine/impostor-bake-target"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        },
    )
}

fn bake_targets(device: &wgpu::Device, shape: AtlasShape) -> BakeTargets {
    let size = shape.size();
    let tile = tile_frames(shape) * shape.texels;
    let sampled = |format| {
        bake_texture(
            device,
            tile,
            BAKE_SAMPLES,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
    };
    let textures = [
        bake_texture(
            device,
            (size * SHADE_SCALE).min(SHADE_MOST),
            1,
            BAKE_DEPTH,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        ),
        sampled(FORMATS[0]),
        sampled(FORMATS[1]),
        sampled(FORMATS[2]),
        sampled(FORMATS[3]),
        sampled(BAKE_DEPTH),
    ];
    let resolved = FORMATS.map(|format| {
        bake_texture(
            device,
            tile,
            1,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        )
    });
    let layer = FORMATS.map(|format| {
        crate::memory::texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("engine/impostor-bake-layer"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: shape.levels(),
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )
    });
    let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
    BakeTargets {
        shade: view(&textures[0]),
        colors: [
            view(&textures[1]),
            view(&textures[2]),
            view(&textures[3]),
            view(&textures[4]),
        ],
        depth: view(&textures[5]),
        resolves: resolved.each_ref().map(view),
        resolved,
        layer,
        _textures: textures,
    }
}
