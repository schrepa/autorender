use super::*;
use crate::gpu_layouts::impostors::ImpostorBakeUniform;

const BAKE_SAMPLES: u32 = 4;
const BAKE_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
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

fn tile_frames(shape: AtlasShape) -> u32 {
    (TILE_TEXELS / shape.texels).clamp(1, shape.frames)
}

impl ImpostorAtlas {
    pub(super) fn bake(
        &mut self,
        gpu: BakeGpu<'_>,
        source: &BakeSource<'_>,
        content: (u128, u128),
    ) {
        let BakeGpu {
            device,
            queue: _,
            encoder,
            mips,
            pipelines,
        } = gpu;
        let encoder = encoder.mips(device);
        let shape = self.shape;
        let targets = self
            .targets
            .get_or_insert_with(|| bake_targets(device, shape));
        draw_layer(device, encoder, pipelines, targets, source, shape);
        for texture in &targets.layer {
            mips.record(device, encoder, texture);
        }
        self.read_back(device, encoder, content);
    }
}

type BakeDraw<'a> = (&'a MeshDraw<'a>, bool, &'a wgpu::BindGroup, Variant);

fn draw_layer(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &mut ImpostorPipelines,
    targets: &BakeTargets,
    source: &BakeSource<'_>,
    shape: AtlasShape,
) {
    let draws: Vec<BakeDraw<'_>> =
        std::iter::once((&source.draw, source.extended, source.group, source.variant))
            .chain(
                source
                    .parts
                    .iter()
                    .map(|part| (&part.draw, part.extended, part.group, part.variant)),
            )
            .map(|(draw, extended, group, variant)| {
                (
                    draw,
                    extended,
                    group,
                    variant.with_extended(extended).uber(),
                )
            })
            .collect();
    let layout = &pipelines.bake_pipeline_layout;
    for &(_, _, _, variant) in &draws {
        pipelines
            .bake_pipelines
            .entry(variant)
            .or_insert_with(|| bake_pipeline(device, layout, variant, false));
        pipelines
            .shade_pipelines
            .entry(variant)
            .or_insert_with(|| bake_pipeline(device, layout, variant, true));
    }
    let no_shade = pipelines
        .no_shade
        .get_or_insert_with(|| {
            bake_texture(
                device,
                1,
                1,
                BAKE_DEPTH,
                wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .create_view(&Default::default())
        })
        .clone();
    let drawn = (&*pipelines, draws.as_slice());
    let frames = shape.frames;
    {
        let group = bake_group(
            device,
            drawn.0,
            (source, shape),
            [0, 0, frames, frames],
            &no_shade,
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
        pass.set_bind_group(2, &group, &[]);
        draw_frames(&mut pass, drawn, true, frames * frames);
    }
    let side = tile_frames(shape) as usize;
    for row in (0..frames).step_by(side) {
        for column in (0..frames).step_by(side) {
            draw_tile(
                (device, &mut *encoder),
                drawn,
                targets,
                (source, shape),
                (column, row),
            );
        }
    }
}

fn draw_frames(
    pass: &mut wgpu::RenderPass<'_>,
    (pipelines, draws): (&ImpostorPipelines, &[BakeDraw<'_>]),
    shade: bool,
    frames: u32,
) {
    let kept = if shade {
        &pipelines.shade_pipelines
    } else {
        &pipelines.bake_pipelines
    };
    for &(mesh, extended, group, variant) in draws {
        let Some(pipeline) = kept.get(&variant) else {
            continue;
        };
        pass.set_pipeline(pipeline);
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
    drawn: (&ImpostorPipelines, &[BakeDraw<'_>]),
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
        drawn.0,
        (source, shape),
        [column, row, columns, side],
        &targets.shade,
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
        pass.set_bind_group(2, &group, &[]);
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
    pipelines: &ImpostorPipelines,
    (source, shape): (&BakeSource<'_>, AtlasShape),
    tile: [u32; 4],
    shade: &wgpu::TextureView,
) -> wgpu::BindGroup {
    let baking = ImpostorRecord::new(
        source.bounds.0,
        source.bounds.1,
        0,
        ([0.0; 3], false, false),
        shape,
    );
    let uniform = crate::memory::buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("engine/impostor-bake"),
            contents: bytemuck::bytes_of(&ImpostorBakeUniform {
                sphere: baking.sphere,
                flags: [baking.flags, 0, 0, 0],
                tile,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    );
    crate::pipelines::group(
        device,
        "engine/impostor-bake-group",
        &IMPOSTOR_BAKE,
        &pipelines.bake_layout,
        &[
            ("impostor_bake", uniform.as_entire_binding()),
            ("impostor_shade", crate::pipelines::view(shade)),
        ],
    )
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

fn bake_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    variant: Variant,
    shade: bool,
) -> wgpu::RenderPipeline {
    let constants = variant.constants();
    let extended = variant.extended();
    let (vertex, fragment) = crate::entries::impostor_bake(extended, shade);
    let mut buffers = vec![crate::renderer::vertex_layout()];
    if extended {
        buffers.push(crate::renderer::extra_layout());
    }
    let targets = if shade {
        Vec::new()
    } else {
        FORMATS
            .map(|format| Some(wgpu::ColorTargetState::from(format)))
            .to_vec()
    };
    crate::pipelines::sampled(
        device,
        &crate::pipelines::Geometry {
            label: "engine/impostor-bake",
            layout,
            vertex,
            vertex_constants: &[],
            buffers: &buffers,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: (!variant.double_sided()).then_some(wgpu::Face::Back),
                ..Default::default()
            },
            depth: Some(wgpu::DepthStencilState {
                format: BAKE_DEPTH,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            fragment: Some(crate::pipelines::Fragment {
                entry: fragment,
                constants: &constants,
                targets: &targets,
            }),
        },
        if shade { 1 } else { BAKE_SAMPLES },
    )
}
