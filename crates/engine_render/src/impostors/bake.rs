//! Baking an impostor no host holds the record of: its mesh drawn with its
//! material, and its parts with theirs, from every frame's direction into a
//! layer at the atlas's shape (the frames' nearest depths first, the whole
//! grid at once, which the colors' bake looks up for the sky each surface
//! sees past its mesh; then the colors a tile of frames at a time, each
//! tile copied into the layer, so the multisampled targets stay a tile's
//! size however many texels the frames hold), then the levels below, read
//! back as its record (`super::records`), which then loads as any record
//! does.

use super::*;
use crate::gpu_layouts::impostors::ImpostorBakeUniform;

/// Bakes rasterize four samples a texel; their resolve averages
/// fragments with the cleared zeros, premultiplying coverage.
const BAKE_SAMPLES: u32 = 4;
const BAKE_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// The depth of every frame's nearest surface, which bakes look up to
/// find the directions a surface sees past its mesh, has this many
/// times the layer's texels across, up to [`SHADE_MOST`].
const SHADE_SCALE: u32 = 4;
const SHADE_MOST: u32 = 2048;
/// Texels across the tiles of frames whose colors a bake draws at once,
/// or a frame's where one is wider.
const TILE_TEXELS: u32 = 1024;

/// Targets every bake renders into: the frames' nearest depths, a tile's
/// multisampled colors and their depth, the colors they resolve to, and
/// the layer the tiles are copied into at the bake's shape, with its
/// levels, which records read back.
pub(super) struct BakeTargets {
    shade: wgpu::TextureView,
    colors: [wgpu::TextureView; 4],
    depth: wgpu::TextureView,
    resolves: [wgpu::TextureView; 4],
    resolved: [wgpu::Texture; 4],
    pub(super) layer: [wgpu::Texture; 4],
    _textures: [wgpu::Texture; 6],
}

/// Frames along each side of the tiles a bake at `shape` draws its colors
/// in.
fn tile_frames(shape: AtlasShape) -> u32 {
    (TILE_TEXELS / shape.texels).clamp(1, shape.frames)
}

impl ImpostorAtlas {
    /// Records the bake of `source`, whose mesh and material hold
    /// `content`, into `gpu`'s encoder: the layer drawn at the bake's shape
    /// with its levels, read back as its record.
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
            shader,
            pipelines,
        } = gpu;
        let encoder = prelude(device, encoder);
        let shape = self.shape;
        let targets = self
            .targets
            .get_or_insert_with(|| bake_targets(device, shape));
        draw_layer(device, encoder, shader, pipelines, targets, source, shape);
        // Premultiplied texels average into the levels below: a frame's
        // cell halves with each level and never mixes with another's.
        for texture in &targets.layer {
            mips.record(device, encoder, texture);
        }
        self.read_back(device, encoder, content);
    }
}

/// What a bake draws, the mesh and then each part: its draw, whether it
/// reads extra streams, its material's group and its variant, an uber
/// variant as bakes are rare.
type BakeDraw<'a> = (&'a MeshDraw<'a>, bool, &'a wgpu::BindGroup, Variant);

/// Draws `source` and its parts from every frame's direction of `shape`
/// into `targets`' layer, with the bake pipelines of `shader` that
/// `pipelines` keep: the frames' nearest depths first, which the colors'
/// bake looks up, then the colors a tile at a time.
fn draw_layer(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    shader: &wgpu::ShaderModule,
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
                (draw, extended, group, variant.with_extended(extended).uber())
            })
            .collect();
    let layout = &pipelines.bake_pipeline_layout;
    for &(_, _, _, variant) in &draws {
        pipelines
            .bake_pipelines
            .entry(variant)
            .or_insert_with(|| bake_pipeline(device, shader, layout, variant, false));
        pipelines
            .shade_pipelines
            .entry(variant)
            .or_insert_with(|| bake_pipeline(device, shader, layout, variant, true));
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

/// Draws `draws` into `pass` with the bake pipelines `pipelines` keep, the
/// frames' nearest depths alone with `shade`: `frames` instances each,
/// the frames of the pass's tile.
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

/// Draws the colors of `source` at `shape` for the tile of frames from
/// the grid's `column` and `row` into `targets`, then copies what they
/// resolve to into the tile's place in the layer. Tiles at the grid's far
/// edges hold fewer frames, in the cells nearest their targets' top left.
fn draw_tile(
    (device, encoder): (&wgpu::Device, &mut wgpu::CommandEncoder),
    drawn: (&ImpostorPipelines, &[BakeDraw<'_>]),
    targets: &BakeTargets,
    (source, shape): (&BakeSource<'_>, AtlasShape),
    (column, row): (u32, u32),
) {
    let side = tile_frames(shape);
    let (columns, rows) = (side.min(shape.frames - column), side.min(shape.frames - row));
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

/// The group of a bake pass of `source` at `shape` drawing `tile` (the
/// grid's column and row of its first frame, its frames along each row and
/// the cells along each side of the pass's target), reading the frames'
/// nearest depths from `shade`, or a stand-in while they are drawn.
fn bake_group(
    device: &wgpu::Device,
    pipelines: &ImpostorPipelines,
    (source, shape): (&BakeSource<'_>, AtlasShape),
    tile: [u32; 4],
    shade: &wgpu::TextureView,
) -> wgpu::BindGroup {
    // The bake's own record: its sphere and the bake's flags.
    let baking = ImpostorRecord::new(
        source.bounds.0,
        source.bounds.1,
        0,
        ([0.0; 3], false, false, false),
        shape,
    );
    let uniform = wgpu::util::DeviceExt::create_buffer_init(
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
    device.create_texture(&wgpu::TextureDescriptor {
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
    })
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
    // What a tile's colors resolve to, copied into the layer.
    let resolved = FORMATS.map(|format| {
        bake_texture(
            device,
            tile,
            1,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        )
    });
    // The layer the tiles are copied into, with its levels, which records
    // read back.
    let layer = FORMATS.map(|format| {
        device.create_texture(&wgpu::TextureDescriptor {
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
        })
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

/// A bake pipeline of `variant`: the colors', or with `shade` the
/// frames' nearest depths alone.
fn bake_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
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
            module: shader,
            vertex,
            vertex_constants: &[],
            buffers: &buffers,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: (!variant.double_sided()).then_some(wgpu::Face::Back),
                ..Default::default()
            },
            // Nearer the frame's viewer is greater, as with reversed depth.
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
