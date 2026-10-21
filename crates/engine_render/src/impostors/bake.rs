//! Baking an impostor no host holds the record of: its mesh drawn with its
//! material from every frame's direction into a layer at the atlas's shape
//! (the frames' nearest depths first, which the colors' bake looks up for
//! the sky each surface sees past its mesh), then the levels below, read
//! back as its record (`super::records`), which then loads as any record
//! does.

use super::*;
use crate::gpu_layouts::impostors::ImpostorBakeUniform;

/// Bakes rasterize four samples a texel; their resolve averages
/// fragments with the cleared zeros, premultiplying coverage.
const SAMPLES: u32 = 4;
const BAKE_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// The depth of every frame's nearest surface, which bakes look up to
/// find the directions a surface sees past its mesh, has this many
/// times the layer's texels across, up to [`SHADE_MOST`].
const SHADE_SCALE: u32 = 4;
const SHADE_MOST: u32 = 2048;

/// Targets every bake renders into: the frames' nearest depths, the
/// multisampled colors and their depth, and the layer they resolve to at
/// the bake's shape, with its levels, which records read back.
pub(super) struct BakeTargets {
    shade: wgpu::TextureView,
    colors: [wgpu::TextureView; 4],
    depth: wgpu::TextureView,
    pub(super) layer: [wgpu::Texture; 4],
    _textures: [wgpu::Texture; 6],
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

/// Draws `source` from every frame's direction of `shape` into `targets`'
/// layer, with the bake pipelines of `shader` that `pipelines` keep: the
/// frames' nearest depths first, which the colors' bake looks up.
fn draw_layer(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    shader: &wgpu::ShaderModule,
    pipelines: &mut ImpostorPipelines,
    targets: &BakeTargets,
    source: &BakeSource<'_>,
    shape: AtlasShape,
) {
    let (shade_group, color_group) = bake_groups(device, pipelines, targets, source, shape);
    // Bakes are rare: uber variants.
    let variant = source.variant.with_extended(source.extended).uber();
    let layout = &pipelines.bake_pipeline_layout;
    let pipeline = pipelines
        .bake_pipelines
        .entry(variant)
        .or_insert_with(|| bake_pipeline(device, shader, layout, variant, false));
    let shade_pipeline = pipelines
        .shade_pipelines
        .entry(variant)
        .or_insert_with(|| bake_pipeline(device, shader, layout, variant, true));
    let frames = shape.frames * shape.frames;
    let draw = |pass: &mut wgpu::RenderPass<'_>| {
        pass.set_bind_group(1, source.group, &[]);
        pass.set_vertex_buffer(0, source.draw.vertices.slice(..));
        if source.extended
            && let Some(extras) = source.draw.extras
        {
            pass.set_vertex_buffer(1, extras.slice(..));
        }
        pass.set_index_buffer(source.draw.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(
            source.draw.first_index..source.draw.first_index + source.draw.index_count,
            source.draw.base_vertex,
            0..frames,
        );
    };
    {
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
        pass.set_pipeline(shade_pipeline);
        pass.set_bind_group(2, &shade_group, &[]);
        draw(&mut pass);
    }
    let resolves = targets.layer.each_ref().map(first_level);
    let attachments: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = targets
        .colors
        .iter()
        .zip(&resolves)
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
    pass.set_pipeline(pipeline);
    pass.set_bind_group(2, &color_group, &[]);
    draw(&mut pass);
}

/// The bake's groups of `source` at `shape`: the frames' nearest depths'
/// pass, which reads none, and the colors', which reads `targets'`.
fn bake_groups(
    device: &wgpu::Device,
    pipelines: &mut ImpostorPipelines,
    targets: &BakeTargets,
    source: &BakeSource<'_>,
    shape: AtlasShape,
) -> (wgpu::BindGroup, wgpu::BindGroup) {
    // The bake's own record: its sphere and the bake's flags.
    let baking = ImpostorRecord::new(
        source.bounds.0,
        source.bounds.1,
        0,
        ([0.0; 3], false, false),
        shape,
    );
    let uniform = wgpu::util::DeviceExt::create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("engine/impostor-bake"),
            contents: bytemuck::bytes_of(&ImpostorBakeUniform {
                sphere: baking.sphere,
                flags: [baking.flags, 0, 0, 0],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    );
    let no_shade = pipelines.no_shade.get_or_insert_with(|| {
        bake_texture(
            device,
            1,
            1,
            BAKE_DEPTH,
            wgpu::TextureUsages::TEXTURE_BINDING,
        )
        .create_view(&Default::default())
    });
    let group = |shade: &wgpu::TextureView| {
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
    };
    (group(no_shade), group(&targets.shade))
}

/// The first level of `texture`, which bakes resolve into.
fn first_level(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("engine/impostor-layer"),
        base_mip_level: 0,
        mip_level_count: Some(1),
        ..Default::default()
    })
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
    let target = |format| {
        bake_texture(
            device,
            size,
            SAMPLES,
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
        target(FORMATS[0]),
        target(FORMATS[1]),
        target(FORMATS[2]),
        target(FORMATS[3]),
        target(BAKE_DEPTH),
    ];
    // The layer the colors resolve to, with its levels, which records
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
                | wgpu::TextureUsages::COPY_SRC,
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
        if shade { 1 } else { SAMPLES },
    )
}
