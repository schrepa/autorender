use engine_core::{
    MeshHandle, RenderFrame, RenderMaterial, RenderMesh, RenderSnapshot, RenderTexture,
};
use glam::Mat4;

use crate::CameraUniform;
use crate::entries::PassKind;
use crate::graph::{Clear, FrameShape, Load, Ops, Planned, Targets};
use crate::renderer::History;

#[derive(Clone, Copy, Debug)]
pub struct FrameKind {
    pub gpu_mode: bool,
    pub temporal: bool,
    pub display: bool,
    pub debug: bool,
    pub instance_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Passes {
    pub opaque_kind: PassKind,
    pub impostor_pass: Option<Targets>,
}

#[derive(Clone, Copy)]
pub struct TemporalFrame {
    pub parity: usize,
    pub camera: CameraUniform,
    pub rasterized: Mat4,
    pub history_valid: bool,
    pub motion_valid: bool,
    pub(crate) history: History,
}

pub struct FrameInfo<'s> {
    pub snapshot: &'s RenderSnapshot,
    pub frame: RenderFrame<'s>,
    pub kind: FrameKind,
    pub temporal: Option<TemporalFrame>,
    pub passes: Option<Passes>,
    pub shape: Option<FrameShape>,
}

#[derive(Default)]
pub struct PartAssets<'s> {
    pub meshes: Vec<(MeshHandle, &'s RenderMesh)>,
    pub materials: Vec<&'s RenderMaterial>,
    pub textures: Vec<&'s RenderTexture>,
}

pub struct NodeViews {
    pub colors: Vec<Option<wgpu::TextureView>>,
    pub depth: Option<wgpu::TextureView>,
    pub region: Option<(u32, u32)>,
}

pub fn render_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    planned: &Planned,
    views: &NodeViews,
    label: &'static str,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) -> wgpu::RenderPass<'e> {
    let colors: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> = planned
        .colors
        .iter()
        .zip(&views.colors)
        .map(|(&(_, ops), view)| {
            view.as_ref().map(|view| wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: color_operations(ops),
            })
        })
        .collect();
    let count = colors.iter().take_while(|color| color.is_some()).count();
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &colors[..count],
        depth_stencil_attachment: planned.depth.zip(views.depth.as_ref()).map(|((_, ops), view)| {
            wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: ops.map(depth_operations),
                stencil_ops: None,
            }
        }),
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    if let Some(region) = views.region {
        crate::targets::draw_within(&mut pass, region);
    }
    pass
}

fn color_operations(ops: Ops) -> wgpu::Operations<wgpu::Color> {
    wgpu::Operations {
        load: match ops.load {
            Load::Clear(Clear::Color([r, g, b, a])) => {
                wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a })
            }
            Load::Clear(Clear::Depth(_)) | Load::Load => wgpu::LoadOp::Load,
        },
        store: store_operation(ops),
    }
}

fn depth_operations(ops: Ops) -> wgpu::Operations<f32> {
    wgpu::Operations {
        load: match ops.load {
            Load::Clear(Clear::Depth(depth)) => wgpu::LoadOp::Clear(depth),
            Load::Clear(Clear::Color(_)) | Load::Load => wgpu::LoadOp::Load,
        },
        store: store_operation(ops),
    }
}

fn store_operation(ops: Ops) -> wgpu::StoreOp {
    if ops.store {
        wgpu::StoreOp::Store
    } else {
        wgpu::StoreOp::Discard
    }
}
