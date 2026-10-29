#![expect(
    clippy::expect_used,
    reason = "groups are built from every contribution before passes bind them"
)]

use super::*;
use crate::bindings::{GroupRef, SharedBinding, SharedGroups};
use crate::cached::GrowableGroupCache;
use crate::pieces::{PIECE_ALIGN, PIECES_BINDING, PieceAt};
use crate::pipeline_layouts::PipelineLayout;

const FRAME: &str = "frame";
const LABEL: &str = "engine/shared-group";

struct SharedView {
    group: &'static str,
    without: &'static [&'static str],
    layout: crate::pipelines::GroupLayout,
    bound: GrowableGroupCache,
    views: Vec<GrowableGroupCache>,
}

pub(crate) struct CameraBindings {
    pub(crate) buffer: wgpu::Buffer,
    pieces: Versioned<wgpu::Buffer>,
    placed: Vec<(usize, PieceAt)>,
    staged: Vec<u8>,
    written: Option<Vec<u8>>,
    views: Vec<SharedView>,
}

type Sources<'a> = Vec<(&'static str, Bound<'a>)>;

impl CameraBindings {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let buffer = crate::buffers::uniform(
            device,
            "engine/camera-uniform",
            size_of::<CameraUniform>() as u64,
        );
        Self {
            buffer,
            pieces: Versioned::new(pieces_buffer(device, PIECE_ALIGN)),
            placed: Vec::new(),
            staged: vec![0; PIECE_ALIGN as usize],
            written: None,
            views: Vec::new(),
        }
    }

    pub(crate) fn install(
        &mut self,
        device: &wgpu::Device,
        layouts: &[(&'static str, &'static PipelineLayout)],
    ) {
        self.views.clear();
        for &(_, layout) in crate::pipeline_layouts::PIPELINE_LAYOUTS.iter().chain(layouts) {
            for reference in layout.groups {
                let GroupRef::Shared { group, without } = *reference else {
                    continue;
                };
                if self
                    .views
                    .iter()
                    .any(|view| view.group == group && view.without == without)
                {
                    continue;
                }
                self.views.push(SharedView {
                    group,
                    without,
                    layout: crate::pipelines::shared_layout(device, group, without),
                    bound: GrowableGroupCache::default(),
                    views: Vec::new(),
                });
            }
        }
    }

    pub(crate) fn place_pieces(
        &mut self,
        device: &wgpu::Device,
        placed: Vec<(usize, PieceAt)>,
        size: u64,
    ) {
        if self.pieces.size() != size {
            self.pieces.replace(pieces_buffer(device, size));
        }
        self.placed = placed;
        self.staged = vec![0; size as usize];
        self.written = None;
    }

    pub(crate) fn pieces(&self) -> &wgpu::Buffer {
        &self.pieces
    }

    pub(crate) fn write_pieces(&mut self, queue: &crate::gpu_calls::Queue, parts: &Parts) -> u64 {
        for &(index, at) in &self.placed {
            let Some(bytes) = parts.part(index).and_then(|part| part.piece()) else {
                continue;
            };
            let start = at.offset as usize;
            let length = bytes.len().min(at.size as usize);
            self.staged[start..start + length].copy_from_slice(&bytes[..length]);
        }
        if self.written.as_deref() == Some(self.staged.as_slice()) {
            return 0;
        }
        queue.write_buffer(&self.pieces, 0, &self.staged);
        self.written = Some(self.staged.clone());
        self.staged.len() as u64
    }

    pub(crate) fn refresh(
        &mut self,
        device: &wgpu::Device,
        shared: &SharedGroups,
        parts: &Parts,
        view_cameras: Option<&[wgpu::Buffer]>,
    ) {
        let Self {
            buffer,
            pieces,
            views,
            ..
        } = self;
        let mut contributed: Vec<(&'static str, Sources<'_>)> = Vec::new();
        for view in views.iter() {
            if contributed.iter().any(|(group, _)| *group == view.group) {
                continue;
            }
            let mut sources: Sources<'_> = vec![
                ("camera", Bound::fixed(buffer.as_entire_binding())),
                (PIECES_BINDING, pieces.bound()),
            ];
            for (_, part) in parts.iter() {
                part.contribute(view.group, &mut sources);
            }
            contributed.push((view.group, sources));
        }
        for view in views.iter_mut() {
            let Some((_, sources)) = contributed.iter().find(|(group, _)| *group == view.group)
            else {
                continue;
            };
            let bindings = || shared.view(view.group, view.without);
            bind_view(device, (bindings(), &view.layout), &mut view.bound, sources);
            let Some(cameras) = view_cameras.filter(|_| view.group == FRAME) else {
                continue;
            };
            if view.views.len() < cameras.len() {
                view.views
                    .resize_with(cameras.len(), GrowableGroupCache::default);
            }
            for (index, camera) in cameras.iter().enumerate() {
                let mut sources = sources.clone();
                if let Some((_, bound)) = sources.iter_mut().find(|(name, _)| *name == "camera") {
                    *bound = Bound::fixed(camera.as_entire_binding());
                }
                bind_view(
                    device,
                    (bindings(), &view.layout),
                    &mut view.views[index],
                    &sources,
                );
            }
        }
    }

    pub(crate) fn group(&self, group: &str, without: &[&str]) -> &wgpu::BindGroup {
        self.built(group, without).expect("shared_group_built")
    }

    pub(crate) fn built(&self, group: &str, without: &[&str]) -> Option<&wgpu::BindGroup> {
        self.view(group, without).and_then(|view| view.bound.get())
    }

    pub(crate) fn view_group(&self, index: usize, without: &[&str]) -> &wgpu::BindGroup {
        self.view(FRAME, without)
            .and_then(|view| view.views.get(index)?.get())
            .expect("prepared_before_views")
    }

    fn view(&self, group: &str, without: &[&str]) -> Option<&SharedView> {
        self.views
            .iter()
            .find(|view| view.group == group && view.without == without)
    }
}

fn pieces_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    crate::buffers::uniform(device, "engine/pieces-uniform", size)
}

fn bind_view<'s>(
    device: &wgpu::Device,
    (bindings, layout): (
        impl Iterator<Item = &'s SharedBinding>,
        &crate::pipelines::GroupLayout,
    ),
    cache: &mut GrowableGroupCache,
    sources: &[(&'static str, Bound<'_>)],
) {
    let mut entries: Vec<(u32, Bound<'_>)> = Vec::new();
    for binding in bindings {
        let Some((_, bound)) = sources.iter().find(|(name, _)| *name == binding.decl.name) else {
            cache.clear();
            return;
        };
        entries.push((binding.number, bound.clone()));
    }
    cache.bind_with(&entries, |entries| bind_group(device, layout, entries));
}

#[expect(
    clippy::disallowed_methods,
    reason = "a shared group binds each contribution at the number its view gives it"
)]
fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    entries: &[wgpu::BindGroupEntry<'_>],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(LABEL),
        layout,
        entries,
    })
}

impl Core {
    pub fn pieces_buffer(&self) -> &wgpu::Buffer {
        self.camera.pieces()
    }
}
