#![expect(
    clippy::expect_used,
    reason = "groups are built from every contribution before passes bind them"
)]

use super::*;
use crate::bindings::{GroupRef, SharedBinding, SharedGroups};
use crate::cached::GrowableGroupCache;

const FRAME: &str = "frame";
const LABEL: &str = "engine/shared-group";

struct SharedView {
    group: &'static str,
    without: &'static [&'static str],
    layout: crate::pipelines::GroupLayout,
    bound: GrowableGroupCache,
    faces: [GrowableGroupCache; FACES],
}

pub(crate) struct CameraBindings {
    pub(crate) buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    lighting: Option<LightingUniform>,
    views: Vec<SharedView>,
}

type Sources<'a> = Vec<(&'static str, Bound<'a>)>;

impl CameraBindings {
    pub(crate) fn new(device: &wgpu::Device, lighting_buffer: wgpu::Buffer) -> Self {
        let buffer = crate::buffers::uniform(
            device,
            "engine/camera-uniform",
            size_of::<CameraUniform>() as u64,
        );
        Self {
            buffer,
            lighting_buffer,
            lighting: None,
            views: Vec::new(),
        }
    }

    pub(crate) fn install(&mut self, device: &wgpu::Device) {
        self.views.clear();
        for &(_, layout) in crate::pipeline_layouts::PIPELINE_LAYOUTS {
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
                    faces: Default::default(),
                });
            }
        }
    }

    pub(crate) fn lighting_buffer(&self) -> &wgpu::Buffer {
        &self.lighting_buffer
    }

    pub(crate) fn forget_lighting(&mut self) {
        self.lighting = None;
    }

    pub(crate) fn write_lighting(
        &mut self,
        queue: &crate::gpu_calls::Queue,
        lighting: LightingUniform,
    ) -> u64 {
        if self.lighting == Some(lighting) {
            return 0;
        }
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&lighting));
        self.lighting = Some(lighting);
        size_of::<LightingUniform>() as u64
    }

    pub(crate) fn refresh(
        &mut self,
        device: &wgpu::Device,
        shared: &SharedGroups,
        parts: &Parts,
        probe_cameras: Option<[&wgpu::Buffer; FACES]>,
    ) {
        let Self {
            buffer,
            lighting_buffer,
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
                (
                    "lighting",
                    Bound::fixed(lighting_buffer.as_entire_binding()),
                ),
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
            let Some(cameras) = probe_cameras.filter(|_| view.group == FRAME) else {
                continue;
            };
            for (face, camera) in cameras.into_iter().enumerate() {
                let mut sources = sources.clone();
                if let Some((_, bound)) = sources.iter_mut().find(|(name, _)| *name == "camera") {
                    *bound = Bound::fixed(camera.as_entire_binding());
                }
                bind_view(
                    device,
                    (bindings(), &view.layout),
                    &mut view.faces[face],
                    &sources,
                );
            }
        }
    }

    pub(crate) fn group(&self, group: &str, without: &[&str]) -> &wgpu::BindGroup {
        self.view(group, without)
            .and_then(|view| view.bound.get())
            .expect("shared_group_built")
    }

    pub(crate) fn probe_group(&self, face: usize, without: &[&str]) -> &wgpu::BindGroup {
        self.view(FRAME, without)
            .and_then(|view| view.faces.get(face)?.get())
            .expect("prepared_before_captures")
    }

    fn view(&self, group: &str, without: &[&str]) -> Option<&SharedView> {
        self.views
            .iter()
            .find(|view| view.group == group && view.without == without)
    }
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
