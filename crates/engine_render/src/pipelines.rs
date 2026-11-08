use std::cell::RefCell;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use crate::bindings::{Kind, Sample, Sampling, Table, View};

pub const UNIFORM_BINDING_BYTES: u64 = 16_384;

#[derive(Clone, Copy, Debug)]
pub struct StageLimits {
    pub bind_groups: u32,
    pub uniform_buffers: u32,
    pub storage_buffers: u32,
    pub sampled_textures: u32,
    pub samplers: u32,
    pub storage_textures: u32,
    pub dynamic_uniform_buffers: u32,
    pub dynamic_storage_buffers: u32,
}

impl StageLimits {
    pub(crate) fn of(limits: &wgpu::Limits) -> Self {
        Self {
            bind_groups: limits.max_bind_groups,
            uniform_buffers: limits.max_uniform_buffers_per_shader_stage,
            storage_buffers: limits.max_storage_buffers_per_shader_stage,
            sampled_textures: limits.max_sampled_textures_per_shader_stage,
            samplers: limits.max_samplers_per_shader_stage,
            storage_textures: limits.max_storage_textures_per_shader_stage,
            dynamic_uniform_buffers: limits.max_dynamic_uniform_buffers_per_pipeline_layout,
            dynamic_storage_buffers: limits.max_dynamic_storage_buffers_per_pipeline_layout,
        }
    }
}

pub const REQUIRED_LIMITS: StageLimits = StageLimits {
    bind_groups: 4,
    uniform_buffers: 12,
    storage_buffers: 8,
    sampled_textures: 16,
    samplers: 16,
    storage_textures: 4,
    dynamic_uniform_buffers: 8,
    dynamic_storage_buffers: 4,
};

thread_local! {
    static CREATED: RefCell<Vec<(&'static str, u32)>> = const { RefCell::new(Vec::new()) };
}

fn add(counts: &mut Vec<(&'static str, u32)>, pass: &'static str, created: u32) {
    match counts.iter_mut().find(|(known, _)| *known == pass) {
        Some((_, held)) => *held = held.saturating_add(created),
        None => counts.push((pass, created)),
    }
}

pub(crate) fn count(pass: &'static str, created: u32) {
    CREATED.with(|counted| add(&mut counted.borrow_mut(), pass, created));
}

fn take_created() -> Vec<(&'static str, u32)> {
    CREATED.with(|created| std::mem::take(&mut *created.borrow_mut()))
}

#[derive(Default)]
pub(crate) struct Ledger {
    created: u32,
    per_pass: Vec<(&'static str, u32)>,
    frames: u64,
    compiled: Arc<AtomicU64>,
}

impl Ledger {
    pub(crate) fn frame_submitted(&mut self, queue: &crate::gpu_calls::Queue) -> u32 {
        let mut created = 0_u32;
        for (pass, more) in take_created() {
            add(&mut self.per_pass, pass, more);
            created = created.saturating_add(more);
        }
        self.created = self.created.saturating_add(created);
        if created > 0 {
            self.frames += 1;
            let (compiled, frame) = (Arc::clone(&self.compiled), self.frames);
            queue.on_submitted_work_done(move || {
                compiled.fetch_max(frame, Ordering::Relaxed);
            });
        }
        created
    }

    pub(crate) fn created(&self) -> u32 {
        self.created
    }

    pub(crate) fn per_pass(&self) -> &[(&'static str, u32)] {
        &self.per_pass
    }

    pub(crate) fn compiling(&self) -> u32 {
        let left = self
            .frames
            .saturating_sub(self.compiled.load(Ordering::Relaxed));
        u32::try_from(left).unwrap_or(u32::MAX)
    }
}

fn binding_type(kind: Kind) -> wgpu::BindingType {
    match kind {
        Kind::Uniform { min_size, dynamic } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: wgpu::BufferSize::new(min_size),
        },
        Kind::Texture { sample, view } => wgpu::BindingType::Texture {
            sample_type: match sample {
                Sample::Float => wgpu::TextureSampleType::Float { filterable: true },
            },
            view_dimension: match view {
                View::D2 => wgpu::TextureViewDimension::D2,
            },
            multisampled: false,
        },
        Kind::Sampler(sampling) => wgpu::BindingType::Sampler(match sampling {
            Sampling::Filtering => wgpu::SamplerBindingType::Filtering,
        }),
    }
}

pub fn layout(device: &wgpu::Device, table: &'static Table) -> GroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = table
        .bindings
        .iter()
        .zip(0..)
        .map(|(binding, number)| wgpu::BindGroupLayoutEntry {
            binding: number,
            visibility: crate::machine::shader_stages(binding.stages),
            ty: binding_type(binding.kind),
            count: None,
        })
        .collect();
    GroupLayout {
        label: table.label,
        layout: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(table.label),
            entries: &entries,
        }),
    }
}

#[derive(Clone, Debug)]
pub struct GroupLayout {
    label: &'static str,
    layout: wgpu::BindGroupLayout,
}

impl std::ops::Deref for GroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }
}

pub fn view(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(view)
}

pub fn sampler(sampler: &wgpu::Sampler) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Sampler(sampler)
}

pub fn group(
    device: &wgpu::Device,
    label: &str,
    table: &Table,
    layout: &GroupLayout,
    resources: &[(&str, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    debug_assert_eq!(layout.label, table.label, "layout_matches {label}");
    debug_assert_eq!(
        resources.len(),
        table.bindings.len(),
        "binds_every_binding {label} {}",
        table.label
    );
    let entries: Vec<wgpu::BindGroupEntry<'_>> = resources
        .iter()
        .map(|(name, resource)| wgpu::BindGroupEntry {
            binding: table
                .bindings
                .iter()
                .zip(0..)
                .find(|(binding, _)| binding.name == *name)
                .map(|(_, number)| number)
                .unwrap_or_else(|| panic!("binding_unknown {} {name}", table.label)),
            resource: resource.clone(),
        })
        .collect();
    bind_group(device, label, layout, &entries)
}

#[expect(clippy::disallowed_methods, reason = "the one way to make bind groups")]
pub(crate) fn bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    entries: &[wgpu::BindGroupEntry<'_>],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries,
    })
}
