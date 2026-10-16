//! Optional roughness queues. Limits select the original whole-frame path.

use crate::bindings::{REFLECTION_EXPORT, REFLECTION_INPUTS, REFLECTION_TRACE, REFLECTION_WORK};
use crate::cached::{Bound, GroupCache};
use crate::gpu_data::QueueParams;
use crate::pipeline_layouts::{
    REFLECTION_EXPORT_PIPELINES, REFLECTION_TRACE_PIPELINES, REFLECTION_WORK_PIPELINES,
};
use crate::pipelines::{self, GroupLayout};
use crate::reflection_queue_plan::Plan;
use crate::shaders::{
    self,
    entry::{reflection_export, reflection_trace, reflection_work},
};
use crate::targets::FrameSize;

struct Pipelines {
    inputs: GroupLayout,
    work: GroupLayout,
    trace: GroupLayout,
    export: GroupLayout,
    classify: wgpu::ComputePipeline,
    scan_up: wgpu::ComputePipeline,
    scan_down: wgpu::ComputePipeline,
    finalize: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    trace_rays: wgpu::ComputePipeline,
    export_raw: wgpu::RenderPipeline,
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        let inputs = pipelines::layout(device, &REFLECTION_INPUTS);
        let work = pipelines::layout(device, &REFLECTION_WORK);
        let trace = pipelines::layout(device, &REFLECTION_TRACE);
        let export = pipelines::layout(device, &REFLECTION_EXPORT);
        let work_layout =
            pipelines::pipeline_layout(device, &REFLECTION_WORK_PIPELINES, &[&inputs, &work]);
        let trace_layout =
            pipelines::pipeline_layout(device, &REFLECTION_TRACE_PIPELINES, &[&inputs, &trace]);
        let export_layout =
            pipelines::pipeline_layout(device, &REFLECTION_EXPORT_PIPELINES, &[&export]);
        let module = |label, source: &'static str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        };
        let work_module = module("engine/reflection-work", shaders::REFLECTION_WORK_SHADER);
        let trace_module = module("engine/reflection-trace", shaders::REFLECTION_TRACE_SHADER);
        let export_module = module(
            "engine/reflection-export",
            shaders::REFLECTION_EXPORT_SHADER,
        );
        let work_pipeline =
            |entry| pipelines::compute_entry(device, &work_layout, &work_module, entry);
        Self {
            inputs,
            work,
            trace,
            export,
            classify: work_pipeline(reflection_work::classify),
            scan_up: work_pipeline(reflection_work::scan_up),
            scan_down: work_pipeline(reflection_work::scan_down),
            finalize: work_pipeline(reflection_work::finalize),
            scatter: work_pipeline(reflection_work::scatter),
            trace_rays: pipelines::compute_entry(
                device,
                &trace_layout,
                &trace_module,
                reflection_trace::trace,
            ),
            export_raw: pipelines::fullscreen(
                device,
                &pipelines::Fullscreen {
                    label: "engine/reflection-export",
                    layout: &export_layout,
                    module: &export_module,
                    vertex: reflection_export::vs_fullscreen,
                    fragment: reflection_export::export_reflections,
                    constants: &[],
                    targets: &[Some(super::post::REFLECTION_FORMAT.into())],
                    depth: None,
                },
            ),
        }
    }
}

struct Scratch {
    plan: Plan,
    buffers: [wgpu::Buffer; 7],
    work: wgpu::BindGroup,
    trace: wgpu::BindGroup,
    export: wgpu::BindGroup,
    inputs: GroupCache<8>,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        for buffer in &self.buffers {
            buffer.destroy();
        }
    }
}

impl Scratch {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, plan: Plan, pipelines: &Pipelines) -> Self {
        let buffers = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/reflection-scratch"),
                size: plan.bytes[i],
                usage: if i == 6 {
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
                } else if i == 5 {
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT
                } else {
                    wgpu::BufferUsages::STORAGE
                },
                mapped_at_creation: false,
            })
        });
        Self::write_params(queue, &buffers[6], &plan);
        let params = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &buffers[6],
            offset: 0,
            size: wgpu::BufferSize::new(48),
        });
        let work = pipelines::group(
            device,
            "engine/reflection-work",
            &REFLECTION_WORK,
            &pipelines.work,
            &[
                ("params", params.clone()),
                ("tags", buffers[0].as_entire_binding()),
                ("prefixes", buffers[1].as_entire_binding()),
                ("rays", buffers[2].as_entire_binding()),
                ("raw", buffers[3].as_entire_binding()),
                ("metadata", buffers[4].as_entire_binding()),
                ("arguments", buffers[5].as_entire_binding()),
            ],
        );
        let trace = pipelines::group(
            device,
            "engine/reflection-trace",
            &REFLECTION_TRACE,
            &pipelines.trace,
            &[
                ("params", params.clone()),
                ("rays", buffers[2].as_entire_binding()),
                ("metadata", buffers[4].as_entire_binding()),
                ("raw", buffers[3].as_entire_binding()),
            ],
        );
        let export = pipelines::group(
            device,
            "engine/reflection-export",
            &REFLECTION_EXPORT,
            &pipelines.export,
            &[("params", params), ("raw", buffers[3].as_entire_binding())],
        );
        Self {
            plan,
            buffers,
            work,
            trace,
            export,
            inputs: GroupCache::default(),
        }
    }

    fn write_params(queue: &wgpu::Queue, buffer: &wgpu::Buffer, plan: &Plan) {
        let mut params = QueueParams {
            size: plan.size,
            scan: [0; 4],
            extra: [0, plan.pixels, plan.row_log2, 0],
        };
        let write = |record: u32, params: &QueueParams| {
            queue.write_buffer(
                buffer,
                u64::from(record) * u64::from(plan.stride),
                bytemuck::bytes_of(params),
            );
        };
        write(0, &params);
        for (index, level) in plan.levels.iter().enumerate() {
            let parent = plan
                .levels
                .get(index + 1)
                .map_or(u32::MAX, |level| level.offset);
            params.scan = [level.offset, level.length, parent, level.grid[0]];
            write(index as u32 + 1, &params);
        }
        params.scan = [0; 4];
        for bin in 0..4 {
            params.extra[0] = bin;
            write(plan.levels.len() as u32 + 1 + bin, &params);
        }
    }

    fn construct(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &Pipelines,
        inputs: &wgpu::BindGroup,
    ) {
        let plan = &self.plan;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/reflection-classify-scan-scatter"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, inputs, &[]);
        pass.set_bind_group(1, &self.work, &[0]);
        pass.set_pipeline(&pipelines.classify);
        pass.dispatch_workgroups(plan.size[2], plan.size[3], 1);
        pass.set_pipeline(&pipelines.scan_up);
        for (index, level) in plan.levels.iter().enumerate() {
            pass.set_bind_group(1, &self.work, &[(index as u32 + 1) * plan.stride]);
            pass.dispatch_workgroups(level.grid[0], level.grid[1], 1);
        }
        pass.set_pipeline(&pipelines.scan_down);
        for (index, level) in plan.levels[..plan.levels.len() - 1]
            .iter()
            .enumerate()
            .rev()
        {
            pass.set_bind_group(1, &self.work, &[(index as u32 + 1) * plan.stride]);
            pass.dispatch_workgroups(level.grid[0], level.grid[1], 1);
        }
        pass.set_bind_group(1, &self.work, &[0]);
        pass.set_pipeline(&pipelines.finalize);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&pipelines.scatter);
        pass.dispatch_workgroups(plan.size[2], plan.size[3], 1);
    }
}

#[derive(Default)]
pub(crate) struct ReflectionQueue {
    pipelines: Option<Pipelines>,
    scratch: Option<Scratch>,
    attempted: Option<FrameSize>,
}

impl ReflectionQueue {
    pub(crate) fn release(&mut self) {
        self.scratch = None;
        self.attempted = None;
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: FrameSize,
        enabled: bool,
    ) {
        if !enabled {
            self.release();
            return;
        }
        if self.attempted == Some(size) {
            return;
        }
        self.release();
        self.attempted = Some(size);
        let Some(plan) = Plan::new(size.width, size.height, &device.limits()) else {
            return;
        };
        let pipelines = self.pipelines.get_or_insert_with(|| Pipelines::new(device));
        self.scratch = Some(Scratch::new(device, queue, plan, pipelines));
    }

    pub(crate) fn bind(&mut self, device: &wgpu::Device, inputs: [(&str, Bound<'_>); 8]) {
        if let (Some(scratch), Some(pipelines)) = (&mut self.scratch, &self.pipelines) {
            scratch.inputs.bind(
                device,
                "engine/reflection-inputs",
                (&REFLECTION_INPUTS, &pipelines.inputs),
                inputs,
            );
        }
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.scratch
            .as_ref()
            .map_or(0, |scratch| scratch.plan.capacity_bytes())
    }

    /// True when queues traced and exported the frame; false selects the fallback.
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) -> bool {
        let (Some(scratch), Some(pipelines)) = (&self.scratch, &self.pipelines) else {
            return false;
        };
        let Some(inputs) = scratch.inputs.get() else {
            return false;
        };
        scratch.construct(encoder, pipelines, inputs);
        // End construction before the arguments become INDIRECT, and use only
        // read-only queue/metadata bindings during tracing.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/reflection-trace"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.trace_rays);
            pass.set_bind_group(0, inputs, &[]);
            for bin in 0..4 {
                let record = scratch.plan.levels.len() as u32 + 1 + bin;
                pass.set_bind_group(1, &scratch.trace, &[record * scratch.plan.stride]);
                pass.dispatch_workgroups_indirect(&scratch.buffers[5], u64::from(bin) * 12);
            }
        }
        let mut pass = super::post::fullscreen_pass(
            encoder,
            "engine/reflection-export",
            target,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            None,
        );
        pass.set_pipeline(&pipelines.export_raw);
        pass.set_bind_group(0, &scratch.export, &[0]);
        pass.draw(0..3, 0..1);
        true
    }
}
