use std::sync::mpsc;
use std::time::Instant;

use wgpu::util::DeviceExt;

use crate::Occluders;

const FEWEST_RAYS: u64 = 1 << 18;
const SHORT_SECONDS: f64 = 0.05;
const LONG_SECONDS: f64 = 0.2;
const WORKGROUP: u32 = 64;

pub struct Gpu {
    adapter: wgpu::Adapter,
}

struct Opened {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    run: wgpu::Buffer,
    work: wgpu::Buffer,
    found: wgpu::Buffer,
    read: wgpu::Buffer,
}

struct Upload<'a> {
    nodes: &'a [u32],
    triangles: &'a [u32],
    targets: &'a [f32],
    eyes: &'a [f32],
    widest: u64,
}

fn taken(items: &[[u32; 3]], rays: u64, most_items: usize) -> usize {
    let mut cast = 0_u64;
    let within = items.iter().take(most_items).take_while(|item| {
        cast += u64::from(item[2]);
        cast <= rays
    });
    within.count().max(1)
}

impl Gpu {
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|error| format!("the rays are cast on a GPU and none was found: {error}"))?;
        Ok(Self { adapter })
    }

    pub fn adapter(&self) -> (String, &'static str) {
        let info = self.adapter.get_info();
        (info.name, info.backend.to_str())
    }

    pub fn first_unblocked(
        &self,
        occluders: &Occluders,
        targets: &[[f32; 3]],
        eyes: &[[f32; 3]],
        items: &[[u32; 3]],
        gap: f32,
    ) -> Result<Vec<u32>, String> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        if occluders.triangles.is_empty() {
            return Ok(vec![0; items.len()]);
        }
        let limits = self.adapter.limits();
        let most_items = (limits.max_compute_workgroups_per_dimension as usize
            * WORKGROUP as usize)
            .min(items.len());
        let upload = Upload {
            nodes: &occluders.nodes,
            triangles: &occluders.triangles,
            targets: targets.as_flattened(),
            eyes: eyes.as_flattened(),
            widest: most_items as u64,
        };
        let opened = self.open(&upload, limits)?;
        let mut answers = Vec::with_capacity(items.len());
        let mut rays = FEWEST_RAYS;
        while answers.len() < items.len() {
            let left = &items[answers.len()..];
            let began = Instant::now();
            answers.extend(opened.dispatch(&left[..taken(left, rays, most_items)], gap)?);
            let seconds = began.elapsed().as_secs_f64();
            if seconds < SHORT_SECONDS {
                rays = rays.saturating_mul(2);
            } else if seconds > LONG_SECONDS {
                rays = (rays / 2).max(FEWEST_RAYS);
            }
        }
        Ok(answers)
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "the cook's own device for casting rays: its pipeline, buffers and error scope never reach the renderer's"
    )]
    fn open(&self, upload: &Upload<'_>, limits: wgpu::Limits) -> Result<Opened, String> {
        use wgpu::BufferUsages as Usage;
        let (name, _) = self.adapter();
        let size = |words: usize| words as u64 * 4;
        let sizes = [
            ("nodes", size(upload.nodes.len())),
            ("triangles", size(upload.triangles.len())),
            ("targets", size(upload.targets.len())),
            ("eyes", size(upload.eyes.len())),
            ("items", upload.widest * 12),
            ("found", upload.widest * 4),
        ];
        let most = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        if let Some((buffer, bytes)) = sizes.iter().find(|(_, bytes)| *bytes > most) {
            return Err(format!(
                "the {buffer} buffer needs {bytes} bytes; {name} binds at most {most}"
            ));
        }
        let (device, queue) = pollster::block_on(self.adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("cook_rays"),
                required_limits: limits,
                ..Default::default()
            },
        ))
        .map_err(|error| format!("{name}: {error}"))?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cook_rays"),
            source: wgpu::ShaderSource::Wgsl(include_str!("unblocked.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cook_rays"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(format!("{name}: {error}"));
        }
        let filled = |label: &str, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: Usage::STORAGE,
            })
        };
        let empty = |label: &str, size: u64, usage: Usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let run = empty("run", 8, Usage::UNIFORM | Usage::COPY_DST);
        let work = empty("items", upload.widest * 12, Usage::STORAGE | Usage::COPY_DST);
        let found = empty("found", upload.widest * 4, Usage::STORAGE | Usage::COPY_SRC);
        let read = empty("read", upload.widest * 4, Usage::MAP_READ | Usage::COPY_DST);
        let held = [
            filled("nodes", bytemuck::cast_slice(upload.nodes)),
            filled("triangles", bytemuck::cast_slice(upload.triangles)),
            filled("targets", bytemuck::cast_slice(upload.targets)),
            filled("eyes", bytemuck::cast_slice(upload.eyes)),
        ];
        let entries: Vec<wgpu::BindGroupEntry<'_>> = std::iter::once(&run)
            .chain(&held)
            .chain([&work, &found])
            .zip(0..)
            .map(|(buffer, binding)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cook_rays"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(Opened {
            device,
            queue,
            pipeline,
            group,
            run,
            work,
            found,
            read,
        })
    }
}

impl Opened {
    #[expect(
        clippy::disallowed_methods,
        reason = "the cook's own device for casting rays: its writes and submissions never reach the renderer's"
    )]
    fn dispatch(&self, items: &[[u32; 3]], gap: f32) -> Result<Vec<u32>, String> {
        let count = u32::try_from(items.len()).expect("dispatch_items_fit_u32");
        let bytes = u64::from(count) * 4;
        self.queue
            .write_buffer(&self.run, 0, bytemuck::cast_slice(&[count, gap.to_bits()]));
        self.queue
            .write_buffer(&self.work, 0, bytemuck::cast_slice(items));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(count.div_ceil(WORKGROUP), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.found, 0, &self.read, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let slice = self.read.slice(..bytes);
        let (send, receive) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |outcome| {
            let _ = send.send(outcome);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| error.to_string())?;
        receive
            .recv()
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())?;
        let view = slice.get_mapped_range().map_err(|error| error.to_string())?;
        let answers = view
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        drop(view);
        self.read.unmap();
        Ok(answers)
    }
}
