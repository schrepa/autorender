use bevy_ecs::world::World;
use engine_core::{RenderFrame, RenderStats};

use super::*;
use crate::CameraUniform;
use crate::graph::Bits;
use crate::install::Work;
use crate::machine::Target;

fn camera_of(
    snapshot: &engine_core::RenderSnapshot,
    resolve: Option<crate::Resolve>,
    previous: Option<Mat4>,
    region: crate::targets::FrameRegion,
    seconds: (f32, f32),
) -> CameraUniform {
    let temporal = resolve.filter(|resolve| *resolve != crate::Resolve::default());
    let mut camera = match temporal {
        Some(resolve) => {
            let mut camera = CameraUniform::temporal(
                snapshot.view_projection,
                snapshot.camera_position,
                snapshot.elapsed_seconds,
                previous.unwrap_or(snapshot.view_projection),
                resolve.offset.to_array(),
                region.render.pixels(),
                0,
            );
            camera.options[2] = resolve.turn - resolve.turn.floor();
            camera
        }
        None => CameraUniform::exact(
            snapshot.view_projection,
            snapshot.camera_position,
            snapshot.elapsed_seconds,
        ),
    };
    camera.set_region(snapshot.view_projection, region);
    camera.set_detail(snapshot.detail_eye);
    camera.wind = [0.0, 0.0, 0.0, if temporal.is_some() { seconds.0 } else { seconds.1 }];
    camera
}

impl Renderer {
    pub fn render_frame(&mut self, frame: RenderFrame<'_>) -> Result<RenderStats, Error> {
        self.render(frame, false)
    }

    pub fn render_display_frame(&mut self, frame: RenderFrame<'_>) -> Result<RenderStats, Error> {
        self.render(frame, true)
    }

    pub fn deliver(&mut self, world: &mut World) {
        let Some(run) = self.run.as_mut() else {
            return;
        };
        run.deliver(world);
        let gates = world.get_resource::<engine_registry::Gates>();
        let live: Vec<bool> = run
            .gates
            .iter()
            .map(|gate| gate.is_none_or(|gate| gates.is_some_and(|gates| gates.on(gate))))
            .collect();
        for (index, on) in live.iter().enumerate() {
            let was = run.live.get(index).copied().unwrap_or(true);
            if *on && !was
                && let Some(Some(held)) = run.parts.get_mut(index)
            {
                held.refresh(world);
            }
        }
        run.live = live;
    }

    fn render(&mut self, frame: RenderFrame<'_>, display: bool) -> Result<RenderStats, Error> {
        let _profile = engine_profile::span("render", "render_frame");
        self.check_health()?;
        let snapshot = frame.snapshot();
        let world = frame.world();
        let debug = world
            .get_resource::<crate::DebugShown>()
            .is_some_and(|shown| shown.owner.is_some());
        let region = self.core.display.plan_region(display && !debug);
        if self.core.display.suspended {
            return Ok(RenderStats::default());
        }
        let Some(surface) = self.core.display.acquire(&self.core.device)? else {
            return Ok(RenderStats::default());
        };
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        let device = self.core.device.clone();
        let mesh_bytes = upload_meshes(&device, &self.core.queue, &mut run.assets, &snapshot.meshes);
        let (texture_bytes, mips) = run.assets.materials.upload_frame(&device, &self.core.queue, snapshot);
        if let Some(encoder) = mips {
            self.core.queue.hand(crate::gpu_calls::Handover::from(encoder));
        }
        run.frame += 1;
        run.parity = (run.frame % 2) as usize;
        run.debug = debug;
        run.drawn = 0;
        run.region = region.upscales().then(|| region.render.pixels());
        run.drop_unfilled();
        let seconds = snapshot.elapsed_seconds;
        let previous_seconds = self.seconds.replace(seconds).unwrap_or(seconds);
        let resolve = world.get_resource::<crate::Resolve>().copied();
        run.cameras[0] = camera_of(snapshot, resolve, self.previous, region, (previous_seconds, seconds));
        self.previous = Some(snapshot.view_projection);
        run.post = crate::shared::post_uniform(
            snapshot.view_projection,
            snapshot.camera_position,
            snapshot.detail_eye,
            region.render.pixels(),
            region.display.pixels(),
        );
        let canvas = surface
            .texture()
            .create_view(&wgpu::TextureViewDescriptor::default());
        run.size_targets(region.display.pixels(), Some(canvas));
        self.core.profiler.begin_frame(&device);
        run.runs = Bits::of(run.machine.graph.passes());
        for at in run.machine.declared.passes.len()..run.machine.graph.passes() {
            run.runs.insert(at);
        }
        for index in 0..run.parts.len() {
            let live = run.live.get(index).copied().unwrap_or(true)
                && run.faulted.get(index).is_some_and(Option::is_none);
            if !live {
                continue;
            }
            let Some(mut held) = run.parts[index].take() else {
                continue;
            };
            run.part = index as u32;
            if let Err(error) = held.prepare(world, run) {
                run.fault(error);
            }
            run.parts[index] = Some(held);
        }
        run.write_cameras();
        run.write_post();
        let bytes = piece_bytes(world, &run.piece_layout);
        run.write_pieces(bytes);
        run.refresh_groups();
        let mut outputs = Bits::of(run.machine.graph.resources());
        if let Some(canvas) = run.machine.ids.target(Target::Canvas) {
            outputs.insert(canvas.index());
        }
        for asked in &run.asked {
            if let Some(id) = run.machine.ids.buffers.get(asked.buffer as usize) {
                outputs.insert(id.index());
            }
        }
        let plan = self
            .plans
            .plan(&run.machine.graph, &run.runs, &outputs, crate::graph::PLANS)?
            .clone();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine/frame-encoder"),
        });
        self.core.display.time_begin(&mut encoder);
        run.encoder = Some(encoder);
        let mut batch = 0;
        let declared = run.machine.declared.passes.len();
        for planned in &plan.passes {
            if planned.batch != batch {
                run.end_render();
                if let Some(encoder) = run.encoder.take() {
                    self.core.queue.hand(crate::gpu_calls::Handover::from(encoder));
                }
                run.encoder = Some(device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine/frame-encoder"),
                }));
                batch = planned.batch;
            }
            let index = planned.pass.index();
            if index >= declared {
                if let Some(&(layer, view)) = run.machine.view_copies.get(index - declared) {
                    run.copy_view(layer, view);
                }
                continue;
            }
            let pass = &run.machine.declared.passes[index];
            let (part, work) = (pass.part, pass.work);
            let live = run.live.get(part as usize).copied().unwrap_or(true)
                && run.faulted.get(part as usize).is_some_and(Option::is_none);
            if !live {
                continue;
            }
            run.part = part;
            run.pass = index as u32;
            match work {
                Work::Copy { .. } => run.copy(index as u32),
                Work::Clear(buffer) => run.clear_buffer(buffer),
                Work::Render | Work::Shared(_) => {
                    let Some(mut held) = run.parts[part as usize].take() else {
                        continue;
                    };
                    run.planned = Some(planned.clone());
                    #[cfg(any(not(target_arch = "wasm32"), feature = "profiling"))]
                    if let Some(encoder) = run.encoder.as_mut()
                        && run.open.is_none()
                    {
                        encoder.insert_debug_marker(pass_label(&run.machine, index));
                    }
                    run.drawn += held.encode(index as u32, run);
                    run.parts[part as usize] = Some(held);
                }
            }
        }
        run.end_render();
        run.record_readbacks();
        let mut encoder = run.encoder.take().unwrap_or_else(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/frame-encoder"),
            })
        });
        self.core.display.time_end(&device, &mut encoder);
        self.core.profiler.end_frame(&mut encoder);
        self.core.queue.hand(crate::gpu_calls::Handover::from(encoder));
        self.core.queue.settle();
        self.core.profiler.submitted();
        surface.present(&self.core.queue);
        run.map_readbacks();
        crate::platform::poll(&device);
        Ok(RenderStats {
            draw_calls: run.drawn,
            upload_bytes: mesh_bytes + texture_bytes,
            mesh_upload_bytes: mesh_bytes,
            texture_upload_bytes: texture_bytes,
            resident_meshes: run.assets.meshes.len() as u32,
            submitted: true,
            pipelines: run.machine.pipelines.iter().map(|pipeline| pipeline.made.len() as u32).sum(),
            ..RenderStats::default()
        })
    }
}

fn pass_label(machine: &crate::machine::Machine, index: usize) -> &'static str {
    machine.declared.passes.get(index).map_or("", |pass| pass.name)
}

fn piece_bytes(world: &World, layout: &[(usize, crate::pieces::PieceAt)]) -> Vec<u8> {
    let Some(registry) = world.get_resource::<engine_registry::Registry>() else {
        return Vec::new();
    };
    let entries: Vec<_> = registry.pieces().map(|(_, entry)| entry).collect();
    let end = layout.iter().map(|(_, at)| at.offset + at.size).max().unwrap_or(0);
    let mut bytes = vec![0_u8; end as usize];
    for &(index, at) in layout {
        let Some(found) = entries.get(index).and_then(|entry| entry.bytes(world)) else {
            continue;
        };
        let start = at.offset as usize;
        let length = found.len().min(at.size as usize);
        bytes[start..start + length].copy_from_slice(&found[..length]);
    }
    bytes
}
