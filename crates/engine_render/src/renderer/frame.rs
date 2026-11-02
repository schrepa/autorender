use std::any::TypeId;

use bevy_ecs::world::World;
use engine_core::{AssetResidency, FrameTime, RenderFrame, RenderStats, View};

use super::*;
use crate::CameraUniform;
use crate::gpu_calls::Handover;
use crate::graph::Bits;
use crate::install::Work;
use crate::machine::Target;
use crate::profiling::{GpuProfiler, Span};

const BATCHES: [&str; 3] = ["batch_0", "batch_1", "batch_2"];
const RAN_ORDERS: usize = 256;
const OPAQUE_MESHES: &str = "opaque_meshes";
const MEMORY: [&str; 18] = [
    "memory.gpu_shadow_bytes",
    "memory.gpu_post_bytes",
    "memory.gpu_environment_bytes",
    "memory.gpu_light_bytes",
    "memory.gpu_impostor_bytes",
    "memory.gpu_terrain_bytes",
    "memory.gpu_transparent_bytes",
    "memory.gpu_temporal_bytes",
    "memory.gpu_presentation_bytes",
    "memory.render_upload_cache_bytes",
    "memory.render_upload_range_bytes",
    "memory.render_staging_capacity_bytes",
    "memory.gpu_pose_staging_bytes",
    "memory.gpu_instance_buffer_bytes",
    "memory.gpu_mesh_payload_bytes",
    "memory.gpu_mesh_arena_bytes",
    "memory.gpu_texture_payload_bytes",
    "memory.gpu_texture_mip_bytes",
];

struct Viewed {
    view_projection: Mat4,
    eye: glam::Vec3,
    detail: glam::Vec3,
    seconds: f32,
    scene: u64,
}

fn mark_between(encoder: &mut wgpu::CommandEncoder, end: Option<Span<'_>>, begin: Option<Span<'_>>) {
    if let Some(end) = end {
        end.mark(encoder, true);
    }
    if let Some(begin) = begin {
        begin.mark(encoder, false);
    }
}

fn mark_span(profiler: &GpuProfiler, encoder: Option<&mut wgpu::CommandEncoder>, name: &'static str, end: bool) {
    if let (Some(encoder), Some(span)) = (encoder, profiler.span(name)) {
        span.mark(encoder, end);
    }
}

fn opaque_span(passes: &[crate::install::PassDecl], plan: &crate::graph::Plan) -> Option<(usize, usize)> {
    let opaque = |planned: &crate::graph::Planned| {
        passes.get(planned.pass.index()).is_some_and(|pass| {
            crate::machine::shared(pass) && crate::machine::pass_layer(pass).is_some_and(|layer| layer <= crate::Layer::Backdrop)
        })
    };
    Some((plan.passes.iter().position(opaque)?, plan.passes.iter().rposition(opaque)?))
}

fn camera_of(
    viewed: &Viewed,
    temporal: Option<(crate::Resolve, Mat4)>,
    region: crate::targets::FrameRegion,
    (previous_seconds, wind): (f32, engine_core::ViewWind),
) -> CameraUniform {
    let mut camera = match temporal {
        Some((resolve, previous)) => {
            let (width, height) = region.render.pixels();
            let clip = crate::gpu_data::jitter_clip(resolve.offset.to_array(), width, height);
            let mut camera = CameraUniform::temporal(
                viewed.view_projection,
                viewed.eye,
                viewed.seconds,
                previous,
                clip,
                (width, height),
                0,
            );
            camera.options[2] = resolve.turn - resolve.turn.floor();
            camera
        }
        None => CameraUniform::exact(viewed.view_projection, viewed.eye, viewed.seconds),
    };
    camera.set_region(viewed.view_projection, region);
    camera.set_detail(viewed.detail);
    camera.wind = [
        wind.velocity[0],
        wind.velocity[1],
        wind.gusts,
        if temporal.is_some() {
            previous_seconds
        } else {
            viewed.seconds
        },
    ];
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
        run.assets.impostors.deliver(world);
        let mut draws = engine_core::PartDraws::default();
        for (part, drawn) in run.machine.declared.parts.iter().zip(&run.part_draws) {
            draws.add(part.ty, *drawn);
        }
        if world.get_resource::<engine_core::PartDraws>() != Some(&draws) {
            world.insert_resource(draws);
        }
        let mut faults = engine_core::PartFaults::default();
        for (part, fault) in run.machine.declared.parts.iter().zip(&run.faulted) {
            if fault.is_some() {
                faults.add(part.ty);
            }
        }
        if world.get_resource::<engine_core::PartFaults>() != Some(&faults) {
            world.insert_resource(faults);
        }
        if let Some(mut gates) = world.get_resource_mut::<engine_registry::Gates>() {
            for (fault, gate) in run.faulted.iter().zip(&run.gates) {
                if let (Some(_), Some(gate)) = (fault, gate)
                    && !gates.faulted(*gate)
                {
                    gates.fault(*gate);
                }
            }
        }
        self.follow_gates(world);
    }

    pub fn follow_gates(&mut self, world: &mut World) {
        let Some(run) = self.run.as_mut() else {
            return;
        };
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
            if !*on && was {
                run.release_part(index as u32);
            }
        }
        run.live = live;
    }

    fn mark(&self, encoder: Option<&mut wgpu::CommandEncoder>, end: Option<&'static str>, begin: Option<&'static str>) {
        let Some(encoder) = encoder else {
            return;
        };
        let span = |name: Option<&'static str>| self.core.profiler.span(name?);
        mark_between(encoder, span(end), span(begin));
    }

    fn frame_encoder(&self, device: &wgpu::Device, batch: u8) -> wgpu::CommandEncoder {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine/frame-encoder"),
        });
        let busy = |batch: u8| self.core.profiler.busy(BATCHES.get(usize::from(batch))?);
        mark_between(&mut encoder, None, busy(batch));
        encoder
    }

    fn close_batch(&self, encoder: &mut wgpu::CommandEncoder, batch: u8) {
        let busy = self
            .core
            .profiler
            .busy(BATCHES.get(usize::from(batch)).copied().unwrap_or_default());
        mark_between(encoder, busy, None);
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn render(&mut self, frame: RenderFrame<'_>, display: bool) -> Result<RenderStats, Error> {
        let _profile = engine_profile::span("render", "render_snapshot");
        let _row = engine_profile::timed(engine_profile::row("render.total"));
        self.check_health()?;
        let gpu_mode = frame.gpu().is_some();
        let world = frame.world();
        let view = world.get_resource::<View>();
        let debug = world
            .get_resource::<crate::DebugShown>()
            .is_some_and(|shown| shown.owner.is_some());
        let temporal_frame = display && gpu_mode && !debug && view.is_some_and(View::temporal);
        let region = self.core.display.plan_region(view.map(View::render_size));
        if self.core.display.suspended {
            return Ok(RenderStats::default());
        }
        let row_acquire = engine_profile::timed(engine_profile::row("render.acquire"));
        let Some(surface) = self.core.display.acquire(&self.core.device)? else {
            return Ok(RenderStats::default());
        };
        drop(row_acquire);
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        let device = self.core.device.clone();
        let profile_upload = engine_profile::span("render", "asset_uploads");
        let row_upload = engine_profile::timed(engine_profile::row("render.uploads"));
        let mesh_bytes = run.assets.meshes.sync(&device, &self.core.queue, world);
        let mut mips = None;
        let (mut texture_bytes, mut mip_levels) =
            run.assets
                .materials
                .sync(&device, &self.core.queue, world, &mut mips);
        run.assets.impostors.run(
            &device,
            &self.core.queue,
            world,
            (&run.assets.meshes, &run.assets.materials),
            &mut mips,
        );
        if let Some(encoder) = mips.take() {
            self.core.queue.hand(Handover::from(encoder));
        }
        drop(profile_upload);
        drop(row_upload);
        run.frame += 1;
        run.parity = (run.frame % 2) as usize;
        run.debug = debug;
        run.temporal = temporal_frame;
        run.drawn = 0;
        run.uploads.counted = 0;
        run.uploads.memory.clear();
        run.queued.ending.clear();
        run.part_draws.fill(0);
        run.region = region.upscales().then(|| region.render.pixels());
        run.asked_frame = world
            .get_resource::<FrameTime>()
            .map_or(run.frame, |time| time.frame);
        let settings = world
            .get_resource::<crate::RenderSettings>()
            .cloned()
            .unwrap_or_default();
        run.readback_slots = settings.readback_slots.max(1) as usize;
        run.readback_slot_bytes = settings.readback_slot_bytes;
        let faulted: Vec<engine_registry::GateId> = run
            .faulted
            .iter()
            .zip(&run.gates)
            .filter(|(fault, _)| fault.is_some())
            .filter_map(|(_, gate)| *gate)
            .collect();
        let switches = super::install::switches_of(&run.machine, world, &faulted);
        run.set_switches(switches);
        run.drop_unfilled();
        let viewed = match view {
            Some(view) => Viewed {
                view_projection: view.view_projection(),
                eye: view.state().eye(),
                detail: view.detail_eye(),
                seconds: world.get_resource::<FrameTime>().map_or(0.0, |time| time.elapsed_seconds as f32),
                scene: view.scene(),
            },
            None => Viewed {
                view_projection: Mat4::IDENTITY,
                eye: glam::Vec3::ZERO,
                detail: glam::Vec3::ZERO,
                seconds: 0.0,
                scene: 0,
            },
        };
        let previous_seconds = self.seconds.replace(viewed.seconds).unwrap_or(viewed.seconds);
        let resolve = world.get_resource::<crate::Resolve>().copied().unwrap_or_default();
        let size = region.display.pixels();
        let temporal = temporal_frame.then(|| {
            let previous = self
                .previous
                .filter(|(_, known, scene)| *known == size && *scene == viewed.scene)
                .map_or(viewed.view_projection, |(previous, _, _)| previous);
            (resolve, previous)
        });
        let wind = world.get_resource::<engine_core::ViewWind>().copied().unwrap_or_default();
        run.cameras[0] = camera_of(&viewed, temporal, region, (previous_seconds, wind));
        self.previous = temporal.map(|_| (viewed.view_projection, size, viewed.scene));
        run.post = crate::shared::post_uniform(
            viewed.view_projection,
            viewed.eye,
            viewed.detail,
            region.render.pixels(),
            region.display.pixels(),
        );
        let canvas = surface
            .texture()
            .create_view(&wgpu::TextureViewDescriptor::default());
        run.size_targets(region.display.pixels(), Some(canvas));
        run.canvas_texture = Some(surface.texture().clone());
        let profile_writes = engine_profile::span("render", "queue_writes");
        let row_writes = engine_profile::timed(engine_profile::row("render.writes"));
        run.runs = Bits::of(run.machine.graph.passes());
        for at in run.machine.declared.passes.len()..run.machine.graph.passes() {
            run.runs.insert(at);
        }
        for index in 0..run.parts.len() {
            if !run.drawable(index as u32) {
                continue;
            }
            let Some(mut held) = run.parts[index].take() else {
                continue;
            };
            run.part = index as u32;
            let _owner = run.machine.declared.parts.get(index).map(crate::install::PartInfo::owned);
            if let Err(error) = held.prepare(world, run) {
                run.fault(error);
            }
            run.parts[index] = Some(held);
        }
        if let Some(library) = world.get_resource::<engine_core::AssetLibrary>() {
            for texture in std::mem::take(&mut run.wanted_textures) {
                if let Some(source) = library.texture(texture) {
                    let (bytes, levels) = run.assets.materials.upload_texture(&device, &self.core.queue, (texture, true), source, &mut mips);
                    texture_bytes += bytes;
                    mip_levels += levels;
                }
            }
        }
        if let Some(encoder) = mips {
            self.core.queue.hand(Handover::from(encoder));
        }
        if let Some(encoder) = run.queued.before.take() {
            self.core.queue.hand(Handover::from(encoder));
        }
        if let Some(error) = run.queued.refused.take() {
            return Err(error);
        }
        let mut camera_bytes = run.write_cameras();
        run.write_post();
        let bytes = piece_bytes(world, &run.piece_layout);
        camera_bytes += run.write_pieces(bytes);
        run.refresh_groups();
        drop(profile_writes);
        drop(row_writes);
        self.core.profiler.begin_frame(&device);
        let capturing = self.core.profiler.capturing();
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        run.merge = !capturing;
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
            .plan(&run.machine.graph, &run.runs, &outputs, crate::graph::PLANS)?;
        run.plan = Some(std::sync::Arc::clone(&plan));
        let profile_encoding = engine_profile::span("render", "command_encoding");
        let row_encoding = engine_profile::timed(engine_profile::row("render.encoding"));
        let mut encoder = self.frame_encoder(&device, 0);
        self.core.display.time_begin(&mut encoder);
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        run.encoder = Some(encoder);
        let mut batch = 0;
        let mut timed: Option<&'static str> = None;
        let mut parted: Option<(&'static str, u32)> = None;
        let mut ran: Vec<&'static str> = Vec::new();
        let mut ran_last: Option<u32> = None;
        let declared = run.machine.declared.passes.len();
        let opaque = opaque_span(&run.machine.declared.passes, &plan);
        let mut opaque_open = false;
        for (at, planned) in plan.passes.iter().enumerate() {
            if opaque_open && opaque.is_some_and(|(_, last)| at > last) {
                let Some(run) = self.run.as_mut() else { break };
                run.leave();
                mark_span(&self.core.profiler, run.encoder.as_mut(), OPAQUE_MESHES, true);
                opaque_open = false;
            }
            if planned.batch != batch {
                let Some(run) = self.run.as_mut() else { break };
                run.leave();
                let mut ended = run.encoder.take();
                self.mark(ended.as_mut(), timed.take(), None);
                if let Some(mut encoder) = ended {
                    self.close_batch(&mut encoder, batch);
                    self.core.queue.hand(Handover::from(encoder));
                }
                let encoder = self.frame_encoder(&device, planned.batch);
                let Some(run) = self.run.as_mut() else { break };
                run.encoder = Some(encoder);
                batch = planned.batch;
            }
            let index = planned.pass.index();
            let Some(run) = self.run.as_mut() else { break };
            if index >= declared {
                if let Some(&(layer, view)) = run.machine.view_copies.get(index - declared) {
                    run.copy_view(layer, view);
                }
                continue;
            }
            let pass = &run.machine.declared.passes[index];
            let (part, work) = (pass.part, pass.work);
            let elsewhere = pass
                .claims
                .iter()
                .any(|point| *point != TypeId::of::<crate::MainView>());
            if !run.drawable(part) {
                continue;
            }
            run.part = part;
            run.pass = index as u32;
            run.planned_at = at;
            if let Work::Clear(buffer) = work {
                run.queued.cleared.push(buffer);
                continue;
            }
            let continues = ran_last.is_some_and(|last| last != index as u32 && run.same_group(last, index as u32));
            if !continues {
                ran.push(run.group_of(index as u32));
            }
            ran_last = Some(index as u32);
            let wanted = run.span_of(index as u32);
            let keeps = match (parted, wanted) {
                (Some((open, owner)), Some(name)) => open == name && owner == part,
                (Some((_, owner)), None) => owner == part && matches!(work, Work::Copy { .. }),
                (None, _) => false,
            };
            if run.enter(index as u32) {
                let span = run.stage_of(index as u32).or_else(|| self.spans.get(index).copied());
                let mut encoder = run.encoder.take();
                if !keeps && let Some((open, _)) = parted.take() {
                    mark_span(&self.core.profiler, encoder.as_mut(), open, true);
                }
                if !(continues && span == timed) {
                    self.mark(encoder.as_mut(), timed, span);
                }
                timed = span;
                if !opaque_open && opaque.is_some_and(|(first, _)| first == at) {
                    mark_span(&self.core.profiler, encoder.as_mut(), OPAQUE_MESHES, false);
                    opaque_open = true;
                }
                if !keeps && let Some(began) = wanted {
                    mark_span(&self.core.profiler, encoder.as_mut(), began, false);
                    parted = Some((began, part));
                }
                let Some(run) = self.run.as_mut() else { break };
                run.encoder = encoder;
                run.group(index as u32);
            }
            let Some(run) = self.run.as_mut() else { break };
            run.flush_clears();
            let _owner = run.machine.declared.parts.get(part as usize).map(crate::install::PartInfo::owned);
            match work {
                Work::Copy { .. } => run.copy(index as u32),
                Work::Clear(_) => {}
                Work::Render | Work::Shared(_) => {
                    let Some(mut held) = run.parts[part as usize].take() else {
                        continue;
                    };
                    let drawn = held.encode(index as u32, run, &self.core.profiler);
                    let next = plan.passes.get(at + 1).map(|next| next.pass.index() as u32);
                    if !next.is_some_and(|next| next != index as u32 && run.same_group(index as u32, next)) {
                        run.close_encode_span(&self.core.profiler);
                    }
                    if let Some(sum) = run.part_draws.get_mut(part as usize) {
                        *sum += drawn;
                    }
                    if !elsewhere {
                        run.drawn += drawn;
                    }
                    run.parts[part as usize] = Some(held);
                }
            }
        }
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        run.close_encode_span(&self.core.profiler);
        run.flush_clears();
        run.leave();
        let mut encoder = run.encoder.take().unwrap_or_else(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/frame-encoder"),
            })
        });
        if opaque_open {
            mark_span(&self.core.profiler, Some(&mut encoder), OPAQUE_MESHES, true);
        }
        if let Some((open, _)) = parted.take() {
            mark_span(&self.core.profiler, Some(&mut encoder), open, true);
        }
        self.mark(Some(&mut encoder), timed, None);
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        run.encoder = Some(encoder);
        run.copy_at_end();
        run.record_readbacks();
        let mut encoder = run.encoder.take().unwrap_or_else(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/frame-encoder"),
            })
        });
        self.core.display.time_end(&device, &mut encoder);
        self.close_batch(&mut encoder, batch);
        self.core.profiler.end_frame(&mut encoder);
        drop(profile_encoding);
        drop(row_encoding);
        let profile_submit = engine_profile::span("render", "submit_and_present");
        let row_submit = engine_profile::timed(engine_profile::row("render.submit"));
        self.core.queue.hand(Handover::from(encoder));
        self.core.queue.settle();
        self.core.profiler.submitted();
        surface.present(&self.core.queue);
        let pipelines_created = self.core.pipeline_counts.frame_submitted(&self.core.queue);
        let render_passes = self.ran_passes(ran);
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        run.assets.impostors.after_submit();
        run.queued.staging.after_submit();
        run.map_readbacks();
        crate::platform::poll(&device);
        drop(profile_submit);
        drop(row_submit);
        let (resident_textures, resident_texture_bytes, mip_bytes) =
            run.assets.materials.texture_totals();
        let resident_mesh_bytes = world
            .get_resource::<AssetResidency>()
            .map_or(0, |residency| {
                residency
                    .meshes()
                    .filter(|(handle, _)| run.assets.meshes.contains(*handle))
                    .map(|(_, mesh)| mesh.byte_len())
                    .sum()
            });
        let stats = RenderStats {
            draw_calls: run.drawn,
            upload_bytes: camera_bytes + run.uploads.counted + mesh_bytes + texture_bytes,
            resident_meshes: run.assets.meshes.len() as u32,
            mesh_upload_bytes: mesh_bytes,
            resident_mesh_bytes,
            resident_textures,
            texture_upload_bytes: texture_bytes,
            resident_texture_bytes,
            submitted: true,
            pipelines: self.core.pipeline_counts.created(),
            pipelines_created,
            render_passes,
            linked: self.linked,
            readback_overflows: run.overflows,
            ..RenderStats::default()
        };
        let immediate =
            self.core.queue.take_immediate() + std::mem::take(&mut run.queued.immediate);
        rebuilt_counters(&mut run.rebuilt);
        if engine_profile::active() {
            stat_counters(&stats);
            engine_profile::counter("render.immediate_submissions", f64::from(immediate));
            engine_profile::counter("render.texture_mip_levels", f64::from(mip_levels));
            engine_profile::counter("render.camera_write_bytes", camera_bytes as f64);
            engine_profile::counter("render.gpu_presentation", f64::from(u8::from(gpu_mode)));
            engine_profile::counter("render.camera_upload_reused", f64::from(u8::from(camera_bytes == 0)));
            for name in MEMORY {
                let held = match name {
                    "memory.gpu_impostor_bytes" => run.assets.impostors.capacity_bytes(),
                    "memory.render_upload_cache_bytes" => size_of::<Option<CameraUniform>>() as u64,
                    "memory.render_staging_capacity_bytes" => run.assets.meshes.staging_bytes(),
                    "memory.gpu_pose_staging_bytes" => run.queued.staging.capacity_bytes(),
                    "memory.gpu_mesh_payload_bytes" => stats.resident_mesh_bytes,
                    "memory.gpu_mesh_arena_bytes" => run.assets.meshes.capacity_bytes(),
                    "memory.gpu_texture_payload_bytes" => stats.resident_texture_bytes,
                    "memory.gpu_texture_mip_bytes" => mip_bytes,
                    _ => 0,
                };
                let parts: u64 = run.uploads.memory.iter().filter(|(known, _)| *known == name).map(|(_, bytes)| *bytes).sum();
                engine_profile::counter(name, (held + parts) as f64);
            }
        }
        Ok(stats)
    }

    fn ran_passes(&mut self, ran: Vec<&'static str>) -> engine_core::Names {
        let names = match self.ran.iter().find(|&&known| known == ran.as_slice()) {
            Some(&known) => known,
            None if self.ran.len() < RAN_ORDERS => {
                let leaked: &'static [&'static str] = Box::leak(ran.into_boxed_slice());
                self.ran.push(leaked);
                leaked
            }
            None => {
                let names = self.passes;
                let bits = names
                    .iter()
                    .enumerate()
                    .filter(|(_, name)| ran.contains(name))
                    .fold(0_u64, |bits, (at, _)| bits | 1_u64.checked_shl(at as u32).unwrap_or(0));
                return engine_core::Names { bits, names };
            }
        };
        engine_core::Names {
            bits: 1_u64
                .checked_shl(names.len() as u32)
                .map_or(u64::MAX, |bit| bit - 1),
            names,
        }
    }
}

fn rebuilt_counters(rebuilt: &mut crate::encode::Rebuilt) {
    engine_profile::counter("render.group_rebuilds", f64::from(rebuilt.groups));
    engine_profile::counter("render.buffer_recreations", f64::from(rebuilt.buffers));
    engine_profile::counter("render.texture_recreations", f64::from(rebuilt.textures));
    engine_profile::counter("render.draw_compiles", f64::from(rebuilt.compiles));
    engine_profile::tally("render.group_rebuilds", u64::from(rebuilt.groups));
    engine_profile::tally("render.buffer_recreations", u64::from(rebuilt.buffers));
    engine_profile::tally("render.texture_recreations", u64::from(rebuilt.textures));
    engine_profile::tally("render.draw_compiles", u64::from(rebuilt.compiles));
    rebuilt.reasons.sort_unstable();
    let mut reasons = std::mem::take(&mut rebuilt.reasons);
    for reason in reasons.chunk_by(|a, b| a == b) {
        engine_profile::counter(rebuilt.reason_name(reason[0]), reason.len() as f64);
        engine_profile::tally(rebuilt.reason_name(reason[0]), reason.len() as u64);
    }
    reasons.clear();
    rebuilt.reasons = reasons;
    rebuilt.groups = 0;
    rebuilt.buffers = 0;
    rebuilt.textures = 0;
    rebuilt.compiles = 0;
}

#[cfg(feature = "profiling")]
#[expect(clippy::expect_used, reason = "render stats always serialize")]
fn stat_counters(stats: &RenderStats) {
    use engine_core::collections::HashMap;
    thread_local! {
        static NAMES: std::cell::RefCell<HashMap<String, &'static str>> =
            std::cell::RefCell::new(HashMap::default());
    }
    fn visit(path: &mut String, value: &serde_json::Value) {
        let number = match value {
            serde_json::Value::Object(fields) => {
                for (key, value) in fields {
                    let length = path.len();
                    path.push('.');
                    path.push_str(key);
                    visit(path, value);
                    path.truncate(length);
                }
                return;
            }
            serde_json::Value::Number(number) => number.as_f64(),
            serde_json::Value::Bool(flag) => Some(f64::from(u8::from(*flag))),
            serde_json::Value::Null
            | serde_json::Value::String(_)
            | serde_json::Value::Array(_) => None,
        };
        if let Some(number) = number {
            let name = NAMES.with_borrow_mut(|names| {
                *names
                    .entry(path.clone())
                    .or_insert_with(|| Box::leak(path.clone().into_boxed_str()))
            });
            engine_profile::counter(name, number);
        }
    }
    let value = serde_json::to_value(stats).expect("render_stats_serialize");
    visit(&mut "render".to_owned(), &value);
}

#[cfg(not(feature = "profiling"))]
fn stat_counters(_: &RenderStats) {}

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
