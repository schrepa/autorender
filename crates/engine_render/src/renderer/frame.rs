
use bevy_ecs::world::World;
use engine_core::{AssetResidency, FrameTime, PrimaryView, Readback, RenderFrame, RenderStats};

use super::*;
use crate::CameraUniform;
use crate::encode::{Queued, Recording};
use crate::gpu_calls::Handover;
use crate::graph::Bits;
use crate::install::Work;
use crate::machine::Target;
use crate::profiling::{GpuProfiler, Span};
use crate::readback::{Arrival, Reader};

const BATCHES: [&str; 3] = ["batch_0", "batch_1", "batch_2"];
const OPAQUE_MESHES: &str = "opaque_meshes";
const MEMORY: [&str; 9] = [
    "memory.gpu_impostor_bytes",
    "memory.gpu_previous_pose_bytes",
    "memory.gpu_presentation_bytes",
    "memory.render_staging_capacity_bytes",
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
}

fn mark_between(
    encoder: &mut wgpu::CommandEncoder,
    end: Option<Span<'_>>,
    begin: Option<Span<'_>>,
) {
    if let Some(end) = end {
        end.mark(encoder, true);
    }
    if let Some(begin) = begin {
        begin.mark(encoder, false);
    }
}

fn mark_span(
    profiler: &GpuProfiler,
    encoder: Option<&mut wgpu::CommandEncoder>,
    name: &'static str,
    end: bool,
) {
    if let (Some(encoder), Some(span)) = (encoder, profiler.span(name)) {
        span.mark(encoder, end);
    }
}

fn opaque_span(
    passes: &[crate::install::PassDecl],
    plan: &crate::graph::Plan,
) -> Option<(usize, usize)> {
    let opaque = |planned: &crate::graph::Planned| {
        passes.get(planned.pass.index()).is_some_and(|pass| {
            crate::machine::shared(pass)
                && crate::machine::pass_layer(pass)
                    .is_some_and(|layer| layer <= engine_core::AttachPoint::Sky)
        })
    };
    Some((
        plan.passes.iter().position(opaque)?,
        plan.passes.iter().rposition(opaque)?,
    ))
}

fn camera_of(
    viewed: &Viewed,
    facts: &engine_core::ViewFacts,
    previous_seconds: f32,
) -> CameraUniform {
    let region = view_region(facts);
    let (width, height) = region.render.pixels();
    let clip = facts.jitter.map_or([0.0; 2], |offset| {
        crate::gpu_data::jitter_clip(offset.to_array(), width, height)
    });
    let mut camera = CameraUniform::viewed(
        (viewed.view_projection, facts.previous_view_projection),
        viewed.eye,
        (viewed.seconds, previous_seconds),
        clip,
        (width, height),
    );
    camera.noise_phase = facts.noise - facts.noise.floor();
    camera.set_region(viewed.view_projection, region);
    camera.set_detail(viewed.detail);
    camera
}

impl Renderer {
    pub(super) fn collect(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        crate::platform::run_arrived_maps();
        let Some(run) = self.run.as_mut() else {
            return;
        };
        let (display, profiler) = (&mut self.core.display, &mut self.core.profiler);
        let impostors = &mut run.assets.impostors;
        let mut parts = Vec::new();
        run.reading.list
            .take(|reader, written, deliver, arrival| match (reader, arrival) {
                (Reader::Part { part, tag }, Arrival::Read(bytes)) => {
                    if let Some(deliver) = deliver {
                        let read = Readback {
                            value: bytes.to_vec(),
                            written,
                        };
                        parts.push(((part, tag), read, deliver));
                    }
                }
                (Reader::Part { .. }, Arrival::Failed) => {}
                (Reader::FrameTimer, arrival) => display.timed(written, arrival),
                (Reader::Profiler(slot), arrival) => profiler.arrived(slot, arrival),
                (Reader::Impostor(serial), arrival) => impostors.arrived(serial, arrival),
            });
        for (asked, read, deliver) in parts {
            run.arrived(asked, read, deliver);
        }
    }

    pub fn deliver(&mut self, world: &mut World) {
        self.fault_bricks(world);
        self.recover(world);
        self.collect();
        let Some(run) = self.run.as_mut() else {
            return;
        };
        run.deliver(world);
        run.assets.impostors.deliver(world);
        let refused = run.assets.materials.take_refused();
        if !refused.is_empty()
            && let Some(mut registry) = world.get_resource_mut::<engine_registry::Registry>()
        {
            for error in refused {
                registry.report(error);
            }
        }
        let mut draws = engine_core::PartDraws::default();
        for (part, drawn) in run.machine.declared.parts.iter().zip(&run.tallies.part_draws) {
            draws.add(part.ty, *drawn);
        }
        if world.get_resource::<engine_core::PartDraws>() != Some(&draws) {
            world.insert_resource(draws);
        }
        for (part, error) in std::mem::take(&mut run.installs.faults) {
            let brick = run.installs.gates.get(part as usize).copied().flatten().and_then(|gate| {
                world
                    .get_resource::<engine_registry::BrickStates>()
                    .and_then(|states| states.brick(gate))
            });
            match brick {
                Some(brick) => engine_registry::fault(world, brick, error),
                None => {
                    if let Some(mut registry) = world.get_resource_mut::<engine_registry::Registry>() {
                        registry.report(error);
                    }
                }
            }
        }
        self.follow_states(world);
    }

    pub fn follow_states(&mut self, world: &mut World) {
        let Some(run) = self.run.as_mut() else {
            return;
        };
        run.settle_parts(world.get_resource::<engine_registry::BrickStates>());
        for part in 0..run.parts.len() {
            let Some(gate) = run.installs.gates.get(part).copied().flatten() else {
                continue;
            };
            let states = world.get_resource::<engine_registry::BrickStates>();
            let (Some(count), Some(brick)) = (
                states.and_then(|states| states.installed(gate)),
                states.and_then(|states| states.brick(gate)),
            ) else {
                continue;
            };
            if run.installs.turned.get(part).copied().flatten().is_some() {
                continue;
            }
            match super::install::reinstall(run, world, part as u32) {
                Ok(held) => {
                    run.parts[part] = Some(held);
                    run.installs.turned[part] = Some(count);
                }
                Err(error) => engine_registry::fault(world, brick, error),
            }
        }
    }

    fn mark(
        &self,
        encoder: Option<&mut wgpu::CommandEncoder>,
        end: Option<&'static str>,
        begin: Option<&'static str>,
    ) {
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

    pub fn acquire_frame(&mut self) -> Result<bool, Error> {
        let _row = engine_profile::timed(engine_profile::row("render.acquire"));
        self.check_health()?;
        drop(self.surface.take());
        if self.core.display.suspended() || self.run.is_none() || !self.usable() {
            return Ok(false);
        }
        self.surface = self.core.display.acquire(&self.core.device)?;
        Ok(self.surface.is_some())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub fn render_frame(&mut self, frame: RenderFrame<'_>) -> Result<RenderStats, Error> {
        let _profile = engine_profile::span("render", "render_snapshot");
        let _row = engine_profile::timed(engine_profile::row("render.total"));
        let Some(surface) = self.surface.take() else {
            return Ok(RenderStats::default());
        };
        let world = frame.world();
        let view = world.get_resource::<PrimaryView>();
        let region = self.core.display.plan_region(view.map(PrimaryView::render_size));
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        let device = self.core.device.clone();
        let profile_upload = engine_profile::span("render", "asset_uploads");
        let row_upload = engine_profile::timed(engine_profile::row("render.uploads"));
        let mesh_bytes = run.assets.meshes.sync(&device, &self.core.queue, world);
        let mut mips = None;
        let written = run.written();
        let (mut compiler, assets, readbacks) = run.split();
        let (mut texture_bytes, mut mip_levels) =
            assets
                .materials
                .sync(&device, &self.core.queue, world, (&mut mips, &mut compiler));
        assets.impostors.run(
            &device,
            &self.core.queue,
            world,
            (&assets.meshes, &assets.materials),
            (&mut mips, &mut compiler),
            (readbacks, written),
        );
        if let Some(encoder) = mips.take() {
            self.core.queue.hand(Handover::from(encoder));
        }
        drop(profile_upload);
        drop(row_upload);
        run.frames.current += 1;
        run.frames.parity = (run.frames.current % 2) as usize;
        run.tallies.drawn = 0;
        run.tallies.uploads.counted = 0;
        run.tallies.uploads.memory.clear();
        run.queued = Queued {
            before: run.queued.before.take(),
            ..Queued::default()
        };
        run.tallies.part_draws.fill(0);
        run.region = region.upscales().then(|| region.render.pixels());
        run.frames.asked = world
            .get_resource::<FrameTime>()
            .map_or(run.frames.current, |time| time.frame);
        let settings = world
            .get_resource::<crate::RenderSettings>()
            .cloned()
            .unwrap_or_default();
        run.reading.slots = settings.readback_slots.max(1) as usize;
        run.reading.slot_bytes = settings.readback_slot_bytes;
        run.settle_parts(world.get_resource::<engine_registry::BrickStates>());
        run.set_switches(super::install::switches_of(&run.machine, world));
        warm_on_flip(run, &mut self.installing, &device, world);
        run.drop_unfilled();
        let seconds = world
            .get_resource::<FrameTime>()
            .map_or(0.0, |time| time.elapsed as f32);
        let previous_seconds = self.seconds.replace(seconds).unwrap_or(seconds);
        let listed = listed_views(world, view, region, run.written());
        let entities: Vec<bevy_ecs::entity::Entity> = listed.iter().map(|facts| facts.view).collect();
        run.settle_sets(&entities);
        let mut cameras: Vec<CameraUniform> = listed
            .iter()
            .enumerate()
            .map(|(index, facts)| {
                let viewed = Viewed {
                    view_projection: facts.view_projection,
                    eye: facts.eye,
                    detail: detail_of(view, index, facts),
                    seconds,
                };
                camera_of(&viewed, facts, previous_seconds)
            })
            .collect();
        let canvas = surface
            .texture()
            .create_view(&wgpu::TextureViewDescriptor::default());
        run.canvas.view = Some(canvas.clone());
        run.canvas.texture = Some(surface.texture().clone());
        run.canvas.format = run.canvas.surface;
        run.uniforms.camera_slot = 0;
        run.uniforms.owned = owned_draws(world, &mut cameras, seconds);
        let profile_writes = engine_profile::span("render", "queue_writes");
        let row_writes = engine_profile::timed(engine_profile::row("render.writes"));
        run.uniforms.cameras = cameras;
        run.planning.runs = Bits::of(run.machine.graph.passes());
        for at in run.machine.declared.passes.len()..run.machine.graph.passes() {
            run.planning.runs.insert(at);
        }
        for index in 0..run.parts.len() {
            if !run.drawable(index as u32) {
                continue;
            }
            let Some(mut held) = run.parts[index].take() else {
                continue;
            };
            run.part = index as u32;
            let _owner = run
                .machine
                .declared
                .parts
                .get(index)
                .map(crate::install::PartInfo::owned);
            if let Err(error) = held.prepare(world, run) {
                run.fault(error);
            }
            run.forget_global(index as u32);
            run.parts[index] = Some(held);
        }
        let gated = world.get_resource::<engine_registry::BrickStates>();
        for &(pass, gate) in &run.installs.pass_gates {
            if gated.is_some_and(|states| !states.on(gate)) {
                run.planning.runs.remove(pass as usize);
            }
        }
        let mut live: Vec<bool> = run
            .machine
            .outputs
            .iter()
            .map(crate::machine::layered)
            .collect();
        for pass in run.planning.runs.iter() {
            for (target, ..) in run.planning.pass_sets.get(pass).into_iter().flat_map(|set| set.iter()) {
                if let Target::Output(index) = target
                    && let Some(on) = live.get_mut(*index as usize)
                {
                    *on = true;
                }
            }
        }
        run.planning.live = live;
        if let Some(library) = world.get_resource::<engine_core::AssetLibrary>() {
            for texture in std::mem::take(&mut run.bindings.wanted_textures) {
                if let Some(source) = library.texture(texture) {
                    let revision = library.revision_of(texture).unwrap_or_default();
                    let (mut compiler, assets, _) = run.split();
                    let (bytes, levels) = assets.materials.upload_texture(
                        &device,
                        &self.core.queue,
                        (texture, revision),
                        source,
                        (&mut mips, &mut compiler),
                    );
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
        let mut pieces = std::mem::take(&mut run.spare.pieces);
        piece_bytes(world, &run.uniforms.piece_layout, &mut pieces);
        camera_bytes += run.write_pieces(&mut pieces);
        run.spare.pieces = pieces;
        run.refresh_groups();
        drop(profile_writes);
        drop(row_writes);
        self.core.profiler.begin_frame(&device);
        let capturing = self.core.profiler.capturing();
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        run.planning.merge = !capturing;
        let mut asked = Bits::of(run.machine.graph.resources());
        if let Some(canvas) = run.machine.ids.target(Target::Canvas) {
            asked.insert(canvas.index());
        }
        for wanted in &run.queued.asked {
            if let Some(id) = run.machine.ids.buffers.get(wanted.buffer as usize) {
                asked.insert(id.index());
            }
        }
        let base = run.planning.runs.clone();
        let states = world.get_resource::<engine_registry::BrickStates>();
        let order = drawn_order(&listed);
        let last_surface = order.iter().rposition(|&index| listed[index].target.surface());
        let profile_encoding = engine_profile::span("render", "command_encoding");
        let row_encoding = engine_profile::timed(engine_profile::row("render.encoding"));
        let mut encoder = self.frame_encoder(&device, 0);
        self.core.display.time_begin(&mut encoder);
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        run.recording = Recording::Encoding(encoder);
        let mut batch = 0;
        let mut timed: Option<&'static str> = None;
        let mut parted: Option<(&'static str, u32)> = None;
        let mut ran: Vec<&'static str> = Vec::new();
        let declared = run.machine.declared.passes.len();
        let mut ready = Bits::of(run.machine.graph.resources());
        let mut drawn_on: Vec<(engine_core::Target, usize)> = Vec::new();
        let mut first = true;
        for (step, &index) in order.iter().enumerate() {
            let facts = listed[index];
            let Some(run) = self.run.as_mut() else { break };
            let Some((drawing, texture, format)) =
                canvas_of(run, &facts, (&canvas, surface.texture()))
            else {
                continue;
            };
            run.leave();
            let set = run.select(facts.view);
            let earlier = drawn_on
                .iter()
                .rev()
                .find(|(target, _)| *target == facts.target)
                .map(|&(_, from)| from);
            let size = facts.size();
            run.uniforms.camera_slot = index as u32;
            run.canvas.clear = facts.clear_color.map(f64::from);
            run.region = (facts.render != size).then_some((facts.render.x, facts.render.y));
            run.canvas.format = format;
            run.canvas.rect = (!facts.whole(glam::UVec2::new(texture.width(), texture.height())))
                .then_some(facts.pixels);
            run.canvas.texture = Some(texture);
            run.canvas.load = earlier.is_some()
                || (!facts.target.surface() && facts.clear != engine_core::Clear::All);
            run.size_targets((size.x, size.y), Some(drawing));
            run.planning.runs = base.clone();
            for part in 0..run.parts.len() {
                if !run.drawable(part as u32) {
                    continue;
                }
                let Some(mut held) = run.parts[part].take() else {
                    continue;
                };
                run.part = part as u32;
                held.view(&facts, run);
                run.parts[part] = Some(held);
            }
            run.forget_viewed();
            keep_eligible(run, &facts, (first, index == 0), Some(step) == last_surface);
            let mut outputs = asked.clone();
            if order.len() > 1 {
                for pass in run.planning.runs.iter().filter(|&pass| !run.sets.passes.contains(pass)) {
                    outputs = union(&outputs, &run.machine.graph.writes(pass));
                }
            }
            run.refresh_groups();
            let plan = self.plans.plan(
                &run.machine.graph,
                (&run.planning.runs, &outputs, &ready),
                crate::graph::PLANS,
            )?;
            for planned in &plan.passes {
                let pass = planned.pass.index();
                if !run.sets.passes.contains(pass) {
                    ready = union(&ready, &run.machine.graph.writes(pass));
                }
            }
            run.planning.plan = Some(std::sync::Arc::clone(&plan));
            let carried = match earlier {
                Some(from) if facts.clear == engine_core::Clear::Nothing => run.carry(from),
                Some(_) | None => false,
            };
            run.canvas.carried = carried;
            let mut post = crate::shared::post_uniform(
                facts.view_projection,
                facts.eye,
                detail_of(view, index, &facts),
                (facts.render.x, facts.render.y),
                (size.x, size.y),
            );
            post.kept[0] = f32::from(u8::from(
                run.canvas.load && !run.canvas.carried && facts.clear != engine_core::Clear::All,
            ));
            post.applied = applied_mask(run, &plan, &facts, states);
            run.uniforms.post = post;
            run.write_post();
            if earlier.is_none()
                && run.canvas.rect.is_some()
                && (facts.target.surface() || facts.clear == engine_core::Clear::All)
            {
                run.clear_canvas();
            }
            if run.canvas.load {
                run.copy_canvas(true);
            }
            let opaque = opaque_span(&run.machine.declared.passes, &plan);
            let mut opaque_open = false;
            let mut ran_last: Option<u32> = None;
            for (at, planned) in plan.passes.iter().enumerate() {
                if opaque_open && opaque.is_some_and(|(_, last)| at > last) {
                    let Some(run) = self.run.as_mut() else { break };
                    run.leave();
                    mark_span(
                        &self.core.profiler,
                        run.recording.encoder(),
                        OPAQUE_MESHES,
                        true,
                    );
                    opaque_open = false;
                }
                if planned.batch > batch {
                    let Some(run) = self.run.as_mut() else { break };
                    run.leave();
                    let mut ended = run.take_encoder();
                    self.mark(ended.as_mut(), timed.take(), None);
                    if let Some(mut encoder) = ended {
                        self.close_batch(&mut encoder, batch);
                        self.core.queue.hand(Handover::from(encoder));
                    }
                    let encoder = self.frame_encoder(&device, planned.batch);
                    let Some(run) = self.run.as_mut() else { break };
                    run.recording = Recording::Encoding(encoder);
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
                let elsewhere = pass.scene.is_some_and(|scene| !scene.core);
                if !run.drawable(part) {
                    continue;
                }
                run.part = part;
                run.pass = index as u32;
                run.planning.planned_at = at;
                if let Work::Clear(buffer) = work {
                    run.queued.cleared.push(buffer);
                    run.ran_now(index);
                    continue;
                }
                let continues = ran_last
                    .is_some_and(|last| last != index as u32 && run.same_group(last, index as u32));
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
                    let span = self.spans.get(index).copied();
                    let mut encoder = run.take_encoder();
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
                    run.recording = Recording::of(encoder);
                    run.group(index as u32);
                }
                let Some(run) = self.run.as_mut() else { break };
                run.flush_clears();
                let _owner = run
                    .machine
                    .declared
                    .parts
                    .get(part as usize)
                    .map(crate::install::PartInfo::owned);
                match work {
                    Work::Copy { .. } => {
                        if run.copy(index as u32) {
                            run.ran_now(index);
                        }
                    }
                    Work::Clear(_) => {}
                    Work::Render => {
                        let Some(mut held) = run.parts[part as usize].take() else {
                            continue;
                        };
                        let drawn = held.encode(index as u32, run, &self.core.profiler);
                        run.ran_now(index);
                        let next = plan.passes.get(at + 1).map(|next| next.pass.index() as u32);
                        if !next.is_some_and(|next| {
                            next != index as u32 && run.same_group(index as u32, next)
                        }) {
                            run.close_encode_span(&self.core.profiler);
                        }
                        if let Some(sum) = run.tallies.part_draws.get_mut(part as usize) {
                            *sum += drawn;
                        }
                        if !elsewhere {
                            run.tallies.drawn += drawn;
                        }
                        run.parts[part as usize] = Some(held);
                    }
                }
            }
            let Some(run) = self.run.as_mut() else { break };
            run.close_encode_span(&self.core.profiler);
            run.flush_clears();
            run.leave();
            if opaque_open {
                mark_span(&self.core.profiler, run.recording.encoder(), OPAQUE_MESHES, true);
            }
            if run.canvas.rect.is_some() {
                run.copy_canvas(false);
            }
            drawn_on.push((facts.target, set));
            first = false;
        }
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        run.close_encode_span(&self.core.profiler);
        run.flush_clears();
        run.leave();
        run.select(entities[0]);
        run.uniforms.camera_slot = 0;
        run.canvas.view = Some(canvas);
        run.canvas.texture = Some(surface.texture().clone());
        run.canvas.format = run.canvas.surface;
        run.canvas.rect = None;
        let mut encoder = run.take_encoder().unwrap_or_else(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/frame-encoder"),
            })
        });
        if let Some((open, _)) = parted.take() {
            mark_span(&self.core.profiler, Some(&mut encoder), open, true);
        }
        self.mark(Some(&mut encoder), timed, None);
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        run.recording = Recording::Encoding(encoder);
        run.copy_at_end();
        run.record_readbacks();
        let mut encoder = run.take_encoder().unwrap_or_else(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/frame-encoder"),
            })
        });
        let (written, slots) = (run.written(), run.reading.slots);
        self.core.display.time_end(
            (&device, &mut encoder),
            (&mut run.reading.list, written, slots),
        );
        self.close_batch(&mut encoder, batch);
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        self.core
            .profiler
            .end_frame((&device, &mut encoder), &mut run.reading.list, written);
        drop(profile_encoding);
        drop(row_encoding);
        let profile_submit = engine_profile::span("render", "submit_and_present");
        let row_submit = engine_profile::timed(engine_profile::row("render.submit"));
        self.core.queue.hand(Handover::from(encoder));
        self.core.queue.settle();
        surface.present(&self.core.queue);
        let pipelines_created = self.core.ledger.frame_submitted(&self.core.queue);
        let render_passes = self.ran_passes(&ran);
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        run.reading.list.map();
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
            draw_calls: run.tallies.drawn,
            upload_bytes: camera_bytes + run.tallies.uploads.counted + mesh_bytes + texture_bytes,
            resident_meshes: run.assets.meshes.len() as u32,
            mesh_upload_bytes: mesh_bytes,
            resident_mesh_bytes,
            resident_textures,
            texture_upload_bytes: texture_bytes,
            resident_texture_bytes,
            pipelines: self.core.ledger.created(),
            pipelines_created,
            render_passes,
            linked: self.linked,
            readback_overflows: run.reading.list.overflows(),
            ..RenderStats::default()
        };
        let immediate = self.core.queue.take_immediate();
        rebuilt_counters(&mut run.tallies.rebuilt);
        if engine_profile::active() {
            stat_counters(&stats);
            engine_profile::counter("render.immediate_submissions", f64::from(immediate));
            engine_profile::counter("render.texture_mip_levels", f64::from(mip_levels));
            engine_profile::counter("render.camera_write_bytes", camera_bytes as f64);
            for name in MEMORY {
                let held = match name {
                    "memory.gpu_impostor_bytes" => run.assets.impostors.capacity_bytes(),
                    "memory.render_staging_capacity_bytes" => run.assets.meshes.staging_bytes(),
                    "memory.gpu_mesh_payload_bytes" => stats.resident_mesh_bytes,
                    "memory.gpu_mesh_arena_bytes" => run.assets.meshes.capacity_bytes(),
                    "memory.gpu_texture_payload_bytes" => stats.resident_texture_bytes,
                    "memory.gpu_texture_mip_bytes" => mip_bytes,
                    _ => 0,
                };
                let parts: u64 = run
                    .tallies.uploads
                    .memory
                    .iter()
                    .filter(|(known, _)| *known == name)
                    .map(|(_, bytes)| *bytes)
                    .sum();
                engine_profile::counter(name, (held + parts) as f64);
            }
        }
        Ok(stats)
    }

    fn ran_passes(&self, ran: &[&'static str]) -> engine_core::Names {
        let names = self.passes;
        let bits = names
            .iter()
            .enumerate()
            .filter(|(_, name)| ran.contains(name))
            .fold(0_u128, |bits, (at, _)| {
                bits | 1_u128.checked_shl(at as u32).unwrap_or(0)
            });
        engine_core::Names { bits, names }
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

fn warm_on_flip(
    run: &mut Run,
    installing: &mut Option<Validation>,
    device: &wgpu::Device,
    world: &World,
) {
    let generation = world
        .get_resource::<engine_registry::BrickStates>()
        .map_or(0, engine_registry::BrickStates::generation);
    if run.reached.generation == Some(generation) {
        return;
    }
    if run.reached.generation.is_some() {
        run.warm_reached(world);
        return;
    }
    let scope = crate::gpu_calls::validation_scope(device);
    run.warm_reached(world);
    *installing = Some(Box::pin(scope.pop()));
}

fn listed_views(
    world: &World,
    view: Option<&PrimaryView>,
    region: crate::targets::FrameRegion,
    frame: engine_core::FrameId,
) -> Vec<engine_core::ViewFacts> {
    let listed: Vec<engine_core::ViewFacts> = world
        .get_resource::<engine_core::Views>()
        .map(|views| views.listed().to_vec())
        .unwrap_or_default();
    if !listed.is_empty() {
        return listed;
    }
    let (width, height) = region.display.pixels();
    let (drawn_width, drawn_height) = region.render.pixels();
    let view_projection = view.map_or(Mat4::IDENTITY, PrimaryView::view_projection);
    vec![engine_core::ViewFacts {
        view: bevy_ecs::entity::Entity::PLACEHOLDER,
        camera: bevy_ecs::entity::Entity::PLACEHOLDER,
        target: engine_core::Target::Surface,
        pixels: [0, 0, width, height],
        render: glam::UVec2::new(drawn_width, drawn_height),
        layers: engine_core::Layers::ALL,
        clear: engine_core::Clear::All,
        clear_color: engine_core::View::BLACK,
        order: 0,
        lens: view.map(PrimaryView::lens).unwrap_or_default(),
        eye: view.map_or(glam::Vec3::ZERO, PrimaryView::eye),
        forward: view.map_or(glam::Vec3::NEG_Z, PrimaryView::forward),
        view_projection,
        previous_view_projection: view_projection,
        history: view.map_or(engine_core::ViewHistory::Fresh, PrimaryView::history),
        jitter: view.and_then(PrimaryView::jitter),
        noise: view.and_then(PrimaryView::facts).map_or(0.0, |facts| facts.noise),
        frame,
    }]
}

fn drawn_order(listed: &[engine_core::ViewFacts]) -> Vec<usize> {
    let (surfaces, textures): (Vec<usize>, Vec<usize>) =
        (0..listed.len()).partition(|&index| listed[index].target.surface());
    textures.into_iter().chain(surfaces).collect()
}

fn detail_of(view: Option<&PrimaryView>, index: usize, facts: &engine_core::ViewFacts) -> glam::Vec3 {
    match view {
        Some(view) if index == 0 => view.detail_eye(),
        Some(_) | None => facts.eye,
    }
}

fn view_region(facts: &engine_core::ViewFacts) -> crate::targets::FrameRegion {
    let size = facts.size();
    crate::targets::FrameRegion::new(
        crate::targets::FrameSize {
            width: size.x,
            height: size.y,
        },
        crate::targets::FrameSize {
            width: facts.render.x,
            height: facts.render.y,
        },
    )
}

fn canvas_of(
    run: &Run,
    facts: &engine_core::ViewFacts,
    (canvas, surface): (&wgpu::TextureView, &wgpu::Texture),
) -> Option<(wgpu::TextureView, wgpu::Texture, wgpu::TextureFormat)> {
    match facts.target {
        engine_core::Target::Surface => Some((canvas.clone(), surface.clone(), run.canvas.surface)),
        engine_core::Target::Texture(handle) => {
            let (texture, _) = run.assets.materials.target(handle)?;
            let format = texture.format().remove_srgb_suffix();
            let drawing = texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(format),
                ..Default::default()
            });
            Some((drawing, texture.clone(), format))
        }
    }
}

fn owned_draws(
    world: &World,
    cameras: &mut Vec<CameraUniform>,
    seconds: f32,
) -> Vec<crate::encode::OwnedDraw> {
    let Some(views) = world.get_resource::<engine_core::Views>() else {
        return Vec::new();
    };
    let mut owned = Vec::new();
    for (drawn, view) in views.drawn().iter().enumerate() {
        let Some(pass) = view.owner else {
            continue;
        };
        owned.push(crate::encode::OwnedDraw {
            pass,
            local: view.local,
            drawn: drawn as u32,
            slot: cameras.len() as u32,
            layer: view.target.layer,
            rect: view.target.rect,
            clear: view.clear,
            passes: view.passes,
            draws: view.draws,
        });
        cameras.push(owned_camera(view, seconds));
    }
    owned
}

fn owned_camera(view: &engine_core::DrawnView, seconds: f32) -> CameraUniform {
    let size = crate::targets::FrameSize {
        width: view.size.x,
        height: view.size.y,
    };
    let mut camera = CameraUniform::viewed(
        (view.view_projection, view.view_projection),
        view.position,
        (seconds, seconds),
        [0.0; 2],
        (size.width, size.height),
    );
    camera.set_region(view.view_projection, crate::targets::FrameRegion::whole(size));
    camera.set_detail(view.eye);
    camera.capture = u32::from(view.capture);
    camera.view[0] = 1 + view.local;
    camera
}

fn keep_eligible(
    run: &mut Run,
    facts: &engine_core::ViewFacts,
    (first, main): (bool, bool),
    overlay: bool,
) {
    let declared = run.machine.declared.passes.len();
    let listed: Vec<usize> = run.planning.runs.iter().collect();
    for index in listed {
        let viewed = run.sets.passes.contains(index);
        let mained = run.sets.main.contains(index);
        if (mained && !main) || (!mained && !viewed && !first) {
            run.planning.runs.remove(index);
            continue;
        }
        let shown = index >= declared || {
            let pass = &run.machine.declared.passes[index];
            let views = match pass.views {
                engine_core::PassViews::Every | engine_core::PassViews::Owned => true,
                engine_core::PassViews::Main => facts.target.surface(),
            };
            (!viewed || views)
                && (pass.layer != Some(engine_core::AttachPoint::Overlay) || overlay)
        };
        if !shown {
            run.planning.runs.remove(index);
        }
    }
}

fn union(first: &Bits, second: &Bits) -> Bits {
    let mut joined = first.clone();
    for index in second.iter() {
        joined.insert(index);
    }
    joined
}

fn applied_mask(
    run: &Run,
    plan: &crate::graph::Plan,
    facts: &engine_core::ViewFacts,
    states: Option<&engine_registry::BrickStates>,
) -> [u32; 4] {
    let mut mask = [0_u32; 4];
    for (bit, member) in run.installs.applied.iter().enumerate() {
        let passes: Vec<usize> = run
            .machine
            .declared
            .passes
            .iter()
            .enumerate()
            .filter(|(_, pass)| pass.name == member.pass)
            .map(|(index, _)| index)
            .collect();
        let on = if passes.is_empty() {
            member
                .gate
                .is_none_or(|gate| states.is_none_or(|states| states.on(gate)))
                && match member.views {
                    engine_core::PassViews::Every => true,
                    engine_core::PassViews::Main => facts.target.surface(),
                    engine_core::PassViews::Owned => false,
                }
        } else {
            plan.passes
                .iter()
                .any(|planned| passes.contains(&planned.pass.index()))
        };
        if on {
            mask[bit / 32 % 4] |= 1 << (bit % 32);
        }
    }
    mask
}

fn piece_bytes(world: &World, layout: &[(usize, crate::pieces::PieceAt)], bytes: &mut Vec<u8>) {
    bytes.clear();
    let Some(registry) = world.get_resource::<engine_registry::Registry>() else {
        return;
    };
    let entries: Vec<_> = registry.pieces().map(|(_, entry)| entry).collect();
    let end = layout
        .iter()
        .map(|(_, at)| at.offset + at.size)
        .max()
        .unwrap_or(0);
    bytes.resize(end as usize, 0);
    for &(index, at) in layout {
        let Some(found) = entries.get(index).and_then(|entry| entry.bytes(world)) else {
            continue;
        };
        let start = at.offset as usize;
        let length = found.len().min(at.size as usize);
        bytes[start..start + length].copy_from_slice(&found[..length]);
    }
}
