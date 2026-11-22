use bevy_ecs::world::World;
use engine_core::{FrameTime, RenderFrame, RenderStats};

use super::*;
use crate::CameraUniform;
use crate::encode::Recording;
use crate::gpu_calls::Handover;
use crate::graph::Bits;
use crate::machine::Target;
use crate::profiling::{GpuProfiler, Span};
use crate::readback::Reader;

const BATCHES: [&str; 3] = ["batch_0", "batch_1", "batch_2"];
const OPAQUE_SCENE: &str = "opaque_scene";

#[derive(Default)]
pub(super) struct Hidden {
    asked: bool,
    submitted: u64,
    finished: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl Hidden {
    fn submitted(&mut self, queue: &crate::gpu_calls::Queue) {
        self.submitted += 1;
        let (finished, frame) = (std::sync::Arc::clone(&self.finished), self.submitted);
        queue.on_submitted_work_done(move || {
            finished.fetch_max(frame, std::sync::atomic::Ordering::Relaxed);
        });
    }

    fn behind(&self) -> bool {
        self.finished.load(std::sync::atomic::Ordering::Relaxed) < self.submitted
    }
}

type Pixels = (u32, u32, Vec<u8>);

#[derive(Default)]
pub(super) struct Picture {
    asked: bool,
    reading: Option<Result<Reading, Error>>,
}

pub(super) struct Reading {
    buffer: wgpu::Buffer,
    size: (u32, u32),
    row: u32,
    bgra: bool,
    mapped: std::sync::Arc<std::sync::OnceLock<Result<(), wgpu::BufferAsyncError>>>,
}

impl Reading {
    fn map(self) -> Self {
        let mapped = std::sync::Arc::clone(&self.mapped);
        self.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |outcome| {
                let _ = mapped.set(outcome);
            });
        self
    }

    fn pixels(&self) -> Result<Vec<u8>, Error> {
        let view = self
            .buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|error| failed_with("$.picture", error))?;
        let (width, height) = self.size;
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for rows in view.chunks_exact(self.row as usize) {
            for texel in rows[..(width * 4) as usize].as_chunks::<4>().0 {
                let [first, green, third, alpha] = *texel;
                let (red, blue) = if self.bgra {
                    (third, first)
                } else {
                    (first, third)
                };
                pixels.extend_from_slice(&[red, green, blue, alpha]);
            }
        }
        drop(view);
        self.buffer.unmap();
        Ok(pixels)
    }
}

fn copy_picture(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: Option<&wgpu::Texture>,
) -> Result<Reading, Error> {
    let texture = texture.ok_or_else(|| failed("$.picture").expected(Expected::Present))?;
    let format = texture.format().remove_srgb_suffix();
    let bgra = format == wgpu::TextureFormat::Bgra8Unorm;
    if !bgra && format != wgpu::TextureFormat::Rgba8Unorm {
        return Err(failed("$.picture").expected(Expected::Supported));
    }
    if !texture.usage().contains(wgpu::TextureUsages::COPY_SRC) {
        return Err(failed("$.picture").expected(Expected::Readable));
    }
    let (width, height) = (texture.width(), texture.height());
    let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = crate::gpu_calls::create_buffer(device, &wgpu::BufferDescriptor {
        label: Some("engine/picture-readback"),
        size: u64::from(row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(height),
            },
        },
        texture.size(),
    );
    Ok(Reading {
        buffer,
        size: (width, height),
        row,
        bgra,
        mapped: std::sync::Arc::default(),
    })
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

impl Renderer {
    pub(super) fn collect(&mut self) {
        crate::platform::poll(&self.core.device);
        #[cfg(not(target_arch = "wasm32"))]
        crate::platform::run_arrived_maps();
        let Some(run) = self.run.as_mut() else {
            return;
        };
        let (display, profiler) = (&mut self.core.display, &mut self.core.profiler);
        let declared = &run.machine.declared.readbacks;
        let arrived = &mut run.reading.arrived;
        run.reading
            .list
            .take(|reader, written, arrival| match reader {
                Reader::FrameTimer => display.timed(written, arrival),
                Reader::Profiler(slot) => profiler.arrived(slot, arrival),
                Reader::Work => run.pipelines.work_landed(written, arrival),
                Reader::Part(part, id) => {
                    if let crate::readback::Arrival::Read(bytes) = arrival {
                        crate::encode::Reading::land_into(arrived, declared, (part, id), written, bytes);
                    }
                }
            });
    }

    pub fn deliver(&mut self, world: &mut World) {
        self.recover(world);
        self.collect();
        let Some(run) = self.run.as_mut() else {
            return;
        };
        let refused = run.assets.textures.take_refused();
        if !refused.is_empty()
            && let Some(mut registry) = world.get_resource_mut::<engine_registry::Registry>()
        {
            for error in refused {
                registry.report(error);
            }
        }
        let faults = std::mem::take(&mut run.installs.faults);
        if let Some(mut registry) = world.get_resource_mut::<engine_registry::Registry>() {
            for error in faults {
                registry.report(error);
            }
        }
        self.follow_states(world);
    }

    pub fn read_picture(&mut self) {
        self.core
            .display
            .require_usage(wgpu::TextureUsages::COPY_SRC);
        self.picture.asked = true;
        self.picture.reading = None;
    }

    fn ensure_surface_copy(
        &mut self,
        surface: &mut Option<DisplayFrame>,
        listed: &[engine_core::ViewFacts],
        display: glam::UVec2,
    ) -> Result<(), Error> {
        let partial = surface.is_some()
            && listed
                .iter()
                .any(|facts| facts.target.surface() && !facts.whole(display));
        if partial
            && self
                .core
                .display
                .require_usage(wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST)
        {
            drop(surface.take());
            *surface = self.core.display.acquire(&self.core.device)?;
        }
        Ok(())
    }

    pub fn take_picture(&mut self) -> Option<Result<Pixels, Error>> {
        crate::platform::poll(&self.core.device);
        let reading = match self.picture.reading.take()? {
            Ok(reading) => reading,
            Err(error) => return Some(Err(error)),
        };
        let Some(outcome) = reading.mapped.get() else {
            self.picture.reading = Some(Ok(reading));
            return None;
        };
        let (width, height) = reading.size;
        Some(match outcome {
            Ok(()) => reading.pixels().map(|pixels| (width, height, pixels)),
            Err(error) => Err(failed_with("$.picture", error)),
        })
    }

    pub fn take_draw_reports(&mut self) -> Option<engine_core::DrawReports> {
        super::kind_passes::take_reports(self.run.as_mut()?)
    }

    fn follow_states(&mut self, world: &mut World) {
        let Some(run) = self.run.as_mut() else {
            return;
        };
        settle_states(run, world);
        for part in 0..run.parts.len() {
            let Some(gate) = run.installs.gates.get(part).copied().flatten() else {
                continue;
            };
            let states = world.get_resource::<engine_registry::BrickStates>();
            let Some(count) = states.and_then(|states| states.installed(gate)) else {
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
                Err(error) => {
                    if let Some(mut registry) =
                        world.get_resource_mut::<engine_registry::Registry>()
                    {
                        registry.report(error);
                    }
                }
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

    fn frame_encoder(
        &self,
        device: &wgpu::Device,
        kept: Option<wgpu::CommandEncoder>,
        batch: u8,
    ) -> wgpu::CommandEncoder {
        let mut encoder = kept.unwrap_or_else(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: crate::encode::debug_label("engine/frame-encoder"),
            })
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

    pub fn acquire_frame(&mut self, picture: crate::Picture) -> Result<bool, Error> {
        let acquired = self.acquire(picture);
        if !matches!(acquired, Ok(true)) {
            self.picture.asked = false;
        }
        acquired
    }

    fn answer_picture(&mut self, surface: Option<DisplayFrame>) -> RenderStats {
        if std::mem::take(&mut self.picture.asked) {
            let device = &self.core.device;
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/picture-readback"),
            });
            let texture = surface.as_ref().map(DisplayFrame::texture);
            #[cfg(not(target_arch = "wasm32"))]
            let texture = texture.or(self.core.display.offscreen());
            let picture = copy_picture(device, &mut encoder, texture);
            self.core.queue.hand(Handover::from(encoder));
            self.core.queue.settle();
            self.picture.reading = Some(picture.map(Reading::map));
            if let Some(surface) = surface {
                surface.present(&self.core.queue);
            }
        }
        RenderStats::default()
    }

    fn acquire(&mut self, picture: crate::Picture) -> Result<bool, Error> {
        self.check_health()?;
        drop(self.surface.take());
        self.hidden.asked = false;
        if let Some(run) = self.run.as_mut() {
            run.tallies.held = None;
        }
        if self.core.display.suspended() || self.run.is_none() || !self.usable() {
            return Ok(false);
        }
        match picture {
            crate::Picture::Shown => {
                self.surface = self.core.display.acquire(&self.core.device)?;
                Ok(self.surface.is_some())
            }
            crate::Picture::Hidden => {
                crate::platform::poll(&self.core.device);
                self.hidden.asked = !self.hidden.behind();
                Ok(self.hidden.asked)
            }
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub fn render_frame(&mut self, frame: RenderFrame<'_>) -> Result<RenderStats, Error> {
        let _profile = engine_profile::span("render", "render_snapshot");
        let mut surface = self.surface.take();
        let hidden = std::mem::take(&mut self.hidden.asked);
        if surface.is_none() && !hidden {
            return Ok(self.answer_picture(surface));
        }
        let world = frame.world();
        #[cfg(feature = "dev")]
        self.core
            .queue
            .mark_unchanged(world.contains_resource::<crate::gpu_calls::UnchangedWrites>());
        let views = world.get_resource::<engine_core::Views>();
        let listed = views.map_or(&[][..], engine_core::Views::listed);
        if listed.is_empty() {
            return Ok(self.answer_picture(surface));
        }
        let display = views.map_or(glam::UVec2::ONE, |views| views.frame().display_size);
        self.ensure_surface_copy(&mut surface, listed, display)?;
        if surface.is_none() && !hidden {
            return Ok(self.answer_picture(surface));
        }
        let Some(run) = self.run.as_mut() else {
            return Ok(self.answer_picture(surface));
        };
        let device = self.core.device.clone();
        run.frames.current += 1;
        run.frames.parity = (run.frames.current % 2) as usize;
        run.frames.asked = world
            .get_resource::<FrameTime>()
            .map_or(run.frames.current, |time| time.frame);
        let profile_upload = engine_profile::span("render", "asset_uploads");
        let mesh_bytes = run.assets.meshes.sync(&device, &self.core.queue, world);
        let texture_bytes = run.assets.textures.sync(&device, &self.core.queue, world);
        drop(profile_upload);
        let drawn_before = run.tallies.drawn.total;
        run.tallies.kinds.start_frame();
        run.tallies.held = Some(false);
        let defaults;
        let settings = match world.get_resource::<crate::RenderSettings>() {
            Some(settings) => settings,
            None => {
                defaults = crate::RenderSettings::default();
                &defaults
            }
        };
        run.reading.slots = settings.readback_slots.max(1) as usize;
        settle_states(run, world);
        warm_on_flip(run, &mut self.installing, &device, world);
        run.drop_unfilled();
        let seconds = world
            .get_resource::<FrameTime>()
            .map_or(0.0, |time| time.elapsed as f32);
        run.settle_sets(listed);
        run.uniforms.cameras.clear();
        let cameras = listed.iter().map(|facts| CameraUniform::of(facts, seconds));
        run.uniforms.cameras.extend(cameras);
        let shown = surface.as_ref().map(|surface| {
            let texture = surface.texture().clone();
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (view, texture)
        });
        run.canvas.view = shown.as_ref().map(|(view, _)| view.clone());
        run.canvas.texture = shown.as_ref().map(|(_, texture)| texture.clone());
        run.canvas.format = run.canvas.surface;
        let canvas_size = shown
            .as_ref()
            .map_or((display.x, display.y), |(_, texture)| {
                (texture.width(), texture.height())
            });
        run.canvas.size = (canvas_size.0.max(1), canvas_size.1.max(1));
        run.uniforms.camera_slot = 0;
        let profile_writes = engine_profile::span("render", "queue_writes");
        run.start_runs();
        for index in 0..run.parts.len() {
            if !run.drawable(index as u32) {
                continue;
            }
            let Some(mut held) = run.parts[index].take() else {
                continue;
            };
            run.part = index as u32;
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
        run.warm_running();
        if let Some(error) = run.refused.take() {
            return Err(error);
        }
        run.write_cameras(views.map_or(0, engine_core::Views::slots));
        run.refresh_groups();
        drop(profile_writes);
        self.core.profiler.begin_frame(&device, settings);
        let Some(run) = self.run.as_mut() else {
            return Ok(self.answer_picture(surface));
        };
        run.planning.merge = !self.core.profiler.splits();
        let scratch = &mut run.sets.scratch;
        scratch.asked.clear();
        if let Some(canvas) = run.machine.ids.target(Target::Canvas) {
            scratch.asked.insert(canvas.index());
        }
        scratch.base.assign(&run.planning.runs);
        #[cfg(feature = "dev")]
        if crate::whatif::changed(world, &mut scratch.seen) {
            for set in &mut run.sets.list {
                set.plans.retain(|(skips, ..)| *skips == 0);
            }
        }
        scratch.ready.clear();
        scratch.drawn_on.clear();
        let mut order = std::mem::take(&mut scratch.order);
        drawn_order(listed, &mut order);
        let profile_encoding = engine_profile::span("render", "command_encoding");
        let mut encoder = self.frame_encoder(&device, None, 0);
        let stamped = !hidden && self.core.display.time_begin(&mut encoder);
        let Some(run) = self.run.as_mut() else {
            return Ok(self.answer_picture(surface));
        };
        run.recording = Recording::Encoding(encoder);
        let (finishers, mut encoders) = (self.core.queue.finishers(), 1);
        let mut batch = 0;
        let mut timed: Option<&'static str> = None;
        let mut parted: Option<(&'static str, u32)> = None;
        let mut ran: u128 = 0;
        let mut first = true;
        for &index in &order {
            let facts = listed[index];
            let Some(run) = self.run.as_mut() else { break };
            let Some((drawing, texture, format, extent)) =
                canvas_of(run, &facts, shown.as_ref(), display)
            else {
                continue;
            };
            let collecting = hidden && facts.target.surface();
            run.leave();
            let set = run.select(facts.slot);
            let earlier = run
                .sets
                .scratch
                .drawn_on
                .iter()
                .rev()
                .find(|(target, _)| *target == facts.target)
                .map(|&(_, from)| from);
            let size = facts.size();
            run.uniforms.camera_slot = index as u32;
            run.canvas.clear = facts.clear_color.map(f64::from);
            run.canvas.format = format;
            run.canvas.rect = (!facts.whole(extent)).then_some(facts.pixels);
            run.canvas.texture = texture;
            run.canvas.load = earlier.is_some()
                || (!facts.target.surface() && facts.clear != engine_core::Clear::All);
            run.size_targets((size.x, size.y), drawing);
            run.planning.runs.assign(&run.sets.scratch.base);
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
            keep_eligible(run, &facts, (first, index == 0));
            let skipped = skip_scenes(run, world, facts.view);
            #[cfg(feature = "dev")]
            {
                run.sets.scratch.skips =
                    crate::whatif::skip(world, run.frames.asked, &mut run.planning.runs);
            }
            let sets = &mut run.sets;
            sets.scratch.outputs.assign(&sets.scratch.asked);
            if order.len() > 1 {
                for pass in run.planning.runs.iter() {
                    if !sets.passes.contains(pass) {
                        sets.scratch
                            .outputs
                            .union_with(run.machine.graph.writes(pass));
                    }
                }
            }
            run.refresh_groups();
            let plan = run.plan_view()?;
            for planned in &plan.passes {
                let pass = planned.pass.index();
                if !run.sets.passes.contains(pass) {
                    run.sets
                        .scratch
                        .ready
                        .union_with(run.machine.graph.writes(pass));
                }
            }
            run.planning.plan = Some(std::sync::Arc::clone(&plan));
            let carried = match earlier {
                Some(from) if !collecting && facts.clear == engine_core::Clear::Nothing => {
                    run.carry(from)
                }
                Some(_) | None => false,
            };
            run.canvas.carried = carried;
            let mut post = crate::shared::post_uniform((size.x, size.y));
            post.kept[0] = u32::from(
                run.canvas.load && !run.canvas.carried && facts.clear != engine_core::Clear::All,
            );
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
            let opaque = plan.opaque;
            let mut opaque_open = false;
            for (at, planned) in plan.passes.iter().enumerate() {
                if let Some(run) = self.run.as_mut() {
                    run.record();
                }
                if opaque_open && opaque.is_some_and(|(_, last)| at > last) {
                    let Some(run) = self.run.as_mut() else { break };
                    run.leave();
                    mark_span(
                        &self.core.profiler,
                        run.recording.encoder(),
                        OPAQUE_SCENE,
                        true,
                    );
                    opaque_open = false;
                }
                if planned.batch > batch {
                    let Some(run) = self.run.as_mut() else { break };
                    run.leave();
                    let mut ended = run.take_encoder();
                    self.mark(ended.as_mut(), timed.take(), None);
                    if let Some(encoder) = ended.as_mut() {
                        self.close_batch(encoder, batch);
                    }
                    if encoders < finishers
                        && let Some(encoder) = ended.take()
                    {
                        self.core.queue.hand(Handover::from(encoder));
                        encoders += 1;
                    }
                    let encoder = self.frame_encoder(&device, ended, planned.batch);
                    let Some(run) = self.run.as_mut() else { break };
                    run.recording = Recording::Encoding(encoder);
                    batch = planned.batch;
                }
                let index = planned.pass.index();
                let Some(run) = self.run.as_mut() else { break };
                let collected = collecting
                    && (run.sets.passes.contains(index) || run.sets.main.contains(index));
                let part = run.machine.declared.passes[index].part;
                if !run.drawable(part) {
                    continue;
                }
                run.part = part;
                run.pass = index as u32;
                run.planning.planned_at = at;
                if !collected {
                    let name = run.sets.fixed.names.get(index).copied().unwrap_or(u32::MAX);
                    ran |= 1_u128.checked_shl(name).unwrap_or(0);
                }
                let wanted = run.span_of(index as u32);
                let keeps = matches!(
                    (parted, wanted),
                    (Some((open, owner)), Some(name)) if open == name && owner == part
                );
                if run.enter(index as u32) {
                    let span = self.spans.get(index).copied();
                    let mut encoder = run.take_encoder();
                    if !keeps && let Some((open, _)) = parted.take() {
                        mark_span(&self.core.profiler, encoder.as_mut(), open, true);
                    }
                    self.mark(encoder.as_mut(), timed, span);
                    timed = span;
                    if !opaque_open && opaque.is_some_and(|(first, _)| first == at) {
                        mark_span(&self.core.profiler, encoder.as_mut(), OPAQUE_SCENE, false);
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
                self.core
                    .profiler
                    .ran(timed, run.machine.declared.passes[index].name);
                if collected {
                    run.collect();
                }
                let Some(mut held) = run.parts[part as usize].take() else {
                    continue;
                };
                held.encode(index as u32, run, &self.core.profiler);
                if !collected {
                    run.ran_now(index);
                }
                run.parts[part as usize] = Some(held);
            }
            let Some(run) = self.run.as_mut() else { break };
            run.record();
            run.leave();
            if opaque_open {
                mark_span(
                    &self.core.profiler,
                    run.recording.encoder(),
                    OPAQUE_SCENE,
                    true,
                );
            }
            walk_held(run, &self.core.profiler, skipped);
            if run.canvas.rect.is_some() {
                run.copy_canvas(false);
            }
            run.sets.scratch.drawn_on.push((facts.target, set));
            first = false;
        }
        let Some(run) = self.run.as_mut() else {
            return Ok(self.answer_picture(surface));
        };
        run.sets.scratch.order = order;
        run.leave();
        run.select(listed[0].slot);
        run.uniforms.camera_slot = 0;
        run.canvas.view = shown.as_ref().map(|(view, _)| view.clone());
        run.canvas.texture = shown.map(|(_, texture)| texture);
        run.canvas.format = run.canvas.surface;
        run.canvas.rect = None;
        let mut encoder = run.take_encoder().unwrap_or_else(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: crate::encode::debug_label("engine/frame-encoder"),
            })
        });
        if let Some((open, _)) = parted.take() {
            mark_span(&self.core.profiler, Some(&mut encoder), open, true);
        }
        self.mark(Some(&mut encoder), timed, None);
        let Some(run) = self.run.as_mut() else {
            return Ok(self.answer_picture(surface));
        };
        run.recording = Recording::Encoding(encoder);
        run.record_readbacks();
        let mut encoder = run.take_encoder().unwrap_or_else(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: crate::encode::debug_label("engine/frame-encoder"),
            })
        });
        let (written, slots) = (run.written(), run.reading.slots);
        if stamped {
            self.core.display.time_end(
                (&device, &mut encoder),
                (&mut run.reading.list, written, slots),
            );
        }
        self.close_batch(&mut encoder, batch);
        let Some(run) = self.run.as_mut() else {
            return Ok(self.answer_picture(surface));
        };
        self.core
            .profiler
            .end_frame((&device, &mut encoder), &mut run.reading.list, written);
        let picture = std::mem::take(&mut self.picture.asked).then(|| {
            let texture = surface.as_ref().map(DisplayFrame::texture);
            #[cfg(not(target_arch = "wasm32"))]
            let texture = texture.or(self.core.display.offscreen());
            copy_picture(&device, &mut encoder, texture)
        });
        drop(profile_encoding);
        let profile_submit = engine_profile::span("render", "submit_and_present");
        self.core.queue.hand(Handover::from(encoder));
        self.core.queue.settle();
        if let Some(picture) = picture {
            self.picture.reading = Some(picture.map(Reading::map));
        }
        if hidden {
            self.hidden.submitted(&self.core.queue);
        }
        if let Some(surface) = surface {
            surface.present(&self.core.queue);
        }
        let pipelines_created = self.core.ledger.frame_submitted(&self.core.queue);
        let render_passes = engine_core::Names {
            bits: ran,
            names: self.passes,
        };
        let Some(run) = self.run.as_mut() else {
            return Ok(RenderStats::default());
        };
        run.reading.list.map();
        crate::platform::poll(&device);
        drop(profile_submit);
        super::kind_passes::settle(run, world);
        let drawn = run.tallies.drawn.total.since(drawn_before);
        Ok(RenderStats {
            draw_calls: drawn.draws,
            instances: drawn.instances,
            upload_bytes: self.core.queue.take_written(),
            mesh_upload_bytes: mesh_bytes,
            texture_upload_bytes: texture_bytes,
            pipelines: self
                .core
                .ledger
                .created()
                .saturating_add(run.pipelines.warming.made()),
            pipelines_created,
            render_passes,
        })
    }
}

fn warm_on_flip(
    run: &mut Run,
    installing: &mut Option<Validation>,
    device: &wgpu::Device,
    world: &World,
) {
    run.pipelines.warming.open(
        world
            .get_resource::<engine_core::Host>()
            .is_some_and(|host| host.showing),
    );
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
    super::validate_also(installing, Box::pin(scope.pop()));
}

fn settle_states(run: &mut Run, world: &World) {
    let states = world.get_resource::<engine_registry::BrickStates>();
    let generation = states.map(engine_registry::BrickStates::generation);
    if run.sets.settled == Some(generation) {
        return;
    }
    run.settle_parts(states);
    run.sets.settled = Some(generation);
}

fn drawn_order(listed: &[engine_core::ViewFacts], order: &mut Vec<usize>) {
    order.clear();
    order.extend((0..listed.len()).filter(|&index| !listed[index].target.surface()));
    order.extend((0..listed.len()).filter(|&index| listed[index].target.surface()));
}

fn canvas_of(
    run: &mut Run,
    facts: &engine_core::ViewFacts,
    shown: Option<&(wgpu::TextureView, wgpu::Texture)>,
    display: glam::UVec2,
) -> Option<(
    Option<wgpu::TextureView>,
    Option<wgpu::Texture>,
    wgpu::TextureFormat,
    glam::UVec2,
)> {
    match (facts.target, shown) {
        (engine_core::Target::Surface, Some((canvas, surface))) => Some((
            Some(canvas.clone()),
            Some(surface.clone()),
            run.canvas.surface,
            glam::UVec2::new(surface.width(), surface.height()),
        )),
        (engine_core::Target::Surface, None) => Some((None, None, run.canvas.surface, display)),
        (engine_core::Target::Texture(handle), _) => {
            let (drawing, texture) = run.drawing_of(facts.slot, handle)?;
            let format = texture.format().remove_srgb_suffix();
            let extent = glam::UVec2::new(texture.width(), texture.height());
            Some((Some(drawing), Some(texture), format, extent))
        }
    }
}

fn keep_eligible(
    run: &mut Run,
    facts: &engine_core::ViewFacts,
    (first, main): (bool, bool),
) {
    let (runs, sets) = (&mut run.planning.runs, &run.sets);
    if !main {
        runs.remove_all(&sets.main);
    }
    if !first {
        runs.remove_all(&sets.fixed.global);
    }
    if !facts.target.surface() {
        runs.remove_all(&sets.fixed.main_views);
    }
}

fn skip_scenes(run: &mut Run, world: &World, view: bevy_ecs::entity::Entity) -> bool {
    let skipped = world.get::<engine_core::ViewSkip>(view).is_some();
    for (index, pass) in run.machine.declared.passes.iter().enumerate() {
        if skipped && pass.scene.is_some_and(|scene| scene.core) {
            run.planning.runs.remove(index);
        }
    }
    skipped
}

fn walk_held(run: &mut Run, timer: &GpuProfiler, skipped: bool) {
    if !skipped || !run.installs.draw_entries.iter().any(|member| member.batches.holds) {
        return;
    }
    let graph = &run.machine.graph;
    let (mut scenes, mut outputs) = (Bits::of(graph.passes()), Bits::of(graph.resources()));
    let mut written = outputs.clone();
    for (index, pass) in run.machine.declared.passes.iter().enumerate() {
        if pass.scene.is_some_and(|scene| scene.core) && run.sets.scratch.base.contains(index) {
            scenes.insert(index);
            outputs.union_with(graph.writes(index));
        }
    }
    (0..graph.resources()).for_each(|resource| written.insert(resource));
    let plan = std::sync::Arc::new(graph.plan(&scenes, &outputs, &written).unwrap_or_default());
    run.planning.plan = Some(std::sync::Arc::clone(&plan));
    for (at, planned) in plan.passes.iter().enumerate() {
        let index = planned.pass.index();
        let part = run.machine.declared.passes[index].part;
        if !run.drawable(part) {
            continue;
        }
        let Some(mut held) = run.parts[part as usize].take() else {
            continue;
        };
        run.part = part;
        run.pass = index as u32;
        run.planning.planned_at = at;
        run.collect();
        held.encode(index as u32, run, timer);
        run.record();
        run.parts[part as usize] = Some(held);
    }
}

#[cfg(feature = "dev")]
pub(super) mod names {
    use bevy_ecs::world::World;
    use engine_registry::{Kind, NameEntry, register, unavailable};
    use serde_json::{Map, Value, json};

    use super::super::Renderer;
    use crate::encode::Run;
    use crate::graph::keyed;
    use crate::gpu_calls::{Count, Written};

    type Reader = fn(&Renderer, &Run, &World) -> Value;

    pub fn gpu_names(world: &mut World) {
        let names: [(&'static str, Kind, Option<&'static str>, Reader); 19] = [
            ("gpu.plans.built", Kind::Count, Some("plans"), plans_built),
            ("gpu.passes.opened", Kind::Count, Some("passes"), passes_opened),
            ("gpu.created", Kind::Count, None, |_, _, _| crate::gpu_calls::created()),
            ("gpu.created.bytes", Kind::Bytes, Some("bytes"), |_, _, _| {
                crate::gpu_calls::created_bytes()
            }),
            ("gpu.views", Kind::State, None, views),
            ("gpu.resources", Kind::State, None, resources),
            ("gpu.resources.written", Kind::Bytes, Some("bytes"), |renderer, run, _| {
                written(renderer, run, |written| written.bytes)
            }),
            ("gpu.resources.unchanged", Kind::Bytes, Some("bytes"), unchanged),
            ("gpu.submitted_at_once", Kind::Count, None, |renderer, _, _| {
                json!(renderer.core.queue.submitted_at_once())
            }),
            ("gpu.assets.resident", Kind::Count, None, resident),
            ("gpu.assets.bytes", Kind::Bytes, Some("bytes"), asset_bytes),
            ("gpu.kinds", Kind::State, None, kinds),
            ("gpu.passes", Kind::State, None, passes),
            ("gpu.passes.draws", Kind::Count, None, |_, run, _| {
                pass_counts(run, |count| count.draws)
            }),
            ("gpu.passes.instances", Kind::Count, None, |_, run, _| {
                pass_counts(run, |count| count.instances)
            }),
            ("gpu.draws", Kind::State, None, draws),
            ("gpu.draws.draws", Kind::Count, None, |_, run, _| {
                part_counts(run, |count| count.draws)
            }),
            ("gpu.draws.instances", Kind::Count, None, |_, run, _| {
                part_counts(run, |count| count.instances)
            }),
            ("gpu.draws.dropped", Kind::Count, Some("draws"), |_, run, _| {
                super::super::kind_passes::dropped_totals(run)
            }),
        ];
        for (name, kind, unit, read) in names {
            let entry = NameEntry::read(name, kind, move |reach, _| {
                Ok(reach.part::<Renderer>().map_or_else(
                    || unavailable("no_renderer"),
                    |renderer| {
                        renderer.run.as_ref().map_or_else(
                            || unavailable("renderer_not_installed"),
                            |run| read(renderer, run, reach.world),
                        )
                    },
                ))
            });
            register(
                world,
                match unit {
                    Some(unit) => entry.unit(unit),
                    None => entry,
                },
            );
        }
        Renderer::timer_names(world);
        crate::whatif::names(world);
        super::super::install::declared_names(world);
    }

    fn part_of(run: &Run, part: u32) -> (&'static str, &'static str) {
        run.machine
            .declared
            .parts
            .get(part as usize)
            .map_or(("", engine_registry::CORE_ROW), |info| {
                (info.name, info.package)
            })
    }

    enum Declared {
        Buffer(usize),
        Texture(usize),
    }

    fn declared(run: &Run, mut row: impl FnMut(Declared) -> Value) -> Map<String, Value> {
        let declared = &run.machine.declared;
        let mut map = Map::new();
        for (at, texture) in declared.textures.iter().enumerate() {
            keyed(&mut map, texture.label, row(Declared::Texture(at)));
        }
        for (at, buffer) in declared.buffers.iter().enumerate() {
            keyed(&mut map, buffer.label, row(Declared::Buffer(at)));
        }
        map
    }

    fn views(_: &Renderer, run: &Run, world: &World) -> Value {
        let listed = world
            .get_resource::<engine_core::Views>()
            .map_or(&[][..], engine_core::Views::listed);
        let declared = &run.machine.declared.passes;
        let mut order = Vec::with_capacity(listed.len());
        super::drawn_order(listed, &mut order);
        let views: Vec<Value> = order
            .into_iter()
            .map(|index| {
                let facts = &listed[index];
                let passes: Vec<&'static str> = run
                    .sets
                    .by_slot
                    .get(facts.slot as usize)
                    .and_then(|&at| run.sets.list.get(at as usize))
                    .filter(|set| set.view == facts.view)
                    .map(|set| {
                        set.ran
                            .iter()
                            .filter(|&pass| run.sets.passes.contains(pass))
                            .filter_map(|pass| declared.get(pass).map(|pass| pass.name))
                            .collect()
                    })
                    .unwrap_or_default();
                let target = match facts.target {
                    engine_core::Target::Surface => json!("surface"),
                    engine_core::Target::Texture(handle) => json!({"texture": handle.slot()}),
                };
                let size = facts.size();
                json!({
                    "camera": facts.camera.to_string(),
                    "target": target,
                    "size": [size.x, size.y],
                    "layers": facts.layers.0,
                    "passes": passes,
                })
            })
            .collect();
        Value::Array(views)
    }

    fn resources(_: &Renderer, run: &Run, _: &World) -> Value {
        let declared_of = &run.machine.declared;
        Value::Object(declared(run, |resource| {
            let (kind, part, lifetime, size) = match resource {
                Declared::Buffer(at) => (
                    "buffer",
                    declared_of.buffers[at].part,
                    declared_of.buffers[at].lifetime,
                    run.buffer_held(at as u32)
                        .map_or_else(|| unavailable("not_allocated"), |(_, bytes)| json!(bytes)),
                ),
                Declared::Texture(at) => (
                    "texture",
                    declared_of.textures[at].part,
                    declared_of.textures[at].lifetime,
                    run.tables
                        .textures
                        .get(at)
                        .and_then(Option::as_ref)
                        .map_or_else(
                            || unavailable("not_allocated"),
                            |made| {
                                let size = made.size;
                                json!([size.width, size.height, size.depth_or_array_layers])
                            },
                        ),
                ),
            };
            let (part, brick) = part_of(run, part);
            let lifetime = crate::graph::Life::from(lifetime).name();
            json!({
                "kind": kind,
                "part": part,
                "brick": brick,
                "lifetime": lifetime,
                "size": size,
            })
        }))
    }

    fn written(renderer: &Renderer, run: &Run, pick: fn(Written) -> u64) -> Value {
        let writes = renderer.core.queue.writes();
        let of = |slots: &[Written], at: usize| json!(slots.get(at).copied().map_or(0, pick));
        let mut map = declared(run, |resource| match resource {
            Declared::Buffer(at) => of(&writes.buffers, at),
            Declared::Texture(at) => of(&writes.textures, at),
        });
        keyed(&mut map, "engine", json!(pick(writes.engine)));
        keyed(&mut map, "total", json!(pick(writes.total)));
        Value::Object(map)
    }

    fn unchanged(renderer: &Renderer, run: &Run, _: &World) -> Value {
        if renderer.core.queue.writes().marking() {
            written(renderer, run, |written| written.unchanged)
        } else {
            unavailable("unchanged_writes_off")
        }
    }

    fn resident(_: &Renderer, run: &Run, _: &World) -> Value {
        let (textures, _) = run.assets.textures.texture_totals();
        let formats: Map<String, Value> = run
            .assets
            .textures
            .texture_formats()
            .into_iter()
            .map(|(format, (count, bytes))| (format, json!({"textures": count, "bytes": bytes})))
            .collect();
        json!({"meshes": run.assets.meshes.len(), "textures": textures, "texture_formats": formats})
    }

    fn asset_bytes(_: &Renderer, run: &Run, _: &World) -> Value {
        let (_, textures) = run.assets.textures.texture_totals();
        json!({"meshes": run.assets.meshes.bytes(), "textures": textures})
    }

    fn kinds(_: &Renderer, run: &Run, _: &World) -> Value {
        json!({
            "passes": run.tallies.kinds.passes,
            "problems": run.tallies.kinds.problems,
            "dropped": super::super::kind_passes::dropped_draws(run),
            "joins": crate::registration::kind_passes(
                &run.machine.declared,
                &run.installs.draw_entries,
            ),
        })
    }

    fn passes(_: &Renderer, run: &Run, _: &World) -> Value {
        let mut map = Map::new();
        for (index, pass) in run.machine.declared.passes.iter().enumerate() {
            let (part, brick) = part_of(run, pass.part);
            let mut facts = run.machine.graph.pass_facts(index);
            facts.insert("part".to_owned(), json!(part));
            facts.insert("brick".to_owned(), json!(brick));
            keyed(&mut map, pass.name, Value::Object(facts));
        }
        Value::Object(map)
    }

    fn plans_built(_: &Renderer, run: &Run, _: &World) -> Value {
        let plans = run.sets.counted.plans;
        json!({
            "first": plans[0],
            "runs": plans[1],
            "outputs": plans[2],
            "ready": plans[3],
            "total": plans.iter().sum::<u64>(),
        })
    }

    fn passes_opened(_: &Renderer, run: &Run, _: &World) -> Value {
        let counted = &run.sets.counted;
        let (mut opened, mut merged) = (Map::new(), Map::new());
        for (index, pass) in run.machine.declared.passes.iter().enumerate() {
            let count = |counts: &[u64]| json!(counts.get(index).copied().unwrap_or(0));
            keyed(&mut opened, pass.name, count(&counted.opened));
            keyed(&mut merged, pass.name, count(&counted.merged));
        }
        json!({
            "opened": opened,
            "merged": merged,
            "total": counted.opened.iter().sum::<u64>(),
        })
    }

    fn pass_counts(run: &Run, pick: fn(Count) -> u64) -> Value {
        let mut map = Map::new();
        let counted = run.tallies.drawn.passes.iter();
        for (pass, count) in run.machine.declared.passes.iter().zip(counted) {
            keyed(&mut map, pass.name, json!(pick(*count)));
        }
        Value::Object(map)
    }

    fn draws(_: &Renderer, run: &Run, _: &World) -> Value {
        let mut map = Map::new();
        for part in &run.machine.declared.parts {
            keyed(&mut map, part.name, json!({"brick": part.package}));
        }
        Value::Object(map)
    }

    fn part_counts(run: &Run, pick: fn(Count) -> u64) -> Value {
        let mut map = Map::new();
        let counted = run.tallies.drawn.parts.iter();
        for (part, count) in run.machine.declared.parts.iter().zip(counted) {
            keyed(&mut map, part.name, json!(pick(*count)));
        }
        keyed(&mut map, "total", json!(pick(run.tallies.drawn.total)));
        Value::Object(map)
    }
}
