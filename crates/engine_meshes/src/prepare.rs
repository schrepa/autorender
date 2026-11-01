use bevy_ecs::system::SystemParamItem;
use engine_core::{GpuPresentation, NodeInfo, NodePose, PoseRows};
use engine_render::{AfterRead, CopyRegion, GpuPart, Prepare, Resolve, wgpu};

use crate::batches::BatchPlan;
use crate::casters::{CasterPlan, ID_CAPACITIES, MAPS, STILLNESS_BYTES, Source};
use crate::gpu::{FACES, MeshesGpu, NODE_LEVELS, PYRAMID_LEVELS};
use crate::layouts::{
    BatchData, COMPOSED_BYTES, CandidateData, CasterFrame, DRAW_ARGS_BYTES, GROUP, IMPOSTOR_VERTICES,
    LOD_LEVELS, PHASE1, PHASE2, PROBE, PresentationFrame, STILLNESS, VISIBLE,
};
use crate::order::{DrawOrder, MeshPlan};
use crate::presentation::{
    CASTER_TOTALS_BYTES, CasterTotals, DrawnBatch, FADE_MARGIN, Kept, PresentationTotals,
    TOTALS_BYTES, frustum_uniform, grow, jitter_clip, jittered, mip_levels, pyramid_extents,
    storage_bytes,
};
use crate::retained::RetainedRender;
use crate::uploads::{ColorData, PoseData};

const TOTALS: u32 = 0;
const CASTERS: u32 = 1;
const GROUNDS_BYTES: u64 = 16;

struct View {
    view_projection: glam::Mat4,
    eye: glam::Vec3,
    viewport: (u32, u32),
    motion: bool,
    occlusion: Option<u32>,
    lod_error: f32,
    impostors: bool,
    stillness: Option<bool>,
    fade: f32,
}

impl MeshesGpu {
    pub(crate) fn prepare_frame(
        &mut self,
        (
            retained,
            plan,
            order,
            decided,
            caster_plan,
            presented,
            hierarchy,
            view,
            detail,
            impostors,
            resolve,
            time,
            maps,
            captures,
            limits,
        ): SystemParamItem<'_, '_, <Self as GpuPart>::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.copy_draws(&plan, &order);
        self.state.groups.first_instance = limits.indirect_first_instance;
        let gpu = decided.gpu;
        let state = &mut self.state;
        if state.gpu_frame && !gpu {
            state.uploads.invalidate();
            state.slot_revision = None;
        }
        state.gpu_frame = gpu;
        state.instance_count = plan.ordered_instances.len();
        let temporal = gpu && *resolve != Resolve::default();
        self.prepare_instances(&retained, &plan, gpu, temporal, cx);
        let size = view.render_size();
        let motion = self.prepare_motion(&retained, size, temporal, cx);
        let fade = self.fade_share(gpu, time.presented_seconds as f32, detail.fade_seconds);
        let view_projection = view.view_projection();
        let rasterized = if temporal {
            jittered(view_projection, jitter_clip(resolve.offset.to_array(), size.x, size.y))
        } else {
            view_projection
        };
        let stillness = if captures.faces.is_empty() && !captures.still {
            None
        } else {
            Some(captures.still)
        };
        let occlusion_on = gpu && decided.occlusion_enabled;
        let culling = View {
            view_projection: rasterized,
            eye: view.detail_eye(),
            viewport: (size.x, size.y),
            motion,
            occlusion: occlusion_on.then(|| self.prepare_pyramid((size.x, size.y), cx)),
            lod_error: if detail.enabled {
                detail.pixel_error * view.render_share()
            } else {
                0.0
            },
            impostors: impostors.enabled,
            stillness,
            fade,
        };
        if gpu {
            let nodes = presented.gpu_presentation(&hierarchy);
            self.prepare_presentation(&retained, &plan, &caster_plan, &nodes, &culling, &decided, cx);
            self.prepare_faces(&captures, cx);
        } else {
            self.state.frame = None;
            self.state.occlusion = false;
            self.state.faces.clear();
        }
        self.prepare_casters(&caster_plan, &maps, cx);
    }

    fn prepare_motion(
        &mut self,
        retained: &RetainedRender,
        size: glam::UVec2,
        temporal: bool,
        cx: &mut Prepare<'_, Self>,
    ) -> bool {
        let state = &mut self.state;
        let key = (
            retained.identity(),
            retained.structure_revision(),
            state.capacity_generation,
            size,
        );
        let motion = temporal && state.motion_key == Some(key);
        state.motion_key = temporal.then_some(key);
        state.temporal = temporal;
        if temporal {
            let bytes = (state.capacity * size_of::<PoseData>()) as u64;
            cx.copy(
                self.passes.keep_poses,
                &[CopyRegion::Bytes {
                    from: 0,
                    to: 0,
                    size: bytes,
                }],
            );
        }
        motion
    }

    fn fade_share(&mut self, gpu: bool, seconds: f32, fade_seconds: f32) -> f32 {
        let since = seconds - self.state.presented_seconds.replace(seconds).unwrap_or(seconds);
        if !gpu {
            -1.0
        } else if fade_seconds > 0.0 && since < 1.0 {
            engine_core::math::clamp(since / fade_seconds, 0.0, 1.0)
        } else {
            1.0
        }
    }

    fn copy_draws(&mut self, plan: &BatchPlan, order: &DrawOrder) {
        let draws = &mut self.state.draws;
        let generation = (plan.generation(), order.generation);
        if draws.generation == Some(generation) {
            return;
        }
        draws.generation = Some(generation);
        draws.batches.clear();
        draws.batches.extend(plan.batches.iter().map(DrawnBatch::of));
        draws.order.clone_from(&order.order);
        draws.masked.clone_from(&order.masked);
        draws.looks.clone_from(&order.looks);
        draws.metas.clone_from(&order.metas);
    }

    fn prepare_instances(
        &mut self,
        retained: &RetainedRender,
        plan: &BatchPlan,
        gpu: bool,
        temporal: bool,
        cx: &mut Prepare<'_, Self>,
    ) {
        let state = &mut self.state;
        let instances = &self.instances;
        let payload = retained.snapshot().instances.len();
        let limit = engine_places::meshes::MOVED_SLOT as usize;
        if state.capacity == 0 || payload > state.capacity {
            state.capacity = payload.max(INITIAL_CAPACITY).next_power_of_two().min(limit);
            state.capacity_generation += 1;
            state.uploads.invalidate();
        }
        let capacity = state.capacity as u64;
        cx.allocate_buffer(instances.poses, capacity * size_of::<PoseData>() as u64);
        cx.allocate_buffer(instances.colors, capacity * size_of::<ColorData>() as u64);
        cx.allocate_buffer(instances.grounds, GROUNDS_BYTES);
        if temporal {
            cx.allocate_buffer(instances.previous_poses, capacity * size_of::<PoseData>() as u64);
        } else {
            cx.allocate_buffer(instances.previous_poses, storage_bytes(0));
        }
        let slot_key = Some((retained.identity(), retained.structure_revision()));
        if !gpu {
            let count = plan.ordered_instances.len();
            if state.slot_capacity == 0 || count > state.slot_capacity {
                state.slot_capacity = count.max(INITIAL_CAPACITY).next_power_of_two().min(limit);
                state.slot_revision = None;
            }
            cx.allocate_buffer(instances.slots, (state.slot_capacity * size_of::<u32>()) as u64);
            if state.slot_revision != slot_key {
                state.slot_staging.clear();
                state.slot_staging.reserve_exact(count);
                state
                    .slot_staging
                    .extend(plan.ordered_instances.iter().map(|&slot| slot as u32));
                if !state.slot_staging.is_empty() {
                    cx.write(instances.slots, 0, &state.slot_staging);
                }
            }
        }
        if state.color_staging.capacity() < state.capacity {
            state
                .color_staging
                .reserve_exact(state.capacity - state.color_staging.len());
        }
        state.uploads.prepare(retained, &mut state.color_staging);
        if !gpu {
            let rows: &[PoseRows] = retained.pose_rows();
            let stride = size_of::<PoseData>();
            for range in state.uploads.pose_ranges() {
                cx.write(instances.poses, (range.start * stride) as u64, &rows[range.start..range.end]);
            }
        }
        for range in state.uploads.color_ranges() {
            cx.write(
                instances.colors,
                (range.start * size_of::<ColorData>()) as u64,
                &state.color_staging[range.start..range.end],
            );
        }
        state.uploads.commit();
        state.slot_revision = slot_key;
    }

    fn prepare_pyramid(&mut self, display: (u32, u32), cx: &mut Prepare<'_, Self>) -> u32 {
        let pyramid = &mut self.state.pyramid;
        let base = (
            display.0.div_ceil(2).max(1).next_power_of_two(),
            display.1.div_ceil(2).max(1).next_power_of_two(),
        );
        let count = mip_levels(base.0, base.1).min(PYRAMID_LEVELS as u32);
        if pyramid.base != base || pyramid.count != count {
            pyramid.base = base;
            pyramid.count = count;
            pyramid.drawn = (0, 0);
            cx.allocate(
                self.presented.pyramid,
                wgpu::Extent3d {
                    width: base.0,
                    height: base.1,
                    depth_or_array_layers: 1,
                },
                count,
            );
        }
        if pyramid.drawn != display {
            pyramid.drawn = display;
            let extents = pyramid_extents(display, count);
            pyramid.written.clear();
            for (level, extent) in extents.iter().enumerate() {
                let buffer = self.presented.extents[level];
                cx.allocate_buffer(buffer, 16);
                cx.write(buffer, 0, extent);
                pyramid.written.push((extent[0], extent[1]));
            }
        }
        count
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the presentation joins the plan, the casters, the hierarchy and the view"
    )]
    fn prepare_presentation(
        &mut self,
        retained: &RetainedRender,
        plan: &BatchPlan,
        casters: &CasterPlan,
        gpu: &GpuPresentation<'_>,
        view: &View,
        decided: &MeshPlan,
        cx: &mut Prepare<'_, Self>,
    ) {
        let held = self.state.nodes.holds(gpu);
        self.ensure_nodes(gpu.info.len(), if held { 0 } else { gpu.parents.len() }, cx);
        let kept = Kept {
            phase_two: view.occlusion.is_some(),
            probe_faces: view.stillness.is_some(),
            impostors: view.impostors && view.lod_error > 0.0,
        };
        self.ensure_groups((plan.ordered_instances.len(), plan.batches.len()), kept, cx);
        self.ensure_stillness(view.stillness, cx);
        self.ensure_fades(cx);
        if let Some((since, shift)) = self.state.nodes.gather(gpu, held) {
            self.state.nodes.shift = shift;
            self.upload_nodes(gpu, since, cx);
        } else {
            self.state.nodes.shift = false;
        }
        self.rebuild_groups(retained, plan, casters, gpu, cx);
        self.write_frame_uniform(gpu.alpha, view, decided, cx);
        cx.run(self.passes.clear_counts);
        cx.run(self.passes.presentation);
        if self.state.occlusion {
            cx.run(self.passes.occlusion);
        }
        if self.state.candidates != 0 {
            cx.read_back::<PresentationTotals>(
                TOTALS,
                self.presented.totals,
                0..TOTALS_BYTES,
                AfterRead::Keep,
            );
        }
    }

    fn ensure_nodes(&mut self, nodes: usize, parents: usize, cx: &mut Prepare<'_, Self>) {
        let inputs = &mut self.state.nodes;
        let presented = &self.presented;
        if nodes > inputs.capacity {
            inputs.capacity = grow(nodes);
            inputs.uploaded = None;
            inputs.composed_capacity = 0;
        }
        let capacity = inputs.capacity as u64;
        cx.allocate_buffer(presented.locals, storage_bytes(2 * capacity * size_of::<NodePose>() as u64));
        cx.allocate_buffer(presented.info, storage_bytes(capacity * size_of::<NodeInfo>() as u64));
        cx.allocate_buffer(presented.authority, storage_bytes(capacity * size_of::<PoseRows>() as u64));
        if parents != 0 && inputs.composed_capacity < inputs.capacity {
            inputs.composed_capacity = inputs.capacity;
        }
        cx.allocate_buffer(
            presented.composed,
            storage_bytes(inputs.composed_capacity as u64 * COMPOSED_BYTES),
        );
        if parents > inputs.parent_capacity {
            inputs.parent_capacity = grow(parents);
        }
        cx.allocate_buffer(presented.parents, storage_bytes(inputs.parent_capacity as u64 * 4));
        for &level in &presented.levels {
            cx.allocate_buffer(level, 16);
        }
    }

    fn ensure_groups(&mut self, (candidates, batches): (usize, usize), kept: Kept, cx: &mut Prepare<'_, Self>) {
        let groups = &mut self.state.groups;
        let presented = &self.presented;
        if candidates > groups.capacity {
            groups.capacity = grow(candidates);
            groups.key = None;
        }
        if batches > groups.batch_capacity {
            groups.batch_capacity = batches.max(64).next_power_of_two();
            groups.key = None;
        }
        let capacity = groups.capacity as u64;
        let scan_groups = capacity.div_ceil(u64::from(GROUP));
        cx.allocate_buffer(presented.candidates, storage_bytes(capacity * size_of::<CandidateData>() as u64));
        cx.allocate_buffer(presented.slot_candidates, storage_bytes(capacity * size_of::<CandidateData>() as u64));
        cx.allocate_buffer(presented.scan, storage_bytes((capacity + 4 * (scan_groups + 1) + 1) * 4));
        let batch_capacity = groups.batch_capacity as u64;
        cx.allocate_buffer(presented.batches, storage_bytes(batch_capacity * size_of::<BatchData>() as u64));
        cx.allocate_buffer(presented.batch_bases, storage_bytes(batch_capacity * 16));
        let planes = u64::from(LOD_LEVELS);
        let args = (planes * batch_capacity + 1) * DRAW_ARGS_BYTES;
        let ids = planes * capacity * 4;
        let compactions = [true, kept.phase_two];
        for (index, wanted) in compactions.into_iter().enumerate() {
            cx.allocate_buffer(presented.visible[index], storage_bytes(if wanted { ids } else { 0 }));
            cx.allocate_buffer(presented.draws[index], storage_bytes(if wanted { args } else { 0 }));
        }
        for face in 0..FACES {
            let wanted = kept.probe_faces;
            cx.allocate_buffer(presented.face_visible[face], storage_bytes(if wanted { ids } else { 0 }));
            cx.allocate_buffer(presented.face_draws[face], storage_bytes(if wanted { args } else { 0 }));
        }
        let impostors = [kept.impostors, kept.impostors && kept.phase_two];
        for (index, wanted) in impostors.into_iter().enumerate() {
            cx.allocate_buffer(
                presented.impostor_ids[index],
                storage_bytes(if wanted { capacity * 8 } else { 0 }),
            );
        }
        cx.allocate_buffer(presented.counts, storage_bytes(16));
        cx.allocate_buffer(presented.totals, storage_bytes(TOTALS_BYTES));
        if groups.kept != kept {
            groups.key = None;
        }
        groups.kept = kept;
    }

    fn ensure_stillness(&mut self, stillness: Option<bool>, cx: &mut Prepare<'_, Self>) {
        let still = &mut self.state.stillness;
        let slots = if stillness.is_some() {
            self.state.capacity as u64
        } else {
            0
        };
        if slots > still.slots || (slots == 0 && still.slots != 0) {
            still.slots = slots;
            still.counted = 0;
        }
        cx.allocate_buffer(self.instances.still_frames, storage_bytes(still.slots * 8));
        let counting = stillness == Some(true);
        if counting && !still.counting {
            still.counted = 0;
        }
        still.counting = counting;
        if counting {
            still.counted = (still.counted + 1).min(engine_places::meshes::STILL_FRAMES);
        }
    }

    fn ensure_fades(&mut self, cx: &mut Prepare<'_, Self>) {
        let state = &mut self.state;
        let slots = state.capacity as u64;
        if slots > state.fade_slots {
            state.fade_slots = slots;
        }
        if state.groups.capacity > state.card_capacity {
            state.card_capacity = state.groups.capacity;
        }
        cx.allocate_buffer(self.presented.lod_states, storage_bytes(state.fade_slots * 4));
        cx.allocate_buffer(
            self.presented.fade_cards,
            storage_bytes(16 + state.card_capacity as u64 * 8),
        );
        cx.write(self.presented.fade_cards, 0, &[IMPOSTOR_VERTICES, 0, 0, 0]);
    }

    fn upload_nodes(&mut self, gpu: &GpuPresentation<'_>, since: Option<u64>, cx: &mut Prepare<'_, Self>) {
        let nodes = &mut self.state.nodes;
        let presented = &self.presented;
        nodes.plan_ranges(gpu, since);
        if nodes.shift {
            nodes.stage_locals(gpu);
            if !nodes.rows.is_empty() {
                cx.allocate_buffer(presented.node_staging, storage_bytes((nodes.staged.len() * 4) as u64));
                cx.allocate_buffer(presented.node_ranges, storage_bytes((nodes.rows.len() * 16) as u64));
                cx.write(presented.node_staging, 0, &nodes.staged);
                cx.write(presented.node_ranges, 0, &nodes.rows);
            }
        } else {
            nodes.rows.clear();
            let pose = size_of::<NodePose>();
            for range in &nodes.ranges[0] {
                cx.write(presented.locals, (range.start * pose) as u64, &gpu.current[range.start..range.end]);
            }
            let previous = (nodes.capacity * pose) as u64;
            for range in &nodes.ranges[1] {
                cx.write(
                    presented.locals,
                    previous + (range.start * pose) as u64,
                    &gpu.previous[range.start..range.end],
                );
            }
        }
        for range in &nodes.ranges[2] {
            cx.write(
                presented.info,
                (range.start * size_of::<NodeInfo>()) as u64,
                &gpu.info[range.start..range.end],
            );
        }
        for range in &nodes.ranges[3] {
            cx.write(
                presented.authority,
                (range.start * size_of::<PoseRows>()) as u64,
                &gpu.authority[range.start..range.end],
            );
        }
        if !gpu.parents.is_empty() {
            cx.write(presented.parents, 0, gpu.parents);
        }
        nodes.levels(gpu);
        for (level, &(start, end)) in nodes.level_ranges.iter().enumerate().take(NODE_LEVELS) {
            cx.write(presented.levels[level], 0, &[start, end, 0, 0]);
        }
        nodes.uploaded = Some((gpu.identity, gpu.sequence));
    }

    fn rebuild_groups(
        &mut self,
        retained: &RetainedRender,
        plan: &BatchPlan,
        casters: &CasterPlan,
        gpu: &GpuPresentation<'_>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let groups = &mut self.state.groups;
        let key = (
            retained.identity(),
            plan.generation(),
            retained.source_revision(),
            gpu.identity,
            casters.generation,
        );
        if groups.key == Some(key) {
            return;
        }
        groups.stage(
            &plan.batches,
            &plan.ordered_instances,
            retained.slot_nodes(),
            &self.state.draws.metas,
            |index| casters.batch_group(index),
        );
        if !groups.candidate_staging.is_empty() {
            cx.write(self.presented.candidates, 0, &groups.candidate_staging);
            cx.write(self.presented.slot_candidates, 0, &groups.slot_staging);
        }
        if !groups.batch_staging.is_empty() {
            cx.write(self.presented.batches, 0, &groups.batch_staging);
        }
        groups.key = Some(key);
    }

    fn write_frame_uniform(&mut self, alpha: f32, view: &View, decided: &MeshPlan, cx: &mut Prepare<'_, Self>) {
        let state = &mut self.state;
        let (planes, absolute) = frustum_uniform(view.view_projection);
        let candidates = state.groups.count;
        state.candidates = candidates;
        state.occlusion = decided.occlusion;
        let unbudgeted = decided.culling && decided.budget as usize >= candidates as usize;
        let (width, height) = view.viewport;
        let mut uniform = PresentationFrame {
            alpha,
            candidates,
            budget: decided.budget,
            culling: u32::from(decided.culling),
            groups: candidates.div_ceil(GROUP),
            batches: state.groups.batch_count,
            node_capacity: state.nodes.capacity as u32,
            blend: u32::from(state.nodes.blend),
            planes,
            absolute_view_projection: absolute,
            motion: u32::from(view.motion),
            mask: if state.occlusion { PHASE1 } else { VISIBLE },
            occlusion: u32::from(state.occlusion),
            subpixel: u32::from(decided.subpixel),
            view_projection: view.view_projection.to_cols_array_2d(),
            viewport: [width as f32, height as f32, view.occlusion.unwrap_or(0) as f32, 0.0],
            plane: state.groups.capacity as u32,
            lod_error: if unbudgeted { view.lod_error } else { 0.0 },
            pad: [0; 2],
            eye: view
                .eye
                .extend(if view.impostors { 1.0 } else { 0.0 })
                .to_array(),
            fade: [view.fade, FADE_MARGIN, 0.0, 0.0],
        };
        cx.allocate_buffer(self.presented.frame, size_of::<PresentationFrame>() as u64);
        cx.write(self.presented.frame, 0, &[uniform]);
        state.lod = uniform.lod_error > 0.0;
        state.frame = Some(uniform);
        if state.occlusion {
            uniform.mask = PHASE2;
            cx.allocate_buffer(self.presented.frame_two, size_of::<PresentationFrame>() as u64);
            cx.write(self.presented.frame_two, 0, &[uniform]);
        }
    }

    fn prepare_faces(&mut self, captures: &engine_places::probes::ProbeCaptures, cx: &mut Prepare<'_, Self>) {
        self.state.faces.clear();
        let Some(base) = self.state.frame else {
            return;
        };
        if self.state.candidates == 0 || !self.state.groups.kept.probe_faces {
            return;
        }
        for face in &captures.faces {
            let slot = face.face as usize;
            if slot >= FACES || self.state.faces.contains(&face.face) {
                continue;
            }
            let (planes, absolute) = frustum_uniform(face.view);
            let mut uniform = base;
            uniform.planes = planes;
            uniform.absolute_view_projection = absolute;
            uniform.view_projection = face.view.to_cols_array_2d();
            uniform.culling = 1;
            uniform.mask = PROBE;
            uniform.occlusion = 0;
            uniform.subpixel = 1;
            uniform.viewport = [captures.size as f32, captures.size as f32, 0.0, 0.0];
            uniform.motion = 0;
            uniform.budget = u32::MAX;
            uniform.lod_error = 0.0;
            let frame = self.presented.faces[slot];
            cx.allocate_buffer(frame, size_of::<PresentationFrame>() as u64);
            cx.write(frame, 0, &[uniform]);
            self.state.faces.push(face.face);
        }
        if !self.state.faces.is_empty() {
            cx.run(self.passes.probes);
        }
    }

    fn prepare_casters(
        &mut self,
        plan: &CasterPlan,
        maps: &engine_places::shadows::ShadowMaps,
        cx: &mut Prepare<'_, Self>,
    ) {
        let buffers = &self.casters;
        let draws = &mut self.state.casters;
        draws.maps.clone_from(&plan.maps);
        draws.runs = maps.on && !maps.regions.is_empty() && !plan.groups.is_empty();
        if !draws.runs {
            return;
        }
        let classify = plan.source == Some(Source::Presented);
        cx.allocate_buffer(buffers.stillness, storage_bytes(plan.stillness_capacity as u64 * STILLNESS_BYTES));
        cx.allocate_buffer(buffers.ids, storage_bytes((ID_CAPACITIES * plan.id_capacity * 4) as u64));
        let args = (MAPS * plan.arg_capacity) as u64 * DRAW_ARGS_BYTES;
        cx.allocate_buffer(buffers.args, storage_bytes(args));
        cx.allocate_buffer(buffers.template, storage_bytes(args));
        cx.allocate_buffer(buffers.totals, storage_bytes(CASTER_TOTALS_BYTES));
        cx.allocate_buffer(buffers.starts, storage_bytes((plan.starts.len() * 4) as u64));
        cx.allocate_buffer(buffers.no_flags, storage_bytes(0));
        if draws.generation != Some(plan.generation) {
            draws.generation = Some(plan.generation);
            draws.groups.clear();
            draws.groups.extend(plan.groups.iter().map(|group| (group.mesh, group.material, group.start)));
            draws.masked.clone_from(&plan.masked);
            draws.regions_in_args = plan.regions_in_args;
            draws.id_capacity = plan.id_capacity;
            if !plan.template.is_empty() {
                cx.write(buffers.template, 0, &plan.template);
            }
            if !plan.starts.is_empty() {
                cx.write(buffers.starts, 0, &plan.starts);
            }
            if !classify {
                cx.allocate_buffer(buffers.listed, storage_bytes((plan.listed.len().max(1) * size_of::<CandidateData>()) as u64));
                cx.allocate_buffer(
                    buffers.listed_groups,
                    storage_bytes((plan.records.len().max(1) * size_of::<BatchData>()) as u64),
                );
                if !plan.listed.is_empty() {
                    cx.write(buffers.listed, 0, &plan.listed);
                }
                cx.write(buffers.listed_groups, 0, &plan.records);
            }
        }
        let candidates = if classify {
            self.state.groups.count
        } else {
            plan.listed.len() as u32
        };
        draws.candidates = candidates;
        draws.listed = !classify;
        let frame = CasterFrame {
            candidates,
            groups: plan.groups.len() as u32,
            capacity: plan.id_capacity as u32,
            mode: if classify { STILLNESS } else { 0 },
        };
        cx.allocate_buffer(buffers.select, size_of::<CasterFrame>() as u64);
        cx.write(buffers.select, 0, &[frame]);
        cx.copy(
            self.passes.caster_args,
            &[CopyRegion::Bytes {
                from: 0,
                to: 0,
                size: MAPS as u64 * plan.groups.len() as u64 * DRAW_ARGS_BYTES,
            }],
        );
        cx.run(self.passes.casters);
        cx.read_back::<CasterTotals>(CASTERS, buffers.totals, 0..CASTER_TOTALS_BYTES, AfterRead::Clear);
    }
}

const INITIAL_CAPACITY: usize = 1_024;
