use bevy_ecs::system::SystemParamItem;
use engine_core::{
    DrawnPoses, GlobalTransforms, GpuBits, ImpostorRecord, NodeInfo, NodePose, NodeStreams,
    PoseRows, Since,
};
use engine_render::{
    AfterRead, Allocation, CopyRegion, GpuPart, Prepare, mip_levels, pyramid_extents, wgpu,
};

use crate::batches::BatchPlan;
use crate::casters::{CACHE_BYTES, CasterPlan, ID_CAPACITIES, MAPS};
use crate::gpu::{FACES, MeshesGpu, NODE_LEVELS, PYRAMID_LEVELS};
use crate::layouts::{
    BatchData, COMPOSED_BYTES, CandidateData, CandidateState, CasterFrame, DRAW_ARGS_BYTES, GROUP,
    LOD_LEVELS, NO_NODE, PresentationFrame, UNWATCHED,
};
use crate::order::MeshPlan;
use crate::presentation::{
    CASTER_TOTALS_BYTES, CasterTotals, FADE_MARGIN, Kept, POSE, PresentationTotals,
    TOTALS_BYTES, UploadSources, WATCHED, frustum_uniform, grow, rasterized, storage_bytes,
};
use crate::retained::RetainedRender;
use crate::uploads::{ColorData, PoseData, UploadRange, color_data, listed_ranges};

const TOTALS: u32 = 0;
const CASTERS: u32 = 1;

struct View {
    view_projection: glam::Mat4,
    eye: glam::Vec3,
    viewport: (u32, u32),
    motion: bool,
    occlusion: Option<u32>,
    lod_error: f32,
    impostors: bool,
    probes: bool,
    fade: f32,
    watched: Option<u32>,
}

struct Prepared {
    uploaded: Option<(bool, u64)>,
    group_bytes: u64,
}

impl MeshesGpu {
    pub(crate) fn prepare_frame(
        &mut self,
        (
            (
                retained,
                plan,
                assets,
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
            ),
            (records, watched, totals, residency),
        ): SystemParamItem<'_, '_, <Self as GpuPart>::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        if let Some(error) = &plan.refused {
            cx.refuse(error.clone());
            return;
        }
        self.state.draws.plan(&plan.batches, &assets, &residency);
        let held = records
            .single()
            .ok()
            .map_or(&[][..], |held| held.records.as_slice());
        self.write_impostors(held, cx);
        self.state.casts = maps.on;
        self.state.groups.first_instance = limits.indirect_first_instance;
        self.state.instance_count = plan.ordered.len();
        let temporal = view.temporal();
        let fresh = self.prepare_instances(&retained, &plan, temporal, cx);
        let size = view.render_size();
        let motion = self.prepare_motion(&plan, (view.display_size(), temporal, fresh), cx);
        let shown = cx.ran(self.passes.presentation);
        let fade = self.fade_share(time.real_elapsed as f32, detail.fade_seconds, shown);
        if !decided.occlusion_enabled {
            self.release_pyramid(cx);
        }
        let culling = View {
            view_projection: rasterized(&view, resolve.offset, temporal),
            eye: view.detail_eye(),
            viewport: (size.x, size.y),
            motion,
            occlusion: decided.occlusion_enabled.then(|| {
                let display = view.display_size();
                self.prepare_pyramid(((display.x, display.y), (size.x, size.y)), cx)
            }),
            lod_error: if detail.enabled {
                detail.pixel_error * view.render_share()
            } else {
                0.0
            },
            impostors: impostors.enabled,
            probes: captures.views,
            fade,
            watched: watched.slot,
        };
        let nodes = presented.streams(&hierarchy);
        let prepared = self.prepare_presentation(
            &retained,
            &plan,
            &caster_plan,
            (&*presented, &nodes),
            &culling,
            &decided,
            UploadSources {
                assets: &assets,
                residency: &residency,
                hierarchy: &hierarchy,
                impostors: held,
            },
            cx,
        );
        if engine_core::profile::active() {
            self.gpu_counters(&prepared, totals.last().map(|last| last.visible));
        }
        self.prepare_faces(&captures, cx);
        self.prepare_casters(&caster_plan, &maps, cx);
    }

    fn prepare_motion(
        &mut self,
        plan: &BatchPlan,
        (size, temporal, fresh): (glam::UVec2, bool, bool),
        cx: &mut Prepare<'_, Self>,
    ) -> bool {
        let shown = cx.ran(self.passes.presentation);
        let state = &mut self.state;
        let key = (plan.generation(), size);
        let motion = temporal && !fresh && shown && state.motion_key == Some(key);
        state.motion_key = temporal.then_some(key);
        state.temporal = temporal;
        if temporal {
            engine_core::profile::counter("render.taa_object_motion", f64::from(u8::from(motion)));
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

    fn fade_share(&mut self, seconds: f32, fade_seconds: f32, shown: bool) -> f32 {
        let state = &mut self.state;
        if shown {
            state.faded_at = state.prepared_at;
        }
        state.prepared_at = Some(seconds);
        let since = state.faded_at.map_or(f32::INFINITY, |at| seconds - at);
        if fade_seconds > 0.0 && since < 1.0 {
            engine_core::math::clamp(since / fade_seconds, 0.0, 1.0)
        } else {
            1.0
        }
    }

    fn write_impostors(&mut self, records: &[ImpostorRecord], cx: &mut Prepare<'_, Self>) {
        self.state.cards = records.iter().any(|record| record.sphere[3] > 0.0);
        let bytes = (records.len().max(1) * size_of::<ImpostorRecord>()) as u64;
        cx.allocate_buffer(self.instances.impostors, bytes);
        if !records.is_empty() {
            cx.write(self.instances.impostors, 0, records);
        }
    }

    fn prepare_instances(
        &mut self,
        retained: &RetainedRender,
        plan: &BatchPlan,
        temporal: bool,
        cx: &mut Prepare<'_, Self>,
    ) -> bool {
        let _profile = engine_core::profile::span("render", "instance_buffer_and_packing");
        let state = &mut self.state;
        let instances = &self.instances;
        let capacity = plan.instance_capacity as u64;
        let records_bytes = crate::debug::records_bytes();
        if cx.allocate_buffer(instances.records, records_bytes) == Allocation::Fresh {
            cx.write(instances.records, 0, &crate::debug::record_words());
        }
        let poses = cx.allocate_buffer(instances.poses, capacity * size_of::<PoseData>() as u64);
        if poses == Allocation::Fresh && state.capacity != 0 {
            engine_core::profile::counter("render.instance_buffer_growth", 1.0);
        }
        state.capacity = plan.instance_capacity;
        let colors =
            cx.allocate_buffer(instances.colors, capacity * size_of::<ColorData>() as u64);
        let previous_bytes = if temporal {
            capacity * size_of::<PoseData>() as u64
        } else {
            storage_bytes(0)
        };
        let previous = cx.allocate_buffer(instances.previous_poses, previous_bytes);
        let records = retained.instances();
        let stride = size_of::<ColorData>();
        state.color_ranges.clear();
        match (colors, retained.colors().since(&mut state.colors_seen)) {
            (Allocation::Fresh, _) | (Allocation::Kept, Since::All) => {
                if !records.is_empty() {
                    state.color_ranges.push(UploadRange {
                        start: 0,
                        end: records.len(),
                    });
                }
            }
            (Allocation::Kept, Since::Listed(slots)) => {
                listed_ranges(
                    slots,
                    records.len(),
                    stride,
                    &mut state.color_bits,
                    &mut state.color_ranges,
                );
            }
            (Allocation::Kept, Since::Same) => {}
        }
        let mut color_bytes = 0;
        for range in &state.color_ranges {
            state.color_staging.clear();
            state
                .color_staging
                .extend(records[range.start..range.end].iter().map(color_data));
            cx.write(
                instances.colors,
                (range.start * stride) as u64,
                &state.color_staging,
            );
            color_bytes += std::mem::size_of_val(state.color_staging.as_slice()) as u64;
        }
        cx.count_upload(color_bytes);
        let (pose, color) = (size_of::<PoseData>() as u64, size_of::<ColorData>() as u64);
        cx.memory(
            "memory.gpu_temporal_bytes",
            if temporal { capacity * pose } else { 0 },
        );
        cx.memory(
            "memory.render_staging_capacity_bytes",
            (state.color_staging.capacity() * size_of::<ColorData>()) as u64,
        );
        cx.memory(
            "memory.gpu_instance_buffer_bytes",
            capacity * (pose + color),
        );
        poses == Allocation::Fresh || previous == Allocation::Fresh
    }

    fn prepare_pyramid(
        &mut self,
        (display, render): ((u32, u32), (u32, u32)),
        cx: &mut Prepare<'_, Self>,
    ) -> u32 {
        let pyramid = &mut self.state.pyramid;
        let base = (
            display.0.div_ceil(2).max(1).next_power_of_two(),
            display.1.div_ceil(2).max(1).next_power_of_two(),
        );
        let count = mip_levels(base.0, base.1).min(PYRAMID_LEVELS as u32);
        pyramid.count = count;
        cx.allocate(
            self.presented.pyramid,
            wgpu::Extent3d {
                width: base.0,
                height: base.1,
                depth_or_array_layers: 1,
            },
            count,
        );
        pyramid.written.clear();
        for (level, extent) in pyramid_extents(render, count).iter().enumerate() {
            let buffer = self.presented.extents[level];
            cx.allocate_buffer(buffer, 16);
            cx.write(buffer, 0, extent);
            pyramid.written.push((extent[0], extent[1]));
        }
        count
    }

    fn release_pyramid(&mut self, cx: &mut Prepare<'_, Self>) {
        let pyramid = std::mem::take(&mut self.state.pyramid);
        if pyramid.count == 0 {
            return;
        }
        cx.release(self.presented.pyramid);
        for level in 0..pyramid.written.len() {
            cx.release_buffer(self.presented.extents[level]);
        }
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
        (poses, streams): (&DrawnPoses, &NodeStreams<'_>),
        view: &View,
        decided: &MeshPlan,
        sources: UploadSources<'_>,
        cx: &mut Prepare<'_, Self>,
    ) -> Prepared {
        self.ensure_nodes(streams.info.len(), streams.parents.len(), cx);
        let kept = Kept {
            phase_two: view.occlusion.is_some(),
            probe_faces: view.probes,
            impostors: view.impostors && view.lod_error > 0.0,
        };
        self.ensure_groups((plan.ordered.len(), plan.batches.len()), kept, cx);
        self.ensure_fades(cx);
        let states = storage_bytes(self.state.fade_slots * 4);
        self.state.watched = view
            .watched
            .filter(|&slot| u64::from(slot) * 4 + 4 <= states);
        let uploaded = self
            .state
            .nodes
            .plan(poses, streams.info.len())
            .map(|full| (full, self.upload_nodes(streams, cx)));
        self.upload_words(sources.hierarchy, streams.info.len(), cx);
        let group_bytes = self.write_batches(plan, casters, &sources, cx)
            + self.write_candidates(retained, plan, sources.hierarchy, cx);
        self.write_frame_uniform(streams.alpha, view, decided, cx);
        cx.run(self.passes.presentation);
        if self.state.occlusion {
            cx.run(self.passes.occlusion);
        }
        if self.state.candidates != 0
            && cx.read_back::<PresentationTotals>(
                TOTALS,
                self.presented.totals,
                0..TOTALS_BYTES,
                AfterRead::Keep,
            )
            && let Some(slot) = self.state.watched
        {
            cx.deliver::<PresentationTotals>(WATCHED, slot.to_le_bytes().to_vec());
            if (slot as usize) < self.state.capacity {
                let (at, size) = (u64::from(slot), size_of::<PoseData>() as u64);
                cx.read_back::<PresentationTotals>(
                    POSE,
                    self.instances.poses,
                    at * size..(at + 1) * size,
                    AfterRead::Keep,
                );
            }
        }
        Prepared {
            uploaded,
            group_bytes,
        }
    }

    fn gpu_counters(&self, prepared: &Prepared, visible: Option<u32>) {
        use engine_core::profile::counter;
        let nodes = &self.state.nodes;
        let (full, bytes) = prepared.uploaded.unwrap_or((false, 0));
        counter("render.gpu_candidates", f64::from(self.state.groups.count));
        counter("render.gpu_node_upload_bytes", bytes as f64);
        counter("render.gpu_group_upload_bytes", prepared.group_bytes as f64);
        counter("render.gpu_full_node_upload", f64::from(u8::from(full)));
        counter("render.gpu_parent_levels", nodes.level_ranges.len() as f64);
        if let Some(visible) = visible {
            counter("render.gpu_visible_readback", f64::from(visible));
        }
    }

    fn ensure_nodes(&mut self, nodes: usize, parents: usize, cx: &mut Prepare<'_, Self>) {
        let inputs = &mut self.state.nodes;
        let presented = &self.presented;
        if nodes > inputs.capacity {
            inputs.capacity = grow(nodes);
            inputs.composed_capacity = 0;
        }
        let capacity = inputs.capacity as u64;
        let sizes = [
            (
                presented.locals,
                storage_bytes(2 * capacity * size_of::<NodePose>() as u64),
            ),
            (
                presented.info,
                storage_bytes(capacity * size_of::<NodeInfo>() as u64),
            ),
            (
                presented.authority,
                storage_bytes(capacity * size_of::<PoseRows>() as u64),
            ),
            (
                presented.node_words,
                storage_bytes(capacity * size_of::<u32>() as u64),
            ),
        ];
        let mut fresh = false;
        for (buffer, bytes) in sizes {
            fresh |= cx.allocate_buffer(buffer, bytes) == Allocation::Fresh;
        }
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
        let parent_bytes = storage_bytes(inputs.parent_capacity as u64 * 4);
        fresh |= cx.allocate_buffer(presented.parents, parent_bytes) == Allocation::Fresh;
        for &level in &presented.levels {
            fresh |= cx.allocate_buffer(level, 16) == Allocation::Fresh;
        }
        if fresh {
            inputs.whole = true;
            inputs.words_whole = true;
        }
    }

    fn ensure_groups(
        &mut self,
        (candidates, batches): (usize, usize),
        kept: Kept,
        cx: &mut Prepare<'_, Self>,
    ) {
        let groups = &mut self.state.groups;
        let presented = &self.presented;
        if candidates > groups.capacity {
            groups.capacity = grow(candidates);
        }
        if batches > groups.batch_capacity {
            groups.batch_capacity = batches.max(64).next_power_of_two();
        }
        let capacity = groups.capacity as u64;
        let scan_groups = capacity.div_ceil(u64::from(GROUP));
        let staged = [
            cx.allocate_buffer(
                presented.candidates,
                storage_bytes(capacity * size_of::<CandidateData>() as u64),
            ),
            cx.allocate_buffer(
                presented.slot_candidates,
                storage_bytes(capacity * size_of::<CandidateData>() as u64),
            ),
        ];
        if staged.contains(&Allocation::Fresh) {
            groups.key = None;
        }
        cx.allocate_buffer(
            presented.scan,
            storage_bytes((capacity + 4 * (scan_groups + 1) + 1) * 4),
        );
        let batch_capacity = groups.batch_capacity as u64;
        cx.allocate_buffer(
            presented.batches,
            storage_bytes(batch_capacity * size_of::<BatchData>() as u64),
        );
        cx.allocate_buffer(presented.batch_bases, storage_bytes(batch_capacity * 16));
        let planes = u64::from(LOD_LEVELS);
        let args = (planes * batch_capacity + 1) * DRAW_ARGS_BYTES;
        let ids = planes * capacity * 4;
        let compactions = [true, kept.phase_two];
        for (index, wanted) in compactions.into_iter().enumerate() {
            cx.allocate_buffer(
                presented.visible[index],
                storage_bytes(if wanted { ids } else { 0 }),
            );
            cx.allocate_buffer(
                presented.draws[index],
                storage_bytes(if wanted { args } else { 0 }),
            );
        }
        for face in 0..FACES {
            let wanted = kept.probe_faces;
            cx.allocate_buffer(
                presented.face_visible[face],
                storage_bytes(if wanted { ids } else { 0 }),
            );
            cx.allocate_buffer(
                presented.face_draws[face],
                storage_bytes(if wanted { args } else { 0 }),
            );
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
        groups.kept = kept;
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
        cx.allocate_buffer(
            self.presented.lod_states,
            storage_bytes(state.fade_slots * 4),
        );
        cx.allocate_buffer(
            self.presented.fade_cards,
            storage_bytes(16 + state.card_capacity as u64 * 8),
        );
        cx.write(
            self.presented.fade_cards,
            0,
            &[engine_render::IMPOSTOR_VERTICES, 0, 0, 0],
        );
    }

    fn upload_nodes(&mut self, streams: &NodeStreams<'_>, cx: &mut Prepare<'_, Self>) -> u64 {
        let nodes = &mut self.state.nodes;
        let presented = &self.presented;
        let bytes = nodes.planned_bytes();
        let pose = size_of::<NodePose>();
        let previous = (nodes.capacity * pose) as u64;
        for range in &nodes.ranges {
            let (start, end) = (range.start, range.end);
            cx.write(
                presented.locals,
                (start * pose) as u64,
                &streams.current[start..end],
            );
            cx.write(
                presented.locals,
                previous + (start * pose) as u64,
                &streams.previous[start..end],
            );
            cx.write(
                presented.info,
                (start * size_of::<NodeInfo>()) as u64,
                &streams.info[start..end],
            );
            cx.write(
                presented.authority,
                (start * size_of::<PoseRows>()) as u64,
                &streams.authority[start..end],
            );
        }
        if !streams.parents.is_empty() {
            cx.write(presented.parents, 0, streams.parents);
        }
        nodes.levels(streams);
        for (level, &(start, end)) in nodes.level_ranges.iter().enumerate().take(NODE_LEVELS) {
            cx.write(presented.levels[level], 0, &[start, end, 0, 0]);
        }
        bytes
    }

    fn upload_words(
        &mut self,
        hierarchy: &GlobalTransforms,
        nodes: usize,
        cx: &mut Prepare<'_, Self>,
    ) {
        let inputs = &mut self.state.nodes;
        inputs.plan_words(hierarchy, nodes);
        let words = hierarchy.packed_words();
        for range in &inputs.word_ranges {
            cx.write(
                self.presented.node_words,
                (range.start * size_of::<u32>()) as u64,
                &words[range.start..range.end],
            );
        }
    }

    fn write_batches(
        &mut self,
        plan: &BatchPlan,
        casters: &CasterPlan,
        sources: &UploadSources<'_>,
        cx: &mut Prepare<'_, Self>,
    ) -> u64 {
        let groups = &mut self.state.groups;
        groups.stage_batches(
            &plan.batches,
            sources.assets,
            sources.residency,
            (
                |index| casters.batch_group(index),
                |index| {
                    sources
                        .impostors
                        .get(index)
                        .map_or(0.0, ImpostorRecord::error)
                },
            ),
        );
        if !groups.batch_staging.is_empty() {
            cx.write(self.presented.batches, 0, &groups.batch_staging);
        }
        (groups.batch_staging.len() * size_of::<BatchData>()) as u64
    }

    fn write_candidates(
        &mut self,
        retained: &RetainedRender,
        plan: &BatchPlan,
        hierarchy: &GlobalTransforms,
        cx: &mut Prepare<'_, Self>,
    ) -> u64 {
        let groups = &mut self.state.groups;
        let key = (plan.generation(), hierarchy.topology_revision().get());
        if groups.key == Some(key) {
            return 0;
        }
        let replanned = groups.key.map(|(generation, _)| generation) != Some(plan.generation());
        if replanned {
            let words = groups.capacity as u64 * 4;
            if words != 0 {
                cx.clear_buffer_range(self.presented.scan, 0..words);
            }
            let fades = storage_bytes(self.state.fade_slots * 4);
            cx.clear_buffer_range(self.presented.lod_states, 0..fades);
        }
        groups.key = Some(key);
        groups.stage_candidates(&plan.batches, &plan.ordered, |slot| {
            retained
                .owner(slot)
                .and_then(|owner| hierarchy.node_index(owner))
                .map_or(NO_NODE, |node| node as u32)
        });
        if !groups.candidate_staging.is_empty() {
            cx.write(self.presented.candidates, 0, &groups.candidate_staging);
            cx.write(self.presented.slot_candidates, 0, &groups.slot_staging);
        }
        (2 * groups.candidate_staging.len() * size_of::<CandidateData>()) as u64
    }

    fn write_frame_uniform(
        &mut self,
        alpha: f32,
        view: &View,
        decided: &MeshPlan,
        cx: &mut Prepare<'_, Self>,
    ) {
        let state = &mut self.state;
        let (planes, absolute) = frustum_uniform(view.view_projection);
        let bounded = absolute.iter().flatten().all(|value| {
            value.is_finite() && *value <= engine_core::presentation_bounds::MAX_COEFFICIENT
        }) && planes.iter().flatten().all(|value| value.is_finite());
        let culling = decided.culling && bounded;
        let candidates = state.groups.count;
        state.candidates = candidates;
        let unbudgeted = culling && decided.budget as usize >= candidates as usize;
        state.occlusion = view.occlusion.is_some() && unbudgeted && candidates != 0;
        let subpixel = decided.subpixel_enabled && unbudgeted && candidates != 0;
        let (width, height) = view.viewport;
        let mut uniform = PresentationFrame {
            alpha,
            candidates,
            budget: decided.budget,
            culling: u32::from(culling),
            groups: candidates.div_ceil(GROUP),
            batches: state.groups.batch_count,
            node_capacity: state.nodes.capacity as u32,
            pad_nodes: 0,
            planes,
            absolute_view_projection: absolute,
            motion: u32::from(view.motion),
            mask: if state.occlusion {
                CandidateState {
                    phase_one: true,
                    ..CandidateState::default()
                }
            } else {
                CandidateState {
                    visible: true,
                    ..CandidateState::default()
                }
            }
            .pack(),
            occlusion: u32::from(state.occlusion),
            subpixel: u32::from(subpixel),
            view_projection: view.view_projection.to_cols_array_2d(),
            viewport: [
                width as f32,
                height as f32,
                view.occlusion.unwrap_or(0) as f32,
                0.0,
            ],
            plane: state.groups.capacity as u32,
            lod_error: if unbudgeted { view.lod_error } else { 0.0 },
            watched: state.watched.unwrap_or(UNWATCHED),
            pad: 0,
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
            uniform.mask = CandidateState {
                phase_two: true,
                ..CandidateState::default()
            }
            .pack();
            cx.allocate_buffer(
                self.presented.frame_two,
                size_of::<PresentationFrame>() as u64,
            );
            cx.write(self.presented.frame_two, 0, &[uniform]);
        }
    }

    fn prepare_faces(
        &mut self,
        captures: &engine_places_rendering::probes::ProbeCaptures,
        cx: &mut Prepare<'_, Self>,
    ) {
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
            uniform.mask = CandidateState {
                probe: true,
                ..CandidateState::default()
            }
            .pack();
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
        maps: &engine_places_rendering::shadows::ShadowMaps,
        cx: &mut Prepare<'_, Self>,
    ) {
        let buffers = &self.casters;
        let draws = &mut self.state.casters;
        draws.maps.clone_from(&plan.maps);
        draws.runs = maps.on && !maps.regions.is_empty() && !plan.groups.is_empty();
        if !draws.runs {
            return;
        }
        cx.allocate_buffer(
            buffers.cache,
            storage_bytes(plan.cache_capacity as u64 * CACHE_BYTES),
        );
        let starts = (ID_CAPACITIES * plan.id_capacity * 4) as u64;
        cx.allocate_buffer(
            buffers.ids,
            storage_bytes(starts + (plan.arg_capacity * 4) as u64),
        );
        let args = (MAPS * plan.arg_capacity) as u64 * DRAW_ARGS_BYTES;
        cx.allocate_buffer(buffers.args, storage_bytes(args));
        cx.allocate_buffer(buffers.template, storage_bytes(args));
        cx.allocate_buffer(buffers.totals, storage_bytes(CASTER_TOTALS_BYTES));
        draws.groups.clear();
        draws.groups.extend(
            plan.groups
                .iter()
                .map(|group| (group.mesh, group.material, group.start)),
        );
        draws.regions_in_args = plan.regions_in_args;
        draws.id_capacity = plan.id_capacity;
        cx.write(buffers.template, 0, &plan.template);
        cx.write(buffers.ids, starts, &plan.starts);
        let candidates = self.state.groups.count;
        draws.candidates = candidates;
        let frame = CasterFrame {
            candidates,
            groups: plan.groups.len() as u32,
            capacity: plan.id_capacity as u32,
            pad: 0,
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
        cx.read_back::<CasterTotals>(
            CASTERS,
            buffers.totals,
            0..CASTER_TOTALS_BYTES,
            AfterRead::Clear,
        );
    }
}
