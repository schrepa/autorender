use bevy_ecs::system::SystemParamItem;
use engine_core::{
    AssetLibrary, DrawnPoses, DrawnTable, Error, ErrorCode, Expected, GlobalTransforms, GpuBits,
    ImpostorRecord, NodeInfo, NodePose, NodeStreams, PoseRows, Since, Views,
};
use engine_render::{
    AfterRead, Allocation, CopyRegion, GpuPart, Prepare, TargetName, mip_levels, pyramid_extents,
    wgpu,
};

use crate::gpu::{Drawing, MeshesGpu, NODE_LEVELS, PYRAMID_LEVELS, VIEW_FRAME_BYTES};
use crate::layouts::{
    BatchData, BlendedView, COMPOSED_BYTES, CandidateData, CandidateState, DRAW_ARGS_BYTES, GROUP,
    LOD_LEVELS, NO_NODE, PresentationFrame, UNWATCHED,
};
use crate::order::{MeshPlan, capacity_for, instance_limit};
use crate::presentation::{
    FADE_MARGIN, Kept, POSE, PresentationTotals, TOTALS_BYTES, UploadSources, VIEWS, ViewSlot,
    WATCHED, frustum_uniform, grow, rasterized, storage_bytes,
};
use crate::uploads::{ColorData, PoseData, UploadRange, color_data, listed_ranges};

const TOTALS: u32 = 0;

struct View {
    view_projection: glam::Mat4,
    eye: glam::Vec3,
    viewport: (u32, u32),
    motion: bool,
    occlusion: Option<u32>,
    lod_error: f32,
    impostors: bool,
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
                table,
                assets,
                decided,
                presented,
                hierarchy,
                view,
                detail,
                impostors,
                time,
                limits,
            ),
            (records, views, watched, totals, residency),
        ): SystemParamItem<'_, '_, <Self as GpuPart>::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let slots = table.records().len();
        let limit = instance_limit(&limits);
        if slots > limit {
            cx.refuse(
                Error::new(ErrorCode::RenderFailed, "$.snapshot.instances")
                    .at_most(limit as u64)
                    .got(slots),
            );
            return;
        }
        if !view.view_projection().is_finite() {
            cx.refuse(
                Error::new(ErrorCode::RenderFailed, "$.snapshot.view_projection")
                    .expected(Expected::Finite),
            );
            return;
        }
        let state = &mut self.state;
        state.batches.settle(&table);
        state.draws.plan(&state.batches.batches, &assets, &residency);
        state.blended.plan(&state.draws.items, &state.batches);
        let generation = state.batches.generation();
        let held = records
            .single()
            .ok()
            .filter(|held| held.generation == generation)
            .map_or(&[][..], |held| held.records.as_slice());
        self.write_impostors(held, cx);
        self.state.groups.first_instance = limits.indirect_first_instance;
        self.state.instance_count = self.state.batches.ordered.len();
        if self.state.capacity == 0 || slots > self.state.capacity {
            self.state.capacity = capacity_for(slots, limit);
        }
        let kept = cx.read_target(TargetName::MOTION);
        let fresh = self.prepare_instances(&table, &assets, kept, cx);
        let size = view.render_size();
        let motion = self.prepare_motion(generation, (kept, view.history().valid(), fresh), cx);
        let shown = cx.ran(self.passes.presentation);
        let fade = self.fade_share(time.real_elapsed as f32, detail.fade_seconds, shown);
        if !decided.occlusion_enabled {
            self.release_pyramid(cx);
        }
        let culling = View {
            view_projection: rasterized(&view),
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
            fade,
            watched: watched.slot,
        };
        let nodes = presented.streams(&hierarchy);
        let prepared = self.prepare_presentation(
            &table,
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
        self.prepare_views(&views, limits.storage_room, cx);
        self.prepare_blended(cx);
    }

    fn prepare_blended(&mut self, cx: &mut Prepare<'_, Self>) {
        let blended = &mut self.state.blended;
        let bytes = storage_bytes((blended.rows.len() * size_of::<[u32; 4]>()) as u64);
        if cx.allocate_buffer(self.sorted.rows, bytes) == Allocation::Fresh {
            blended.uploaded = false;
        }
        if !blended.uploaded && !blended.rows.is_empty() {
            cx.write(self.sorted.rows, 0, &blended.rows);
            cx.count_upload(std::mem::size_of_val(blended.rows.as_slice()) as u64);
        }
        blended.uploaded = true;
    }

    pub(crate) fn prepare_blended_view(&mut self, cx: &mut Prepare<'_, Self>) {
        let rows = self.state.blended.rows.len() as u64;
        if rows == 0 || self.state.instance_count == 0 {
            return;
        }
        let batches = self.state.blended.batches.len() as u64;
        let sorted = &self.sorted;
        cx.allocate_buffer(sorted.view, size_of::<BlendedView>() as u64);
        cx.allocate_buffer(sorted.keys, storage_bytes(rows * 4));
        cx.allocate_buffer(sorted.ids, storage_bytes(rows * 4));
        cx.allocate_buffer(sorted.args, storage_bytes(batches * DRAW_ARGS_BYTES));
        let view = cx.view_index();
        cx.write(sorted.view, 0, &[view, rows as u32, batches as u32, 0]);
        cx.run(self.passes.blended);
    }

    fn prepare_motion(
        &mut self,
        generation: u64,
        (kept, history, fresh): (bool, bool, bool),
        cx: &mut Prepare<'_, Self>,
    ) -> bool {
        let shown = cx.ran(self.passes.presentation);
        let state = &mut self.state;
        let motion =
            kept && history && !fresh && shown && state.motion_generation == Some(generation);
        state.motion_generation = kept.then_some(generation);
        if kept {
            engine_core::profile::counter("render.object_motion", f64::from(u8::from(motion)));
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
        table: &DrawnTable,
        assets: &AssetLibrary,
        kept: bool,
        cx: &mut Prepare<'_, Self>,
    ) -> bool {
        let _profile = engine_core::profile::span("render", "instance_buffer_and_packing");
        let state = &mut self.state;
        let instances = &self.instances;
        let capacity = state.capacity as u64;
        let records_bytes = crate::debug::records_bytes();
        if cx.allocate_buffer(instances.records, records_bytes) == Allocation::Fresh {
            cx.write(instances.records, 0, &crate::debug::record_words());
        }
        let poses = cx.allocate_buffer(instances.poses, capacity * size_of::<PoseData>() as u64);
        if poses == Allocation::Fresh && state.poses_capacity != 0 {
            engine_core::profile::counter("render.instance_buffer_growth", 1.0);
        }
        state.poses_capacity = state.capacity;
        let colors =
            cx.allocate_buffer(instances.colors, capacity * size_of::<ColorData>() as u64);
        let previous_bytes = if kept {
            capacity * size_of::<PoseData>() as u64
        } else {
            storage_bytes(0)
        };
        let previous = cx.allocate_buffer(instances.previous_poses, previous_bytes);
        let records = table.records().len();
        let stride = size_of::<ColorData>();
        state.color_ranges.clear();
        let library = Some(assets.revision());
        let restyled = state.colors_library != library;
        state.colors_library = library;
        match (colors, table.changes_since(&mut state.colors_seen)) {
            (Allocation::Fresh, _) | (Allocation::Kept, Since::All) => {
                if records != 0 {
                    state.color_ranges.push(UploadRange {
                        start: 0,
                        end: records,
                    });
                }
            }
            (Allocation::Kept, _) if restyled => {
                if records != 0 {
                    state.color_ranges.push(UploadRange {
                        start: 0,
                        end: records,
                    });
                }
            }
            (Allocation::Kept, Since::Listed(slots)) => {
                listed_ranges(
                    slots,
                    records,
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
                .extend((range.start..range.end).map(|slot| color_data(table, assets, slot)));
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
            "memory.gpu_previous_pose_bytes",
            if kept { capacity * pose } else { 0 },
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

    fn prepare_presentation(
        &mut self,
        table: &DrawnTable,
        (poses, streams): (&DrawnPoses, &NodeStreams<'_>),
        view: &View,
        decided: &MeshPlan,
        sources: UploadSources<'_>,
        cx: &mut Prepare<'_, Self>,
    ) -> Prepared {
        self.ensure_nodes(streams.info.len(), streams.parents.len(), cx);
        let kept = Kept {
            phase_two: view.occlusion.is_some(),
            impostors: view.impostors && view.lod_error > 0.0,
        };
        let batches = &self.state.batches;
        let counts = (batches.ordered.len(), batches.batches.len());
        self.ensure_groups(counts, kept, cx);
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
        let group_bytes = self.write_batches(&sources, table, cx)
            + self.write_candidates(table, sources.hierarchy, cx);
        self.write_frame_uniform(streams.alpha, view, decided, cx);
        cx.run(self.passes.presentation);
        if self.state.occlusion {
            cx.run(self.passes.occlusion);
            cx.run(self.passes.occluded);
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

    fn write_batches(
        &mut self,
        sources: &UploadSources<'_>,
        table: &DrawnTable,
        cx: &mut Prepare<'_, Self>,
    ) -> u64 {
        let state = &mut self.state;
        let groups = &mut state.groups;
        groups.stage_batches(
            &state.batches.batches,
            (sources.assets, sources.residency),
            |index| {
                sources
                    .impostors
                    .get(index)
                    .map_or(0.0, ImpostorRecord::error)
            },
            |material| {
                material
                    .and_then(|material| sources.assets.material(material))
                    .map_or(0.0, |material| table.reach(material))
            },
        );
        if !groups.batch_staging.is_empty() {
            cx.write(self.presented.batches, 0, &groups.batch_staging);
        }
        (groups.batch_staging.len() * size_of::<BatchData>()) as u64
    }

    fn write_candidates(
        &mut self,
        table: &DrawnTable,
        hierarchy: &GlobalTransforms,
        cx: &mut Prepare<'_, Self>,
    ) -> u64 {
        let state = &mut self.state;
        let groups = &mut state.groups;
        let generation = state.batches.generation();
        let key = (generation, hierarchy.topology_revision().get());
        if groups.key == Some(key) {
            return 0;
        }
        let replanned = groups.key.map(|(generation, _)| generation) != Some(generation);
        if replanned {
            let words = groups.capacity as u64 * 4;
            if words != 0 {
                cx.clear_buffer_range(self.presented.scan, 0..words);
            }
            let fades = storage_bytes(state.fade_slots * 4);
            cx.clear_buffer_range(self.presented.lod_states, 0..fades);
        }
        groups.key = Some(key);
        let records = table.records();
        groups.stage_candidates(&state.batches.batches, &state.batches.ordered, |slot| {
            records.get(slot).map_or(NO_NODE, |record| record.node)
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
            bases: [0; 4],
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

    fn prepare_views(&mut self, views: &Views, room: u64, cx: &mut Prepare<'_, Self>) {
        let state = &mut self.state;
        state.views.clear();
        let base = state.frame.filter(|_| state.candidates != 0);
        let capacity = state.groups.capacity as u64;
        let arg_count = u64::from(LOD_LEVELS) * u64::from(state.groups.batch_count) + 1;
        let (mut ids, mut args, mut refused) = (0_u64, 0_u64, 0_u32);
        let mut frames: Vec<PresentationFrame> = Vec::new();
        for (global, view) in views.drawn().iter().enumerate() {
            let Some(pass) = view.owner else {
                continue;
            };
            let drawn = self.scenes.iter().filter(|scene| scene.name == pass).any(|scene| {
                matches!(scene.drawing, Drawing::Views { .. })
            });
            if !drawn || !view.draws {
                continue;
            }
            let index = view.local;
            {
                let (Some(base), detailed) = (base, view.detail > 0.0) else {
                    continue;
                };
                let id_words = if detailed { u64::from(LOD_LEVELS) } else { 1 } * capacity;
                let over = (ids + id_words) * 4 > room || (args + arg_count) * DRAW_ARGS_BYTES > room;
                if room != 0 && over {
                    refused += 1;
                    continue;
                }
                let view_projection = view.view_projection;
                let (planes, absolute) = frustum_uniform(view_projection);
                let mut uniform = base;
                uniform.planes = view.planes.unwrap_or(planes);
                uniform.absolute_view_projection = absolute;
                uniform.view_projection = view_projection.to_cols_array_2d();
                uniform.culling = 1;
                uniform.mask = CandidateState {
                    view: true,
                    ..CandidateState::default()
                }
                .pack();
                uniform.occlusion = 0;
                uniform.subpixel = 1;
                uniform.viewport = [view.size.x as f32, view.size.y as f32, 0.0, 0.0];
                uniform.motion = 0;
                uniform.budget = u32::MAX;
                uniform.lod_error = view.detail;
                uniform.watched = UNWATCHED;
                uniform.eye = view.eye.extend(0.0).to_array();
                uniform.bases = [ids as u32, args as u32, global as u32, 0];
                state.views.push(ViewSlot {
                    pass,
                    index,
                    frame_at: (frames.len() as u64 * VIEW_FRAME_BYTES) as u32,
                    ids_base: ids,
                    args_base: args,
                    detailed,
                    mirrored: view_projection.determinant() < 0.0,
                });
                frames.push(uniform);
                ids += id_words;
                args += arg_count;
            }
        }
        let counts = [state.views.len() as u32, refused];
        cx.deliver::<PresentationTotals>(VIEWS, bytemuck::cast_slice(&counts).to_vec());
        if frames.is_empty() {
            return;
        }
        let presented = &self.presented;
        cx.allocate_buffer(
            presented.view_frames,
            frames.len() as u64 * VIEW_FRAME_BYTES,
        );
        for (at, frame) in frames.iter().enumerate() {
            cx.write(
                presented.view_frames,
                at as u64 * VIEW_FRAME_BYTES,
                std::slice::from_ref(frame),
            );
        }
        cx.allocate_buffer(presented.view_visible, storage_bytes(ids * 4));
        cx.allocate_buffer(
            presented.view_draws,
            storage_bytes(args * DRAW_ARGS_BYTES),
        );
        cx.run(self.passes.views);
    }
}
