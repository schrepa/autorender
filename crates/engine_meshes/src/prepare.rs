use bevy_ecs::system::SystemParamItem;
use engine_core::{
    AssetLibrary, CullPlan, DrawnTable, Error, ErrorCode, GlobalTransforms, GpuBits, Since, Views,
};
use engine_places::impostors::{IMPOSTOR_VERTICES, ImpostorRecord};
use engine_render::{AfterRead, Allocation, GpuPart, Prepare, UploadRange, listed_ranges};

use crate::gpu::{Drawing, MeshesGpu, VIEW_FRAME_BYTES};
use crate::layouts::{
    BatchData, BlendedView, CandidateData, CandidateState, DRAW_ARGS_BYTES, GROUP,
    LOD_LEVELS, PresentationFrame, UNWATCHED,
};
use crate::order::{capacity_for, instance_limit};
use crate::presentation::{
    FADE_MARGIN, Kept, PresentationTotals, TOTALS_BYTES, UploadSources, VIEWS, ViewSlot,
    WATCHED, grow, storage_bytes,
};
use crate::uploads::{ColorData, color_data};

const TOTALS: u32 = 0;

struct View {
    eye: glam::Vec3,
    occlusion: bool,
    lod_error: f32,
    impostors: bool,
    fade: f32,
    watched: Option<u32>,
}

impl MeshesGpu {
    pub(crate) fn prepare_frame(
        &mut self,
        (
            (
                table,
                assets,
                plan,
                hierarchy,
                detail,
                impostors,
                time,
                limits,
            ),
            (records, views, watched, totals, residency),
        ): SystemParamItem<'_, '_, <Self as GpuPart>::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let Some(main) = views.main() else {
            return;
        };
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
        self.prepare_instances(&table, &assets, cx);
        let shown = cx.ran(self.passes.presentation);
        let fade = self.fade_share(time.real_elapsed as f32, detail.fade_seconds, shown);
        let culling = View {
            eye: main.detail_eye,
            occlusion: plan.occlusion,
            lod_error: if detail.enabled {
                detail.pixel_error * main.render.y as f32 / main.size().y.max(1) as f32
            } else {
                0.0
            },
            impostors: impostors.enabled,
            fade,
            watched: watched.slot,
        };
        let group_bytes = self.prepare_presentation(
            &table,
            &culling,
            &plan,
            UploadSources {
                assets: &assets,
                residency: &residency,
                hierarchy: &hierarchy,
                impostors: held,
            },
            cx,
        );
        if engine_core::profile::active() {
            self.gpu_counters(group_bytes, totals.last().map(|last| last.visible));
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
        let view = cx.view_index();
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(sorted.view, size_of::<BlendedView>() as u64);
        cx.write(sorted.view, 0, &[view, rows as u32, batches as u32, 0]);
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(sorted.keys, storage_bytes(rows * 4));
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(sorted.ids, storage_bytes(rows * 4));
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(sorted.args, storage_bytes(batches * DRAW_ARGS_BYTES));
        cx.run(self.passes.blended);
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
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.instances.impostors, bytes);
        if !records.is_empty() {
            cx.write(self.instances.impostors, 0, records);
        }
    }

    fn prepare_instances(
        &mut self,
        table: &DrawnTable,
        assets: &AssetLibrary,
        cx: &mut Prepare<'_, Self>,
    ) {
        let _profile = engine_core::profile::span("render", "instance_buffer_and_packing");
        let state = &mut self.state;
        let instances = &self.instances;
        let capacity = state.capacity as u64;
        let records_bytes = crate::debug::records_bytes();
        if cx.allocate_buffer(instances.records, records_bytes) == Allocation::Fresh {
            cx.write(instances.records, 0, &crate::debug::record_words());
        }
        let colors =
            cx.allocate_buffer(instances.colors, capacity * size_of::<ColorData>() as u64);
        if colors == Allocation::Fresh && state.colors_capacity != 0 {
            engine_core::profile::counter("render.instance_buffer_growth", 1.0);
        }
        state.colors_capacity = state.capacity;
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
        let color = size_of::<ColorData>() as u64;
        cx.memory(
            "memory.render_staging_capacity_bytes",
            (state.color_staging.capacity() * size_of::<ColorData>()) as u64,
        );
        cx.memory("memory.gpu_instance_buffer_bytes", capacity * color);
    }

    fn prepare_presentation(
        &mut self,
        table: &DrawnTable,
        view: &View,
        plan: &CullPlan,
        sources: UploadSources<'_>,
        cx: &mut Prepare<'_, Self>,
    ) -> u64 {
        let kept = Kept {
            phase_two: view.occlusion,
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
        let group_bytes =
            self.write_batches(&sources, table, cx) + self.write_candidates(sources.hierarchy, cx);
        self.write_frame_uniform(view, plan, cx);
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
        }
        group_bytes
    }

    fn gpu_counters(&self, group_bytes: u64, visible: Option<u32>) {
        use engine_core::profile::counter;
        counter("render.gpu_candidates", f64::from(self.state.groups.count));
        counter("render.gpu_group_upload_bytes", group_bytes as f64);
        if let Some(visible) = visible {
            counter("render.gpu_visible_readback", f64::from(visible));
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
            cx.allocate_buffer(
                presented.scan,
                storage_bytes((capacity + 4 * (scan_groups + 1) + 1) * 4),
            ),
        ];
        if staged.contains(&Allocation::Fresh) {
            groups.key = None;
        }
        let batch_capacity = groups.batch_capacity as u64;
        let planes = u64::from(LOD_LEVELS);
        let args = (planes * batch_capacity + 1) * DRAW_ARGS_BYTES;
        let ids = planes * capacity * 4;
        let [two_ids, two_args, impostor_ids, two_impostor_ids] = [
            (kept.phase_two, ids),
            (kept.phase_two, args),
            (kept.impostors, capacity * 8),
            (kept.impostors && kept.phase_two, capacity * 8),
        ]
        .map(|(wanted, bytes)| storage_bytes(if wanted { bytes } else { 0 }));
        let rebuilt = [
            (
                presented.batches,
                storage_bytes(batch_capacity * size_of::<BatchData>() as u64),
            ),
            (presented.batch_bases, storage_bytes(batch_capacity * 16)),
            (presented.visible[0], storage_bytes(ids)),
            (presented.draws[0], storage_bytes(args)),
            (presented.visible[1], two_ids),
            (presented.draws[1], two_args),
            (presented.impostor_ids[0], impostor_ids),
            (presented.impostor_ids[1], two_impostor_ids),
            (presented.counts, storage_bytes(16)),
            (presented.totals, storage_bytes(TOTALS_BYTES)),
        ];
        for (buffer, bytes) in rebuilt {
            let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(buffer, bytes);
        }
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
        let states = cx.allocate_buffer(
            self.presented.lod_states,
            storage_bytes(state.fade_slots * 4),
        );
        if states == Allocation::Fresh {
            state.groups.key = None;
        }
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(
            self.presented.fade_cards,
            storage_bytes(16 + state.card_capacity as u64 * 8),
        );
        cx.write(self.presented.fade_cards, 0, &[IMPOSTOR_VERTICES, 0, 0, 0]);
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
        groups.stage_candidates(&state.batches.batches, &state.batches.ordered);
        if !groups.candidate_staging.is_empty() {
            cx.write(self.presented.candidates, 0, &groups.candidate_staging);
            cx.write(self.presented.slot_candidates, 0, &groups.slot_staging);
        }
        (2 * groups.candidate_staging.len() * size_of::<CandidateData>()) as u64
    }

    fn write_frame_uniform(&mut self, view: &View, plan: &CullPlan, cx: &mut Prepare<'_, Self>) {
        let state = &mut self.state;
        let candidates = state.groups.count;
        state.candidates = candidates;
        let unbudgeted = plan.frustum && plan.budget >= candidates;
        state.occlusion = view.occlusion && candidates != 0;
        let mut uniform = PresentationFrame {
            candidates,
            budget: plan.budget,
            groups: candidates.div_ceil(GROUP),
            batches: state.groups.batch_count,
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
            plane: state.groups.capacity as u32,
            lod_error: if unbudgeted { view.lod_error } else { 0.0 },
            watched: state.watched.unwrap_or(UNWATCHED),
            pad_0: 0,
            pad_1: 0,
            pad_2: 0,
            eye: view
                .eye
                .extend(if view.impostors { 1.0 } else { 0.0 })
                .to_array(),
            fade: [view.fade, FADE_MARGIN, 0.0, 0.0],
            bases: [0; 4],
        };
        let (Allocation::Fresh | Allocation::Kept) =
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
            let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(
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
                let mut uniform = base;
                uniform.mask = CandidateState {
                    view: true,
                    ..CandidateState::default()
                }
                .pack();
                uniform.occlusion = 0;
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
                    mirrored: view.view_projection.determinant() < 0.0,
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
        let rebuilt = [
            (
                presented.view_frames,
                frames.len() as u64 * VIEW_FRAME_BYTES,
            ),
            (presented.view_visible, storage_bytes(ids * 4)),
            (presented.view_draws, storage_bytes(args * DRAW_ARGS_BYTES)),
        ];
        for (buffer, bytes) in rebuilt {
            let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(buffer, bytes);
        }
        for (at, frame) in frames.iter().enumerate() {
            cx.write(
                presented.view_frames,
                at as u64 * VIEW_FRAME_BYTES,
                std::slice::from_ref(frame),
            );
        }
        cx.run(self.passes.views);
    }
}
