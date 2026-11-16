use bevy_ecs::system::SystemParamItem;
use engine_core::{
    AssetLibrary, CullPlan, DrawnTable, Error, ErrorCode, GlobalTransforms, GpuBits, Since,
    ViewDetail, Views,
};
use engine_render::{AfterRead, Allocation, GpuPart, Prepare, UploadRange, listed_ranges};

use crate::gpu::{Drawing, INSTANCE_BUFFER_ROW, INSTANCE_STAGING_ROW, MeshesGpu, VIEW_FRAME_BYTES};
use crate::layouts::{
    BatchData, CandidateData, CandidateState, DRAW_ARGS_BYTES, GROUP, LOD_LEVELS,
    PresentationFrame, UNWATCHED,
};
use crate::order::{capacity_for, instance_limit};
use crate::presentation::{
    Kept, PresentationTotals, TOTALS_BYTES, UploadSources, VIEWS, ViewSlot, WATCHED, grow,
    storage_bytes,
};
use crate::uploads::{ColorData, color_data, restyled_slots};

const TOTALS: u32 = 0;
const LIST_HEADER: u64 = 16;

struct View {
    detail: ViewDetail,
    occlusion: bool,
    watched: Option<u32>,
}

impl MeshesGpu {
    pub(crate) fn prepare_frame(
        &mut self,
        (
            (table, assets, plan, hierarchy, detail, limits),
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
        state
            .draws
            .plan(&state.batches.batches, &assets, &residency);
        let generation = state.batches.generation();
        let held = records
            .single()
            .ok()
            .filter(|held| held.generation == generation)
            .map_or(&[][..], |held| held.levels.as_slice());
        self.state.groups.first_instance = limits.indirect_first_instance;
        self.state.instance_count = self.state.batches.ordered.len();
        if self.state.capacity == 0 || slots > self.state.capacity {
            self.state.capacity = capacity_for(slots, limit);
        }
        self.prepare_instances(&table, &assets, cx);
        let culling = View {
            detail: measured(main.detail(&detail), main.view_projection),
            occlusion: plan.occlusion,
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
                last: held,
            },
            cx,
        );
        if engine_core::profile::active() {
            self.gpu_counters(group_bytes, totals.last().map(|last| last.visible));
        }
        self.prepare_views(&views, limits.storage_room, cx);
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
        let colors = cx.allocate_buffer(instances.colors, capacity * size_of::<ColorData>() as u64);
        if colors == Allocation::Fresh && state.colors_capacity != 0 {
            engine_core::profile::counter("render.instance_buffer_growth", 1.0);
        }
        state.colors_capacity = state.capacity;
        let records = table.records().len();
        let stride = size_of::<ColorData>();
        state.color_ranges.clear();
        let library = assets.revision();
        let before = state.colors_library.replace(library);
        let restyled = before != Some(library);
        let narrow = before
            .filter(|before| before.0 == library.0)
            .map(|(_, revision)| revision);
        match (colors, table.changes_since(&mut state.colors_seen)) {
            (Allocation::Fresh, _) | (Allocation::Kept, Since::All) => {
                if records != 0 {
                    state.color_ranges.push(UploadRange {
                        start: 0,
                        end: records,
                    });
                }
            }
            (Allocation::Kept, since) if restyled => match narrow {
                Some(revision) => {
                    state.color_slots.clear();
                    if let Since::Listed(slots) = since {
                        state.color_slots.extend_from_slice(slots);
                    }
                    restyled_slots(table, assets, revision, &mut state.color_slots);
                    listed_ranges(
                        &state.color_slots,
                        records,
                        &mut state.color_bits,
                        &mut state.color_ranges,
                    );
                }
                None => {
                    if records != 0 {
                        state.color_ranges.push(UploadRange {
                            start: 0,
                            end: records,
                        });
                    }
                }
            },
            (Allocation::Kept, Since::Listed(slots)) => {
                listed_ranges(
                    slots,
                    records,
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
            INSTANCE_STAGING_ROW,
            (state.color_staging.capacity() * size_of::<ColorData>()) as u64,
        );
        cx.memory(INSTANCE_BUFFER_ROW, capacity * color);
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
            last: view.detail.pixel_error > 0.0
                && sources.last.iter().any(|level| level.error > 0.0),
        };
        let batches = &self.state.batches;
        let counts = (batches.ordered.len(), batches.batches.len());
        self.ensure_groups(counts, kept, cx);
        self.ensure_fades(cx);
        self.state.watched = view.watched;
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
        let args = planes * batch_capacity * DRAW_ARGS_BYTES;
        let ids = planes * capacity * 4;
        let [two_ids, two_args, last_ids, two_last_ids] = [
            (kept.phase_two, ids),
            (kept.phase_two, args),
            (kept.last, LIST_HEADER + capacity * 8),
            (kept.last && kept.phase_two, LIST_HEADER + capacity * 8),
        ]
        .map(|(wanted, bytes)| storage_bytes(if wanted { bytes } else { 0 }));
        let batches = cx.allocate_buffer(
            presented.batches,
            storage_bytes(batch_capacity * size_of::<BatchData>() as u64),
        );
        if batches == Allocation::Fresh {
            groups.batch_sent.clear();
        }
        let rebuilt = [
            (presented.batch_bases, storage_bytes(batch_capacity * 16)),
            (presented.visible[0], storage_bytes(ids)),
            (presented.draws[0], storage_bytes(args)),
            (presented.visible[1], two_ids),
            (presented.draws[1], two_args),
            (presented.last_ids[0], last_ids),
            (presented.last_ids[1], two_last_ids),
            (presented.counts, storage_bytes(16)),
            (presented.totals, storage_bytes(TOTALS_BYTES)),
        ];
        for (buffer, bytes) in rebuilt {
            let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(buffer, bytes);
        }
        for buffer in presented.last_ids {
            cx.write(buffer, 0, &[0_u32; 4]);
        }
        groups.kept = kept;
    }

    fn ensure_fades(&mut self, cx: &mut Prepare<'_, Self>) {
        let state = &mut self.state;
        if state.groups.capacity > state.fading_capacity {
            state.fading_capacity = state.groups.capacity;
        }
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(
            self.presented.fading_ids,
            storage_bytes(LIST_HEADER + state.fading_capacity as u64 * 8),
        );
        cx.write(self.presented.fading_ids, 0, &[0_u32; 4]);
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
            |index| sources.last.get(index).copied().unwrap_or_default(),
            |material| {
                material
                    .and_then(|material| sources.assets.material(material))
                    .map_or(0.0, |material| table.reach(material))
            },
        );
        let sent: &[u8] = bytemuck::cast_slice(&groups.batch_sent);
        let staged: &[u8] = bytemuck::cast_slice(&groups.batch_staging);
        if staged.is_empty() || staged == sent {
            return 0;
        }
        cx.write(self.presented.batches, 0, &groups.batch_staging);
        groups.batch_sent.clone_from(&groups.batch_staging);
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
            lod_error: view.detail.pixel_error,
            watched: state.watched.unwrap_or(UNWATCHED),
            last_level: u32::from(state.groups.kept.last),
            pixel_angle: view.detail.pixel_angle,
            texel: view.detail.texel,
            eye: view.detail.eye.extend(0.0).to_array(),
            bases: [0; 4],
        };
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.presented.frame, size_of::<PresentationFrame>() as u64);
        cx.write(self.presented.frame, 0, &[uniform]);
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
        let arg_count = u64::from(LOD_LEVELS) * u64::from(state.groups.batch_count);
        let (mut ids, mut args, mut refused) = (0_u64, 0_u64, 0_u32);
        let mut frames = std::mem::take(&mut state.view_frames);
        frames.clear();
        for (global, view) in views.drawn().iter().enumerate() {
            let Some(pass) = view.owner else {
                continue;
            };
            let drawn = self
                .scenes
                .named(pass)
                .is_some_and(|drawing| matches!(drawing, Drawing::Views(_)));
            if !drawn || !view.draws {
                continue;
            }
            {
                let (Some(base), detailed) = (base, view.detail > 0.0) else {
                    continue;
                };
                let id_words = if detailed { u64::from(LOD_LEVELS) } else { 1 } * capacity;
                let over =
                    (ids + id_words) * 4 > room || (args + arg_count) * DRAW_ARGS_BYTES > room;
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
                let seen = measured(
                    ViewDetail::perspective(
                        view.eye,
                        view.view_projection,
                        view.size.y,
                        view.detail,
                    ),
                    view.view_projection,
                );
                uniform.lod_error = seen.pixel_error;
                uniform.watched = UNWATCHED;
                uniform.last_level = 0;
                uniform.pixel_angle = seen.pixel_angle;
                uniform.texel = seen.texel;
                uniform.eye = seen.eye.extend(0.0).to_array();
                uniform.bases = [ids as u32, args as u32, global as u32, 0];
                state.views.push(ViewSlot {
                    drawn: global as u32,
                    frame_at: (frames.len() as u64 * VIEW_FRAME_BYTES) as u32,
                    ids_base: ids,
                    args_base: args,
                    detailed,
                });
                frames.push(uniform);
                ids += id_words;
                args += arg_count;
            }
        }
        let counts = [state.views.len() as u32, refused];
        cx.deliver::<PresentationTotals>(VIEWS, bytemuck::cast_slice(&counts).to_vec());
        if !frames.is_empty() {
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
        self.state.view_frames = frames;
    }
}

fn measured(detail: ViewDetail, view_projection: glam::Mat4) -> ViewDetail {
    if view_projection.row(3).truncate().length() > 0.5 {
        detail
    } else {
        ViewDetail {
            eye: detail.eye,
            ..ViewDetail::orthographic(detail.pixel_angle, detail.pixel_error)
        }
    }
}
