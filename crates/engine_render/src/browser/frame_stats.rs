//! A frame's [`RenderStats`] and, while profiling, its counters.

use super::frame::{EncodeReport, FrameKind, FrameState};
use super::*;

impl Renderer {
    /// The frame's stats, after submission (read-backs arrive a few frames
    /// late), and its profile counters while a capture runs.
    pub(super) fn frame_stats(
        &mut self,
        state: &FrameState<'_>,
        report: &EncodeReport,
    ) -> RenderStats {
        let FrameKind {
            gpu_mode,
            instance_count,
            ..
        } = state.kind;
        let (snapshot, retained) = (state.snapshot, state.retained);
        let (virtual_frame, resolving) = (state.virtual_frame, state.resolving);
        let (shadow_draws, probe_faces) = (report.shadow_draws, report.probe_faces);
        let features = &state.features;
        let (shadow_frame, light_frame, post_frame) = (
            &features.shadow_frame,
            &features.light_frame,
            features.post_frame,
        );
        let (particle_frame, decal_frame, refraction) = (
            &features.particle_frame,
            &features.decal_frame,
            features.refraction,
        );
        let dropped_probes = features.dropped_probes;
        let (mesh_upload_bytes, texture_upload_bytes) =
            (state.assets.mesh_bytes, state.assets.texture_bytes);
        let (camera_upload_bytes, instance_upload_bytes) =
            (state.writes.camera_bytes, state.writes.instance_bytes);

        let gpu_visible = if gpu_mode {
            self.gpu_presentation.visible()
        } else {
            None
        };
        // Read back a few frames late (decision 044).
        let virtual_stats = virtual_frame
            .then(|| self.virtual_geometry.stats())
            .flatten();
        let pipelines_created = crate::pipelines::take_created();
        self.pipelines = self.pipelines.saturating_add(pipelines_created);
        if pipelines_created > 0 {
            // Pipelines compile before the work submitted after them runs:
            // this frame's completion ends their compiles (decision 049).
            self.pipeline_frames += 1;
            let (compiled, frame) = (Arc::clone(&self.pipelines_compiled), self.pipeline_frames);
            self.queue.on_submitted_work_done(move || {
                compiled.fetch_max(frame, Ordering::Relaxed);
            });
        }
        let stats = RenderStats {
            draw_calls: self.plan.batches.len() as u32,
            upload_bytes: camera_upload_bytes
                + instance_upload_bytes
                + mesh_upload_bytes
                + texture_upload_bytes,
            // GPU-culled frames report the latest read-back visible count
            // within the budget the GPU applied.
            instances: match (retained, gpu_visible) {
                (Some(frame), Some(visible)) if gpu_mode => {
                    visible.min(frame.instance_budget().min(u32::MAX as usize) as u32)
                }
                (Some(frame), None) if gpu_mode => {
                    instance_count.min(frame.instance_budget()) as u32
                }
                _ => instance_count as u32,
            },
            gpu_visible,
            occluded_instances: if gpu_mode && self.occlusion_culling {
                self.gpu_presentation.occluded()
            } else {
                None
            },
            subpixel_instances: if gpu_mode && self.subpixel_culling {
                self.gpu_presentation.subpixel()
            } else {
                None
            },
            lod_instances: if gpu_mode && snapshot.level_of_detail.enabled {
                self.gpu_presentation.lod_instances()
            } else {
                None
            },
            triangles: if gpu_mode {
                self.gpu_presentation.triangles()
            } else {
                // CPU-prepared frames draw every planned instance as its
                // mesh, but virtual geometry as its clusters.
                let triangles: u64 = self
                    .plan
                    .batches
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| self.virtual_drawn.get(*index) != Some(&true))
                    .map(|(_, batch)| {
                        u64::from(batch.instances.end - batch.instances.start)
                            * u64::from(
                                self.meshes
                                    .get(&batch.mesh)
                                    .map_or(0, |mesh| mesh.index_count)
                                    / 3,
                            )
                    })
                    .sum();
                Some(triangles.min(u64::from(u32::MAX)) as u32)
            },
            mesh_upload_bytes,
            texture_upload_bytes,
            submitted: true,
            shadows: shadow_frame.as_ref().map(|frame| ShadowStats {
                draws: shadow_draws,
                kept_redraw: frame.refresh_kind(),
                casters: self.shadows.caster_counts(),
            }),
            ambient_occlusion: post_frame.is_some_and(|frame| frame.occlusion),
            bloom: post_frame.is_some_and(|frame| frame.bloom),
            lights: light_frame.as_ref().map(|_| {
                let totals = self.lights.as_ref().and_then(LightPass::totals);
                LightStats {
                    lights: snapshot.lights.len() as u32,
                    dropped: snapshot.dropped_lights.min(u32::MAX as usize) as u32,
                    visible: totals.map(|totals| totals[0]),
                    overflow: totals.map(|totals| totals[1]),
                }
            }),
            transparent_draws: if instance_count > 0 {
                self.plan
                    .batches
                    .iter()
                    .filter(|batch| batch.transparent)
                    .count() as u32
            } else {
                0
            },
            particles: particle_frame.as_ref().map_or(0, |frame| frame.total),
            dropped_particles: (snapshot.dropped_particles as u64
                + particle_frame.as_ref().map_or(0, |frame| frame.dropped))
            .min(u64::from(u32::MAX)) as u32,
            sun_shafts: post_frame.is_some_and(|frame| frame.shafts),
            reflections: post_frame.is_some_and(|frame| frame.reflections),
            refraction: refraction && post_frame.is_some(),
            visibility_buffer: resolving,
            decals: decal_frame.as_ref().map_or(0, |frame| frame.drawn),
            dropped_decals: (snapshot.dropped_decals as u64
                + decal_frame
                    .as_ref()
                    .map_or(0, |frame| u64::from(frame.dropped)))
            .min(u64::from(u32::MAX)) as u32,
            impostors: self.impostors.live(),
            dropped_impostors: self.dropped_impostors,
            reflection_probes: self.probes.slots.ready(),
            reflection_probe_faces: probe_faces,
            dropped_reflection_probes: dropped_probes.min(u64::from(u32::MAX)) as u32,
            pipelines: self.pipelines,
            pipelines_created,
            virtual_clusters: virtual_stats.map_or(0, |stats| stats.clusters),
            virtual_triangles: virtual_stats.map_or(0, |stats| stats.triangles),
            virtual_shadow_clusters: virtual_stats.map_or(0, |stats| stats.shadow_clusters),
            virtual_shadow_triangles: virtual_stats.map_or(0, |stats| stats.shadow_triangles),
            virtual_occluded_clusters: virtual_stats.map_or(0, |stats| stats.occluded),
            virtual_resident_pages: virtual_stats.map_or(0, |stats| stats.resident_pages),
            virtual_loading_pages: virtual_stats.map_or(0, |stats| stats.loading_pages),
            virtual_missing_pages: virtual_stats.map_or(0, |stats| stats.missing_pages),
            dropped_virtual_pages: virtual_stats.map_or(0, |stats| stats.dropped_pages),
            dropped_virtual: virtual_stats.map_or(0, |stats| stats.dropped),
            ..self.residency_stats()
        };
        self.profile_counters(state, report, &stats, gpu_visible, virtual_stats);
        stats
    }

    /// The frame's profile counters while a capture runs.
    #[allow(clippy::too_many_arguments)]
    fn profile_counters(
        &self,
        state: &FrameState<'_>,
        report: &EncodeReport,
        stats: &RenderStats,
        gpu_visible: Option<u32>,
        virtual_stats: Option<crate::virtual_geometry::VirtualStats>,
    ) {
        let FrameKind {
            affine,
            gpu_mode,
            instance_count,
            ..
        } = state.kind;
        let shadow_draws = report.shadow_draws;
        let features = &state.features;
        let (shadow_frame, post_frame, gpu_prepared) = (
            &features.shadow_frame,
            features.post_frame,
            &features.gpu_prepared,
        );
        let temporal_frame = &state.packing.temporal_frame;
        let (mesh_upload_bytes, texture_upload_bytes, mip_levels) = (
            state.assets.mesh_bytes,
            state.assets.texture_bytes,
            state.assets.mip_levels,
        );
        let writes = &state.writes;
        let (camera_upload_bytes, instance_upload_bytes) =
            (writes.camera_bytes, writes.instance_bytes);
        let (pose_upload_bytes, color_upload_bytes, slot_upload_bytes) =
            (writes.pose_bytes, writes.color_bytes, writes.slot_bytes);
        let instance_write_ranges = writes.ranges;
        let packing = &state.packing;
        let (repacked, poses_repacked, colors_repacked) = (
            packing.repacked,
            packing.poses_repacked,
            packing.colors_repacked,
        );
        let (full_upload, fallback) = (packing.full_upload, packing.fallback);

        if engine_profile::active() {
            engine_profile::counter("render.instance_capacity", self.instance_capacity as f64);
            engine_profile::counter("render.instances", f64::from(stats.instances));
            engine_profile::counter("render.draw_calls", self.plan.batches.len() as f64);
            engine_profile::counter("render.mesh_upload_bytes", mesh_upload_bytes as f64);
            engine_profile::counter("render.texture_upload_bytes", texture_upload_bytes as f64);
            engine_profile::counter("render.texture_mip_levels", f64::from(mip_levels));
            engine_profile::counter("render.camera_write_bytes", camera_upload_bytes as f64);
            engine_profile::counter("render.instance_write_bytes", instance_upload_bytes as f64);
            engine_profile::counter("render.pose_write_bytes", pose_upload_bytes as f64);
            engine_profile::counter("render.color_write_bytes", color_upload_bytes as f64);
            engine_profile::counter("render.instance_id_write_bytes", slot_upload_bytes as f64);
            engine_profile::counter("render.retained_frame", f64::from(u8::from(affine)));
            engine_profile::counter("render.gpu_presentation", f64::from(u8::from(gpu_mode)));
            engine_profile::counter("render.taa", f64::from(u8::from(temporal_frame.is_some())));
            engine_profile::counter(
                "render.shadow_refresh",
                shadow_frame
                    .as_ref()
                    .map_or(0.0, |frame| f64::from(frame.refresh_kind())),
            );
            engine_profile::counter("render.shadow_groups", self.shadows.groups().len() as f64);
            engine_profile::counter("render.shadow_draws", f64::from(shadow_draws));
            if let Some([near, moving, kept]) = self.shadows.caster_counts() {
                engine_profile::counter("render.shadow_near_casters", f64::from(near));
                engine_profile::counter("render.shadow_moving_casters", f64::from(moving));
                engine_profile::counter("render.shadow_kept_casters", f64::from(kept));
            }
            engine_profile::counter(
                "render.ambient_occlusion",
                f64::from(u8::from(post_frame.is_some_and(|frame| frame.occlusion))),
            );
            engine_profile::counter(
                "render.bloom",
                f64::from(u8::from(post_frame.is_some_and(|frame| frame.bloom))),
            );
            engine_profile::counter(
                "memory.gpu_shadow_bytes",
                self.shadows.capacity_bytes() as f64,
            );
            engine_profile::counter(
                "memory.gpu_post_bytes",
                self.post.as_ref().map_or(0, PostPass::capacity_bytes) as f64,
            );
            engine_profile::counter(
                "memory.gpu_environment_bytes",
                (self.environment.capacity_bytes() + self.probes.capacity_bytes()) as f64,
            );
            engine_profile::counter(
                "memory.gpu_virtual_bytes",
                (self.virtual_geometry.capacity_bytes()
                    + self
                        .visibility
                        .as_ref()
                        .map_or(0, VisibilityBuffer::capacity_bytes)) as f64,
            );
            if let Some(virtual_stats) = virtual_stats {
                engine_profile::counter(
                    "render.virtual_clusters",
                    f64::from(virtual_stats.clusters),
                );
                engine_profile::counter(
                    "render.virtual_triangles",
                    f64::from(virtual_stats.triangles),
                );
                engine_profile::counter(
                    "render.virtual_shadow_triangles",
                    f64::from(virtual_stats.shadow_triangles),
                );
                engine_profile::counter(
                    "render.virtual_occluded_clusters",
                    f64::from(virtual_stats.occluded),
                );
                engine_profile::counter(
                    "render.virtual_resident_pages",
                    f64::from(virtual_stats.resident_pages),
                );
                engine_profile::counter(
                    "render.virtual_loading_pages",
                    f64::from(virtual_stats.loading_pages),
                );
                engine_profile::counter(
                    "render.virtual_missing_pages",
                    f64::from(virtual_stats.missing_pages),
                );
                engine_profile::counter(
                    "render.dropped_virtual_pages",
                    f64::from(virtual_stats.dropped_pages),
                );
                engine_profile::counter("render.dropped_virtual", f64::from(virtual_stats.dropped));
            }
            if let Some(lights) = &stats.lights {
                engine_profile::counter("render.local_lights", f64::from(lights.lights));
                if let Some([visible, overflow]) = self.lights.as_ref().and_then(LightPass::totals)
                {
                    engine_profile::counter("render.lights_visible", f64::from(visible));
                    engine_profile::counter("render.light_cluster_overflow", f64::from(overflow));
                }
            }
            engine_profile::counter(
                "memory.gpu_light_bytes",
                self.lights.as_ref().map_or(0, LightPass::capacity_bytes) as f64,
            );
            engine_profile::counter(
                "render.transparent_draws",
                f64::from(stats.transparent_draws),
            );
            engine_profile::counter("render.particles", f64::from(stats.particles));
            engine_profile::counter("render.decals", f64::from(stats.decals));
            engine_profile::counter("render.impostors", f64::from(stats.impostors));
            engine_profile::counter(
                "render.dropped_impostors",
                f64::from(stats.dropped_impostors),
            );
            engine_profile::counter(
                "render.pipelines_created",
                f64::from(stats.pipelines_created),
            );
            engine_profile::counter(
                "memory.gpu_impostor_bytes",
                self.impostors.capacity_bytes() as f64,
            );
            if let Some(occluded) = stats.occluded_instances {
                engine_profile::counter("render.occluded_instances", f64::from(occluded));
            }
            if let Some(subpixel) = stats.subpixel_instances {
                engine_profile::counter("render.subpixel_instances", f64::from(subpixel));
            }
            if let Some(triangles) = stats.triangles {
                engine_profile::counter("render.triangles", f64::from(triangles));
            }
            if let Some(levels) = stats.lod_instances {
                for (name, count) in [
                    "render.lod0_instances",
                    "render.lod1_instances",
                    "render.lod2_instances",
                    "render.impostor_instances",
                ]
                .into_iter()
                .zip(levels)
                {
                    engine_profile::counter(name, f64::from(count));
                }
            }
            engine_profile::counter(
                "memory.gpu_transparent_bytes",
                (self
                    .transparent
                    .as_ref()
                    .map_or(0, TransparentPass::capacity_bytes)
                    + self
                        .particles
                        .as_ref()
                        .map_or(0, ParticlePass::capacity_bytes)
                    + self.decals.as_ref().map_or(0, DecalPass::capacity_bytes))
                    as f64,
            );
            if let Some(frame) = &temporal_frame {
                engine_profile::counter(
                    "render.taa_history",
                    f64::from(u8::from(frame.history_valid)),
                );
                engine_profile::counter(
                    "render.taa_object_motion",
                    f64::from(u8::from(frame.motion_valid)),
                );
            }
            engine_profile::counter(
                "memory.gpu_temporal_bytes",
                (self
                    .temporal
                    .as_ref()
                    .map_or(0, TemporalPass::capacity_bytes)
                    + self.alternate_poses.as_ref().map_or(0, wgpu::Buffer::size))
                    as f64,
            );
            if let Some(prepared) = &gpu_prepared {
                engine_profile::counter("render.gpu_candidates", f64::from(prepared.candidates));
                engine_profile::counter(
                    "render.gpu_node_upload_bytes",
                    prepared.node_upload_bytes as f64,
                );
                engine_profile::counter(
                    "render.gpu_group_upload_bytes",
                    prepared.group_upload_bytes as f64,
                );
                engine_profile::counter(
                    "render.gpu_full_node_upload",
                    f64::from(u8::from(prepared.full_node_upload)),
                );
                engine_profile::counter(
                    "render.gpu_node_upload_staged",
                    f64::from(u8::from(prepared.staged)),
                );
                engine_profile::counter("render.gpu_parent_levels", prepared.parent_levels as f64);
                engine_profile::counter("render.gpu_shift", f64::from(u8::from(prepared.shifted)));
                engine_profile::counter("render.gpu_blend", f64::from(u8::from(prepared.blend)));
                engine_profile::counter(
                    "render.gpu_current_prestaged",
                    f64::from(u8::from(prepared.prestaged)),
                );
                engine_profile::counter("render.gpu_held", f64::from(u8::from(prepared.held)));
                if let Some(visible) = gpu_visible {
                    engine_profile::counter("render.gpu_visible_readback", f64::from(visible));
                }
            }
            engine_profile::counter(
                "memory.gpu_presentation_bytes",
                (self.gpu_presentation.capacity_bytes()
                    + self
                        .depth_pyramid
                        .as_ref()
                        .map_or(0, DepthPyramid::capacity_bytes)) as f64,
            );
            engine_profile::counter("render.instance_write_ranges", instance_write_ranges as f64);
            engine_profile::counter(
                "render.instance_full_upload",
                f64::from(u8::from(full_upload)),
            );
            engine_profile::counter(
                "render.instance_upload_fallback",
                f64::from(u8::from(fallback)),
            );
            engine_profile::counter(
                "render.camera_upload_reused",
                f64::from(u8::from(camera_upload_bytes == 0)),
            );
            engine_profile::counter(
                "render.instance_upload_reused",
                f64::from(u8::from(instance_count != 0 && instance_upload_bytes == 0)),
            );
            engine_profile::counter("render.instances_repacked", repacked as f64);
            engine_profile::counter("render.poses_repacked", poses_repacked as f64);
            engine_profile::counter("render.colors_repacked", colors_repacked as f64);
            engine_profile::counter(
                "memory.render_upload_cache_bytes",
                (size_of::<UploadCache>() + self.retained_uploads.capacity_bytes()) as f64,
            );
            engine_profile::counter(
                "memory.render_upload_range_bytes",
                (UploadCache::range_scratch_bytes() + RetainedUploads::range_scratch_bytes())
                    as f64,
            );
            engine_profile::counter(
                "memory.render_staging_capacity_bytes",
                (self.staging.capacity() * size_of::<InstanceData>()
                    + self.color_staging.capacity() * size_of::<ColorData>()
                    + self.slot_staging.capacity() * size_of::<u32>()
                    + self.vertex_staging.capacity() * size_of::<Vertex>()) as f64,
            );
            engine_profile::counter(
                "memory.gpu_pose_staging_bytes",
                self.pose_uploads.capacity_bytes() as f64,
            );
            engine_profile::counter(
                "memory.gpu_instance_buffer_bytes",
                (self.instance_capacity
                    * if affine {
                        size_of::<PoseData>() + size_of::<ColorData>()
                    } else {
                        size_of::<InstanceData>()
                    }
                    + self.slot_capacity * size_of::<u32>()) as f64,
            );
            engine_profile::counter("render.upload_bytes", stats.upload_bytes as f64);
            engine_profile::counter(
                "memory.gpu_mesh_payload_bytes",
                stats.resident_mesh_bytes as f64,
            );
            engine_profile::counter(
                "memory.gpu_mesh_arena_bytes",
                self.mesh_arena.capacity_bytes() as f64,
            );
            engine_profile::counter(
                "memory.gpu_texture_payload_bytes",
                stats.resident_texture_bytes as f64,
            );
            engine_profile::counter(
                "memory.gpu_texture_mip_bytes",
                self.materials.texture_totals().2 as f64,
            );
        }
    }
}
