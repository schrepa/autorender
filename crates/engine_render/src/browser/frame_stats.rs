//! A frame's [`RenderStats`] and, while profiling, its counters.

use super::frame::{EncodeReport, FrameKind, FrameState};
use super::*;
use crate::gpu_presentation::GpuPrepared;

/// A profile counter read from a frame's stats; `None` records nothing.
type StatCounter = (&'static str, fn(&RenderStats) -> Option<f64>);

/// Counters of every frame's stats.
const STAT_COUNTERS: [StatCounter; 26] = [
    ("render.instances", |stats| Some(stats.instances.into())),
    ("render.draw_calls", |stats| Some(stats.draw_calls.into())),
    ("render.upload_bytes", |stats| {
        Some(stats.upload_bytes as f64)
    }),
    ("render.mesh_upload_bytes", |stats| {
        Some(stats.mesh_upload_bytes as f64)
    }),
    ("render.texture_upload_bytes", |stats| {
        Some(stats.texture_upload_bytes as f64)
    }),
    ("render.shadow_draws", |stats| {
        Some(stats.shadow_draws.into())
    }),
    ("render.shadow_near_casters", |stats| {
        stats.shadow_near_casters.map(f64::from)
    }),
    ("render.shadow_moving_casters", |stats| {
        stats.shadow_moving_casters.map(f64::from)
    }),
    ("render.shadow_kept_casters", |stats| {
        stats.shadow_kept_casters.map(f64::from)
    }),
    ("render.shadow_level_tiles", |stats| {
        Some(stats.shadow_level_tiles.into())
    }),
    ("render.shadow_level_casters", |stats| {
        stats.shadow_level_casters.map(f64::from)
    }),
    ("render.ambient_occlusion", |stats| {
        Some(u8::from(stats.ambient_occlusion).into())
    }),
    ("render.bloom", |stats| Some(u8::from(stats.bloom).into())),
    ("render.transparent_draws", |stats| {
        Some(stats.transparent_draws.into())
    }),
    ("render.particles", |stats| Some(stats.particles.into())),
    ("render.decals", |stats| Some(stats.decals.into())),
    ("render.impostors", |stats| Some(stats.impostors.into())),
    ("render.dropped_impostors", |stats| {
        Some(stats.dropped_impostors.into())
    }),
    ("render.pipelines_created", |stats| {
        Some(stats.pipelines_created.into())
    }),
    ("render.occluded_instances", |stats| {
        stats.occluded_instances.map(f64::from)
    }),
    ("render.subpixel_instances", |stats| {
        stats.subpixel_instances.map(f64::from)
    }),
    ("render.triangles", |stats| stats.triangles.map(f64::from)),
    ("render.lod0_instances", |stats| {
        stats.lod0_instances.map(f64::from)
    }),
    ("render.lod1_instances", |stats| {
        stats.lod1_instances.map(f64::from)
    }),
    ("render.lod2_instances", |stats| {
        stats.lod2_instances.map(f64::from)
    }),
    ("render.impostor_instances", |stats| {
        stats.impostor_instances.map(f64::from)
    }),
];

/// Counters of frames that shade local lights (decision 023).
const LIGHT_COUNTERS: [StatCounter; 3] = [
    ("render.local_lights", |stats| {
        Some(stats.local_lights.into())
    }),
    ("render.lights_visible", |stats| {
        stats.lights_visible.map(f64::from)
    }),
    ("render.light_cluster_overflow", |stats| {
        stats.light_cluster_overflow.map(f64::from)
    }),
];

/// Counters of frames that read virtual geometry's selection back
/// (decision 044).
const VIRTUAL_COUNTERS: [StatCounter; 9] = [
    ("render.virtual_clusters", |stats| {
        Some(stats.virtual_clusters.into())
    }),
    ("render.virtual_triangles", |stats| {
        Some(stats.virtual_triangles.into())
    }),
    ("render.virtual_shadow_triangles", |stats| {
        Some(stats.virtual_shadow_triangles.into())
    }),
    ("render.virtual_occluded_clusters", |stats| {
        Some(stats.virtual_occluded_clusters.into())
    }),
    ("render.virtual_resident_pages", |stats| {
        Some(stats.virtual_resident_pages.into())
    }),
    ("render.virtual_loading_pages", |stats| {
        Some(stats.virtual_loading_pages.into())
    }),
    ("render.virtual_missing_pages", |stats| {
        Some(stats.virtual_missing_pages.into())
    }),
    ("render.dropped_virtual_pages", |stats| {
        Some(stats.dropped_virtual_pages.into())
    }),
    ("render.dropped_virtual", |stats| {
        Some(stats.dropped_virtual.into())
    }),
];

/// Records `counters` of `stats`.
fn stat_counters(stats: &RenderStats, counters: &[StatCounter]) {
    for (name, read) in counters {
        if let Some(value) = read(stats) {
            engine_profile::counter(name, value);
        }
    }
}

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
        let (snapshot, frame) = (state.snapshot, state.frame);
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
            .then(|| self.virtual_draws.pass.stats())
            .flatten();
        // Pipelines compile before the work submitted after them runs: this
        // frame's completion ends their compiles (decision 049).
        let pipelines_created = self.pipeline_counts.frame_submitted(&self.queue);
        // Read back a few frames late, like the visible count.
        let lod_instances = if gpu_mode && snapshot.environment.level_of_detail.enabled {
            self.gpu_presentation.lod_instances()
        } else {
            None
        };
        let shadow_casters = shadow_frame.as_ref().and(self.shadows.caster_counts());
        let lit = light_frame.is_some();
        let light_totals = light_frame
            .as_ref()
            .and(self.lights.as_ref().and_then(LightPass::totals));
        let stats = RenderStats {
            draw_calls: report.draws,
            upload_bytes: camera_upload_bytes
                + instance_upload_bytes
                + mesh_upload_bytes
                + texture_upload_bytes,
            // GPU-culled frames report the latest read-back visible count
            // within the budget the GPU applied.
            instances: match gpu_visible {
                Some(visible) if gpu_mode => {
                    visible.min(frame.instance_budget().min(u32::MAX as usize) as u32)
                }
                None if gpu_mode => instance_count.min(frame.instance_budget()) as u32,
                _ => instance_count as u32,
            },
            gpu_visible,
            occluded_instances: if gpu_mode && self.culling.occlusion {
                self.gpu_presentation.occluded()
            } else {
                None
            },
            subpixel_instances: if gpu_mode && self.culling.subpixel {
                self.gpu_presentation.subpixel()
            } else {
                None
            },
            lod0_instances: lod_instances.map(|levels| levels[0]),
            lod1_instances: lod_instances.map(|levels| levels[1]),
            lod2_instances: lod_instances.map(|levels| levels[2]),
            impostor_instances: lod_instances.map(|levels| levels[3]),
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
                    .filter(|(index, _)| self.virtual_draws.drawn.get(*index) != Some(&true))
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
            shadows_enabled: shadow_frame.is_some(),
            shadow_draws: if shadow_frame.is_some() {
                shadow_draws
            } else {
                0
            },
            shadow_kept_redraw: shadow_frame
                .as_ref()
                .map_or(0, |frame| u32::from(frame.refresh_kind())),
            shadow_near_casters: shadow_casters.map(|casters| casters[0]),
            shadow_moving_casters: shadow_casters.map(|casters| casters[1]),
            shadow_kept_casters: shadow_casters.map(|casters| casters[2]),
            shadow_level_tiles: shadow_frame
                .as_ref()
                .map_or(0, crate::shadows::ShadowFrame::level_tiles),
            shadow_levels_pending: shadow_frame
                .as_ref()
                .map_or(0, crate::shadows::ShadowFrame::level_pending),
            shadow_level_casters: shadow_casters.map(|casters| casters[3]),
            ambient_occlusion: post_frame.is_some_and(|frame| frame.occlusion),
            contact_shadows: post_frame.is_some_and(|frame| frame.contact),
            bloom: post_frame.is_some_and(|frame| frame.bloom),
            local_lights: if lit { snapshot.lights.len() as u32 } else { 0 },
            dropped_lights: if lit {
                snapshot.dropped_lights.min(u32::MAX as usize) as u32
            } else {
                0
            },
            lights_visible: light_totals.map(|totals| totals[0]),
            light_cluster_overflow: light_totals.map(|totals| totals[1]),
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
            dropped_impostors: self.impostors.dropped,
            reflection_probes: self.probes.slots.ready(),
            reflection_probe_faces: probe_faces,
            dropped_reflection_probes: dropped_probes.min(u64::from(u32::MAX)) as u32,
            pipelines: self.pipeline_counts.created(),
            pipelines_created,
            passes: report.passes,
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
            terrain_patches: self.terrain.frame.patches,
            terrain_triangles: self.terrain.frame.triangles,
            terrain_shadow_patches: self.terrain.frame.shadow_patches,
            terrain_tiles: self.terrain.resident(),
            terrain_loading_tiles: snapshot.terrain.stats.loading_tiles,
            dropped_terrain_tiles: snapshot.terrain.stats.dropped_tiles + self.terrain.dropped,
            terrain_fallback_patches: snapshot.terrain.stats.fallback_patches,
            terrain_horizons: self.terrain.horizons.count(),
            terrain_horizon_builds: self.terrain.horizons.builds,
            terrain_sky_maps: self.terrain.sky.count(),
            terrain_sky_rows: self.terrain.sky.rows,
            ground_cover_blades: self.ground_cover.totals()[0],
            dropped_ground_cover: self.ground_cover.totals()[1],
            ground_cover_reach: self.ground_cover.reach().0.round() as u32,
            ground_cover_tiers: self.ground_cover.reach().1,
            placed_objects: self.scatter.resident,
            water_chunks: if state.shape.water {
                self.water.chunks_drawn()
            } else {
                0
            },
            ocean_sectors: if state.shape.water {
                self.water.ocean_drawn().0
            } else {
                0
            },
            ocean_triangles: if state.shape.water {
                self.water.ocean_drawn().1
            } else {
                0
            },
            cloud_rows: self.clouds.rows,
            cloud_view_blocks: self.clouds.view_blocks,
            sky_tables: self.sky.built,
            cells: snapshot.cells.cells,
            cells_loaded: snapshot.cells.loaded,
            cells_pending: snapshot.cells.pending,
            streamed_entities: snapshot.cells.entities,
            cell_spawns: snapshot.cells.spawned,
            cell_despawns: snapshot.cells.despawned,
            deformation_stamps: self.terrain.deformation.pressed,
            deformation_texels: self.terrain.deformation.texels,
            placed_far_objects: self.scatter_draws.far_objects(),
            placed_impostors: self.impostors.placed.live(),
            occupancy_redraws: self.occupancy.redraws,
            dropped_terrain_materials: self.terrain.dropped_materials,
            ..self.residency_stats()
        };
        self.profile_counters(state, &stats, gpu_visible, virtual_stats);
        stats
    }

    /// The frame's profile counters while a capture runs.
    fn profile_counters(
        &self,
        state: &FrameState<'_>,
        stats: &RenderStats,
        gpu_visible: Option<u32>,
        virtual_stats: Option<crate::virtual_geometry::VirtualStats>,
    ) {
        if !engine_profile::active() {
            return;
        }
        stat_counters(stats, &STAT_COUNTERS);
        if state.features.light_frame.is_some() {
            stat_counters(stats, &LIGHT_COUNTERS);
        }
        if virtual_stats.is_some() {
            stat_counters(stats, &VIRTUAL_COUNTERS);
        }
        self.upload_counters(state);
        if let Some(prepared) = &state.features.gpu_prepared {
            gpu_counters(prepared, gpu_visible);
        }
        self.memory_counters(stats);
    }

    /// What the frame wrote and repacked, and which passes it ran.
    fn upload_counters(&self, state: &FrameState<'_>) {
        let FrameKind {
            gpu_mode,
            instance_count,
            ..
        } = state.kind;
        let (writes, packing) = (&state.writes, &state.packing);
        engine_profile::counter("render.instance_capacity", self.instances.capacity as f64);
        engine_profile::counter(
            "render.texture_mip_levels",
            f64::from(state.assets.mip_levels),
        );
        engine_profile::counter("render.camera_write_bytes", writes.camera_bytes as f64);
        engine_profile::counter("render.instance_write_bytes", writes.instance_bytes as f64);
        engine_profile::counter("render.pose_write_bytes", writes.pose_bytes as f64);
        engine_profile::counter("render.color_write_bytes", writes.color_bytes as f64);
        engine_profile::counter("render.instance_id_write_bytes", writes.slot_bytes as f64);
        engine_profile::counter("render.gpu_presentation", f64::from(u8::from(gpu_mode)));
        engine_profile::counter(
            "render.taa",
            f64::from(u8::from(packing.temporal_frame.is_some())),
        );
        engine_profile::counter(
            "render.shadow_refresh",
            state
                .features
                .shadow_frame
                .as_ref()
                .map_or(0.0, |frame| f64::from(frame.refresh_kind())),
        );
        engine_profile::counter("render.shadow_groups", self.shadows.groups().len() as f64);
        if let Some(frame) = &packing.temporal_frame {
            engine_profile::counter(
                "render.taa_history",
                f64::from(u8::from(frame.history_valid)),
            );
            engine_profile::counter(
                "render.taa_object_motion",
                f64::from(u8::from(frame.motion_valid)),
            );
        }
        engine_profile::counter("render.instance_write_ranges", writes.ranges as f64);
        engine_profile::counter(
            "render.instance_full_upload",
            f64::from(u8::from(packing.full_upload)),
        );
        engine_profile::counter(
            "render.instance_upload_fallback",
            f64::from(u8::from(packing.fallback)),
        );
        engine_profile::counter(
            "render.camera_upload_reused",
            f64::from(u8::from(writes.camera_bytes == 0)),
        );
        engine_profile::counter(
            "render.instance_upload_reused",
            f64::from(u8::from(instance_count != 0 && writes.instance_bytes == 0)),
        );
        engine_profile::counter("render.instances_repacked", packing.repacked as f64);
        engine_profile::counter("render.poses_repacked", packing.poses_repacked as f64);
        engine_profile::counter("render.colors_repacked", packing.colors_repacked as f64);
    }

    /// The renderer's GPU and CPU storage, by part.
    fn memory_counters(&self, stats: &RenderStats) {
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
            (self.environment.capacity_bytes()
                + self.probes.capacity_bytes()
                + self.clouds.capacity_bytes()
                + self.sky.capacity_bytes()) as f64,
        );
        engine_profile::counter(
            "memory.gpu_virtual_bytes",
            (self.virtual_draws.pass.capacity_bytes()
                + self
                    .visibility
                    .buffer
                    .as_ref()
                    .map_or(0, VisibilityBuffer::capacity_bytes)) as f64,
        );
        engine_profile::counter(
            "memory.gpu_light_bytes",
            self.lights.as_ref().map_or(0, LightPass::capacity_bytes) as f64,
        );
        engine_profile::counter(
            "memory.gpu_impostor_bytes",
            self.impostors.capacity_bytes() as f64,
        );
        engine_profile::counter(
            "memory.gpu_terrain_bytes",
            (self.terrain.capacity_bytes()
                + self.ground_cover.capacity_bytes()
                + self.scatter.capacity_bytes()
                + self.occupancy.capacity_bytes()
                + self.water.capacity_bytes()) as f64,
        );
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
                + self.decals.as_ref().map_or(0, DecalPass::capacity_bytes)) as f64,
        );
        engine_profile::counter(
            "memory.gpu_temporal_bytes",
            (self
                .temporal
                .as_ref()
                .map_or(0, TemporalPass::capacity_bytes)
                + self.instances.alternate_bytes()) as f64,
        );
        engine_profile::counter(
            "memory.gpu_presentation_bytes",
            (self.gpu_presentation.capacity_bytes()
                + self
                    .culling
                    .pyramid
                    .as_ref()
                    .map_or(0, DepthPyramid::capacity_bytes)) as f64,
        );
        engine_profile::counter(
            "memory.render_upload_cache_bytes",
            (size_of::<UploadCache>() + self.instances.retained_uploads.capacity_bytes()) as f64,
        );
        engine_profile::counter(
            "memory.render_upload_range_bytes",
            RetainedUploads::range_scratch_bytes() as f64,
        );
        engine_profile::counter(
            "memory.render_staging_capacity_bytes",
            (self.instances.color_staging.capacity() * size_of::<ColorData>()
                + self.instances.slot_staging.capacity() * size_of::<u32>()
                + self.vertex_staging.capacity() * size_of::<Vertex>()) as f64,
        );
        engine_profile::counter(
            "memory.gpu_pose_staging_bytes",
            self.instances.pose_uploads.capacity_bytes() as f64,
        );
        engine_profile::counter(
            "memory.gpu_instance_buffer_bytes",
            (self.instances.capacity * (size_of::<PoseData>() + size_of::<ColorData>())
                + self.instances.slot_capacity * size_of::<u32>()) as f64,
        );
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

/// What GPU presentation gathered and uploaded for the frame.
fn gpu_counters(prepared: &GpuPrepared, gpu_visible: Option<u32>) {
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
