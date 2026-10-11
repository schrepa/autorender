//! A frame's [`RenderStats`] and, while profiling, its counters.

use super::frame::{EncodeReport, FrameKind, FrameState};
use super::*;
use crate::clouds::CloudPass;
use crate::decals::DecalPass;
use crate::gpu_presentation::GpuPrepared;
use crate::ground_cover::GroundCoverPass;
use crate::occupancy::OccupancyPass;
use crate::transparent::Transparency;
use crate::water::WaterPass;

use engine_core::{
    Applied, CullingStats, DeformationStats, DetailStats, ShadowStats, TerrainRenderStats,
    VirtualGeometryStats,
};

/// Records every number and flag of `stats` as the counter `render.<path>`,
/// its path in the JSON hosts read: no list to keep. Fields reading `null`
/// record nothing.
#[cfg(feature = "profiling")]
#[expect(clippy::expect_used, reason = "render stats always serialize")]
fn stat_counters(stats: &RenderStats) {
    use engine_core::collections::HashMap;
    thread_local! {
        // Counter names live as long as the profiler: one per stats field.
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
    let value = serde_json::to_value(stats).expect("render stats serialize");
    visit(&mut "render".to_owned(), &value);
}

#[cfg(not(feature = "profiling"))]
fn stat_counters(_: &RenderStats) {}

impl Renderer {
    /// The frame's stats, after submission (read-backs arrive a few frames
    /// late), and its profile counters while a capture runs.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
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
        let shadow_draws = report.shadow_draws;
        let features = &state.features;
        let (shadow_frame, post_frame) = (&features.shadow_frame, features.post_frame());
        let effects = PostPass::stats(
            self.features.post.get(),
            features.frames.post.as_ref().map(Ready::frame),
            snapshot,
        );
        let refraction = features.refraction;
        let (water, ocean) = WaterPass::stats(
            Some(self.features.water.get()),
            features.frames.water.as_ref().map(Ready::frame),
            snapshot,
        );
        let (mesh_upload_bytes, texture_upload_bytes) =
            (state.assets.mesh_bytes, state.assets.texture_bytes);
        let (camera_upload_bytes, instance_upload_bytes) =
            (state.writes.camera_bytes, state.writes.instance_bytes);

        let gpu_visible = if gpu_mode {
            self.gpu_presentation.visible()
        } else {
            None
        };
        // Read back a few frames late.
        let virtual_stats = virtual_frame
            .then(|| self.virtual_draws.pass.stats())
            .flatten();
        // Pipelines compile before the work submitted after them runs: this
        // frame's completion ends their compiles.
        let pipelines_created = self.pipeline_counts.frame_submitted(&self.queue);
        // Read back a few frames late, like the visible count.
        let lod_instances = if gpu_mode && snapshot.environment.level_of_detail.enabled {
            self.gpu_presentation.lod_instances()
        } else {
            None
        };
        let shadow_casters = shadow_frame.as_ref().and(self.shadows.caster_counts());
        let residency = self.residency();
        let stats = RenderStats {
            draw_calls: report.draws,
            upload_bytes: camera_upload_bytes
                + instance_upload_bytes
                + mesh_upload_bytes
                + texture_upload_bytes
                + self.camera.basin_fog.upload_bytes,
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
            read_back_age: if gpu_mode {
                self.gpu_presentation
                    .read_back_age()
                    .map(|age| age.min(u64::from(u32::MAX)) as u32)
            } else {
                None
            },
            resident_meshes: residency.meshes,
            mesh_upload_bytes,
            resident_mesh_bytes: residency.mesh_bytes,
            resident_textures: residency.textures,
            texture_upload_bytes,
            resident_texture_bytes: residency.texture_bytes,
            submitted: true,
            transparent_draws: if instance_count > 0 {
                self.plan
                    .batches
                    .iter()
                    .filter(|batch| batch.transparent)
                    .count() as u32
            } else {
                0
            },
            pipelines: self.pipeline_counts.created(),
            pipelines_created,
            render_passes: crate::render_graph::render_passes(report.passes),
            shadows: ShadowStats {
                active: shadow_frame.is_some(),
                draws: if shadow_frame.is_some() {
                    shadow_draws
                } else {
                    0
                },
                kept_redraw: shadow_frame
                    .as_ref()
                    .map_or(0, |frame| u32::from(frame.refresh_kind())),
                near_casters: shadow_casters.map(|casters| casters[0]),
                moving_casters: shadow_casters.map(|casters| casters[1]),
                kept_casters: shadow_casters.map(|casters| casters[2]),
                level_tiles: shadow_frame
                    .as_ref()
                    .map_or(0, crate::shadows::ShadowFrame::level_tiles),
                levels_pending: shadow_frame
                    .as_ref()
                    .map_or(0, crate::shadows::ShadowFrame::level_pending),
                level_casters: shadow_casters.map(|casters| casters[3]),
            },
            ambient_occlusion: effects.ambient_occlusion,
            contact_shadows: effects.contact_shadows,
            bloom: effects.bloom,
            local_lights: LightPass::stats(
                self.features.lights.get(),
                features.light_frame(),
                snapshot,
            ),
            sun_shafts: effects.sun_shafts,
            reflections: effects.reflections,
            particles: Transparency::stats(
                Some(self.features.transparency.get()),
                features.frames.transparency.as_ref().map(Ready::frame),
                snapshot,
            ),
            refraction: Applied {
                active: refraction && post_frame.is_some(),
            },
            decals: DecalPass::stats(
                self.features.decals.get(),
                features.frames.decals.as_ref().map(Ready::frame),
                snapshot,
            ),
            reflection_probes: ProbePass::stats(Some(self.features.probes.get()), None, snapshot),
            virtual_geometry: VirtualGeometryStats {
                clusters: virtual_stats.map_or(0, |stats| stats.clusters),
                triangles: virtual_stats.map_or(0, |stats| stats.triangles),
                shadow_clusters: virtual_stats.map_or(0, |stats| stats.shadow_clusters),
                shadow_triangles: virtual_stats.map_or(0, |stats| stats.shadow_triangles),
                occluded_clusters: virtual_stats.map_or(0, |stats| stats.occluded),
                dropped: virtual_stats.map_or(0, |stats| stats.dropped),
                resident_pages: virtual_stats.map_or(0, |stats| stats.resident_pages),
                loading_pages: virtual_stats.map_or(0, |stats| stats.loading_pages),
                missing_pages: virtual_stats.map_or(0, |stats| stats.missing_pages),
                dropped_pages: virtual_stats.map_or(0, |stats| stats.dropped_pages),
                failed_pages: virtual_stats.map_or(0, |stats| stats.failed_pages),
                visibility_buffer: resolving,
            },
            level_of_detail: DetailStats {
                lod0_instances: lod_instances.map(|levels| levels[0]),
                lod1_instances: lod_instances.map(|levels| levels[1]),
                lod2_instances: lod_instances.map(|levels| levels[2]),
                impostor_instances: lod_instances.map(|levels| levels[3]),
                impostors: self.impostors.live(),
                dropped_impostors: self.impostors.dropped,
                failed_meshes: self.mesh_uploads.lod_failures.0,
            },
            terrain: TerrainRenderStats {
                patches: self.terrain.patches.frame.patches,
                triangles: self.terrain.patches.frame.triangles,
                shadow_patches: self.terrain.patches.frame.shadow_patches,
                tiles: self.terrain.resident(),
                loading_tiles: snapshot.terrain.stats.loading_tiles,
                unplaced_tiles: snapshot.terrain.stats.unplaced_tiles,
                gpu_controls: self.terrain.atlas.controls,
                dropped_tiles: snapshot.terrain.stats.dropped_tiles + self.terrain.atlas.dropped,
                fallback_patches: snapshot.terrain.stats.fallback_patches,
                horizons: self.terrain.horizons.count(),
                horizon_builds: self.terrain.horizons.builds,
                sky_maps: self.terrain.sky.count(),
                sky_rows: self.terrain.sky.rows,
                dropped_materials: self.terrain.tables.dropped_materials,
                vt_slots: self.terrain.vt.stats.slots,
                vt_pages: self.terrain.vt.stats.pages,
                vt_bakes: self.terrain.vt.stats.bakes,
                vt_missing: self.terrain.vt.stats.missing,
                dropped_vt_pages: self.terrain.vt.stats.dropped,
                vt_evictions: self.terrain.vt.stats.evictions,
                placed_objects: self.features.scatter.get().resident,
                placed_far_objects: self.scatter_draws.far_objects(),
                placed_impostors: self.impostors.placed.live(),
                dropped_placed_objects: self.features.scatter.get().left_out(),
            },
            ground_cover: GroundCoverPass::stats(
                Some(self.features.ground_cover.get()),
                features.frames.ground_cover.as_ref().map(Ready::frame),
                snapshot,
            ),
            occupancy: OccupancyPass::stats(
                Some(self.features.occupancy.get()),
                features.frames.occupancy.as_ref().map(Ready::frame),
                snapshot,
            ),
            water,
            ocean,
            deformation: DeformationStats {
                stamps: self.terrain.deformation.pressed,
                texels: self.terrain.deformation.texels,
            },
            clouds: CloudPass::stats(
                Some(self.features.clouds.get()),
                features.frames.clouds.as_ref().map(Ready::frame),
                snapshot,
            ),
            basin_fog: engine_core::BasinFogStats {
                gpu_bytes: self.camera.basin_fog.bytes,
                upload_bytes: self.camera.basin_fog.upload_bytes,
                nodes: self.camera.basin_fog.nodes,
            },
            world_streaming: snapshot.cells,
            sky: crate::sky::SkyPass::stats(Some(self.features.sky.get()), None, snapshot),
            occlusion_culling: CullingStats {
                active: gpu_mode && self.gpu_presentation.occlusion(),
                instances: if gpu_mode && snapshot.environment.occlusion_culling.enabled {
                    self.gpu_presentation.occluded()
                } else {
                    None
                },
            },
            subpixel_culling: CullingStats {
                active: gpu_mode && self.gpu_presentation.culls_subpixel(),
                instances: if gpu_mode && snapshot.environment.subpixel_culling.enabled {
                    self.gpu_presentation.subpixel()
                } else {
                    None
                },
            },
            temporal_antialiasing: TemporalPass::stats(
                self.features.temporal.get(),
                features.frames.temporal.as_ref().map(Ready::frame),
                snapshot,
            ),
        };
        // Counted every frame, recorded while a capture runs.
        let immediate = self.queue.take_immediate();
        self.profile_counters(state, &stats, gpu_visible, immediate);
        stats
    }

    /// The frame's profile counters while a capture runs, with the
    /// submissions made at once since the last frame, `immediate`.
    fn profile_counters(
        &self,
        state: &FrameState<'_>,
        stats: &RenderStats,
        gpu_visible: Option<u32>,
        immediate: u32,
    ) {
        if !engine_profile::active() {
            return;
        }
        stat_counters(stats);
        engine_profile::counter("render.immediate_submissions", f64::from(immediate));
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
            self.features.post.get().map_or(0, PostPass::capacity_bytes) as f64,
        );
        engine_profile::counter(
            "memory.gpu_environment_bytes",
            (self.features.environment.get().capacity_bytes()
                + self.features.probes.get().capacity_bytes()
                + self.features.clouds.get().capacity_bytes()
                + self.features.sky.get().capacity_bytes()) as f64,
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
            self.features
                .lights
                .get()
                .map_or(0, LightPass::capacity_bytes) as f64,
        );
        engine_profile::counter(
            "memory.gpu_impostor_bytes",
            self.impostors.capacity_bytes() as f64,
        );
        engine_profile::counter(
            "memory.gpu_basin_fog_bytes",
            self.camera.basin_fog.bytes as f64,
        );
        engine_profile::counter(
            "memory.gpu_terrain_bytes",
            (self.terrain.capacity_bytes()
                + self.features.ground_cover.get().capacity_bytes()
                + self.features.scatter.get().capacity_bytes()
                + self.features.occupancy.get().capacity_bytes()
                + self.features.water.get().capacity_bytes()) as f64,
        );
        engine_profile::counter(
            "memory.gpu_transparent_bytes",
            (self.features.transparency.get().capacity_bytes()
                + self
                    .features
                    .decals
                    .get()
                    .map_or(0, DecalPass::capacity_bytes)) as f64,
        );
        engine_profile::counter(
            "memory.gpu_temporal_bytes",
            (self
                .features
                .temporal
                .get()
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
                + self.mesh_uploads.staging.capacity() * size_of::<Vertex>()) as f64,
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
