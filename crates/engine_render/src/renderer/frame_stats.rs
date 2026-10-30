use super::frame::{EncodeReport, Uploaded};
use super::*;
use crate::gpu_presentation::{GpuPrepared, PresentationPart};

use engine_core::{CullingStats, DetailStats};

const RAN_ORDERS: usize = 256;

const MEMORY: [(&str, &str); 13] = [
    ("shadows", "memory.gpu_shadow_bytes"),
    ("post", "memory.gpu_post_bytes"),
    ("exposure", "memory.gpu_post_bytes"),
    ("sky", "memory.gpu_environment_bytes"),
    ("probes", "memory.gpu_environment_bytes"),
    ("lights", "memory.gpu_light_bytes"),
    ("terrain", "memory.gpu_terrain_bytes"),
    ("ground_cover", "memory.gpu_terrain_bytes"),
    ("scatter", "memory.gpu_terrain_bytes"),
    ("occupancy", "memory.gpu_terrain_bytes"),
    ("water", "memory.gpu_terrain_bytes"),
    ("transparency", "memory.gpu_transparent_bytes"),
    ("temporal", "memory.gpu_temporal_bytes"),
];

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

impl Renderer {
    pub(super) fn frame_stats(
        &mut self,
        info: &FrameInfo<'_>,
        uploaded: &Uploaded,
        report: &EncodeReport,
    ) -> RenderStats {
        let FrameKind {
            gpu_mode,
            instance_count,
            ..
        } = info.kind;
        let (snapshot, frame) = (info.snapshot, info.frame);
        let (mesh_upload_bytes, texture_upload_bytes) =
            (uploaded.assets.mesh_bytes, uploaded.assets.texture_bytes);
        let (camera_upload_bytes, instance_upload_bytes) =
            (uploaded.writes.camera_bytes, uploaded.writes.instance_bytes);
        let pipelines_created = self.core.pipeline_counts.frame_submitted(&self.core.queue);
        let render_passes = self.ran_passes();
        let core = &self.core;
        let gpu_visible = if gpu_mode {
            core.gpu_presentation.visible()
        } else {
            None
        };
        let residency = core.residency();
        let mut stats = RenderStats {
            draw_calls: report.draws,
            upload_bytes: camera_upload_bytes
                + instance_upload_bytes
                + mesh_upload_bytes
                + texture_upload_bytes,
            instances: match gpu_visible {
                Some(visible) if gpu_mode => {
                    visible.min(frame.instance_budget().min(u32::MAX as usize) as u32)
                }
                None if gpu_mode => instance_count.min(frame.instance_budget()) as u32,
                _ => instance_count as u32,
            },
            gpu_visible,
            triangles: if gpu_mode {
                core.gpu_presentation.triangles()
            } else {
                let triangles: u64 = core
                    .plan
                    .batches
                    .iter()
                    .map(|batch| {
                        u64::from(batch.instances.end - batch.instances.start)
                            * u64::from(
                                core.meshes
                                    .get(&batch.mesh)
                                    .map_or(0, |mesh| mesh.index_count)
                                    / 3,
                            )
                    })
                    .sum();
                Some(triangles.min(u64::from(u32::MAX)) as u32)
            },
            read_back_age: if gpu_mode {
                core.gpu_presentation
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
            pipelines: core.pipeline_counts.created(),
            pipelines_created,
            render_passes,
            ..RenderStats::default()
        };
        core.culling_stats(snapshot, gpu_mode, &mut stats);
        let view = self.view(info);
        for (decl, part) in self.parts.iter() {
            let _owner = crate::memory::owned(decl.brick);
            part.stats(&view, &mut stats);
        }
        let immediate = core.queue.take_immediate();
        self.profile_counters(info, uploaded, &stats, (gpu_visible, immediate));
        stats
    }

    fn ran_passes(&mut self) -> engine_core::Names {
        let ran: Vec<&'static str> = self
            .graph
            .kept_passes()
            .map(|node| self.table.entry(node.pass()).name)
            .collect();
        let names = match self.ran.iter().find(|&&known| known == ran.as_slice()) {
            Some(&known) => known,
            None if self.ran.len() < RAN_ORDERS => {
                let leaked: &'static [&'static str] = Box::leak(ran.into_boxed_slice());
                self.ran.push(leaked);
                leaked
            }
            None => return self.table.names_of(self.graph.kept()),
        };
        engine_core::Names {
            bits: 1_u64
                .checked_shl(names.len() as u32)
                .map_or(u64::MAX, |bit| bit - 1),
            names,
        }
    }

    fn profile_counters(
        &self,
        info: &FrameInfo<'_>,
        uploaded: &Uploaded,
        stats: &RenderStats,
        (gpu_visible, immediate): (Option<u32>, u32),
    ) {
        if !engine_profile::active() {
            return;
        }
        stat_counters(stats);
        engine_profile::counter("render.immediate_submissions", f64::from(immediate));
        self.upload_counters(info, uploaded);
        if let Some(prepared) = self
            .parts
            .presentation()
            .and_then(PresentationPart::prepared)
        {
            gpu_counters(prepared, gpu_visible);
        }
        self.memory_counters(stats);
    }

    fn upload_counters(&self, info: &FrameInfo<'_>, uploaded: &Uploaded) {
        let FrameKind {
            gpu_mode,
            instance_count,
            ..
        } = info.kind;
        let (writes, packing) = (&uploaded.writes, &uploaded.packing);
        let core = &self.core;
        engine_profile::counter("render.instance_capacity", core.instances.capacity as f64);
        engine_profile::counter(
            "render.texture_mip_levels",
            f64::from(uploaded.assets.mip_levels),
        );
        engine_profile::counter("render.camera_write_bytes", writes.camera_bytes as f64);
        engine_profile::counter("render.instance_write_bytes", writes.instance_bytes as f64);
        engine_profile::counter("render.pose_write_bytes", writes.pose_bytes as f64);
        engine_profile::counter("render.color_write_bytes", writes.color_bytes as f64);
        engine_profile::counter("render.instance_id_write_bytes", writes.slot_bytes as f64);
        engine_profile::counter("render.gpu_presentation", f64::from(u8::from(gpu_mode)));
        if let Some(frame) = &info.temporal {
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

    fn part_bytes(&self, counter: &str) -> u64 {
        self.parts
            .iter()
            .filter(|(decl, _)| {
                MEMORY
                    .iter()
                    .any(|&(part, named)| part == decl.name && named == counter)
            })
            .map(|(_, part)| part.capacity_bytes())
            .sum()
    }

    fn memory_counters(&self, stats: &RenderStats) {
        let core = &self.core;
        for counter in [
            "memory.gpu_shadow_bytes",
            "memory.gpu_post_bytes",
            "memory.gpu_environment_bytes",
            "memory.gpu_light_bytes",
        ] {
            engine_profile::counter(counter, self.part_bytes(counter) as f64);
        }
        engine_profile::counter(
            "memory.gpu_impostor_bytes",
            core.impostors.capacity_bytes() as f64,
        );
        for counter in ["memory.gpu_terrain_bytes", "memory.gpu_transparent_bytes"] {
            engine_profile::counter(counter, self.part_bytes(counter) as f64);
        }
        engine_profile::counter(
            "memory.gpu_temporal_bytes",
            (self.part_bytes("memory.gpu_temporal_bytes")
                + core.instances.alternate_bytes()
                + core.previous.capacity_bytes()) as f64,
        );
        engine_profile::counter(
            "memory.gpu_presentation_bytes",
            (core.gpu_presentation.capacity_bytes()
                + core
                    .culling
                    .pyramid
                    .as_ref()
                    .map_or(0, DepthPyramid::capacity_bytes)) as f64,
        );
        engine_profile::counter(
            "memory.render_upload_cache_bytes",
            (size_of::<UploadCache>() + core.instances.retained_uploads.capacity_bytes()) as f64,
        );
        engine_profile::counter(
            "memory.render_upload_range_bytes",
            RetainedUploads::range_scratch_bytes() as f64,
        );
        engine_profile::counter(
            "memory.render_staging_capacity_bytes",
            (core.instances.color_staging.capacity() * size_of::<ColorData>()
                + core.instances.slot_staging.capacity() * size_of::<u32>()
                + core.mesh_uploads.staging.capacity() * size_of::<Vertex>()) as f64,
        );
        engine_profile::counter(
            "memory.gpu_pose_staging_bytes",
            core.instances.pose_uploads.capacity_bytes() as f64,
        );
        engine_profile::counter(
            "memory.gpu_instance_buffer_bytes",
            (core.instances.capacity * (size_of::<PoseData>() + size_of::<ColorData>())
                + core.instances.slot_capacity * size_of::<u32>()) as f64,
        );
        engine_profile::counter(
            "memory.gpu_mesh_payload_bytes",
            stats.resident_mesh_bytes as f64,
        );
        engine_profile::counter(
            "memory.gpu_mesh_arena_bytes",
            core.mesh_arena.capacity_bytes() as f64,
        );
        engine_profile::counter(
            "memory.gpu_texture_payload_bytes",
            stats.resident_texture_bytes as f64,
        );
        engine_profile::counter(
            "memory.gpu_texture_mip_bytes",
            core.materials.texture_totals().2 as f64,
        );
    }
}

impl Core {
    fn culling_stats(&self, snapshot: &RenderSnapshot, gpu_mode: bool, stats: &mut RenderStats) {
        let lod_instances = if gpu_mode && snapshot.environment.level_of_detail.enabled {
            self.gpu_presentation.lod_instances()
        } else {
            None
        };
        stats.level_of_detail = DetailStats {
            lod0_instances: lod_instances.map(|levels| levels[0]),
            lod1_instances: lod_instances.map(|levels| levels[1]),
            lod2_instances: lod_instances.map(|levels| levels[2]),
            impostor_instances: lod_instances.map(|levels| levels[3]),
            impostors: self.impostors.live(),
            dropped_impostors: self.impostors.dropped,
            meshes_without_levels: self
                .meshes
                .values()
                .filter(|mesh| mesh.without_levels)
                .count() as u32,
            impostor_cut: engine_core::Cut::default(),
        };
        stats.occlusion_culling = CullingStats {
            active: gpu_mode && self.gpu_presentation.occlusion(),
            instances: if gpu_mode && snapshot.environment.occlusion_culling.enabled {
                self.gpu_presentation.occluded()
            } else {
                None
            },
        };
        stats.subpixel_culling = CullingStats {
            active: gpu_mode && self.gpu_presentation.culls_subpixel(),
            instances: if gpu_mode && snapshot.environment.subpixel_culling.enabled {
                self.gpu_presentation.subpixel()
            } else {
                None
            },
        };
    }
}

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
