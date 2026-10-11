//! One frame of [`Renderer`], stage by stage: planning, the surface, asset
//! uploads, instance packing and writes, each feature's preparation, the
//! frame's pipelines, encoding and submission. `frame_stats` reports it.

use super::*;
use crate::decals::DecalFrame;
use crate::gpu_presentation::{CullingView, GpuPrepared, PresentationInputs, RecordsKey, Staging};
use crate::particles::ParticleFrame;
use crate::post::PostFrame;
use crate::render_graph::{self, Clear, Effects, FrameShape, Load, Node, Ops, Pass, Resource};
use crate::shadows::{MapSources, ShadowFrame, ShadowInputs};

/// How far switches between how objects show fade each frame:
/// the presentation time of the last displayed GPU-presented
/// frame, and this frame's share of a switch's fade.
#[derive(Default)]
pub(super) struct SwitchFades {
    presented: f32,
    pub(super) share: f32,
}

impl SwitchFades {
    /// This frame's share of a fade `fade_seconds` long, at presentation time
    /// `seconds`: displayed GPU-presented frames advance by the time since
    /// the last (none for one presenting the same moment again, as the
    /// first after a pause does); previews switch at once (1), and exact
    /// frames show objects as they stand (a negative share).
    fn advance(&mut self, kind: FrameKind, seconds: f32, fade_seconds: f32) {
        self.share = match (kind.gpu_mode, kind.display) {
            (false, _) => -1.0,
            (true, false) => 1.0,
            (true, true) => {
                let since = seconds - std::mem::replace(&mut self.presented, seconds);
                if fade_seconds > 0.0 && since < 1.0 {
                    engine_core::math::clamp(since / fade_seconds, 0.0, 1.0)
                } else {
                    1.0
                }
            }
        };
    }
}

/// What kind of frame this is, decided before any GPU work.
#[derive(Clone, Copy)]
pub(super) struct FrameKind {
    /// Presented on the GPU.
    pub(super) gpu_mode: bool,
    /// Temporally antialiased.
    pub(super) temporal: bool,
    /// Shown as the page's next frame, not a preview or capture.
    pub(super) display: bool,
    pub(super) instance_count: usize,
}

/// A frame's asset uploads, and the encoder of new textures' mip levels and
/// impostor bakes, submitted first.
pub(super) struct AssetUploads {
    pub(super) mesh_bytes: u64,
    pub(super) texture_bytes: u64,
    pub(super) mip_levels: u32,
    pub(super) encoder: Option<wgpu::CommandEncoder>,
}

/// How a frame packed its instance storage.
pub(super) struct Packing {
    /// This frame's temporal antialiasing.
    pub(super) temporal_frame: Option<TemporalFrame>,
    write_slots: bool,
    slot_key: Option<(u64, u64)>,
    pub(super) repacked: usize,
    pub(super) poses_repacked: usize,
    pub(super) colors_repacked: usize,
    pub(super) full_upload: bool,
    pub(super) fallback: bool,
}

/// A frame's uniform and instance writes, and the staged pose copies
/// submitted before it.
pub(super) struct Writes {
    pub(super) lighting: LightingUniform,
    pub(super) camera_bytes: u64,
    pub(super) instance_bytes: u64,
    pub(super) pose_bytes: u64,
    pub(super) color_bytes: u64,
    pub(super) slot_bytes: u64,
    pub(super) ranges: usize,
    pub(super) staged_poses: Option<wgpu::CommandBuffer>,
}

/// What each feature prepared for a frame.
pub(super) struct Features<'s> {
    /// GPU presentation's input copies, submitted before the frame.
    pub(super) input_copies: Option<wgpu::CommandEncoder>,
    /// The viewport and depth pyramid levels of occlusion culling.
    pub(super) occlusion: Option<(u32, u32, u32)>,
    pub(super) gpu_prepared: Option<GpuPrepared>,
    pub(super) shadow_frame: Option<ShadowFrame>,
    /// The physically based sky's work, submitted first:
    /// what reads the light it gives follows it.
    pub(super) sky_encoder: Option<wgpu::CommandEncoder>,
    /// Environment maps' and reflection probes' work,
    /// submitted before the frame.
    pub(super) environment_encoder: Option<wgpu::CommandEncoder>,
    pub(super) probe_list: &'s [engine_core::RenderReflectionProbe],
    pub(super) dropped_probes: u64,
    pub(super) light_frame: Option<LightFrame>,
    pub(super) post_frame: Option<PostFrame>,
    pub(super) refraction: bool,
    pub(super) transparent_draws: bool,
    pub(super) particle_frame: Option<ParticleFrame>,
    pub(super) transparency: bool,
    pub(super) decal_frame: Option<DecalFrame>,
}

/// GPU presentation's preparation: the input copies submitted before the
/// frame, occlusion culling's viewport and depth pyramid levels, and what
/// presentation prepared.
struct PresentationPrep {
    input_copies: Option<wgpu::CommandEncoder>,
    occlusion: Option<(u32, u32, u32)>,
    gpu_prepared: Option<GpuPrepared>,
}

/// The mesh pass kinds a frame draws with, and its impostors' pipeline.
#[derive(Clone, Copy)]
pub(super) struct Passes {
    pub(super) opaque_kind: PassKind,
    pub(super) transparent_kind: PassKind,
    pub(super) impostor_pass: Option<(Targets, bool)>,
}

/// Everything a frame's encoding, submission and stats read.
pub(super) struct FrameState<'s> {
    pub(super) snapshot: &'s RenderSnapshot,
    pub(super) frame: RenderFrame<'s>,
    pub(super) kind: FrameKind,
    pub(super) assets: AssetUploads,
    pub(super) packing: Packing,
    pub(super) writes: Writes,
    pub(super) features: Features<'s>,
    pub(super) passes: Passes,
    /// Virtual geometry draws its clusters, and the
    /// visibility buffer resolves some of them.
    pub(super) virtual_frame: bool,
    /// Terrain draws.
    pub(super) terrain: bool,
    /// Placed objects draw.
    pub(super) scatter: bool,
    pub(super) resolving: bool,
    /// What the frame's render graph declares.
    pub(super) shape: FrameShape,
}

/// What encoding counted: the opaque passes' draws, shadow draws,
/// reflection probe faces, and the render graph's passes.
#[derive(Default)]
pub(super) struct EncodeReport {
    pub(super) draws: u32,
    pub(super) shadow_draws: u32,
    pub(super) probe_faces: u32,
    pub(super) passes: u32,
}

impl Renderer {
    pub(super) fn render_inner(
        &mut self,
        frame: RenderFrame<'_>,
        display: bool,
    ) -> Result<RenderStats, RenderError> {
        let snapshot = frame.snapshot();
        let _profile = engine_profile::span("render", "render_snapshot");
        if let Ok(failure) = self.failure.lock()
            && let Some(message) = failure.as_ref()
        {
            return Err(RenderError(message.to_string()));
        }
        let (kind, gpu) = self.plan_frame(frame, display)?;
        if self.display.suspended {
            return Ok(self.residency_stats());
        }
        self.switch_fades.advance(
            kind,
            snapshot.elapsed_seconds,
            snapshot.environment.level_of_detail.fade_seconds,
        );
        let Some(surface) = self.display.acquire(&self.device, &mut self.depth)? else {
            return Ok(self.residency_stats());
        };
        let assets = self.upload_assets(snapshot, kind);
        let packing = self.pack_instances(frame, kind);
        let profile_writes = engine_profile::span("render", "queue_writes");
        let writes = self.write_frame_data(frame, kind, &packing);
        // Profiling captures time this frame's passes from here.
        self.profiler.begin_frame(&self.device);
        let features = self.prepare_features(
            frame,
            gpu.as_ref(),
            kind,
            &packing.temporal_frame,
            &writes.lighting,
        );
        drop(profile_writes);
        let profile_encoding = engine_profile::span("render", "command_encoding");
        let view = surface
            .texture()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/frame-encoder"),
            });
        let passes = self.prepare_pipelines(kind, &packing.temporal_frame, &features);
        let (virtual_frame, resolving) = self.prepare_virtual_frame(
            snapshot,
            kind,
            &packing.temporal_frame,
            features.occlusion,
            passes.opaque_kind,
        );
        let terrain = self.prepare_terrain(
            snapshot,
            &packing.temporal_frame,
            passes.opaque_kind,
            (
                kind.gpu_mode && !features.probe_list.is_empty(),
                kind.gpu_mode,
            ),
        );
        // Lookups loop over the horizon records the terrain now binds.
        let horizon_records = self.terrain.horizons.record_count();
        self.shadows
            .set_horizon_records(&self.queue, horizon_records);
        let scatter =
            self.prepare_scatter(snapshot, passes.opaque_kind, passes.impostor_pass.is_some());
        // Water on terrains.
        let water = self.prepare_water(snapshot, passes.opaque_kind);
        // Terrain's horizon maps may have been replaced.
        self.refresh_camera_group();
        let mut shape = self.frame_shape(
            kind,
            &packing,
            &features,
            (virtual_frame, resolving, scatter),
        );
        shape.water = water;
        shape.clouds = self.prepare_cloud_composite(shape.targets);
        let mut state = FrameState {
            snapshot,
            frame,
            kind,
            assets,
            packing,
            writes,
            features,
            passes,
            virtual_frame,
            resolving,
            terrain,
            scatter,
            shape,
        };
        self.prepare_frame_groups(&state);
        // The frame's passes as a render graph: declared,
        // compiled into attachment operations, then encoded in order.
        let mut graph = std::mem::take(&mut self.graph);
        render_graph::plan(&mut graph, &state.shape);
        if let Err(unwritten) = graph.compile() {
            self.graph = graph;
            return Err(RenderError(format!(
                "The render graph's {} pass reads {:?} before any pass writes it.",
                unwritten.pass.name(),
                unwritten.resource,
            )));
        }
        let mut report = EncodeReport {
            passes: graph.kept(),
            ..EncodeReport::default()
        };
        for node in graph.kept_passes() {
            // Profiling captures time every pass whole, with whatever it
            // records.
            let stage = node.pass().stage();
            if let Some(span) = self.profiler.span(stage) {
                span.mark(&mut encoder, false);
            }
            // GPU debuggers (PIX, RenderDoc, Xcode) name each pass's work;
            // ordinary page builds make no calls for them.
            #[cfg(any(not(target_arch = "wasm32"), feature = "profiling"))]
            encoder.push_debug_group(node.pass().name());
            self.encode_pass(node, &state, &mut encoder, &view, &mut report);
            #[cfg(any(not(target_arch = "wasm32"), feature = "profiling"))]
            encoder.pop_debug_group();
            if let Some(span) = self.profiler.span(stage) {
                span.mark(&mut encoder, true);
            }
        }
        self.graph = graph;
        // ...and the command buffers submitted before the frame's, whose
        // spans must exist before the frame resolves its queries.
        for (span, present) in [
            ("pre_mips", state.assets.encoder.is_some()),
            ("pre_poses", state.writes.staged_poses.is_some()),
            ("pre_inputs", state.features.input_copies.is_some()),
            (
                "pre_environment",
                state.features.environment_encoder.is_some(),
            ),
        ] {
            if present {
                let _ = self.profiler.span(span);
            }
        }
        self.profiler.end_frame(&mut encoder);
        drop(profile_encoding);
        self.submit_frame(&mut state, surface, encoder);
        Ok(self.frame_stats(&state, &report))
    }

    /// Plans the frame's batches and shadow casters, retires assets it no
    /// longer uses and decides its kind; also returns GPU presentation's
    /// inputs.
    fn plan_frame<'f>(
        &mut self,
        frame: RenderFrame<'f>,
        display: bool,
    ) -> Result<(FrameKind, Option<engine_core::GpuPresentation<'f>>), RenderError> {
        let snapshot = frame.snapshot();
        {
            let _profile = engine_profile::span("render", "batch_plan");
            self.plan
                .prepare_frame(&frame)
                .map_err(|error| render_error("Invalid render frame", error))?;
        }
        let gpu = frame.gpu();
        let gpu_mode = gpu.is_some();
        if self.gpu_frame && !gpu_mode {
            // GPU frames wrote poses and skipped the slot-ID stream; CPU frames
            // must not trust either upload cache.
            self.instances.retained_uploads.invalidate();
            self.instances.slot_revision = None;
        }
        self.gpu_frame = gpu_mode;
        self.apply_temporal_antialiasing(snapshot.environment.temporal_antialiasing.enabled);
        let temporal = display && gpu_mode && self.temporal.is_some();
        if !temporal && let Some(pass) = &mut self.temporal {
            pass.reset();
        }
        let profile_residency = engine_profile::span("render", "validate_and_retire");
        for source in snapshot.textures.iter().chain(&snapshot.decal_textures) {
            if source.texture.width().max(source.texture.height()) > self.display.max_dimension {
                return Err(RenderError(format!(
                    "Texture {:?} exceeds the device's {}-pixel dimension limit.",
                    source.handle, self.display.max_dimension,
                )));
            }
        }
        // Shadow draw groups: CPU-prepared frames also draw
        // culled casters, whose meshes stay resident and upload with the
        // batches' meshes.
        self.shadows.plan_groups(&self.plan, frame, gpu_mode);
        // Terrains' placed objects keep their meshes resident.
        let placed: BTreeSet<MeshHandle> = snapshot
            .terrain
            .terrains
            .iter()
            .flat_map(|terrain| terrain.assets.iter().flatten())
            .flat_map(|asset| {
                std::iter::once(asset.mesh.handle)
                    .chain(asset.lods.iter().map(|(lod, _)| lod.handle))
            })
            .collect();
        if placed != self.scatter_meshes.0 {
            self.scatter_meshes = (placed, self.scatter_meshes.1 + 1);
        }
        // Supplied sources keep residency across culling; uploads below follow
        // only emitted draw batches, shadow casters and referenced textures.
        // Residency changes only with the plan's sources, the casters or the
        // placed objects' meshes.
        let residency = (
            self.plan.rebuilds(),
            self.shadows.groups_generation(),
            self.scatter_meshes.1,
        );
        if self.structure_keys.residency != Some(residency) {
            self.structure_keys.residency = Some(residency);
            let shadow_meshes: BTreeSet<MeshHandle> = self
                .shadows
                .groups()
                .iter()
                .map(|group| group.mesh)
                .collect();
            let (plan, arena, placed) = (&self.plan, &mut self.mesh_arena, &self.scatter_meshes.0);
            let resident = self.meshes.len();
            self.meshes.retain(|handle, mesh| {
                let keep = plan.contains_mesh(*handle)
                    || shadow_meshes.contains(handle)
                    || placed.contains(handle);
                if !keep && let Ok(placement) = mesh.placement {
                    arena.remove(placement);
                }
                keep
            });
            if self.meshes.len() != resident {
                self.mesh_layout += 1;
            }
            let meshes = &self.meshes;
            self.virtual_draws
                .pass
                .retain(|handle| meshes.contains_key(&handle));
        }
        self.materials.retain(&self.plan, snapshot);
        drop(profile_residency);
        Ok((
            FrameKind {
                gpu_mode,
                temporal,
                display,
                instance_count: self.plan.ordered_instances.len(),
            },
            gpu,
        ))
    }

    /// Uploads the frame's new meshes (with their levels of detail and
    /// virtual geometry) and textures, bakes new impostors, and orders the
    /// opaque draws.
    fn upload_assets(&mut self, snapshot: &RenderSnapshot, kind: FrameKind) -> AssetUploads {
        let gpu_mode = kind.gpu_mode;

        let profile_upload = engine_profile::span("render", "asset_uploads");
        let mut mesh_upload_bytes = 0_u64;
        // Every batch's and caster's mesh is resident unless the plan, the
        // casters or residency changed since the last upload.
        let uploads = (
            self.plan.rebuilds(),
            self.shadows.groups_generation(),
            self.mesh_layout,
            self.scatter_meshes.1,
        );
        let mesh_uploads = self.structure_keys.uploads != Some(uploads);
        // The plan's and casters' meshes, and terrains' placed objects'.
        let sources = self
            .plan
            .batches
            .iter()
            .map(|batch| (batch.mesh, &snapshot.meshes[batch.source_index].mesh))
            .chain(
                self.shadows
                    .groups()
                    .iter()
                    .map(|group| (group.mesh, &snapshot.meshes[group.source_index].mesh)),
            )
            .chain(
                snapshot
                    .terrain
                    .terrains
                    .iter()
                    .flat_map(|terrain| terrain.assets.iter().flatten())
                    .flat_map(|asset| {
                        std::iter::once(&asset.mesh).chain(asset.lods.iter().map(|(lod, _)| lod))
                    })
                    .map(|mesh| (mesh.handle, &mesh.mesh)),
            )
            .filter(|_| mesh_uploads);
        for (handle, source) in sources {
            if let std::collections::btree_map::Entry::Vacant(entry) = self.meshes.entry(handle) {
                self.vertex_staging.clear();
                self.vertex_staging.reserve_exact(source.vertices().len());
                self.vertex_staging
                    .extend(source.vertices().iter().map(Vertex::from));
                // The arena holds the mesh in its streams' class; the device's
                // limits alone leave a mesh buffers of its own.
                let extras = ExtraVertex::stream(source);
                let skin = SkinVertex::stream(source);
                let skin_envelopes = skin
                    .as_ref()
                    .map_or_else(Vec::new, |skin| SkinVertex::envelopes(source, skin));
                let mut streams: Vec<&[u8]> = vec![bytemuck::cast_slice(&self.vertex_staging)];
                if let Some(extras) = &extras {
                    streams.push(bytemuck::cast_slice(extras));
                }
                if let Some(skin) = &skin {
                    streams.push(bytemuck::cast_slice(skin));
                }
                // Levels of detail follow the mesh's own
                // indices in its range; placed objects' meshes
                // simplify further, for shadow maps and distant views. Draw
                // groups read their first levels alike.
                let levels = if self.scatter_meshes.0.contains(&handle) {
                    crate::scatter::LEVELS as usize - 1
                } else {
                    LOD_LEVELS - 1
                };
                let (indices, lods) = lod_chain(source, levels);
                let placement = self
                    .mesh_arena
                    .insert(
                        &self.device,
                        &self.queue,
                        &streams,
                        self.vertex_staging.len() as u32,
                        &indices,
                    )
                    .ok_or_else(|| {
                        let buffer = |label, contents: &[u8], usage| {
                            self.device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some(label),
                                    contents,
                                    usage,
                                })
                        };
                        let vertex = wgpu::BufferUsages::VERTEX;
                        OwnMesh {
                            vertices: buffer("engine/mesh-vertices", streams[0], vertex),
                            extras: extras.as_ref().map(|stream| {
                                buffer("engine/mesh-extras", bytemuck::cast_slice(stream), vertex)
                            }),
                            skin: skin.as_ref().map(|stream| {
                                buffer("engine/mesh-skin", bytemuck::cast_slice(stream), vertex)
                            }),
                            indices: buffer(
                                "engine/mesh-indices",
                                bytemuck::cast_slice(&indices),
                                wgpu::BufferUsages::INDEX,
                            ),
                        }
                    });
                self.mesh_layout += 1;
                let byte_len = source.byte_len() as u64;
                entry.insert(GpuMesh {
                    placement,
                    extended: extras.is_some(),
                    skinned: skin.is_some(),
                    skin_envelopes,
                    index_count: source.indices().len() as u32,
                    byte_len,
                    lods,
                });
                mesh_upload_bytes += byte_len;
                if let Some(geometry) = source.virtual_geometry() {
                    self.virtual_draws.pass.insert(handle, geometry);
                }
            }
        }
        if mesh_uploads {
            self.structure_keys.uploads = Some((
                self.plan.rebuilds(),
                self.shadows.groups_generation(),
                self.mesh_layout,
                self.scatter_meshes.1,
            ));
        }
        self.virtual_draws.pass.upload(
            &self.device,
            &self.queue,
            snapshot.environment.virtual_geometry.max_resident_pages,
        );
        // New textures' mip levels render from their uploaded base level in
        // a command buffer submitted ahead of this frame's.
        let mut mip_encoder = None;
        // Masked shadow casters bind their materials even when culled.
        let shadow_materials = self
            .shadows
            .groups()
            .iter()
            .filter_map(|group| group.material);
        let (texture_upload_bytes, mip_levels) = self.materials.upload(
            &self.device,
            &self.queue,
            snapshot,
            &self.plan,
            shadow_materials,
            &mut mip_encoder,
        );
        // Impostors bake after their materials' levels, the
        // plan's and then placed objects'.
        self.prepare_impostors(snapshot, gpu_mode, &mut mip_encoder);
        self.prepare_scatter_impostors(snapshot, gpu_mode, &mut mip_encoder);
        drop(profile_upload);
        self.order_draws(snapshot);
        AssetUploads {
            mesh_bytes: mesh_upload_bytes,
            texture_bytes: texture_upload_bytes,
            mip_levels,
            encoder: mip_encoder,
        }
    }

    /// Sizes instance, pose, color and palette storage for the frame, begins
    /// its temporal antialiasing, and packs the records to upload.
    fn pack_instances(&mut self, frame: RenderFrame<'_>, kind: FrameKind) -> Packing {
        let FrameKind {
            gpu_mode, temporal, ..
        } = kind;
        let snapshot = frame.snapshot();

        let profile_packing = engine_profile::span("render", "instance_buffer_and_packing");
        // Every slot, and joint palettes as many as the
        // frame's joints.
        self.instances.reserve(
            &self.device,
            snapshot.instances.len(),
            snapshot.joints.len(),
            temporal,
        );
        let temporal_frame: Option<TemporalFrame> = match self.temporal.as_mut() {
            Some(pass) if temporal => Some(pass.begin(
                &self.device,
                &self.queue,
                &frame,
                self.display.size(),
                self.instances.generation,
            )),
            _ => None,
        };
        // Pose/color buffers retain core slot positions. Only compact IDs follow
        // changing visibility and mesh/texture grouping, so order changes never
        // shift payload records or force otherwise unchanged colors to upload.
        let slot_key = Some((frame.identity(), frame.structure_revision()));
        let write_slots = !gpu_mode
            && self
                .instances
                .stage_slots(&self.device, &self.plan.ordered_instances, slot_key);
        if self.instances.color_staging.capacity() < self.instances.capacity {
            self.instances
                .color_staging
                .reserve_exact(self.instances.capacity - self.instances.color_staging.len());
        }
        let prepared = self
            .instances
            .retained_uploads
            .prepare(&frame, &mut self.instances.color_staging);
        drop(profile_packing);
        Packing {
            temporal_frame,
            write_slots,
            slot_key,
            repacked: prepared.repacked,
            poses_repacked: prepared.poses_repacked,
            colors_repacked: prepared.colors_repacked,
            full_upload: prepared.full_upload,
            fallback: prepared.fallback,
        }
    }

    /// Writes the camera and lighting uniforms and the frame's changed
    /// poses, colors, palettes and slot IDs.
    fn write_frame_data(
        &mut self,
        frame: RenderFrame<'_>,
        kind: FrameKind,
        packing: &Packing,
    ) -> Writes {
        let snapshot = frame.snapshot();
        let gpu_mode = kind.gpu_mode;
        let temporal_frame = &packing.temporal_frame;
        let (write_slots, slot_key) = (packing.write_slots, packing.slot_key);

        let mut camera = temporal_frame.as_ref().map_or_else(
            || {
                CameraUniform::exact(
                    snapshot.view_projection,
                    snapshot.camera_position,
                    snapshot.elapsed_seconds,
                )
            },
            |frame| frame.camera,
        );
        camera.options[1] =
            crate::gpu_data::pixel_angle(snapshot.view_projection, self.display.size().1);
        // The wind placed objects sway in, and the time they
        // swayed at last frame for temporal frames' motion.
        let seconds = snapshot.elapsed_seconds;
        let previous = self.last_seconds.replace(seconds).unwrap_or(seconds);
        let wind = snapshot.environment.wind;
        camera.wind = [
            wind.velocity[0],
            wind.velocity[1],
            wind.gusts,
            if temporal_frame.is_some() {
                previous
            } else {
                seconds
            },
        ];
        let mut camera_upload_bytes = 0_u64;
        if self.uploads.camera_needs_write(&camera) {
            self.queue
                .write_buffer(&self.camera.buffer, 0, bytemuck::bytes_of(&camera));
            self.uploads.commit_camera(camera);
            camera_upload_bytes = size_of::<CameraUniform>() as u64;
        }
        let mut lighting = LightingUniform::new(
            &snapshot.environment.atmosphere,
            &snapshot.environment.bloom,
            &snapshot.environment.reflections,
            &snapshot.environment.refraction,
            &snapshot.environment.sky,
        );
        // Under the sea, its water veils everything.
        lighting.underwater = crate::water::underwater(
            &snapshot.environment,
            snapshot.camera_position,
            snapshot.presented_seconds,
        );
        let ocean = snapshot.environment.ocean;
        if ocean.enabled {
            lighting.sea = [ocean.level, ocean.clarity, 1.0, 0.0];
        }
        // Clouds' shadows on everything the sun lights.
        (lighting.cloud_shadow, lighting.cloud_layer) =
            crate::clouds::shadow_lighting(&snapshot.environment.clouds, snapshot.camera_position);
        // What placed objects cover around the camera: the
        // map shading reads, drawn this frame when it no longer holds the
        // camera's surroundings or the objects changed.
        if let Some((area, far, strength)) = self.occupancy.prepare(
            &self.device,
            &self.queue,
            &snapshot.environment.occupancy,
            snapshot.camera_position,
            (
                self.scatter_draws.covers(),
                self.scatter.objects(),
                self.scatter.version(),
            ),
        ) {
            lighting.occupancy = [area.center[0], area.center[1], 1.0 / area.size, strength];
            lighting.occupancy_heights = [
                area.reference,
                area.resolution as f32,
                area.reach,
                far.map_or(0.0, |far| far.reference),
            ];
            // The far map.
            if let Some(far) = far {
                lighting.occupancy_far = [far.center[0], far.center[1], 1.0 / far.size, far.reach];
            }
        }
        camera_upload_bytes += self.camera.write_lighting(&self.queue, lighting);
        let mut color_upload_bytes = 0_u64;
        let mut slot_upload_bytes = 0_u64;
        // Large pose uploads copy into a persistently mapped staging buffer
        // and a GPU copy submitted before this frame's draws.
        let mut staged_poses = None;
        // GPU-presented frames write poses on the GPU.
        let pose_ranges = if gpu_mode {
            &[][..]
        } else {
            self.instances.retained_uploads.pose_ranges()
        };
        let instance_write_ranges = pose_ranges.len()
            + self.instances.retained_uploads.color_ranges().len()
            + usize::from(write_slots);
        // Core keeps each slot's pose in this upload layout.
        let pose_rows: &[u8] = bytemuck::cast_slice(frame.pose_rows());
        let stride = size_of::<PoseData>();
        let mut pose_upload_bytes: u64 = pose_ranges
            .iter()
            .map(|range| ((range.end - range.start) * stride) as u64)
            .sum();
        let mut staged = false;
        if pose_upload_bytes as usize >= staging::MIN_STAGED_BYTES {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine/pose-staging-copies"),
                });
            let outcome = self.instances.pose_uploads.record(
                &self.device,
                &mut encoder,
                &self.instances.buffer,
                pose_rows,
                stride,
                pose_ranges,
            );
            staged = outcome == staging::Staged::Recorded;
            engine_profile::counter("render.pose_staged", f64::from(u8::from(staged)));
            engine_profile::counter(
                "render.pose_staging_unavailable",
                f64::from(u8::from(!staged)),
            );
            staged_poses = staged.then(|| encoder.finish());
        }
        if !staged {
            for range in pose_ranges {
                self.queue.write_buffer(
                    &self.instances.buffer,
                    (range.start * stride) as u64,
                    &pose_rows[range.start * stride..range.end * stride],
                );
            }
        }
        for range in self.instances.retained_uploads.color_ranges() {
            let records = &self.instances.color_staging[range.start..range.end];
            self.queue.write_buffer(
                self.instances.colors(),
                (range.start * size_of::<ColorData>()) as u64,
                bytemuck::cast_slice(records),
            );
            color_upload_bytes += std::mem::size_of_val(records) as u64;
        }
        // CPU-prepared frames upload skinned slots' palettes;
        // GPU-presented ones compute them.
        if !gpu_mode && !snapshot.joints.is_empty() {
            self.instances.palette_staging.clear();
            self.instances
                .palette_staging
                .extend(snapshot.joints.iter().map(|joint| joint.matrix));
            self.queue.write_buffer(
                &self.instances.palettes[0],
                0,
                bytemuck::cast_slice(&self.instances.palette_staging),
            );
            pose_upload_bytes += std::mem::size_of_val(&self.instances.palette_staging[..]) as u64;
        }
        if write_slots {
            self.queue.write_buffer(
                self.instances.slots(),
                0,
                bytemuck::cast_slice(&self.instances.slot_staging),
            );
            slot_upload_bytes = (self.instances.slot_staging.len() * size_of::<u32>()) as u64;
        }
        let instance_upload_bytes = pose_upload_bytes + color_upload_bytes + slot_upload_bytes;
        self.instances.retained_uploads.commit();
        self.instances.slot_revision = slot_key;
        Writes {
            lighting,
            camera_bytes: camera_upload_bytes,
            instance_bytes: instance_upload_bytes,
            pose_bytes: pose_upload_bytes,
            color_bytes: color_upload_bytes,
            slot_bytes: slot_upload_bytes,
            ranges: instance_write_ranges,
            staged_poses,
        }
    }

    /// Prepares every feature the frame runs, in order: GPU presentation
    /// with occlusion culling, shadows, environment maps and reflection
    /// probes, local lights, screen effects, transparency and particles,
    /// and decals.
    fn prepare_features<'s>(
        &mut self,
        frame: RenderFrame<'s>,
        gpu: Option<&engine_core::GpuPresentation<'_>>,
        kind: FrameKind,
        temporal_frame: &Option<TemporalFrame>,
        lighting: &LightingUniform,
    ) -> Features<'s> {
        let snapshot = frame.snapshot();
        let PresentationPrep {
            input_copies,
            occlusion,
            gpu_prepared,
        } = self.prepare_presentation(frame, gpu, kind, temporal_frame);
        let shadow_frame = self.prepare_shadows(snapshot, kind, temporal_frame);
        // Clouds: this frame's march of their dome, which the
        // sky and its environment map show; exact frames march all of it.
        self.clouds.prepare(
            &self.device,
            &self.queue,
            (
                &snapshot.environment.clouds,
                snapshot.camera_position,
                snapshot.presented_seconds,
            ),
            lighting,
            !kind.gpu_mode,
        );
        // And their march in the view, after the opaque
        // scene and water.
        self.clouds.prepare_view(
            &self.device,
            &self.queue,
            (
                &snapshot.environment.clouds,
                snapshot.camera_position,
                snapshot.presented_seconds,
            ),
            snapshot.view_projection,
            [self.depth.texture.width(), self.depth.texture.height()],
            lighting,
        );
        self.clouds
            .prepare_sky_group(&self.device, self.sky.bindings());
        self.environment
            .set_clouds(&self.device, self.clouds.dome());
        let sky_encoder = self.prepare_sky(snapshot);
        let (environment_encoder, probe_list, dropped_probes) =
            self.prepare_environment(snapshot, lighting);
        let light_frame = self.prepare_lights(frame, kind, temporal_frame);
        let (post_frame, refraction) = self.prepare_post(snapshot, temporal_frame);
        let (transparent_draws, particle_frame, transparency) = self.prepare_transparency(
            snapshot,
            (frame.identity(), frame.emitters_revision()),
            temporal_frame,
            post_frame,
        );
        let decal_frame = self.prepare_decals(snapshot, temporal_frame, &light_frame, post_frame);
        Features {
            input_copies,
            occlusion,
            gpu_prepared,
            shadow_frame,
            sky_encoder,
            environment_encoder,
            probe_list,
            dropped_probes,
            light_frame,
            post_frame,
            refraction,
            transparent_draws,
            particle_frame,
            transparency,
            decal_frame,
        }
    }

    /// Occlusion culling's depth pyramid and GPU presentation's inputs;
    /// returns the input copies to submit first, the
    /// occlusion viewport and levels, and what presentation prepared.
    fn prepare_presentation(
        &mut self,
        frame: RenderFrame<'_>,
        gpu: Option<&engine_core::GpuPresentation<'_>>,
        kind: FrameKind,
        temporal_frame: &Option<TemporalFrame>,
    ) -> PresentationPrep {
        let gpu_mode = kind.gpu_mode;
        let snapshot = frame.snapshot();

        let mut input_copies = None;
        // Occlusion culling: the depth pyramid of this frame's
        // depth buffer, sized before presentation plans two phases; turning
        // it off releases the pyramid.
        let occlusion_enabled = snapshot.environment.occlusion_culling.enabled;
        if !occlusion_enabled {
            self.culling.pyramid = None;
        }
        let occlusion = if gpu_mode && occlusion_enabled {
            let device = &self.device;
            let pyramid = self
                .culling
                .pyramid
                .get_or_insert_with(|| DepthPyramid::new(device, Reduction::Farthest));
            let (width, height) = self.display.size();
            let levels = pyramid.prepare(
                device,
                &self.depth.view,
                self.depth.generation,
                (width, height),
            );
            Some((width, height, levels))
        } else {
            None
        };
        // Batches drawing virtual geometry instead of their
        // fallback meshes, by the arena and whether it draws.
        let virtual_enabled = snapshot.environment.virtual_geometry.enabled;
        let arena = self
            .virtual_draws
            .pass
            .arena()
            .map_or(0, |(_, generation)| generation);
        let virtual_key = arena * 2 + u64::from(virtual_enabled);
        let gpu_prepared = if let Some(gpu) = gpu {
            let meshes = &self.meshes;
            let arena = &self.mesh_arena;
            let shadows = &self.shadows;
            let virtual_meshes = &self.virtual_draws.pass;
            let impostor_records = &self.impostors.records;
            let batch_meta = |index, batch: &DrawBatch| {
                let mesh = &snapshot.meshes[batch.source_index].mesh;
                let gpu_mesh = &meshes[&batch.mesh];
                let draw = gpu_mesh.draw(arena);
                let virtual_geometry = virtual_enabled
                    && !batch.transparent
                    && virtual_meshes.mesh_index(batch.mesh).is_some();
                let levels = lod_levels(batch, gpu_mesh, virtual_geometry);
                BatchMeta {
                    index_count: draw.index_count,
                    first_index: draw.first_index,
                    base_vertex: draw.base_vertex,
                    bounds: mesh.bounds(),
                    group: shadows.batch_group(index),
                    joints: mesh.joint_count(),
                    skin_envelopes: &gpu_mesh.skin_envelopes,
                    virtual_geometry,
                    lods: gpu_mesh.lods[..levels - 1]
                        .iter()
                        .map(|&(offset, count, error)| (draw.first_index + offset, count, error))
                        .collect(),
                    impostor: impostor_records
                        .get(index)
                        .map_or(0.0, ImpostorRecord::error),
                }
            };
            let records = RecordsKey {
                shadow_groups: !shadows.groups().is_empty(),
                meshes: virtual_key ^ (self.mesh_layout << 32),
                // Batch records carry impostor errors: rebuilt with the
                // impostors, and when turning them on or off.
                impostors: if snapshot.environment.level_of_detail.impostors {
                    self.impostors.atlas.generation() + 1
                } else {
                    0
                },
            };
            let view = CullingView {
                view_projection: temporal_frame
                    .as_ref()
                    .map_or(snapshot.view_projection, |temporal| temporal.rasterized),
                eye: snapshot.camera_position,
                viewport: self.display.size(),
                motion: temporal_frame
                    .as_ref()
                    .is_some_and(|temporal| temporal.motion_valid),
                occlusion: occlusion.map(|(_, _, levels)| levels),
                subpixel: snapshot.environment.subpixel_culling.enabled,
                lod_error: if snapshot.environment.level_of_detail.enabled {
                    snapshot.environment.level_of_detail.pixel_error
                } else {
                    0.0
                },
                impostors: snapshot.environment.level_of_detail.impostors,
                // Reflection probes capture still slots,
                // counted while captures wait.
                stillness: (snapshot.environment.reflection_probes.enabled
                    && !snapshot.reflection_probes.is_empty())
                .then(|| self.probes.slots.waiting()),
                fade: self.switch_fades.share,
            };
            // Temporal frames alternate pose buffers; the other keeps the
            // previous frame's poses.
            let parity = temporal_frame.as_ref().map_or(0, |frame| frame.parity);
            let (ring, poses) = self.instances.presentation_inputs(parity);
            Some(self.gpu_presentation.prepare(
                &self.device,
                &self.queue,
                Staging {
                    ring,
                    copies: &mut input_copies,
                },
                &PresentationInputs {
                    frame: &frame,
                    gpu,
                    plan: &self.plan,
                    batch_meta: &batch_meta,
                    records,
                    poses,
                },
                &view,
                &self.profiler,
            ))
        } else {
            None
        };
        PresentationPrep {
            input_copies,
            occlusion,
            gpu_prepared,
        }
    }

    /// The frame's shadow maps, selecting from its poses.
    fn prepare_shadows(
        &mut self,
        snapshot: &RenderSnapshot,
        kind: FrameKind,
        temporal_frame: &Option<TemporalFrame>,
    ) -> Option<ShadowFrame> {
        let gpu_mode = kind.gpu_mode;

        // Shadow maps select from this frame's poses.
        let parity = temporal_frame.as_ref().map_or(0, |frame| frame.parity);
        let meshes = &self.meshes;
        let arena = &self.mesh_arena;
        let placement = |mesh| {
            meshes.get(&mesh).map_or(
                CasterDraw {
                    index_count: 0,
                    first_index: 0,
                    base_vertex: 0,
                    lods: Vec::new(),
                },
                |mesh| {
                    let draw = mesh.draw(arena);
                    // Levels of detail of meshes without
                    // joints, which move their vertices.
                    let lods = if mesh.skinned || !snapshot.environment.level_of_detail.enabled {
                        Vec::new()
                    } else {
                        mesh.lods
                            .iter()
                            .map(|&(offset, count, error)| {
                                (draw.first_index + offset, count, error)
                            })
                            .collect()
                    };
                    CasterDraw {
                        index_count: draw.index_count,
                        first_index: draw.first_index,
                        base_vertex: draw.base_vertex,
                        lods,
                    }
                },
            )
        };
        self.shadows.prepare(
            &self.device,
            &self.queue,
            &ShadowInputs {
                snapshot,
                presented: gpu_mode.then(|| self.gpu_presentation.shadow_inputs()),
                poses: self.instances.frame_poses(parity),
                slot_capacity: self.instances.capacity,
                mesh_layout: self.mesh_layout,
                placement: &placement,
                terrain: snapshot
                    .terrain
                    .terrains
                    .iter()
                    .any(|terrain| !terrain.patches.is_empty()),
            },
        )
    }

    /// Environment maps and reflection probes, and the camera
    /// group binding them; returns the environment's encoder, the probes the
    /// frame keeps and those beyond the GPU's arrays.
    /// The physically based sky's work this frame: its
    /// tables as they need building, then the light it gives, copied over
    /// the authored atmosphere's colors in the lighting uniform and the
    /// clouds' march's uniform.
    fn prepare_sky(&mut self, snapshot: &RenderSnapshot) -> Option<wgpu::CommandEncoder> {
        use crate::sky::{
            DERIVED_AIR, DERIVED_GLOW, DERIVED_GROUND, DERIVED_HORIZON, DERIVED_SUN, DERIVED_ZENITH,
        };
        use std::mem::offset_of;
        let lighting = self.camera.lighting_buffer();
        let at = |offset: usize| offset as u64;
        let mut copies = vec![
            (
                DERIVED_SUN,
                lighting,
                at(offset_of!(LightingUniform, sun_color)),
                12,
            ),
            (
                DERIVED_ZENITH,
                lighting,
                at(offset_of!(LightingUniform, zenith)),
                12,
            ),
            // With the exposure adapted to the sky's light.
            (
                DERIVED_HORIZON,
                lighting,
                at(offset_of!(LightingUniform, horizon)),
                16,
            ),
            (
                DERIVED_GROUND,
                lighting,
                at(offset_of!(LightingUniform, ground)),
                12,
            ),
            (
                DERIVED_AIR,
                lighting,
                at(offset_of!(LightingUniform, aerial_color)),
                12,
            ),
            // The glow rides in the aerial vector's w.
            (
                DERIVED_GLOW,
                lighting,
                at(offset_of!(LightingUniform, aerial)) + 12,
                4,
            ),
        ];
        copies.extend(self.clouds.sky_copies());
        self.sky.encode(
            &self.device,
            &self.queue,
            (
                &snapshot.environment.sky,
                snapshot.environment.atmosphere.sun_direction,
                snapshot.camera_position,
                snapshot.environment.atmosphere.exposure,
                snapshot.presented_seconds,
            ),
            &copies,
            &self.profiler,
        )
    }

    fn prepare_environment<'s>(
        &mut self,
        snapshot: &'s RenderSnapshot,
        lighting: &LightingUniform,
    ) -> (
        Option<wgpu::CommandEncoder>,
        &'s [engine_core::RenderReflectionProbe],
        u64,
    ) {
        // Environment maps: as many cubes as the reflection
        // probes kept need (the nearest first, as many as the GPU's texture
        // arrays hold), the sky's redrawn when it changes, then the probes'
        // cubes assigned and their grid uploaded.
        let probe_settings = snapshot.environment.reflection_probes;
        let probe_list: &[engine_core::RenderReflectionProbe] = if probe_settings.enabled {
            &snapshot.reflection_probes
        } else {
            &[]
        };
        let most_cubes = (self.display.max_layers / FACES as u32).max(1);
        let wanted_cubes = 1 + (probe_list.len() as u32).min(most_cubes - 1);
        let cubes = if wanted_cubes == 1 {
            1
        } else if wanted_cubes > self.environment.cubes() {
            wanted_cubes.next_power_of_two().min(most_cubes)
        } else {
            self.environment.cubes()
        };
        // Frames whose maps and sky stand record nothing here.
        let keys = (self.clouds.environment_key(), self.sky.environment_key());
        let mut environment_encoder = self
            .environment
            .needs_encoding(probe_settings.resolution, cubes, lighting, keys)
            .then(|| {
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("engine/environment-encoder"),
                    })
            });
        let (lost, sky_changed) = match environment_encoder.as_mut() {
            Some(encoder) => (
                self.environment
                    .ensure(&self.device, encoder, probe_settings.resolution, cubes),
                self.environment.encode_sky(encoder, lighting, keys).1,
            ),
            None => (false, false),
        };
        self.probes
            .slots
            .plan(probe_list, (cubes - 1) as usize, lost, sky_changed);
        self.probes.upload(&self.device, &self.queue);
        let dropped_probes = snapshot.dropped_reflection_probes as u64
            + (probe_list.len() as u64).saturating_sub(u64::from(most_cubes - 1));
        self.refresh_camera_group();
        (environment_encoder, probe_list, dropped_probes)
    }

    /// Rebuilds the main passes' first group when what it binds was
    /// replaced.
    fn refresh_camera_group(&mut self) {
        self.camera.refresh(
            &self.device,
            &camera::GroupSources {
                shadows: &self.shadows,
                environment: &self.environment,
                probes: &self.probes,
                horizons: &self.terrain.horizons,
                occupancy: &self.occupancy,
            },
        );
    }

    /// Local lights, binned from the frame's poses.
    fn prepare_lights(
        &mut self,
        frame: RenderFrame<'_>,
        kind: FrameKind,
        temporal_frame: &Option<TemporalFrame>,
    ) -> Option<LightFrame> {
        let temporal = kind.temporal;
        let snapshot = frame.snapshot();

        // Local lights: binned on the GPU from this frame's
        // poses, so slot lights follow GPU-presented meshes.
        if snapshot.environment.local_lights.active() && !snapshot.lights.is_empty() {
            let device = &self.device;
            let lights = self.lights.get_or_insert_with(|| LightPass::new(device));
            if temporal && let Some(pass) = self.temporal.as_mut() {
                pass.ensure_lit_layout(
                    device,
                    [
                        &self.camera.layout,
                        self.materials.layout(),
                        lights.fragment_layout(),
                    ],
                );
            }
            let parity = temporal_frame.as_ref().map_or(0, |frame| frame.parity);
            let poses = self.instances.frame_poses(parity);
            lights.prepare(
                device,
                &self.queue,
                snapshot,
                self.display.size(),
                poses,
                (frame.identity(), frame.lights_revision()),
            )
        } else {
            None
        }
    }

    /// Screen effects and refraction's targets;
    /// returns the effects and whether glass refracts.
    fn prepare_post(
        &mut self,
        snapshot: &RenderSnapshot,
        temporal_frame: &Option<TemporalFrame>,
    ) -> (Option<PostFrame>, bool) {
        // Screen effects; temporal frames apply them in their
        // resolve, whose history averages the occlusion noise.
        let effects = snapshot.environment.ambient_occlusion.sanitized().enabled()
            || snapshot.environment.bloom.sanitized().enabled()
            || snapshot.environment.reflections.sanitized().enabled()
            || (snapshot.environment.sun_shafts.sanitized().enabled()
                && snapshot.environment.shadows.enabled)
            || snapshot.environment.contact_shadows.sanitized().traces();
        // Glass refracts the opaque color of the scene target:
        // frames drawing it render into the target.
        let refraction = snapshot.environment.refraction.enabled()
            && self.plan.batches.iter().any(|batch| {
                batch.transparent && self.materials.binding(batch.material).1.transmission()
            });
        let post_frame = if temporal_frame.is_some() || effects || refraction {
            let format = self.display.format();
            let (device, camera_layout) = (&self.device, &self.camera.layout);
            let post = self
                .post
                .get_or_insert_with(|| PostPass::new(device, format, camera_layout));
            Some(post.begin(
                &self.device,
                &self.queue,
                snapshot,
                self.display.size(),
                &self.depth.view,
                self.depth.generation,
                temporal_frame.is_some(),
            ))
        } else {
            // Frames without effects release their targets.
            if let Some(post) = self.post.as_mut() {
                post.release_targets();
            }
            None
        };
        if let (Some(_), Some(pass), Some(post)) =
            (&temporal_frame, self.temporal.as_mut(), self.post.as_ref())
        {
            pass.prepare_resolve(&self.device, post);
        }
        (post_frame, refraction)
    }

    /// Transparent surfaces and particles; returns whether
    /// transparent batches draw, the particles, and whether either does.
    /// `emitters` is the frame's stream identity and emitters revision.
    fn prepare_transparency(
        &mut self,
        snapshot: &RenderSnapshot,
        emitters: (u64, u64),
        temporal_frame: &Option<TemporalFrame>,
        post_frame: Option<PostFrame>,
    ) -> (bool, Option<ParticleFrame>, bool) {
        // Transparent surfaces and particles. Particles
        // sample the depth buffer for their soft edges.
        let transparent_draws = self.plan.batches.iter().any(|batch| batch.transparent);
        let particle_frame =
            if snapshot.environment.particles.enabled && !snapshot.emitters.is_empty() {
                let device = &self.device;
                let (camera_layout, format) = (&self.camera.layout, self.display.format());
                let particles = self.particles.get_or_insert_with(|| {
                    ParticlePass::new(
                        device,
                        camera_layout,
                        format,
                        DEPTH_FORMAT,
                        &accumulation_targets(),
                    )
                });
                let prepared = particles.prepare(
                    device,
                    &self.queue,
                    snapshot,
                    emitters,
                    self.display.size(),
                    &self.depth.view,
                    self.depth.generation,
                );
                if prepared.is_some() {
                    particles.ensure_additive(device, post_frame.is_some());
                }
                prepared
            } else {
                None
            };
        let transparency = transparent_draws || particle_frame.is_some();
        if transparency {
            let (device, format) = (&self.device, self.display.format());
            let pass = self
                .transparent
                .get_or_insert_with(|| TransparentPass::new(device, format, DEPTH_FORMAT));
            pass.begin(device, self.display.size());
            pass.ensure_composite(device, post_frame.is_some());
            if transparent_draws
                && temporal_frame
                    .as_ref()
                    .is_some_and(|frame| frame.parity == 1)
            {
                self.instances.prepare_alternate_group(device);
            }
        } else if let Some(pass) = self.transparent.as_mut() {
            pass.release_targets();
        }
        (transparent_draws, particle_frame, transparency)
    }

    /// Decals over the opaque surfaces.
    fn prepare_decals(
        &mut self,
        snapshot: &RenderSnapshot,
        temporal_frame: &Option<TemporalFrame>,
        light_frame: &Option<LightFrame>,
        post_frame: Option<PostFrame>,
    ) -> Option<DecalFrame> {
        // Decals cover the opaque surfaces the depth buffer
        // holds and read their displayed poses from the joint palette.
        let decal_lights = light_frame
            .as_ref()
            .and(self.lights.as_ref())
            .map(LightPass::fragment_layout);
        if snapshot.environment.decals.enabled && !snapshot.decals.is_empty() {
            let (device, format) = (&self.device, self.display.format());
            let decals = self
                .decals
                .get_or_insert_with(|| DecalPass::new(device, format));
            let parity = temporal_frame.as_ref().map_or(0, |frame| frame.parity);
            let prepared = decals.prepare(
                device,
                &self.queue,
                snapshot,
                (
                    &self.instances.palettes[parity],
                    (self.instances.generation, parity),
                ),
                &self.depth.view,
                self.depth.generation,
            );
            if prepared.is_some() {
                decals.ensure_pipelines(
                    device,
                    &self.mesh_shader,
                    [&self.camera.layout, self.materials.layout()],
                    decal_lights,
                    post_frame.is_some(),
                );
            }
            prepared
        } else {
            None
        }
    }

    /// The pipelines of the material variants the frame's
    /// mesh passes draw, reflection probe captures' and the
    /// impostors'.
    fn prepare_pipelines(
        &mut self,
        kind: FrameKind,
        temporal_frame: &Option<TemporalFrame>,
        features: &Features<'_>,
    ) -> Passes {
        let gpu_mode = kind.gpu_mode;
        let (light_frame, post_frame, probe_list) = (
            &features.light_frame,
            features.post_frame,
            features.probe_list,
        );

        // Pipelines of the material variants this frame draws.
        let lit_frame = light_frame.is_some() && self.lights.is_some();
        let targets = self.frame_targets(temporal_frame, post_frame);
        let opaque_kind = PassKind::opaque(targets, lit_frame);
        let transparent_kind = PassKind::transparent(lit_frame);
        self.ensure_variants(opaque_kind, transparent_kind);
        // Reflection probes capture once their slots stand
        // still: their pipelines are prepared with the first frame that has
        // probes, not in the middle of play.
        if gpu_mode && !probe_list.is_empty() {
            self.ensure_probe_pipelines(PassKind::probe(lit_frame));
        }
        // Impostors: this frame's records, group and pipeline.
        let impostor_pass = if gpu_mode
            && self
                .impostors
                .records
                .iter()
                .chain(self.scatter_draws.impostors())
                .any(|record| record.sphere[3] > 0.0)
        {
            self.prepare_impostor_draws(opaque_kind)
        } else {
            None
        };
        // Placed objects' impostors cast cards.
        if impostor_pass.is_some()
            && self
                .scatter_draws
                .impostors()
                .iter()
                .any(|record| record.sphere[3] > 0.0)
        {
            self.shadows
                .prepare_impostor_casters(&self.device, self.impostors.pipelines.draw_layout());
        }
        Passes {
            opaque_kind,
            transparent_kind,
            impostor_pass,
        }
    }

    /// Virtual geometry's selection for the frame and the
    /// visibility buffer's bins; returns whether clusters
    /// draw and whether the visibility buffer resolves.
    fn prepare_virtual_frame(
        &mut self,
        snapshot: &RenderSnapshot,
        kind: FrameKind,
        temporal_frame: &Option<TemporalFrame>,
        occlusion: Option<(u32, u32, u32)>,
        opaque_kind: PassKind,
    ) -> (bool, bool) {
        let FrameKind {
            gpu_mode,
            instance_count,
            ..
        } = kind;

        // Virtual geometry: the clusters the visible
        // instances of virtual meshes draw this frame.
        let odd = temporal_frame
            .as_ref()
            .is_some_and(|frame| frame.parity == 1);
        let rasterized = temporal_frame
            .as_ref()
            .map_or(snapshot.view_projection, |temporal| temporal.rasterized);
        let virtual_frame = self.prepare_virtual(
            snapshot,
            gpu_mode,
            odd,
            (rasterized, occlusion.map(|(_, _, levels)| levels)),
        );
        if virtual_frame {
            self.ensure_virtual_pipelines(opaque_kind);
            self.ensure_virtual_pose_groups();
        }
        // The visibility buffer resolves the camera's bins it
        // can shade; the others draw forward.
        let resolving =
            virtual_frame && instance_count > 0 && self.prepare_visibility(snapshot, opaque_kind);
        if resolving {
            self.ensure_visibility_pipelines(opaque_kind);
            self.ensure_visibility_groups();
        }
        (virtual_frame, resolving)
    }

    /// Where the frame's main passes draw: the temporal targets of temporal
    /// frames, the screen effects' targets of frames with
    /// effects, otherwise the canvas.
    fn frame_targets(
        &self,
        temporal_frame: &Option<TemporalFrame>,
        post_frame: Option<PostFrame>,
    ) -> Targets {
        match (temporal_frame, post_frame.filter(|_| self.post.is_some())) {
            (Some(_), Some(_)) if self.temporal.is_some() => Targets::Temporal,
            (_, Some(_)) => Targets::Post,
            (_, None) => Targets::Canvas,
        }
    }

    /// The frame's shape: which passes its render graph
    /// declares, from what each feature prepared.
    fn frame_shape(
        &self,
        kind: FrameKind,
        packing: &Packing,
        features: &Features<'_>,
        (virtual_frame, resolving, scatter): (bool, bool, bool),
    ) -> FrameShape {
        let post = features.post_frame.filter(|_| self.post.is_some());
        let targets = self.frame_targets(&packing.temporal_frame, features.post_frame);
        // Occlusion culling draws in two phases: what the
        // last test left visible, then, after the depth pyramid of those
        // depths and the test, what the test newly finds visible.
        let phases = if kind.gpu_mode && self.gpu_presentation.occlusion() {
            2
        } else {
            1
        };
        FrameShape {
            gpu: kind.gpu_mode,
            targets,
            canvas_clear: canvas_clear(),
            phases,
            shadows: features.shadow_frame.is_some(),
            scatter,
            occupancy: self.occupancy.redrawing(),
            clusters: virtual_frame,
            shadow_clusters: virtual_frame && !self.virtual_draws.shadow_bins.is_empty(),
            resolves: [
                resolving,
                resolving && phases == 2 && self.virtual_draws.phase_two.is_some(),
            ],
            lights: features.light_frame.is_some() && self.lights.is_some(),
            decals: features.decal_frame.is_some() && self.decals.is_some(),
            water: false,
            clouds: false,
            transparency: features.transparency && self.transparent.is_some(),
            refraction: features.refraction && post.is_some(),
            effects: post.map_or(Effects::default(), |frame| Effects {
                occlusion: frame.occlusion,
                bloom: frame.bloom,
                reflections: frame.reflections,
                shafts: frame.shafts,
                contact: frame.contact,
            }),
            probes: kind.gpu_mode && !features.probe_list.is_empty(),
        }
    }

    /// Temporal antialiasing's pose groups and the camera
    /// group of glass refracting the opaque color.
    fn prepare_frame_groups(&mut self, state: &FrameState<'_>) {
        let (temporal_frame, refraction) =
            (&state.packing.temporal_frame, state.features.refraction);

        if let (Some(_), Some(pass)) = (&temporal_frame, self.temporal.as_mut()) {
            pass.prepare_pose_groups(
                &self.device,
                self.instances.generation,
                [&self.instances.buffer, self.instances.poses(1)],
                self.instances.colors(),
                [&self.instances.palettes[0], &self.instances.palettes[1]],
                self.instances.grounds(),
            );
        }
        if refraction && let Some(post) = self.post.as_ref() {
            let (scene, ..) = post.targets();
            self.camera.prepare_refraction(
                &self.device,
                (scene, post.generation()),
                &camera::GroupSources {
                    shadows: &self.shadows,
                    environment: &self.environment,
                    probes: &self.probes,
                    horizons: &self.terrain.horizons,
                    occupancy: &self.occupancy,
                },
            );
        }
    }

    /// Encodes `node`, a pass the frame's render graph kept,
    /// into `encoder`; `view` is the canvas's.
    fn encode_pass(
        &mut self,
        node: &Node,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        report: &mut EncodeReport,
    ) {
        match node.pass() {
            Pass::Presentation => self.encode_presentation(encoder),
            // Placed objects cull before the shadow maps and the first
            // opaque phase draw them.
            Pass::Scatter => self.scatter.encode(encoder, &self.profiler),
            Pass::Occupancy => {
                self.occupancy.encode(
                    &self.device,
                    encoder,
                    self.scatter.slot_buffers(),
                    &self.profiler,
                );
            }
            Pass::ShadowSelect => self.encode_shadow_selection(state, encoder),
            Pass::VirtualSelect => self.encode_cluster_selection(encoder),
            Pass::ShadowMaps => self.encode_shadow_maps(state, encoder, report),
            Pass::Lights => self.encode_light_binning(state, encoder),
            Pass::Opaque { phase } => {
                // Tracks press into the deformation map, and
                // ground cover grows on the terrain, before the first phase
                // draws them.
                if phase == 0 {
                    self.clouds.encode(
                        &self.device,
                        encoder,
                        &self.camera.cloud_shadows,
                        &self.profiler,
                    );
                    self.terrain.deformation.encode(
                        &self.device,
                        &self.queue,
                        encoder,
                        &self.profiler,
                    );
                    // The runtime virtual texture's pages bake before the
                    // resolves read them.
                    self.terrain.encode_vt_bakes(
                        &self.device,
                        encoder,
                        &self.mesh_shader,
                        (&self.camera.terrain_layout, &self.camera.terrain_group),
                    );
                }
                if phase == 0 && state.terrain {
                    self.ground_cover
                        .encode(&self.device, encoder, &self.profiler);
                }
                let draws = self.encode_opaque(state, encoder, view, node, usize::from(phase));
                // The pages the first phase's terrain resolves wanted of the
                // runtime virtual texture.
                if phase == 0
                    && self.terrain_deferred(state)
                    && let Some(ids) = &self.terrain_ids
                {
                    let size = self.depth.texture.size();
                    self.terrain.encode_vt_feedback(
                        &self.device,
                        encoder,
                        &self.mesh_shader,
                        &self.camera.buffer,
                        (&ids.view, (size.width, size.height)),
                    );
                }
                // Phase two issues phase one's draws again for
                // the instances its test revealed: count each once.
                if phase == 0 {
                    report.draws += draws;
                }
            }
            Pass::VisibilityIds { phase } => {
                self.encode_visibility_ids(state, encoder, view, node, usize::from(phase));
            }
            Pass::Classify { .. } => self.encode_visibility_tiles(encoder),
            Pass::Resolve { phase } => {
                self.encode_visibility_resolve(state, encoder, view, node, usize::from(phase));
            }
            Pass::OcclusionTest => self.encode_occlusion_test(state, encoder),
            Pass::Decals => self.encode_decals(state, encoder, view, node),
            Pass::Water => report.draws += self.encode_water(state, encoder, view, node),
            Pass::CloudMarch => self.clouds.encode_view(
                &self.device,
                encoder,
                (&self.depth.view, self.depth.generation),
                &self.profiler,
            ),
            Pass::CloudComposite => self.encode_cloud_composite(state, encoder, view, node),
            Pass::Transparent => {
                report.draws += self.encode_transparent(state, encoder, view, node);
            }
            Pass::TransparentComposite => {
                self.encode_transparent_composite(state, encoder, view, node);
            }
            Pass::Effects => self.encode_screen_effects(state, encoder),
            Pass::Composite => self.encode_composite(encoder, view, node),
            Pass::TemporalResolve => self.encode_temporal_resolve(state, encoder, view, node),
            Pass::ProbeCaptures => self.encode_probes(state, encoder, report),
        }
    }

    /// The view of attachment `resource` this frame, `canvas` being the
    /// canvas's.
    fn attachment_view<'a>(
        &'a self,
        resource: Resource,
        canvas: &'a wgpu::TextureView,
    ) -> &'a wgpu::TextureView {
        let post = || {
            self.post
                .as_ref()
                .expect("frames drawing the effects' targets create them")
                .targets()
        };
        match resource {
            Resource::Canvas => canvas,
            Resource::Depth => &self.depth.view,
            Resource::Scene => post().0,
            Resource::Glow => post().1,
            Resource::Surface => post().2,
            Resource::Motion => self
                .temporal
                .as_ref()
                .expect("temporal frames create it")
                .motion(),
            Resource::Ids => self
                .visibility
                .buffer
                .as_ref()
                .expect("resolving frames create it")
                .ids_view(),
            Resource::Accumulated | Resource::Revealed => {
                let [accumulated, revealed] = self
                    .transparent
                    .as_ref()
                    .expect("transparent frames create it")
                    .views();
                if resource == Resource::Accumulated {
                    accumulated
                } else {
                    revealed
                }
            }
            _ => unreachable!("{resource:?} is no attachment of the renderer's passes"),
        }
    }

    /// `node`'s color attachments in target order with the operations the
    /// graph derived, and how many there are; `canvas` is the canvas's view.
    fn color_attachments<'a>(
        &'a self,
        node: &Node,
        canvas: &'a wgpu::TextureView,
    ) -> ([Option<wgpu::RenderPassColorAttachment<'a>>; 4], usize) {
        let mut attachments = [None, None, None, None];
        let mut count = 0;
        for (resource, ops) in node.colors() {
            attachments[count] = Some(wgpu::RenderPassColorAttachment {
                view: self.attachment_view(resource, canvas),
                depth_slice: None,
                resolve_target: None,
                ops: color_operations(ops),
            });
            count += 1;
        }
        (attachments, count)
    }

    /// `node`'s depth attachment with the operations the graph derived,
    /// read-only without them.
    fn depth_attachment(&self, node: &Node) -> Option<wgpu::RenderPassDepthStencilAttachment<'_>> {
        node.depth()
            .map(|(_, ops)| wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth.view,
                depth_ops: ops.map(depth_operations),
                stencil_ops: None,
            })
    }

    /// The sky of the frame's opaque targets, drawn after its last opaque
    /// geometry.
    fn sky_pipeline(&self, state: &FrameState<'_>) -> &wgpu::RenderPipeline {
        self.mesh_pipelines
            .sky(SkyTargets::Frame(state.shape.targets))
    }

    /// The pose buffers of this frame and the previous one, as the virtual
    /// pose groups pick them.
    fn pose_parities(&self, state: &FrameState<'_>) -> (usize, usize) {
        match state
            .packing
            .temporal_frame
            .as_ref()
            .zip(self.temporal.as_ref())
        {
            Some((frame, _)) if frame.motion_valid => (frame.parity, 1 - frame.parity),
            Some((frame, _)) => (frame.parity, frame.parity),
            None => (0, 0),
        }
    }

    /// GPU presentation's blending, composition, culling and compaction.
    fn encode_presentation(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.gpu_presentation
            .encode(&self.device, encoder, &self.profiler);
    }

    /// Shadow caster selection.
    fn encode_shadow_selection(
        &mut self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let shadow_frame = state
            .features
            .shadow_frame
            .as_ref()
            .expect("planned with shadows");
        self.shadows
            .encode_selection(&self.device, encoder, shadow_frame, &self.profiler);
    }

    /// Virtual geometry's cluster selection for the camera
    /// and the shadow maps.
    fn encode_cluster_selection(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let span = self.profiler.span("virtual_selection");
        if let Some(span) = span {
            span.mark(encoder, false);
        }
        self.virtual_draws.pass.encode(&self.device, encoder);
        if let Some(span) = span {
            span.mark(encoder, true);
        }
    }

    /// The shadow maps, with the clusters virtual
    /// geometry selected for them.
    fn encode_shadow_maps(
        &mut self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        report: &mut EncodeReport,
    ) {
        // Terrain's horizon maps where the sun or terrain
        // changed, before the passes that read them.
        self.terrain
            .encode_horizons(&self.device, encoder, &self.profiler);
        let virtual_frame = state.virtual_frame;
        let shadow_frame = state
            .features
            .shadow_frame
            .as_ref()
            .expect("planned with shadows");
        let meshes = &self.meshes;
        let arena = &self.mesh_arena;
        let (virtual_geometry, shadow_bins) =
            (&self.virtual_draws.pass, &self.virtual_draws.shadow_bins);
        let (terrain, terrain_frame) = (&self.terrain, state.terrain);
        report.shadow_draws = self.shadows.encode_maps(
            encoder,
            shadow_frame,
            &MapSources {
                meshes: &|mesh| {
                    meshes
                        .get(&mesh)
                        .map(|mesh| (mesh.draw(arena), mesh.skinned))
                },
                materials: &|material| self.materials.group(material),
                virtual_draw: &|pass, map| {
                    shadow_bins
                        .iter()
                        .find(|(bin_map, _, _)| *bin_map == map)
                        .map_or(0, |&(_, bin, offset)| {
                            u32::from(
                                virtual_frame && virtual_geometry.draw_shadow(pass, bin, offset),
                            )
                        })
                },
                // Terrain: the camera's patches in this map.
                terrain_draw: &|pass, map| {
                    let Some(pipeline) = terrain.shadow_pipeline().filter(|_| terrain_frame) else {
                        return 0;
                    };
                    let mut draws = 0;
                    for frame_terrain in &terrain.frame.terrains {
                        let Some((_, range)) = frame_terrain
                            .maps
                            .iter()
                            .find(|(drawn, range)| *drawn == map && !range.is_empty())
                        else {
                            continue;
                        };
                        pass.set_pipeline(pipeline);
                        draws += terrain.draw(pass, range, 0, frame_terrain.uniform);
                    }
                    draws
                },
                // Placed objects culled for this map.
                scatter_draw: &|pass, map, casters, cards| {
                    if state.scatter {
                        self.draw_scatter_shadow(pass, map, casters, cards)
                    } else {
                        0
                    }
                },
            },
            &self.profiler,
        );
    }

    /// Light binning.
    fn encode_light_binning(&mut self, state: &FrameState<'_>, encoder: &mut wgpu::CommandEncoder) {
        if let (Some(frame), Some(lights)) = (&state.features.light_frame, self.lights.as_mut()) {
            lights.encode(
                &self.device,
                encoder,
                frame,
                self.profiler.span("light_bins"),
            );
        }
    }

    /// The opaque pass of `phase` (two around the occlusion test), with the sky
    /// after the last one unless the visibility buffer resolves it; returns its
    /// draws, the sky's aside.
    fn encode_opaque(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
        phase: usize,
    ) -> u32 {
        let phases = usize::from(state.shape.phases);
        let resolve_phase = state.shape.resolves[phase];
        // The first of this phase's camera bins.
        let virtual_first = match phase {
            0 => Some(0),
            _ => self.virtual_draws.phase_two,
        }
        .filter(|_| state.virtual_frame);
        // Profiled frames time both phases as one opaque span, which the
        // visibility buffer's last resolve ends when it resolves this phase.
        let (begins, ends) = (phase == 0, phase + 1 == phases && !resolve_phase);
        // Terrain, the main occluder, draws in the first phase
        // so occlusion culling's pyramid holds it; shaded deferred
        // where it can be.
        if phase == 0 && self.terrain_deferred(state) {
            return self.encode_opaque_deferred(
                state,
                encoder,
                view,
                node,
                (phase, phases),
                (begins, ends),
                virtual_first,
            );
        }
        let timestamp_writes = (begins || ends)
            .then(|| self.profiler.span("opaque_meshes"))
            .flatten()
            .map(|span| span.render_part(begins, ends));
        let (colors, count) = self.color_attachments(node, view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/opaque-meshes"),
            color_attachments: &colors[..count],
            depth_stencil_attachment: self.depth_attachment(node),
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let mut draws = if state.kind.instance_count > 0 {
            self.draw_opaque(state, &mut pass, phase, virtual_first)
        } else {
            0
        };
        // Terrain, the main occluder: in the first phase, so
        // occlusion culling's pyramid holds it.
        if phase == 0 {
            draws += self.draw_terrain(state, &mut pass);
            draws += self.draw_scatter(state, &mut pass);
        }
        // Last, so depth testing leaves the sky only uncovered pixels.
        if phase + 1 == phases && !resolve_phase {
            self.draw_sky(state, &mut pass);
        }
        draws
    }

    /// The first opaque phase with terrain shaded deferred:
    /// the meshes; terrain's visibility, writing each pixel's terrain and
    /// depth; each terrain's resolve, shading its pixels once each; then
    /// its cover, placed objects and the sky; each in a pass of its own.
    #[allow(clippy::too_many_arguments)]
    fn encode_opaque_deferred(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
        (phase, phases): (usize, usize),
        (begins, ends): (bool, bool),
        virtual_first: Option<usize>,
    ) -> u32 {
        let resolve_phase = state.shape.resolves[phase];
        let loaded = wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        };
        let loaded_depth = wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        };
        // The meshes, as the opaque pass's start.
        let mut draws = 0;
        {
            let timestamp_writes = begins
                .then(|| self.profiler.span("opaque_meshes"))
                .flatten()
                .map(|span| span.render_part(true, false));
            let (colors, count) = self.color_attachments(node, view);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/opaque-meshes"),
                color_attachments: &colors[..count],
                depth_stencil_attachment: self.depth_attachment(node),
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if state.kind.instance_count > 0 {
                draws += self.draw_opaque(state, &mut pass, phase, virtual_first);
            }
        }
        let Some(ids) = &self.terrain_ids else {
            return draws;
        };
        let depth = wgpu::RenderPassDepthStencilAttachment {
            view: &self.depth.view,
            depth_ops: Some(loaded_depth),
            stencil_ops: None,
        };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/terrain-ids"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &ids.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(depth.clone()),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            draws += self.draw_terrain_ids(&mut pass);
        }
        let (mut colors, count) = self.color_attachments(node, view);
        for color in colors.iter_mut().flatten() {
            color.ops = loaded;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/terrain-resolve"),
                color_attachments: &colors[..count],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            draws += self.draw_terrain_resolve(state, &mut pass);
        }
        let timestamp_writes = ends
            .then(|| self.profiler.span("opaque_meshes"))
            .flatten()
            .map(|span| span.render_part(false, true));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/opaque-meshes"),
            color_attachments: &colors[..count],
            depth_stencil_attachment: Some(depth),
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        draws += self.draw_terrain_cover(state, &mut pass);
        draws += self.draw_scatter(state, &mut pass);
        if phase + 1 == phases && !resolve_phase {
            self.draw_sky(state, &mut pass);
        }
        draws
    }

    /// The sky over the pixels nothing covers, after the opaque surfaces.
    fn draw_sky(&self, state: &FrameState<'_>, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(self.sky_pipeline(state));
        pass.set_bind_group(0, &self.camera.group, &[]);
        pass.set_bind_group(1, self.clouds.sky_group(), &[]);
        pass.draw(0..3, 0..1);
    }

    /// Occlusion culling's test between the opaque phases: the
    /// depth pyramid of phase one's depths, the test and phase two's
    /// compaction, and virtual geometry's retest of the camera's clusters the
    /// last frame's pyramid hid.
    fn encode_occlusion_test(
        &mut self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let virtual_frame = state.virtual_frame;
        let temporal_frame = &state.packing.temporal_frame;
        // One span from the pyramid's reduction to the test's compaction or
        // the virtual retest.
        let span = self.profiler.span("occlusion");
        if let Some((pyramid, generation)) =
            self.culling.pyramid.as_ref().and_then(DepthPyramid::view)
        {
            let odd = temporal_frame
                .as_ref()
                .is_some_and(|frame| frame.parity == 1);
            let poses = self.instances.poses(usize::from(odd));
            self.culling
                .pyramid
                .as_ref()
                .expect("viewed above")
                .encode(encoder, span);
            // The camera's clusters the last frame's pyramid hid
            // are retested against this frame's within
            // the same span.
            self.gpu_presentation.encode_occlusion(
                &self.device,
                encoder,
                (pyramid, generation),
                poses,
                span.filter(|_| !virtual_frame),
            );
            if virtual_frame {
                self.virtual_draws.pass.encode_retest(&self.device, encoder);
                if let Some(span) = span {
                    span.mark(encoder, true);
                }
            }
        }
    }

    /// The opaque meshes of `phase` (GPU-presented frames draw each level of
    /// detail's visible slots), the impostors' cards,
    /// and virtual geometry's bins the visibility buffer leaves to
    /// forward drawing; returns the draws.
    fn draw_opaque(
        &self,
        state: &FrameState<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        phase: usize,
        virtual_first: Option<usize>,
    ) -> u32 {
        let gpu_mode = state.kind.gpu_mode;
        let Passes {
            opaque_kind,
            impostor_pass,
            ..
        } = state.passes;
        let temporal_pass = state
            .packing
            .temporal_frame
            .as_ref()
            .zip(self.temporal.as_ref());
        // Lit frames shade with the lit pipelines and the lights' group.
        let lit = state
            .features
            .light_frame
            .as_ref()
            .zip(self.lights.as_ref());
        pass.set_bind_group(0, &self.camera.group, &[]);
        if let Some((_, lights)) = lit {
            pass.set_bind_group(3, lights.fragment_group(), &[]);
        }
        pass.set_bind_group(
            2,
            temporal_pass.map_or_else(
                || self.instances.pose_group(0),
                |(frame, temporal)| temporal.pose_group(frame),
            ),
            &[],
        );
        if !gpu_mode {
            pass.set_vertex_buffer(1, self.instances.slots().slice(..));
        }
        let mut draws = 0;
        let mut binds = BatchBinds::default();
        for &index in &self.draw_order.order {
            let index = index as usize;
            let batch = &self.plan.batches[index];
            // Batches drawing virtual geometry draw its
            // clusters below.
            if self.virtual_draws.drawn.get(index) == Some(&true) {
                continue;
            }
            let (mesh, draw) = self.bind_batch(pass, &mut binds, (opaque_kind, false), batch);
            if gpu_mode {
                // Visible slots of this group, compacted and counted
                // on the GPU, each level of detail
                // with its own.
                for level in 0..lod_levels(batch, mesh, false) {
                    // An empty level draws nothing.
                    if level > 0 && mesh.lods[level - 1].1 == 0 {
                        continue;
                    }
                    let (ids, args, offset) = self
                        .gpu_presentation
                        .draw_source(batch, index, phase, level);
                    binds.bindings.ids(pass, 1, ids);
                    let timed = self
                        .profiler
                        .begin_draw(pass, || draw_label(batch, phase, Some(level)));
                    pass.draw_indexed_indirect(args, offset);
                    self.profiler.end_draw(pass, timed);
                    draws += 1;
                }
            } else {
                let timed = self
                    .profiler
                    .begin_draw(pass, || draw_label(batch, phase, None));
                pass.draw_indexed(
                    draw.index_range(),
                    draw.base_vertex,
                    batch.instances.clone(),
                );
                self.profiler.end_draw(pass, timed);
                draws += 1;
            }
        }
        // Impostors: every draw group's cards in one
        // draw, over the pose and light groups the meshes bound.
        if gpu_mode
            && let Some((targets, lit)) = impostor_pass
            && let (Some(pipeline), Some(group)) = (
                self.impostors.pipelines.pipeline(targets, lit),
                self.impostors.atlas.group(),
            )
        {
            let (ids, args, offset) = self.gpu_presentation.impostor_source(phase);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, group, &[]);
            pass.set_vertex_buffer(0, ids.slice(..));
            pass.draw_indirect(args, offset);
            draws += 1;
            // Cards fading over their meshes, once.
            if phase == 0 {
                let (cards, args, offset) = self.gpu_presentation.fading_cards();
                pass.set_vertex_buffer(0, cards);
                pass.draw_indirect(args, offset);
                draws += 1;
            }
        }
        // Virtual geometry: each bin's selected
        // clusters, in phase one, then each bin's twin of clusters
        // the retest found visible in phase two; bins the visibility
        // buffer resolves draw below.
        if let Some(first_bin) = virtual_first {
            let groups = self.virtual_draws.pose_groups.get().expect("built above");
            let group = match temporal_pass {
                Some((frame, _)) => {
                    let previous = if frame.motion_valid {
                        1 - frame.parity
                    } else {
                        frame.parity
                    };
                    &groups[2 * frame.parity + previous]
                }
                None => &groups[0],
            };
            pass.set_bind_group(2, group, &[]);
            for (bin, (material, variant)) in self.virtual_draws.bins.iter().enumerate() {
                if self.visibility.bins.get(bin) == Some(&true) {
                    continue;
                }
                pass.set_pipeline(&self.virtual_draws.pipelines[&(opaque_kind, *variant)]);
                pass.set_bind_group(1, self.materials.binding(*material).0, &[]);
                let (stream, args, offset) = self.virtual_draws.pass.bin_source(first_bin + bin);
                pass.set_vertex_buffer(0, stream);
                pass.draw_indirect(args, offset);
                draws += 1;
            }
        }
        draws
    }

    /// The triangle IDs of `phase`'s camera bins the visibility buffer
    /// resolves.
    fn encode_visibility_ids(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
        phase: usize,
    ) {
        let pipelines = self.visibility.pipelines.as_ref().expect("created above");
        let (pass_groups, _) = self.visibility.groups.get().expect("built above");
        let bins = self.virtual_draws.bins.len();
        let first_bin = match phase {
            0 => 0,
            _ => self
                .virtual_draws
                .phase_two
                .expect("phase two resolves its bins' twins"),
        };
        let (current, _) = self.pose_parities(state);
        let (colors, count) = self.color_attachments(node, view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/visibility"),
            color_attachments: &colors[..count],
            depth_stencil_attachment: self.depth_attachment(node),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &self.camera.group, &[]);
        pass.set_bind_group(1, &self.mesh_pipelines.empty_group, &[]);
        for (bin, (_, variant)) in self.virtual_draws.bins.iter().enumerate() {
            if !self.visibility.bins[bin] {
                continue;
            }
            pass.set_pipeline(&pipelines[usize::from(variant.double_sided())]);
            pass.set_bind_group(
                2,
                &pass_groups[current],
                &[VisibilityBuffer::draw_offset(phase, bins, bin)],
            );
            let (stream, args, offset) = self.virtual_draws.pass.bin_source(first_bin + bin);
            pass.set_vertex_buffer(0, stream);
            pass.draw_indirect(args, offset);
        }
    }

    /// The tiles of each bin the visibility buffer resolves.
    fn encode_visibility_tiles(&self, encoder: &mut wgpu::CommandEncoder) {
        self.visibility
            .buffer
            .as_ref()
            .expect("prepared above")
            .encode_classify(encoder, self.virtual_draws.bins.len());
    }

    /// Each resolved bin's pixels of `phase` shaded into the opaque targets,
    /// and the sky after the last phase.
    fn encode_visibility_resolve(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
        phase: usize,
    ) {
        let phases = usize::from(state.shape.phases);
        let opaque_kind = state.passes.opaque_kind;
        let lit = state
            .features
            .light_frame
            .as_ref()
            .zip(self.lights.as_ref());
        let visibility = self.visibility.buffer.as_ref().expect("prepared above");
        let (_, resolve_groups) = self.visibility.groups.get().expect("built above");
        let bins = self.virtual_draws.bins.len();
        let (current, previous) = self.pose_parities(state);
        // The last resolve ends the opaque span.
        let timestamp_writes = (phase + 1 == phases)
            .then(|| self.profiler.span("opaque_meshes"))
            .flatten()
            .map(|span| span.render_part(false, true));
        let (colors, count) = self.color_attachments(node, view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/visibility-resolve"),
            color_attachments: &colors[..count],
            // Read-only: the resolves test no depth and the sky tests it.
            depth_stencil_attachment: self.depth_attachment(node),
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &self.camera.group, &[]);
        if let Some((_, lights)) = lit {
            pass.set_bind_group(3, lights.fragment_group(), &[]);
        }
        let (_, _, resolve_args) = visibility.buffers();
        for (bin, (material, variant)) in self.virtual_draws.bins.iter().enumerate() {
            if !self.visibility.bins[bin] {
                continue;
            }
            pass.set_pipeline(&self.visibility.resolve_pipelines[&(opaque_kind, *variant)]);
            pass.set_bind_group(1, self.materials.binding(*material).0, &[]);
            pass.set_bind_group(
                2,
                &resolve_groups[2 * current + previous],
                &[VisibilityBuffer::draw_offset(phase, bins, bin)],
            );
            pass.draw_indirect(resolve_args, bin as u64 * 16);
        }
        if phase + 1 == phases {
            pass.set_pipeline(self.sky_pipeline(state));
            pass.set_bind_group(0, &self.camera.group, &[]);
            pass.set_bind_group(1, self.clouds.sky_group(), &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Decals into the frame's color target.
    fn encode_decals(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
    ) {
        let features = &state.features;
        let decals = self.decals.as_ref().expect("planned with decals");
        let (colors, count) = self.color_attachments(node, view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/decals"),
            color_attachments: &colors[..count],
            depth_stencil_attachment: None,
            timestamp_writes: self.profiler.span("decals").map(Span::render),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        decals.draw(
            &mut pass,
            &self.camera.group,
            &self.materials,
            features
                .light_frame
                .as_ref()
                .and(self.lights.as_ref())
                .map(LightPass::fragment_group),
            state.shape.targets != Targets::Canvas,
        );
    }

    /// Water over the opaque scene, writing its depth;
    /// returns its draws.
    fn encode_water(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
    ) -> u32 {
        // The ground behind the water, before it draws.
        self.water.copy_depth(encoder, &self.depth.texture);
        let (colors, count) = self.color_attachments(node, view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/water"),
            color_attachments: &colors[..count],
            depth_stencil_attachment: self.depth_attachment(node),
            timestamp_writes: self.profiler.span("water").map(Span::render),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.water
            .draw(&mut pass, &self.camera.group, state.shape.targets)
    }

    /// The view's clouds laid over the scene's color.
    fn encode_cloud_composite(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
    ) {
        let (colors, count) = self.color_attachments(node, view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/cloud-composite"),
            color_attachments: &colors[..count],
            depth_stencil_attachment: None,
            timestamp_writes: self.profiler.span("cloud_composite").map(Span::render),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.clouds.draw_composite(
            &mut pass,
            &self.camera.group,
            self.cloud_target(state.shape.targets),
        );
    }

    /// The format of the scene's color for frames drawing into `targets`,
    /// and whether that is the canvas.
    fn cloud_target(&self, targets: Targets) -> (wgpu::TextureFormat, bool) {
        if targets == Targets::Canvas {
            (self.display.format(), true)
        } else {
            (crate::post::SCENE_FORMAT, false)
        }
    }

    /// Clouds in the view: their composite's pipeline into
    /// `targets` and its group; whether they draw this frame.
    fn prepare_cloud_composite(&mut self, targets: Targets) -> bool {
        if !self.clouds.marches_view() {
            return false;
        }
        let target = self.cloud_target(targets);
        self.clouds.prepare_composite(
            &self.device,
            &self.mesh_shader,
            &self.camera.layout,
            target,
            (&self.depth.view, self.depth.generation),
        );
        true
    }

    /// Transparent surfaces and alpha particles accumulated;
    /// returns the surfaces' draws.
    fn encode_transparent(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
    ) -> u32 {
        let FrameKind {
            gpu_mode,
            instance_count,
            ..
        } = state.kind;
        let transparent_kind = state.passes.transparent_kind;
        let temporal_frame = &state.packing.temporal_frame;
        let features = &state.features;
        let (light_frame, post_frame, particle_frame) = (
            &features.light_frame,
            features.post_frame,
            &features.particle_frame,
        );
        let (transparent_draws, refraction) = (features.transparent_draws, features.refraction);
        let lit = light_frame.as_ref().zip(self.lights.as_ref());
        // Read-only depth: surfaces behind the opaque ones stay hidden, and
        // particles sample it.
        let (colors, count) = self.color_attachments(node, view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/transparent-accumulation"),
            color_attachments: &colors[..count],
            depth_stencil_attachment: self.depth_attachment(node),
            timestamp_writes: self.profiler.span("transparent").map(Span::render),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let mut draws = 0;
        if transparent_draws && instance_count > 0 {
            // Glass reads the opaque color the pass draws over.
            let camera_group = match self.camera.refraction_group() {
                Some(group) if refraction && post_frame.is_some() => group,
                _ => &self.camera.group,
            };
            pass.set_bind_group(0, camera_group, &[]);
            if let Some((_, lights)) = lit {
                pass.set_bind_group(3, lights.fragment_group(), &[]);
            }
            let parity = temporal_frame.as_ref().map_or(0, |frame| frame.parity);
            pass.set_bind_group(2, self.instances.pose_group(parity), &[]);
            if !gpu_mode {
                pass.set_vertex_buffer(1, self.instances.slots().slice(..));
            }
            let mut binds = BatchBinds::default();
            for (index, batch) in self.plan.batches.iter().enumerate() {
                if !batch.transparent {
                    continue;
                }
                let (_, draw) =
                    self.bind_batch(&mut pass, &mut binds, (transparent_kind, false), batch);
                if gpu_mode {
                    let (ids, args, offset) = self.gpu_presentation.draw_source(batch, index, 0, 0);
                    binds.bindings.ids(&mut pass, 1, ids);
                    pass.draw_indexed_indirect(args, offset);
                } else {
                    pass.draw_indexed(
                        draw.index_range(),
                        draw.base_vertex,
                        batch.instances.clone(),
                    );
                }
                draws += 1;
            }
        }
        if let (Some(frame), Some(particles)) = (particle_frame, self.particles.as_ref()) {
            particles.draw_alpha(&mut pass, &self.camera.group, frame);
        }
        draws
    }

    /// The accumulated transparent surfaces composited over the frame's
    /// color, then additive particles.
    fn encode_transparent_composite(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
    ) {
        let transparent = self
            .transparent
            .as_ref()
            .expect("planned with transparency");
        let post = state.shape.targets != Targets::Canvas;
        let (colors, count) = self.color_attachments(node, view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/transparent-composite"),
            color_attachments: &colors[..count],
            depth_stencil_attachment: self.depth_attachment(node),
            timestamp_writes: self
                .profiler
                .span("transparent_composite")
                .map(Span::render),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        transparent.composite(&mut pass, post);
        if let (Some(frame), Some(particles)) =
            (&state.features.particle_frame, self.particles.as_ref())
        {
            particles.draw_additive(&mut pass, &self.camera.group, frame, post);
        }
    }

    /// The screen effects: ambient occlusion, bloom,
    /// reflections and sun shafts after the opaque and transparent passes.
    fn encode_screen_effects(&self, state: &FrameState<'_>, encoder: &mut wgpu::CommandEncoder) {
        let post_frame = state.features.post_frame.expect("planned with effects");
        let post = self.post.as_ref().expect("post frames create it");
        post.encode_effects(encoder, post_frame, &self.camera.group, &self.profiler);
    }

    /// The screen effects' composite into the canvas.
    fn encode_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
    ) {
        let post = self.post.as_ref().expect("post frames create it");
        let (_, canvas) = node
            .colors()
            .next()
            .expect("the composite draws the canvas");
        post.encode_composite(encoder, view, color_operations(canvas), &self.profiler);
    }

    /// Temporal antialiasing's resolve into the canvas and the next history.
    fn encode_temporal_resolve(
        &mut self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
    ) {
        let frame = state
            .packing
            .temporal_frame
            .as_ref()
            .expect("temporal frames begin it");
        let mut ops = node.colors().map(|(_, ops)| color_operations(ops));
        let ops = [
            ops.next().expect("the canvas"),
            ops.next().expect("the history"),
        ];
        self.temporal
            .as_ref()
            .expect("temporal frames create it")
            .encode_resolve(
                encoder,
                view,
                frame,
                ops,
                self.profiler.span("taa_resolve").map(Span::render),
            );
    }

    /// Reflection probe captures after the frame's own
    /// drawing, once their slots stand still.
    fn encode_probes(
        &mut self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        report: &mut EncodeReport,
    ) {
        let snapshot = state.snapshot;
        let temporal_frame = &state.packing.temporal_frame;
        let (probe_list, light_frame) = (state.features.probe_list, &state.features.light_frame);
        self.probes.slots.tick();
        if !self.gpu_presentation.stillness_settled() {
            return;
        }
        let odd = temporal_frame
            .as_ref()
            .is_some_and(|frame| frame.parity == 1);
        let faces = self.encode_probe_captures(
            encoder,
            probe_list,
            snapshot.environment.reflection_probes.faces_per_frame,
            odd,
            snapshot.elapsed_seconds,
            light_frame
                .as_ref()
                .map(|frame| (frame, &snapshot.environment.local_lights)),
        );
        report.probe_faces = faces;
    }

    /// Submits the frame after the work it reads (asset levels, staged pose
    /// and input copies, environment maps) and presents it.
    fn submit_frame(
        &mut self,
        state: &mut FrameState<'_>,
        frame: DisplayFrame,
        encoder: wgpu::CommandEncoder,
    ) {
        let FrameKind { gpu_mode, .. } = state.kind;
        let virtual_frame = state.virtual_frame;
        let sky_encoder = state.features.sky_encoder.take();
        let mip_encoder = state.assets.encoder.take();
        let staged_poses = state.writes.staged_poses.take();
        let input_copies = state.features.input_copies.take();
        let environment_encoder = state.features.environment_encoder.take();
        let temporal_frame = &state.packing.temporal_frame;

        let profile_submit = engine_profile::span("render", "submit_and_present");
        // Profiling captures time each of the command buffers before the
        // frame's between marks of their own.
        let mark = |span: &'static str, end: bool| {
            let span = self.profiler.span(span)?;
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine/profile-mark"),
                });
            span.mark(&mut encoder, end);
            Some(encoder.finish())
        };
        let timed = |span: &'static str, buffer: Option<wgpu::CommandBuffer>| {
            let marked = buffer.is_some() && self.profiler.capturing();
            let begin = marked.then(|| mark(span, false)).flatten();
            let end = marked.then(|| mark(span, true)).flatten();
            begin.into_iter().chain(buffer).chain(end)
        };
        // The sky's light precedes everything that reads it; staged
        // pose/input copies precede the passes that read them.
        let before: Vec<wgpu::CommandBuffer> = sky_encoder
            .map(wgpu::CommandEncoder::finish)
            .into_iter()
            .chain(timed(
                "pre_mips",
                mip_encoder.map(wgpu::CommandEncoder::finish),
            ))
            .chain(timed("pre_poses", staged_poses))
            .chain(timed(
                "pre_inputs",
                input_copies.map(wgpu::CommandEncoder::finish),
            ))
            .chain(timed(
                "pre_environment",
                environment_encoder.map(wgpu::CommandEncoder::finish),
            ))
            .collect();
        self.queue
            .submit(before.into_iter().chain([encoder.finish()]));
        self.instances.pose_uploads.after_submit();
        if gpu_mode {
            self.gpu_presentation.after_submit();
        }
        self.shadows.after_submit();
        self.ground_cover.after_submit();
        self.terrain.vt.after_submit();
        if let Some(lights) = self.lights.as_mut() {
            lights.after_submit();
        }
        if virtual_frame {
            self.virtual_draws.pass.after_submit();
        }
        if let (Some(frame), Some(pass)) = (&temporal_frame, &mut self.temporal) {
            pass.submitted(frame);
        }
        self.profiler.submitted();
        frame.present(&self.queue);
        // Native interfaces run finished work's callbacks, read-backs and
        // staging maps among them, when polled.
        crate::platform::poll(&self.device);
        drop(profile_submit);
    }
}

/// A color attachment's operations as the render graph derived them.
fn color_operations(ops: Ops) -> wgpu::Operations<wgpu::Color> {
    wgpu::Operations {
        load: match ops.load {
            Load::Clear(Clear::Color([r, g, b, a])) => {
                wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a })
            }
            Load::Clear(Clear::Depth(_)) => unreachable!("color attachments clear to colors"),
            Load::Load => wgpu::LoadOp::Load,
        },
        store: store_operation(ops),
    }
}

/// A depth attachment's operations as the render graph derived them.
fn depth_operations(ops: Ops) -> wgpu::Operations<f32> {
    wgpu::Operations {
        load: match ops.load {
            Load::Clear(Clear::Depth(depth)) => wgpu::LoadOp::Clear(depth),
            Load::Clear(Clear::Color(_)) => unreachable!("depth attachments clear to depths"),
            Load::Load => wgpu::LoadOp::Load,
        },
        store: store_operation(ops),
    }
}

fn store_operation(ops: Ops) -> wgpu::StoreOp {
    if ops.store {
        wgpu::StoreOp::Store
    } else {
        wgpu::StoreOp::Discard
    }
}

/// What profiling reports name a draw of `batch` by: its mesh, material,
/// opaque phase and level of detail (`None` for exact frames' draws of
/// every instance).
#[cfg_attr(not(feature = "profiling"), allow(dead_code))]
fn draw_label(batch: &DrawBatch, phase: usize, level: Option<usize>) -> String {
    let material = batch
        .material
        .map_or_else(|| "none".to_owned(), |material| material.slot().to_string());
    let mesh = batch.mesh.slot();
    match level {
        Some(level) => format!("mesh {mesh} material {material} phase {phase} level {level}"),
        None => format!("mesh {mesh} material {material} phase {phase}"),
    }
}
