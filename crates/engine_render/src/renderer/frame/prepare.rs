//! A frame's preparation before its passes encode: its plan, kind and
//! residency, asset uploads, instance packing and writes, each feature's
//! preparation, the pipelines it draws with, and its render graph's shape.

use super::*;

/// GPU presentation's preparation: occlusion culling's viewport and depth
/// pyramid levels, and what presentation prepared.
struct PresentationPrep {
    occlusion: Option<(u32, u32, u32)>,
    gpu_prepared: Option<GpuPrepared>,
}

impl Renderer {
    /// Plans the frame's batches and shadow casters, retires assets it no
    /// longer uses and decides its kind; also returns GPU presentation's
    /// inputs.
    pub(super) fn plan_frame<'f>(
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
        for source in snapshot.textures.iter().chain(&snapshot.decals.textures) {
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

    /// Follows the `TemporalAntialiasing` settings: enabling creates the
    /// pipelines of temporally antialiased display frames, disabling releases
    /// their targets and pose buffer. Frame targets follow the surface size,
    /// 16 bytes per pixel.
    fn apply_temporal_antialiasing(&mut self, enabled: bool) {
        if enabled == self.temporal.is_some() {
            return;
        }
        if enabled {
            self.temporal = Some(TemporalPass::new(
                &self.device,
                &self.camera.layout,
                self.materials.layout(),
                self.display.format(),
            ));
        } else {
            self.temporal = None;
            // Pipelines over the temporal pass's layouts go with it.
            let current =
                |&(kind, _): &(PassKind, Variant)| kind.targets() != Some(Targets::Temporal);
            self.mesh_pipelines.variants.retain(|key, _| current(key));
            self.virtual_draws.pipelines.retain(|key, _| current(key));
            self.visibility
                .resolve_pipelines
                .retain(|key, _| current(key));
            self.impostors.pipelines.forget_pipelines(Targets::Temporal);
            self.mesh_pipelines
                .forget_sky(SkyTargets::Frame(Targets::Temporal));
            self.instances.release_alternate();
        }
    }

    /// Uploads the frame's new meshes (with their levels of detail and
    /// virtual geometry) and textures, bakes new impostors, and orders the
    /// opaque draws. New textures' mip levels and impostor bakes record into
    /// `mips`, the prelude's.
    pub(super) fn upload_assets(
        &mut self,
        snapshot: &RenderSnapshot,
        kind: FrameKind,
        mips: &mut Option<wgpu::CommandEncoder>,
    ) -> AssetUploads {
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
        // Masked shadow casters bind their materials even when culled.
        let shadow_materials = self
            .shadows
            .groups()
            .iter()
            .filter_map(|group| group.material);
        // New textures' mip levels render from their uploaded base level in
        // the prelude, submitted ahead of this frame's command buffer.
        let (texture_upload_bytes, mip_levels) = self.materials.upload(
            &self.device,
            &self.queue,
            snapshot,
            &self.plan,
            shadow_materials,
            mips,
        );
        // Impostors bake after their materials' levels, the
        // plan's and then placed objects'.
        self.prepare_impostors(snapshot, gpu_mode, mips);
        self.prepare_scatter_impostors(snapshot, gpu_mode, mips);
        drop(profile_upload);
        self.order_draws(snapshot);
        AssetUploads {
            mesh_bytes: mesh_upload_bytes,
            texture_bytes: texture_upload_bytes,
            mip_levels,
        }
    }

    /// Sizes instance, pose, color and palette storage for the frame, begins
    /// its temporal antialiasing, and packs the records to upload.
    pub(super) fn pack_instances(&mut self, frame: RenderFrame<'_>, kind: FrameKind) -> Packing {
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
    /// poses, colors, palettes and slot IDs; large pose uploads stage their
    /// copies into `poses`, the prelude's.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(super) fn write_frame_data(
        &mut self,
        frame: RenderFrame<'_>,
        kind: FrameKind,
        packing: &Packing,
        poses: &mut Option<wgpu::CommandBuffer>,
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
        // Large pose uploads copy into a persistently mapped staging buffer
        // and a GPU copy submitted before this frame's draws.
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
            *poses = staged.then(|| encoder.finish());
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
        }
    }

    /// Prepares every feature the frame runs, in order: GPU presentation
    /// with occlusion culling, shadows, clouds, the physically based sky,
    /// environment maps and reflection probes, local lights, screen effects,
    /// transparency and particles, and decals. Presentation's input copies,
    /// the sky's light and the environment's work go in `prelude`.
    pub(super) fn prepare_features<'s>(
        &mut self,
        frame: RenderFrame<'s>,
        gpu: Option<&engine_core::GpuPresentation<'_>>,
        kind: FrameKind,
        (temporal_frame, lighting): (&Option<TemporalFrame>, &LightingUniform),
        prelude: &mut Prelude,
    ) -> Features<'s> {
        let snapshot = frame.snapshot();
        let PresentationPrep {
            occlusion,
            gpu_prepared,
        } = self.prepare_presentation(frame, gpu, kind, temporal_frame, &mut prelude.inputs);
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
        prelude.sky = self.prepare_sky(snapshot);
        let (environment, probe_list, dropped_probes) =
            self.prepare_environment(snapshot, lighting);
        prelude.environment = environment;
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
            occlusion,
            gpu_prepared,
            shadow_frame,
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

    /// Occlusion culling's depth pyramid and GPU presentation's inputs, whose
    /// copies record into `input_copies`, the prelude's; returns the
    /// occlusion viewport and levels, and what presentation prepared.
    fn prepare_presentation(
        &mut self,
        frame: RenderFrame<'_>,
        gpu: Option<&engine_core::GpuPresentation<'_>>,
        kind: FrameKind,
        temporal_frame: &Option<TemporalFrame>,
        input_copies: &mut Option<wgpu::CommandEncoder>,
    ) -> PresentationPrep {
        let gpu_mode = kind.gpu_mode;
        let snapshot = frame.snapshot();

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
                    && !snapshot.reflection_probes.list.is_empty())
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
                    copies: input_copies,
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

    /// Environment maps and reflection probes, and the camera
    /// group binding them; returns the environment's encoder, the probes the
    /// frame keeps and those beyond the GPU's arrays.
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
            &snapshot.reflection_probes.list
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
        let dropped_probes = snapshot.reflection_probes.dropped as u64
            + (probe_list.len() as u64).saturating_sub(u64::from(most_cubes - 1));
        self.refresh_camera_group();
        (environment_encoder, probe_list, dropped_probes)
    }

    /// Rebuilds the main passes' first group when what it binds was
    /// replaced.
    pub(super) fn refresh_camera_group(&mut self) {
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
        if snapshot.environment.local_lights.active() && !snapshot.lights.list.is_empty() {
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
            if snapshot.environment.particles.enabled && !snapshot.particles.list.is_empty() {
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
        if snapshot.environment.decals.enabled && !snapshot.decals.list.is_empty() {
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
    pub(super) fn prepare_pipelines(
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
    pub(super) fn prepare_virtual_frame(
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
    pub(super) fn frame_shape(
        &self,
        kind: FrameKind,
        (packing, features, passes): (&Packing, &Features<'_>, &Passes),
        (virtual_frame, resolving, scatter, terrain): (bool, bool, bool, bool),
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
            horizons: self.terrain.horizons.pending(),
            cloud_dome: self.clouds.marches_dome(),
            deformation: self.terrain.deformation.pending(),
            vt_bakes: self.terrain.vt.baking(),
            cover: terrain && self.ground_cover.growing(),
            deferred: self.terrain_deferred(terrain, passes.opaque_kind),
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

    /// Clouds in the view: their composite's pipeline into
    /// `targets` and its group; whether they draw this frame.
    pub(super) fn prepare_cloud_composite(&mut self, targets: Targets) -> bool {
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

    /// Temporal antialiasing's pose groups and the camera
    /// group of glass refracting the opaque color.
    pub(super) fn prepare_frame_groups(&mut self, state: &FrameState<'_>) {
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
}

/// The canvas's clear color: the scene's in linear light, encoded for the
/// UNORM canvas.
fn canvas_clear() -> [f64; 4] {
    let encode = |linear: f64| 1.055 * linear.powf(1.0 / 2.4) - 0.055;
    let [r, g, b, a] = crate::render_graph::SCENE_CLEAR;
    [encode(r), encode(g), encode(b), a]
}
