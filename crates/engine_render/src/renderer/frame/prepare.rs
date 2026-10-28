#![expect(
    clippy::expect_used,
    reason = "temporal frames create the temporal pass"
)]

use super::*;
use crate::clouds::CloudFrame;
use crate::temporal::ResolveFrame;
use crate::transparent::TransparencyFrame;

struct PresentationPrep {
    occlusion: Option<(u32, u32, u32)>,
    gpu_prepared: Option<GpuPrepared>,
}

impl Renderer {
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
            self.instances.retained_uploads.invalidate();
            self.instances.slot_revision = None;
        }
        self.gpu_frame = gpu_mode;
        self.apply_temporal_antialiasing(snapshot.environment.temporal_antialiasing.enabled);
        let debug = self.shows_debug_view(snapshot);
        let temporal = display
            && gpu_mode
            && !debug
            && self.features.temporal.get().is_some()
            && !self.features.post.disabled();
        if !temporal && let Some(pass) = self.features.temporal.get_mut() {
            pass.reset();
        }
        self.display
            .plan_region(display && !debug && !self.features.post.disabled());
        let profile_residency = engine_profile::span("render", "validate_and_retire");
        for source in &snapshot.textures {
            if source.texture.width().max(source.texture.height()) > self.display.max_dimension {
                return Err(RenderError(format!(
                    "Texture {:?} exceeds the device's {}-pixel dimension limit.",
                    source.handle, self.display.max_dimension,
                )));
            }
        }
        self.shadows.plan_groups(&self.plan, frame, gpu_mode);
        let placed: BTreeSet<MeshHandle> = snapshot
            .terrain
            .terrains
            .iter()
            .flat_map(|terrain| terrain.assets.iter().flatten())
            .flat_map(|asset| asset.meshes().map(|mesh| mesh.handle))
            .collect();
        if placed != self.scatter_meshes.0 {
            self.scatter_meshes = (placed, self.scatter_meshes.1 + 1);
        }
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
                debug,
                instance_count: self.plan.ordered_instances.len(),
            },
            gpu,
        ))
    }

    fn apply_temporal_antialiasing(&mut self, enabled: bool) {
        if enabled == self.features.temporal.get().is_some() {
            return;
        }
        if enabled {
            let (device, queue, camera) = (&self.device, &self.queue, &self.camera.layout);
            let (materials, format) = (self.materials.layout(), self.display.format());
            self.features.temporal.create(device, || {
                TemporalPass::new(device, queue, camera, materials, format)
            });
        } else {
            self.features.temporal.release();
            self.forget_temporal();
        }
    }

    pub(in crate::renderer) fn forget_temporal(&mut self) {
        let current = |&(kind, _): &(PassKind, Variant)| kind.targets() != Some(Targets::Temporal);
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

    pub(super) fn specialize_shading(&mut self, snapshot: &RenderSnapshot) {
        let shading = SceneShading::of(snapshot);
        if shading != self.mesh_pipelines.shading {
            self.mesh_pipelines.shading = shading;
            self.mesh_pipelines.variants.clear();
            self.structure_keys.variants = None;
            self.structure_keys.probes = None;
            self.virtual_draws.pipelines.clear();
            self.visibility.resolve_pipelines.clear();
            self.impostors.pipelines.forget_shading();
        }
        self.terrain.pipelines.specialize(shading);
        self.features.ground_cover.get_mut().specialize(shading);
        let water_views = crate::debug_view::shows_water(snapshot.environment.debug_view.view);
        self.features
            .water
            .get_mut()
            .specialize(shading, water_views);
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(super) fn upload_assets(
        &mut self,
        snapshot: &RenderSnapshot,
        kind: FrameKind,
        mips: &mut Option<wgpu::CommandEncoder>,
    ) -> AssetUploads {
        let gpu_mode = kind.gpu_mode;

        let profile_upload = engine_profile::span("render", "asset_uploads");
        let mut mesh_upload_bytes = 0_u64;
        let uploads = (
            self.plan.rebuilds(),
            self.shadows.groups_generation(),
            self.mesh_layout,
            self.scatter_meshes.1,
        );
        let mesh_uploads = self.structure_keys.uploads != Some(uploads);
        let sources = self
            .plan
            .batches
            .iter()
            .map(|batch| (batch.mesh, &snapshot.meshes[batch.source_index]))
            .chain(
                self.shadows
                    .groups()
                    .iter()
                    .map(|group| (group.mesh, &snapshot.meshes[group.source_index])),
            )
            .chain(
                snapshot
                    .terrain
                    .terrains
                    .iter()
                    .flat_map(|terrain| terrain.assets.iter().flatten())
                    .flat_map(engine_core::RenderPlacementAsset::meshes)
                    .map(|mesh| (mesh.handle, mesh)),
            )
            .filter(|_| mesh_uploads);
        for (handle, rendered) in sources {
            let source = &rendered.mesh;
            if let std::collections::btree_map::Entry::Vacant(entry) = self.meshes.entry(handle) {
                self.mesh_uploads.staging.clear();
                self.mesh_uploads
                    .staging
                    .reserve_exact(source.vertices().len());
                self.mesh_uploads
                    .staging
                    .extend(source.vertices().iter().map(Vertex::from));
                let extras = ExtraVertex::stream(source);
                let skin = SkinVertex::stream(source);
                let mut streams: Vec<&[u8]> =
                    vec![bytemuck::cast_slice(&self.mesh_uploads.staging)];
                if let Some(extras) = &extras {
                    streams.push(bytemuck::cast_slice(extras));
                }
                if let Some(skin) = &skin {
                    streams.push(bytemuck::cast_slice(skin));
                }
                let (indices, lods, casters) = with_levels(source, rendered.lods.as_deref());
                let index_count = source.indices().len() as u32;
                let without_levels = rendered.lods.is_none()
                    && source.joint_count() == 0
                    && source.virtual_geometry().is_none();
                let placement = self
                    .mesh_arena
                    .insert(
                        &self.device,
                        &self.queue,
                        &streams,
                        self.mesh_uploads.staging.len() as u32,
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
                    index_count,
                    byte_len,
                    plan: plan_levels(&lods),
                    lods,
                    casters,
                    without_levels,
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
            mips,
        );
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

    pub(super) fn pack_instances(&mut self, frame: RenderFrame<'_>, kind: FrameKind) -> Packing {
        let FrameKind {
            gpu_mode, temporal, ..
        } = kind;
        let snapshot = frame.snapshot();

        let profile_packing = engine_profile::span("render", "instance_buffer_and_packing");
        self.instances.reserve(
            &self.device,
            snapshot.instances.len(),
            snapshot.joints.len(),
            temporal,
        );
        let temporal_frame: Option<TemporalFrame> = match self.features.temporal.get_mut() {
            Some(pass) if temporal => Some(pass.begin(
                &self.device,
                &self.queue,
                &frame,
                self.display.region,
                self.instances.generation(),
            )),
            _ => None,
        };
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
        camera.set_region(snapshot.view_projection, self.display.region);
        camera.set_detail(snapshot.detail_eye);
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
            (&snapshot.environment.sky, &snapshot.environment.highlights),
            (&snapshot.environment.foliage, temporal_frame.is_some()),
        );
        lighting.grade[3] = f32::from(u8::from(snapshot.environment.dither.enabled));
        crate::sky::apply_night(&mut lighting, &snapshot.environment);
        crate::sky::apply_haze(
            &mut lighting,
            &snapshot.environment.haze,
            snapshot.view_projection,
            snapshot.camera_position,
            !kind.debug
                && !self.features.post.disabled()
                && crate::screen_effects::haze_runs(&snapshot.environment),
        );
        lighting.underwater = crate::water::underwater(
            &snapshot.environment,
            snapshot.camera_position,
            snapshot.presented_seconds,
        );
        let ocean = snapshot.environment.ocean;
        if ocean.enabled {
            lighting.sea = [ocean.level, ocean.clarity, 1.0, 0.0];
        }
        (lighting.cloud_shadow, lighting.cloud_layer) =
            crate::clouds::shadow_lighting(&snapshot.environment.clouds, snapshot.camera_position);
        if let Some(maps) = self.features.occupancy.get_mut().prepare(
            &self.device,
            &self.queue,
            &snapshot.environment.occupancy,
            snapshot.detail_eye,
            (
                self.scatter_draws.covers(),
                self.features.scatter.get().objects(),
            ),
        ) {
            lighting.occupancy = maps.occupancy;
            lighting.occupancy_maps = maps.maps;
            lighting.occupancy_centers = maps.centers;
        }
        camera_upload_bytes += self.camera.write_lighting(&self.queue, lighting);
        let mut color_upload_bytes = 0_u64;
        let mut slot_upload_bytes = 0_u64;
        let pose_ranges = if gpu_mode {
            &[][..]
        } else {
            self.instances.retained_uploads.pose_ranges()
        };
        let instance_write_ranges = pose_ranges.len()
            + self.instances.retained_uploads.color_ranges().len()
            + usize::from(write_slots);
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
        self.features.clouds.get_mut().prepare(
            &self.device,
            &self.queue,
            (
                &snapshot.environment.clouds,
                snapshot.tables.cloud_noise.as_ref(),
                (snapshot.camera_position, snapshot.surface_under_eye),
                snapshot.presented_seconds,
            ),
            (lighting, self.features.sky.get().bindings()),
            !kind.gpu_mode,
        );
        self.features.clouds.get_mut().prepare_view(
            &self.device,
            &self.queue,
            (
                &snapshot.environment.clouds,
                snapshot.camera_position,
                snapshot.presented_seconds,
            ),
            snapshot.view_projection,
            self.display.region,
            (lighting, self.features.sky.get().bindings()),
            &self.depth.view,
        );
        self.features
            .environment
            .get_mut()
            .set_clouds(&self.device, self.features.clouds.get().environment_dome());
        prelude.sky = self.prepare_sky(snapshot);
        let (environment, probe_list) = self.prepare_environment(snapshot, lighting);
        prelude.environment = environment;
        if kind.debug {
            return Features::debug(occlusion, gpu_prepared, shadow_frame, probe_list);
        }
        let lights = self.prepare_lights(frame, kind, temporal_frame);
        let post = self.prepare_post(snapshot, temporal_frame, kind.display);
        let transparency = self.prepare_transparency(
            snapshot,
            (frame.identity(), frame.emitters_revision()),
            post.is_some(),
        );
        let temporal = self.prepare_resolve(temporal_frame, transparency.is_some());
        Features {
            occlusion,
            gpu_prepared,
            shadow_frame,
            probe_list,
            frames: FeatureFrames {
                lights,
                transparency,
                post,
                temporal,
                ..FeatureFrames::default()
            },
        }
    }

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

        let occlusion_enabled = snapshot.environment.occlusion_culling.enabled;
        if !occlusion_enabled {
            self.culling.pyramid = None;
        }
        let region = self.display.region;
        let occlusion = if gpu_mode && occlusion_enabled {
            let device = &self.device;
            let pyramid = self
                .culling
                .pyramid
                .get_or_insert_with(|| DepthPyramid::new(device, Reduction::Farthest));
            let levels = pyramid.prepare(device, &self.queue, &self.depth.view, region);
            Some((region.render.width, region.render.height, levels))
        } else {
            None
        };
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
                let virtual_geometry =
                    virtual_enabled && virtual_meshes.mesh_index(batch.mesh).is_some();
                let levels = lod_levels(gpu_mesh, virtual_geometry);
                BatchMeta {
                    index_count: draw.index_count,
                    first_index: draw.first_index,
                    base_vertex: draw.base_vertex,
                    bounds: mesh.bounds(),
                    group: shadows.batch_group(index),
                    joints: mesh.joint_count(),
                    virtual_geometry,
                    lods: gpu_mesh.plan[..levels - 1]
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
                eye: snapshot.detail_eye,
                viewport: region.render.pixels(),
                motion: temporal_frame
                    .as_ref()
                    .is_some_and(|temporal| temporal.motion_valid),
                occlusion: occlusion.map(|(_, _, levels)| levels),
                subpixel: snapshot.environment.subpixel_culling.enabled,
                lod_error: if snapshot.environment.level_of_detail.enabled {
                    snapshot.environment.level_of_detail.pixel_error * region.share()[1]
                } else {
                    0.0
                },
                impostors: snapshot.environment.level_of_detail.impostors,
                stillness: (snapshot.environment.reflection_probes.enabled
                    && !snapshot.reflection_probes.list.is_empty())
                .then(|| self.features.probes.get().slots.waiting()),
                fade: self.switch_fades.share,
            };
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

    fn prepare_shadows(
        &mut self,
        snapshot: &RenderSnapshot,
        kind: FrameKind,
        temporal_frame: &Option<TemporalFrame>,
    ) -> Option<ShadowFrame> {
        let gpu_mode = kind.gpu_mode;

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
                    let lods = if mesh.skinned || !snapshot.environment.level_of_detail.enabled {
                        Vec::new()
                    } else {
                        mesh.lods
                            .iter()
                            .enumerate()
                            .map(|(level, &(offset, count, error))| {
                                let (offset, count) = mesh.caster_range(level + 1, (offset, count));
                                (draw.first_index + offset, count, error)
                            })
                            .collect()
                    };
                    let (offset, count) = mesh.caster_range(0, (0, draw.index_count));
                    CasterDraw {
                        index_count: count,
                        first_index: draw.first_index + offset,
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
                    .any(|terrain| terrain.drawn),
                pixel_angle: crate::gpu_data::pixel_angle(
                    snapshot.view_projection,
                    self.display.region.display.height,
                ),
            },
        )
    }

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
            (
                DERIVED_GLOW,
                lighting,
                at(offset_of!(LightingUniform, aerial)) + 12,
                4,
            ),
        ];
        copies.extend(self.features.clouds.get().sky_copies());
        let sun = snapshot.environment.atmosphere.sun_direction;
        let [night, _, moon] = crate::sky::night_lighting(&snapshot.environment, sun);
        self.features.sky.get_mut().encode(
            &self.device,
            &self.queue,
            (&snapshot.environment.sky, snapshot.tables.sky.as_ref()),
            (
                sun,
                snapshot.camera_position,
                snapshot.environment.atmosphere.exposure,
                snapshot.presented_seconds,
                [moon[0], moon[1], moon[2], night[1] * night[3]],
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
    ) {
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
        } else if wanted_cubes > self.features.environment.get().cubes() {
            wanted_cubes.next_power_of_two().min(most_cubes)
        } else {
            self.features.environment.get().cubes()
        };
        let keys = (
            self.features.clouds.get().environment_key(),
            self.features.sky.get().environment_key(),
        );
        let mut environment_encoder = self
            .features
            .environment
            .get()
            .needs_encoding(probe_settings.resolution, cubes, lighting, keys)
            .then(|| {
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("engine/environment-encoder"),
                    })
            });
        let (lost, sky_changed) = match environment_encoder.as_mut() {
            Some(encoder) => (
                self.features.environment.get_mut().ensure(
                    &self.device,
                    encoder,
                    probe_settings.resolution,
                    cubes,
                ),
                self.features
                    .environment
                    .get_mut()
                    .encode_sky(encoder, lighting, keys)
                    .1,
            ),
            None => (false, false),
        };
        let probes = self.features.probes.get_mut();
        probes
            .slots
            .plan(probe_list, (cubes - 1) as usize, lost, sky_changed);
        probes.upload(&self.device, &self.queue);
        probes.faces = 0;
        probes.dropped = (snapshot.reflection_probes.dropped as u64
            + (probe_list.len() as u64).saturating_sub(u64::from(most_cubes - 1)))
        .min(u64::from(u32::MAX)) as u32;
        self.refresh_camera_group();
        (environment_encoder, probe_list)
    }

    pub(super) fn refresh_camera_group(&mut self) {
        self.camera.refresh(
            &self.device,
            &camera::GroupSources {
                shadows: &self.shadows,
                environment: self.features.environment.get(),
                probes: self.features.probes.get(),
                horizons: &self.terrain.horizons,
                occupancy: self.features.occupancy.get(),
                sky: self.features.sky.get().bindings(),
                clouds: self.features.clouds.get().dome(),
            },
        );
    }

    fn prepare_lights(
        &mut self,
        frame: RenderFrame<'_>,
        kind: FrameKind,
        temporal_frame: &Option<TemporalFrame>,
    ) -> Option<Ready<LightFrame>> {
        let temporal = kind.temporal;
        let snapshot = frame.snapshot();

        let needs = snapshot.environment.local_lights.active() && !snapshot.lights.list.is_empty();
        let (device, queue, size) = (&self.device, &self.queue, self.display.region.render);
        let layouts = (&self.camera.layout, self.materials.layout());
        let temporal_pass = &mut self.features.temporal;
        let instances = &self.instances;
        self.features.lights.prepare(
            (device, needs),
            || LightPass::new(device),
            |lights| {
                if temporal && let Some(pass) = temporal_pass.get_mut() {
                    pass.ensure_lit_layout(
                        device,
                        [layouts.0, layouts.1, lights.fragment_layout()],
                    );
                }
                let parity = temporal_frame.as_ref().map_or(0, |frame| frame.parity);
                lights.prepare(
                    device,
                    queue,
                    snapshot,
                    size,
                    instances.frame_poses(parity),
                    (frame.identity(), frame.lights_revision()),
                )
            },
        )
    }

    fn prepare_post(
        &mut self,
        snapshot: &RenderSnapshot,
        temporal_frame: &Option<TemporalFrame>,
        display: bool,
    ) -> Option<Ready<Effects>> {
        let effects = crate::screen_effects::targets_read(&snapshot.environment);
        let region = self.display.region;
        let needs = temporal_frame.is_some() || effects || region.upscales();
        let (device, queue) = (&self.device, &self.queue);
        let format = self.display.format();
        let layouts = (&self.camera.layout, &self.camera.terrain_layout);
        let depth = &self.depth.view;
        let post = self.features.post.prepare(
            (device, needs),
            || PostPass::new(device, format, layouts),
            |post| {
                Some(post.begin(
                    device,
                    queue,
                    snapshot,
                    (region, depth),
                    display,
                    self.shadows.sky_bindings(),
                ))
            },
        );
        if !needs && let Some(post) = self.features.post.get_mut() {
            post.release_targets();
        }
        post
    }

    fn prepare_resolve(
        &mut self,
        temporal_frame: &Option<TemporalFrame>,
        transparent: bool,
    ) -> Option<Ready<ResolveFrame>> {
        let (Some(frame), Some(post)) = (temporal_frame, self.features.post.get()) else {
            return None;
        };
        let revealed = transparent
            .then(|| self.features.transparency.get().revealed())
            .flatten();
        let (device, depth) = (&self.device, &self.depth.view);
        self.features.temporal.prepare_created(|pass| {
            Some(pass.prepare_resolve(device, post, frame, depth, revealed))
        })
    }

    fn prepare_transparency(
        &mut self,
        snapshot: &RenderSnapshot,
        emitters: (u64, u64),
        post: bool,
    ) -> Option<Ready<TransparencyFrame>> {
        let (device, queue) = (&self.device, &self.queue);
        let inputs = (
            (self.display.region, &self.depth.view),
            (&self.camera.layout, (self.display.format(), DEPTH_FORMAT)),
        );
        self.features.transparency.prepare(|transparency| {
            transparency.prepare(device, queue, snapshot, emitters, inputs.0, inputs.1, post)
        })
    }

    pub(super) fn prepare_pipelines(
        &mut self,
        kind: FrameKind,
        temporal_frame: &Option<TemporalFrame>,
        features: &Features<'_>,
    ) -> Passes {
        let gpu_mode = kind.gpu_mode;
        let (light_frame, post_frame, probe_list) = (
            features.light_frame(),
            features.post_frame(),
            features.probe_list,
        );

        let lit_frame = light_frame.is_some() && self.features.lights.get().is_some();
        let targets = self.frame_targets(kind, temporal_frame, post_frame);
        let opaque_kind = PassKind::opaque(targets, lit_frame);
        self.ensure_variants(opaque_kind);
        if gpu_mode && !kind.debug && !probe_list.is_empty() {
            self.ensure_probe_pipelines(PassKind::probe(lit_frame));
        }
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
            impostor_pass,
        }
    }

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
        let resolving =
            virtual_frame && instance_count > 0 && self.prepare_visibility(snapshot, opaque_kind);
        if resolving {
            self.ensure_visibility_pipelines(opaque_kind);
            self.ensure_visibility_groups();
        }
        (virtual_frame, resolving)
    }

    pub(super) fn frame_shape(
        &self,
        kind: FrameKind,
        (packing, features, passes): (&Packing, &Features<'_>, &Passes),
        (virtual_frame, resolving, scatter, terrain): (bool, bool, bool, bool),
    ) -> FrameShape {
        let post = features.post_frame();
        let targets = self.frame_targets(kind, &packing.temporal_frame, post);
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
            deformation: self.terrain.deformation.pending(),
            bounce: self.terrain.bounce.relights(),
            subdivision: terrain && self.terrain.subdivides(),
            deferred: self.terrain_deferred(terrain, passes.opaque_kind),
            scatter,
            clusters: virtual_frame,
            shadow_clusters: virtual_frame && !self.virtual_draws.shadow_bins.is_empty(),
            resolves: [
                resolving,
                resolving && phases == 2 && self.virtual_draws.phase_two.is_some(),
            ],
            lights: features.light_frame().is_some(),
            effects: post.unwrap_or_default(),
            probes: kind.gpu_mode && !kind.debug && !features.probe_list.is_empty(),
        }
    }

    pub(super) fn prepare_cloud_composite(
        &mut self,
        targets: Targets,
    ) -> Option<Ready<CloudFrame>> {
        if targets == Targets::Debug {
            return None;
        }
        let target = self.cloud_target(targets);
        let (device, shader) = (&self.device, &self.mesh_shader);
        let camera = &self.camera.layout;
        self.features.clouds.prepare(|clouds| {
            if clouds.marches_view() {
                clouds.prepare_composite(device, shader, camera, target);
            }
            clouds.frame(target)
        })
    }

    pub(super) fn prepare_frame_groups(&mut self, state: &FrameState<'_>) {
        let temporal_frame = &state.packing.temporal_frame;

        if let (Some(_), Some(pass)) = (&temporal_frame, self.features.temporal.get_mut()) {
            let poses = self.instances.frame_poses(0);
            let both = [0, 1].map(|parity| poses.bound(parity).expect("temporal frames create it"));
            pass.prepare_pose_groups(
                &self.device,
                both,
                &poses.colors.bound(),
                poses.palettes.map(|palette| palette.bound()),
                &Bound::fixed(self.instances.grounds().as_entire_binding()),
            );
        }
        self.prepare_reflections(state);
    }

    fn prepare_reflections(&mut self, state: &FrameState<'_>) {
        let Some(frame) = state
            .features
            .frames
            .post
            .as_ref()
            .map(|post| *post.frame())
        else {
            return;
        };
        let Some(post) = self.features.post.get_mut() else {
            return;
        };
        let terrains = &state.snapshot.terrain;
        let infos: Vec<crate::scatter::TerrainInfo> = terrains
            .terrains
            .iter()
            .enumerate()
            .filter(|_| frame.reflections)
            .map(|(index, terrain)| crate::world_trace::terrain_info(&self.terrain, index, terrain))
            .collect();
        let cover = state.snapshot.environment.ground_cover.sanitized();
        let trace = frame.reflections.then(|| crate::world_trace::TraceFrame {
            terrains,
            infos: &infos,
            terrain: &self.terrain,
            placed: self.scatter_draws.placements(),
            cover: if cover.enabled { cover.density } else { 0.0 },
        });
        post.prepare_reflections(
            (&self.device, &self.queue),
            (frame, &self.depth.view),
            trace.as_ref(),
            (
                self.terrain.trace_sources(),
                self.features.scatter.get().slot_buffers(),
                self.impostors.placed.frames(),
            ),
        );
    }
}

fn canvas_clear() -> [f64; 4] {
    let encode = |linear: f64| 1.055 * linear.powf(1.0 / 2.4) - 0.055;
    let [r, g, b, a] = crate::render_graph::SCENE_CLEAR;
    [encode(r), encode(g), encode(b), a]
}
