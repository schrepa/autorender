use super::*;
use crate::graph::FrameShape;
use crate::memory::buffer_init;
use crate::parts::PartAssets;

impl Renderer {
    pub(super) fn plan_frame(
        &mut self,
        frame: RenderFrame<'_>,
        display: bool,
    ) -> Result<FrameKind, Error> {
        let snapshot = frame.snapshot();
        {
            let _profile = engine_profile::span("render", "batch_plan");
            self.core.plan.prepare_frame(&frame)?;
        }
        let gpu_mode = frame.gpu().is_some();
        if self.core.gpu_frame && !gpu_mode {
            self.core.instances.retained_uploads.invalidate();
            self.core.instances.slot_revision = None;
        }
        self.core.gpu_frame = gpu_mode;
        let mut out = BeginOut::default();
        let (parts, core) = (&mut self.parts, &mut self.core);
        for index in 0..parts.len() {
            let _owner = crate::memory::owned(parts.decl(index).brick);
            parts.with_part(index, |part, parts| {
                part.begin(
                    &mut Begin {
                        core,
                        parts,
                        snapshot,
                        frame,
                        gpu_mode,
                    },
                    &mut out,
                );
            });
        }
        let motion = ["motion", "previous_poses"]
            .into_iter()
            .any(|name| self.parts.wants(self.table.resource(name)));
        if out.forget_temporal || (self.core.previous.wanted && !motion) {
            self.forget_temporal();
        }
        self.core.previous.wanted = motion;
        let debug = self.shows_debug_view();
        let post = self.parts.post().is_some_and(|post| !post.disabled());
        let temporal = display && gpu_mode && !debug && motion && post;
        if !temporal {
            self.core.previous.reset();
        }
        self.core.display.plan_region(display && !debug && post);
        let profile_residency = engine_profile::span("render", "validate_and_retire");
        for (index, source) in snapshot.textures.iter().enumerate() {
            let side = source.texture.width().max(source.texture.height());
            if side > self.core.display.max_dimension {
                return Err(failed(&format!("$.snapshot.textures[{index}]"))
                    .at_most(u64::from(self.core.display.max_dimension))
                    .got(side));
            }
        }
        let assets = self.parts.assets(snapshot);
        self.retire_meshes(&assets, self.parts.asset_generation());
        self.core.materials.retain(&self.core.plan, &assets);
        drop(profile_residency);
        Ok(FrameKind {
            gpu_mode,
            temporal,
            display,
            debug,
            instance_count: self.core.plan.ordered_instances.len(),
        })
    }

    fn retire_meshes(&mut self, assets: &PartAssets<'_>, generation: u64) {
        let core = &mut self.core;
        let residency = (core.plan.rebuilds(), generation);
        if core.structure_keys.residency == Some(residency) {
            return;
        }
        core.structure_keys.residency = Some(residency);
        let kept: BTreeSet<MeshHandle> = assets.meshes.iter().map(|&(mesh, _)| mesh).collect();
        let (plan, arena) = (&core.plan, &mut core.mesh_arena);
        let resident = core.meshes.len();
        core.meshes.retain(|handle, mesh| {
            let keep = plan.contains_mesh(*handle) || kept.contains(handle);
            if !keep && let Ok(placement) = mesh.placement {
                arena.remove(placement);
            }
            keep
        });
        if core.meshes.len() != resident {
            core.mesh_layout += 1;
        }
    }

    pub(in crate::renderer) fn forget_temporal(&mut self) {
        self.core.forget_temporal();
        self.core.previous.release();
        for index in 0..self.parts.len() {
            let _owner = crate::memory::owned(self.parts.decl(index).brick);
            self.parts
                .with_part(index, |part, _| part.forget_targets(Targets::Temporal));
        }
    }

    pub(super) fn upload_assets(
        &mut self,
        snapshot: &RenderSnapshot,
        kind: FrameKind,
        prelude: &mut Prelude,
    ) -> AssetUploads {
        let gpu_mode = kind.gpu_mode;
        let assets = self.parts.assets(snapshot);
        let generation = self.parts.asset_generation();
        let core = &mut self.core;

        let profile_upload = engine_profile::span("render", "asset_uploads");
        let mut mesh_upload_bytes = 0_u64;
        let uploads = (core.plan.rebuilds(), generation, core.mesh_layout);
        let mesh_uploads = core.structure_keys.uploads != Some(uploads);
        let sources = core
            .plan
            .batches
            .iter()
            .map(|batch| (batch.mesh, &snapshot.meshes[batch.source_index]))
            .chain(assets.meshes.iter().copied())
            .filter(|_| mesh_uploads);
        for (handle, rendered) in sources {
            let source = &rendered.mesh;
            if let std::collections::btree_map::Entry::Vacant(entry) = core.meshes.entry(handle) {
                core.mesh_uploads.staging.clear();
                core.mesh_uploads
                    .staging
                    .reserve_exact(source.vertices().len());
                core.mesh_uploads
                    .staging
                    .extend(source.vertices().iter().map(Vertex::from));
                let extras = ExtraVertex::stream(source);
                let mut streams: Vec<&[u8]> =
                    vec![bytemuck::cast_slice(&core.mesh_uploads.staging)];
                if let Some(extras) = &extras {
                    streams.push(bytemuck::cast_slice(extras));
                }
                let (indices, lods, casters) = with_levels(source, rendered.lods.as_deref());
                let index_count = source.indices().len() as u32;
                let without_levels = rendered.lods.is_none();
                let placement = core
                    .mesh_arena
                    .insert(
                        &core.device,
                        &core.queue,
                        &streams,
                        core.mesh_uploads.staging.len() as u32,
                        &indices,
                    )
                    .ok_or_else(|| {
                        let buffer = |label, contents: &[u8], usage| {
                            buffer_init(
                                &core.device,
                                &wgpu::util::BufferInitDescriptor {
                                    label: Some(label),
                                    contents,
                                    usage,
                                },
                            )
                        };
                        let vertex = wgpu::BufferUsages::VERTEX;
                        OwnMesh {
                            vertices: buffer("engine/mesh-vertices", streams[0], vertex),
                            extras: extras.as_ref().map(|stream| {
                                buffer("engine/mesh-extras", bytemuck::cast_slice(stream), vertex)
                            }),
                            indices: buffer(
                                "engine/mesh-indices",
                                bytemuck::cast_slice(&indices),
                                wgpu::BufferUsages::INDEX,
                            ),
                        }
                    });
                core.mesh_layout += 1;
                let byte_len = source.byte_len() as u64;
                entry.insert(GpuMesh {
                    placement,
                    extended: extras.is_some(),
                    index_count,
                    byte_len,
                    plan: plan_levels(&lods),
                    lods,
                    casters,
                    without_levels,
                });
                mesh_upload_bytes += byte_len;
            }
        }
        if mesh_uploads {
            core.structure_keys.uploads =
                Some((core.plan.rebuilds(), generation, core.mesh_layout));
        }
        let mut mips = None;
        let (texture_upload_bytes, mip_levels) = core.materials.upload(
            &core.device,
            &core.queue,
            snapshot,
            &core.plan,
            &assets,
            &mut mips,
        );
        core.prepare_impostors(snapshot, gpu_mode, &mut mips);
        match mips {
            Some(encoder) => prelude.push_encoder(PreludeStage::Uploads, "pre_mips", encoder),
            None => prelude.reserve("pre_mips"),
        }
        drop(profile_upload);
        core.order_draws(snapshot);
        AssetUploads {
            mesh_bytes: mesh_upload_bytes,
            texture_bytes: texture_upload_bytes,
            mip_levels,
        }
    }

    pub(super) fn pack_instances(&mut self, info: &mut FrameInfo<'_>) -> Packing {
        let FrameKind {
            gpu_mode, temporal, ..
        } = info.kind;
        let frame = info.frame;
        let snapshot = info.snapshot;

        let profile_packing = engine_profile::span("render", "instance_buffer_and_packing");
        self.core
            .instances
            .reserve(&self.core.device, snapshot.instances.len(), temporal);
        if temporal {
            let mut jitter = None;
            self.steps(info, |part, cx| {
                if jitter.is_none() {
                    jitter = part.jitter(cx);
                }
            });
            let core = &mut self.core;
            let generation = core.instances.generation();
            info.temporal = Some(core.previous.frame(
                &core.device,
                &frame,
                (core.display.region, generation),
                jitter,
            ));
        }
        let core = &mut self.core;
        let slot_key = Some((frame.identity(), frame.structure_revision()));
        let write_slots = !gpu_mode
            && core
                .instances
                .stage_slots(&core.device, &core.plan.ordered_instances, slot_key);
        if core.instances.color_staging.capacity() < core.instances.capacity {
            core.instances
                .color_staging
                .reserve_exact(core.instances.capacity - core.instances.color_staging.len());
        }
        let prepared = core
            .instances
            .retained_uploads
            .prepare(&frame, &mut core.instances.color_staging);
        drop(profile_packing);
        Packing {
            write_slots,
            slot_key,
            repacked: prepared.repacked,
            poses_repacked: prepared.poses_repacked,
            colors_repacked: prepared.colors_repacked,
            full_upload: prepared.full_upload,
            fallback: prepared.fallback,
        }
    }

    pub(super) fn write_frame_data(
        &mut self,
        info: &FrameInfo<'_>,
        packing: &Packing,
        prelude: &mut Prelude,
    ) -> Writes {
        let frame = info.frame;
        let snapshot = info.snapshot;
        let gpu_mode = info.kind.gpu_mode;
        let temporal_frame = &info.temporal;
        let (write_slots, slot_key) = (packing.write_slots, packing.slot_key);
        let core = &mut self.core;

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
        camera.set_region(snapshot.view_projection, core.display.region);
        camera.set_detail(snapshot.detail_eye);
        let seconds = snapshot.elapsed_seconds;
        let previous = core.last_seconds.replace(seconds).unwrap_or(seconds);
        let wind = snapshot.wind;
        camera.wind = [
            wind[0],
            wind[1],
            wind[2],
            if temporal_frame.is_some() {
                previous
            } else {
                seconds
            },
        ];
        let mut camera_upload_bytes = 0_u64;
        if core.uploads.camera_needs_write(&camera) {
            core.queue
                .write_buffer(&core.camera.buffer, 0, bytemuck::bytes_of(&camera));
            core.uploads.commit_camera(camera);
            camera_upload_bytes = size_of::<CameraUniform>() as u64;
        }
        let mut color_upload_bytes = 0_u64;
        let mut slot_upload_bytes = 0_u64;
        let pose_ranges = if gpu_mode {
            &[][..]
        } else {
            core.instances.retained_uploads.pose_ranges()
        };
        let instance_write_ranges = pose_ranges.len()
            + core.instances.retained_uploads.color_ranges().len()
            + usize::from(write_slots);
        let pose_rows: &[u8] = bytemuck::cast_slice(frame.pose_rows());
        let stride = size_of::<PoseData>();
        let pose_upload_bytes: u64 = pose_ranges
            .iter()
            .map(|range| ((range.end - range.start) * stride) as u64)
            .sum();
        let mut staged = false;
        if pose_upload_bytes as usize >= staging::MIN_STAGED_BYTES {
            let mut encoder = core
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine/pose-staging-copies"),
                });
            let outcome = core.instances.pose_uploads.record(
                &core.device,
                &mut encoder,
                &core.instances.buffer,
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
            if staged {
                prelude.push_buffer(PreludeStage::Uploads, "pre_poses", encoder.finish());
            }
        }
        if !staged {
            for range in pose_ranges {
                core.queue.write_buffer(
                    &core.instances.buffer,
                    (range.start * stride) as u64,
                    &pose_rows[range.start * stride..range.end * stride],
                );
            }
        }
        for range in core.instances.retained_uploads.color_ranges() {
            let records = &core.instances.color_staging[range.start..range.end];
            core.queue.write_buffer(
                core.instances.colors(),
                (range.start * size_of::<ColorData>()) as u64,
                bytemuck::cast_slice(records),
            );
            color_upload_bytes += std::mem::size_of_val(records) as u64;
        }
        if write_slots {
            core.queue.write_buffer(
                core.instances.slots(),
                0,
                bytemuck::cast_slice(&core.instances.slot_staging),
            );
            slot_upload_bytes = (core.instances.slot_staging.len() * size_of::<u32>()) as u64;
        }
        let instance_upload_bytes = pose_upload_bytes + color_upload_bytes + slot_upload_bytes;
        core.instances.retained_uploads.commit();
        core.instances.slot_revision = slot_key;
        Writes {
            camera_bytes: camera_upload_bytes,
            instance_bytes: instance_upload_bytes,
            pose_bytes: pose_upload_bytes,
            color_bytes: color_upload_bytes,
            slot_bytes: slot_upload_bytes,
            ranges: instance_write_ranges,
        }
    }

    pub(super) fn prepare_pipelines(&mut self, info: &FrameInfo<'_>) -> Passes {
        let kind = info.kind;
        let gpu_mode = kind.gpu_mode;
        let targets = Self::frame_targets(info);
        let opaque_kind = PassKind::opaque(targets);
        let meshes = self.core.mesh_entries.is_some();
        if meshes {
            self.core.ensure_variants(opaque_kind);
        }
        let views = (gpu_mode && !kind.debug)
            .then(|| self.parts.views())
            .flatten()
            .map(|views| (views.format, views.mirrored));
        self.core.views = views;
        if meshes && let Some((format, mirrored)) = views {
            self.core.ensure_view_variants(format, mirrored);
        }
        let impostor_pass = if meshes
            && gpu_mode
            && self
                .core
                .impostors
                .records
                .iter()
                .any(|record| record.sphere[3] > 0.0)
        {
            self.core.prepare_impostor_draws(opaque_kind)
        } else {
            None
        };
        Passes {
            opaque_kind,
            impostor_pass,
        }
    }

    fn frame_targets(info: &FrameInfo<'_>) -> Targets {
        match &info.temporal {
            _ if info.kind.debug => Targets::Debug,
            Some(_) => Targets::Temporal,
            None => Targets::Post,
        }
    }

    pub(super) fn frame_shape(&self, info: &FrameInfo<'_>) -> FrameShape {
        let gpu_mode = info.kind.gpu_mode;
        let phases = if gpu_mode && self.core.gpu_presentation.occlusion() {
            2
        } else {
            1
        };
        FrameShape {
            gpu: gpu_mode,
            targets: Self::frame_targets(info),
            phases,
            deferred: self.parts.iter().any(|(_, part)| part.defers()),
        }
    }

    pub(super) fn refresh_shared(&mut self, info: &FrameInfo<'_>) {
        let (parts, core) = (&self.parts, &mut self.core);
        let cameras = (info.kind.gpu_mode && !info.kind.debug)
            .then(|| parts.views())
            .flatten()
            .map(|views| views.cameras);
        core.camera
            .refresh(&core.device, &core.shared, parts, cameras);
    }
}
