use super::frame::FrameState;
use super::*;
use crate::terrain::TerrainViews;
use crate::water::WaterFrame;
use crate::water_particles::ParticleView;

pub(super) struct TerrainIds {
    pub(super) view: wgpu::TextureView,
    group: wgpu::BindGroup,
}

impl Renderer {
    pub(super) fn prepare_terrain(
        &mut self,
        snapshot: &RenderSnapshot,
        temporal_frame: &Option<TemporalFrame>,
        opaque_kind: PassKind,
        (probes, fades): (bool, bool),
    ) -> bool {
        let terrains = &snapshot.terrain;
        if terrains.terrains.is_empty() && self.terrain.resident() == 0 {
            self.terrain.rest();
            return false;
        }
        let jitter = temporal_frame.as_ref().map_or([0.0; 2], |frame| {
            [frame.camera.jitter[0], frame.camera.jitter[1]]
        });
        let culling = terrains.relative_view_projection;
        let casts = snapshot.environment.shadows.enabled;
        let cover = snapshot.environment.ground_cover.sanitized();
        let canopy = (cover.enabled && cover.density > 0.0).then(|| {
            (
                cover,
                crate::ground_cover::pixels_per_meter(culling, self.display.size().height as f32),
            )
        });
        let drawn = self.terrain.prepare(
            &self.device,
            &self.queue,
            terrains,
            &TerrainViews {
                camera: crate::gpu_data::jittered(culling, jitter),
                culling,
                eye: snapshot.camera_position,
                detail_eye: snapshot.detail_eye,
                regions: self.shadows.region_views(),
                settings: snapshot.environment.terrain.sanitized(),
                shadows: snapshot.environment.shadows.enabled,
                water: snapshot.environment.water.sanitized(),
                fades,
                now: snapshot.elapsed_seconds,
                cover: canopy,
            },
            (self.display.max_layers, self.materials.mips()),
        );
        for change in self.terrain.ground_changes() {
            self.shadows
                .dirty_ground((change.min, change.max), (&change.pyramid, change.level));
        }
        self.warm_terrain(snapshot, opaque_kind, probes);
        if !drawn {
            return false;
        }
        let covered = self.features.ground_cover.get_mut().prepare(
            &self.device,
            &self.queue,
            &self.terrain,
            terrains,
            (
                &snapshot.environment.ground_cover,
                snapshot.environment.wind.velocity,
                snapshot.view_projection,
                snapshot.detail_eye,
                snapshot.elapsed_seconds,
            ),
            (
                self.display.size().height as f32,
                snapshot.environment.foliage.sanitized().ground_blend > 0.0,
            ),
        );
        let lit = opaque_kind.lit;
        let lights = self.features.lights.get().map(LightPass::fragment_layout);
        let layouts = (
            &self.camera.layout,
            &self.mesh_pipelines.empty_layout,
            lights,
        );
        let terrain_layouts = (
            &self.camera.terrain_layout,
            &self.mesh_pipelines.empty_layout,
            lights,
        );
        let variants: &[bool] = if self.terrain.has_holes() {
            &[false, true]
        } else {
            &[false]
        };
        if let (true, Some(targets)) = (covered, opaque_kind.targets()) {
            self.features.ground_cover.get_mut().ensure_pipeline(
                &self.device,
                &self.mesh_shader,
                layouts,
                (targets, lit),
                self.display.format(),
                DEPTH_FORMAT,
            );
        }
        for &holes in variants {
            if let Some(targets) = opaque_kind.targets() {
                self.terrain.pipelines.ensure_pipeline(
                    &self.device,
                    &self.mesh_shader,
                    terrain_layouts,
                    (targets, lit, holes),
                    self.display.format(),
                    DEPTH_FORMAT,
                );
            }
            if probes {
                self.terrain.pipelines.ensure_probe_pipeline(
                    &self.device,
                    &self.mesh_shader,
                    terrain_layouts,
                    (lit, holes),
                    ENVIRONMENT_FORMAT,
                    DEPTH_FORMAT,
                );
            }
        }
        if casts {
            self.terrain.pipelines.ensure_shadow_pipeline(&self.device);
        }
        if opaque_kind.targets().is_some() {
            self.prepare_terrain_ids();
        }
        true
    }

    fn prepare_terrain_ids(&mut self) {
        let size = self.display.size();
        if self.terrain_ids.resizes(size) {
            let texture = crate::targets::texture_2d(
                &self.device,
                "engine/terrain-ids",
                size.pixels(),
                1,
                1,
                wgpu::TextureFormat::Rg32Uint,
                crate::targets::RENDER_TARGET,
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let group = crate::pipelines::group(
                &self.device,
                "engine/terrain-resolve-group",
                &crate::bindings::TERRAIN_RESOLVE,
                &self.terrain.pipelines.resolve_layout,
                &[("terrain_ids_texture", crate::pipelines::view(&view))],
            );
            self.terrain_ids.set(size, TerrainIds { view, group });
        }
    }

    fn warm_terrain(&mut self, snapshot: &RenderSnapshot, opaque_kind: PassKind, probes: bool) {
        let terrains = &snapshot.terrain.terrains;
        if terrains.is_empty() {
            return;
        }
        self.terrain.warm(&self.device);
        let lit = opaque_kind.lit;
        let lights = self.features.lights.get().map(LightPass::fragment_layout);
        let layouts = (
            &self.camera.layout,
            &self.mesh_pipelines.empty_layout,
            lights,
        );
        let terrain_layouts = (
            &self.camera.terrain_layout,
            &self.mesh_pipelines.empty_layout,
            lights,
        );
        let holes = terrains.iter().any(|terrain| terrain.may_hole) || self.terrain.has_holes();
        let variants: &[bool] = if holes { &[false, true] } else { &[false] };
        for &holes in variants {
            if let Some(targets) = opaque_kind.targets() {
                self.terrain.pipelines.ensure_pipeline(
                    &self.device,
                    &self.mesh_shader,
                    terrain_layouts,
                    (targets, lit, holes),
                    self.display.format(),
                    DEPTH_FORMAT,
                );
            }
            if probes {
                self.terrain.pipelines.ensure_probe_pipeline(
                    &self.device,
                    &self.mesh_shader,
                    terrain_layouts,
                    (lit, holes),
                    ENVIRONMENT_FORMAT,
                    DEPTH_FORMAT,
                );
            }
        }
        if snapshot.environment.shadows.enabled {
            self.terrain.pipelines.ensure_shadow_pipeline(&self.device);
        }
        let cover = snapshot.environment.ground_cover.sanitized();
        let grows = cover.enabled
            && cover.density > 0.0
            && terrains.iter().any(|terrain| {
                !terrain.surroundings && terrain.surface.iter().any(|look| look.cover.is_some())
            });
        if grows {
            self.features.ground_cover.get_mut().warm(&self.device);
            if let Some(targets) = opaque_kind.targets() {
                self.features.ground_cover.get_mut().ensure_pipeline(
                    &self.device,
                    &self.mesh_shader,
                    layouts,
                    (targets, lit),
                    self.display.format(),
                    DEPTH_FORMAT,
                );
            }
        }
        if terrains.iter().any(|terrain| !terrain.assets.is_empty()) {
            self.features.scatter.get_mut().warm(&self.device);
            if snapshot.environment.occupancy.enabled {
                self.features.occupancy.get_mut().warm(&self.device);
            }
        }
    }

    pub(super) fn prepare_water(
        &mut self,
        snapshot: &RenderSnapshot,
        opaque_kind: PassKind,
    ) -> Option<Ready<WaterFrame>> {
        let targets = opaque_kind.targets()?;
        let coming = snapshot
            .terrain
            .terrains
            .iter()
            .any(|terrain| terrain.hydrology);
        let (device, shader, camera) = (&self.device, &self.mesh_shader, &self.camera.water_layout);
        let formats = (self.display.format(), DEPTH_FORMAT);
        let tiles = self.terrain.water_draws(&snapshot.terrain);
        let display = self.display.region.display;
        let seen = ParticleView {
            detail: snapshot.detail_eye,
            pixel_angle: crate::gpu_data::pixel_angle(snapshot.view_projection, display.height),
            display,
        };
        self.features.water.prepare(|water| {
            if coming {
                water.ensure_pipeline(device, shader, camera, (targets, true), formats);
            }
            let environment = &snapshot.environment;
            let prepared = water.prepare(
                device,
                &self.queue,
                (tiles, &snapshot.terrain.terrains),
                environment.water,
                (seen, environment.wind, snapshot.presented_seconds),
                (
                    &self.depth.texture,
                    matches!(targets, Targets::Post | Targets::Temporal),
                ),
            )?;
            water.ensure_pipeline(device, shader, camera, (targets, false), formats);
            Some(prepared)
        })
    }

    pub(super) fn terrain_deferred(&self, terrain: bool, kind: PassKind) -> bool {
        terrain
            && self.terrain_ids.get().is_some()
            && kind.targets().is_some_and(|targets| {
                self.terrain
                    .pipelines
                    .resolves
                    .contains_key(&(targets, kind.lit))
            })
            && !self.terrain.patches.frame.terrains.is_empty()
    }

    pub(super) fn draw_terrain_ids(&self, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        pass.set_bind_group(0, self.camera.terrain_group(), &[]);
        let mut draws = 0;
        for holes in [false, true] {
            let Some(pipeline) = self.terrain.pipelines.deferred_ids.get(&holes) else {
                continue;
            };
            pass.set_pipeline(pipeline);
            for terrain in &self.terrain.patches.frame.terrains {
                if terrain.holes == holes {
                    draws += self.terrain.draw_surface(pass, terrain, 0);
                }
            }
        }
        draws
    }

    pub(super) fn draw_terrain_resolve(
        &self,
        state: &FrameState<'_>,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> u32 {
        let kind = state.passes.opaque_kind;
        let (Some(targets), Some(TerrainIds { group, .. })) =
            (kind.targets(), self.terrain_ids.get().map(|ids| &**ids))
        else {
            return 0;
        };
        let Some(pipeline) = self.terrain.pipelines.resolves.get(&(targets, kind.lit)) else {
            return 0;
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, self.camera.terrain_group(), &[]);
        pass.set_bind_group(2, group, &[]);
        if kind.lit {
            let Some(lights) = self.features.lights.get() else {
                return 0;
            };
            pass.set_bind_group(3, lights.fragment_group(), &[]);
        }
        let render = self.display.region.render;
        let mut draws = 0;
        for terrain in &self.terrain.patches.frame.terrains {
            let Some([x, y, width, height]) = terrain.scissor(render.pixels()) else {
                continue;
            };
            if !self.terrain.bind(pass, 0, terrain.uniform) {
                continue;
            }
            pass.set_scissor_rect(x, y, width, height);
            pass.draw(0..3, 0..1);
            draws += 1;
        }
        draws
    }

    pub(super) fn draw_terrain_cover(
        &self,
        state: &FrameState<'_>,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> u32 {
        let kind = state.passes.opaque_kind;
        let Some(targets) = kind.targets().filter(|_| state.terrain) else {
            return 0;
        };
        pass.set_bind_group(0, self.camera.group(), &[]);
        if kind.lit {
            let Some(lights) = self.features.lights.get() else {
                return 0;
            };
            pass.set_bind_group(2, &self.mesh_pipelines.empty_group, &[]);
            pass.set_bind_group(3, lights.fragment_group(), &[]);
        }
        self.features
            .ground_cover
            .get()
            .draw(pass, targets, kind.lit)
    }

    pub(super) fn draw_terrain_patches(
        &self,
        state: &FrameState<'_>,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> u32 {
        let kind = state.passes.opaque_kind;
        let Some(targets) = kind.targets().filter(|_| state.terrain) else {
            return 0;
        };
        pass.set_bind_group(0, self.camera.terrain_group(), &[]);
        if kind.lit {
            let Some(lights) = self.features.lights.get() else {
                return 0;
            };
            pass.set_bind_group(2, &self.mesh_pipelines.empty_group, &[]);
            pass.set_bind_group(3, lights.fragment_group(), &[]);
        }
        let mut draws = 0;
        for (holes, pipeline) in [false, true].into_iter().filter_map(|holes| {
            Some((
                holes,
                self.terrain.pipelines.pipeline(targets, kind.lit, holes)?,
            ))
        }) {
            pass.set_pipeline(pipeline);
            for terrain in &self.terrain.patches.frame.terrains {
                if terrain.holes == holes {
                    draws += self.terrain.draw_surface(pass, terrain, 0);
                }
            }
        }
        draws
    }

    pub(super) fn draw_terrain_face(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        face: usize,
        center: engine_core::glam::Vec3,
        lit: bool,
    ) {
        if self.terrain.patches.frame.terrains.is_empty() {
            return;
        }
        let view = self.terrain.write_probe_view(
            &self.queue,
            face,
            face_view_projection(engine_core::glam::Vec3::ZERO, face),
            self.terrain.eye(),
            center,
        );
        pass.set_bind_group(0, self.camera.terrain_probe_group(face), &[]);
        if lit {
            pass.set_bind_group(2, &self.mesh_pipelines.empty_group, &[]);
        }
        for holes in [false, true] {
            let Some(pipeline) = self.terrain.pipelines.probe_pipeline(lit, holes) else {
                continue;
            };
            pass.set_pipeline(pipeline);
            for terrain in &self.terrain.patches.frame.terrains {
                if terrain.holes == holes {
                    self.terrain.draw_surface(pass, terrain, view);
                }
            }
        }
    }
}
