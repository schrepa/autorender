//! Terrain's draws (decision 051): its preparation each frame, and its
//! patches in the opaque pass, shadow maps and reflection probe faces.

use super::frame::FrameState;
use super::*;
use crate::terrain::{FrameTerrain, TerrainViews};

impl Renderer {
    /// Prepares this frame's terrain: tile uploads, each view's patches and
    /// the pipelines drawing them into `opaque_kind`'s targets, the shadow
    /// maps and, when `probes` capture, reflection probe faces. Returns
    /// whether terrain draws.
    pub(super) fn prepare_terrain(
        &mut self,
        snapshot: &RenderSnapshot,
        temporal_frame: &Option<TemporalFrame>,
        opaque_kind: PassKind,
        probes: bool,
    ) -> bool {
        let terrains = &snapshot.terrain;
        if terrains.terrains.is_empty() && self.terrain.resident() == 0 {
            return false;
        }
        let jitter = temporal_frame.as_ref().map_or([0.0; 2], |frame| {
            [frame.camera.jitter[0], frame.camera.jitter[1]]
        });
        let culling = terrains.relative_view_projection;
        let maps = self.shadows.drawn_maps().to_vec();
        let drawn = self.terrain.prepare(
            &self.device,
            &self.queue,
            terrains,
            &TerrainViews {
                camera: crate::gpu_data::jittered(culling, jitter),
                culling,
                eye: snapshot.camera_position,
                maps: &maps,
                probes,
                settings: snapshot.environment.terrain.sanitized(),
                shadows: snapshot.environment.shadows.enabled,
                sun: snapshot
                    .environment
                    .atmosphere
                    .sanitized()
                    .sun_direction
                    .normalize(),
            },
            (self.display.max_layers, self.materials.mips()),
        );
        if !drawn {
            return false;
        }
        // Ground cover on the terrain (decision 056), seen through the
        // camera's unjittered view.
        let covered = self.ground_cover.prepare(
            &self.device,
            &self.queue,
            &self.terrain,
            terrains,
            (
                &snapshot.environment.ground_cover,
                snapshot.environment.wind.velocity,
                snapshot.view_projection,
                snapshot.camera_position,
                snapshot.elapsed_seconds,
            ),
        );
        let lit = opaque_kind.lit;
        let lights = self.lights.as_ref().map(LightPass::fragment_layout);
        let layouts = (
            &self.camera.layout,
            &self.mesh_pipelines.empty_layout,
            lights,
        );
        // Patches with holes draw with their own pipelines, created with the
        // first frame that has them.
        let variants: &[bool] = if self.terrain.has_holes() {
            &[false, true]
        } else {
            &[false]
        };
        if let (true, Some(targets)) = (covered, opaque_kind.targets()) {
            self.ground_cover.ensure_pipeline(
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
                self.terrain.ensure_pipeline(
                    &self.device,
                    &self.mesh_shader,
                    layouts,
                    (targets, lit, holes),
                    self.display.format(),
                    DEPTH_FORMAT,
                );
            }
            if probes {
                self.terrain.ensure_probe_pipeline(
                    &self.device,
                    &self.mesh_shader,
                    layouts,
                    (lit, holes),
                    ENVIRONMENT_FORMAT,
                    DEPTH_FORMAT,
                );
            }
        }
        if !maps.is_empty() {
            self.terrain.ensure_shadow_pipeline(&self.device);
        }
        true
    }

    /// Prepares this frame's water (decision 067) and its pipeline into
    /// `opaque_kind`'s targets; returns whether any of it draws.
    pub(super) fn prepare_water(
        &mut self,
        snapshot: &RenderSnapshot,
        opaque_kind: PassKind,
    ) -> bool {
        let Some(targets) = opaque_kind.targets() else {
            return false;
        };
        let environment = &snapshot.environment;
        if !self.water.prepare(
            &self.device,
            &self.queue,
            &snapshot.terrain,
            (environment.water, environment.ocean),
            (
                snapshot.camera_position,
                environment.wind.velocity,
                snapshot.elapsed_seconds,
            ),
            &self.depth.texture,
        ) {
            return false;
        }
        self.water.ensure_pipeline(
            &self.device,
            &self.mesh_shader,
            &self.camera.layout,
            targets,
            (self.display.format(), DEPTH_FORMAT),
        );
        true
    }

    /// Draws the camera's terrain patches into the opaque pass; returns the
    /// draws.
    pub(super) fn draw_terrain(
        &self,
        state: &FrameState<'_>,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> u32 {
        let kind = state.passes.opaque_kind;
        let Some(targets) = kind.targets().filter(|_| state.terrain) else {
            return 0;
        };
        pass.set_bind_group(0, &self.camera.group, &[]);
        if kind.lit {
            let Some(lights) = self.lights.as_ref() else {
                return 0;
            };
            pass.set_bind_group(2, &self.mesh_pipelines.empty_group, &[]);
            pass.set_bind_group(3, lights.fragment_group(), &[]);
        }
        let mut draws = 0;
        for (holes, pipeline) in [false, true]
            .into_iter()
            .filter_map(|holes| Some((holes, self.terrain.pipeline(targets, kind.lit, holes)?)))
        {
            pass.set_pipeline(pipeline);
            for terrain in &self.terrain.frame.terrains {
                let range =
                    &FrameTerrain::split(&terrain.camera, terrain.camera_holes)[usize::from(holes)];
                draws += self.terrain.draw(pass, range, 0, terrain.uniform);
            }
        }
        // Its cover, after it so depth tests leave only blades in view.
        draws += self.ground_cover.draw(pass, targets, kind.lit);
        draws
    }

    /// Draws every terrain patch into reflection probe face `face` of the
    /// probe at `center`, after its meshes (`lit` when lights shade it).
    pub(super) fn draw_terrain_face(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        face: usize,
        center: engine_core::glam::Vec3,
        lit: bool,
    ) {
        if self.terrain.frame.terrains.is_empty() {
            return;
        }
        let view = self.terrain.write_probe_view(
            &self.queue,
            face,
            face_view_projection(engine_core::glam::Vec3::ZERO, face),
            self.terrain.eye(),
            center,
        );
        if lit {
            pass.set_bind_group(2, &self.mesh_pipelines.empty_group, &[]);
        }
        for holes in [false, true] {
            let Some(pipeline) = self.terrain.probe_pipeline(lit, holes) else {
                continue;
            };
            pass.set_pipeline(pipeline);
            for terrain in &self.terrain.frame.terrains {
                let range =
                    &FrameTerrain::split(&terrain.all, terrain.all_holes)[usize::from(holes)];
                self.terrain.draw(pass, range, view, terrain.uniform);
            }
        }
    }
}
