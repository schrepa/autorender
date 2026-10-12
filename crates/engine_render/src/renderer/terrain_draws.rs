//! Terrain's draws: its preparation each frame, and its
//! patches in the opaque pass, shadow maps and reflection probe faces.

use super::frame::FrameState;
use super::*;
use crate::terrain::{FrameTerrain, TerrainViews};

/// Terrain's visibility target: each pixel's terrain (its
/// directory start plus one; 0 where none shows) and depth's bits, of the
/// depth target generation `generation`, and the group its resolves read
/// it through.
pub(super) struct TerrainIds {
    generation: u64,
    pub(super) view: wgpu::TextureView,
    group: wgpu::BindGroup,
}

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
        (probes, fades): (bool, bool),
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
        // Runtime deformation's map before the terrain's
        // uniforms, which shade with it.
        self.terrain.deformation.prepare(
            &self.device,
            &snapshot.deformation,
            snapshot.presented_seconds,
        );
        // Ground under cover turns to its canopy as blades thin out.
        let cover = snapshot.environment.ground_cover.sanitized();
        let canopy = (cover.enabled && cover.density > 0.0).then(|| {
            (
                cover,
                crate::ground_cover::pixels_per_meter(culling, self.display.size().1 as f32),
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
                fades,
                now: snapshot.elapsed_seconds,
                cover: canopy,
            },
            (self.display.max_layers, self.materials.mips()),
        );
        // What terrains draw with is created before they first draw,
        // so the GPU compiles it while they bake and load.
        self.warm_terrain(snapshot, opaque_kind, probes);
        if !drawn {
            return false;
        }
        // Ground cover on the terrain, seen through the
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
            self.display.size().1 as f32,
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
                self.terrain.pipelines.ensure_pipeline(
                    &self.device,
                    &self.mesh_shader,
                    layouts,
                    (targets, lit, holes),
                    self.display.format(),
                    DEPTH_FORMAT,
                    &self.camera.terrain_layout,
                );
            }
            if probes {
                self.terrain.pipelines.ensure_probe_pipeline(
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
            self.terrain.pipelines.ensure_shadow_pipeline(&self.device);
        }
        true
    }

    /// Creates the pipelines `snapshot`'s terrains draw with unless they
    /// exist, before any draws: their patches into
    /// `opaque_kind`'s targets, with holes where tiles may hold any, into
    /// reflection probe faces when `probes` capture and into shadow maps;
    /// their tiles' normal and sky maps; their cover's growth and blades;
    /// and their placed objects' culling and occupancy.
    fn warm_terrain(&mut self, snapshot: &RenderSnapshot, opaque_kind: PassKind, probes: bool) {
        let terrains = &snapshot.terrain.terrains;
        if terrains.is_empty() {
            return;
        }
        self.terrain.warm(&self.device);
        let lit = opaque_kind.lit;
        let lights = self.lights.as_ref().map(LightPass::fragment_layout);
        let layouts = (
            &self.camera.layout,
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
                    layouts,
                    (targets, lit, holes),
                    self.display.format(),
                    DEPTH_FORMAT,
                    &self.camera.terrain_layout,
                );
            }
            if probes {
                self.terrain.pipelines.ensure_probe_pipeline(
                    &self.device,
                    &self.mesh_shader,
                    layouts,
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
            self.ground_cover.warm(&self.device);
            if let Some(targets) = opaque_kind.targets() {
                self.ground_cover.ensure_pipeline(
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
            self.scatter.warm(&self.device);
            if snapshot.environment.occupancy.enabled {
                self.occupancy.warm(&self.device);
            }
        }
    }

    /// Prepares this frame's water and its pipeline into
    /// `opaque_kind`'s targets, created while terrains with hydrology still
    /// bake; returns whether any of it draws.
    pub(super) fn prepare_water(
        &mut self,
        snapshot: &RenderSnapshot,
        opaque_kind: PassKind,
    ) -> bool {
        let Some(targets) = opaque_kind.targets() else {
            return false;
        };
        let coming = snapshot
            .terrain
            .terrains
            .iter()
            .any(|terrain| terrain.hydrology && terrain.water.is_none());
        if coming {
            self.water.ensure_pipeline(
                &self.device,
                &self.mesh_shader,
                &self.camera.layout,
                (targets, true),
                (self.display.format(), DEPTH_FORMAT),
            );
        }
        let environment = &snapshot.environment;
        if !self.water.prepare(
            &self.device,
            &self.queue,
            &snapshot.terrain,
            (environment.water, environment.ocean),
            (
                snapshot.camera_position,
                environment.wind.velocity,
                snapshot.presented_seconds,
            ),
            &self.depth.texture,
        ) {
            return false;
        }
        self.water.ensure_pipeline(
            &self.device,
            &self.mesh_shader,
            &self.camera.layout,
            (targets, false),
            (self.display.format(), DEPTH_FORMAT),
        );
        // Terrain's visibility target, the depth target's
        // size.
        let generation = self.depth.view.generation();
        if self
            .terrain_ids
            .as_ref()
            .is_none_or(|ids| ids.generation != generation)
        {
            let size = self.depth.texture.size();
            let texture = crate::targets::texture_2d(
                &self.device,
                "engine/terrain-ids",
                (size.width, size.height),
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
            self.terrain_ids = Some(TerrainIds {
                generation,
                view,
                group,
            });
        }
        true
    }

    /// Whether this frame's terrain draws deferred, given whether it draws
    /// and the opaque passes' `kind`: shaded (its development views read what
    /// only its patches hold) into the effects' targets, its pipelines and
    /// target ready.
    pub(super) fn terrain_deferred(&self, terrain: bool, kind: PassKind) -> bool {
        terrain
            && self.terrain.patches.frame.shaded
            && self.terrain_ids.is_some()
            && kind.targets().is_some_and(|targets| {
                self.terrain
                    .pipelines
                    .resolves
                    .contains_key(&(targets, kind.lit))
            })
            && !self.terrain.patches.frame.terrains.is_empty()
    }

    /// The camera's terrain patches into terrain's visibility target;
    /// returns the draws.
    pub(super) fn draw_terrain_ids(&self, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        pass.set_bind_group(0, self.camera.group(), &[]);
        let mut draws = 0;
        for holes in [false, true] {
            let Some(pipeline) = self.terrain.pipelines.deferred_ids.get(&holes) else {
                continue;
            };
            pass.set_pipeline(pipeline);
            for terrain in &self.terrain.patches.frame.terrains {
                let range =
                    &FrameTerrain::split(&terrain.camera, terrain.camera_holes)[usize::from(holes)];
                draws += self.terrain.draw(pass, range, 0, terrain.uniform);
            }
        }
        draws
    }

    /// Each terrain's pixels shaded once from its visibility target;
    /// returns the draws.
    pub(super) fn draw_terrain_resolve(
        &self,
        state: &FrameState<'_>,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> u32 {
        let kind = state.passes.opaque_kind;
        let (Some(targets), Some(TerrainIds { group, .. })) = (kind.targets(), &self.terrain_ids)
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
            let Some(lights) = self.lights.as_ref() else {
                return 0;
            };
            pass.set_bind_group(3, lights.fragment_group(), &[]);
        }
        // Each within the pixels its patches cover: elsewhere it would only
        // read the visibility target and discard.
        let size = self.depth.texture.size();
        let mut draws = 0;
        for terrain in &self.terrain.patches.frame.terrains {
            if terrain.camera.is_empty() {
                continue;
            }
            let Some([x, y, width, height]) = terrain.scissor((size.width, size.height)) else {
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

    /// The terrain's cover, after it so depth tests leave only blades in
    /// view; returns the draws.
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
            let Some(lights) = self.lights.as_ref() else {
                return 0;
            };
            pass.set_bind_group(2, &self.mesh_pipelines.empty_group, &[]);
            pass.set_bind_group(3, lights.fragment_group(), &[]);
        }
        self.ground_cover.draw(pass, targets, kind.lit)
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
        pass.set_bind_group(0, self.camera.group(), &[]);
        if kind.lit {
            let Some(lights) = self.lights.as_ref() else {
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
                let range =
                    &FrameTerrain::split(&terrain.camera, terrain.camera_holes)[usize::from(holes)];
                draws += self.terrain.draw(pass, range, 0, terrain.uniform);
            }
        }
        draws + self.draw_terrain_cover(state, pass)
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
        if lit {
            pass.set_bind_group(2, &self.mesh_pipelines.empty_group, &[]);
        }
        for holes in [false, true] {
            let Some(pipeline) = self.terrain.pipelines.probe_pipeline(lit, holes) else {
                continue;
            };
            pass.set_pipeline(pipeline);
            for terrain in &self.terrain.patches.frame.terrains {
                let range =
                    &FrameTerrain::split(&terrain.all, terrain.all_holes)[usize::from(holes)];
                self.terrain.draw(pass, range, view, terrain.uniform);
            }
        }
    }
}
