//! Terrain's draws: its preparation each frame, and its
//! subdivisions' leaves in the opaque pass and reflection probe faces.

use super::frame::FrameState;
use super::*;
use crate::terrain::TerrainViews;
use crate::water::WaterFrame;
use crate::water_foam_field::FoamView;

/// Terrain's visibility target: each pixel's terrain (its
/// directory start plus one; 0 where none shows) and depth's bits, and the
/// group its resolves read it through.
pub(super) struct TerrainIds {
    pub(super) view: wgpu::TextureView,
    group: wgpu::BindGroup,
}

impl Renderer {
    /// Prepares this frame's terrain: tile uploads, its subdivisions and the
    /// pipelines drawing their leaves into `opaque_kind`'s targets, the
    /// shadow maps and, when `probes` capture, reflection probe faces.
    /// Returns whether terrain draws.
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
        // The shadow clipmap redraws its tiles over ground that arrived, left
        // or changed.
        for change in self.terrain.ground_changes() {
            self.shadows
                .dirty_ground((change.min, change.max), (&change.pyramid, change.level));
        }
        // What terrains draw with is created before they first draw,
        // so the GPU compiles it while they bake and load.
        self.warm_terrain(snapshot, opaque_kind, probes);
        if !drawn {
            return false;
        }
        // Ground cover on the terrain, seen through the
        // camera's unjittered view, growing and retiring by its distance
        // from the detail eye.
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
        // Terrain's pipelines over its own first group, its cover's over the
        // main one.
        let terrain_layouts = (
            &self.camera.terrain_layout,
            &self.mesh_pipelines.empty_layout,
            lights,
        );
        // Terrains with holes draw with their own pipelines, created with the
        // first frame that has them.
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

    /// Terrain's visibility target and its resolves' group, at the frame's
    /// size, for frames whose terrain may shade deferred.
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

    /// Creates the pipelines `snapshot`'s terrains draw with unless they
    /// exist, before any draws: their leaves into
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
        let lights = self.features.lights.get().map(LightPass::fragment_layout);
        let layouts = (
            &self.camera.layout,
            &self.mesh_pipelines.empty_layout,
            lights,
        );
        // Terrain's pipelines over its own first group, its cover's over the
        // main one.
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

    /// Prepares this frame's water and its pipelines into
    /// `opaque_kind`'s targets, created while terrains with hydrology stream
    /// in, tiles' water on the leaves of the terrains' subdivisions; the sea
    /// draws through the terrain's projection, jittered as
    /// `temporal_frame` jitters it; `None` when none of it draws.
    pub(super) fn prepare_water(
        &mut self,
        snapshot: &RenderSnapshot,
        opaque_kind: PassKind,
        temporal_frame: &Option<TemporalFrame>,
    ) -> Option<Ready<WaterFrame>> {
        let targets = opaque_kind.targets()?;
        let coming = snapshot
            .terrain
            .terrains
            .iter()
            .any(|terrain| terrain.hydrology);
        let jitter = temporal_frame.as_ref().map_or([0.0; 2], |frame| {
            [frame.camera.jitter[0], frame.camera.jitter[1]]
        });
        let cull = snapshot.terrain.relative_view_projection;
        let view = crate::gpu_data::jittered(cull, jitter);
        let (device, shader, camera) = (&self.device, &self.mesh_shader, &self.camera.water_layout);
        let formats = (self.display.format(), DEPTH_FORMAT);
        let tiles = self.terrain.water_draws(&snapshot.terrain);
        // The foam field reaches from the detail eye out to where its cells
        // span a few of the display's pixels.
        let display = self.display.region.display;
        let seen = FoamView {
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
                (environment.water, environment.ocean),
                (
                    snapshot.camera_position,
                    seen,
                    environment.wind,
                    snapshot.presented_seconds,
                ),
                (
                    cull,
                    view,
                    &self.depth.texture,
                    matches!(targets, Targets::Post | Targets::Temporal),
                ),
            )?;
            water.ensure_pipeline(device, shader, camera, (targets, false), formats);
            Some(prepared)
        })
    }

    /// Whether this frame's terrain draws deferred, given whether it draws
    /// and the opaque passes' `kind`: into the effects' targets (debug views
    /// draw it forward), its pipelines and target ready.
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

    /// The leaves of the terrains' subdivisions into terrain's visibility
    /// target; returns the draws.
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

    /// Each terrain's pixels shaded once from its visibility target;
    /// returns the draws.
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
        // Each within the pixels its ground covers: elsewhere it would only
        // read the visibility target and discard. Pixels of the frame's
        // render size, which it draws.
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

    /// Draws the leaves of the terrains' subdivisions into the opaque pass;
    /// returns the draws.
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

    /// Draws every terrain's leaves into reflection probe face `face` of the
    /// probe at `center`, after its meshes (`lit` when lights shade it), over
    /// terrain's first group of that face: what draws after it binds its own.
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
