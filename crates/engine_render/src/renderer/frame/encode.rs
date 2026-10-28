#![expect(clippy::expect_used, reason = "passes read what their frames created")]

use super::*;
use crate::render_features::{AttachmentViews, Encode, MeshDraws};

impl Renderer {
    pub(super) fn encode_pass(
        &mut self,
        node: &Node,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        report: &mut EncodeReport,
    ) {
        match node.pass() {
            Pass::Presentation => self.encode_presentation(encoder),
            Pass::Subdivision => self.terrain.encode_subdivision(
                &self.device,
                encoder,
                &self.mesh_shader,
                &self.camera.buffer,
                &self.profiler,
            ),
            Pass::ShadowSelect => self.encode_shadow_selection(state, encoder),
            Pass::VirtualSelect => self.encode_cluster_selection(encoder),
            Pass::ShadowMaps => self.encode_shadow_maps(state, encoder, report),
            Pass::Deformation => {
                self.terrain
                    .deformation
                    .encode(&self.device, &self.queue, encoder, &self.profiler)
            }
            Pass::Bounce => self.terrain.encode_bounce(
                &self.device,
                encoder,
                &self.profiler,
                crate::terrain_bounce::SceneSources {
                    lighting: self.camera.lighting_buffer(),
                    shadow: self.shadows.entries()[0].clone(),
                    cloud_shadows: &self.camera.cloud_shadows,
                },
            ),
            Pass::Opaque { phase } => {
                let draws = self.encode_opaque(state, encoder, view, node, phase.index());
                if phase == Phase::One {
                    report.draws += draws;
                }
            }
            Pass::TerrainIds => report.draws += self.encode_terrain_ids(encoder, view, node),
            Pass::TerrainResolve => {
                report.draws += self.encode_terrain_resolve(state, encoder, view, node);
            }
            Pass::OpaqueLate => report.draws += self.encode_opaque_late(state, encoder, view, node),
            Pass::VisibilityIds { phase } => {
                self.encode_visibility_ids(state, encoder, view, node, phase.index());
            }
            Pass::Classify { .. } => self.encode_visibility_tiles(encoder),
            Pass::Resolve { phase } => {
                self.encode_visibility_resolve(state, encoder, view, node, phase.index());
            }
            Pass::OcclusionTest => self.encode_occlusion_test(state, encoder),
            Pass::ShadowMarks => self.encode_shadow_marks(encoder),
            Pass::Feature(pass) => {
                report.draws += self.encode_feature(pass, state, encoder, view, node);
            }
            Pass::ProbeCaptures => self.encode_probes(state, encoder),
        }
    }

    fn attachment_view<'a>(
        &'a self,
        resource: Resource,
        canvas: &'a wgpu::TextureView,
    ) -> &'a wgpu::TextureView {
        let post = || {
            self.features
                .post
                .get()
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
                .features
                .temporal
                .get()
                .expect("temporal frames create it")
                .motion(),
            Resource::Ids => self
                .visibility
                .buffer
                .as_ref()
                .expect("resolving frames create it")
                .ids_view(),
            Resource::TerrainIds => {
                &self
                    .terrain_ids
                    .get()
                    .expect("deferred terrain's frames create it")
                    .view
            }
            Resource::DebugIds | Resource::DebugCount | Resource::DebugTerrain => {
                let [ids, count, terrain] = self
                    .features
                    .debug_view
                    .get()
                    .expect("debug views' frames create it")
                    .views();
                if resource == Resource::DebugIds {
                    ids
                } else if resource == Resource::DebugCount {
                    count
                } else {
                    terrain
                }
            }
            Resource::Accumulated | Resource::Revealed => {
                let [accumulated, revealed] = self.features.transparency.get().views();
                if resource == Resource::Accumulated {
                    accumulated
                } else {
                    revealed
                }
            }
            Resource::History
            | Resource::Tiles
            | Resource::Poses
            | Resource::Visible
            | Resource::Placed
            | Resource::Occupancy
            | Resource::Casters
            | Resource::ShadowMaps
            | Resource::ShadowMarks
            | Resource::Clusters
            | Resource::Lights
            | Resource::Pyramid
            | Resource::Effects
            | Resource::Haze
            | Resource::Exposure
            | Resource::Probes
            | Resource::CloudView
            | Resource::CloudDome
            | Resource::Tracks
            | Resource::Bounce
            | Resource::Leaves
            | Resource::Cover => {
                unreachable!("{resource:?} is no attachment of the renderer's passes")
            }
        }
    }

    pub(super) fn render_pass<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        node: &Node,
        canvas: &wgpu::TextureView,
        label: &'static str,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) -> wgpu::RenderPass<'e> {
        let views = FrameViews {
            renderer: self,
            canvas,
            history: None,
        };
        crate::render_features::render_pass(encoder, node, &views, label, timestamp_writes)
    }

    fn encode_feature(
        &self,
        pass: FeaturePass,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
    ) -> u32 {
        let history = state
            .features
            .frames
            .temporal
            .as_ref()
            .zip(self.features.temporal.get())
            .map(|(ready, temporal)| temporal.history_target(ready.frame()));
        let views = FrameViews {
            renderer: self,
            canvas: view,
            history,
        };
        let meshes = FrameMeshes {
            renderer: self,
            state,
        };
        let lights = state
            .features
            .light_frame()
            .and(self.features.lights.get())
            .map(LightPass::fragment_group);
        let scene = matches!(state.shape.targets, Targets::Post | Targets::Temporal)
            .then(|| self.features.post.get())
            .flatten()
            .map(crate::post::PostPass::scene_texture);
        let encode = Encode::new(
            (&views, &meshes),
            (&self.device, &self.profiler),
            (
                (
                    self.camera.group(),
                    self.camera.terrain_group(),
                    self.camera.water_group(),
                ),
                &self.materials,
                lights,
            ),
            (state.shape.targets, &self.depth.texture, scene),
        );
        self.features
            .encode(pass, &state.features.frames, node, &encode, encoder)
    }

    fn encode_presentation(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.gpu_presentation
            .encode(&self.device, encoder, &self.profiler);
    }

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

    fn encode_shadow_marks(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.shadows.encode_marks(
            &self.device,
            encoder,
            (&self.camera.buffer, self.depth.view.bound()),
            self.display.region.render.pixels(),
            &self.profiler,
        );
    }

    fn encode_shadow_maps(
        &mut self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        report: &mut EncodeReport,
    ) {
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
                terrain_draw: &|pass, map| {
                    if terrain_frame {
                        terrain.draw_shadow(pass, map)
                    } else {
                        0
                    }
                },
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

    pub(super) fn cloud_target(&self, targets: Targets) -> (wgpu::TextureFormat, bool) {
        if targets == Targets::Canvas {
            (self.display.format(), true)
        } else {
            (crate::post::SCENE_FORMAT, false)
        }
    }

    fn encode_probes(&mut self, state: &FrameState<'_>, encoder: &mut wgpu::CommandEncoder) {
        let snapshot = state.snapshot;
        let temporal_frame = &state.packing.temporal_frame;
        let (probe_list, light_frame) = (state.features.probe_list, state.features.light_frame());
        self.features.probes.get_mut().slots.tick();
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
            light_frame.map(|frame| (frame, &snapshot.environment.local_lights)),
        );
        self.features.probes.get_mut().faces = faces;
    }
}

struct FrameMeshes<'a> {
    renderer: &'a Renderer,
    state: &'a FrameState<'a>,
}

impl MeshDraws for FrameMeshes<'_> {
    fn transparent(&self, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        let (renderer, state) = (self.renderer, self.state);
        let FrameKind {
            gpu_mode,
            instance_count,
            ..
        } = state.kind;
        if instance_count == 0 {
            return 0;
        }
        let features = &state.features;
        let lit = features.light_frame().zip(renderer.features.lights.get());
        let camera_group = match renderer.camera.refraction_group() {
            Some(group) if features.refraction && features.post_frame().is_some() => group,
            _ => renderer.camera.group(),
        };
        pass.set_bind_group(0, camera_group, &[]);
        if let Some((_, lights)) = lit {
            pass.set_bind_group(3, lights.fragment_group(), &[]);
        }
        let parity = state
            .packing
            .temporal_frame
            .as_ref()
            .map_or(0, |frame| frame.parity);
        pass.set_bind_group(2, renderer.instances.pose_group(parity), &[]);
        if !gpu_mode {
            pass.set_vertex_buffer(1, renderer.instances.slots().slice(..));
        }
        let mut binds = BatchBinds::default();
        let mut draws = 0;
        for (index, batch) in renderer.plan.batches.iter().enumerate() {
            if !batch.transparent {
                continue;
            }
            let (_, draw) = renderer.bind_batch(
                pass,
                &mut binds,
                (state.passes.transparent_kind, false),
                batch,
            );
            if gpu_mode {
                let (ids, args, offset) = renderer.gpu_presentation.draw_source(batch, index, 0, 0);
                binds.bindings.ids(pass, 1, ids);
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
        draws
    }
}

struct FrameViews<'a> {
    renderer: &'a Renderer,
    canvas: &'a wgpu::TextureView,
    history: Option<&'a wgpu::TextureView>,
}

impl AttachmentViews for FrameViews<'_> {
    fn view(&self, resource: Resource) -> &wgpu::TextureView {
        match (resource, self.history) {
            (Resource::History, Some(history)) => history,
            _ => self.renderer.attachment_view(resource, self.canvas),
        }
    }

    fn region(&self, node: &Node) -> Option<(u32, u32)> {
        let whole = node
            .colors()
            .any(|(resource, _)| matches!(resource, Resource::Canvas | Resource::History));
        (!whole).then(|| self.renderer.display.region.render.pixels())
    }
}
