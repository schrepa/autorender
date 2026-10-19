//! A frame's passes encoded in its render graph's order: each kept pass's
//! work, the attachments the graph derived operations for, and every pass
//! but the opaque ones (`opaque`).

#![expect(clippy::expect_used, reason = "passes read what their frames created")]

use super::*;
use crate::render_features::{AttachmentViews, Encode, MeshDraws};

impl Renderer {
    /// Encodes `node`, a pass the frame's render graph kept,
    /// into `encoder`; `view` is the canvas's.
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
            Pass::ShadowSelect => self.encode_shadow_selection(state, encoder),
            Pass::VirtualSelect => self.encode_cluster_selection(encoder),
            Pass::TerrainHorizons => {
                self.terrain
                    .encode_horizons(&self.device, encoder, &self.profiler);
            }
            Pass::ShadowMaps => self.encode_shadow_maps(state, encoder, report),
            Pass::Deformation => {
                self.terrain
                    .deformation
                    .encode(&self.device, &self.queue, encoder, &self.profiler)
            }
            Pass::VtBakes => self.terrain.encode_vt_bakes(
                &self.device,
                encoder,
                &self.mesh_shader,
                (&self.camera.terrain_layout, self.camera.terrain_group()),
            ),
            Pass::Opaque { phase } => {
                let draws = self.encode_opaque(state, encoder, view, node, phase.index());
                // Phase two issues phase one's draws again for
                // the instances its test revealed: count each once.
                if phase == Phase::One {
                    report.draws += draws;
                }
            }
            Pass::TerrainIds => report.draws += self.encode_terrain_ids(encoder, view, node),
            Pass::TerrainResolve => {
                report.draws += self.encode_terrain_resolve(state, encoder, view, node);
            }
            Pass::OpaqueLate => report.draws += self.encode_opaque_late(state, encoder, view, node),
            Pass::VtFeedback => self.encode_vt_feedback(encoder),
            Pass::VisibilityIds { phase } => {
                self.encode_visibility_ids(state, encoder, view, node, phase.index());
            }
            Pass::Classify { .. } => self.encode_visibility_tiles(encoder),
            Pass::Resolve { phase } => {
                self.encode_visibility_resolve(state, encoder, view, node, phase.index());
            }
            Pass::OcclusionTest => self.encode_occlusion_test(state, encoder),
            Pass::Feature(pass) => {
                report.draws += self.encode_feature(pass, state, encoder, view, node);
            }
            Pass::ProbeCaptures => self.encode_probes(state, encoder),
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
            | Resource::Clusters
            | Resource::Lights
            | Resource::Pyramid
            | Resource::Effects
            | Resource::Probes
            | Resource::CloudView
            | Resource::Horizons
            | Resource::CloudDome
            | Resource::Tracks
            | Resource::Pages
            | Resource::Cover
            | Resource::Feedback => {
                unreachable!("{resource:?} is no attachment of the renderer's passes")
            }
        }
    }

    /// Begins `node`'s render pass, labeled `label`, over its attachments
    /// (`canvas` is the canvas's view), timed by `timestamp_writes`: as
    /// features' passes begin (`render_features::render_pass`).
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

    /// `node`, a feature's pass `pass`, encoded by its feature; returns
    /// the draws it adds to the frame's draw calls.
    fn encode_feature(
        &self,
        pass: FeaturePass,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
    ) -> u32 {
        // The history temporal antialiasing's resolve writes this frame.
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
        let scene = (state.shape.targets != Targets::Canvas)
            .then(|| self.features.post.get())
            .flatten()
            .map(crate::post::PostPass::scene_texture);
        let encode = Encode::new(
            (&views, &meshes),
            (&self.device, &self.profiler),
            (self.camera.group(), &self.materials, lights),
            (state.shape.targets, &self.depth.texture, scene),
        );
        self.features
            .encode(pass, &state.features.frames, node, &encode, encoder)
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
                    let Some(pipeline) = terrain
                        .pipelines
                        .shadow_pipeline()
                        .filter(|_| terrain_frame)
                    else {
                        return 0;
                    };
                    let mut draws = 0;
                    for frame_terrain in &terrain.patches.frame.terrains {
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

    /// The pages the first phase's terrain resolves wanted of the runtime
    /// virtual texture, read from terrain's visibility target.
    fn encode_vt_feedback(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some(ids) = self.terrain_ids.get() else {
            return;
        };
        let size = self.depth.texture.size();
        self.terrain.encode_vt_feedback(
            &self.device,
            encoder,
            &self.mesh_shader,
            &self.camera.buffer,
            (&ids.view, (size.width, size.height)),
        );
    }

    /// The format of the scene's color for frames drawing into `targets`,
    /// and whether that is the canvas.
    pub(super) fn cloud_target(&self, targets: Targets) -> (wgpu::TextureFormat, bool) {
        if targets == Targets::Canvas {
            (self.display.format(), true)
        } else {
            (crate::post::SCENE_FORMAT, false)
        }
    }

    /// Reflection probe captures after the frame's own
    /// drawing, once their slots stand still.
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

/// A frame's own mesh draws, which features' passes take in.
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
        // Glass reads the opaque color the pass draws over.
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

/// A frame's attachments: the renderer's targets, the canvas's view
/// `canvas`, and the history temporal antialiasing resolves into, in
/// frames resolving it.
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
}
