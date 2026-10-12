//! A frame's passes encoded in its render graph's order: each kept pass's
//! work, the attachments the graph derived operations for, and every pass
//! but the opaque ones (`opaque`).

use super::*;
use crate::render_features::{AttachmentViews, Encode, color_operations};

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
            Pass::TerrainHorizons => {
                self.terrain
                    .encode_horizons(&self.device, encoder, &self.profiler);
            }
            Pass::ShadowMaps => self.encode_shadow_maps(state, encoder, report),
            Pass::Lights => self.encode_light_binning(state, encoder),
            Pass::CloudDome => self.clouds.encode(
                &self.device,
                encoder,
                &self.camera.cloud_shadows,
                &self.profiler,
            ),
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
            Pass::GroundCover => self
                .ground_cover
                .encode(&self.device, encoder, &self.profiler),
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
            Pass::CloudMarch => {
                self.clouds
                    .encode_view(&self.device, encoder, &self.depth.view, &self.profiler);
            }
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
            Resource::TerrainIds => {
                &self
                    .terrain_ids
                    .get()
                    .expect("deferred terrain's frames create it")
                    .view
            }
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
        let views = FrameViews {
            renderer: self,
            canvas: view,
        };
        let lights = state
            .features
            .light_frame
            .as_ref()
            .and(self.lights.as_ref())
            .map(LightPass::fragment_group);
        let encode = Encode::new(
            &views,
            &self.profiler,
            (self.camera.group(), &self.materials, lights),
            (state.shape.targets, &self.depth.texture),
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

    /// The view's clouds laid over the scene's color.
    fn encode_cloud_composite(
        &self,
        state: &FrameState<'_>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        node: &Node,
    ) {
        let mut pass = self.render_pass(
            encoder,
            node,
            view,
            "engine/cloud-composite",
            self.profiler.span("cloud_composite").map(Span::render),
        );
        self.clouds.draw_composite(
            &mut pass,
            self.camera.group(),
            self.cloud_target(state.shape.targets),
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
        let mut pass = self.render_pass(
            encoder,
            node,
            view,
            "engine/transparent-accumulation",
            self.profiler.span("transparent").map(Span::render),
        );
        let mut draws = 0;
        if transparent_draws && instance_count > 0 {
            // Glass reads the opaque color the pass draws over.
            let camera_group = match self.camera.refraction_group() {
                Some(group) if refraction && post_frame.is_some() => group,
                _ => self.camera.group(),
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
            particles.draw_alpha(&mut pass, self.camera.group(), frame);
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
        let mut pass = self.render_pass(
            encoder,
            node,
            view,
            "engine/transparent-composite",
            self.profiler
                .span("transparent_composite")
                .map(Span::render),
        );
        transparent.composite(&mut pass, post);
        if let (Some(frame), Some(particles)) =
            (&state.features.particle_frame, self.particles.as_ref())
        {
            particles.draw_additive(&mut pass, self.camera.group(), frame, post);
        }
    }

    /// The screen effects: ambient occlusion, bloom,
    /// reflections and sun shafts after the opaque and transparent passes.
    fn encode_screen_effects(&self, state: &FrameState<'_>, encoder: &mut wgpu::CommandEncoder) {
        let post_frame = state.features.post_frame.expect("planned with effects");
        let post = self.post.as_ref().expect("post frames create it");
        post.encode_effects(encoder, post_frame, self.camera.group(), &self.profiler);
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
}

/// A frame's attachments: the renderer's targets, and the canvas's view
/// `canvas`.
struct FrameViews<'a> {
    renderer: &'a Renderer,
    canvas: &'a wgpu::TextureView,
}

impl AttachmentViews for FrameViews<'_> {
    fn view(&self, resource: Resource) -> &wgpu::TextureView {
        self.renderer.attachment_view(resource, self.canvas)
    }
}
