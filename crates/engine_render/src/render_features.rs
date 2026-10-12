//! The renderer's features, each in a module of its own: what it prepares
//! for a frame (its own typed method, other features' outputs as its
//! parameters), the render graph nodes that declares, how it encodes them,
//! its work after the frame is submitted, its stats and the GPU memory it
//! holds. `render_features!` lists them; the renderer dispatches to each
//! statically.

use engine_core::RenderSnapshot;

use crate::decals::DecalPass;
use crate::lights::LightPass;
use crate::materials::MaterialStore;
use crate::post::PostPass;
use crate::profiling::GpuProfiler;
use crate::render_graph::{Clear, FrameShape, Load, Node, Ops, RenderGraph, Resource};
use crate::targets::Targets;
use crate::temporal::TemporalPass;
use crate::transparent::Transparency;
use crate::water::WaterPass;

/// A feature of the renderer.
pub(crate) trait RenderFeature {
    /// What the feature prepared for a frame, which its own preparation
    /// makes and its holder ([`Lazy`] or [`Eager`]) hands back as a
    /// [`Ready`].
    type Frame;
    /// The feature's part of a frame's stats.
    type Stats;

    /// Declares the feature's nodes into `graph`, a frame of `shape` it
    /// prepared `frame` for.
    fn declare(frame: &Self::Frame, shape: &FrameShape, graph: &mut RenderGraph);

    /// Encodes `node`, one of the feature's nodes the graph kept, into
    /// `encoder`; returns the draws it adds to the frame's draw calls.
    fn encode(
        &self,
        frame: &Self::Frame,
        node: &Node,
        encode: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32;

    /// Work after a frame it prepared `frame` for (`None` when it did not
    /// run) is submitted.
    fn after_submit(&mut self, _frame: Option<&Self::Frame>) {}

    /// The GPU memory the feature holds.
    fn capacity_bytes(&self) -> u64;

    /// The feature's stats, `this` once created, of a frame it prepared
    /// `frame` for (`None` when it did not run), showing `snapshot`.
    fn stats(
        this: Option<&Self>,
        frame: Option<&Self::Frame>,
        snapshot: &RenderSnapshot,
    ) -> Self::Stats;
}

/// A feature created with the first frame needing it, then kept.
pub(crate) struct Lazy<F> {
    feature: Option<F>,
}

impl<F> Default for Lazy<F> {
    fn default() -> Self {
        Self { feature: None }
    }
}

impl<F: RenderFeature> Lazy<F> {
    /// Prepares the feature for a frame that `needs` it, created by `create`
    /// the first time: what `prepare` made of it, which encoding the
    /// feature requires. `None` when the frame does not need it, or it
    /// prepared nothing.
    pub(crate) fn prepare(
        &mut self,
        needs: bool,
        create: impl FnOnce() -> F,
        prepare: impl FnOnce(&mut F) -> Option<F::Frame>,
    ) -> Option<Ready<F::Frame>> {
        if !needs {
            return None;
        }
        prepare(self.feature.get_or_insert_with(create)).map(|frame| Ready { frame })
    }

    /// Prepares the feature for a frame when one created it: what
    /// `prepare` made of it, as [`Self::prepare`] hands it back.
    pub(crate) fn prepare_created(
        &mut self,
        prepare: impl FnOnce(&mut F) -> Option<F::Frame>,
    ) -> Option<Ready<F::Frame>> {
        prepare(self.feature.as_mut()?).map(|frame| Ready { frame })
    }

    /// The feature, once a frame created it.
    pub(crate) fn get(&self) -> Option<&F> {
        self.feature.as_ref()
    }

    pub(crate) fn get_mut(&mut self) -> Option<&mut F> {
        self.feature.as_mut()
    }

    /// Creates the feature with `create` unless it exists.
    pub(crate) fn create(&mut self, create: impl FnOnce() -> F) {
        self.feature.get_or_insert_with(create);
    }

    /// Drops the feature until it is created again.
    pub(crate) fn release(&mut self) {
        self.feature = None;
    }

    /// The feature, once a frame created it, for work after the frame.
    fn each_mut(&mut self) -> Option<&mut F> {
        self.feature.as_mut()
    }

    /// The feature `ready` was prepared by.
    fn ready(&self, _ready: &Ready<F::Frame>) -> &F {
        self.feature
            .as_ref()
            .expect("only Lazy::prepare makes a Ready, once it created its feature")
    }
}

/// A feature the renderer creates with itself.
pub(crate) struct Eager<F> {
    feature: F,
}

impl<F: RenderFeature> Eager<F> {
    pub(crate) fn new(feature: F) -> Self {
        Self { feature }
    }

    /// What `prepare` made of the feature for a frame, which encoding the
    /// feature requires; `None` when it prepared nothing.
    pub(crate) fn prepare(
        &mut self,
        prepare: impl FnOnce(&mut F) -> Option<F::Frame>,
    ) -> Option<Ready<F::Frame>> {
        prepare(&mut self.feature).map(|frame| Ready { frame })
    }

    pub(crate) fn get(&self) -> &F {
        &self.feature
    }

    /// The feature, for work after the frame.
    fn each_mut(&mut self) -> Option<&mut F> {
        Some(&mut self.feature)
    }

    /// The feature `ready` was prepared by.
    fn ready(&self, _ready: &Ready<F::Frame>) -> &F {
        &self.feature
    }
}

/// What a feature prepared for a frame. Only its holder's `prepare` makes
/// one, so holding one proves the feature exists and is prepared.
pub(crate) struct Ready<F> {
    frame: F,
}

impl<F> Ready<F> {
    pub(crate) fn frame(&self) -> &F {
        &self.frame
    }
}

/// The renderer's own mesh draws that a feature's pass takes in.
pub(crate) trait MeshDraws {
    /// The frame's transparent mesh batches, into the transparent
    /// accumulation pass `pass`; returns their draws.
    fn transparent(&self, pass: &mut wgpu::RenderPass<'_>) -> u32;
}

/// The views of a frame's attachments.
pub(crate) trait AttachmentViews {
    /// The view of attachment `resource` this frame.
    fn view(&self, resource: Resource) -> &wgpu::TextureView;
}

/// What a feature's encoding reads of the frame, and the one way to begin a
/// node's render pass.
pub(crate) struct Encode<'f> {
    views: &'f dyn AttachmentViews,
    /// The renderer's mesh draws features' passes take in.
    pub(crate) meshes: &'f dyn MeshDraws,
    pub(crate) device: &'f wgpu::Device,
    pub(crate) profiler: &'f GpuProfiler,
    /// The main passes' first group.
    pub(crate) camera: &'f wgpu::BindGroup,
    pub(crate) materials: &'f MaterialStore,
    /// The local lights' group, in frames shading them.
    pub(crate) lights: Option<&'f wgpu::BindGroup>,
    /// Where the frame's main passes draw.
    pub(crate) targets: Targets,
    /// The depth buffer's texture.
    pub(crate) depth: &'f wgpu::Texture,
}

impl<'f> Encode<'f> {
    pub(crate) fn new(
        (views, meshes): (&'f dyn AttachmentViews, &'f dyn MeshDraws),
        (device, profiler): (&'f wgpu::Device, &'f GpuProfiler),
        (camera, materials, lights): (
            &'f wgpu::BindGroup,
            &'f MaterialStore,
            Option<&'f wgpu::BindGroup>,
        ),
        (targets, depth): (Targets, &'f wgpu::Texture),
    ) -> Self {
        Self {
            views,
            meshes,
            device,
            profiler,
            camera,
            materials,
            lights,
            targets,
            depth,
        }
    }

    /// The view of attachment `resource` this frame.
    pub(crate) fn view(&self, resource: Resource) -> &wgpu::TextureView {
        self.views.view(resource)
    }

    /// Begins `node`'s render pass, labeled `label` and timed by
    /// `timestamp_writes`.
    pub(crate) fn render_pass<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        node: &Node,
        label: &'static str,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) -> wgpu::RenderPass<'e> {
        render_pass(encoder, node, self.views, label, timestamp_writes)
    }
}

/// Begins `node`'s render pass, labeled `label`, over its attachments'
/// `views` with the operations the render graph derived for them, timed by
/// `timestamp_writes`: how every pass drawing into the graph's attachments
/// begins, so none writes operations of its own.
pub(crate) fn render_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    node: &Node,
    views: &dyn AttachmentViews,
    label: &'static str,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) -> wgpu::RenderPass<'e> {
    let mut colors = [None, None, None, None];
    let mut count = 0;
    for (resource, ops) in node.colors() {
        colors[count] = Some(wgpu::RenderPassColorAttachment {
            view: views.view(resource),
            depth_slice: None,
            resolve_target: None,
            ops: color_operations(ops),
        });
        count += 1;
    }
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &colors[..count],
        depth_stencil_attachment: node.depth().map(|(resource, ops)| {
            wgpu::RenderPassDepthStencilAttachment {
                view: views.view(resource),
                depth_ops: ops.map(depth_operations),
                stencil_ops: None,
            }
        }),
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// A color attachment's operations as the render graph derived them.
pub(crate) fn color_operations(ops: Ops) -> wgpu::Operations<wgpu::Color> {
    wgpu::Operations {
        load: match ops.load {
            Load::Clear(Clear::Color([r, g, b, a])) => {
                wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a })
            }
            Load::Clear(Clear::Depth(_)) => unreachable!("color attachments clear to colors"),
            Load::Load => wgpu::LoadOp::Load,
        },
        store: store_operation(ops),
    }
}

/// A depth attachment's operations as the render graph derived them.
fn depth_operations(ops: Ops) -> wgpu::Operations<f32> {
    wgpu::Operations {
        load: match ops.load {
            Load::Clear(Clear::Depth(depth)) => wgpu::LoadOp::Clear(depth),
            Load::Clear(Clear::Color(_)) => unreachable!("depth attachments clear to depths"),
            Load::Load => wgpu::LoadOp::Load,
        },
        store: store_operation(ops),
    }
}

fn store_operation(ops: Ops) -> wgpu::StoreOp {
    if ops.store {
        wgpu::StoreOp::Store
    } else {
        wgpu::StoreOp::Discard
    }
}

/// Builds the renderer's features from one list: each one's field, its
/// holder ([`Lazy`] or [`Eager`]) and type, and the render graph passes it
/// owns, which become [`FeaturePass`]es the feature declares and encodes.
macro_rules! render_features {
    ($(
        $(#[$doc:meta])*
        $field:ident: $holder:ident<$feature:ty> => [$($pass:ident),+ $(,)?],
    )*) => {
        /// The render graph passes features own (`Pass::Feature`), each
        /// declared and encoded by its feature.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum FeaturePass {
            $($($pass,)+)*
        }

        /// The features, in their holders.
        pub(crate) struct RenderFeatures {
            $($(#[$doc])* pub(crate) $field: $holder<$feature>,)*
        }

        /// What each feature prepared for a frame.
        #[derive(Default)]
        pub(crate) struct FeatureFrames {
            $(pub(crate) $field: Option<Ready<<$feature as RenderFeature>::Frame>>,)*
        }

        impl RenderFeatures {
            /// Declares into `graph` the nodes of each feature `frames`
            /// prepared, in a frame of `shape`.
            pub(crate) fn declare(
                frames: &FeatureFrames,
                shape: &FrameShape,
                graph: &mut RenderGraph,
            ) {
                $(if let Some(ready) = &frames.$field {
                    <$feature as RenderFeature>::declare(ready.frame(), shape, graph);
                })*
            }

            /// Encodes `node`, a kept node of the feature pass `pass`, with
            /// what its feature prepared in `frames`; returns the draws it
            /// adds to the frame's draw calls.
            pub(crate) fn encode(
                &self,
                pass: FeaturePass,
                frames: &FeatureFrames,
                node: &Node,
                encode: &Encode<'_>,
                encoder: &mut wgpu::CommandEncoder,
            ) -> u32 {
                match pass {
                    $($(FeaturePass::$pass)|+ => {
                        let ready = frames
                            .$field
                            .as_ref()
                            .expect("features declare their passes only with what they prepared");
                        self.$field.ready(ready).encode(ready.frame(), node, encode, encoder)
                    })*
                }
            }

            /// Each feature's work after the frame it prepared `frames` for
            /// is submitted.
            pub(crate) fn after_submit(&mut self, frames: &FeatureFrames) {
                $(if let Some(feature) = self.$field.each_mut() {
                    feature.after_submit(frames.$field.as_ref().map(Ready::frame));
                })*
            }
        }
    };
}

render_features! {
    /// Local lights binned into the view's clusters, created with the first
    /// frame that has some.
    lights: Lazy<LightPass> => [Lights],
    /// Decals over the opaque surfaces.
    decals: Lazy<DecalPass> => [Decals],
    /// Terrains' lakes and rivers, and the sea, over the opaque scene.
    water: Eager<WaterPass> => [Water],
    /// Transparent surfaces and particles, accumulated, then composited.
    transparency: Eager<Transparency> => [Transparent, TransparentComposite],
    /// The screen effects, then their composite into the canvas, created
    /// with the first frame drawing into their targets.
    post: Lazy<PostPass> => [Effects, Composite],
    /// Temporal antialiasing's resolve into the canvas, created while the
    /// setting is on.
    temporal: Lazy<TemporalPass> => [TemporalResolve],
}
