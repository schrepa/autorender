//! The renderer's features, each in a module of its own: what it prepares
//! for a frame (its own typed method, other features' outputs as its
//! parameters), the render graph nodes that declares, how it encodes them,
//! its work after the frame is submitted, its stats and the GPU memory it
//! holds. `render_features!` lists them; the renderer dispatches to each
//! statically. A lazily created feature that fails is disabled alone:
//! frames render without it.

#![expect(
    clippy::expect_used,
    reason = "only a holder's preparation makes a Ready, with its feature"
)]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use engine_core::RenderSnapshot;

use crate::clouds::CloudPass;
use crate::decals::DecalPass;
use crate::ground_cover::GroundCoverPass;
use crate::lights::LightPass;
use crate::materials::MaterialStore;
use crate::occupancy::OccupancyPass;
use crate::post::PostPass;
use crate::profiling::GpuProfiler;
use crate::render_graph::{Clear, FrameShape, Load, Node, Ops, RenderGraph, Resource};
use crate::scatter::ScatterPass;
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

    /// How the labels of the feature's GPU objects begin: an uncaptured
    /// error naming one of them is the feature's. Features created with the
    /// renderer name none; an error in them stays fatal.
    const LABELS: &'static [&'static str] = &[];
}

/// A feature created with the first frame needing it, then kept. Its
/// creation runs under an error scope, which later frames poll: a creation
/// that failed validation, or an uncaptured error naming the feature,
/// disables it, released for good.
pub(crate) struct Lazy<F> {
    state: State<F>,
    /// The fault a development session injected into the feature.
    #[cfg(feature = "dev")]
    fault: std::cell::Cell<Fault>,
}

/// Where a lazily created feature is.
enum State<F> {
    /// No frame needed it yet, or a setting released it.
    Absent,
    /// Created; `validation` is its creation's error scope until the device
    /// resolves it.
    Created {
        feature: F,
        validation: Option<Validation>,
    },
    /// Released for good: its creation failed, or an error named it.
    Disabled,
}

/// A creation's error scope: the error it caught, once the device says.
type Validation = Pin<Box<dyn Future<Output = Option<wgpu::Error>>>>;

/// A fault a development session injects into a lazily created feature, to
/// check that the renderer disables it alone: its next creation makes an
/// invalid buffer of its own, which its creation's error scope catches, and
/// its first frame then uses, which fails that frame's submission with
/// uncaptured errors naming it where the device validates later.
#[cfg(feature = "dev")]
#[derive(Default)]
enum Fault {
    #[default]
    None,
    /// For the feature's next creation.
    Injected,
    /// The invalid buffer its creation made, for its first frame.
    Made(wgpu::Buffer),
}

impl<F> Default for Lazy<F> {
    fn default() -> Self {
        Self {
            state: State::Absent,
            #[cfg(feature = "dev")]
            fault: std::cell::Cell::default(),
        }
    }
}

impl<F: RenderFeature> Lazy<F> {
    /// Whether the holder creates its feature lazily.
    #[cfg(feature = "dev")]
    const LAZY: bool = true;

    /// Prepares the feature for a frame that `needs` it, created by `create`
    /// on `device` the first time: what `prepare` made of it, which encoding
    /// the feature requires. `None` when the frame does not need it, it is
    /// disabled, or it prepared nothing.
    pub(crate) fn prepare(
        &mut self,
        (device, needs): (&wgpu::Device, bool),
        create: impl FnOnce() -> F,
        prepare: impl FnOnce(&mut F) -> Option<F::Frame>,
    ) -> Option<Ready<F::Frame>> {
        if !needs {
            return None;
        }
        self.create(device, create);
        prepare(self.get_mut()?).map(|frame| Ready { frame })
    }

    /// Prepares the feature for a frame when one created it: what
    /// `prepare` made of it, as [`Self::prepare`] hands it back.
    pub(crate) fn prepare_created(
        &mut self,
        prepare: impl FnOnce(&mut F) -> Option<F::Frame>,
    ) -> Option<Ready<F::Frame>> {
        prepare(self.get_mut()?).map(|frame| Ready { frame })
    }

    /// The feature, once a frame created it, unless it is disabled.
    pub(crate) fn get(&self) -> Option<&F> {
        match &self.state {
            State::Created { feature, .. } => Some(feature),
            State::Absent | State::Disabled => None,
        }
    }

    pub(crate) fn get_mut(&mut self) -> Option<&mut F> {
        match &mut self.state {
            State::Created { feature, .. } => Some(feature),
            State::Absent | State::Disabled => None,
        }
    }

    /// Creates the feature with `create` on `device` unless it exists or is
    /// disabled, under an error scope: a device that validates at once
    /// (natively) disables a failed creation before any frame uses it,
    /// others when a later frame polls the scope.
    pub(crate) fn create(&mut self, device: &wgpu::Device, create: impl FnOnce() -> F) {
        if !matches!(self.state, State::Absent) {
            return;
        }
        let scope = crate::gpu_calls::validation_scope(device);
        let feature = create();
        #[cfg(feature = "dev")]
        if let Fault::Injected = self.fault.take() {
            self.fault.set(Fault::Made(invalid_buffer::<F>(device)));
        }
        let mut validation: Validation = Box::pin(scope.pop());
        self.state = match poll(&mut validation) {
            Poll::Ready(Some(_)) => {
                #[cfg(feature = "dev")]
                drop(self.fault.take());
                State::Disabled
            }
            Poll::Ready(None) => State::Created {
                feature,
                validation: None,
            },
            Poll::Pending => State::Created {
                feature,
                validation: Some(validation),
            },
        };
    }

    /// Releases the feature until it is created again; a disabled one stays
    /// disabled.
    pub(crate) fn release(&mut self) {
        if matches!(self.state, State::Created { .. }) {
            self.state = State::Absent;
        }
    }

    /// Whether the feature is disabled for good.
    pub(crate) fn disabled(&self) -> bool {
        matches!(self.state, State::Disabled)
    }

    /// Polls the feature's creation scope while it is unresolved: whether
    /// this disabled the feature, whose creation failed.
    fn poll(&mut self) -> bool {
        let State::Created {
            validation: Some(validation),
            ..
        } = &mut self.state
        else {
            return false;
        };
        match poll(validation) {
            Poll::Pending => false,
            Poll::Ready(None) => {
                if let State::Created { validation, .. } = &mut self.state {
                    *validation = None;
                }
                false
            }
            Poll::Ready(Some(_)) => self.disable(),
        }
    }

    /// Disables the feature for good, releasing it: whether it was not
    /// disabled already.
    fn disable(&mut self) -> bool {
        #[cfg(feature = "dev")]
        drop(self.fault.take());
        !matches!(
            std::mem::replace(&mut self.state, State::Disabled),
            State::Disabled
        )
    }

    /// Injects a fault into the feature's next creation, releasing it now
    /// so the next frame needing it creates it again: always, as lazily
    /// created features take faults.
    #[cfg(feature = "dev")]
    fn inject(&mut self) -> bool {
        self.release();
        self.fault.set(Fault::Injected);
        true
    }

    /// Encodes, once, a command on the invalid buffer an injected fault made
    /// the feature's creation make into `encoder`.
    #[cfg(feature = "dev")]
    fn encode_fault(&self, encoder: &mut wgpu::CommandEncoder) {
        match self.fault.take() {
            Fault::Made(buffer) => encoder.clear_buffer(&buffer, 0, None),
            fault @ (Fault::None | Fault::Injected) => self.fault.set(fault),
        }
    }

    /// The feature, once a frame created it, for work after the frame.
    fn each_mut(&mut self) -> Option<&mut F> {
        self.get_mut()
    }

    /// The feature `ready` was prepared by.
    fn ready(&self, _ready: &Ready<F::Frame>) -> &F {
        self.get()
            .expect("only Lazy::prepare makes a Ready, once it created its feature")
    }
}

/// Polls a creation's error scope once, without waiting: its result, once
/// the device resolved it.
fn poll(validation: &mut Validation) -> Poll<Option<wgpu::Error>> {
    validation
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
}

/// A buffer whose usages no device allows together, labeled as one of `F`'s
/// objects: creating it is a validation error naming the feature.
#[cfg(feature = "dev")]
fn invalid_buffer<F: RenderFeature>(device: &wgpu::Device) -> wgpu::Buffer {
    let label = format!("{}fault", F::LABELS.first().copied().unwrap_or("engine/"));
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&label),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE,
        mapped_at_creation: false,
    })
}

/// A feature the renderer creates with itself.
pub(crate) struct Eager<F> {
    feature: F,
}

impl<F: RenderFeature> Eager<F> {
    /// Whether the holder creates its feature lazily.
    #[cfg(feature = "dev")]
    const LAZY: bool = false;

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

    pub(crate) fn get_mut(&mut self) -> &mut F {
        &mut self.feature
    }

    /// The feature, for work after the frame.
    fn each_mut(&mut self) -> Option<&mut F> {
        Some(&mut self.feature)
    }

    /// The feature `ready` was prepared by.
    fn ready(&self, _ready: &Ready<F::Frame>) -> &F {
        &self.feature
    }

    /// Never disabled: the renderer's startup validated it.
    fn poll(&mut self) -> bool {
        false
    }

    /// Never disabled: an error naming it fails the renderer.
    fn disable(&mut self) -> bool {
        false
    }

    fn disabled(&self) -> bool {
        false
    }

    /// Never injected: faults go to lazily created features.
    #[cfg(feature = "dev")]
    fn inject(&mut self) -> bool {
        false
    }

    #[cfg(feature = "dev")]
    fn encode_fault(&self, _encoder: &mut wgpu::CommandEncoder) {}
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

    /// Begins `node`'s render pass, labeled `label` and timed by
    /// `timestamp_writes`: the one way a feature reaches the frame's
    /// attachments.
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
fn color_operations(ops: Ops) -> wgpu::Operations<wgpu::Color> {
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
                        #[cfg(feature = "dev")]
                        self.$field.encode_fault(encoder);
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

            /// Every feature's name, in the list's order: a set of
            /// features has bit `i` for the `i`th.
            pub(crate) const NAMES: &'static [&'static str] = &[$(stringify!($field)),*];

            /// The one feature an error's `message` names by its objects'
            /// labels, as a set of features; empty when it names none, or
            /// several.
            pub(crate) fn named_by(message: &str) -> u64 {
                let labels: [&[&str]; Self::NAMES.len()] =
                    [$(<$feature as RenderFeature>::LABELS),*];
                let mut named = 0;
                for (index, labels) in labels.iter().enumerate() {
                    if labels.iter().any(|label| message.contains(label)) {
                        if named != 0 {
                            return 0;
                        }
                        named = 1 << index;
                    }
                }
                named
            }

            /// Polls the features' creation scopes still unresolved and
            /// disables those whose creation failed, and the features in
            /// `named`: the set of features this disabled.
            pub(crate) fn disable(&mut self, named: u64) -> u64 {
                let mut bits = (0..Self::NAMES.len()).map(|index| 1 << index);
                let mut disabled = 0;
                $(
                    let bit = bits.next().unwrap_or_default();
                    if self.$field.poll() || (named & bit != 0 && self.$field.disable()) {
                        disabled |= bit;
                    }
                )*
                disabled
            }

            /// The names of the features created lazily, which errors
            /// may disable.
            #[cfg(feature = "dev")]
            pub(crate) fn lazy_names() -> impl Iterator<Item = &'static str> {
                let lazy = [$(<$holder<$feature>>::LAZY),*];
                Self::NAMES
                    .iter()
                    .zip(lazy)
                    .filter_map(|(name, lazy)| lazy.then_some(*name))
            }

            /// Injects a fault into the lazily created feature `name`:
            /// whether one has that name.
            #[cfg(feature = "dev")]
            pub(crate) fn inject(&mut self, name: &str) -> bool {
                $(if name == stringify!($field) {
                    return self.$field.inject();
                })*
                false
            }

            /// The features disabled for good, as a set.
            pub(crate) fn disabled(&self) -> u64 {
                let mut bits = (0..Self::NAMES.len()).map(|index| 1 << index);
                let mut disabled = 0;
                $(
                    let bit = bits.next().unwrap_or_default();
                    if self.$field.disabled() {
                        disabled |= bit;
                    }
                )*
                disabled
            }
        }
    };
}

render_features! {
    /// Objects placed on terrain, culled for the camera and the shadow
    /// maps.
    scatter: Eager<ScatterPass> => [Scatter],
    /// What placed objects cover around the camera, drawn as it moves.
    occupancy: Eager<OccupancyPass> => [Occupancy],
    /// Local lights binned into the view's clusters, created with the first
    /// frame that has some.
    lights: Lazy<LightPass> => [Lights],
    /// Ground cover grown on the terrain.
    ground_cover: Eager<GroundCoverPass> => [GroundCover],
    /// Clouds: a slice of their dome each frame, which the sky shows, and
    /// their march in the view, laid over the scene.
    clouds: Eager<CloudPass> => [CloudDome, CloudMarch, CloudComposite],
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
