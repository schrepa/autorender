#![expect(
    clippy::expect_used,
    reason = "only a holder's preparation makes a Ready, with its feature"
)]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use engine_core::RenderSnapshot;

use crate::clouds::CloudPass;
use crate::debug_view::DebugViewPass;
use crate::decals::DecalPass;
use crate::environment::Environment;
use crate::exposure::ExposurePass;
use crate::ground_cover::GroundCoverPass;
use crate::lights::LightPass;
use crate::materials::MaterialStore;
use crate::occupancy::OccupancyPass;
use crate::post::PostPass;
use crate::probes::ProbePass;
use crate::profiling::GpuProfiler;
use crate::render_graph::{Clear, FrameShape, Load, Node, Ops, RenderGraph, Resource};
use crate::scatter::ScatterPass;
use crate::sky::SkyPass;
use crate::targets::Targets;
use crate::temporal::TemporalPass;
use crate::transparent::Transparency;
use crate::water::WaterPass;

pub(crate) trait RenderFeature {
    type Frame;
    type Stats;

    fn declare(frame: &Self::Frame, shape: &FrameShape, graph: &mut RenderGraph);

    fn encode(
        &self,
        frame: &Self::Frame,
        node: &Node,
        encode: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32;

    fn after_submit(&mut self, _frame: Option<&Self::Frame>) {}

    fn capacity_bytes(&self) -> u64;

    fn stats(
        this: Option<&Self>,
        frame: Option<&Self::Frame>,
        snapshot: &RenderSnapshot,
    ) -> Self::Stats;

    const LABELS: &'static [&'static str] = &[];
}

pub(crate) struct Lazy<F> {
    state: State<F>,
    #[cfg(feature = "dev")]
    fault: std::cell::Cell<Fault>,
}

enum State<F> {
    Absent,
    Created {
        feature: F,
        validation: Option<Validation>,
    },
    Disabled,
}

type Validation = Pin<Box<dyn Future<Output = Option<wgpu::Error>>>>;

#[cfg(feature = "dev")]
#[derive(Default)]
enum Fault {
    #[default]
    None,
    Injected,
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
    #[cfg(feature = "dev")]
    const LAZY: bool = true;

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

    pub(crate) fn prepare_created(
        &mut self,
        prepare: impl FnOnce(&mut F) -> Option<F::Frame>,
    ) -> Option<Ready<F::Frame>> {
        prepare(self.get_mut()?).map(|frame| Ready { frame })
    }

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

    pub(crate) fn release(&mut self) {
        if matches!(self.state, State::Created { .. }) {
            self.state = State::Absent;
        }
    }

    pub(crate) fn disabled(&self) -> bool {
        matches!(self.state, State::Disabled)
    }

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

    fn disable(&mut self) -> bool {
        #[cfg(feature = "dev")]
        drop(self.fault.take());
        !matches!(
            std::mem::replace(&mut self.state, State::Disabled),
            State::Disabled
        )
    }

    #[cfg(feature = "dev")]
    fn inject(&mut self) -> bool {
        self.release();
        self.fault.set(Fault::Injected);
        true
    }

    #[cfg(feature = "dev")]
    fn encode_fault(&self, encoder: &mut wgpu::CommandEncoder) {
        match self.fault.take() {
            Fault::Made(buffer) => encoder.clear_buffer(&buffer, 0, None),
            fault @ (Fault::None | Fault::Injected) => self.fault.set(fault),
        }
    }

    fn each_mut(&mut self) -> Option<&mut F> {
        self.get_mut()
    }

    fn ready(&self, _ready: &Ready<F::Frame>) -> &F {
        self.get()
            .expect("only Lazy::prepare makes a Ready, once it created its feature")
    }
}

fn poll(validation: &mut Validation) -> Poll<Option<wgpu::Error>> {
    validation
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
}

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

pub(crate) struct Eager<F> {
    feature: F,
}

impl<F: RenderFeature> Eager<F> {
    #[cfg(feature = "dev")]
    const LAZY: bool = false;

    pub(crate) fn new(feature: F) -> Self {
        Self { feature }
    }

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

    fn each_mut(&mut self) -> Option<&mut F> {
        Some(&mut self.feature)
    }

    fn ready(&self, _ready: &Ready<F::Frame>) -> &F {
        &self.feature
    }

    fn poll(&mut self) -> bool {
        false
    }

    fn disable(&mut self) -> bool {
        false
    }

    fn disabled(&self) -> bool {
        false
    }

    #[cfg(feature = "dev")]
    fn inject(&mut self) -> bool {
        false
    }

    #[cfg(feature = "dev")]
    fn encode_fault(&self, _encoder: &mut wgpu::CommandEncoder) {}
}

pub(crate) struct Ready<F> {
    frame: F,
}

impl<F> Ready<F> {
    pub(crate) fn frame(&self) -> &F {
        &self.frame
    }
}

pub(crate) trait MeshDraws {
    fn transparent(&self, pass: &mut wgpu::RenderPass<'_>) -> u32;
}

pub(crate) trait AttachmentViews {
    fn view(&self, resource: Resource) -> &wgpu::TextureView;

    fn region(&self, node: &Node) -> Option<(u32, u32)>;
}

pub(crate) struct Encode<'f> {
    views: &'f dyn AttachmentViews,
    pub(crate) meshes: &'f dyn MeshDraws,
    pub(crate) device: &'f wgpu::Device,
    pub(crate) profiler: &'f GpuProfiler,
    pub(crate) camera: &'f wgpu::BindGroup,
    pub(crate) terrain_camera: &'f wgpu::BindGroup,
    pub(crate) water_camera: &'f wgpu::BindGroup,
    pub(crate) materials: &'f MaterialStore,
    pub(crate) lights: Option<&'f wgpu::BindGroup>,
    pub(crate) targets: Targets,
    pub(crate) depth: &'f wgpu::Texture,
    pub(crate) scene: Option<&'f wgpu::Texture>,
}

impl<'f> Encode<'f> {
    pub(crate) fn new(
        (views, meshes): (&'f dyn AttachmentViews, &'f dyn MeshDraws),
        (device, profiler): (&'f wgpu::Device, &'f GpuProfiler),
        ((camera, terrain_camera, water_camera), materials, lights): (
            (
                &'f wgpu::BindGroup,
                &'f wgpu::BindGroup,
                &'f wgpu::BindGroup,
            ),
            &'f MaterialStore,
            Option<&'f wgpu::BindGroup>,
        ),
        (targets, depth, scene): (Targets, &'f wgpu::Texture, Option<&'f wgpu::Texture>),
    ) -> Self {
        Self {
            views,
            meshes,
            device,
            profiler,
            camera,
            terrain_camera,
            water_camera,
            materials,
            lights,
            targets,
            depth,
            scene,
        }
    }

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
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
    });
    if let Some(region) = views.region(node) {
        crate::targets::draw_within(&mut pass, region);
    }
    pass
}

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

macro_rules! render_features {
    ($(
        $(#[$doc:meta])*
        $field:ident: $holder:ident<$feature:ty> => [$($pass:ident),* $(,)?],
    )*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum FeaturePass {
            $($($pass,)*)*
        }

        pub(crate) struct RenderFeatures {
            $($(#[$doc])* pub(crate) $field: $holder<$feature>,)*
        }

        #[derive(Default)]
        pub(crate) struct FeatureFrames {
            $(pub(crate) $field: Option<Ready<<$feature as RenderFeature>::Frame>>,)*
        }

        impl RenderFeatures {
            pub(crate) fn declare(
                frames: &FeatureFrames,
                shape: &FrameShape,
                graph: &mut RenderGraph,
            ) {
                $(if let Some(ready) = &frames.$field {
                    <$feature as RenderFeature>::declare(ready.frame(), shape, graph);
                })*
            }

            pub(crate) fn encode(
                &self,
                pass: FeaturePass,
                frames: &FeatureFrames,
                node: &Node,
                encode: &Encode<'_>,
                encoder: &mut wgpu::CommandEncoder,
            ) -> u32 {
                match pass {
                    $($(FeaturePass::$pass => {
                        let ready = frames
                            .$field
                            .as_ref()
                            .expect("features declare their passes only with what they prepared");
                        #[cfg(feature = "dev")]
                        self.$field.encode_fault(encoder);
                        self.$field.ready(ready).encode(ready.frame(), node, encode, encoder)
                    })*)*
                }
            }

            pub(crate) fn after_submit(&mut self, frames: &FeatureFrames) {
                $(if let Some(feature) = self.$field.each_mut() {
                    feature.after_submit(frames.$field.as_ref().map(Ready::frame));
                })*
            }

            pub(crate) const NAMES: &'static [&'static str] = &[$(stringify!($field)),*];

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

            #[cfg(feature = "dev")]
            pub(crate) fn lazy_names() -> impl Iterator<Item = &'static str> {
                let lazy = [$(<$holder<$feature>>::LAZY),*];
                Self::NAMES
                    .iter()
                    .zip(lazy)
                    .filter_map(|(name, lazy)| lazy.then_some(*name))
            }

            #[cfg(feature = "dev")]
            pub(crate) fn inject(&mut self, name: &str) -> bool {
                $(if name == stringify!($field) {
                    return self.$field.inject();
                })*
                false
            }

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
    scatter: Eager<ScatterPass> => [Scatter],
    occupancy: Eager<OccupancyPass> => [Occupancy],
    lights: Lazy<LightPass> => [Lights],
    ground_cover: Eager<GroundCoverPass> => [GroundCover],
    clouds: Eager<CloudPass> => [CloudDome, CloudMarch, CloudComposite],
    decals: Lazy<DecalPass> => [Decals],
    water: Eager<WaterPass> => [Water],
    transparency: Eager<Transparency> => [Transparent, TransparentComposite],
    post: Lazy<PostPass> => [Effects, Composite],
    temporal: Lazy<TemporalPass> => [TemporalResolve],
    debug_view: Lazy<DebugViewPass> => [DebugView],
    sky: Eager<SkyPass> => [],
    environment: Eager<Environment> => [],
    probes: Eager<ProbePass> => [],
    exposure: Lazy<ExposurePass> => [Exposure],
}
