#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

use std::any::TypeId;
use std::marker::PhantomData;
use std::ops::Range;

use bevy_ecs::resource::Resource;
use bevy_ecs::system::SystemParamItem;
use engine_core::{
    Bricks, DrawPoint, Entry, Error, List, MaterialHandle, MeshHandle, Read, Readback, Slot,
    TextureHandle, Var, gpu_struct,
};
use engine_registry::RecordExt;

pub use wgpu;

include!(concat!(env!("OUT_DIR"), "/source.rs"));

mod bindings;
mod cached;
mod gpu_calls;
mod gpu_data;
mod pipeline_layouts;

mod depth_pyramid;
mod entries;
mod faces;
mod frame_group;
mod gpu_layouts;
mod gpu_uploads;
mod graph;
mod impostors;
mod lazy;
mod linker;
mod parts;
mod pieces;
mod pipelines;
mod places;
mod registration;
mod shaders;
mod uploads;

mod buffers;
mod materials;
mod memory;
mod mesh_arena;
mod mipmaps;
mod platform;
mod post;
mod profiling;
mod renderer;
mod staging;
mod targets;

#[doc(hidden)]
pub use memory::GpuMemory;
#[doc(hidden)]
pub use platform::{AdapterInfo, RenderTarget, read_range};
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use platform::{GpuPreference, NativeWindow, start_native_instances_with};
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use renderer::GpuMark;
#[doc(hidden)]
pub use renderer::{GpuPassTimes, Renderer, StartingRenderer};

pub trait GpuPart: Sized + Send + Sync + 'static {
    type Reads: Read + 'static;
    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error>;
    fn prepare(&mut self, reads: SystemParamItem<'_, '_, Self::Reads>, cx: &mut Prepare<'_, Self>);
    fn encode(&mut self, _pass: Pass<Self>, _cx: &mut Encode<'_, Self>) -> u32 {
        0
    }
}

pub trait RenderBricks {
    fn part<P: GpuPart>(&mut self) -> &mut Self;
    fn draws<P: GpuPart, D: DrawPoint>(
        &mut self,
        layer: D::Layer,
        draw: fn(&P, &D::Args, &mut Draw<'_, P, D>) -> u32,
    ) -> &mut Self;
}

impl RenderBricks for Bricks<'_> {
    fn part<P: GpuPart>(&mut self) -> &mut Self {
        self.record()
            .render
            .push(Box::new(registration::PartRecord::of::<P>()));
        self
    }

    fn draws<P: GpuPart, D: DrawPoint>(
        &mut self,
        layer: D::Layer,
        draw: fn(&P, &D::Args, &mut Draw<'_, P, D>) -> u32,
    ) -> &mut Self {
        let record = self.record();
        record
            .render
            .push(Box::new(registration::PartRecord::of::<P>()));
        record
            .render
            .push(Box::new(registration::DrawRecord::<D>::of::<P>(layer, draw)));
        self
    }
}

macro_rules! handle {
    ($name:ident <$($t:ident),+>) => {
        pub struct $name<$($t),+>(u32, PhantomData<fn() -> ($($t,)+)>);
        impl<$($t),+> Clone for $name<$($t),+> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<$($t),+> Copy for $name<$($t),+> {}
        impl<$($t),+> PartialEq for $name<$($t),+> {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
        impl<$($t),+> Eq for $name<$($t),+> {}
        impl<$($t),+> std::hash::Hash for $name<$($t),+> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.0.hash(state)
            }
        }
        impl<$($t),+> std::fmt::Debug for $name<$($t),+> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "{}({})", ::core::stringify!($name), self.0)
            }
        }
    };
}

handle!(Texture<P>);
handle!(Buffer<P>);
handle!(Sampler<P>);
handle!(Pass<P>);
handle!(Pipeline<P, S>);
handle!(ComputePipeline<P>);
handle!(DrawInput<P, S>);

pub struct Own;

pub trait Scope<P> {
    type Tag: 'static;
}

impl<P> Scope<P> for Pass<P> {
    type Tag = Own;
}

impl<P, D: DrawPoint> Scope<P> for D {
    type Tag = D;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifetime {
    Frame,
    Kept,
    Previous,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Layer {
    Occluders,
    Opaque,
    Late,
    Backdrop,
    Media,
    Over,
    Blend,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Load,
    Clear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewTarget {
    Scene,
    Depth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfterRead {
    Keep,
    Clear,
}

pub enum Bind<P> {
    Texture(Texture<P>),
    Slice {
        texture: Texture<P>,
        mips: Range<u32>,
        layers: Range<u32>,
    },
    Buffer(Buffer<P>),
    Sampler(Sampler<P>),
    Target(ViewTarget),
    Asset,
}

pub enum Copied<P> {
    Buffer(Buffer<P>),
    Texture(Texture<P>),
}

pub enum CopyRegion {
    Bytes {
        from: u64,
        to: u64,
        size: u64,
    },
    Texels {
        from: wgpu::Origin3d,
        to: wgpu::Origin3d,
        size: wgpu::Extent3d,
    },
}

pub enum Attach<P> {
    Declared,
    Depth(Texture<P>, Option<u32>),
    Color(Texture<P>, Option<u32>),
}

pub enum Count<P> {
    Groups([u32; 3]),
    Indirect(DrawInput<P, Own>, u64),
}

pub enum Shape<'a, P, S> {
    Pulled {
        vertices: Range<u32>,
        instances: Range<u32>,
    },
    Buffers {
        vertices: &'a [DrawInput<P, S>],
        indices: Option<DrawInput<P, S>>,
        elements: Range<u32>,
        instances: Range<u32>,
    },
    Indirect {
        vertices: &'a [DrawInput<P, S>],
        indices: Option<DrawInput<P, S>>,
        args: DrawInput<P, S>,
        offset: u64,
    },
    Mesh {
        mesh: MeshHandle,
        material: Option<MaterialHandle>,
        instances: Range<u32>,
    },
    MeshIndirect {
        mesh: MeshHandle,
        material: Option<MaterialHandle>,
        args: DrawInput<P, S>,
        offset: u64,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum DepthUse {
    None,
    Test,
    Write { bias: wgpu::DepthBiasState },
    Overwrite,
}

#[derive(Clone, Debug)]
pub enum VertexLayout {
    Mesh,
    Buffer(wgpu::VertexBufferLayout<'static>),
}

pub struct RenderPipeline<'a> {
    pub vertex: Entry,
    pub fragment: Option<Entry>,
    pub debug: Option<Entry>,
    pub vertices: &'a [VertexLayout],
    pub primitive: wgpu::PrimitiveState,
    pub depth: DepthUse,
    pub blend: Option<wgpu::BlendState>,
    pub writes: wgpu::ColorWrites,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerDesc {
    pub filter: wgpu::FilterMode,
    pub address: wgpu::AddressMode,
    pub compare: Option<wgpu::CompareFunction>,
}

impl SamplerDesc {
    pub const NEAREST: Self = Self {
        filter: wgpu::FilterMode::Nearest,
        address: wgpu::AddressMode::ClampToEdge,
        compare: None,
    };
    pub const LINEAR_CLAMP: Self = Self {
        filter: wgpu::FilterMode::Linear,
        address: wgpu::AddressMode::ClampToEdge,
        compare: None,
    };
    pub const COMPARE_LINEAR: Self = Self {
        filter: wgpu::FilterMode::Linear,
        address: wgpu::AddressMode::ClampToEdge,
        compare: Some(wgpu::CompareFunction::LessEqual),
    };
}

pub struct Install<'i, P> {
    core: &'i mut engine_registry::InstallCore,
    part: PhantomData<fn() -> P>,
}

impl<P: GpuPart> Install<'_, P> {
    pub fn texture(
        &mut self,
        label: &'static str,
        format: wgpu::TextureFormat,
        lifetime: Lifetime,
    ) -> Texture<P> {
        todo!()
    }

    pub fn buffer(&mut self, label: &'static str, lifetime: Lifetime) -> Buffer<P> {
        todo!()
    }

    pub fn sampler(&mut self, desc: SamplerDesc) -> Sampler<P> {
        todo!()
    }

    pub fn frame(&mut self, variable: Var, bound: Bind<P>) -> Result<(), Error> {
        todo!()
    }

    pub fn pass(&mut self, name: &'static str) -> PassBuilder<'_, P> {
        todo!()
    }

    pub fn copy(&mut self, name: &'static str, from: Copied<P>, to: Copied<P>) -> Pass<P> {
        todo!()
    }

    pub fn clear(&mut self, name: &'static str, buffer: Buffer<P>) -> Pass<P> {
        todo!()
    }

    pub fn render_pipeline<S: Scope<P>>(
        &mut self,
        scope: S,
        desc: &RenderPipeline<'_>,
        bound: &[(Var, Bind<P>)],
    ) -> Result<Pipeline<P, S::Tag>, Error> {
        todo!()
    }

    pub fn compute_pipeline(
        &mut self,
        pass: Pass<P>,
        entry: Entry,
        bound: &[(Var, Bind<P>)],
    ) -> Result<ComputePipeline<P>, Error> {
        todo!()
    }

    pub fn draw_input<S: Scope<P>>(&mut self, scope: S, buffer: Buffer<P>) -> DrawInput<P, S::Tag> {
        todo!()
    }

    pub fn claim<D: DrawPoint>(&mut self, pass: Pass<P>) -> Result<Draws<D>, Error> {
        todo!()
    }
}

pub struct PassBuilder<'b, P> {
    core: &'b mut engine_registry::InstallCore,
    pass: u32,
    part: PhantomData<fn() -> P>,
}

impl<P: GpuPart> PassBuilder<'_, P> {
    pub fn at(self, layer: Layer) -> Self {
        todo!()
    }

    pub fn after(self, pass: Pass<P>) -> Self {
        todo!()
    }

    pub fn color(self, target: Texture<P>, access: Access) -> Self {
        todo!()
    }

    pub fn depth(self, target: Texture<P>, access: Access) -> Self {
        todo!()
    }

    pub fn id(self) -> Pass<P> {
        Pass(self.pass, PhantomData)
    }
}

pub struct Prepare<'f, P> {
    core: &'f mut engine_registry::FrameCore,
    part: PhantomData<fn() -> P>,
}

impl<P: GpuPart> Prepare<'_, P> {
    pub fn run(&mut self, pass: Pass<P>) {
        todo!()
    }

    pub fn allocate(&mut self, texture: Texture<P>, size: wgpu::Extent3d, mips: u32) {
        todo!()
    }

    pub fn allocate_buffer(&mut self, buffer: Buffer<P>, bytes: u64) {
        todo!()
    }

    pub fn write<T: bytemuck::Pod>(&mut self, buffer: Buffer<P>, offset: u64, data: &[T]) {
        todo!()
    }

    pub fn write_texture(
        &mut self,
        texture: Texture<P>,
        at: wgpu::Origin3d,
        bytes: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        todo!()
    }

    pub fn copy(&mut self, pass: Pass<P>, regions: &[CopyRegion]) {
        todo!()
    }

    pub fn cameras(&mut self, views: &[CameraUniform]) {
        todo!()
    }

    pub fn fill<S>(&mut self, pipeline: &Pipeline<P, S>, textures: &[TextureHandle]) -> Pipeline<P, S> {
        todo!()
    }

    pub fn read_back<R: Readback>(
        &mut self,
        tag: u32,
        buffer: Buffer<P>,
        range: Range<u64>,
        then: AfterRead,
    ) -> bool {
        todo!()
    }
}

pub struct Encode<'f, P> {
    core: &'f mut engine_registry::EncodeCore,
    part: PhantomData<fn() -> P>,
}

impl<P: GpuPart> Encode<'_, P> {
    pub fn render(&mut self, attach: Attach<P>, view: Option<u32>) -> Draw<'_, P, Own> {
        todo!()
    }

    pub fn dispatch(&mut self, pipeline: &ComputePipeline<P>, count: Count<P>) -> u32 {
        todo!()
    }
}

pub struct Draw<'e, P, S> {
    raw: engine_registry::RawDraw<'e>,
    part: PhantomData<fn() -> (P, S)>,
}

impl<P: GpuPart, S: 'static> Draw<'_, P, S> {
    pub fn draw(&mut self, pipeline: &Pipeline<P, S>, shape: Shape<'_, P, S>) {
        todo!()
    }
}

impl<P: GpuPart> Draw<'_, P, Own> {
    pub fn scissor(&mut self, rect: [u32; 4]) {
        todo!()
    }
}

pub struct Draws<D: DrawPoint> {
    members: Box<[engine_registry::DrawMember<D>]>,
}

impl<D: DrawPoint> Draws<D> {
    pub fn draw<P: GpuPart>(&self, owner: &mut Draw<'_, P, Own>, layer: D::Layer, args: &D::Args) -> u32 {
        todo!()
    }
}

gpu_struct! {
    pub struct CameraUniform {
        pub view_projection: [[f32; 4]; 4],
        pub previous_view_projection: [[f32; 4]; 4],
        pub inverse_view_projection: [[f32; 4]; 4],
        pub position: [f32; 4],
        pub jitter: [f32; 4],
        pub options: [f32; 4],
        pub wind: [f32; 4],
        pub region: [f32; 4],
        pub detail: [f32; 4],
        pub view: [u32; 4],
    }
}

#[derive(Resource, Default)]
pub struct MainView;

impl Slot for MainView {
    const NAME: &'static str = "main_view";
}

impl DrawPoint for MainView {
    type Layer = Layer;
    type Args = ();
}

pub struct Output {
    pub name: &'static str,
    pub format: wgpu::TextureFormat,
    pub layers: &'static [Layer],
}

pub struct ViewOutputs;

impl List for ViewOutputs {
    const NAME: &'static str = "view_outputs";
    type Member = Output;
    const WGSL: Option<&'static str> = Some(include_str!("places/view_outputs.wgsl"));
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct Resolve {
    pub offset: glam::Vec2,
    pub turn: f32,
}

impl Slot for Resolve {
    const NAME: &'static str = "resolve";
    const WGSL: Option<&'static str> = Some(include_str!("places/resolve.wgsl"));
}

pub struct PostScene;

impl List for PostScene {
    const NAME: &'static str = "post_scene";
    type Member = ();
    const WGSL: Option<&'static str> = Some(include_str!("places/post_scene.wgsl"));
}

pub struct PostLight;

impl List for PostLight {
    const NAME: &'static str = "post_light";
    type Member = ();
    const WGSL: Option<&'static str> = Some(include_str!("places/post_light.wgsl"));
}

pub struct PostDisplay;

impl List for PostDisplay {
    const NAME: &'static str = "post_display";
    type Member = ();
    const WGSL: Option<&'static str> = Some(include_str!("places/post_display.wgsl"));
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DebugShown {
    pub owner: Option<TypeId>,
    pub local: u32,
}

impl Slot for DebugShown {
    const NAME: &'static str = "debug";
    const WGSL: Option<&'static str> = Some(include_str!("places/debug.wgsl"));
}

pub struct DebugMode {
    pub name: &'static str,
    pub decode: fn(ids: [u32; 4], color: [f32; 4]) -> serde_json::Value,
}

pub struct DebugModes;

impl List for DebugModes {
    const NAME: &'static str = "debug_modes";
    type Member = DebugMode;
    const WGSL: Option<&'static str> = Some(include_str!("places/debug_modes.wgsl"));
}
