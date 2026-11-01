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

mod bindings;
mod cached;
mod encode;
mod fixed_linker;
mod frame_group;
mod gpu_calls;
mod gpu_data;
mod gpu_layouts;
mod graph;
mod impostors;
mod install;
mod linker;
mod machine;
mod pieces;
mod pipeline_layouts;
mod pipelines;
mod places;
mod registration;
mod shaders;
mod shared;

mod buffers;
mod materials;
mod memory;
mod mesh_arena;
mod mipmaps;
mod platform;
mod profiling;
mod renderer;
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
        let record = self.record();
        let scope = record.scope();
        record
            .render
            .push(Box::new(registration::PartRecord::of::<P>(scope)));
        self
    }

    fn draws<P: GpuPart, D: DrawPoint>(
        &mut self,
        layer: D::Layer,
        draw: fn(&P, &D::Args, &mut Draw<'_, P, D>) -> u32,
    ) -> &mut Self {
        let record = self.record();
        let scope = record.scope();
        record
            .render
            .push(Box::new(registration::PartRecord::of::<P>(scope)));
        record
            .render
            .push(Box::new(registration::DrawAny::of::<P, D>(scope, layer, draw)));
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
    #[doc(hidden)]
    fn pass_index(&self) -> Option<u32> {
        None
    }
}

impl<P> Scope<P> for Pass<P> {
    type Tag = Own;
    fn pass_index(&self) -> Option<u32> {
        Some(self.0)
    }
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

fn bound_of<P>(bound: Bind<P>) -> install::Bound {
    match bound {
        Bind::Texture(texture) => install::Bound::Texture(texture.0),
        Bind::Slice {
            texture,
            mips,
            layers,
        } => install::Bound::Slice {
            texture: texture.0,
            mips,
            layers,
        },
        Bind::Buffer(buffer) => install::Bound::Buffer(buffer.0),
        Bind::Sampler(sampler) => install::Bound::Sampler(sampler.0),
        Bind::Target(view) => install::Bound::Target(view),
        Bind::Asset => install::Bound::Asset,
    }
}

fn bound_list<P>(bound: &[(Var, Bind<P>)]) -> Vec<(Var, install::Bound)> {
    bound
        .iter()
        .map(|(var, bound)| {
            let owned = match bound {
                Bind::Texture(texture) => Bind::Texture(*texture),
                Bind::Slice {
                    texture,
                    mips,
                    layers,
                } => Bind::Slice {
                    texture: *texture,
                    mips: mips.clone(),
                    layers: layers.clone(),
                },
                Bind::Buffer(buffer) => Bind::Buffer(*buffer),
                Bind::Sampler(sampler) => Bind::Sampler(*sampler),
                Bind::Target(view) => Bind::Target(*view),
                Bind::Asset => Bind::Asset,
            };
            (*var, bound_of(owned))
        })
        .collect()
}

fn scoped<P, S: Scope<P>>(scope: &S) -> install::Scoped {
    match scope.pass_index() {
        Some(pass) => install::Scoped::Pass(pass),
        None => install::Scoped::Point(TypeId::of::<S::Tag>()),
    }
}

fn copied<P>(copied: Copied<P>) -> install::Copied {
    match copied {
        Copied::Buffer(buffer) => install::Copied::Buffer(buffer.0),
        Copied::Texture(texture) => install::Copied::Texture(texture.0),
    }
}

pub struct Install<'i, P> {
    core: &'i mut install::InstallCore,
    part: PhantomData<fn() -> P>,
}

impl<P: GpuPart> Install<'_, P> {
    pub fn texture(
        &mut self,
        label: &'static str,
        format: wgpu::TextureFormat,
        lifetime: Lifetime,
    ) -> Texture<P> {
        Texture(self.core.texture(label, format, lifetime), PhantomData)
    }

    pub fn buffer(&mut self, label: &'static str, lifetime: Lifetime) -> Buffer<P> {
        Buffer(self.core.buffer(label, lifetime), PhantomData)
    }

    pub fn sampler(&mut self, desc: SamplerDesc) -> Sampler<P> {
        Sampler(self.core.sampler(desc), PhantomData)
    }

    pub fn frame(&mut self, variable: Var, bound: Bind<P>) -> Result<(), Error> {
        self.core.frame(variable, bound_of(bound))
    }

    pub fn pass(&mut self, name: &'static str) -> PassBuilder<'_, P> {
        let pass = self.core.pass(name, install::Work::Render);
        PassBuilder {
            core: &mut *self.core,
            pass,
            part: PhantomData,
        }
    }

    pub fn copy(&mut self, name: &'static str, from: Copied<P>, to: Copied<P>) -> Pass<P> {
        let work = install::Work::Copy {
            from: copied(from),
            to: copied(to),
        };
        Pass(self.core.pass(name, work), PhantomData)
    }

    pub fn clear(&mut self, name: &'static str, buffer: Buffer<P>) -> Pass<P> {
        Pass(self.core.pass(name, install::Work::Clear(buffer.0)), PhantomData)
    }

    pub fn render_pipeline<S: Scope<P>>(
        &mut self,
        scope: S,
        desc: &RenderPipeline<'_>,
        bound: &[(Var, Bind<P>)],
    ) -> Result<Pipeline<P, S::Tag>, Error> {
        let program = install::Program::Render(install::RenderDecl {
            vertex: desc.vertex,
            fragment: desc.fragment,
            debug: desc.debug,
            vertices: desc.vertices.to_vec(),
            primitive: desc.primitive,
            depth: desc.depth,
            blend: desc.blend,
            writes: desc.writes,
        });
        let pipeline = self.core.pipeline(scoped(&scope), program, bound_list(bound))?;
        Ok(Pipeline(pipeline, PhantomData))
    }

    pub fn compute_pipeline(
        &mut self,
        pass: Pass<P>,
        entry: Entry,
        bound: &[(Var, Bind<P>)],
    ) -> Result<ComputePipeline<P>, Error> {
        let pipeline = self.core.pipeline(
            install::Scoped::Pass(pass.0),
            install::Program::Compute(entry),
            bound_list(bound),
        )?;
        Ok(ComputePipeline(pipeline, PhantomData))
    }

    pub fn draw_input<S: Scope<P>>(&mut self, scope: S, buffer: Buffer<P>) -> DrawInput<P, S::Tag> {
        DrawInput(self.core.draw_input(scoped(&scope), buffer.0), PhantomData)
    }

    pub fn claim<D: DrawPoint>(&mut self, pass: Pass<P>) -> Result<Draws<D>, Error> {
        self.core.claim(TypeId::of::<D>(), D::NAME, pass.0)?;
        let members = self
            .core
            .members::<registration::DrawRecord<D>>()
            .map(|(part, record)| record.member(part))
            .collect();
        Ok(Draws { members })
    }
}

pub struct PassBuilder<'b, P> {
    core: &'b mut install::InstallCore,
    pass: u32,
    part: PhantomData<fn() -> P>,
}

impl<P: GpuPart> PassBuilder<'_, P> {
    pub fn at(self, layer: Layer) -> Self {
        if let Some(pass) = self.core.pass_mut(self.pass) {
            pass.layer = Some(layer);
        }
        self
    }

    pub fn after(self, pass: Pass<P>) -> Self {
        if let Some(declared) = self.core.pass_mut(self.pass) {
            declared.after.push(pass.0);
        }
        self
    }

    pub fn color(self, target: Texture<P>, access: Access) -> Self {
        if let Some(pass) = self.core.pass_mut(self.pass) {
            pass.colors.push((target.0, access));
        }
        self
    }

    pub fn depth(self, target: Texture<P>, access: Access) -> Self {
        if let Some(pass) = self.core.pass_mut(self.pass) {
            pass.depths.push((target.0, access));
        }
        self
    }

    pub fn id(self) -> Pass<P> {
        Pass(self.pass, PhantomData)
    }
}

pub struct Prepare<'f, P> {
    core: &'f mut encode::FrameCore,
    part: PhantomData<fn() -> P>,
}

fn delivered<R: Readback>(world: &mut bevy_ecs::world::World, frame: u64, tag: u32, bytes: &[u8]) {
    world.get_resource_or_init::<R>().arrived(frame, tag, bytes);
}

impl<P: GpuPart> Prepare<'_, P> {
    pub fn run(&mut self, pass: Pass<P>) {
        self.core.run_pass(pass.0);
    }

    pub fn allocate(&mut self, texture: Texture<P>, size: wgpu::Extent3d, mips: u32) {
        self.core.allocate(texture.0, size, mips);
    }

    pub fn allocate_buffer(&mut self, buffer: Buffer<P>, bytes: u64) {
        self.core.allocate_buffer(buffer.0, bytes);
    }

    pub fn write<T: bytemuck::Pod>(&mut self, buffer: Buffer<P>, offset: u64, data: &[T]) {
        self.core.write(buffer.0, offset, bytemuck::cast_slice(data));
    }

    pub fn write_texture(
        &mut self,
        texture: Texture<P>,
        at: wgpu::Origin3d,
        bytes: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        self.core.write_texture(texture.0, at, bytes, layout, size);
    }

    pub fn copy(&mut self, pass: Pass<P>, regions: &[CopyRegion]) {
        let regions = regions
            .iter()
            .map(|region| match region {
                CopyRegion::Bytes { from, to, size } => CopyRegion::Bytes {
                    from: *from,
                    to: *to,
                    size: *size,
                },
                CopyRegion::Texels { from, to, size } => CopyRegion::Texels {
                    from: *from,
                    to: *to,
                    size: *size,
                },
            })
            .collect();
        self.core.copies.retain(|(known, _)| *known != pass.0);
        self.core.copies.push((pass.0, regions));
        self.core.run_pass(pass.0);
    }

    pub fn cameras(&mut self, views: &[CameraUniform]) {
        self.core.cameras(views);
    }

    pub fn fill<S>(&mut self, pipeline: &Pipeline<P, S>, textures: &[TextureHandle]) -> Pipeline<P, S> {
        Pipeline(self.core.fill(pipeline.0, textures), PhantomData)
    }

    pub fn read_back<R: Readback>(
        &mut self,
        tag: u32,
        buffer: Buffer<P>,
        range: Range<u64>,
        then: AfterRead,
    ) -> bool {
        let part = self.core.part;
        self.core.read_back(encode::Asked {
            part,
            tag,
            buffer: buffer.0,
            range,
            then,
            deliver: delivered::<R>,
        })
    }
}

pub struct Encode<'f, P> {
    core: &'f mut encode::EncodeCore,
    part: PhantomData<fn() -> P>,
}

impl<P: GpuPart> Encode<'_, P> {
    pub fn render(&mut self, attach: Attach<P>, view: Option<u32>) -> Draw<'_, P, Own> {
        let raw = match attach {
            Attach::Declared => encode::AttachRaw::Declared,
            Attach::Depth(texture, layer) => encode::AttachRaw::Depth(texture.0, layer),
            Attach::Color(texture, layer) => encode::AttachRaw::Color(texture.0, layer),
        };
        if !self.core.begin_render(raw, view) {
            self.core.end_render();
        }
        let part = self.core.part;
        Draw {
            raw: encode::RawDraw {
                run: &mut *self.core,
                part,
                point: None,
            },
            part: PhantomData,
        }
    }

    pub fn dispatch(&mut self, pipeline: &ComputePipeline<P>, count: Count<P>) -> u32 {
        let count = match count {
            Count::Groups(groups) => encode::CountRaw::Groups(groups),
            Count::Indirect(input, offset) => encode::CountRaw::Indirect(input.0, offset),
        };
        let part = self.core.part;
        self.core.dispatch(part, pipeline.0, count)
    }
}

pub struct Draw<'e, P, S> {
    raw: encode::RawDraw<'e>,
    part: PhantomData<fn() -> (P, S)>,
}

fn inputs<P, S>(inputs: &[DrawInput<P, S>]) -> Vec<u32> {
    inputs.iter().map(|input| input.0).collect()
}

impl<P: GpuPart, S: 'static> Draw<'_, P, S> {
    pub fn draw(&mut self, pipeline: &Pipeline<P, S>, shape: Shape<'_, P, S>) {
        let shape = match shape {
            Shape::Pulled {
                vertices,
                instances,
            } => encode::RawShape::Pulled {
                vertices,
                instances,
            },
            Shape::Buffers {
                vertices,
                indices,
                elements,
                instances,
            } => encode::RawShape::Buffers {
                vertices: inputs(vertices),
                indices: indices.map(|input| input.0),
                elements,
                instances,
            },
            Shape::Indirect {
                vertices,
                indices,
                args,
                offset,
            } => encode::RawShape::Indirect {
                vertices: inputs(vertices),
                indices: indices.map(|input| input.0),
                args: args.0,
                offset,
            },
            Shape::Mesh {
                mesh,
                material,
                instances,
            } => encode::RawShape::Mesh {
                mesh,
                material,
                instances,
            },
            Shape::MeshIndirect {
                mesh,
                material,
                args,
                offset,
            } => encode::RawShape::MeshIndirect {
                mesh,
                material,
                args: args.0,
                offset,
            },
        };
        let (part, point) = (self.raw.part, self.raw.point);
        self.raw.run.draw(part, point, pipeline.0, shape);
    }
}

impl<P: GpuPart> Draw<'_, P, Own> {
    pub fn scissor(&mut self, rect: [u32; 4]) {
        self.raw.run.scissor(rect);
    }
}

pub struct Draws<D: DrawPoint> {
    members: Box<[registration::DrawMember<D>]>,
}

impl<D: DrawPoint> Draws<D> {
    pub fn draw<P: GpuPart>(&self, owner: &mut Draw<'_, P, Own>, layer: D::Layer, args: &D::Args) -> u32 {
        let run = &mut *owner.raw.run;
        run.reset_scissor();
        let mut drawn = 0;
        for member in self.members.iter().filter(|member| member.layer == layer) {
            let at = member.part as usize;
            let live = run.live.get(at).copied().unwrap_or(false)
                && run.faulted.get(at).is_some_and(Option::is_none);
            if !live {
                continue;
            }
            let Some(held) = run.parts.get_mut(at).and_then(Option::take) else {
                continue;
            };
            let saved = run.part;
            run.part = member.part;
            drawn += (member.call)(
                held.part(),
                args,
                &mut encode::RawDraw {
                    run: &mut *run,
                    part: member.part,
                    point: Some(TypeId::of::<D>()),
                },
            );
            run.part = saved;
            if let Some(slot) = run.parts.get_mut(at) {
                *slot = Some(held);
            }
        }
        drawn
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
