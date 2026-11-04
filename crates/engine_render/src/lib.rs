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

use crate::draws::{DrawMember, RawDraw};

pub use wgpu;

mod bindings;
mod draws;
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
mod pyramid;
mod registration;
mod shaders;
mod shared;
mod sizes;

mod buffers;
mod materials;
mod memory;
mod mesh_arena;
mod mipmaps;
mod platform;
mod profiling;
mod renderer;
mod targets;

pub use gpu_data::{
    ground_data, jitter_clip, mip_levels, pack_color, pack_surface, pixel_angle, pixels_per_meter,
    planes, pyramid_extents,
};
pub use machine::{DEPTH_FORMAT, SCENE_FORMAT};
pub use materials::gpu_format as texture_format;
pub use materials::{MATERIAL_ANISOTROPY, material_variant};
#[doc(hidden)]
pub use memory::GpuMemory;
#[doc(hidden)]
pub use platform::{AdapterInfo, RenderTarget, read_range};
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use platform::{GpuPreference, NativeWindow, start_native_instances_with};
pub use pyramid::{PYRAMID_GROUP, PyramidExtent};
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use renderer::GpuMark;
#[doc(hidden)]
pub use renderer::{
    GpuPassTimes, Installation, InstalledPart, InstalledPiece, Renderer, StartingRenderer,
    install_noop,
};
pub use shaders::FULLSCREEN;
#[doc(hidden)]
pub use shaders::core_modules;

pub const IMPOSTOR_VERTICES: u32 = 6;

pub const DEV: bool = cfg!(feature = "dev");

pub fn level_region(view: &engine_core::View, level: u32) -> (u32, u32) {
    let (display, render) = (view.display_size(), view.render_size());
    targets::FrameRegion::new(
        targets::FrameSize {
            width: display.x,
            height: display.y,
        },
        targets::FrameSize {
            width: render.x,
            height: render.y,
        },
    )
    .level_region(level)
}

pub mod debug {
    macro_rules! debug_consts {
        ($($name:ident = $value:expr;)*) => {
            $(pub const $name: u32 = $value;)*
            pub(crate) const DECLARED: &[(&str, u32)] = &[$((stringify!($name), $name)),*];
        };
    }

    debug_consts! {
        DEBUG_KIND_MESH = 1;
        DEBUG_KIND_PLACED = 2;
        DEBUG_KIND_CARD = 3;
        DEBUG_KIND_PLACED_CARD = 4;
        DEBUG_KIND_TERRAIN = 6;
        DEBUG_KIND_COVER = 7;
        DEBUG_KIND_WATER = 8;
        DEBUG_KIND_FALLS = 10;
        DEBUG_KIND_SKY = 11;
        DEBUG_LEVEL_SHIFT = 4;
        DEBUG_FADING = 0x100;
        DEBUG_CASTS = 0x200;
        DEBUG_SOURCE_SHIFT = 16;
        DEBUG_NO_MATERIAL = 0xff;
    }
}

engine_core::settings! {
    #[section("render")]
    pub struct RenderSettings {
        pub readback_slots: u32 = 3 => at_least(1),
        pub readback_slot_bytes: Option<u64> = None => at_least(4),
    }
}

pub trait GpuPart: Sized + Send + Sync + 'static {
    type Reads: Read + 'static;
    const NAME: &'static str = "";
    const LAZY: bool = false;
    const LABELS: &'static [&'static str] = &[];
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
    fn generated(&mut self, wgsl: &'static [&'static str]) -> &mut Self;
    fn resolve_pass(&mut self, names: PassNames) -> &mut Self;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassNames {
    pub pass: &'static str,
    pub label: &'static str,
    pub span: &'static str,
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
            .push(Box::new(registration::DrawAny::of::<P, D>(layer, draw)));
        self
    }

    fn generated(&mut self, wgsl: &'static [&'static str]) -> &mut Self {
        self.record()
            .render
            .push(Box::new(registration::Generated(wgsl)));
        self
    }

    fn resolve_pass(&mut self, names: PassNames) -> &mut Self {
        self.record()
            .render
            .push(Box::new(registration::ResolvePass(names)));
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
    Debug(u32),
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
    Target(ViewTarget),
    Canvas,
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
    Levels {
        from: (u32, wgpu::Origin3d),
        to: (u32, wgpu::Origin3d),
        size: wgpu::Extent3d,
    },
    ToBuffer {
        from: wgpu::Origin3d,
        to: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    },
}

pub enum Attach<P> {
    Declared,
    Depth(Texture<P>, Option<u32>),
    Color(Texture<P>, Option<u32>),
    ColorMip(Texture<P>, u32, u32),
    ColorDepth(Texture<P>, Option<u32>, Texture<P>),
    Layers {
        texture: Texture<P>,
        mip: u32,
        layers: Range<u32>,
    },
    Colors(Vec<(Texture<P>, u32)>),
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
    IndirectAt {
        vertices: &'a [(DrawInput<P, S>, u64)],
        indices: Option<DrawInput<P, S>>,
        args: DrawInput<P, S>,
        offset: u64,
    },
    MeshIndirect {
        mesh: MeshHandle,
        material: Option<MaterialHandle>,
        args: DrawInput<P, S>,
        offset: u64,
        streams: &'a [(DrawInput<P, S>, u64)],
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshPass {
    pub opaque: bool,
    pub prepass: bool,
    pub fading: bool,
    pub grounding: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum DepthUse {
    None,
    Test,
    Write {
        bias: wgpu::DepthBiasState,
    },
    Overwrite,
    Compare {
        compare: wgpu::CompareFunction,
        write: bool,
        bias: wgpu::DepthBiasState,
    },
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
    pub mipmap: wgpu::MipmapFilterMode,
    pub address: [wgpu::AddressMode; 3],
    pub compare: Option<wgpu::CompareFunction>,
    pub anisotropy: u16,
}

impl SamplerDesc {
    pub const NEAREST: Self = Self {
        filter: wgpu::FilterMode::Nearest,
        mipmap: wgpu::MipmapFilterMode::Nearest,
        address: [wgpu::AddressMode::ClampToEdge; 3],
        compare: None,
        anisotropy: 1,
    };
    pub const LINEAR_CLAMP: Self = Self {
        filter: wgpu::FilterMode::Linear,
        mipmap: wgpu::MipmapFilterMode::Linear,
        address: [wgpu::AddressMode::ClampToEdge; 3],
        compare: None,
        anisotropy: 1,
    };
    pub const COMPARE_LINEAR: Self = Self {
        filter: wgpu::FilterMode::Linear,
        mipmap: wgpu::MipmapFilterMode::Linear,
        address: [wgpu::AddressMode::ClampToEdge; 3],
        compare: Some(wgpu::CompareFunction::LessEqual),
        anisotropy: 1,
    };
}

fn bound_of<P>(bound: &Bind<P>) -> install::Bound {
    match bound {
        Bind::Texture(texture) => install::Bound::Texture(texture.0),
        Bind::Slice {
            texture,
            mips,
            layers,
        } => install::Bound::Slice {
            texture: texture.0,
            mips: mips.clone(),
            layers: layers.clone(),
        },
        Bind::Buffer(buffer) => install::Bound::Buffer(buffer.0),
        Bind::Sampler(sampler) => install::Bound::Sampler(sampler.0),
        Bind::Target(view) => install::Bound::Target(*view),
        Bind::Asset => install::Bound::Asset,
    }
}

fn bound_list<P>(bound: &[(Var, Bind<P>)]) -> Vec<(Var, install::Bound)> {
    bound
        .iter()
        .map(|(var, bound)| (*var, bound_of(bound)))
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
        Copied::Target(view) => install::Copied::Target(view),
        Copied::Canvas => install::Copied::Canvas,
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

    pub fn window(&mut self, buffer: Buffer<P>, size: u64) {
        self.core.window(buffer.0, size);
    }

    pub fn features(&self) -> wgpu::Features {
        self.core.features
    }

    pub fn limits(&self) -> &engine_core::DeviceLimits {
        &self.core.limits
    }

    pub fn frame(&mut self, variable: Var, bound: Bind<P>) -> Result<(), Error> {
        self.core.frame(variable, bound_of(&bound))
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
        Pass(
            self.core.pass(name, install::Work::Clear(buffer.0)),
            PhantomData,
        )
    }

    pub fn after(&mut self, pass: Pass<P>, earlier: Pass<P>) {
        if let Some(declared) = self.core.pass_mut(pass.0) {
            declared.after.push(earlier.0);
        }
    }

    pub fn group(&mut self, pass: Pass<P>, name: &'static str) {
        if let Some(declared) = self.core.pass_mut(pass.0) {
            declared.group = Some(name);
        }
    }

    pub fn reaches(&mut self, entry: Entry, var: Var) -> bool {
        self.core.reaches(entry, var)
    }

    pub fn render_pipeline<S: Scope<P>>(
        &mut self,
        scope: S,
        desc: &RenderPipeline<'_>,
        bound: &[(Var, Bind<P>)],
    ) -> Result<Pipeline<P, S::Tag>, Error> {
        self.render_program(&scope, desc, None, bound)
    }

    pub fn render_pipeline_targets<S: Scope<P>>(
        &mut self,
        scope: S,
        desc: &RenderPipeline<'_>,
        targets: &[(Option<wgpu::BlendState>, wgpu::ColorWrites)],
        bound: &[(Var, Bind<P>)],
    ) -> Result<Pipeline<P, S::Tag>, Error> {
        self.render_program(&scope, desc, Some(targets.to_vec()), bound)
    }

    fn render_program<S: Scope<P>>(
        &mut self,
        scope: &S,
        desc: &RenderPipeline<'_>,
        targets: Option<Vec<(Option<wgpu::BlendState>, wgpu::ColorWrites)>>,
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
            targets,
        });
        let pipeline = self
            .core
            .pipeline(scoped(scope), program, bound_list(bound))?;
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
        self.core
            .claim((TypeId::of::<D>(), D::UBER_MATERIALS), D::NAME, pass.0)?;
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

    #[must_use]
    pub fn span(self, name: &'static str) -> Self {
        if let Some(declared) = self.core.pass_mut(self.pass) {
            declared.span = Some(name);
        }
        self
    }

    #[must_use]
    pub fn label(self, label: &'static str) -> Self {
        if let Some(declared) = self.core.pass_mut(self.pass) {
            declared.label = Some(label);
        }
        self
    }

    pub fn color(self, target: Texture<P>, access: Access) -> Self {
        if let Some(pass) = self.core.pass_mut(self.pass) {
            pass.colors.push((target.0, access));
        }
        self
    }

    pub fn color_cleared(self, target: Texture<P>, clear: wgpu::Color) -> Self {
        if let Some(pass) = self.core.pass_mut(self.pass) {
            pass.colors.push((target.0, Access::Clear));
            pass.clears.push((
                target.0,
                graph::Clear::Color([clear.r, clear.g, clear.b, clear.a]),
            ));
        }
        self
    }

    pub fn depth(self, target: Texture<P>, access: Access) -> Self {
        if let Some(pass) = self.core.pass_mut(self.pass) {
            pass.depths.push((target.0, access));
        }
        self
    }

    pub fn depth_cleared(self, target: Texture<P>, clear: f32) -> Self {
        if let Some(pass) = self.core.pass_mut(self.pass) {
            pass.depths.push((target.0, Access::Clear));
            pass.clears.push((target.0, graph::Clear::Depth(clear)));
        }
        self
    }

    pub fn id(self) -> Pass<P> {
        Pass(self.pass, PhantomData)
    }
}

pub struct Prepare<'f, P> {
    core: &'f mut encode::FrameCore,
    fresh: bool,
    part: PhantomData<fn() -> P>,
}

fn delivered<R: Readback>(world: &mut bevy_ecs::world::World, frame: u64, tag: u32, bytes: &[u8]) {
    world.get_resource_or_init::<R>().arrived(frame, tag, bytes);
}

fn region_of(region: &CopyRegion) -> CopyRegion {
    match region {
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
        CopyRegion::Levels { from, to, size } => CopyRegion::Levels {
            from: *from,
            to: *to,
            size: *size,
        },
        CopyRegion::ToBuffer { from, to, size } => CopyRegion::ToBuffer {
            from: *from,
            to: *to,
            size: *size,
        },
    }
}

impl<P: GpuPart> Prepare<'_, P> {
    pub fn run(&mut self, pass: Pass<P>) {
        self.core.run_pass(pass.0);
    }

    pub fn want_overlay(&mut self) {
        self.core.want_overlay();
    }

    pub fn fresh(&self) -> bool {
        self.fresh
    }

    pub fn allocate(&mut self, texture: Texture<P>, size: wgpu::Extent3d, mips: u32) {
        self.core.allocate(texture.0, size, mips);
    }

    pub fn allocate_as(
        &mut self,
        texture: Texture<P>,
        format: wgpu::TextureFormat,
        size: wgpu::Extent3d,
        mips: u32,
    ) {
        self.core.allocate_as(texture.0, format, size, mips);
    }

    pub fn grow(&mut self, texture: Texture<P>, size: wgpu::Extent3d, mips: u32) {
        self.core.grow(texture.0, size, mips);
    }

    pub fn release(&mut self, texture: Texture<P>) {
        self.core.release(texture.0);
    }

    pub fn allocate_buffer(&mut self, buffer: Buffer<P>, bytes: u64) {
        self.core.allocate_buffer(buffer.0, bytes);
    }

    pub fn release_buffer(&mut self, buffer: Buffer<P>) {
        self.core.release_buffer(buffer.0);
    }

    pub fn renew_buffer(&mut self, buffer: Buffer<P>, bytes: u64) {
        self.core.renew_buffer(buffer.0, bytes);
    }

    pub fn clear_buffer_range(&mut self, buffer: Buffer<P>, range: Range<u64>) {
        self.core.clear_buffer_range(buffer.0, range);
    }

    pub fn write<T: bytemuck::Pod>(&mut self, buffer: Buffer<P>, offset: u64, data: &[T]) {
        self.core
            .write(buffer.0, offset, bytemuck::cast_slice(data));
    }

    pub fn grow_buffer(&mut self, buffer: Buffer<P>, bytes: u64) {
        self.core.grow_buffer(buffer.0, bytes);
    }

    pub fn count_upload(&mut self, bytes: u64) {
        self.core.uploads.counted += bytes;
    }

    pub fn memory(&mut self, category: &'static str, bytes: u64) {
        let memory = &mut self.core.uploads.memory;
        match memory.iter_mut().find(|(known, _)| *known == category) {
            Some((_, held)) => *held += bytes,
            None => memory.push((category, bytes)),
        }
    }

    pub fn write_texture(
        &mut self,
        texture: Texture<P>,
        at: wgpu::Origin3d,
        bytes: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        self.core
            .write_texture(texture.0, 0, at, bytes, layout, size);
    }

    pub fn write_texture_level(
        &mut self,
        texture: Texture<P>,
        level: u32,
        at: wgpu::Origin3d,
        bytes: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        self.core
            .write_texture(texture.0, level, at, bytes, layout, size);
    }

    pub fn write_level(
        &mut self,
        texture: Texture<P>,
        (level, layer): (u32, u32),
        size: (u32, u32),
        format: engine_core::TextureFormat,
        bytes: &[u8],
    ) {
        let (layout, extent) = gpu_data::level_copy(format, size, level);
        let at = wgpu::Origin3d {
            x: 0,
            y: 0,
            z: layer,
        };
        self.core
            .write_texture(texture.0, level, at, bytes, layout, extent);
    }

    pub fn mipmaps(&mut self, texture: Texture<P>, layers: Range<u32>) {
        self.core.mipmaps(texture.0, layers);
    }

    pub fn copy(&mut self, pass: Pass<P>, regions: &[CopyRegion]) {
        let regions = regions.iter().map(region_of).collect();
        self.core.copies.retain(|(known, _)| *known != pass.0);
        self.core.copies.push((pass.0, regions));
        self.core.run_pass(pass.0);
    }

    pub fn copy_at_end(&mut self, from: Copied<P>, to: Copied<P>, regions: &[CopyRegion]) {
        let regions = regions.iter().map(region_of).collect();
        self.core
            .queued
            .ending
            .push((copied(from), copied(to), regions));
    }

    pub fn cameras(&mut self, views: &[CameraUniform]) {
        self.core.cameras(views);
    }

    pub fn fill<S>(
        &mut self,
        pipeline: &Pipeline<P, S>,
        textures: &[TextureHandle],
    ) -> Pipeline<P, S> {
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
            once: false,
            deliver: delivered::<R>,
        })
    }

    pub fn read_back_once<R: Readback>(
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
            once: true,
            deliver: delivered::<R>,
        })
    }

    pub fn deliver<R: Readback>(&mut self, tag: u32, bytes: Vec<u8>) {
        self.core.deliver_bytes(tag, bytes, delivered::<R>);
    }

    pub fn refuse(&mut self, error: Error) {
        self.core.refuse(error);
    }

    pub fn canvas(&self) -> Option<wgpu::TextureFormat> {
        self.core
            .canvas_texture
            .as_ref()
            .filter(|texture| texture.usage().contains(wgpu::TextureUsages::COPY_SRC))
            .map(wgpu::Texture::format)
    }
}

pub struct Encode<'f, P> {
    core: &'f mut encode::EncodeCore,
    timer: &'f profiling::GpuProfiler,
    part: PhantomData<fn() -> P>,
}

impl<P: GpuPart> Encode<'_, P> {
    pub fn span(&mut self, name: &'static str) {
        self.core.close_encode_span(self.timer);
        self.core.timed.asked = Some(name);
        if self.core.group_of(self.core.pass) == name {
            self.core.open_encode_span(self.timer);
        }
    }

    pub fn clear(&mut self, buffer: Buffer<P>) {
        self.core.clear_buffer(buffer.0);
    }

    pub fn render(&mut self, attach: Attach<P>, view: Option<u32>) -> Draw<'_, P, Own> {
        let raw = match attach {
            Attach::Declared => encode::AttachRaw::Declared,
            Attach::Depth(texture, layer) => encode::AttachRaw::Depth(texture.0, layer),
            Attach::Color(texture, layer) => {
                encode::AttachRaw::Color(texture.0, 0, layer.unwrap_or(0)..layer.unwrap_or(0) + 1)
            }
            Attach::ColorMip(texture, mip, layer) => {
                encode::AttachRaw::Color(texture.0, mip, layer..layer + 1)
            }
            Attach::ColorDepth(color, layer, depth) => {
                encode::AttachRaw::ColorDepth(color.0, layer, depth.0)
            }
            Attach::Layers {
                texture,
                mip,
                layers,
            } => encode::AttachRaw::Color(texture.0, mip, layers),
            Attach::Colors(colors) => encode::AttachRaw::Colors(
                colors
                    .iter()
                    .map(|(texture, layer)| (texture.0, *layer))
                    .collect(),
            ),
        };
        self.core.open_encode_span(self.timer);
        if !self.core.begin_render(raw, view) {
            self.core.end_render();
        }
        self.core.point = None;
        let part = self.core.part;
        Draw {
            raw: RawDraw::new(&mut *self.core, part, self.timer),
            part: PhantomData,
        }
    }

    pub fn dispatch(&mut self, pipeline: &ComputePipeline<P>, count: Count<P>) -> u32 {
        self.dispatch_at(pipeline, count, &[])
    }

    pub fn dispatch_at(
        &mut self,
        pipeline: &ComputePipeline<P>,
        count: Count<P>,
        offsets: &[u32],
    ) -> u32 {
        let count = match count {
            Count::Groups(groups) => encode::CountRaw::Groups(groups),
            Count::Indirect(input, offset) => encode::CountRaw::Indirect(input.0, offset),
        };
        self.core.open_encode_span(self.timer);
        let part = self.core.part;
        self.core.dispatch(part, pipeline.0, count, offsets)
    }
}

pub struct Draw<'e, P, S> {
    raw: RawDraw<'e>,
    part: PhantomData<fn() -> (P, S)>,
}

fn inputs<P, S>(inputs: &[DrawInput<P, S>]) -> encode::Streams {
    encode::Streams::of(inputs.iter().map(|input| (input.0, 0)))
}

fn streams<P, S>(streams: &[(DrawInput<P, S>, u64)]) -> encode::Streams {
    encode::Streams::of(streams.iter().map(|(input, offset)| (input.0, *offset)))
}

impl<P: GpuPart, S: 'static> Draw<'_, P, S> {
    pub fn draw(&mut self, pipeline: &Pipeline<P, S>, shape: Shape<'_, P, S>) {
        self.issue(pipeline, (shape, &[], MeshPass::default()), None);
    }

    pub fn draw_at(&mut self, pipeline: &Pipeline<P, S>, shape: Shape<'_, P, S>, offsets: &[u32]) {
        self.issue(pipeline, (shape, offsets, MeshPass::default()), None);
    }

    pub fn draw_with(&mut self, pipeline: &Pipeline<P, S>, shape: Shape<'_, P, S>, mesh: MeshPass) {
        self.issue(pipeline, (shape, &[], mesh), None);
    }

    pub fn draw_timed<'l>(
        &mut self,
        pipeline: &Pipeline<P, S>,
        shape: Shape<'_, P, S>,
        label: impl FnOnce() -> String + 'l,
    ) {
        let timer = self.raw.timer();
        let mut label = Some(label);
        let mut named = || label.take().map_or_else(String::new, |label| label());
        self.issue(
            pipeline,
            (shape, &[], MeshPass::default()),
            Some((timer, &mut named)),
        );
    }

    pub fn draw_timed_with<'l>(
        &mut self,
        pipeline: &Pipeline<P, S>,
        (shape, mesh): (Shape<'_, P, S>, MeshPass),
        label: impl FnOnce() -> String + 'l,
    ) {
        let timer = self.raw.timer();
        let mut label = Some(label);
        let mut named = || label.take().map_or_else(String::new, |label| label());
        self.issue(pipeline, (shape, &[], mesh), Some((timer, &mut named)));
    }

    fn issue(
        &mut self,
        pipeline: &Pipeline<P, S>,
        (shape, offsets, mesh): (Shape<'_, P, S>, &[u32], MeshPass),
        timed: encode::TimedDraw<'_>,
    ) {
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
            Shape::IndirectAt {
                vertices,
                indices,
                args,
                offset,
            } => encode::RawShape::IndirectAt {
                vertices: streams(vertices),
                indices: indices.map(|input| input.0),
                args: args.0,
                offset,
            },
            Shape::MeshIndirect {
                mesh,
                material,
                args,
                offset,
                streams: given,
            } => encode::RawShape::MeshIndirect {
                mesh,
                material,
                args: args.0,
                offset,
                streams: streams(given),
            },
        };
        let part = self.raw.part();
        let run = self.raw.run();
        let point = run.point;
        run.draw(part, point, pipeline.0, (shape, offsets, mesh), timed);
    }
}

impl<P: GpuPart> Draw<'_, P, Own> {
    pub fn scissor(&mut self, rect: [u32; 4]) {
        self.raw.run().scissor(rect);
    }

    pub fn within(&mut self, size: (u32, u32)) {
        self.raw.run().within(size);
    }
}

pub struct Draws<D: DrawPoint> {
    members: Box<[DrawMember<D>]>,
}

impl<D: DrawPoint> Draws<D> {
    pub fn draw<P: GpuPart>(
        &self,
        owner: &mut Draw<'_, P, Own>,
        layer: D::Layer,
        args: &D::Args,
    ) -> u32 {
        owner.raw.run().reset_scissor();
        let mut drawn = 0;
        for member in self.members.iter().filter(|member| member.layer == layer) {
            let run = owner.raw.run();
            let Some(held) = run.take_member(member.part) else {
                continue;
            };
            run.point = Some(TypeId::of::<D>());
            drawn += (member.call)(held.part(), args, &mut owner.raw.member(member.part));
            let run = owner.raw.run();
            run.point = None;
            run.give_member(member.part, held);
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

#[derive(Resource, Default)]
pub struct Overlay;

impl Slot for Overlay {
    const NAME: &'static str = "overlay";
}

impl DrawPoint for Overlay {
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

#[derive(Resource, Default)]
pub struct PostTone;

impl Slot for PostTone {
    const NAME: &'static str = "post_tone";
    const WGSL: Option<&'static str> = Some(include_str!("places/post_tone.wgsl"));
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
    pub next_debug: bool,
    pub next_whole: bool,
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
