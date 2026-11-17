#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

use std::marker::PhantomData;
use std::ops::Range;

use bevy_ecs::system::SystemParamItem;
use engine_core::{
    Entry, Error, GeometryHandle, MaterialHandle, PassBit, PassName, Read, Readable, Readback,
    TextureHandle, Var, gpu_struct,
};

use crate::draws::RawDraw;

pub use engine_core::TargetName;
pub use wgpu;

mod drawn;
mod draws;
mod encode;
mod gpu_calls;
mod gpu_data;
mod graph;
mod install;
mod kinds;
mod linker;
mod machine;
mod pipelines;
mod registration;
mod render_bricks;
mod shaders;
mod shared;
mod sizes;
mod surface;
mod uploads;

mod materials;
mod mesh_arena;
mod mipmaps;
mod platform;
mod profiling;
mod readback;
mod renderer;
mod targets;

pub use engine_place::{PointInput, PointSet};
pub use gpu_data::{mip_levels, pack_color, pack_surface};
pub use install::FragmentUse;
pub use kinds::{
    Form, FormPass, FormSide, KindForm, KindSurface, OwnFragments, PassFragment, PerPass,
    RoleFragments, Varyings,
};
pub use materials::gpu_format as texture_format;
#[doc(hidden)]
pub use platform::{AdapterInfo, RenderTarget};
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use platform::{GpuPreference, NativeWindow, keep_pipeline_cache_in};
pub use render_bricks::RenderBricks;
#[doc(hidden)]
pub use renderer::{
    Ahead, DeviceHealth, Installation, Renderer, StartingRenderer, device_limits, install_noop,
};
pub use shaders::FULLSCREEN;
pub use surface::{
    DrawState, FaceCull, LightingModule, MaterialValue, MaterialVariant, ModelForm, OpenSlot,
    SlotModule, SlotPrepare, SurfaceKind, SurfaceTerms, material_state,
};
pub use targets::SharedTarget;
pub use uploads::{UploadRange, listed_ranges};

pub const DEV: bool = cfg!(feature = "dev");

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
    const MEMORY: &'static [&'static str] = &[];
    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error>;
    fn prepare(&mut self, reads: SystemParamItem<'_, '_, Self::Reads>, cx: &mut Prepare<'_, Self>);
    fn view(&mut self, _view: &engine_core::ViewFacts, _cx: &mut Prepare<'_, Self>) {}
    fn encode(&mut self, _pass: Pass<Self>, _cx: &mut Encode<'_, Self>) -> u32 {
        0
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Scene(&'static str);

impl Scene {
    pub const fn of<N: PassName>() -> Self {
        Self(N::NAME)
    }

    pub const fn named(name: &'static str) -> Self {
        Self(name)
    }

    pub const fn name(self) -> &'static str {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SceneDraw {
    pub(crate) pass: Scene,
    pub(crate) bit: Option<PassBit>,
}

pub trait Scope<P> {
    type Tag: 'static;
    #[doc(hidden)]
    fn pass_index(&self) -> Option<u32> {
        None
    }
    #[doc(hidden)]
    fn scene_pass(&self) -> Option<&'static str> {
        None
    }
}

impl<P> Scope<P> for Pass<P> {
    type Tag = Own;
    fn pass_index(&self) -> Option<u32> {
        Some(self.0)
    }
}

impl<P> Scope<P> for Scene {
    type Tag = Scene;
    fn scene_pass(&self) -> Option<&'static str> {
        Some(self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifetime {
    Frame,
    Kept,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Picture {
    Shown,
    Hidden,
}

#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Allocation {
    Kept,
    Fresh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Load,
    Clear,
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
    Target(TargetName),
    Asset,
}

pub enum Copied<P> {
    Buffer(Buffer<P>),
    Texture(Texture<P>),
    Target(TargetName),
    Canvas,
}

#[derive(Clone, Copy, Debug)]
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
    LevelToBuffer {
        from: (u32, wgpu::Origin3d),
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
        material: Option<MaterialHandle>,
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
        material: Option<MaterialHandle>,
    },
    IndirectAt {
        vertices: &'a [(DrawInput<P, S>, u64)],
        indices: Option<DrawInput<P, S>>,
        args: DrawInput<P, S>,
        offset: u64,
    },
    MeshIndirect {
        mesh: GeometryHandle,
        material: Option<MaterialHandle>,
        args: DrawInput<P, S>,
        offset: u64,
        streams: &'a [(DrawInput<P, S>, u64)],
    },
    Mesh {
        mesh: GeometryHandle,
        material: Option<MaterialHandle>,
        instances: Range<u32>,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum DepthUse {
    None,
    Compare {
        compare: wgpu::CompareFunction,
        write: bool,
        bias: wgpu::DepthBiasState,
    },
}

#[derive(Clone, Debug)]
pub enum VertexLayout {
    Geometry(engine_core::GeometryStream),
    Buffer(wgpu::VertexBufferLayout<'static>),
}

pub struct RenderPipeline<'a> {
    pub vertex: Entry,
    pub fragment: Option<Entry>,
    pub vertices: &'a [VertexLayout],
    pub primitive: wgpu::PrimitiveState,
    pub depth: DepthUse,
    pub stencil: wgpu::StencilState,
    pub samples: u32,
    pub targets: &'a [(Option<wgpu::BlendState>, wgpu::ColorWrites)],
    pub constants: &'a [(&'static str, f64)],
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
        Bind::Target(name) => install::Bound::Target(*name),
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
        None => install::Scoped::Scene(scope.scene_pass().unwrap_or_default()),
    }
}

fn copied<P>(copied: Copied<P>) -> install::Copied {
    match copied {
        Copied::Buffer(buffer) => install::Copied::Buffer(buffer.0),
        Copied::Texture(texture) => install::Copied::Texture(texture.0),
        Copied::Target(name) => install::Copied::Target(name),
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
        Texture(
            self.core.texture(label, format, (lifetime, false)),
            PhantomData,
        )
    }

    pub fn buffer(&mut self, label: &'static str, lifetime: Lifetime) -> Buffer<P> {
        Buffer(self.core.buffer(label, (lifetime, false)), PhantomData)
    }

    pub fn view_texture(
        &mut self,
        label: &'static str,
        format: wgpu::TextureFormat,
        lifetime: Lifetime,
    ) -> Texture<P> {
        Texture(
            self.core.texture(label, format, (lifetime, true)),
            PhantomData,
        )
    }

    pub fn view_buffer(&mut self, label: &'static str, lifetime: Lifetime) -> Buffer<P> {
        Buffer(self.core.buffer(label, (lifetime, true)), PhantomData)
    }

    pub fn sampler(&mut self, desc: engine_core::Sampler) -> Sampler<P> {
        Sampler(self.core.sampler(desc), PhantomData)
    }

    pub fn window(&mut self, buffer: Buffer<P>, size: u64) {
        self.core.window(buffer.0, size);
    }

    pub fn frame(&mut self, variable: Var, bound: Bind<P>) -> Result<(), Error> {
        self.core.frame(variable, bound_of(&bound))
    }

    pub fn shared_target(&mut self, target: SharedTarget) {
        let name = target.name.name();
        if self.core.declared.targets.iter().any(|known| known.name == target.name) {
            let path = format!("$.targets.{name}");
            self.core
                .problems
                .push(Error::new(engine_core::ErrorCode::NameTwice, path).got(name));
        } else {
            self.core.declared.targets.push(target);
        }
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
        let program = install::Program::Render(install::RenderDecl {
            vertex: desc.vertex,
            fragment: desc.fragment,
            vertices: desc.vertices.to_vec(),
            primitive: desc.primitive,
            depth: desc.depth,
            stencil: desc.stencil.clone(),
            targets: desc.targets.to_vec(),
            samples: desc.samples,
            faces: install::Faces::Declared,
        });
        let pipeline = self
            .core
            .pipeline(scoped(&scope), program, bound_list(bound))?;
        self.constants(pipeline, desc.constants);
        Ok(Pipeline(pipeline, PhantomData))
    }

    pub fn compute_pipeline(
        &mut self,
        pass: Pass<P>,
        entry: Entry,
        constants: &[(&'static str, f64)],
        bound: &[(Var, Bind<P>)],
    ) -> Result<ComputePipeline<P>, Error> {
        let pipeline = self.core.pipeline(
            install::Scoped::Pass(pass.0),
            install::Program::Compute(entry),
            bound_list(bound),
        )?;
        self.constants(pipeline, constants);
        Ok(ComputePipeline(pipeline, PhantomData))
    }

    fn constants(&mut self, pipeline: u32, constants: &[(&'static str, f64)]) {
        if let Some(decl) = self.core.declared.pipelines.get_mut(pipeline as usize) {
            decl.constants = constants.to_vec();
        }
    }

    pub fn draw_input<S: Scope<P>>(&mut self, scope: S, buffer: Buffer<P>) -> DrawInput<P, S::Tag> {
        DrawInput(self.core.draw_input(scoped(&scope), buffer.0), PhantomData)
    }

    pub fn args_input<S: Scope<P>>(&mut self, scope: S, buffer: Buffer<P>) -> DrawInput<P, S::Tag> {
        DrawInput(self.core.args_input(scoped(&scope), buffer.0), PhantomData)
    }

    pub fn scene<N: PassName>(&mut self) -> PassBuilder<'_, P> {
        let pass = self.core.pass(N::NAME, install::Work::Render);
        self.core.draws_scene(pass, N::NAME, N::PASS);
        PassBuilder {
            core: &mut *self.core,
            pass,
            part: PhantomData,
        }
    }

    pub fn screen<N: PassName>(&mut self) -> PassBuilder<'_, P> {
        if N::PASS.scene() {
            let path = format!("$.passes.{}", N::NAME);
            self.core
                .problems
                .push(Error::new(engine_core::ErrorCode::PassMismatch, path));
        }
        let pass = self.core.pass(N::NAME, install::Work::Render);
        if let Some(declared) = self.core.pass_mut(pass) {
            declared.views = N::PASS.views;
        }
        PassBuilder {
            core: &mut *self.core,
            pass,
            part: PhantomData,
        }
    }

    pub fn passes(&self) -> &engine_core::ScenePasses {
        &self.core.passes
    }
}

pub struct PassBuilder<'b, P> {
    core: &'b mut install::InstallCore,
    pass: u32,
    part: PhantomData<fn() -> P>,
}

impl<P: GpuPart> PassBuilder<'_, P> {
    #[must_use]
    pub fn views(self, views: engine_core::PassViews) -> Self {
        if let Some(pass) = self.core.pass_mut(self.pass) {
            pass.views = views;
        }
        self
    }

    #[must_use]
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

    #[must_use]
    pub fn makes_records(self) -> Self {
        if let Some(declared) = self.core.pass_mut(self.pass) {
            declared.makes_records = true;
        }
        self
    }

    #[must_use]
    pub fn color(self, target: Texture<P>, access: Access) -> Self {
        if let Some(pass) = self.core.pass_mut(self.pass) {
            pass.colors.push((target.0, access));
        }
        self
    }

    #[must_use]
    pub fn target(self, name: TargetName, access: Access) -> Self {
        if let Some(pass) = self.core.pass_mut(self.pass) {
            pass.targets.push((name, access));
        }
        self
    }

    #[must_use]
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

    #[must_use]
    pub fn depth(self, target: Texture<P>, access: Access) -> Self {
        if let Some(pass) = self.core.pass_mut(self.pass) {
            pass.depths.push((target.0, access));
        }
        self
    }

    #[must_use]
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
    part: PhantomData<fn() -> P>,
}

impl<P: GpuPart> Prepare<'_, P> {
    pub fn run(&mut self, pass: Pass<P>) {
        self.core.run_pass(pass.0);
    }

    pub fn skip_kind(&mut self, kind: engine_core::DrawKind, reason: &'static str) {
        self.core.skip_kind(kind, reason);
    }

    pub fn ran(&self, pass: Pass<P>) -> bool {
        self.core.set().ran.contains(pass.0 as usize)
    }

    pub fn warming(&self) -> u32 {
        self.core.pipelines.warming.left()
    }

    pub fn features(&self) -> wgpu::Features {
        self.core.installs.features
    }

    pub fn view_index(&self) -> u32 {
        self.core.uniforms.camera_slot
    }

    pub fn drawable(&self, mesh: GeometryHandle, material: Option<MaterialHandle>) -> bool {
        self.core.assets.meshes.get(mesh).is_some()
            && self.core.assets.materials.group(material).is_some()
    }

    pub fn mipped(&self, material: Option<MaterialHandle>) -> bool {
        self.core.assets.materials.mipped(material)
    }

    pub fn allocate(&mut self, texture: Texture<P>, size: wgpu::Extent3d, mips: u32) -> Allocation {
        self.core.allocate(texture.0, size, mips)
    }

    pub fn allocate_as(
        &mut self,
        texture: Texture<P>,
        format: wgpu::TextureFormat,
        size: wgpu::Extent3d,
        mips: u32,
    ) -> Allocation {
        self.core.allocate_as(texture.0, format, size, mips)
    }

    pub fn grow(&mut self, texture: Texture<P>, size: wgpu::Extent3d, mips: u32) -> Allocation {
        self.core.grow(texture.0, size, mips)
    }

    pub fn release(&mut self, texture: Texture<P>) {
        self.core.release(texture.0);
    }

    pub fn allocate_buffer(&mut self, buffer: Buffer<P>, bytes: u64) -> Allocation {
        self.core.allocate_buffer(buffer.0, bytes)
    }

    pub fn release_buffer(&mut self, buffer: Buffer<P>) {
        self.core.release_buffer(buffer.0);
    }

    pub fn renew_buffer(&mut self, buffer: Buffer<P>, bytes: u64) -> Allocation {
        self.core.renew_buffer(buffer.0, bytes)
    }

    pub fn clear_buffer_range(&mut self, buffer: Buffer<P>, range: Range<u64>) {
        self.core.clear_buffer_range(buffer.0, range);
    }

    pub fn write<T: bytemuck::Pod>(&mut self, buffer: Buffer<P>, offset: u64, data: &[T]) {
        self.core
            .write(buffer.0, offset, bytemuck::cast_slice(data));
    }

    pub fn grow_buffer(&mut self, buffer: Buffer<P>, bytes: u64) -> Allocation {
        self.core.grow_buffer(buffer.0, bytes)
    }

    pub fn count_upload(&mut self, bytes: u64) {
        self.core.tallies.uploads.counted += bytes;
    }

    pub fn memory(&mut self, category: &'static str, bytes: u64) {
        let memory = &mut self.core.tallies.uploads.memory;
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

    pub fn mipmaps(&mut self, texture: Texture<P>, layers: Range<u32>) -> bool {
        self.core.mipmaps(texture.0, layers)
    }

    pub fn copy(&mut self, pass: Pass<P>, regions: &[CopyRegion]) {
        let regions = regions.to_vec();
        let copies = &mut self.core.queued.copies;
        copies.retain(|(known, _)| *known != pass.0);
        copies.push((pass.0, regions));
        self.core.run_pass(pass.0);
    }

    pub fn copy_at_end(&mut self, from: Copied<P>, to: Copied<P>, regions: &[CopyRegion]) {
        let (from, to) = (copied(from), copied(to));
        let unknown = [from, to].into_iter().find_map(|copied| match copied {
            install::Copied::Target(name) => (name != TargetName::CANVAS
                && self.core.machine.declared.shared_target(name).is_none())
            .then_some(name.name()),
            install::Copied::Buffer(_) | install::Copied::Texture(_) | install::Copied::Canvas => {
                None
            }
        });
        match unknown {
            Some(name) => {
                let path = format!("$.targets.{name}");
                let error = Error::new(engine_core::ErrorCode::UnknownVariable, path).got(name);
                self.core.refuse(error);
            }
            None => self
                .core
                .queued
                .ending
                .push((from, to, regions.to_vec())),
        }
    }

    pub fn fill<S>(
        &mut self,
        pipeline: &Pipeline<P, S>,
        textures: &[TextureHandle],
    ) -> Pipeline<P, S> {
        Pipeline(self.core.fill(pipeline.0, textures), PhantomData)
    }

    pub fn read_back<R: Readable>(
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
            deliver: readback::delivered::<R>,
        })
    }

    pub fn read_back_once<R: Readable>(
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
            deliver: readback::delivered::<R>,
        })
    }

    pub fn deliver<R: Readable>(&mut self, tag: u32, bytes: Vec<u8>) {
        let (part, written) = (self.core.part, self.core.written());
        self.core.arrived(
            (part, tag),
            Readback {
                value: Some(bytes),
                written,
            },
            readback::delivered::<R>,
        );
    }

    pub fn refuse(&mut self, error: Error) {
        self.core.refuse(error);
    }

    pub fn canvas(&self) -> Option<wgpu::TextureFormat> {
        self.core
            .canvas
            .texture
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
    pub fn view_index(&self) -> u32 {
        self.core.uniforms.camera_slot
    }

    pub fn recorded(&mut self) -> Recorded {
        std::mem::take(&mut self.core.tallies.worst)
    }

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

    pub fn render(&mut self, attach: Attach<P>) -> Draw<'_, P, Own> {
        self.core.open_encode_span(self.timer);
        if !self.core.begin_render(raw_attach(&attach)) {
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Recorded {
    #[default]
    Exact,
    Refused,
    StandIn,
    Waiting,
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
    pub fn draw(&mut self, pipeline: &Pipeline<P, S>, shape: Shape<'_, P, S>) -> Recorded {
        self.issue(pipeline, shape, None)
    }

    pub fn speedup(&mut self, pipeline: Pipeline<P, S>) -> Option<Pipeline<P, S>> {
        let part = self.raw.part();
        let run = self.raw.run();
        let point = run.point;
        run.speedup_made(part, point, pipeline.0).then_some(pipeline)
    }

    pub fn draw_timed<'l>(
        &mut self,
        pipeline: &Pipeline<P, S>,
        shape: Shape<'_, P, S>,
        label: impl FnOnce() -> String + 'l,
    ) -> Recorded {
        let timer = self.raw.timer();
        let mut label = Some(label);
        let mut named = || label.take().map_or_else(String::new, |label| label());
        self.issue(pipeline, shape, Some((timer, &mut named)))
    }

    fn issue(
        &mut self,
        pipeline: &Pipeline<P, S>,
        shape: Shape<'_, P, S>,
        timed: encode::TimedDraw<'_>,
    ) -> Recorded {
        let shape = match shape {
            Shape::Pulled {
                vertices,
                instances,
                material,
            } => encode::RawShape::Pulled {
                vertices,
                instances,
                material,
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
                material,
            } => encode::RawShape::Indirect {
                vertices: inputs(vertices),
                indices: indices.map(|input| input.0),
                args: args.0,
                offset,
                material,
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
            Shape::Mesh {
                mesh,
                material,
                instances,
            } => encode::RawShape::Mesh {
                mesh,
                material,
                instances,
            },
        };
        let part = self.raw.part();
        let run = self.raw.run();
        let point = run.point;
        run.draw(part, point, pipeline.0, shape, timed)
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

fn raw_attach<P>(attach: &Attach<P>) -> encode::AttachRaw {
    match attach {
        Attach::Declared => encode::AttachRaw::Declared,
        Attach::Depth(texture, layer) => encode::AttachRaw::Depth(texture.0, *layer),
        Attach::Color(texture, layer) => {
            let layer = layer.unwrap_or(0);
            encode::AttachRaw::Color(texture.0, 0, layer..layer + 1)
        }
        Attach::ColorMip(texture, mip, layer) => {
            encode::AttachRaw::Color(texture.0, *mip, *layer..*layer + 1)
        }
        Attach::ColorDepth(color, layer, depth) => {
            encode::AttachRaw::ColorDepth(color.0, *layer, depth.0)
        }
        Attach::Layers {
            texture,
            mip,
            layers,
        } => encode::AttachRaw::Color(texture.0, *mip, layers.clone()),
        Attach::Colors(colors) => encode::AttachRaw::Colors(
            colors
                .iter()
                .map(|(texture, layer)| (texture.0, *layer))
                .collect(),
        ),
    }
}

impl<P: GpuPart> Draw<'_, P, Own> {
    pub fn scene(&mut self, own: &P) -> u32 {
        let run = self.raw.run();
        run.reset_scissor();
        let Some(scene) = run.scene_draw() else {
            return 0;
        };
        let members = run.scene_members();
        run.tallies.kinds.scened(scene.pass.0);
        let encoding = self.raw.part();
        let mut drawn = 0;
        for member in members.iter() {
            let batches = (&member.batches, scene.bit);
            let run = self.raw.run();
            if member.batches.drawn(scene.bit).next().is_none() {
                run.tallies.kinds.begin(member.kind.kind, scene.pass.0);
                run.tallies.kinds.skip("no object of this kind is drawn here");
                run.tallies.kinds.end();
                continue;
            }
            let held = if member.part == encoding {
                None
            } else {
                let Some(held) = run.take_member(member.part) else {
                    continue;
                };
                Some(held)
            };
            run.point = Some(scene.pass.0);
            run.tallies.kinds.begin(member.kind.kind, scene.pass.0);
            let before = reached(run);
            let part = held.as_ref().map_or(own as &dyn std::any::Any, |held| held.part());
            (member.call)(part, &mut self.raw.member(member.part, batches));
            let run = self.raw.run();
            drawn += reached(run) - before;
            run.tallies.kinds.end();
            run.point = None;
            if let Some(held) = held {
                run.give_member(member.part, held);
            }
        }
        drawn
    }
}

fn reached(run: &encode::EncodeCore) -> u32 {
    let kinds = &run.tallies.kinds;
    kinds
        .drawing
        .and_then(|at| kinds.listed.get(at))
        .map_or(0, |tally| tally.reached)
}

gpu_struct! {
    pub struct CameraUniform {
        pub view_projection: [[f32; 4]; 4],
        pub position: [f32; 4],
        pub mip_bias: f32,
        pub pad_0: u32,
        pub pad_1: u32,
        pub pad_2: u32,
    }
}
