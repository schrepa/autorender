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
mod linker;
mod machine;
mod pipelines;
mod registration;
mod render_bricks;
mod shaders;
mod shared;
mod uploads;

mod mesh_arena;
mod mipmaps;
mod platform;
mod profiling;
mod readback;
mod renderer;
mod targets;
mod textures;
mod variant;
#[cfg(any(feature = "dev", feature = "profiling"))]
mod whatif;

pub use gpu_data::mip_levels;
#[cfg(any(feature = "dev", feature = "profiling"))]
pub use profiling::INSTRUMENTS as GPU_INSTRUMENTS;
#[doc(hidden)]
pub use platform::{AdapterInfo, RenderTarget};
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use platform::{GpuPreference, NativeWindow, keep_pipeline_cache_in};
pub use registration::{Entries, Made};
pub use render_bricks::RenderBricks;
#[doc(hidden)]
#[cfg(any(feature = "dev", feature = "profiling"))]
pub use gpu_calls::{UnchangedWrites, unchanged_writes};
pub use renderer::{
    Ahead, DeviceHealth, Installation, Renderer, StartingRenderer, device_limits, install_noop,
};
#[cfg(any(feature = "dev", feature = "profiling"))]
pub use renderer::gpu_names;
pub use shaders::FULLSCREEN;
pub use targets::{SharedTarget, TargetClear};
pub use textures::gpu_format as texture_format;
pub use uploads::{UploadRange, listed_ranges};
pub use variant::{PerBatch, Variant};

pub const DEV: bool = cfg!(feature = "dev");

engine_core::settings! {
    #[section("render")]
    pub struct RenderSettings {
        pub readback_slots: u32 = 3 => at_least(1),
        pub readback_slot_bytes: Option<u64> = None => at_least(4),
        pub timer_slots: u32 = 3 => at_least(1),
        pub timer_spans: u32 = 128 => at_least(1),
        pub timer_draws: u32 = 2048 => at_least(1),
    }
}

pub trait GpuPart: Sized + Send + Sync + 'static {
    type Reads: Read + 'static;
    const NAME: &'static str = "";
    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error>;
    fn prepare(&mut self, reads: SystemParamItem<'_, '_, Self::Reads>, cx: &mut Prepare<'_, Self>);
    fn view(&mut self, _view: &engine_core::ViewFacts, _cx: &mut Prepare<'_, Self>) {}
    fn encode(&mut self, _pass: Pass<Self>, _cx: &mut Encode<'_, Self>) {}
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
handle!(BatchGroup<P>);

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

impl<P> Clone for Bind<P> {
    fn clone(&self) -> Self {
        match self {
            Self::Texture(texture) => Self::Texture(*texture),
            Self::Slice {
                texture,
                mips,
                layers,
            } => Self::Slice {
                texture: *texture,
                mips: mips.clone(),
                layers: layers.clone(),
            },
            Self::Buffer(buffer) => Self::Buffer(*buffer),
            Self::Sampler(sampler) => Self::Sampler(*sampler),
            Self::Target(name) => Self::Target(*name),
            Self::Asset => Self::Asset,
        }
    }
}

pub enum GroupEntry<P> {
    Range {
        buffer: Buffer<P>,
        offset: u64,
        size: u64,
    },
    Texture(Texture<P>),
    Asset(TextureHandle),
    AssetSampler(engine_core::Sampler),
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
        key: Option<MaterialHandle>,
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
        key: Option<MaterialHandle>,
    },
    IndirectAt {
        vertices: &'a [(DrawInput<P, S>, u64)],
        indices: Option<DrawInput<P, S>>,
        args: DrawInput<P, S>,
        offset: u64,
    },
    MeshIndirect {
        mesh: GeometryHandle,
        key: Option<MaterialHandle>,
        args: DrawInput<P, S>,
        offset: u64,
        streams: &'a [(DrawInput<P, S>, u64)],
    },
    Mesh {
        mesh: GeometryHandle,
        key: Option<MaterialHandle>,
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
    pub per_batch: PerBatch,
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

fn scoped<P, S: Scope<P>>(core: &mut install::InstallCore, scope: &S) -> install::Scoped {
    if let Some(pass) = scope.pass_index() {
        return install::Scoped::Pass(pass);
    }
    let name = scope.scene_pass().unwrap_or_default();
    match core.passes.bit(name) {
        Some(bit) => install::Scoped::Scene(bit),
        None => {
            let path = core.path("scene");
            core.problems
                .push(Error::new(engine_core::ErrorCode::PassMismatch, path).got(name));
            install::Scoped::Outside(name)
        }
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
            per_batch: desc.per_batch,
        });
        let scope = scoped(self.core, &scope);
        let pipeline = self.core.pipeline(scope, program, bound_list(bound))?;
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
        let scope = scoped(self.core, &scope);
        DrawInput(self.core.draw_input(scope, buffer.0), PhantomData)
    }

    pub fn args_input<S: Scope<P>>(&mut self, scope: S, buffer: Buffer<P>) -> DrawInput<P, S::Tag> {
        let scope = scoped(self.core, &scope);
        DrawInput(self.core.args_input(scope, buffer.0), PhantomData)
    }

    pub fn batch_group(
        &mut self,
        module: engine_core::Module,
        index: u32,
        buffers: &[(Var, Buffer<P>)],
    ) -> Result<BatchGroup<P>, Error> {
        let core = &mut *self.core;
        core.linker.add_source(module.source());
        let (path, _) = linker::path_of((module, ""));
        let at = format!("$.batch_groups.{path}.{index}");
        if index == 0 {
            return Err(Error::new(engine_core::ErrorCode::UnknownVariable, at).got(index));
        }
        let groups = &core.declared.batch_groups;
        if groups
            .iter()
            .any(|known| known.module == module && known.index == index)
        {
            return Err(Error::new(engine_core::ErrorCode::NameTwice, at));
        }
        for (var, buffer) in buffers {
            let declared = core.linker.declared(*var);
            if var.module() != module || declared.is_none_or(|found| found.group != index) {
                return Err(Error::new(engine_core::ErrorCode::UnknownVariable, at).got(var.name()));
            }
            let usage = core.linker.buffer_usage(*var);
            core.use_buffer(buffer.0, usage);
        }
        core.declared.batch_groups.push(install::BatchDecl { module, index });
        Ok(BatchGroup(
            (core.declared.batch_groups.len() - 1) as u32,
            PhantomData,
        ))
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

    pub fn set_batch(
        &mut self,
        group: BatchGroup<P>,
        key: Option<MaterialHandle>,
        entries: &[(Var, GroupEntry<P>)],
        variant: Option<Variant>,
    ) -> Result<(), Error> {
        let raw: Vec<(Var, encode::BatchEntryRaw)> = entries
            .iter()
            .map(|(var, entry)| {
                let raw = match entry {
                    GroupEntry::Range {
                        buffer,
                        offset,
                        size,
                    } => encode::BatchEntryRaw::Range {
                        buffer: buffer.0,
                        offset: *offset,
                        size: *size,
                    },
                    GroupEntry::Texture(texture) => encode::BatchEntryRaw::Texture(texture.0),
                    GroupEntry::Asset(texture) => encode::BatchEntryRaw::Asset(*texture),
                    GroupEntry::AssetSampler(sampler) => {
                        encode::BatchEntryRaw::AssetSampler(*sampler)
                    }
                };
                (*var, raw)
            })
            .collect();
        self.core.set_batch(group.0, key, &raw, variant)
    }

    pub fn clear_batch(&mut self, group: BatchGroup<P>, key: Option<MaterialHandle>) {
        self.core.clear_batch(group.0, key);
    }

    pub fn texture_resident(&self, texture: TextureHandle) -> bool {
        self.core.assets.textures.is_texture_resident(texture)
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

    pub fn report(&mut self, error: Error) {
        self.core.fault(error);
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

    pub fn dispatch(&mut self, pipeline: &ComputePipeline<P>, count: Count<P>) {
        self.dispatch_at(pipeline, count, &[]);
    }

    pub fn dispatch_at(
        &mut self,
        pipeline: &ComputePipeline<P>,
        count: Count<P>,
        offsets: &[u32],
    ) {
        let count = match count {
            Count::Groups(groups) => encode::CountRaw::Groups(groups),
            Count::Indirect(input, offset) => encode::CountRaw::Indirect(input.0, offset),
        };
        self.core.open_encode_span(self.timer);
        let part = self.core.part;
        self.core.dispatch(part, pipeline.0, count, offsets);
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

fn inputs<P, S>(held: &mut Vec<(u32, u64)>, inputs: &[DrawInput<P, S>]) -> encode::Streams {
    encode::Streams::of(held, inputs.iter().map(|input| (input.0, 0)))
}

fn streams<P, S>(
    held: &mut Vec<(u32, u64)>,
    streams: &[(DrawInput<P, S>, u64)],
) -> encode::Streams {
    encode::Streams::of(held, streams.iter().map(|(input, offset)| (input.0, *offset)))
}

impl<P: GpuPart, S: 'static> Draw<'_, P, S> {
    pub fn draw(&mut self, pipeline: &Pipeline<P, S>, shape: Shape<'_, P, S>) -> Recorded {
        self.issue(pipeline, shape)
    }

    pub fn speedup(&mut self, pipeline: Pipeline<P, S>) -> Option<Pipeline<P, S>> {
        let part = self.raw.part();
        let run = self.raw.run();
        let point = run.point;
        run.speedup_made(part, point, pipeline.0).then_some(pipeline)
    }

    fn issue(&mut self, pipeline: &Pipeline<P, S>, shape: Shape<'_, P, S>) -> Recorded {
        let (part, timer) = (self.raw.part(), self.raw.timer());
        let run = self.raw.run();
        let held = &mut run.spare.streams;
        held.clear();
        let shape = match shape {
            Shape::Pulled {
                vertices,
                instances,
                key,
            } => encode::RawShape::Pulled {
                vertices,
                instances,
                key,
            },
            Shape::Buffers {
                vertices,
                indices,
                elements,
                instances,
            } => encode::RawShape::Buffers {
                vertices: inputs(held, vertices),
                indices: indices.map(|input| input.0),
                elements,
                instances,
            },
            Shape::Indirect {
                vertices,
                indices,
                args,
                offset,
                key,
            } => encode::RawShape::Indirect {
                vertices: inputs(held, vertices),
                indices: indices.map(|input| input.0),
                args: args.0,
                offset,
                key,
            },
            Shape::IndirectAt {
                vertices,
                indices,
                args,
                offset,
            } => encode::RawShape::IndirectAt {
                vertices: streams(held, vertices),
                indices: indices.map(|input| input.0),
                args: args.0,
                offset,
            },
            Shape::MeshIndirect {
                mesh,
                key,
                args,
                offset,
                streams: given,
            } => encode::RawShape::MeshIndirect {
                mesh,
                key,
                args: args.0,
                offset,
                streams: streams(held, given),
            },
            Shape::Mesh {
                mesh,
                key,
                instances,
            } => encode::RawShape::Mesh {
                mesh,
                key,
                instances,
            },
        };
        let point = run.point;
        run.draw(part, point, pipeline.0, shape, timer)
    }
}

impl<P: GpuPart> Draw<'_, P, Scene> {
    pub fn batches(&mut self, pipeline: &Pipeline<P, Scene>, vertices: Range<u32>) {
        for batch in self.raw.batches() {
            let shape = match batch.geometry {
                Some(mesh) => Shape::Mesh {
                    mesh,
                    key: batch.material,
                    instances: batch.instances.clone(),
                },
                None => Shape::Pulled {
                    vertices: vertices.clone(),
                    instances: batch.instances.clone(),
                    key: batch.material,
                },
            };
            let recorded = self.draw(pipeline, shape);
            #[cfg(any(feature = "dev", feature = "profiling"))]
            if engine_profile::tracing()
                && matches!(recorded, Recorded::Exact | Recorded::StandIn)
                && !batch.drawn.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                self.raw.run().first_drawn(pipeline.0, batch.cause);
            }
            #[cfg(not(any(feature = "dev", feature = "profiling")))]
            let _ = recorded;
        }
    }

    pub fn pass(&mut self) -> Option<PassBit> {
        self.raw.run().point
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
    pub fn scene(&mut self, own: &P) {
        let run = self.raw.run();
        run.reset_scissor();
        let Some(bit) = run.scene_bit() else {
            return;
        };
        let scene = bit.index();
        run.tallies.kinds.scened(scene);
        let Some(listed) = run.installs.members.get_mut(scene as usize).map(std::mem::take) else {
            return;
        };
        let entries = std::mem::take(&mut run.installs.draw_entries);
        run.point = Some(bit);
        let encoding = self.raw.part();
        for &at in &listed {
            let Some(member) = entries.get(at as usize) else {
                continue;
            };
            let run = self.raw.run();
            if member.batches.drawn(Some(bit)).next().is_none() {
                run.tallies.kinds.begin(scene, member.kind);
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
            run.tallies.kinds.begin(scene, member.kind);
            let part = held.as_ref().map_or(own as &dyn std::any::Any, |held| held.part());
            let batches = (&member.batches, Some(bit));
            (member.call)(part, &mut self.raw.member(member.part, batches));
            let run = self.raw.run();
            run.tallies.kinds.end();
            if let Some(held) = held {
                run.give_member(member.part, held);
            }
        }
        let run = self.raw.run();
        run.point = None;
        run.installs.draw_entries = entries;
        if let Some(slot) = run.installs.members.get_mut(scene as usize) {
            *slot = listed;
        }
    }
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
