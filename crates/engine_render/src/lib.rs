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
    Bricks, DrawKind, Entry, Error, List, MaterialHandle, MeshHandle, PassBit, PassEntry, PassName,
    Read, Readable, Readback, Slot, SlotSet, TextureHandle, Var, Writes, gpu_struct,
};
use engine_registry::RecordExt;

use crate::draws::RawDraw;

pub use engine_core::{AttachPoint, TargetName};
pub use wgpu;

mod bindings;
mod draws;
mod drawn;
mod encode;
mod frame_group;
mod gpu_calls;
mod gpu_data;
mod gpu_layouts;
mod graph;
mod install;
mod linker;
mod machine;
mod pieces;
mod pipelines;
mod places;
mod pyramid;
mod registration;
mod shaders;
mod shared;
mod sizes;
pub mod slots;
mod surface;

mod materials;
mod memory;
mod mesh_arena;
mod mipmaps;
mod platform;
mod profiling;
mod readback;
mod renderer;
mod targets;

pub use gpu_data::{
    jitter_clip, mip_levels, pack_color, pack_surface, pixel_angle, pixels_per_meter,
    planes, pyramid_extents,
};
pub use machine::{DEPTH_FORMAT, SCENE_FORMAT};
pub use materials::gpu_format as texture_format;
pub use materials::MATERIAL_ANISOTROPY;
pub use surface::{
    DrawState, LightingModule, MaterialVariant, SlotModule, SurfaceKind, material_state,
};
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
    DeviceHealth, GpuPassTimes, Installation, InstalledPart, InstalledPiece, Renderer,
    StartingRenderer, device_limits, install_noop,
};
pub use shaders::FULLSCREEN;
#[doc(hidden)]
pub use shaders::core_modules;

pub const DEV: bool = cfg!(feature = "dev");

pub fn level_region(view: &engine_core::ViewFacts, level: u32) -> (u32, u32) {
    let (display, render) = (view.size(), view.render);
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
    use engine_core::gpu_bits;

    #[gpu_bits]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DebugRecord {
        #[bits(8)]
        pub kind: u32,
        #[bits(4)]
        pub level: u32,
        pub fading: bool,
        pub card: bool,
        #[bits(8, at = 16)]
        pub source: u32,
    }

    pub const DEBUG_NO_MATERIAL: u32 = 0xff;
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
    const TURNS_ON_IN_PLAY: bool = false;
    const LABELS: &'static [&'static str] = &[];
    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error>;
    fn prepare(&mut self, reads: SystemParamItem<'_, '_, Self::Reads>, cx: &mut Prepare<'_, Self>);
    fn view(&mut self, _view: &engine_core::ViewFacts, _cx: &mut Prepare<'_, Self>) {}
    fn encode(&mut self, _pass: Pass<Self>, _cx: &mut Encode<'_, Self>) -> u32 {
        0
    }
}

pub trait RenderBricks {
    fn part<P: GpuPart>(&mut self) -> &mut Self;
    fn pass<N: PassName>(&mut self) -> &mut Self;
    fn applies<N: PassName>(&mut self, module: engine_core::Module) -> &mut Self;
    fn draws<P: GpuPart>(
        &mut self,
        kind: DrawKind,
        draw: fn(&P, &SceneDraw, &mut Draw<'_, P, Scene>) -> u32,
    ) -> &mut Self;
    fn view_target(&mut self, target: TargetDecl) -> &mut Self;
    fn generated(&mut self, wgsl: &'static [&'static str]) -> &mut Self;
    fn material_slot(&mut self, module: SlotModule) -> &mut Self;
    fn lighting_model(&mut self, module: LightingModule) -> &mut Self;
    fn surface_kind(&mut self, kind: SurfaceKind) -> &mut Self;
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

    fn pass<N: PassName>(&mut self) -> &mut Self {
        let record = self.record();
        if !record
            .render_entries::<PassEntry>()
            .any(|entry| entry.name == N::NAME)
        {
            record.render.push(Box::new(PassEntry::of::<N>()));
        }
        self
    }

    fn applies<N: PassName>(&mut self, module: engine_core::Module) -> &mut Self {
        self.pass::<N>();
        let record = self.record();
        let scope = record.scope();
        record.render.push(Box::new(registration::Applied {
            pass: PassEntry::of::<N>(),
            module,
            scope,
        }));
        self
    }

    fn draws<P: GpuPart>(
        &mut self,
        kind: DrawKind,
        draw: fn(&P, &SceneDraw, &mut Draw<'_, P, Scene>) -> u32,
    ) -> &mut Self {
        let record = self.record();
        let scope = record.scope();
        record
            .render
            .push(Box::new(registration::PartRecord::of::<P>(scope)));
        record
            .render
            .push(Box::new(registration::KindDraw::of::<P>(kind, draw)));
        self
    }

    fn view_target(&mut self, target: TargetDecl) -> &mut Self {
        self.record().render.push(Box::new(target));
        self
    }

    fn generated(&mut self, wgsl: &'static [&'static str]) -> &mut Self {
        self.record()
            .render
            .push(Box::new(registration::Generated(wgsl)));
        self
    }

    fn material_slot(&mut self, module: SlotModule) -> &mut Self {
        if module.slot == engine_core::SurfaceSlot::Position {
            self.record()
                .render
                .push(Box::new(engine_core::PositionReach {
                    module: module.name,
                    reach: module.reach,
                }));
        }
        self.record().render.push(Box::new(module));
        self
    }

    fn lighting_model(&mut self, module: LightingModule) -> &mut Self {
        self.record().render.push(Box::new(module));
        self
    }

    fn surface_kind(&mut self, kind: SurfaceKind) -> &mut Self {
        self.record().render.push(Box::new(kind));
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
pub struct SceneDraw {
    pub pass: Scene,
    pub bit: Option<PassBit>,
    pub at: AttachPoint,
    pub reads: SlotSet,
    pub writes: Writes,
    pub view: Option<u32>,
    pub index: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct TargetDecl {
    pub name: TargetName,
    pub format: wgpu::TextureFormat,
    pub written: &'static [AttachPoint],
    pub module: Option<engine_core::Module>,
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
    Previous,
}

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
    Mesh {
        mesh: MeshHandle,
        material: Option<MaterialHandle>,
        instances: Range<u32>,
    },
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

pub const DEBUG_TARGETS: [TargetName; 3] = [
    TargetName("debug_ids"),
    TargetName("debug_count"),
    TargetName("debug_words"),
];

fn target_of(name: TargetName) -> Result<install::ViewTarget, Error> {
    install::ViewTarget::of(name).ok_or_else(|| {
        Error::new(
            engine_core::ErrorCode::UnknownVariable,
            format!("$.targets.{}", name.name()),
        )
        .got(name.name())
    })
}

fn bound_of<P>(bound: &Bind<P>) -> Result<install::Bound, Error> {
    Ok(match bound {
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
        Bind::Target(name) => install::Bound::Target(target_of(*name)?),
        Bind::Asset => install::Bound::Asset,
    })
}

fn bound_list<P>(bound: &[(Var, Bind<P>)]) -> Result<Vec<(Var, install::Bound)>, Error> {
    bound
        .iter()
        .map(|(var, bound)| Ok((*var, bound_of(bound)?)))
        .collect()
}

fn scoped<P, S: Scope<P>>(scope: &S) -> install::Scoped {
    match scope.pass_index() {
        Some(pass) => install::Scoped::Pass(pass),
        None => install::Scoped::Scene(scope.scene_pass().unwrap_or_default()),
    }
}

fn copied<P>(copied: Copied<P>) -> Result<install::Copied, Error> {
    Ok(match copied {
        Copied::Buffer(buffer) => install::Copied::Buffer(buffer.0),
        Copied::Texture(texture) => install::Copied::Texture(texture.0),
        Copied::Target(name) => install::Copied::Target(target_of(name)?),
        Copied::Canvas => install::Copied::Canvas,
    })
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
        Texture(self.core.texture(label, format, (lifetime, false)), PhantomData)
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
        Texture(self.core.texture(label, format, (lifetime, true)), PhantomData)
    }

    pub fn view_buffer(&mut self, label: &'static str, lifetime: Lifetime) -> Buffer<P> {
        Buffer(self.core.buffer(label, (lifetime, true)), PhantomData)
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
        self.core.frame(variable, bound_of(&bound)?)
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
        let work = match (copied(from), copied(to)) {
            (Ok(from), Ok(to)) => install::Work::Copy { from, to },
            (Err(error), _) | (_, Err(error)) => {
                self.core.problems.push(error);
                install::Work::Render
            }
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
            samples: 1,
        });
        let pipeline = self
            .core
            .pipeline(scoped(scope), program, bound_list(bound)?)?;
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
            bound_list(bound)?,
        )?;
        Ok(ComputePipeline(pipeline, PhantomData))
    }

    pub fn draw_input<S: Scope<P>>(&mut self, scope: S, buffer: Buffer<P>) -> DrawInput<P, S::Tag> {
        DrawInput(self.core.draw_input(scoped(&scope), buffer.0), PhantomData)
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
            declared.layer = Some(N::PASS.at);
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
    pub fn at(self, point: AttachPoint) -> Self {
        if let Some(pass) = self.core.pass_mut(self.pass) {
            pass.layer = Some(point);
        }
        self
    }

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
    pub fn color(self, target: Texture<P>, access: Access) -> Self {
        if let Some(pass) = self.core.pass_mut(self.pass) {
            pass.colors.push((target.0, access));
        }
        self
    }

    #[must_use]
    pub fn target(self, name: TargetName) -> Self {
        match install::ViewTarget::of(name).filter(|view| *view == install::ViewTarget::Scene) {
            Some(view) => {
                if let Some(pass) = self.core.pass_mut(self.pass) {
                    pass.targets.push(view);
                }
            }
            None => {
                let pass = self.core.pass_mut(self.pass).map_or("", |pass| pass.name);
                let path = format!("$.passes.{pass}.targets.{}", name.name());
                self.core
                    .problems
                    .push(Error::new(engine_core::ErrorCode::PassMismatch, path));
            }
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

fn delivered<R: Readable>(world: &mut bevy_ecs::world::World, tag: u32, read: Readback<&[u8]>) {
    world.get_resource_or_init::<R>().arrived(tag, read);
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
        CopyRegion::LevelToBuffer { from, to, size } => CopyRegion::LevelToBuffer {
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

    pub fn ran(&self, pass: Pass<P>) -> bool {
        self.core.set().ran.contains(pass.0 as usize)
    }

    pub fn view_index(&self) -> u32 {
        self.core.uniforms.camera_slot
    }

    pub fn drawable(&self, mesh: MeshHandle, material: Option<MaterialHandle>) -> bool {
        self.core.assets.meshes.get(mesh).is_some()
            && self.core.assets.materials.group(material).is_some()
    }

    pub fn read_target(&self, name: TargetName) -> bool {
        let machine = &self.core.machine;
        let Some(target) = machine
            .outputs
            .iter()
            .position(|output| output.name == name.name())
            .and_then(|index| machine.ids.target(machine::Target::Output(index as u32)))
        else {
            return false;
        };
        self.core
            .set()
            .ran
            .iter()
            .any(|pass| machine.graph.reads(pass).contains(target.index()))
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

    pub fn mipmaps(&mut self, texture: Texture<P>, layers: Range<u32>) {
        self.core.mipmaps(texture.0, layers);
    }

    pub fn copy(&mut self, pass: Pass<P>, regions: &[CopyRegion]) {
        let regions = regions.iter().map(region_of).collect();
        let copies = &mut self.core.queued.copies;
        copies.retain(|(known, _)| *known != pass.0);
        copies.push((pass.0, regions));
        self.core.run_pass(pass.0);
    }

    pub fn copy_at_end(&mut self, from: Copied<P>, to: Copied<P>, regions: &[CopyRegion]) {
        let regions = regions.iter().map(region_of).collect();
        match (copied(from), copied(to)) {
            (Ok(from), Ok(to)) => self.core.queued.ending.push((from, to, regions)),
            (Err(error), _) | (_, Err(error)) => self.core.refuse(error),
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
            deliver: delivered::<R>,
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
            deliver: delivered::<R>,
        })
    }

    pub fn deliver<R: Readable>(&mut self, tag: u32, bytes: Vec<u8>) {
        let (part, written) = (self.core.part, self.core.written());
        self.core.arrived(
            (part, tag),
            Readback {
                value: bytes,
                written,
            },
            delivered::<R>,
        );
    }

    pub fn refuse(&mut self, error: Error) {
        self.core.refuse(error);
    }

    pub fn canvas(&self) -> Option<wgpu::TextureFormat> {
        self.core
            .canvas.texture
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
        if !self.core.begin_render(raw_attach(&attach), None) {
            self.core.end_render();
        }
        self.core.point = None;
        let part = self.core.part;
        Draw {
            raw: RawDraw::new(&mut *self.core, part, self.timer),
            view: None,
            part: PhantomData,
        }
    }

    pub fn view(&mut self, target: &Attach<P>, local: u32) -> Option<Draw<'_, P, Own>> {
        let view = self.core.owned_views(local..local + 1).pop()?;
        self.open_view(target, view)
    }

    pub fn views(&mut self, own: &P, target: &Attach<P>, views: Range<u32>) -> u32 {
        let mut drawn = 0;
        for view in self.core.owned_views(views) {
            if let Some(mut draw) = self.open_view(target, view) {
                drawn += draw.scene(own);
            }
        }
        drawn
    }

    fn open_view(&mut self, target: &Attach<P>, view: encode::OwnedDraw) -> Option<Draw<'_, P, Own>> {
        self.core.open_encode_span(self.timer);
        if !self.core.begin_render(raw_attach(target), Some(&view)) {
            self.core.end_render();
            return None;
        }
        self.core.point = None;
        self.core.view_rect(view.rect);
        let part = self.core.part;
        Some(Draw {
            raw: RawDraw::new(&mut *self.core, part, self.timer),
            view: Some(view),
            part: PhantomData,
        })
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
    view: Option<encode::OwnedDraw>,
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
        self.issue(pipeline, (shape, &[]), None);
    }

    pub fn draw_at(&mut self, pipeline: &Pipeline<P, S>, shape: Shape<'_, P, S>, offsets: &[u32]) {
        self.issue(pipeline, (shape, offsets), None);
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
        self.issue(pipeline, (shape, &[]), Some((timer, &mut named)));
    }

    fn issue(
        &mut self,
        pipeline: &Pipeline<P, S>,
        (shape, offsets): (Shape<'_, P, S>, &[u32]),
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
        run.draw(part, point, pipeline.0, (shape, offsets), timed);
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
        let view = self.view;
        let run = self.raw.run();
        match &view {
            Some(view) => run.view_rect(view.rect),
            None => run.reset_scissor(),
        }
        let Some(scene) = run.scene_draw(view.as_ref()) else {
            return 0;
        };
        let members = run.scene_members();
        let encoding = self.raw.part();
        let mut drawn = 0;
        for member in members.iter() {
            if member.part == encoding {
                let run = self.raw.run();
                run.point = Some(scene.pass.0);
                drawn += (member.call)(own, &scene, &mut self.raw.member(member.part));
                self.raw.run().point = None;
                continue;
            }
            let run = self.raw.run();
            let Some(held) = run.take_member(member.part) else {
                continue;
            };
            run.point = Some(scene.pass.0);
            drawn += (member.call)(held.part(), &scene, &mut self.raw.member(member.part));
            let run = self.raw.run();
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
        pub previous: [f32; 4],
        pub region: [f32; 4],
        pub detail: [f32; 4],
        pub view: [u32; 4],
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resolve;

impl Slot for Resolve {
    const NAME: &'static str = "resolve";
    const WGSL: Option<&'static str> = Some(include_str!("places/resolve.wgsl"));
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lit;

impl Slot for Lit {
    const NAME: &'static str = "lit";
    const WGSL: Option<&'static str> = Some(include_str!("places/lit.wgsl"));
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
    pub decode: fn(kind: Option<DrawKind>, ids: [u32; 4], color: [f32; 4]) -> serde_json::Value,
}

pub struct DebugModes;

impl List for DebugModes {
    const NAME: &'static str = "debug_modes";
    type Member = DebugMode;
    const WGSL: Option<&'static str> = Some(include_str!("places/debug_modes.wgsl"));
}
