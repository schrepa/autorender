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
    Asset, Entry, Error, Material, MeshFile, PassBit, Read, TargetName, Var, gpu_struct,
};

use crate::draws::RawDraw;
pub use draws::BatchOf;

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
mod platform;
mod profiling;
mod readback;
mod renderer;
mod targets;
mod textures;
mod variant;
#[cfg(feature = "dev")]
mod whatif;

#[cfg(feature = "dev")]
pub use profiling::INSTRUMENTS as GPU_INSTRUMENTS;
#[doc(hidden)]
pub use platform::{AdapterInfo, RenderTarget};
pub use linker::LinkedShaders;
pub use platform::{GpuAsk, GpuRequest, SurfaceForm};
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use platform::{NativeWindow, keep_pipeline_cache_in};
pub use render_bricks::RenderBricks;
#[doc(hidden)]
#[cfg(feature = "dev")]
pub use gpu_calls::unchanged_writes;
pub use renderer::{Ahead, DeviceHealth, Renderer, StartingRenderer, device_limits};
#[doc(hidden)]
#[cfg(feature = "dev")]
pub use renderer::install_noop;
#[cfg(feature = "dev")]
pub use renderer::gpu_names;
pub use shaders::FULLSCREEN;
pub use targets::{SharedTarget, TargetClear};
pub use variant::{PerBatch, Variant};

engine_core::settings! {
    #[section("render")]
    pub struct RenderSettings {
        pub readback_slots: u32 = 3 => at_least(1),
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
handle!(Readback<P>);
handle!(Sampler<P>);
handle!(Pass<P>);
handle!(Pipeline<P, S>);
handle!(ComputePipeline<P>);
handle!(BatchGroup<P>);

pub struct Own;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Scene(&'static str);

impl Scene {
    pub const fn named(name: &'static str) -> Self {
        Self(name)
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

pub enum Bind<P> {
    Buffer(Buffer<P>),
    Texture(Texture<P>),
    TextureMip(Texture<P>, u32),
    Sampler(Sampler<P>),
    Target(TargetName),
    Asset,
}

impl<P> Clone for Bind<P> {
    fn clone(&self) -> Self {
        match self {
            Self::Buffer(buffer) => Self::Buffer(*buffer),
            Self::Texture(texture) => Self::Texture(*texture),
            Self::TextureMip(texture, mip) => Self::TextureMip(*texture, *mip),
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
    TextureMip(Texture<P>, u32),
    Asset(Asset<engine_core::Texture>),
    AssetSampler(engine_core::Sampler),
}

pub enum Shape {
    Pulled {
        vertices: Range<u32>,
        instances: Range<u32>,
        key: Option<Asset<Material>>,
    },
    Mesh {
        mesh: Asset<MeshFile>,
        key: Option<Asset<Material>>,
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

pub struct RenderPipeline<'a> {
    pub vertex: Entry,
    pub fragment: Option<Entry>,
    pub vertices: &'a [engine_core::VertexStream],
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
        Bind::Buffer(buffer) => install::Bound::Buffer(buffer.0),
        Bind::Texture(texture) => install::Bound::Texture(texture.0),
        Bind::TextureMip(texture, mip) => install::Bound::TextureMip(texture.0, *mip),
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

    pub fn texture_3d(
        &mut self,
        label: &'static str,
        format: wgpu::TextureFormat,
        lifetime: Lifetime,
    ) -> Texture<P> {
        Texture(self.core.texture_3d(label, format, lifetime), PhantomData)
    }

    pub fn buffer(&mut self, label: &'static str, lifetime: Lifetime) -> Buffer<P> {
        Buffer(self.core.buffer(label, lifetime), PhantomData)
    }

    pub fn indirect_buffer(&mut self, label: &'static str, lifetime: Lifetime) -> Buffer<P> {
        Buffer(self.core.indirect_buffer(label, lifetime), PhantomData)
    }

    pub fn readback(&mut self, label: &'static str) -> Readback<P> {
        Readback(self.core.readback(label), PhantomData)
    }

    pub fn sampler(&mut self, desc: engine_core::Sampler) -> Sampler<P> {
        Sampler(self.core.sampler(desc), PhantomData)
    }

    pub fn frame(&mut self, variable: Var, bound: Bind<P>) -> Result<(), Error> {
        self.core.frame(variable, bound_of(&bound))
    }

    pub fn shared_target(&mut self, target: SharedTarget) {
        self.push_shared_target(target, None);
    }

    pub fn shared_target_or(&mut self, target: SharedTarget, fallback: wgpu::TextureFormat) {
        self.push_shared_target(target, Some(fallback));
    }

    fn push_shared_target(
        &mut self,
        target: SharedTarget,
        fallback: Option<wgpu::TextureFormat>,
    ) {
        let name = target.name.name();
        if self.core.declared.targets.iter().any(|known| known.name == target.name) {
            let path = format!("$.targets.{name}");
            self.core
                .problems
                .push(Error::new(engine_core::ErrorCode::NameTwice, path).got(name));
        } else {
            self.core.declared.targets.push(target);
            self.core.declared.target_fallbacks.push(fallback);
        }
    }

    pub fn pass(&mut self, name: &'static str) -> PassBuilder<'_, P> {
        let pass = self.core.pass(name);
        PassBuilder {
            core: &mut *self.core,
            pass,
            part: PhantomData,
        }
    }

    pub fn after(&mut self, pass: Pass<P>, earlier: Pass<P>) {
        if let Some(declared) = self.core.pass_mut(pass.0) {
            declared.after.push(earlier.0);
        }
    }

    pub fn reaches(&mut self, entry: Entry, var: Var) -> bool {
        self.core.reaches(entry, var)
    }

    pub fn scene_bit(&self, scene: Scene) -> Option<PassBit> {
        self.core.passes.bit(scene.0)
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

    pub fn batch_group(
        &mut self,
        module: engine_core::Module,
        index: u32,
        buffers: &[(Var, Buffer<P>)],
    ) -> Result<BatchGroup<P>, Error> {
        let core = &mut *self.core;
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
            let declared = core.linked.declared(*var);
            if var.module() != module || declared.is_none_or(|found| found.group != index) {
                return Err(Error::new(engine_core::ErrorCode::UnknownVariable, at).got(var.name()));
            }
            let usage = core.linked.buffer_usage(*var);
            core.use_buffer(buffer.0, usage);
        }
        core.declared.batch_groups.push(install::BatchDecl { module, index });
        Ok(BatchGroup(
            (core.declared.batch_groups.len() - 1) as u32,
            PhantomData,
        ))
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
    pub fn target(self, name: TargetName, access: Access) -> Self {
        if let Some(pass) = self.core.pass_mut(self.pass) {
            pass.targets.push((name, access));
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

    pub fn set_batch(
        &mut self,
        group: BatchGroup<P>,
        key: Option<Asset<Material>>,
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
                    GroupEntry::TextureMip(texture, mip) => {
                        encode::BatchEntryRaw::TextureMip(texture.0, *mip)
                    }
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

    pub fn clear_batch(&mut self, group: BatchGroup<P>, key: Option<Asset<Material>>) {
        self.core.clear_batch(group.0, key);
    }

    pub fn judge(
        &mut self,
        kind: engine_core::DrawKind,
        mut verdict: impl FnMut(BatchOf<'_>, u32) -> Option<u32>,
    ) {
        self.core.judge(kind, &mut verdict);
    }

    pub fn texture_resident(&self, texture: Asset<engine_core::Texture>) -> bool {
        self.core.assets.textures.is_texture_resident(texture)
    }

    pub fn allocate(&mut self, texture: Texture<P>, size: wgpu::Extent3d) -> Allocation {
        self.allocate_mips(texture, size, 1)
    }

    pub fn allocate_mips(
        &mut self,
        texture: Texture<P>,
        size: wgpu::Extent3d,
        mip_levels: u32,
    ) -> Allocation {
        self.core.allocate_mips(texture.0, size, mip_levels)
    }

    pub fn allocate_buffer(&mut self, buffer: Buffer<P>, bytes: u64) -> Allocation {
        self.core.allocate_buffer(buffer.0, bytes)
    }

    pub fn take_readback(&mut self, readback: Readback<P>) -> Option<Vec<u8>> {
        self.core.take_readback(readback.0)
    }

    pub fn write<T: bytemuck::Pod>(&mut self, buffer: Buffer<P>, offset: u64, data: &[T]) {
        self.core
            .write(buffer.0, offset, bytemuck::cast_slice(data));
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

    pub fn fill<S>(
        &mut self,
        pipeline: &Pipeline<P, S>,
        textures: &[Asset<engine_core::Texture>],
    ) -> Pipeline<P, S> {
        Pipeline(self.core.fill(pipeline.0, textures), PhantomData)
    }

    pub fn report(&mut self, error: Error) {
        self.core.fault(error);
    }

    pub fn canvas_size(&self) -> (u32, u32) {
        let (width, height) = self.core.canvas.size;
        (width.max(1), height.max(1))
    }

    pub fn scale_target(&mut self, name: TargetName, fraction: f32) {
        let path = format!("$.targets.{}", name.name());
        if self.core.machine.declared.shared_target(name).is_none() {
            let error = Error::new(engine_core::ErrorCode::UnknownVariable, path).got(name.name());
            self.core.refuse(error);
        } else if fraction.is_finite() && fraction > 0.0 && fraction <= 1.0 {
            self.core.canvas.scale(name, fraction);
        } else {
            let error = Error::new(engine_core::ErrorCode::OutOfRange, format!("{path}.fraction"))
                .allowed(engine_core::Allowed::Range { min: 0.0, max: 1.0 })
                .got(fraction);
            self.core.refuse(error);
        }
    }

    pub fn scale_target_to_pixels(&mut self, name: TargetName, max_pixels: u32) {
        let (width, height) = self.canvas_size();
        let fraction = (f64::from(max_pixels.max(1)) / (f64::from(width) * f64::from(height)))
            .sqrt()
            .clamp(0.25, 1.0) as f32;
        self.scale_target(name, fraction);
    }
}

pub struct Encode<'f, P> {
    core: &'f mut encode::EncodeCore,
    timer: &'f profiling::GpuProfiler,
    part: PhantomData<fn() -> P>,
}

impl<P: GpuPart> Encode<'_, P> {
    pub fn recorded(&mut self) -> Recorded {
        std::mem::take(&mut self.core.tallies.worst)
    }

    pub fn render(&mut self) -> Draw<'_, P, Own> {
        if !self.core.begin_render() {
            self.core.end_render();
        }
        self.core.point = None;
        let part = self.core.part;
        Draw {
            raw: RawDraw::new(&mut *self.core, part, self.timer),
            part: PhantomData,
        }
    }

    pub fn dispatch(&mut self, pipeline: &ComputePipeline<P>, groups: [u32; 3]) {
        let part = self.core.part;
        self.core.dispatch(part, pipeline.0, groups);
    }

    pub fn dispatch_indirect(
        &mut self,
        pipeline: &ComputePipeline<P>,
        buffer: Buffer<P>,
        offset: u64,
    ) {
        let part = self.core.part;
        self.core
            .dispatch_indirect(part, pipeline.0, buffer.0, offset);
    }

    pub fn clear_buffer(&mut self, buffer: Buffer<P>, range: Range<u64>) {
        self.core.clear_buffer(buffer.0, range);
    }

    pub fn copy_buffer(
        &mut self,
        src: Buffer<P>,
        src_offset: u64,
        dst: Buffer<P>,
        dst_offset: u64,
        size: u64,
    ) {
        self.core
            .copy_buffer(src.0, src_offset, dst.0, dst_offset, size);
    }

    pub fn copy_to_readback(&mut self, readback: Readback<P>, buffer: Buffer<P>, range: Range<u64>) {
        self.core.copy_to_readback(readback.0, buffer.0, range);
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

impl<P: GpuPart, S: 'static> Draw<'_, P, S> {
    pub fn draw(&mut self, pipeline: &Pipeline<P, S>, shape: Shape) -> Recorded {
        let (part, timer) = (self.raw.part(), self.raw.timer());
        let run = self.raw.run();
        let point = run.point;
        run.draw(part, point, pipeline.0, shape, timer)
    }

    pub fn draw_indirect(
        &mut self,
        pipeline: &Pipeline<P, S>,
        buffer: Buffer<P>,
        offset: u64,
        key: Option<Asset<Material>>,
    ) -> Recorded {
        let (part, timer) = (self.raw.part(), self.raw.timer());
        let run = self.raw.run();
        let point = run.point;
        run.draw_indirect(part, point, pipeline.0, (buffer.0, offset, key), timer)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawOrder {
    Grouped,
    Sorted,
}

impl<P: GpuPart> Draw<'_, P, Scene> {
    pub fn batches(&mut self, pipeline: &Pipeline<P, Scene>, vertices: Range<u32>, order: DrawOrder) {
        self.raw.batches((pipeline.0, vertices, order));
    }

    pub fn batches_matching(
        &mut self,
        pipeline: &Pipeline<P, Scene>,
        vertices: Range<u32>,
        order: DrawOrder,
        mut keep: impl FnMut(BatchOf<'_>) -> bool,
    ) {
        self.raw
            .batches_matching((pipeline.0, vertices, order), &mut keep);
    }

    pub fn pass(&mut self) -> Option<PassBit> {
        self.raw.run().point
    }
}

impl<P: GpuPart> Draw<'_, P, Own> {
    pub fn scene(&mut self, own: &P) {
        let run = self.raw.run();
        let Some(bit) = run.scene_bit() else {
            return;
        };
        let scene = bit.index();
        run.tallies.kinds.scened(scene);
        let Some(listed) = run.installs.members.get_mut(scene as usize).map(std::mem::take) else {
            return;
        };
        let mut entries = std::mem::take(&mut run.installs.draw_entries);
        run.point = Some(bit);
        let encoding = self.raw.part();
        for &at in &listed {
            let Some(member) = entries.get_mut(at as usize) else {
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
            let batches = (&mut member.batches, Some(bit));
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
    }
}
