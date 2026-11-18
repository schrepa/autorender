use std::any::TypeId;
use std::ops::Range;

use engine_core::{AttachPoint, Entry, Error, ErrorCode, PassBit, PassViews, TargetName, Var};

use crate::graph::Clear;
use crate::linker::Linker;
use crate::registration::KindMember;
use crate::{Access, DepthUse, Lifetime, VertexLayout};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Bound {
    Texture(u32),
    Slice {
        texture: u32,
        mips: Range<u32>,
        layers: Range<u32>,
    },
    Buffer(u32),
    Window {
        buffer: u32,
        size: u64,
    },
    Sampler(u32),
    Target(ViewTarget),
    Asset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ViewTarget {
    Scene,
    Depth,
}

impl ViewTarget {
    pub(crate) fn of(name: TargetName) -> Option<Self> {
        if name == TargetName::COLOR {
            return Some(Self::Scene);
        }
        (name == TargetName::DEPTH).then_some(Self::Depth)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Copied {
    Buffer(u32),
    Texture(u32),
    Target(ViewTarget),
    Canvas,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Work {
    Render,
    Copy { from: Copied, to: Copied },
    Clear(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Scoped {
    Pass(u32),
    Scene(&'static str),
    Outside(&'static str),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SceneOf {
    pub(crate) name: &'static str,
    pub(crate) pass: engine_core::Pass,
    pub(crate) bit: Option<PassBit>,
    pub(crate) raster: engine_core::PassRaster,
    pub(crate) core: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct TextureDecl {
    pub(crate) label: &'static str,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) lifetime: Lifetime,
    pub(crate) part: u32,
    pub(crate) per_view: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct BufferDecl {
    pub(crate) label: &'static str,
    pub(crate) lifetime: Lifetime,
    pub(crate) part: u32,
    pub(crate) usage: wgpu::BufferUsages,
    pub(crate) window: Option<u64>,
    pub(crate) per_view: bool,
    pub(crate) args: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct PassDecl {
    pub(crate) name: &'static str,
    pub(crate) label: Option<&'static str>,
    pub(crate) span: Option<&'static str>,
    pub(crate) group: Option<&'static str>,
    pub(crate) part: u32,
    pub(crate) work: Work,
    pub(crate) layer: Option<AttachPoint>,
    pub(crate) views: PassViews,
    pub(crate) after: Vec<u32>,
    pub(crate) colors: Vec<(u32, Access)>,
    pub(crate) targets: Vec<ViewTarget>,
    pub(crate) depths: Vec<(u32, Access)>,
    pub(crate) clears: Vec<(u32, Clear)>,
    pub(crate) scene: Option<SceneOf>,
    pub(crate) members: Vec<u32>,
    pub(crate) makes_records: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FragmentUse {
    Always,
    Masked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Faces {
    Declared,
    Material(FragmentUse),
}

#[derive(Clone, Debug)]
pub(crate) struct RenderDecl {
    pub(crate) vertex: Entry,
    pub(crate) fragment: Option<Entry>,
    pub(crate) vertices: Vec<VertexLayout>,
    pub(crate) primitive: wgpu::PrimitiveState,
    pub(crate) depth: DepthUse,
    pub(crate) blend: Option<wgpu::BlendState>,
    pub(crate) writes: wgpu::ColorWrites,
    pub(crate) targets: Option<Vec<(Option<wgpu::BlendState>, wgpu::ColorWrites)>>,
    pub(crate) samples: u32,
    pub(crate) faces: Faces,
}

#[derive(Clone, Debug)]
pub(crate) enum Program {
    Render(RenderDecl),
    Compute(Entry),
}

#[derive(Clone, Debug)]
pub(crate) struct PipelineDecl {
    pub(crate) part: u32,
    pub(crate) scope: Scoped,
    pub(crate) program: Program,
    pub(crate) bound: Vec<(Var, Bound)>,
    pub(crate) speedup_only: bool,
}

impl PassDecl {
    pub(crate) fn clear_of(&self, texture: u32) -> Option<Clear> {
        self.clears
            .iter()
            .find(|(known, _)| *known == texture)
            .map(|(_, clear)| *clear)
    }

    pub(crate) fn view_depth(&self, texture: u32, access: Access) -> bool {
        access == Access::Clear
            && self
                .clear_of(texture)
                .is_none_or(|clear| clear == Clear::Depth(crate::graph::FAR))
    }

    pub(crate) fn per_view(&self) -> bool {
        self.layer.is_some() || crate::shared::composites(self)
    }
}

impl Bound {
    pub(crate) fn window(&self) -> bool {
        matches!(self, Self::Window { .. })
    }
}

impl PipelineDecl {
    pub(crate) fn entries(&self) -> Vec<Entry> {
        match &self.program {
            Program::Render(render) => std::iter::once(render.vertex)
                .chain(render.fragment)
                .collect(),
            Program::Compute(entry) => vec![*entry],
        }
    }

    pub(crate) fn holes(&self) -> usize {
        self.bound
            .iter()
            .filter(|(_, bound)| *bound == Bound::Asset)
            .count()
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DrawInputDecl {
    pub(crate) part: u32,
    pub(crate) scope: Scoped,
    pub(crate) buffer: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct FrameBind {
    pub(crate) var: Var,
    pub(crate) bound: Bound,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PartInfo {
    pub(crate) ty: TypeId,
    pub(crate) package: &'static str,
    pub(crate) name: &'static str,
}

#[derive(Clone, Default)]
pub(crate) struct Declared {
    pub(crate) parts: Vec<PartInfo>,
    pub(crate) textures: Vec<TextureDecl>,
    pub(crate) buffers: Vec<BufferDecl>,
    pub(crate) samplers: Vec<engine_core::Sampler>,
    pub(crate) passes: Vec<PassDecl>,
    pub(crate) pipelines: Vec<PipelineDecl>,
    pub(crate) inputs: Vec<DrawInputDecl>,
    pub(crate) frame: Vec<FrameBind>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Counts {
    textures: usize,
    buffers: usize,
    passes: usize,
    pipelines: usize,
    inputs: usize,
    frame: usize,
}

impl Declared {
    pub(crate) fn counts(&self) -> Counts {
        Counts {
            textures: self.textures.len(),
            buffers: self.buffers.len(),
            passes: self.passes.len(),
            pipelines: self.pipelines.len(),
            inputs: self.inputs.len(),
            frame: self.frame.len(),
        }
    }

    pub(crate) fn before(&self, counts: Counts) -> Self {
        fn kept<T: Clone>(all: &[T], count: usize) -> Vec<T> {
            all.iter().take(count).cloned().collect()
        }
        Self {
            parts: self.parts.clone(),
            textures: kept(&self.textures, counts.textures),
            buffers: kept(&self.buffers, counts.buffers),
            samplers: self.samplers.clone(),
            passes: kept(&self.passes, counts.passes),
            pipelines: kept(&self.pipelines, counts.pipelines),
            inputs: kept(&self.inputs, counts.inputs),
            frame: kept(&self.frame, counts.frame),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Reinstall {
    pub(crate) install: crate::registration::InstallFn,
    pub(crate) package: &'static str,
    pub(crate) from: Counts,
    pub(crate) to: Counts,
}

pub(crate) struct InstallCore {
    pub(crate) part: u32,
    pub(crate) package: &'static str,
    pub(crate) declared: Declared,
    pub(crate) draws: Vec<KindMember>,
    pub(crate) passes: engine_core::ScenePasses,
    pub(crate) linker: Linker,
    pub(crate) problems: Vec<Error>,
}

impl InstallCore {
    pub(crate) fn new(linker: Linker) -> Self {
        Self {
            part: 0,
            package: "",
            declared: Declared::default(),
            draws: Vec::new(),
            passes: engine_core::ScenePasses::default(),
            linker,
            problems: Vec::new(),
        }
    }

    fn path(&self, what: &str) -> String {
        let name = self
            .declared
            .parts
            .get(self.part as usize)
            .map_or(self.package, |part| part.name);
        format!("$.parts.{name}.{what}")
    }

    pub(crate) fn texture(
        &mut self,
        label: &'static str,
        format: wgpu::TextureFormat,
        (lifetime, per_view): (Lifetime, bool),
    ) -> u32 {
        self.declared.textures.push(TextureDecl {
            label,
            format,
            lifetime,
            part: self.part,
            per_view,
        });
        (self.declared.textures.len() - 1) as u32
    }

    pub(crate) fn buffer(
        &mut self,
        label: &'static str,
        (lifetime, per_view): (Lifetime, bool),
    ) -> u32 {
        self.declared.buffers.push(BufferDecl {
            label,
            lifetime,
            part: self.part,
            usage: wgpu::BufferUsages::empty(),
            window: None,
            per_view,
            args: false,
        });
        (self.declared.buffers.len() - 1) as u32
    }

    pub(crate) fn window(&mut self, buffer: u32, size: u64) {
        let part = self.part;
        if let Some(declared) = self
            .declared
            .buffers
            .get_mut(buffer as usize)
            .filter(|declared| declared.part == part)
        {
            declared.window = Some(size);
        }
    }

    pub(crate) fn use_buffer(&mut self, buffer: u32, usage: wgpu::BufferUsages) {
        if let Some(declared) = self.declared.buffers.get_mut(buffer as usize) {
            declared.usage |= usage;
        }
    }

    pub(crate) fn sampler(&mut self, desc: engine_core::Sampler) -> u32 {
        match self
            .declared
            .samplers
            .iter()
            .position(|known| *known == desc)
        {
            Some(index) => index as u32,
            None => {
                self.declared.samplers.push(desc);
                (self.declared.samplers.len() - 1) as u32
            }
        }
    }

    pub(crate) fn frame(&mut self, var: Var, bound: Bound) -> Result<(), Error> {
        let path = self.path(&format!("frame.{}", var.name()));
        self.linker.add_source(var.module().source());
        match self.linker.declared(var) {
            Some(declared) if declared.group == 0 => {}
            Some(_) | None => {
                return Err(Error::new(ErrorCode::UnknownVariable, path).got(var.name()));
            }
        }
        if bound == Bound::Asset {
            return Err(Error::new(ErrorCode::UnknownVariable, path).got("asset"));
        }
        if let Bound::Buffer(buffer) = bound {
            let usage = self.linker.buffer_usage(var);
            self.use_buffer(buffer, usage);
        }
        self.declared.frame.push(FrameBind { var, bound });
        Ok(())
    }

    pub(crate) fn pass(&mut self, name: &'static str, work: Work) -> u32 {
        self.declared.passes.push(PassDecl {
            name,
            label: None,
            span: None,
            group: None,
            part: self.part,
            work,
            layer: None,
            views: PassViews::Every,
            after: Vec::new(),
            colors: Vec::new(),
            depths: Vec::new(),
            clears: Vec::new(),
            targets: Vec::new(),
            scene: None,
            members: Vec::new(),
            makes_records: false,
        });
        let pass = (self.declared.passes.len() - 1) as u32;
        match work {
            Work::Copy { from, to } => {
                for (copied, usage) in [
                    (from, wgpu::BufferUsages::COPY_SRC),
                    (to, wgpu::BufferUsages::COPY_DST),
                ] {
                    if let Copied::Buffer(buffer) = copied {
                        self.use_buffer(buffer, usage);
                    }
                }
            }
            Work::Clear(buffer) => self.use_buffer(buffer, wgpu::BufferUsages::COPY_DST),
            Work::Render => {}
        }
        pass
    }

    pub(crate) fn pass_mut(&mut self, pass: u32) -> Option<&mut PassDecl> {
        let part = self.part;
        self.declared
            .passes
            .get_mut(pass as usize)
            .filter(|declared| declared.part == part)
    }

    pub(crate) fn speedup_only(&mut self, pipeline: u32) {
        if let Some(declared) = self.declared.pipelines.get_mut(pipeline as usize) {
            declared.speedup_only = true;
        }
    }

    pub(crate) fn pipeline(
        &mut self,
        scope: Scoped,
        program: Program,
        bound: Vec<(Var, Bound)>,
    ) -> Result<u32, Error> {
        let label = match &program {
            Program::Render(render) => render.vertex.name(),
            Program::Compute(entry) => entry.name(),
        };
        let path = self.path(&format!("pipelines.{label}"));
        if let Scoped::Pass(pass) = scope
            && self.pass_mut(pass).is_none()
        {
            return Err(Error::new(ErrorCode::PassMismatch, path));
        }
        let bound = bound
            .into_iter()
            .map(|(var, bound)| {
                let window = match &bound {
                    Bound::Buffer(buffer) => self
                        .declared
                        .buffers
                        .get(*buffer as usize)
                        .and_then(|declared| declared.window)
                        .map(|size| (*buffer, size)),
                    Bound::Texture(_)
                    | Bound::Slice { .. }
                    | Bound::Window { .. }
                    | Bound::Sampler(_)
                    | Bound::Target(_)
                    | Bound::Asset => None,
                };
                match window {
                    Some((buffer, size)) => (var, Bound::Window { buffer, size }),
                    None => (var, bound),
                }
            })
            .collect();
        let decl = PipelineDecl {
            part: self.part,
            scope,
            program,
            bound,
            speedup_only: false,
        };
        for entry in decl.entries() {
            self.linker.add_source(entry.module().source());
            if !self.linker.has_entry(entry) {
                return Err(Error::new(ErrorCode::UnknownModule, path).got(entry.name()));
            }
        }
        for (var, bound) in &decl.bound {
            match self.linker.declared(*var) {
                Some(declared) if declared.group > 0 => {}
                Some(_) | None => {
                    return Err(Error::new(ErrorCode::UnknownVariable, path).got(var.name()));
                }
            }
            if let Bound::Buffer(buffer) | Bound::Window { buffer, .. } = bound {
                let usage = self.linker.buffer_usage(*var);
                self.use_buffer(*buffer, usage);
            }
        }
        self.declared.pipelines.push(decl);
        Ok((self.declared.pipelines.len() - 1) as u32)
    }

    pub(crate) fn reaches(&mut self, entry: Entry, var: Var) -> bool {
        self.linker.add_source(entry.module().source());
        self.linker.add_source(var.module().source());
        self.linker.reaches(entry, var)
    }

    pub(crate) fn draw_input(&mut self, scope: Scoped, buffer: u32) -> u32 {
        self.use_buffer(
            buffer,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX | wgpu::BufferUsages::INDIRECT,
        );
        self.declared.inputs.push(DrawInputDecl {
            part: self.part,
            scope,
            buffer,
        });
        (self.declared.inputs.len() - 1) as u32
    }

    pub(crate) fn args_input(&mut self, scope: Scoped, buffer: u32) -> u32 {
        if let Some(declared) = self.declared.buffers.get_mut(buffer as usize) {
            declared.args = true;
        }
        self.draw_input(scope, buffer)
    }

    pub(crate) fn draws_scene(&mut self, pass: u32, name: &'static str, listed: engine_core::Pass) {
        let bit = self.passes.bit(name);
        let raster = self
            .passes
            .get(name)
            .map_or(engine_core::PassRaster::PLAIN, |listed| listed.raster);
        let mut members: Vec<u32> = self
            .draws
            .iter()
            .filter(|member| member.kind.joins(name, &listed))
            .map(|member| member.part)
            .collect();
        members.sort_unstable();
        members.dedup();
        let core = self.package == crate::linker::CORE_PACKAGE;
        if let Some(declared) = self.pass_mut(pass) {
            if core {
                declared.layer = Some(listed.at);
            }
            declared.views = listed.views;
            declared.scene = Some(SceneOf {
                name,
                pass: listed,
                bit,
                raster,
                core,
            });
            declared.members = members;
        }
    }
}
