use std::any::{Any, TypeId};
use std::ops::Range;

use engine_core::{Entry, Error, ErrorCode, Var};

use crate::graph::Clear;
use crate::linker::Linker;
use crate::{Access, DepthUse, Layer, Lifetime, SamplerDesc, VertexLayout, ViewTarget};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Bound {
    Texture(u32),
    Slice {
        texture: u32,
        mips: Range<u32>,
        layers: Range<u32>,
    },
    Buffer(u32),
    Sampler(u32),
    Target(ViewTarget),
    Asset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Copied {
    Buffer(u32),
    Texture(u32),
    Target(ViewTarget),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Work {
    Render,
    Copy { from: Copied, to: Copied },
    Clear(u32),
    Shared(Layer),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Scoped {
    Pass(u32),
    Point(TypeId),
}

#[derive(Clone, Debug)]
pub(crate) struct TextureDecl {
    pub(crate) label: &'static str,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) lifetime: Lifetime,
    pub(crate) part: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct BufferDecl {
    pub(crate) label: &'static str,
    pub(crate) lifetime: Lifetime,
    pub(crate) part: u32,
    pub(crate) usage: wgpu::BufferUsages,
}

#[derive(Clone, Debug)]
pub(crate) struct PassDecl {
    pub(crate) name: &'static str,
    pub(crate) part: u32,
    pub(crate) work: Work,
    pub(crate) layer: Option<Layer>,
    pub(crate) after: Vec<u32>,
    pub(crate) colors: Vec<(u32, Access)>,
    pub(crate) depths: Vec<(u32, Access)>,
    pub(crate) clears: Vec<(u32, Clear)>,
    pub(crate) claims: Vec<TypeId>,
    pub(crate) members: Vec<u32>,
}

#[derive(Clone, Debug)]
pub(crate) struct RenderDecl {
    pub(crate) vertex: Entry,
    pub(crate) fragment: Option<Entry>,
    pub(crate) debug: Option<Entry>,
    pub(crate) vertices: Vec<VertexLayout>,
    pub(crate) primitive: wgpu::PrimitiveState,
    pub(crate) depth: DepthUse,
    pub(crate) blend: Option<wgpu::BlendState>,
    pub(crate) writes: wgpu::ColorWrites,
    pub(crate) targets: Option<Vec<(Option<wgpu::BlendState>, wgpu::ColorWrites)>>,
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
}

impl PassDecl {
    pub(crate) fn clear_of(&self, texture: u32) -> Option<Clear> {
        self.clears
            .iter()
            .find(|(known, _)| *known == texture)
            .map(|(_, clear)| *clear)
    }
}

impl PipelineDecl {
    pub(crate) fn entries(&self) -> Vec<Entry> {
        match &self.program {
            Program::Render(render) => std::iter::once(render.vertex)
                .chain(render.fragment)
                .chain(render.debug)
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
    pub(crate) part: u32,
    pub(crate) var: Var,
    pub(crate) bound: Bound,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Claim {
    pub(crate) point: TypeId,
    pub(crate) pass: u32,
    pub(crate) part: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PartInfo {
    pub(crate) package: &'static str,
    pub(crate) name: &'static str,
    pub(crate) brick: u32,
    pub(crate) registered: u32,
    pub(crate) owner: &'static str,
    pub(crate) profile: u16,
}

impl PartInfo {
    pub(crate) fn owned(&self) -> crate::memory::Owned {
        crate::memory::owned(self.owner, self.profile)
    }
}

#[derive(Default)]
pub(crate) struct Declared {
    pub(crate) parts: Vec<PartInfo>,
    pub(crate) textures: Vec<TextureDecl>,
    pub(crate) buffers: Vec<BufferDecl>,
    pub(crate) samplers: Vec<SamplerDesc>,
    pub(crate) passes: Vec<PassDecl>,
    pub(crate) pipelines: Vec<PipelineDecl>,
    pub(crate) inputs: Vec<DrawInputDecl>,
    pub(crate) frame: Vec<FrameBind>,
    pub(crate) claims: Vec<Claim>,
}

pub(crate) struct DrawEntry {
    pub(crate) package: &'static str,
    pub(crate) part: u32,
    pub(crate) record: std::sync::Arc<dyn Any + Send + Sync>,
}

pub(crate) struct InstallCore {
    pub(crate) part: u32,
    pub(crate) package: &'static str,
    pub(crate) fills: Vec<TypeId>,
    pub(crate) declared: Declared,
    pub(crate) draws: Vec<DrawEntry>,
    pub(crate) linker: Linker,
    pub(crate) features: wgpu::Features,
    pub(crate) problems: Vec<Error>,
}

fn core_entry(entry: Entry) -> bool {
    std::ptr::eq(entry.module().source(), crate::shared::core_source())
}

impl InstallCore {
    pub(crate) fn new(linker: Linker, features: wgpu::Features) -> Self {
        Self {
            part: 0,
            package: "",
            fills: Vec::new(),
            declared: Declared::default(),
            draws: Vec::new(),
            linker,
            features,
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
        lifetime: Lifetime,
    ) -> u32 {
        self.declared.textures.push(TextureDecl {
            label,
            format,
            lifetime,
            part: self.part,
        });
        (self.declared.textures.len() - 1) as u32
    }

    pub(crate) fn buffer(&mut self, label: &'static str, lifetime: Lifetime) -> u32 {
        self.declared.buffers.push(BufferDecl {
            label,
            lifetime,
            part: self.part,
            usage: wgpu::BufferUsages::empty(),
        });
        (self.declared.buffers.len() - 1) as u32
    }

    pub(crate) fn use_buffer(&mut self, buffer: u32, usage: wgpu::BufferUsages) {
        if let Some(declared) = self.declared.buffers.get_mut(buffer as usize) {
            declared.usage |= usage;
        }
    }

    pub(crate) fn sampler(&mut self, desc: SamplerDesc) -> u32 {
        match self.declared.samplers.iter().position(|known| *known == desc) {
            Some(index) => index as u32,
            None => {
                self.declared.samplers.push(desc);
                (self.declared.samplers.len() - 1) as u32
            }
        }
    }

    pub(crate) fn frame(&mut self, var: Var, bound: Bound) -> Result<(), Error> {
        let path = self.path(&format!("frame.{}", var.name()));
        self.owns(var.module().source().package(), &path)?;
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
        self.declared.frame.push(FrameBind {
            part: self.part,
            var,
            bound,
        });
        Ok(())
    }

    fn owns(&self, package: &str, path: &str) -> Result<(), Error> {
        if package == self.package {
            return Ok(());
        }
        Err(Error::new(ErrorCode::ForeignAccess, path.to_owned()).got(package))
    }

    pub(crate) fn pass(&mut self, name: &'static str, work: Work) -> u32 {
        self.declared.passes.push(PassDecl {
            name,
            part: self.part,
            work,
            layer: None,
            after: Vec::new(),
            colors: Vec::new(),
            depths: Vec::new(),
            clears: Vec::new(),
            claims: Vec::new(),
            members: Vec::new(),
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
            Work::Render | Work::Shared(_) => {}
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
        let decl = PipelineDecl {
            part: self.part,
            scope,
            program,
            bound,
        };
        for entry in decl.entries() {
            if !core_entry(entry) {
                self.owns(entry.module().source().package(), &path)?;
            }
            self.linker.add_source(entry.module().source());
            if !self.linker.has_entry(entry) {
                return Err(Error::new(ErrorCode::UnknownModule, path).got(entry.name()));
            }
        }
        for (var, bound) in &decl.bound {
            self.owns(var.module().source().package(), &path)?;
            match self.linker.declared(*var) {
                Some(declared) if declared.group > 0 => {}
                Some(_) | None => {
                    return Err(Error::new(ErrorCode::UnknownVariable, path).got(var.name()));
                }
            }
            if let Bound::Buffer(buffer) = bound {
                let usage = self.linker.buffer_usage(*var);
                self.use_buffer(*buffer, usage);
            }
        }
        self.declared.pipelines.push(decl);
        Ok((self.declared.pipelines.len() - 1) as u32)
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

    pub(crate) fn claim(&mut self, point: TypeId, name: &'static str, pass: u32) -> Result<(), Error> {
        let path = self.path(&format!("claims.{name}"));
        if !self.fills.contains(&point) {
            return Err(Error::new(ErrorCode::PlaceNotFilled, path).got(self.package));
        }
        if let Some(known) = self.declared.claims.iter().find(|claim| claim.point == point) {
            let owner = self.declared.parts[known.part as usize].package;
            return Err(Error::new(ErrorCode::SlotFilledTwice, path).got(owner));
        }
        let Some(declared) = self.pass_mut(pass) else {
            return Err(Error::new(ErrorCode::PassMismatch, path));
        };
        declared.claims.push(point);
        self.declared.claims.push(Claim {
            point,
            pass,
            part: self.part,
        });
        Ok(())
    }

    pub(crate) fn members<T: Any>(&self) -> impl Iterator<Item = (u32, &T)> + '_ {
        self.draws
            .iter()
            .filter_map(|entry| Some((entry.part, entry.record.downcast_ref::<T>()?)))
    }
}
