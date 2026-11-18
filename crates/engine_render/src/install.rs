use std::any::TypeId;
use std::ops::Range;

use engine_core::{Entry, Error, ErrorCode, PassBit, PassViews, TargetName, Var};

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
    Target(TargetName),
    Asset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Copied {
    Buffer(u32),
    Texture(u32),
    Target(TargetName),
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
    pub(crate) views: PassViews,
    pub(crate) after: Vec<u32>,
    pub(crate) colors: Vec<(u32, Access)>,
    pub(crate) targets: Vec<(TargetName, Access)>,
    pub(crate) depths: Vec<(u32, Access)>,
    pub(crate) clears: Vec<(u32, Clear)>,
    pub(crate) scene: Option<SceneOf>,
    pub(crate) members: Vec<u32>,
    pub(crate) makes_records: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct RenderDecl {
    pub(crate) vertex: Entry,
    pub(crate) fragment: Option<Entry>,
    pub(crate) vertices: Vec<VertexLayout>,
    pub(crate) primitive: wgpu::PrimitiveState,
    pub(crate) depth: DepthUse,
    pub(crate) stencil: wgpu::StencilState,
    pub(crate) targets: Vec<(Option<wgpu::BlendState>, wgpu::ColorWrites)>,
    pub(crate) samples: u32,
    pub(crate) per_batch: crate::PerBatch,
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
    pub(crate) constants: Vec<(&'static str, f64)>,
    pub(crate) speedup_only: bool,
}

impl PassDecl {
    pub(crate) fn clear_of(&self, texture: u32) -> Option<Clear> {
        self.clears
            .iter()
            .find(|(known, _)| *known == texture)
            .map(|(_, clear)| *clear)
    }

    pub(crate) fn per_view(&self) -> bool {
        self.scene.is_some_and(|scene| scene.core) || !self.targets.is_empty()
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
    pub(crate) targets: Vec<crate::SharedTarget>,
    pub(crate) passes: Vec<PassDecl>,
    pub(crate) pipelines: Vec<PipelineDecl>,
    pub(crate) inputs: Vec<DrawInputDecl>,
    pub(crate) frame: Vec<FrameBind>,
    pub(crate) batch_groups: Vec<BatchDecl>,
}

#[derive(Clone, Copy)]
pub(crate) struct BatchDecl {
    pub(crate) module: engine_core::Module,
    pub(crate) index: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Counts {
    targets: usize,
    textures: usize,
    buffers: usize,
    passes: usize,
    pipelines: usize,
    inputs: usize,
    frame: usize,
    batch_groups: usize,
}

impl Declared {
    pub(crate) fn counts(&self) -> Counts {
        Counts {
            targets: self.targets.len(),
            textures: self.textures.len(),
            buffers: self.buffers.len(),
            passes: self.passes.len(),
            pipelines: self.pipelines.len(),
            inputs: self.inputs.len(),
            frame: self.frame.len(),
            batch_groups: self.batch_groups.len(),
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
            targets: kept(&self.targets, counts.targets),
            passes: kept(&self.passes, counts.passes),
            pipelines: kept(&self.pipelines, counts.pipelines),
            inputs: kept(&self.inputs, counts.inputs),
            frame: kept(&self.frame, counts.frame),
            batch_groups: kept(&self.batch_groups, counts.batch_groups),
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
    pub(crate) entries: crate::registration::Entries,
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
            entries: crate::registration::Entries::default(),
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
            constants: Vec::new(),
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
        let mut members: Vec<u32> = self
            .draws
            .iter()
            .filter(|member| member.kind.joins(name))
            .map(|member| member.part)
            .collect();
        members.sort_unstable();
        members.dedup();
        let core = self.package == crate::linker::CORE_PACKAGE;
        if let Some(declared) = self.pass_mut(pass) {
            declared.views = listed.views;
            declared.scene = Some(SceneOf {
                name,
                pass: listed,
                bit,
                core,
            });
            declared.members = members;
        }
    }
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) fn pipeline_facts(decl: &PipelineDecl) -> serde_json::Value {
    use serde_json::{Map, Value, json};

    let constants: Map<String, Value> = decl
        .constants
        .iter()
        .map(|&(name, value)| (name.to_owned(), json!(value)))
        .collect();
    let Program::Render(render) = &decl.program else {
        let compute = engine_registry::unavailable("compute_pipeline");
        return json!({
            "program": "compute",
            "samples": compute,
            "stencil": compute,
            "blend": compute,
            "depth": compute,
            "per_batch": compute,
            "constants": constants,
        });
    };
    let named = |value: &dyn std::fmt::Debug| format!("{value:?}");
    let face = |face: wgpu::StencilFaceState| {
        json!([
            named(&face.compare),
            named(&face.fail_op),
            named(&face.depth_fail_op),
            named(&face.pass_op)
        ])
    };
    let stencil = if render.stencil.is_enabled() {
        json!({
            "front": face(render.stencil.front),
            "back": face(render.stencil.back),
            "read_mask": render.stencil.read_mask,
            "write_mask": render.stencil.write_mask,
        })
    } else {
        json!("off")
    };
    let component = |component: wgpu::BlendComponent| {
        json!([
            named(&component.src_factor),
            named(&component.dst_factor),
            named(&component.operation)
        ])
    };
    let channels = [
        (wgpu::ColorWrites::RED, "red"),
        (wgpu::ColorWrites::GREEN, "green"),
        (wgpu::ColorWrites::BLUE, "blue"),
        (wgpu::ColorWrites::ALPHA, "alpha"),
    ];
    let blend: Vec<Value> = render
        .targets
        .iter()
        .map(|(blend, writes)| {
            let written: Vec<&str> = channels
                .iter()
                .filter(|(channel, _)| writes.contains(*channel))
                .map(|(_, name)| *name)
                .collect();
            let blend = blend.map_or_else(
                || json!("off"),
                |blend| json!({"color": component(blend.color), "alpha": component(blend.alpha)}),
            );
            json!({"blend": blend, "writes": written})
        })
        .collect();
    let depth = match render.depth {
        DepthUse::None => json!("off"),
        DepthUse::Compare {
            compare,
            write,
            bias,
        } => json!({
            "compare": named(&compare),
            "write": write,
            "bias": {
                "constant": bias.constant,
                "slope_scale": bias.slope_scale,
                "clamp": bias.clamp,
            },
        }),
    };
    json!({
        "program": "render",
        "samples": render.samples,
        "stencil": stencil,
        "blend": blend,
        "depth": depth,
        "per_batch": {
            "state": render.per_batch.state,
            "cull": render.per_batch.cull,
            "fragment": render.per_batch.fragment,
        },
        "constants": constants,
    })
}

#[cfg(any(feature = "dev", feature = "profiling"))]
impl Declared {
    pub(crate) fn target_facts(&self, reinstalls: &[Reinstall]) -> serde_json::Value {
        let mut targets = serde_json::Map::new();
        for (at, target) in self.targets.iter().enumerate() {
            let declaring = reinstalls
                .iter()
                .position(|again| (again.from.targets..again.to.targets).contains(&at))
                .and_then(|part| self.parts.get(part));
            let (part, brick) = declaring.map_or_else(
                || {
                    let reason = engine_registry::unavailable("declared_outside_a_part");
                    (reason.clone(), reason)
                },
                |part| {
                    (
                        serde_json::json!(part.name),
                        serde_json::json!(part.package),
                    )
                },
            );
            let clear = match target.clear {
                Clear::Color(color) => serde_json::json!({"color": color}),
                Clear::Depth(depth) => serde_json::json!({"depth": depth}),
            };
            targets.insert(
                target.name.name().to_owned(),
                serde_json::json!({
                    "format": format!("{:?}", target.format),
                    "samples": target.samples,
                    "clear": clear,
                    "part": part,
                    "brick": brick,
                }),
            );
        }
        serde_json::Value::Object(targets)
    }
}
