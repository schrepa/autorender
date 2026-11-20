use std::any::TypeId;
use std::sync::Arc;

use engine_core::{Entry, Error, ErrorCode, PassBit, PassViews, TargetName, Var, VertexStream};

use crate::linker::Shaders;
use crate::registration::KindMember;
use crate::{Access, DepthUse, Lifetime};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Bound {
    Buffer(u32),
    Texture(u32),
    Sampler(u32),
    Target(TargetName),
    Asset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Scoped {
    Pass(u32),
    Scene(PassBit),
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
    pub(crate) dimension: wgpu::TextureDimension,
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
    pub(crate) label: Option<&'static str>,
    pub(crate) span: Option<&'static str>,
    pub(crate) part: u32,
    pub(crate) views: PassViews,
    pub(crate) after: Vec<u32>,
    pub(crate) targets: Vec<(TargetName, Access)>,
    pub(crate) scene: Option<SceneOf>,
    pub(crate) members: Vec<u32>,
}

#[derive(Clone, Debug)]
pub(crate) struct RenderDecl {
    pub(crate) vertex: Entry,
    pub(crate) fragment: Option<Entry>,
    pub(crate) vertices: Vec<VertexStream>,
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
}

impl PassDecl {
    pub(crate) fn per_view(&self) -> bool {
        self.scene.is_some_and(|scene| scene.core) || !self.targets.is_empty()
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
    pub(crate) target_fallbacks: Vec<Option<wgpu::TextureFormat>>,
    pub(crate) passes: Vec<PassDecl>,
    pub(crate) pipelines: Vec<PipelineDecl>,
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
    frame: usize,
    batch_groups: usize,
}

impl Declared {
    pub(crate) fn scene_name(&self, bit: PassBit) -> &'static str {
        self.passes
            .iter()
            .find_map(|pass| pass.scene.filter(|scene| scene.bit == Some(bit)))
            .map_or("", |scene| scene.name)
    }

    pub(crate) fn counts(&self) -> Counts {
        Counts {
            targets: self.targets.len(),
            textures: self.textures.len(),
            buffers: self.buffers.len(),
            passes: self.passes.len(),
            pipelines: self.pipelines.len(),
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
            target_fallbacks: kept(&self.target_fallbacks, counts.targets),
            passes: kept(&self.passes, counts.passes),
            pipelines: kept(&self.pipelines, counts.pipelines),
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
    pub(crate) linked: Arc<Shaders>,
    pub(crate) problems: Vec<Error>,
}

impl InstallCore {
    pub(crate) fn new(linked: Arc<Shaders>) -> Self {
        Self {
            part: 0,
            package: "",
            declared: Declared::default(),
            draws: Vec::new(),
            passes: engine_core::ScenePasses::default(),
            linked,
            problems: Vec::new(),
        }
    }

    pub(crate) fn path(&self, what: &str) -> String {
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
        self.texture_with(label, format, wgpu::TextureDimension::D2, lifetime)
    }

    pub(crate) fn texture_3d(
        &mut self,
        label: &'static str,
        format: wgpu::TextureFormat,
        lifetime: Lifetime,
    ) -> u32 {
        self.texture_with(label, format, wgpu::TextureDimension::D3, lifetime)
    }

    fn texture_with(
        &mut self,
        label: &'static str,
        format: wgpu::TextureFormat,
        dimension: wgpu::TextureDimension,
        lifetime: Lifetime,
    ) -> u32 {
        self.declared.textures.push(TextureDecl {
            label,
            format,
            dimension,
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
        match self.linked.declared(var) {
            Some(declared) if declared.group == 0 => {}
            Some(_) | None => {
                return Err(Error::new(ErrorCode::UnknownVariable, path).got(var.name()));
            }
        }
        if bound == Bound::Asset {
            return Err(Error::new(ErrorCode::UnknownVariable, path).got("asset"));
        }
        if let Bound::Buffer(buffer) = bound {
            let usage = self.linked.buffer_usage(var);
            self.use_buffer(buffer, usage);
        }
        self.declared.frame.push(FrameBind { var, bound });
        Ok(())
    }

    pub(crate) fn pass(&mut self, name: &'static str) -> u32 {
        self.declared.passes.push(PassDecl {
            name,
            label: None,
            span: None,
            part: self.part,
            views: PassViews::Every,
            after: Vec::new(),
            targets: Vec::new(),
            scene: None,
            members: Vec::new(),
        });
        (self.declared.passes.len() - 1) as u32
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
            constants: Vec::new(),
        };
        for entry in decl.entries() {
            if !self.linked.has_entry(entry) {
                return Err(Error::new(ErrorCode::UnknownModule, path).got(entry.name()));
            }
        }
        for (var, bound) in &decl.bound {
            match self.linked.declared(*var) {
                Some(declared) if declared.group > 0 => {}
                Some(_) | None => {
                    return Err(Error::new(ErrorCode::UnknownVariable, path).got(var.name()));
                }
            }
            if let Bound::Buffer(buffer) = bound {
                let usage = self.linked.buffer_usage(*var);
                self.use_buffer(*buffer, usage);
            }
        }
        self.declared.pipelines.push(decl);
        Ok((self.declared.pipelines.len() - 1) as u32)
    }

    pub(crate) fn reaches(&self, entry: Entry, var: Var) -> bool {
        self.linked.reaches(entry, var)
    }

    pub(crate) fn draws_scene(&mut self, pass: u32, name: &'static str, listed: engine_core::Pass) {
        let bit = self.passes.bit(name);
        let mut members: Vec<u32> = self
            .draws
            .iter()
            .filter(|member| member.decl.joins(name))
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

#[cfg(feature = "dev")]
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

#[cfg(feature = "dev")]
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
                crate::graph::Clear::Color(color) => serde_json::json!({"color": color}),
                crate::graph::Clear::Depth(depth) => serde_json::json!({"depth": depth}),
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
