use engine_core::{DepthBias, DrawKind, Entry, Error, ErrorCode, Pass, ScenePass, Var};
use engine_render::{
    Bind, DepthUse, Draw, GpuPart, Install, PerBatch, Pipeline, RenderPipeline, Scene,
    VertexLayout, wgpu,
};
use serde::Serialize;

use super::records::FaceCull;
use super::shaders::surfaces::kinds;
use crate::bricks::scene_pipeline::{COLOR, DEPTH};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FragmentUse {
    Always,
    Masked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PassRole {
    Lit,
    Depth,
    Blended,
}

pub fn role(pass: &Pass) -> PassRole {
    if !pass.writes.contains(DEPTH) && pass.writes.contains(COLOR) {
        PassRole::Blended
    } else if pass.writes.contains(COLOR) {
        PassRole::Lit
    } else {
        PassRole::Depth
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Varyings {
    Surface(FragmentUse),
    SurfaceExtra(FragmentUse),
}

#[derive(Clone, Copy, Debug)]
pub struct FormSide {
    pub vertex: Entry,
    pub streams: &'static [VertexLayout],
    pub depth_bias: DepthBias,
}

impl FormSide {
    pub const UNBIASED: DepthBias = DepthBias {
        constant: 0,
        slope: 0.0,
    };

    pub const fn pulled(vertex: Entry) -> Self {
        Self {
            vertex,
            streams: &[],
            depth_bias: Self::UNBIASED,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KindForm {
    pub kind: DrawKind,
    pub placed: FormSide,
    pub varyings: Varyings,
}

#[derive(Clone, Copy, Debug)]
pub struct PassFragment {
    pub pass: Scene,
    pub side: Option<FormSide>,
    pub entry: Entry,
    pub cull: FaceCull,
    pub blend: Option<wgpu::BlendState>,
    pub depth: DepthUse,
}

struct FormScene<P> {
    scene: Scene,
    pipeline: Pipeline<P, Scene>,
}

pub struct Form<P> {
    scenes: Vec<FormScene<P>>,
}

pub trait SurfaceForms<P> {
    fn form(&mut self, form: &KindForm, bound: &[(Var, Bind<P>)]) -> Result<Form<P>, Error>;
    fn form_owning(
        &mut self,
        form: &KindForm,
        owns: &[PassFragment],
        bound: &[(Var, Bind<P>)],
    ) -> Result<Form<P>, Error>;
}

pub trait FormDraw<P> {
    fn form(&mut self, form: &Form<P>) -> Option<Pipeline<P, Scene>>;
}

struct Program {
    side: FormSide,
    fragment: Entry,
    cull: Option<wgpu::Face>,
    depth: DepthUse,
    blend: Option<wgpu::BlendState>,
    per_batch: PerBatch,
}

fn generic_entry(role: PassRole, extra: bool) -> Entry {
    match (role, extra) {
        (PassRole::Lit, false) => kinds::FS_KIND_LIT,
        (PassRole::Lit, true) => kinds::FS_KIND_LIT_EXTRA,
        (PassRole::Blended, false) => kinds::FS_KIND_BLENDED,
        (PassRole::Blended, true) => kinds::FS_KIND_BLENDED_EXTRA,
        (PassRole::Depth, false) => kinds::FS_KIND_MASKED_DEPTH,
        (PassRole::Depth, true) => kinds::FS_KIND_MASKED_DEPTH_EXTRA,
    }
}

fn fragment_use(role: PassRole, depth: FragmentUse) -> FragmentUse {
    match role {
        PassRole::Depth => depth,
        PassRole::Lit | PassRole::Blended => FragmentUse::Always,
    }
}

fn primitive(cull: Option<wgpu::Face>) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: cull,
        ..Default::default()
    }
}

fn depth_bias(bias: DepthBias) -> wgpu::DepthBiasState {
    wgpu::DepthBiasState {
        constant: bias.constant,
        slope_scale: bias.slope,
        clamp: 0.0,
    }
}

fn tested(write: bool, bias: wgpu::DepthBiasState) -> DepthUse {
    DepthUse::Compare {
        compare: wgpu::CompareFunction::GreaterEqual,
        write,
        bias,
    }
}

fn kind_program((varyings, side): (Varyings, FormSide), role: PassRole) -> Program {
    let (fragment, depth) = match varyings {
        Varyings::Surface(depth) => (generic_entry(role, false), depth),
        Varyings::SurfaceExtra(depth) => (generic_entry(role, true), depth),
    };
    Program {
        side,
        fragment,
        cull: Some(wgpu::Face::Back),
        depth: match role {
            PassRole::Blended => tested(false, wgpu::DepthBiasState::default()),
            PassRole::Lit | PassRole::Depth => tested(true, depth_bias(side.depth_bias)),
        },
        blend: None,
        per_batch: PerBatch {
            state: true,
            cull: true,
            fragment: fragment_use(role, depth) == FragmentUse::Masked,
        },
    }
}

fn pass_program(form: &KindForm, own: &PassFragment) -> Program {
    Program {
        side: own.side.unwrap_or(form.placed),
        fragment: own.entry,
        cull: own.cull.face(),
        depth: own.depth,
        blend: own.blend,
        per_batch: PerBatch {
            state: true,
            ..PerBatch::NONE
        },
    }
}

fn reached<P: GpuPart>(
    cx: &mut Install<'_, P>,
    program: &Program,
    bound: &[(Var, Bind<P>)],
) -> Vec<(Var, Bind<P>)> {
    let entries = [program.side.vertex, program.fragment];
    bound
        .iter()
        .filter(|(var, _)| entries.iter().any(|entry| cx.reaches(*entry, *var)))
        .map(|(var, bind)| (*var, bind.clone()))
        .collect()
}

fn form_built<P: GpuPart>(
    cx: &mut Install<'_, P>,
    form: &KindForm,
    owns: &[PassFragment],
    bound: &[(Var, Bind<P>)],
) -> Result<Form<P>, Error> {
    let passes = cx.passes();
    let listed: Vec<ScenePass> = passes
        .listed()
        .iter()
        .filter(|listed| listed.bit.is_some() && passes.takes(listed, form.kind))
        .copied()
        .collect();
    for own in owns {
        if !listed.iter().any(|listed| listed.name == own.pass.name()) {
            let path = format!("forms.{}.owns", form.kind.0);
            return Err(Error::new(ErrorCode::PassMismatch, path).got(own.pass.name()));
        }
    }
    let mut scenes = Vec::with_capacity(listed.len());
    for listed in listed {
        let scene = Scene::named(listed.name);
        let program = match owns.iter().find(|own| own.pass.name() == listed.name) {
            Some(own) => pass_program(form, own),
            None => kind_program((form.varyings, form.placed), role(&listed.pass)),
        };
        let cut = reached(cx, &program, bound);
        let targets = [(program.blend, wgpu::ColorWrites::ALL)];
        let desc = RenderPipeline {
            vertex: program.side.vertex,
            fragment: Some(program.fragment),
            vertices: program.side.streams,
            primitive: primitive(program.cull),
            depth: program.depth,
            stencil: wgpu::StencilState::default(),
            samples: 1,
            targets: &targets,
            constants: &[],
            per_batch: program.per_batch,
        };
        let pipeline = cx.render_pipeline(scene, &desc, &cut)?;
        scenes.push(FormScene { scene, pipeline });
    }
    Ok(Form { scenes })
}

impl<P: GpuPart> SurfaceForms<P> for Install<'_, P> {
    fn form(&mut self, form: &KindForm, bound: &[(Var, Bind<P>)]) -> Result<Form<P>, Error> {
        form_built(self, form, &[], bound)
    }

    fn form_owning(
        &mut self,
        form: &KindForm,
        owns: &[PassFragment],
        bound: &[(Var, Bind<P>)],
    ) -> Result<Form<P>, Error> {
        form_built(self, form, owns, bound)
    }
}

impl<P: GpuPart> FormDraw<P> for Draw<'_, P, Scene> {
    fn form(&mut self, form: &Form<P>) -> Option<Pipeline<P, Scene>> {
        let pass = self.pass()?;
        form.scenes
            .iter()
            .find(|known| known.scene == pass)
            .map(|known| known.pipeline)
    }
}
