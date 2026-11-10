use std::marker::PhantomData;

use engine_core::{
    DepthBias, DrawKind, Entry, Error, ErrorCode, PassName, PassRole, PassViews, Var,
};

use crate::install::{Faces, FragmentUse, Program, RenderDecl, Scoped};
use crate::{
    Bind, DEBUG_BLEND, DepthUse, Draw, FaceCull, GpuPart, Install, Own, Pass, Pipeline, Scene,
    VertexLayout, bound_list,
};

pub(crate) const KINDS_MODULE: &str = "engine::kinds";

#[derive(Clone, Copy, Debug)]
pub enum Varyings {
    Surface,
    SurfaceExtra,
    Object,
    ObjectExtra,
    FadingObject,
    FadingObjectExtra,
    Own(&'static OwnFragments),
}

#[derive(Clone, Copy, Debug)]
pub struct RoleFragments {
    pub lit: Option<Entry>,
    pub depth: Option<Entry>,
    pub blended: Option<Entry>,
    pub shown: Option<Entry>,
}

impl RoleFragments {
    pub const NONE: Self = Self {
        lit: None,
        depth: None,
        blended: None,
        shown: None,
    };

    fn of(&self, role: PassRole) -> Option<Entry> {
        match role {
            PassRole::Lit => self.lit,
            PassRole::Depth => self.depth,
            PassRole::Blended => self.blended,
            PassRole::Shown => self.shown,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OwnFragments {
    pub placed: RoleFragments,
    pub owned: RoleFragments,
    pub cull: FaceCull,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FormTopology {
    List,
    Strip,
}

impl FormTopology {
    fn primitive(self) -> wgpu::PrimitiveTopology {
        match self {
            Self::List => wgpu::PrimitiveTopology::TriangleList,
            Self::Strip => wgpu::PrimitiveTopology::TriangleStrip,
        }
    }
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
    pub owned: Option<FormSide>,
    pub varyings: Varyings,
}

impl KindForm {
    fn side(&self, views: PassViews) -> FormSide {
        match views {
            PassViews::Owned => self.owned.unwrap_or(self.placed),
            PassViews::Every | PassViews::Main => self.placed,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormDepth {
    Written,
    Tested,
}

#[derive(Clone, Copy, Debug)]
pub struct PassFragment {
    pub scene: Scene,
    pub side: Option<FormSide>,
    pub fragment: Entry,
    pub cull: FaceCull,
    pub depth: FormDepth,
}

#[derive(Clone, Copy, Debug)]
pub struct FormPass {
    pub scene: Scene,
    pub role: PassRole,
    pub owned: bool,
}

#[derive(Clone, Copy, Debug)]
struct FormScene {
    pass: FormPass,
    plain: Option<u32>,
    mirrored: Option<u32>,
}

const EVERY_ROLE: [PassRole; 4] = [
    PassRole::Lit,
    PassRole::Depth,
    PassRole::Blended,
    PassRole::Shown,
];

pub struct Form<P> {
    kind: DrawKind,
    scenes: Vec<FormScene>,
    part: PhantomData<fn() -> P>,
}

impl<P> Form<P> {
    pub fn kind(&self) -> DrawKind {
        self.kind
    }

    pub fn draws_in(&self, scene: &str) -> bool {
        self.scenes
            .iter()
            .any(|known| known.pass.scene.name() == scene && known.plain.is_some())
    }

    pub fn passes(&self) -> impl Iterator<Item = FormPass> + '_ {
        self.scenes.iter().map(|known| known.pass)
    }
}

struct Fragment {
    entry: Entry,
    faces: Faces,
    cull: Option<wgpu::Face>,
}

fn generic_name(role: PassRole, extra: bool) -> &'static str {
    match (role, extra) {
        (PassRole::Lit, false) => "fs_kind_lit",
        (PassRole::Lit, true) => "fs_kind_lit_extra",
        (PassRole::Blended, false) => "fs_kind_blended",
        (PassRole::Blended, true) => "fs_kind_blended_extra",
        (PassRole::Shown, false) => "fs_kind_shown",
        (PassRole::Shown, true) => "fs_kind_shown_extra",
        (PassRole::Depth, false) => "fs_kind_masked_depth",
        (PassRole::Depth, true) => "fs_kind_masked_depth_extra",
    }
}

fn object_name(role: PassRole, extra: bool) -> &'static str {
    match (role, extra) {
        (PassRole::Lit, false) => "fs_kind_object_lit",
        (PassRole::Lit, true) => "fs_kind_object_lit_extra",
        (PassRole::Blended, false) => "fs_kind_object_blended",
        (PassRole::Blended, true) => "fs_kind_object_blended_extra",
        (PassRole::Shown, false) => "fs_kind_object_shown",
        (PassRole::Shown, true) => "fs_kind_object_shown_extra",
        (PassRole::Depth, false) => "fs_kind_object_masked_depth",
        (PassRole::Depth, true) => "fs_kind_object_masked_depth_extra",
    }
}

fn fragment_use(role: PassRole, fades: bool) -> FragmentUse {
    match (role, fades) {
        (PassRole::Depth, false) => FragmentUse::Masked,
        (PassRole::Depth, true) | (PassRole::Lit | PassRole::Blended | PassRole::Shown, _) => {
            FragmentUse::Always
        }
    }
}

fn fragment_of(role: PassRole, varyings: Varyings, views: PassViews) -> Option<Fragment> {
    let (name, fades) = match varyings {
        Varyings::Surface => (generic_name(role, false), false),
        Varyings::SurfaceExtra => (generic_name(role, true), false),
        Varyings::Object => (object_name(role, false), false),
        Varyings::ObjectExtra => (object_name(role, true), false),
        Varyings::FadingObject => (object_name(role, false), true),
        Varyings::FadingObjectExtra => (object_name(role, true), true),
        Varyings::Own(own) => {
            let side = match views {
                PassViews::Owned => &own.owned,
                PassViews::Every | PassViews::Main => &own.placed,
            };
            return side.of(role).map(|entry| Fragment {
                entry,
                faces: Faces::Declared,
                cull: own.cull.face(),
            });
        }
    };
    Some(Fragment {
        entry: Entry::generated(crate::shaders::core_module(KINDS_MODULE), name),
        faces: Faces::Material(fragment_use(role, fades)),
        cull: Some(wgpu::Face::Back),
    })
}

fn primitive(
    topology: FormTopology,
    mirrored: bool,
    cull: Option<wgpu::Face>,
) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: topology.primitive(),
        front_face: if mirrored {
            wgpu::FrontFace::Cw
        } else {
            wgpu::FrontFace::Ccw
        },
        cull_mode: cull,
        ..Default::default()
    }
}

fn written(bias: DepthBias) -> DepthUse {
    DepthUse::Write {
        bias: wgpu::DepthBiasState {
            constant: bias.constant,
            slope_scale: bias.slope,
            clamp: 0.0,
        },
    }
}

fn kind_program(
    form: &KindForm,
    (role, views): (PassRole, PassViews),
    (mirrored, topology): (bool, FormTopology),
) -> Option<Program> {
    let side = form.side(views);
    let fragment = fragment_of(role, form.varyings, views)?;
    let writes = match (role, views) {
        (PassRole::Depth, PassViews::Owned) => wgpu::ColorWrites::empty(),
        (PassRole::Depth, PassViews::Every | PassViews::Main)
        | (
            PassRole::Lit | PassRole::Blended | PassRole::Shown,
            PassViews::Every | PassViews::Main | PassViews::Owned,
        ) => wgpu::ColorWrites::ALL,
    };
    Some(Program::Render(RenderDecl {
        vertex: side.vertex,
        fragment: Some(fragment.entry),
        vertices: side.streams.to_vec(),
        primitive: primitive(topology, mirrored, fragment.cull),
        depth: match role {
            PassRole::Blended => DepthUse::Test,
            PassRole::Lit | PassRole::Depth | PassRole::Shown => written(side.depth_bias),
        },
        blend: (role == PassRole::Shown).then_some(DEBUG_BLEND),
        writes,
        targets: None,
        samples: 1,
        faces: fragment.faces,
    }))
}

fn pass_program(
    form: &KindForm,
    own: &PassFragment,
    views: PassViews,
    (mirrored, topology): (bool, FormTopology),
) -> Program {
    let side = own.side.unwrap_or_else(|| form.side(views));
    Program::Render(RenderDecl {
        vertex: side.vertex,
        fragment: Some(own.fragment),
        vertices: side.streams.to_vec(),
        primitive: primitive(topology, mirrored, own.cull.face()),
        depth: match own.depth {
            FormDepth::Written => written(side.depth_bias),
            FormDepth::Tested => DepthUse::Test,
        },
        blend: None,
        writes: wgpu::ColorWrites::ALL,
        targets: None,
        samples: 1,
        faces: Faces::Declared,
    })
}

impl<P: GpuPart> Install<'_, P> {
    pub fn form(&mut self, form: &KindForm, bound: &[(Var, Bind<P>)]) -> Result<Form<P>, Error> {
        self.form_built(form, (&EVERY_ROLE, &[]), FormTopology::List, bound)
    }

    pub fn form_for(
        &mut self,
        form: &KindForm,
        roles: &[PassRole],
        bound: &[(Var, Bind<P>)],
    ) -> Result<Form<P>, Error> {
        self.form_built(form, (roles, &[]), FormTopology::List, bound)
    }

    pub fn form_shaped(
        &mut self,
        form: &KindForm,
        topology: FormTopology,
        bound: &[(Var, Bind<P>)],
    ) -> Result<Form<P>, Error> {
        self.form_built(form, (&EVERY_ROLE, &[]), topology, bound)
    }

    pub fn form_owning(
        &mut self,
        form: &KindForm,
        owns: &[PassFragment],
        bound: &[(Var, Bind<P>)],
    ) -> Result<Form<P>, Error> {
        self.form_built(form, (&EVERY_ROLE, owns), FormTopology::List, bound)
    }

    fn form_built(
        &mut self,
        form: &KindForm,
        (roles, owns): (&[PassRole], &[PassFragment]),
        topology: FormTopology,
        bound: &[(Var, Bind<P>)],
    ) -> Result<Form<P>, Error> {
        let bound = bound_list(bound, &self.core.outputs)?;
        let listed: Vec<engine_core::ScenePass> = self
            .core
            .passes
            .listed()
            .iter()
            .filter(|listed| listed.bit.is_some() && self.core.passes.takes(listed, form.kind))
            .copied()
            .collect();
        let joined =
            |own: &&PassFragment| listed.iter().any(|listed| listed.name == own.scene.name());
        if let Some(stray) = owns.iter().find(|own| !joined(own)) {
            let path = format!("forms.{}.owns", form.kind.0);
            return Err(Error::new(ErrorCode::PassMismatch, path).got(stray.scene.name()));
        }
        let mut scenes = Vec::with_capacity(listed.len());
        for listed in listed {
            let facts = (listed.pass.role(), listed.pass.views);
            let scope = Scoped::Scene(listed.name);
            let owned = match listed.pass.views {
                PassViews::Owned => true,
                PassViews::Every | PassViews::Main => false,
            };
            let own = owns.iter().find(|own| own.scene.name() == listed.name);
            let built = own.is_some() || roles.contains(&facts.0);
            let program_of = |mirrored: bool| match own {
                Some(own) => Some(pass_program(form, own, facts.1, (mirrored, topology))),
                None => kind_program(form, facts, (mirrored, topology)),
            };
            let plain = match program_of(false).filter(|_| built) {
                Some(program) => {
                    let reached = self.reached(&program, &bound);
                    Some(self.core.pipeline(scope, program, reached)?)
                }
                None => None,
            };
            let mirrored = match program_of(true).filter(|_| built && owned) {
                Some(program) => {
                    let reached = self.reached(&program, &bound);
                    Some(self.core.pipeline(scope, program, reached)?)
                }
                None => None,
            };
            scenes.push(FormScene {
                pass: FormPass {
                    scene: Scene::named(listed.name),
                    role: facts.0,
                    owned,
                },
                plain,
                mirrored,
            });
        }
        Ok(Form {
            kind: form.kind,
            scenes,
            part: PhantomData,
        })
    }

    fn reached(
        &mut self,
        program: &Program,
        bound: &[(Var, crate::install::Bound)],
    ) -> Vec<(Var, crate::install::Bound)> {
        let entries: Vec<Entry> = match program {
            Program::Render(render) => std::iter::once(render.vertex)
                .chain(render.fragment)
                .collect(),
            Program::Compute(entry) => vec![*entry],
        };
        bound
            .iter()
            .filter(|(var, _)| entries.iter().any(|entry| self.core.reaches(*entry, *var)))
            .cloned()
            .collect()
    }

    pub fn form_as<N: PassName>(
        &mut self,
        pass: Pass<P>,
        form: &KindForm,
        bound: &[(Var, Bind<P>)],
    ) -> Result<Pipeline<P, Own>, Error> {
        let bound = bound_list(bound, &self.core.outputs)?;
        let facts = (N::PASS.role(), N::PASS.views);
        let program = kind_program(form, facts, (false, FormTopology::List)).ok_or_else(|| {
            Error::new(ErrorCode::PassMismatch, format!("forms.{}", form.kind.0)).got(N::NAME)
        })?;
        let pipeline = self.core.pipeline(Scoped::Pass(pass.0), program, bound)?;
        Ok(Pipeline(pipeline, PhantomData))
    }
}

impl<P: GpuPart> Draw<'_, P, Scene> {
    pub fn form(&mut self, form: &Form<P>) -> Option<Pipeline<P, Scene>> {
        let point = self.raw.run().point?;
        let found = form
            .scenes
            .iter()
            .find(|known| known.pass.scene.name() == point)?;
        let pipeline = if self.mirrored {
            found.mirrored.or(found.plain)
        } else {
            found.plain
        }?;
        Some(Pipeline(pipeline, PhantomData))
    }

    pub fn skip(&mut self, reason: &'static str) -> u32 {
        self.raw.run().tallies.kinds.skip(reason);
        0
    }
}
