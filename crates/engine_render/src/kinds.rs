use std::marker::PhantomData;

use engine_core::{DrawKind, Entry, Error, PassName, PassRole, PassViews, Var};

use crate::install::{Faces, FragmentUse, Program, RenderDecl, Scoped};
use crate::{
    Bind, DEBUG_BLEND, DepthUse, Draw, GpuPart, Install, Own, Pass, Pipeline, Scene, VertexLayout,
    bound_list,
};

pub(crate) const KINDS_MODULE: &str = "engine::kinds";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Varyings {
    Surface,
    SurfaceExtra,
}

#[derive(Clone, Copy, Debug)]
pub struct FormSide {
    pub vertex: Entry,
    pub streams: &'static [VertexLayout],
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

#[derive(Clone, Copy, Debug)]
pub struct FormPass {
    pub scene: Scene,
    pub role: PassRole,
    pub owned: bool,
}

#[derive(Clone, Copy, Debug)]
struct FormScene {
    pass: FormPass,
    plain: u32,
    mirrored: Option<u32>,
}

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
        self.scenes.iter().any(|known| known.pass.scene.name() == scene)
    }

    pub fn passes(&self) -> impl Iterator<Item = FormPass> + '_ {
        self.scenes.iter().map(|known| known.pass)
    }
}

fn fragment_of(role: PassRole, varyings: Varyings) -> (Entry, FragmentUse) {
    let (name, uses) = match (role, varyings) {
        (PassRole::Lit, Varyings::Surface) => ("fs_kind_lit", FragmentUse::Always),
        (PassRole::Lit, Varyings::SurfaceExtra) => ("fs_kind_lit_extra", FragmentUse::Always),
        (PassRole::Blended, Varyings::Surface) => ("fs_kind_blended", FragmentUse::Always),
        (PassRole::Blended, Varyings::SurfaceExtra) => {
            ("fs_kind_blended_extra", FragmentUse::Always)
        }
        (PassRole::Shown, Varyings::Surface) => ("fs_kind_shown", FragmentUse::Always),
        (PassRole::Shown, Varyings::SurfaceExtra) => ("fs_kind_shown_extra", FragmentUse::Always),
        (PassRole::Depth, Varyings::Surface) => ("fs_kind_masked_depth", FragmentUse::Masked),
        (PassRole::Depth, Varyings::SurfaceExtra) => {
            ("fs_kind_masked_depth_extra", FragmentUse::Masked)
        }
    };
    (
        Entry::generated(crate::shaders::core_module(KINDS_MODULE), name),
        uses,
    )
}

fn kind_program(
    form: &KindForm,
    (role, views): (PassRole, PassViews),
    mirrored: bool,
) -> Program {
    let side = form.side(views);
    let (fragment, uses) = fragment_of(role, form.varyings);
    let writes = match (role, views) {
        (PassRole::Depth, PassViews::Owned) => wgpu::ColorWrites::empty(),
        (PassRole::Depth, PassViews::Every | PassViews::Main)
        | (
            PassRole::Lit | PassRole::Blended | PassRole::Shown,
            PassViews::Every | PassViews::Main | PassViews::Owned,
        ) => wgpu::ColorWrites::ALL,
    };
    Program::Render(RenderDecl {
        vertex: side.vertex,
        fragment: Some(fragment),
        vertices: side.streams.to_vec(),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: if mirrored {
                wgpu::FrontFace::Cw
            } else {
                wgpu::FrontFace::Ccw
            },
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth: DepthUse::Write {
            bias: wgpu::DepthBiasState {
                constant: 0,
                slope_scale: 0.0,
                clamp: 0.0,
            },
        },
        blend: (role == PassRole::Shown).then_some(DEBUG_BLEND),
        writes,
        targets: None,
        samples: 1,
        faces: Faces::Material(uses),
    })
}

impl<P: GpuPart> Install<'_, P> {
    pub fn form(&mut self, form: &KindForm, bound: &[(Var, Bind<P>)]) -> Result<Form<P>, Error> {
        let bound = bound_list(bound, &self.core.outputs)?;
        let listed: Vec<engine_core::ScenePass> = self
            .core
            .passes
            .listed()
            .iter()
            .filter(|listed| listed.bit.is_some() && self.core.passes.takes(listed, form.kind))
            .copied()
            .collect();
        let mut scenes = Vec::with_capacity(listed.len());
        for listed in listed {
            let facts = (listed.pass.role(), listed.pass.views);
            let scope = Scoped::Scene(listed.name);
            let plain = self
                .core
                .pipeline(scope, kind_program(form, facts, false), bound.clone())?;
            let owned = match listed.pass.views {
                PassViews::Owned => true,
                PassViews::Every | PassViews::Main => false,
            };
            let mirrored = if owned {
                Some(
                    self.core
                        .pipeline(scope, kind_program(form, facts, true), bound.clone())?,
                )
            } else {
                None
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

    pub fn form_as<N: PassName>(
        &mut self,
        pass: Pass<P>,
        form: &KindForm,
        bound: &[(Var, Bind<P>)],
    ) -> Result<Pipeline<P, Own>, Error> {
        let bound = bound_list(bound, &self.core.outputs)?;
        let facts = (N::PASS.role(), N::PASS.views);
        let pipeline =
            self.core
                .pipeline(Scoped::Pass(pass.0), kind_program(form, facts, false), bound)?;
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
        let pipeline = found
            .mirrored
            .filter(|_| self.mirrored)
            .unwrap_or(found.plain);
        Some(Pipeline(pipeline, PhantomData))
    }

    pub fn skip(&mut self, reason: &'static str) -> u32 {
        let run = self.raw.run();
        let point = run.point;
        run.tallies.kinds.skip(point, reason);
        0
    }
}
