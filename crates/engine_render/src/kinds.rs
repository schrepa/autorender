use std::fmt::Write as _;
use std::marker::PhantomData;

use engine_core::{
    DepthBias, DrawKind, DropReason, Entry, Error, ErrorCode, Module, PassRole, PassViews, Var,
};

use crate::install::{Faces, FragmentUse, InstallCore, Program, RenderDecl, Scoped};
use crate::{
    Bind, DEBUG_BLEND, DepthUse, Draw, FaceCull, GpuPart, Install, Pipeline, Scene, VertexLayout,
    bound_list,
};

pub(crate) const KINDS_MODULE: &str = "engine::kinds";

pub(crate) const SURFACED_MODULE: &str = "generated::surfaced";

const SURFACED_IMPORTS: [&str; 1] =
    ["import package::generated::writes::{Outputs, blended_outputs, outputs, point_shaded};"];

#[derive(Clone, Copy, Debug)]
pub enum Varyings {
    Surface(FragmentUse),
    SurfaceExtra(FragmentUse),
    Object(FragmentUse),
    ObjectExtra(FragmentUse),
    Own(&'static OwnFragments),
    Surfaced(&'static KindSurface),
}

#[derive(Clone, Copy, Debug)]
pub struct KindSurface {
    pub module: Module,
    pub varyings: &'static str,
    pub point: &'static str,
    pub own: OwnFragments,
}

#[derive(Clone, Copy, Debug)]
struct SurfacedEntries {
    lit: Entry,
    blended: Entry,
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

impl OwnFragments {
    pub const NONE: Self = Self {
        placed: RoleFragments::NONE,
        owned: RoleFragments::NONE,
        cull: FaceCull::Back,
    };
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

#[derive(Clone, Copy, Debug)]
pub struct PassFragment {
    pub pass: Scene,
    pub side: Option<FormSide>,
    pub entry: Entry,
    pub cull: FaceCull,
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

pub struct PerPass<T> {
    rows: Vec<(&'static str, T)>,
}

impl<T> PerPass<T> {
    pub fn of(
        passes: impl IntoIterator<Item = FormPass>,
        mut make: impl FnMut(FormPass) -> T,
    ) -> Self {
        let mut rows: Vec<(&'static str, T)> = Vec::new();
        for pass in passes {
            let name = pass.scene.name();
            if rows.iter().any(|(known, _)| *known == name) {
                continue;
            }
            let row = make(pass);
            rows.push((name, row));
        }
        Self { rows }
    }

    pub fn named(&self, name: &str) -> Option<&T> {
        self.rows
            .iter()
            .find(|(known, _)| *known == name)
            .map(|(_, row)| row)
    }

    pub fn rows(&self) -> impl Iterator<Item = &T> + '_ {
        self.rows.iter().map(|(_, row)| row)
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

fn fragment_use(role: PassRole, depth: FragmentUse) -> FragmentUse {
    match role {
        PassRole::Depth => depth,
        PassRole::Lit | PassRole::Blended | PassRole::Shown => FragmentUse::Always,
    }
}

fn own_fragment(own: &OwnFragments, role: PassRole, views: PassViews) -> Option<Fragment> {
    let side = match views {
        PassViews::Owned => &own.owned,
        PassViews::Every | PassViews::Main => &own.placed,
    };
    side.of(role).map(|entry| Fragment {
        entry,
        faces: Faces::Declared,
        cull: own.cull.face(),
    })
}

fn surfaced_fragment(
    own: &OwnFragments,
    role: PassRole,
    surfaced: Option<SurfacedEntries>,
) -> Option<Fragment> {
    let surfaced = surfaced?;
    let entry = match role {
        PassRole::Lit => surfaced.lit,
        PassRole::Blended => surfaced.blended,
        PassRole::Depth | PassRole::Shown => return None,
    };
    Some(Fragment {
        entry,
        faces: Faces::Declared,
        cull: own.cull.face(),
    })
}

fn fragment_of(
    role: PassRole,
    varyings: Varyings,
    views: PassViews,
    surfaced: Option<SurfacedEntries>,
) -> Option<Fragment> {
    let (name, depth) = match varyings {
        Varyings::Surface(depth) => (generic_name(role, false), depth),
        Varyings::SurfaceExtra(depth) => (generic_name(role, true), depth),
        Varyings::Object(depth) => (object_name(role, false), depth),
        Varyings::ObjectExtra(depth) => (object_name(role, true), depth),
        Varyings::Own(own) => return own_fragment(own, role, views),
        Varyings::Surfaced(surface) => {
            return own_fragment(&surface.own, role, views)
                .or_else(|| surfaced_fragment(&surface.own, role, surfaced));
        }
    };
    Some(Fragment {
        entry: Entry::generated(crate::shaders::core_module(KINDS_MODULE), name),
        faces: Faces::Material(fragment_use(role, depth)),
        cull: Some(wgpu::Face::Back),
    })
}

fn primitive(mirrored: bool, cull: Option<wgpu::Face>) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        front_face: if mirrored {
            wgpu::FrontFace::Cw
        } else {
            wgpu::FrontFace::Ccw
        },
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

fn written(bias: DepthBias) -> DepthUse {
    DepthUse::Write {
        bias: depth_bias(bias),
    }
}

fn kind_program(
    (varyings, side): (Varyings, FormSide),
    (role, views): (PassRole, PassViews),
    mirrored: bool,
    surfaced: Option<SurfacedEntries>,
) -> Option<Program> {
    let fragment = fragment_of(role, varyings, views, surfaced)?;
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
        primitive: primitive(mirrored, fragment.cull),
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

fn pass_program(form: &KindForm, own: &PassFragment, views: PassViews, mirrored: bool) -> Program {
    let side = own.side.unwrap_or_else(|| form.side(views));
    Program::Render(RenderDecl {
        vertex: side.vertex,
        fragment: Some(own.entry),
        vertices: side.streams.to_vec(),
        primitive: primitive(mirrored, own.cull.face()),
        depth: DepthUse::Test,
        blend: None,
        writes: wgpu::ColorWrites::ALL,
        targets: None,
        samples: 1,
        faces: Faces::Declared,
    })
}

impl<P: GpuPart> Install<'_, P> {
    pub fn form(&mut self, form: &KindForm, bound: &[(Var, Bind<P>)]) -> Result<Form<P>, Error> {
        self.form_built(form, &[], bound)
    }

    pub fn form_owning(
        &mut self,
        form: &KindForm,
        owns: &[PassFragment],
        bound: &[(Var, Bind<P>)],
    ) -> Result<Form<P>, Error> {
        self.form_built(form, owns, bound)
    }

    pub fn speedup_form_owning(
        &mut self,
        form: &KindForm,
        owns: &[PassFragment],
        bound: &[(Var, Bind<P>)],
    ) -> Result<Form<P>, Error> {
        let built = self.form_owning(form, owns, bound)?;
        let scenes = built.scenes.iter().flat_map(|scene| [scene.plain, scene.mirrored]);
        for pipeline in scenes.flatten() {
            self.core.speedup_only(pipeline);
        }
        Ok(built)
    }

    fn form_built(
        &mut self,
        form: &KindForm,
        owns: &[PassFragment],
        bound: &[(Var, Bind<P>)],
    ) -> Result<Form<P>, Error> {
        let bound = bound_list(bound, &self.core.outputs)?;
        let surfaced = self.surfaced(form)?;
        let listed: Vec<engine_core::ScenePass> = self
            .core
            .passes
            .listed()
            .iter()
            .filter(|listed| listed.bit.is_some() && self.core.passes.takes(listed, form.kind))
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
            let facts = (listed.pass.role(), listed.pass.views);
            let scope = Scoped::Scene(listed.name);
            let owned = match listed.pass.views {
                PassViews::Owned => true,
                PassViews::Every | PassViews::Main => false,
            };
            let own = owns.iter().find(|own| own.pass.name() == listed.name);
            let program_of = |mirrored: bool| match own {
                Some(own) => Some(pass_program(form, own, facts.1, mirrored)),
                None => kind_program(
                    (form.varyings, form.side(facts.1)),
                    facts,
                    mirrored,
                    surfaced,
                ),
            };
            let plain = match program_of(false) {
                Some(program) => {
                    let reached = self.reached(&program, &bound);
                    Some(self.core.pipeline(scope, program, reached)?)
                }
                None => None,
            };
            let mirrored = match program_of(true).filter(|_| owned) {
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

    fn surfaced(&mut self, form: &KindForm) -> Result<Option<SurfacedEntries>, Error> {
        match form.varyings {
            Varyings::Surfaced(surface) => {
                surfaced_entries(self.core, form.kind, surface).map(Some)
            }
            Varyings::Surface(_)
            | Varyings::SurfaceExtra(_)
            | Varyings::Object(_)
            | Varyings::ObjectExtra(_)
            | Varyings::Own(_) => Ok(None),
        }
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
}

fn surfaced_name(path: &str, item: &str) -> String {
    format!("{}_{item}", path.replace("::", "_"))
}

fn surfaced_entry(name: String) -> Entry {
    Entry::generated(
        crate::shaders::core_module(SURFACED_MODULE),
        Box::leak(name.into_boxed_str()),
    )
}

fn surfaced_entries(
    core: &mut InstallCore,
    kind: DrawKind,
    surface: &KindSurface,
) -> Result<SurfacedEntries, Error> {
    let at = format!("forms.{}.surface", kind.0);
    let package = surface.module.source().package();
    if package != core.package {
        return Err(Error::new(ErrorCode::ForeignAccess, at).got(package));
    }
    let linker = &mut core.linker;
    linker.add_source(surface.module.source());
    let (path, point) = crate::linker::path_of((surface.module, surface.point));
    if !linker.defines(path, &point) {
        return Err(Error::new(ErrorCode::UnknownModule, at).got(surface.point));
    }
    let varyings = surfaced_name(path, surface.varyings);
    let shape = format!("surfaced_varyings_{varyings}");
    let named = surfaced_name(path, &point);
    let made = format!("surfaced_point_{named}");
    let lit = format!("fs_surfaced_lit_{named}");
    let blended = format!("fs_surfaced_blended_{named}");
    let held = linker.text(SURFACED_MODULE).unwrap_or_default().to_owned();
    let imports = SURFACED_IMPORTS.map(str::to_owned);
    let wanted = [
        format!("import package::{path}::{} as {shape};", surface.varyings),
        format!("import package::{path}::{point} as {made};"),
    ];
    let mut text = String::new();
    for line in imports.iter().chain(&wanted) {
        if !held.contains(line.as_str()) {
            let _ = writeln!(text, "{line}");
        }
    }
    text.push_str(&held);
    for (entry, written) in [(&lit, "outputs"), (&blended, "blended_outputs")] {
        if !linker.defines(SURFACED_MODULE, entry) {
            let _ = writeln!(text, "\n@fragment");
            let _ = writeln!(text, "fn {entry}(input: {shape}) -> Outputs {{");
            let _ = writeln!(text, "    let seen = {made}(input);");
            let _ = writeln!(
                text,
                "    let shaded = point_shaded(seen.point, seen.pixel);"
            );
            let _ = writeln!(text, "    return {written}(seen.point, shaded, seen.at);");
            let _ = writeln!(text, "}}");
        }
    }
    if text != held {
        linker.add_generated(
            SURFACED_MODULE.to_owned(),
            crate::linker::CORE_PACKAGE,
            text,
        );
    }
    linker.keep(path, &point);
    Ok(SurfacedEntries {
        lit: surfaced_entry(lit),
        blended: surfaced_entry(blended),
    })
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

    pub fn per_pass<'t, T>(&mut self, table: &'t PerPass<T>) -> Option<&'t T> {
        let point = self.raw.run().point?;
        table.named(point)
    }

    pub fn skip(&mut self, reason: &'static str) -> u32 {
        self.raw.run().tallies.kinds.skip(reason);
        0
    }

    pub fn wait(&mut self, on: &'static str) -> u32 {
        self.raw.run().tallies.kinds.wait(on);
        0
    }

    pub fn refuse(&mut self, on: &'static str, error: &Error) -> u32 {
        self.raw.run().tallies.kinds.refuse(on, error);
        0
    }

    pub fn drop_rows(&mut self, reason: DropReason, rows: u32) {
        if rows == 0 {
            return;
        }
        let part = self.raw.part();
        let run = self.raw.run();
        let pass = run.point.unwrap_or_default();
        run.tallies
            .kinds
            .drop_draws((part, pass), (reason, None), rows);
    }
}
