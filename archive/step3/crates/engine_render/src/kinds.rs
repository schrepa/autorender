pub(crate) const SURFACED_MODULE: &str = "generated::surfaced";

const SURFACED_IMPORTS: [&str; 1] =
    ["import package::generated::writes::{Outputs, blended_outputs, outputs, point_shaded};"];

    Object(FragmentUse),
    ObjectExtra(FragmentUse),
    Own(&'static OwnFragments),
    Surfaced(&'static KindSurface),

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
}

impl RoleFragments {
    pub const NONE: Self = Self {
        lit: None,
        depth: None,
        blended: None,
    };

    fn of(&self, role: PassRole) -> Option<Entry> {
        match role {
            PassRole::Lit => self.lit,
            PassRole::Depth => self.depth,
            PassRole::Blended => self.blended,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OwnFragments {
    pub placed: RoleFragments,
    pub cull: FaceCull,
}

impl OwnFragments {
    pub const NONE: Self = Self {
        placed: RoleFragments::NONE,
        cull: FaceCull::Back,
    };
}

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

fn object_name(role: PassRole, extra: bool) -> &'static str {
    match (role, extra) {
        (PassRole::Lit, false) => "fs_kind_object_lit",
        (PassRole::Lit, true) => "fs_kind_object_lit_extra",
        (PassRole::Blended, false) => "fs_kind_object_blended",
        (PassRole::Blended, true) => "fs_kind_object_blended_extra",
        (PassRole::Depth, false) => "fs_kind_object_masked_depth",
        (PassRole::Depth, true) => "fs_kind_object_masked_depth_extra",
    }
}

fn own_fragment(own: &OwnFragments, role: PassRole) -> Option<Fragment> {
    own.placed.of(role).map(|entry| Fragment {
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
        PassRole::Depth => return None,
    };
    Some(Fragment {
        entry,
        faces: Faces::Declared,
        cull: own.cull.face(),
    })
}

        Varyings::Object(depth) => (object_name(role, false), depth),
        Varyings::ObjectExtra(depth) => (object_name(role, true), depth),
        Varyings::Own(own) => return own_fragment(own, role),
        Varyings::Surfaced(surface) => {
            return own_fragment(&surface.own, role)
                .or_else(|| surfaced_fragment(&surface.own, role, surfaced));
        }

    pub fn speedup_form_owning(
        &mut self,
        form: &KindForm,
        owns: &[PassFragment],
        bound: &[(Var, Bind<P>)],
    ) -> Result<Form<P>, Error> {
        let built = self.form_owning(form, owns, bound)?;
        for pipeline in built.scenes.iter().filter_map(|scene| scene.plain) {
            self.core.speedup_only(pipeline);
        }
        Ok(built)
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
            let _ = writeln!(text, "    return {written}(seen.point, shaded);");
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

    pub fn per_pass<'t, T>(&mut self, table: &'t PerPass<T>) -> Option<&'t T> {
        let point = self.raw.run().point?;
        table.named(point)
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
