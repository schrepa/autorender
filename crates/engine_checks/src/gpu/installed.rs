use std::collections::BTreeSet;

use engine::bricks::{AddBrick, BrickRegistry};
use engine_render::bindings::{GroupRef, SharedGroups, TABLES, Table};
use engine_render::faces::{face_module, face_of};
use engine_render::gpu_layouts::{Constant, Layout, constants, layouts};
use engine_render::linker::{Linker, ShaderModule, face_imports};
use engine_render::parts::{CORE_BRICK, PartEntry, installed_parts, shared_of};
use engine_render::pieces::{PIECES_MODULE, PieceDecl, pieces_source, place};
use engine_render::pipeline_layouts::{PIPELINE_LAYOUTS, PipelineLayout};
use engine_render::render_bricks::render_bricks;
use engine_render::shaders::MODULES;

const GENERATED: &str = "generated::";

const PIECES_IMPORT: &str = "package::generated::pieces";

pub struct Installed {
    pub sources: Vec<ShaderModule>,
    pub defaults: Vec<ShaderModule>,
    pub defaulted: Vec<String>,
    pub layouts: Vec<(&'static str, &'static PipelineLayout)>,
    pub tables: Vec<(&'static str, &'static Table)>,
    pub mirrors: Vec<&'static Layout>,
    pub constants: Vec<&'static Constant>,
    pub shared: SharedGroups,
    pub pieces: Vec<(&'static str, PieceDecl)>,
    pub crates: Vec<&'static str>,
    pub(super) linker: Linker,
}

fn leaked(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

fn generated(path: String, source: String) -> ShaderModule {
    ShaderModule {
        path: leaked(path),
        brick: CORE_BRICK,
        source: leaked(source),
    }
}

pub fn installed() -> Result<Installed, String> {
    let mut engine = engine::Engine::default();
    engine.app.add_bricks(engine::bricks::standard());
    let registry = engine
        .app
        .world()
        .get_resource::<BrickRegistry>()
        .ok_or("the standard engine registers no brick")?;
    Installed::of(registry)
}

pub fn core(path: &str) -> bool {
    MODULES.iter().any(|module| module.path == path)
}

pub fn generated_path(path: &str) -> bool {
    path.starts_with(GENERATED)
}

impl Installed {
    fn of(registry: &BrickRegistry) -> Result<Self, String> {
        let parts = installed_parts(registry);
        let mut installed = Self {
            sources: MODULES.to_vec(),
            defaults: Vec::new(),
            defaulted: Vec::new(),
            layouts: PIPELINE_LAYOUTS.to_vec(),
            tables: TABLES.to_vec(),
            mirrors: layouts(),
            constants: constants().collect(),
            shared: shared_of(&parts).map_err(|error| error.to_string())?,
            pieces: crate::bricks::pieces(registry),
            crates: Vec::new(),
            linker: Linker::new(MODULES),
        };
        for (name, render) in render_bricks(registry) {
            let modules: Vec<ShaderModule> = render.modules().copied().collect();
            installed
                .linker
                .add_modules(&modules)
                .map_err(|error| error.to_string())?;
            installed.sources.extend(modules);
            for &listed in &render.layouts {
                installed.layouts.extend(listed.iter().copied());
            }
            for &mirrored in &render.mirrors {
                installed.mirrors.extend(mirrored);
            }
            for &declared in &render.constants {
                installed.constants.extend(declared);
            }
            if let Some(brick) = registry.brick(name) {
                installed.crates.push(brick.crate_name);
            }
        }
        for (_, render) in render_bricks(registry) {
            let defaults: Vec<ShaderModule> = render.defaults().copied().collect();
            installed
                .linker
                .add_defaults(&defaults)
                .map_err(|error| error.to_string())?;
            for default in defaults {
                if !installed
                    .defaults
                    .iter()
                    .any(|known| known.path == default.path)
                {
                    installed.defaults.push(default);
                }
            }
        }
        installed.link_pieces()?;
        installed.defaulted = installed
            .linker
            .resolve_faces()
            .map_err(|need| registry.face_unanswered(&need.face, need.brick).to_string())?;
        for face in &installed.defaulted {
            let path = face_module(face);
            if let Some(default) = installed.defaults.iter().find(|known| known.path == path) {
                installed.sources.push(*default);
            }
        }
        installed.needs(registry, &parts)?;
        installed.core_generated(&parts)?;
        installed.own_tables();
        Ok(installed)
    }

    fn link_pieces(&mut self) -> Result<(), String> {
        place(&self.pieces).map_err(|error| error.to_string())?;
        let decls: Vec<PieceDecl> = self.pieces.iter().map(|&(_, piece)| piece).collect();
        let module = generated(PIECES_MODULE.to_owned(), pieces_source(&decls));
        self.linker.add_generated(
            module.path.to_owned(),
            module.brick,
            module.source.to_owned(),
        );
        self.sources.push(module);
        Ok(())
    }

    fn needs(&self, registry: &BrickRegistry, parts: &[&'static PartEntry]) -> Result<(), String> {
        let mut offered: BTreeSet<String> = parts
            .iter()
            .flat_map(|part| part.decl.faces)
            .map(|offer| offer.name().to_string())
            .collect();
        offered.extend(self.linker.paths().filter_map(face_of).map(str::to_string));
        for part in parts {
            for &name in part.decl.needs {
                if registry.filler(name).is_none() && !offered.contains(name) {
                    return Err(registry.face_unanswered(name, part.decl.brick).to_string());
                }
            }
        }
        Ok(())
    }

    fn core_generated(&mut self, parts: &[&'static PartEntry]) -> Result<(), String> {
        let made =
            engine_render::generated::generated(parts).map_err(|error| error.to_string())?;
        for made in made {
            let module = generated(made.path, made.source);
            self.linker.add_generated(
                module.path.to_owned(),
                module.brick,
                module.source.to_owned(),
            );
            self.sources.push(module);
        }
        Ok(())
    }

    fn own_tables(&mut self) {
        for &(_, layout) in &self.layouts {
            for group in layout.groups {
                let GroupRef::Own(table) = *group else {
                    continue;
                };
                if !self
                    .tables
                    .iter()
                    .any(|(_, known)| known.label == table.label)
                {
                    self.tables.push((table.label, table));
                }
            }
        }
    }

    fn brick_of(&self, path: &str) -> Option<&'static str> {
        self.sources
            .iter()
            .find(|module| module.path == path)
            .map(|module| module.brick)
    }

    fn known(&self, path: &str) -> bool {
        self.sources
            .iter()
            .chain(&self.defaults)
            .any(|module| module.path == path)
    }

    pub fn imports(&self, source: &str) -> Vec<String> {
        let mut found: Vec<String> = Vec::new();
        for line in source.lines() {
            let Some(rest) = line.trim_start().strip_prefix("import package::") else {
                continue;
            };
            let path = rest
                .split(['{', ';', ' '])
                .next()
                .unwrap_or_default()
                .trim_end_matches("::");
            let segments: Vec<&str> = path.split("::").collect();
            let module = (1..=segments.len())
                .rev()
                .map(|count| segments[..count].join("::"))
                .find(|candidate| self.known(candidate))
                .unwrap_or_else(|| segments[..segments.len().saturating_sub(1)].join("::"));
            if !found.contains(&module) {
                found.push(module);
            }
        }
        found
    }

    fn is_default(&self, module: &ShaderModule) -> bool {
        self.defaults
            .iter()
            .any(|default| default.path == module.path && default.brick == module.brick)
    }

    pub fn import_rules(&self) -> Vec<String> {
        let mut problems = Vec::new();
        for module in &self.sources {
            if generated_path(module.path) || self.is_default(module) {
                continue;
            }
            if core(module.path) {
                for face in face_imports(module.source) {
                    problems.push(format!(
                        "core module {} imports face::{face}: core uses no faces \
                         (core_uses_brick)",
                        module.path
                    ));
                }
                continue;
            }
            for imported in self.imports(module.source) {
                let Some(owner) = self.brick_of(&imported) else {
                    continue;
                };
                if core(&imported)
                    || generated_path(&imported)
                    || face_of(&imported).is_some()
                    || owner == module.brick
                {
                    continue;
                }
                problems.push(format!(
                    "{} of {} imports {imported} of {owner}: a brick imports only face::*, core \
                     and generated::* modules of others (brick_internal_path)",
                    module.path, module.brick
                ));
            }
        }
        for default in &self.defaults {
            if default.source.contains(PIECES_IMPORT) {
                problems.push(format!(
                    "the default {} imports generated::pieces: a default returns constants",
                    default.path
                ));
            }
            for imported in self.imports(default.source) {
                if !core(&imported) {
                    problems.push(format!(
                        "the default {} imports {imported}: a default imports only core modules",
                        default.path
                    ));
                }
            }
        }
        problems
    }
}
