use engine::bricks::{AddBrick, BrickRegistry};
use engine_render::bindings::{GroupRef, SharedGroups, TABLES, Table};
use engine_render::composite::POST_PLACES;
use engine_render::faces::{FACE_ROOT, face_module, face_of};
use engine_render::gpu_layouts::{Constant, Layout, constants, layouts};
use engine_render::linker::{Linker, ShaderModule};
use engine_render::parts::{CORE_BRICK, PartEntry, installed_parts, shared_of};
use engine_render::pieces::{PIECES_MODULE, PieceDecl, pieces_source, place};
use engine_render::pipeline_layouts::{PIPELINE_LAYOUTS, PipelineLayout};
use engine_render::places::{Answered, PLACE_OWNER, PLACE_ROOT, place_module, place_of, places_of};
use engine_render::render_bricks::render_bricks;
use engine_render::shaders::MODULES;

const GENERATED: &str = "generated::";

pub struct Installed {
    pub sources: Vec<ShaderModule>,
    pub nobody: Vec<&'static str>,
    answers: Vec<(&'static str, String)>,
    pub layouts: Vec<(&'static str, &'static PipelineLayout)>,
    pub tables: Vec<(&'static str, &'static Table)>,
    pub mirrors: Vec<&'static Layout>,
    pub constants: Vec<&'static Constant>,
    pub shared: SharedGroups,
    pub pieces: Vec<(&'static str, PieceDecl)>,
    pub crates: Vec<&'static str>,
    placed: Vec<(&'static str, &'static str)>,
    answering: Vec<(&'static str, &'static str)>,
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

pub fn generated_face(module: &ShaderModule) -> bool {
    module.brick == CORE_BRICK && face_of(module.path).is_some()
}

pub fn contract_module(module: &ShaderModule) -> bool {
    module.brick == PLACE_OWNER && place_of(module.path).is_some()
}

fn core_place(name: &str) -> bool {
    POST_PLACES.iter().any(|place| place.name == name)
}

fn rooted(text: &str, root: &str) -> Vec<String> {
    let prefix = format!("import package::{root}::");
    let mut found: Vec<String> = Vec::new();
    for line in text.lines() {
        let Some(path) = line.trim_start().strip_prefix(prefix.as_str()) else {
            continue;
        };
        let name: String = path
            .chars()
            .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
            .collect();
        if !name.is_empty() && !found.contains(&name) {
            found.push(name);
        }
    }
    found
}

fn placed(registry: &BrickRegistry) -> Vec<(&'static str, &'static str)> {
    registry
        .bricks()
        .iter()
        .flat_map(|brick| {
            brick
                .fills
                .iter()
                .map(|filled| filled.place)
                .chain(brick.joins.iter().map(|joined| joined.place))
                .map(move |place| (brick.name, place))
        })
        .collect()
}

impl Installed {
    fn of(registry: &BrickRegistry) -> Result<Self, String> {
        let parts = installed_parts(registry);
        let mut installed = Self {
            sources: MODULES.to_vec(),
            nobody: Vec::new(),
            answers: Vec::new(),
            layouts: PIPELINE_LAYOUTS.to_vec(),
            tables: TABLES.to_vec(),
            mirrors: layouts(),
            constants: constants().collect(),
            shared: shared_of(&parts).map_err(|error| error.to_string())?,
            pieces: crate::bricks::pieces(registry),
            crates: Vec::new(),
            placed: placed(registry),
            answering: Vec::new(),
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
        installed.link_pieces()?;
        installed.link_places(registry)?;
        installed.needs(registry, &parts)?;
        installed.core_generated(&parts)?;
        installed.answered(registry)?;
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

    fn link_places(&mut self, registry: &BrickRegistry) -> Result<(), String> {
        let places = places_of(registry).map_err(|error| error.to_string())?;
        self.linker
            .add_places(&places)
            .map_err(|error| error.to_string())?;
        for linked in &places {
            let name = linked.place.name;
            if let Some(contract) = linked.place.face.wgsl {
                self.sources.push(ShaderModule {
                    path: leaked(place_module(name)),
                    brick: PLACE_OWNER,
                    source: contract,
                });
            }
            if let Some(source) = &linked.source {
                self.sources
                    .push(generated(face_module(name), source.clone()));
            }
            match &linked.answered {
                Some(Answered::Nobody) => self.nobody.push(name),
                Some(Answered::Filler { module, .. }) => {
                    self.answering.push((*module, name));
                    self.answers.extend(
                        linked
                            .contract
                            .answers
                            .iter()
                            .map(|answer| (*module, answer.name.clone())),
                    );
                }
                Some(Answered::Members(members)) => {
                    for (_, member) in members {
                        self.answering.push((member.module, name));
                        self.answers.extend(
                            member
                                .functions
                                .iter()
                                .map(|function| (member.module, (*function).to_owned())),
                        );
                    }
                }
                None => {}
            }
        }
        Ok(())
    }

    fn answered(&self, registry: &BrickRegistry) -> Result<(), String> {
        for need in self.linker.face_needs() {
            let path = face_module(&need.face);
            if !self.linker.paths().any(|known| known == path) {
                return Err(registry.face_unanswered(&need.face, need.brick).to_string());
            }
        }
        Ok(())
    }

    fn needs(&self, registry: &BrickRegistry, parts: &[&'static PartEntry]) -> Result<(), String> {
        for part in parts {
            for &name in part.decl.needs {
                let answered = registry.filled(name).is_some()
                    || !registry.joined(name).is_empty()
                    || self.linker.paths().any(|path| face_of(path) == Some(name));
                if !answered {
                    return Err(registry.face_unanswered(name, part.decl.brick).to_string());
                }
            }
        }
        Ok(())
    }

    fn core_generated(&mut self, parts: &[&'static PartEntry]) -> Result<(), String> {
        let made = engine_render::generated::generated(parts).map_err(|error| error.to_string())?;
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

    pub fn answers_for(&self, module: &str, item: &str) -> bool {
        self.answers
            .iter()
            .any(|(answering, name)| *answering == module && name == item)
    }

    fn brick_of(&self, path: &str) -> Option<&'static str> {
        self.sources
            .iter()
            .find(|module| module.path == path)
            .map(|module| module.brick)
    }

    fn known(&self, path: &str) -> bool {
        self.sources.iter().any(|module| module.path == path)
    }

    fn holds(&self, brick: &str, place: &str) -> bool {
        self.placed
            .iter()
            .any(|&(holder, held)| holder == brick && held == place)
    }

    fn answers_place(&self, module: &str, place: &str) -> bool {
        self.answering
            .iter()
            .any(|&(answering, answered)| answering == module && answered == place)
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

    pub fn import_rules(&self) -> Vec<String> {
        let mut problems = Vec::new();
        for module in &self.sources {
            if generated_path(module.path) || generated_face(module) || contract_module(module) {
                continue;
            }
            let faces = rooted(module.source, FACE_ROOT);
            let contracts = rooted(module.source, PLACE_ROOT);
            if core(module.path) {
                for name in faces.iter().chain(&contracts) {
                    if !core_place(name) {
                        problems.push(format!(
                            "core module {} imports the place {name}: core reads only the places \
                             core declares (core_uses_brick)",
                            module.path
                        ));
                    }
                }
                continue;
            }
            for name in &faces {
                if self.answers_place(module.path, name) {
                    problems.push(format!(
                        "{} of {} imports face::{name}, the face it answers: it imports the \
                         place's types from place::{name} (brick_internal_path)",
                        module.path, module.brick
                    ));
                }
            }
            for name in &contracts {
                if !self.holds(module.brick, name) {
                    problems.push(format!(
                        "{} of {} imports place::{name}, a place its brick neither fills nor \
                         joins: it reads the place through face::{name} (brick_internal_path)",
                        module.path, module.brick
                    ));
                }
            }
            for imported in self.imports(module.source) {
                if face_of(&imported).is_some() || place_of(&imported).is_some() {
                    continue;
                }
                let Some(owner) = self.brick_of(&imported) else {
                    continue;
                };
                if core(&imported) || generated_path(&imported) || owner == module.brick {
                    continue;
                }
                problems.push(format!(
                    "{} of {} imports {imported} of {owner}: a brick imports only face::*, core \
                     and generated::* modules, its own modules, and place::* for the places it \
                     fills or joins (brick_internal_path)",
                    module.path, module.brick
                ));
            }
        }
        problems
    }
}
