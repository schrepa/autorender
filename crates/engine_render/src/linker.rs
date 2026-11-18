use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Arc;

use engine_core::{Entry, Error, ErrorCode, Source, Var};
use wesl::Mangler as _;

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) mod counting;
mod driver;

pub const CORE_PACKAGE: &str = "engine_render";
pub const CAMERA: &str = "camera";
const CORE_BINDINGS: [&str; 1] = [CAMERA];
const GENERATED_ROOT: &str = "generated::";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sample {
    Float,
    Depth,
    Uint,
    Sint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ty {
    Uniform,
    Storage {
        write: bool,
    },
    Texture {
        sample: Sample,
        view: wgpu::TextureViewDimension,
    },
    StorageTexture {
        format: wgpu::TextureFormat,
        access: wgpu::StorageTextureAccess,
        view: wgpu::TextureViewDimension,
    },
    Sampler,
    Comparison,
}

impl Ty {
    pub(crate) fn binding(self, filterable: bool, dynamic: bool) -> wgpu::BindingType {
        let buffer = |ty| wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        };
        match self {
            Self::Uniform => buffer(wgpu::BufferBindingType::Uniform),
            Self::Storage { write } => {
                buffer(wgpu::BufferBindingType::Storage { read_only: !write })
            }
            Self::Texture { sample, view } => wgpu::BindingType::Texture {
                sample_type: match sample {
                    Sample::Float => wgpu::TextureSampleType::Float { filterable },
                    Sample::Depth => wgpu::TextureSampleType::Depth,
                    Sample::Uint => wgpu::TextureSampleType::Uint,
                    Sample::Sint => wgpu::TextureSampleType::Sint,
                },
                view_dimension: view,
                multisampled: false,
            },
            Self::StorageTexture {
                format,
                access,
                view,
            } => wgpu::BindingType::StorageTexture {
                access,
                format,
                view_dimension: view,
            },
            Self::Sampler => wgpu::BindingType::Sampler(if filterable {
                wgpu::SamplerBindingType::Filtering
            } else {
                wgpu::SamplerBindingType::NonFiltering
            }),
            Self::Comparison => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
        }
    }

    pub(crate) fn writes(self) -> bool {
        match self {
            Self::Storage { write } => write,
            Self::StorageTexture { access, .. } => access != wgpu::StorageTextureAccess::ReadOnly,
            Self::Uniform | Self::Texture { .. } | Self::Sampler | Self::Comparison => false,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Declaration {
    pub(crate) group: u32,
    pub(crate) binding: u32,
    pub(crate) name: String,
    pub(crate) ty: Ty,
}

struct Module {
    path: String,
    package: &'static str,
    text: Cow<'static, str>,
    declared: Vec<Declaration>,
    generated: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct Composed {
    pub(crate) module: String,
    pub(crate) text: String,
    pub(crate) bindings: Vec<String>,
    pub(crate) overrides: Vec<(String, String)>,
}

impl Composed {
    pub(crate) fn constants<'a>(&'a self, values: &[(&str, f64)]) -> Vec<(&'a str, f64)> {
        values
            .iter()
            .filter_map(|&(name, value)| {
                self.overrides
                    .iter()
                    .find(|(plain, _)| plain == name)
                    .map(|(_, mangled)| (mangled.as_str(), value))
            })
            .collect()
    }
}

pub(crate) type Part = (Option<&'static str>, &'static str);

#[derive(Default)]
pub(crate) struct Linker {
    modules: Vec<Module>,
    sources: Vec<&'static Source>,
    handed: Vec<(String, &'static str, Vec<Part>)>,
    numbers: BTreeMap<(String, String), u32>,
    composed: BTreeMap<(String, String), Arc<Composed>>,
    numbered: bool,
    edited: BTreeMap<String, String>,
    kept: Vec<(String, String)>,
    prepared: Option<Result<driver::Prepared, String>>,
    parsed: driver::Parsed,
    wholes: engine_core::collections::HashMap<String, driver::Slot>,
    bound: BTreeMap<(String, String), Option<Vec<String>>>,
}

pub(crate) fn path_of(var_or_entry: (engine_core::Module, &str)) -> (&'static str, String) {
    let (module, name) = var_or_entry;
    let path = module
        .source()
        .modules()
        .get(module.index() as usize)
        .map_or("", |text| text.path());
    (path, name.to_owned())
}

impl Linker {
    pub(crate) fn new(core: &'static Source) -> Self {
        Self::edited(core, BTreeMap::new())
    }

    pub(crate) fn edited(core: &'static Source, edited: BTreeMap<String, String>) -> Self {
        let mut linker = Self {
            edited,
            ..Self::default()
        };
        linker.add_source(core);
        linker
    }

    fn source_text(&self, module: &'static engine_core::ShaderText) -> Cow<'static, str> {
        self.edited
            .get(module.path())
            .map_or(Cow::Borrowed(module.text()), |text| {
                Cow::Owned(text.clone())
            })
    }

    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    pub(crate) fn files(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        self.sources
            .iter()
            .flat_map(|source| source.modules())
            .filter(|module| !module.file().is_empty())
            .map(|module| (module.path(), module.file()))
    }

    pub(crate) fn add_source(&mut self, source: &'static Source) {
        if self
            .sources
            .iter()
            .any(|known| std::ptr::eq(*known, source))
        {
            return;
        }
        self.sources.push(source);
        for module in source.modules() {
            let generated = module.path().starts_with(GENERATED_ROOT);
            let text = self.source_text(module);
            self.add(module.path().to_owned(), source.package(), text, generated);
            self.place_handed(module.path());
        }
        self.numbered = false;
        self.changed();
    }

    pub(crate) fn forget_wholes(&mut self) {
        self.wholes.clear();
    }

    fn changed(&mut self) {
        self.composed.clear();
        self.prepared = None;
        self.parsed.changed();
        self.wholes.clear();
        self.bound.clear();
    }

    pub(crate) fn hand(&mut self, path: String, package: &'static str, parts: Vec<Part>) {
        self.handed.retain(|(known, _, _)| *known != path);
        self.handed.push((path.clone(), package, parts));
        self.place_handed(&path);
    }

    fn place_handed(&mut self, path: &str) {
        let Some((_, package, parts)) = self.handed.iter().find(|(known, _, _)| known == path)
        else {
            return;
        };
        let package = *package;
        let base = self
            .sources
            .iter()
            .flat_map(|source| source.modules())
            .find(|module| module.path() == path)
            .map(|module| self.source_text(module));
        let text = handed_text(base.as_deref(), parts);
        let once = self
            .modules
            .iter()
            .filter(|module| module.path == path)
            .count()
            == 1;
        if !once || self.text(path) != Some(text.as_str()) {
            self.add_generated(path.to_owned(), package, text);
        }
    }

    pub(crate) fn add_generated(&mut self, path: String, package: &'static str, text: String) {
        self.modules.retain(|known| known.path != path);
        self.add(path, package, Cow::Owned(text), true);
        self.numbered = false;
        self.changed();
    }

    fn add(
        &mut self,
        path: String,
        package: &'static str,
        text: Cow<'static, str>,
        generated: bool,
    ) {
        let declared = declarations(&text);
        self.modules.push(Module {
            path,
            package,
            text,
            declared,
            generated,
        });
    }

    fn module(&self, path: &str) -> Option<&Module> {
        self.modules.iter().find(|module| module.path == path)
    }

    pub(crate) fn text(&self, path: &str) -> Option<&str> {
        self.module(path).map(|module| &*module.text)
    }

    pub(crate) fn sources(&self) -> &[&'static Source] {
        &self.sources
    }

    pub(crate) fn paths(&self) -> Vec<String> {
        self.modules
            .iter()
            .map(|module| module.path.clone())
            .collect()
    }

    pub(crate) fn declared(&self, var: Var) -> Option<&Declaration> {
        let (path, name) = path_of((var.module(), var.name()));
        self.module(path)?
            .declared
            .iter()
            .find(|declared| declared.name == name)
    }

    pub(crate) fn has_entry(&self, entry: Entry) -> bool {
        let (path, name) = path_of((entry.module(), entry.name()));
        self.module(path)
            .is_some_and(|module| defines_function(&module.text, &name))
    }

    pub(crate) fn buffer_usage(&self, var: Var) -> wgpu::BufferUsages {
        match self.declared(var).map(|declared| declared.ty) {
            Some(Ty::Uniform) => wgpu::BufferUsages::UNIFORM,
            Some(Ty::Storage { .. }) => wgpu::BufferUsages::STORAGE,
            Some(_) | None => wgpu::BufferUsages::empty(),
        }
    }

    pub(crate) fn check(&self) -> Vec<Error> {
        let mut problems = Vec::new();
        for module in self.modules.iter().filter(|module| !module.generated) {
            let path = format!("$.shaders.{}", module.path);
            let found = hex_floats(&module.text);
            if !found.is_empty() {
                problems.push(Error::new(ErrorCode::ShaderLink, path).got(found));
            }
        }
        problems
    }

    pub(crate) fn number(&mut self) -> Vec<(String, String, Ty)> {
        let mut ordered: Vec<(&'static str, usize, usize)> = Vec::new();
        for (index, module) in self.modules.iter().enumerate() {
            for (at, declared) in module.declared.iter().enumerate() {
                if declared.group == 0 {
                    ordered.push((module.package, index, at));
                }
            }
        }
        ordered.sort_by(|first, second| {
            let core = |package: &str| package != CORE_PACKAGE;
            (core(first.0), first.0, first.1, first.2).cmp(&(
                core(second.0),
                second.0,
                second.1,
                second.2,
            ))
        });
        self.numbers.clear();
        let mut listed = Vec::with_capacity(ordered.len());
        let mut next = CORE_BINDINGS.len() as u32;
        for (package, index, at) in ordered {
            let module = &self.modules[index];
            let declared = &module.declared[at];
            let number = match CORE_BINDINGS.iter().position(|name| *name == declared.name) {
                Some(core) if package == CORE_PACKAGE => core as u32,
                Some(_) | None => {
                    next += 1;
                    next - 1
                }
            };
            let key = (module.path.clone(), declared.name.clone());
            if self.numbers.insert(key.clone(), number).is_none() {
                listed.push((key.0, key.1, declared.ty));
            }
        }
        listed.sort_by_key(|(path, name, _)| {
            self.numbers.get(&(path.clone(), name.clone())).copied()
        });
        self.numbered = true;
        self.changed();
        listed
    }

    pub(crate) fn binding_of(&self, path: &str, name: &str) -> Option<u32> {
        self.numbers
            .get(&(path.to_owned(), name.to_owned()))
            .copied()
    }

    pub(crate) fn compose_in(&mut self, entry: Entry) -> Result<Arc<Composed>, Error> {
        let (path, name) = path_of((entry.module(), entry.name()));
        self.link(path, Some(&name))
    }

    pub(crate) fn compose_named(
        &mut self,
        module: &str,
        entry: &str,
    ) -> Result<Arc<Composed>, Error> {
        self.link(module, Some(entry))
    }

    pub(crate) fn compose_module(&mut self, module: &str) -> Result<Arc<Composed>, Error> {
        self.link(module, None)
    }

    fn link(&mut self, module: &str, kept: Option<&str>) -> Result<Arc<Composed>, Error> {
        let entry = kept.unwrap_or_default();
        let key = (module.to_owned(), entry.to_owned());
        if let Some(composed) = self.composed.get(&key) {
            return Ok(Arc::clone(composed));
        }
        let path = match kept {
            Some(entry) => format!("$.shaders.{module}.{entry}"),
            None => format!("$.shaders.{module}"),
        };
        if self.module(module).is_none() {
            return Err(Error::new(ErrorCode::UnknownModule, path).got(module));
        }
        let failed = |error: &dyn std::fmt::Display| {
            Error::new(ErrorCode::ShaderLink, path.clone()).got(error.to_string())
        };
        let prepared = self
            .prepared
            .get_or_insert_with(|| prepare(&self.modules, &self.numbers))
            .as_ref()
            .map_err(|error| failed(error))?;
        let main = module_path(module).map_err(|error| failed(&error))?;
        let whole = kept.and_then(|entry| {
            driver::entry(
                &mut self.wholes,
                (module.to_owned(), entry),
                (&main, prepared, &mut self.parsed),
            )
        });
        let linked = match whole {
            Some(linked) => linked,
            None => driver::linked_entry(&main, kept, prepared, &mut self.parsed)
                .map_err(|error| failed(&error))?,
        };
        let text = linked.text;
        let mut overrides = Vec::new();
        for (source, names) in self.modules.iter().zip(&prepared.overrides) {
            for (name, other) in names {
                let mangled = if source.path == module { name } else { other };
                if linked.overrides.contains(mangled) {
                    overrides.push((name.clone(), mangled.clone()));
                }
            }
        }
        let composed = Arc::new(Composed {
            module: module.to_owned(),
            text,
            bindings: linked.bindings,
            overrides,
        });
        self.composed.insert(key, Arc::clone(&composed));
        Ok(composed)
    }

    pub(crate) fn reaches(&mut self, entry: Entry, var: Var) -> bool {
        let (module, name) = path_of((entry.module(), entry.name()));
        let (path, var) = path_of((var.module(), var.name()));
        let key = (module.to_owned(), name);
        if !self.bound.contains_key(&key) {
            let bindings = self.bindings(module, &key.1);
            self.bound.insert(key.clone(), bindings);
        }
        let Some(bindings) = &self.bound[&key] else {
            return true;
        };
        named_in(module, path, &var).is_some_and(|named| bindings.contains(&named))
    }

    fn bindings(&mut self, module: &str, entry: &str) -> Option<Vec<String>> {
        self.module(module)?;
        let prepared = self
            .prepared
            .get_or_insert_with(|| prepare(&self.modules, &self.numbers))
            .as_ref()
            .ok()?;
        let main = module_path(module).ok()?;
        driver::bindings(&main, entry, prepared, &mut self.parsed).ok()
    }

    pub(crate) fn reached(&self, composed: &Composed) -> Vec<(String, Declaration)> {
        let Some(Ok(prepared)) = &self.prepared else {
            return self.reached_unprepared(composed);
        };
        let mut found = Vec::new();
        for (module, mangled) in self.modules.iter().zip(&prepared.declared) {
            for (declared, other) in module.declared.iter().zip(mangled) {
                let name = if module.path == composed.module {
                    &declared.name
                } else {
                    other
                };
                if composed.bindings.contains(name) {
                    found.push((module.path.clone(), declared.clone()));
                }
            }
        }
        found
    }

    fn reached_unprepared(&self, composed: &Composed) -> Vec<(String, Declaration)> {
        let mut found = Vec::new();
        for module in &self.modules {
            for declared in &module.declared {
                if named_in(&composed.module, &module.path, &declared.name)
                    .is_some_and(|named| composed.bindings.contains(&named))
                {
                    found.push((module.path.clone(), declared.clone()));
                }
            }
        }
        found
    }

    pub(crate) fn is_core(&self, path: &str) -> bool {
        self.module(path)
            .is_some_and(|module| module.package == CORE_PACKAGE)
    }

    pub(crate) fn keep(&mut self, path: &str, name: &str) {
        let kept = (path.to_owned(), name.to_owned());
        if !self.kept.contains(&kept) {
            self.kept.push(kept);
        }
    }

    pub(crate) fn defines(&self, path: &str, function: &str) -> bool {
        self.module(path)
            .is_some_and(|module| defines_function(&module.text, function))
    }

    pub(crate) fn unused(&mut self, composed: &[Arc<Composed>]) -> Vec<Error> {
        let mut texts: Vec<(String, String)> = composed
            .iter()
            .map(|composed| (composed.module.clone(), composed.text.clone()))
            .collect();
        let entries: Vec<(String, String)> = self
            .modules
            .iter()
            .filter(|module| !module.generated && module.package != CORE_PACKAGE)
            .flat_map(|module| {
                entry_points(&module.text)
                    .into_iter()
                    .map(|name| (module.path.clone(), name))
            })
            .collect();
        let roots: Vec<(String, String)> = self.kept.iter().cloned().chain(entries).collect();
        for (module, name) in &roots {
            if called(&texts, module, name) || !self.defines(module, name) {
                continue;
            }
            if let Ok(composed) = self.compose_named(module, name) {
                texts.push((composed.module.clone(), composed.text.clone()));
            }
        }
        let mut problems = Vec::new();
        for module in self
            .modules
            .iter()
            .filter(|module| !module.generated && module.package != CORE_PACKAGE)
        {
            for item in items(&module.text) {
                let root = roots
                    .iter()
                    .any(|(known, name)| *known == module.path && *name == item);
                if !root && !mentioned(&texts, &module.path, &item) {
                    problems.push(Error::new(
                        ErrorCode::WgslUnused,
                        format!("$.shaders.{}.{item}", module.path),
                    ));
                }
            }
        }
        problems
    }
}

fn named_in(main: &str, path: &str, name: &str) -> Option<String> {
    if path == main {
        return Some(name.to_owned());
    }
    let module = module_path(path).ok()?;
    Some(wesl::mangler::EscapeMangler.mangle(&module, name))
}

fn prepare(
    modules: &[Module],
    numbers: &BTreeMap<(String, String), u32>,
) -> Result<driver::Prepared, String> {
    let mut paths = Vec::with_capacity(modules.len());
    for source in modules {
        paths.push(module_path(&source.path)?);
    }
    let mut at = engine_core::collections::HashMap::default();
    let mut named = engine_core::collections::HashMap::default();
    for (index, (source, path)) in modules.iter().zip(&paths).enumerate() {
        at.insert(path.clone(), index);
        named
            .entry(format!("package::{}", source.path))
            .or_insert(index);
    }
    let mangle = |path, name: &str| driver::mangled(path, name);
    Ok(driver::Prepared {
        at,
        named,
        texts: modules
            .iter()
            .map(|source| numbered_text(numbers, source))
            .collect(),
        overrides: modules
            .iter()
            .zip(&paths)
            .map(|(source, path)| {
                declared_overrides(&source.text)
                    .into_iter()
                    .map(|name| {
                        let mangled = mangle(path, &name);
                        (name, mangled)
                    })
                    .collect()
            })
            .collect(),
        declared: modules
            .iter()
            .zip(&paths)
            .map(|(source, path)| {
                (source.declared.iter())
                    .map(|declared| mangle(path, &declared.name))
                    .collect()
            })
            .collect(),
    })
}

fn numbered_text(numbers: &BTreeMap<(String, String), u32>, module: &Module) -> String {
    let mut placed = String::with_capacity(module.text.len());
    for line in module.text.split_inclusive('\n') {
        match declared_binding(line) {
            Some((indent, 0, name, rest)) => {
                let number = numbers
                    .get(&(module.path.clone(), name))
                    .copied()
                    .unwrap_or(u32::MAX);
                placed.push_str(indent);
                placed.push_str(&format!("@group(0) @binding({number}) "));
                placed.push_str(rest);
            }
            Some(_) | None => placed.push_str(line),
        }
    }
    placed
}

fn handed_text(base: Option<&str>, parts: &[Part]) -> String {
    let mut text = base.unwrap_or_default().to_owned();
    for &(name, part) in parts {
        let declared = match name {
            Some(name) => declares_struct(&text, name),
            None => {
                let listed = items(part);
                let known = items(&text);
                text.contains(part)
                    || (listed.is_empty() && base.is_some())
                    || (!listed.is_empty() && listed.iter().all(|item| known.contains(item)))
            }
        };
        if !declared {
            text.push_str(part);
            if !part.ends_with('\n') {
                text.push('\n');
            }
        }
    }
    text
}

fn declares_struct(text: &str, name: &str) -> bool {
    let declaration = format!("struct {name}");
    text.match_indices(&declaration).any(|(at, _)| {
        let before = text[..at]
            .chars()
            .next_back()
            .is_none_or(|character| !(character.is_ascii_alphanumeric() || character == '_'));
        let after = text[at + declaration.len()..]
            .chars()
            .next()
            .is_none_or(|character| !(character.is_ascii_alphanumeric() || character == '_'));
        before && after
    })
}

fn mentioned(texts: &[(String, String)], path: &str, name: &str) -> bool {
    found(texts, (path, name), |rest| {
        rest.chars().next().is_none_or(|character| !word(character))
    })
}

fn called(texts: &[(String, String)], path: &str, name: &str) -> bool {
    found(texts, (path, name), |rest| {
        rest.trim_start().starts_with('(')
    })
}

fn found(
    texts: &[(String, String)],
    (path, name): (&str, &str),
    ends: impl Fn(&str) -> bool,
) -> bool {
    let Ok(module) = module_path(path) else {
        return false;
    };
    let mangled = wesl::mangler::EscapeMangler.mangle(&module, name);
    texts.iter().any(|(main, text)| {
        let ident = if main == path { name } else { mangled.as_str() };
        text.match_indices(ident).any(|(at, _)| {
            let before = text[..at]
                .chars()
                .next_back()
                .is_none_or(|character| !word(character));
            before && ends(&text[at + ident.len()..])
        })
    })
}

fn word(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '_'
}

fn items(text: &str) -> Vec<String> {
    use wesl::syntax::{GlobalDeclaration, TranslationUnit};
    let Ok(unit) = text.parse::<TranslationUnit>() else {
        return Vec::new();
    };
    unit.global_declarations
        .iter()
        .filter_map(|declaration| match declaration.node() {
            GlobalDeclaration::Function(item) => Some(item.ident.to_string()),
            GlobalDeclaration::Struct(item) => Some(item.ident.to_string()),
            GlobalDeclaration::Declaration(item) => Some(item.ident.to_string()),
            GlobalDeclaration::TypeAlias(item) => Some(item.ident.to_string()),
            GlobalDeclaration::Void
            | GlobalDeclaration::ConstAssert(_)
            | GlobalDeclaration::Compound(_) => None,
        })
        .collect()
}

fn entry_points(text: &str) -> Vec<String> {
    use wesl::syntax::{GlobalDeclaration, TranslationUnit};
    let Ok(unit) = text.parse::<TranslationUnit>() else {
        return Vec::new();
    };
    unit.global_declarations
        .iter()
        .filter_map(|declaration| match declaration.node() {
            GlobalDeclaration::Function(item)
                if item
                    .attributes
                    .iter()
                    .any(|attribute| attribute.node().is_entry_point()) =>
            {
                Some(item.ident.to_string())
            }
            GlobalDeclaration::Void
            | GlobalDeclaration::Declaration(_)
            | GlobalDeclaration::TypeAlias(_)
            | GlobalDeclaration::Struct(_)
            | GlobalDeclaration::Function(_)
            | GlobalDeclaration::ConstAssert(_)
            | GlobalDeclaration::Compound(_) => None,
        })
        .collect()
}

pub(crate) fn module_path(path: &str) -> Result<wesl::syntax::ModulePath, String> {
    format!("package::{path}")
        .parse::<wesl::syntax::ModulePath>()
        .map_err(|_| path.to_owned())
}

fn defines_function(text: &str, name: &str) -> bool {
    let declaration = format!("fn {name}(");
    text.match_indices(&declaration).any(|(at, _)| {
        at == 0
            || !text[..at]
                .ends_with(|character: char| character.is_ascii_alphanumeric() || character == '_')
    })
}

fn declared_overrides(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| line.trim_start().strip_prefix("override "))
        .filter_map(|rest| {
            let name: String = rest
                .chars()
                .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
                .collect();
            (!name.is_empty()).then_some(name)
        })
        .collect()
}

pub(crate) fn declarations(text: &str) -> Vec<Declaration> {
    text.lines()
        .filter_map(|line| {
            let (_, group, name, rest) = declared_binding(line)?;
            let binding = binding_number(line)?;
            let ty = binding_ty(rest)?;
            Some(Declaration {
                group,
                binding,
                name,
                ty,
            })
        })
        .collect()
}

fn binding_number(line: &str) -> Option<u32> {
    let after = &line[line.find("@binding(")? + "@binding(".len()..];
    after[..after.find(')')?].trim().parse().ok()
}

fn binding_ty(rest: &str) -> Option<Ty> {
    let declaration = rest.strip_prefix("var")?;
    let (space, declaration) = match declaration.strip_prefix('<') {
        Some(spaced) => {
            let close = spaced.find('>')?;
            (Some(&spaced[..close]), &spaced[close + 1..])
        }
        None => (None, declaration),
    };
    if let Some(space) = space {
        let space = space.replace(' ', "");
        return Some(match space.as_str() {
            "uniform" => Ty::Uniform,
            "storage,read_write" => Ty::Storage { write: true },
            _ => Ty::Storage { write: false },
        });
    }
    let ty = declaration[declaration.find(':')? + 1..]
        .trim()
        .trim_end_matches(';')
        .replace(' ', "");
    texture_ty(&ty)
}

fn texture_ty(ty: &str) -> Option<Ty> {
    use wgpu::TextureViewDimension as D;
    if ty == "sampler" {
        return Some(Ty::Sampler);
    }
    if ty == "sampler_comparison" {
        return Some(Ty::Comparison);
    }
    let depth = [
        ("texture_depth_2d_array", D::D2Array),
        ("texture_depth_cube_array", D::CubeArray),
        ("texture_depth_cube", D::Cube),
        ("texture_depth_2d", D::D2),
    ];
    if let Some(&(_, view)) = depth.iter().find(|(name, _)| ty == *name) {
        return Some(Ty::Texture {
            sample: Sample::Depth,
            view,
        });
    }
    let (head, inner) = ty.split_once('<')?;
    let inner = inner.strip_suffix('>')?;
    let views = [
        ("2d_array", D::D2Array),
        ("cube_array", D::CubeArray),
        ("cube", D::Cube),
        ("3d", D::D3),
        ("2d", D::D2),
        ("1d", D::D1),
    ];
    if let Some(shape) = head.strip_prefix("texture_storage_") {
        let view = views.iter().find(|(name, _)| shape == *name)?.1;
        let (format, access) = inner.split_once(',')?;
        let format = storage_format(format)?;
        let access = match access {
            "read" => wgpu::StorageTextureAccess::ReadOnly,
            "read_write" => wgpu::StorageTextureAccess::ReadWrite,
            _ => wgpu::StorageTextureAccess::WriteOnly,
        };
        return Some(Ty::StorageTexture {
            format,
            access,
            view,
        });
    }
    let shape = head.strip_prefix("texture_")?;
    let view = views.iter().find(|(name, _)| shape == *name)?.1;
    let sample = match inner {
        "u32" => Sample::Uint,
        "i32" => Sample::Sint,
        _ => Sample::Float,
    };
    Some(Ty::Texture { sample, view })
}

fn storage_format(format: &str) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as F;
    Some(match format {
        "r32float" => F::R32Float,
        "r32uint" => F::R32Uint,
        "r32sint" => F::R32Sint,
        "rg32float" => F::Rg32Float,
        "rg32uint" => F::Rg32Uint,
        "rgba8unorm" => F::Rgba8Unorm,
        "rgba8snorm" => F::Rgba8Snorm,
        "rgba8uint" => F::Rgba8Uint,
        "rgba16float" => F::Rgba16Float,
        "rgba16uint" => F::Rgba16Uint,
        "rgba32float" => F::Rgba32Float,
        "rgba32uint" => F::Rgba32Uint,
        "r16float" => F::R16Float,
        "rg16float" => F::Rg16Float,
        "bgra8unorm" => F::Bgra8Unorm,
        _ => return None,
    })
}

fn hex_floats(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let word = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    let mut found = Vec::new();
    for (x, _) in text.match_indices(['x', 'X']) {
        if x == 0 || bytes[x - 1] != b'0' || (x >= 2 && word(bytes[x - 2])) {
            continue;
        }
        let mut end = x + 1;
        let mut floating = false;
        while end < bytes.len() && (bytes[end].is_ascii_hexdigit() || bytes[end] == b'.') {
            floating |= bytes[end] == b'.';
            end += 1;
        }
        if floating || matches!(bytes.get(end), Some(b'p' | b'P')) {
            if matches!(bytes.get(end), Some(b'p' | b'P')) {
                end += 1;
                if matches!(bytes.get(end), Some(b'+' | b'-')) {
                    end += 1;
                }
            }
            while end < bytes.len() && word(bytes[end]) {
                end += 1;
            }
            found.push(text[x - 1..end].to_owned());
        }
    }
    found
}

fn declared_binding(line: &str) -> Option<(&str, u32, String, &str)> {
    let trimmed = line.trim_start();
    let indent = &line[..line.len() - trimmed.len()];
    let after_group = trimmed.strip_prefix("@group(")?;
    let close = after_group.find(')')?;
    let group = after_group[..close].trim().parse().ok()?;
    let after_group = &after_group[close + 1..];
    let after_binding = after_group.trim_start().strip_prefix("@binding(")?;
    let rest = after_binding[after_binding.find(')')? + 1..].trim_start();
    let declaration = rest.strip_prefix("var")?;
    let declaration = match declaration.strip_prefix('<') {
        Some(spaced) => &spaced[spaced.find('>')? + 1..],
        None => declaration,
    };
    let name: String = declaration
        .trim_start()
        .chars()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
        .collect();
    (!name.is_empty()).then_some((indent, group, name, rest))
}
