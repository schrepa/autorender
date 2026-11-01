use std::borrow::Cow;
use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use engine_core::{Entry, Error, ErrorCode, Source, Var};
use wesl::Mangler as _;

pub const CORE_PACKAGE: &str = "engine_render";
pub const CORE_PREFIX: &str = "engine";
pub const CAMERA: &str = "camera";
pub const PIECES: &str = "pieces";
const CORE_BINDINGS: [&str; 2] = [CAMERA, PIECES];
const RESERVED: [&str; 4] = ["engine", "generated", "face", "place"];
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
    Storage { write: bool },
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
            Self::Storage { write } => buffer(wgpu::BufferBindingType::Storage { read_only: !write }),
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
    pub(crate) entry: String,
    pub(crate) text: String,
    pub(crate) hash: u64,
    pub(crate) switches: Vec<u32>,
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

#[derive(Default)]
pub(crate) struct Linker {
    modules: Vec<Module>,
    sources: Vec<&'static Source>,
    numbers: BTreeMap<(String, String), u32>,
    composed: BTreeMap<(String, String), Arc<Composed>>,
    numbered: bool,
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

fn prefix(path: &str) -> &str {
    path.split("::").next().unwrap_or(path)
}

impl Linker {
    pub(crate) fn new(core: &'static Source) -> Self {
        let mut linker = Self::default();
        linker.add_source(core);
        linker
    }

    pub(crate) fn add_source(&mut self, source: &'static Source) {
        if self.sources.iter().any(|known| std::ptr::eq(*known, source)) {
            return;
        }
        self.sources.push(source);
        for module in source.modules() {
            let generated = module.path().starts_with(GENERATED_ROOT);
            self.add(module.path().to_owned(), source.package(), Cow::Borrowed(module.text()), generated);
        }
        self.numbered = false;
        self.composed.clear();
    }

    pub(crate) fn add_generated(&mut self, path: String, package: &'static str, text: String) {
        self.modules.retain(|known| known.path != path);
        self.add(path, package, Cow::Owned(text), true);
        self.numbered = false;
        self.composed.clear();
    }

    fn add(&mut self, path: String, package: &'static str, text: Cow<'static, str>, generated: bool) {
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
        self.modules.iter().map(|module| module.path.clone()).collect()
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

    pub(crate) fn check(&self, fills: &dyn Fn(&'static str, &str) -> bool) -> Vec<Error> {
        let mut problems = Vec::new();
        let mut prefixes: Vec<(&'static str, &str)> = Vec::new();
        for module in self.modules.iter().filter(|module| !module.generated) {
            let path = format!("$.shaders.{}", module.path);
            let first = prefix(&module.path);
            let core = module.package == CORE_PACKAGE;
            if core != (first == CORE_PREFIX) || (!core && RESERVED.contains(&first)) {
                problems.push(Error::new(ErrorCode::ModuleOutsideBrick, path).got(module.package));
                continue;
            }
            match prefixes.iter().find(|(package, _)| *package == module.package) {
                Some((_, known)) if *known != first => {
                    problems.push(Error::new(ErrorCode::ModuleOutsideBrick, path).got(first));
                    continue;
                }
                Some(_) => {}
                None => {
                    if let Some((owner, _)) = prefixes.iter().find(|(_, known)| *known == first) {
                        problems.push(Error::new(ErrorCode::NameTwice, path).got(*owner));
                        continue;
                    }
                    prefixes.push((module.package, first));
                }
            }
            if !core
                && let Some(declared) = module
                    .declared
                    .iter()
                    .find(|declared| declared.group == 0 && CORE_BINDINGS.contains(&declared.name.as_str()))
            {
                problems.push(Error::new(ErrorCode::BindingForeign, path.clone()).got(&declared.name));
            }
            let found = hex_floats(&module.text);
            if !found.is_empty() {
                problems.push(Error::new(ErrorCode::ShaderLink, path.clone()).got(found));
            }
            for imported in imports(&module.text) {
                if !self.may_import(module, first, &imported, fills) {
                    problems.push(Error::new(ErrorCode::ImportForeign, path.clone()).got(imported));
                }
            }
        }
        problems
    }

    fn may_import(
        &self,
        module: &Module,
        own: &str,
        imported: &str,
        fills: &dyn Fn(&'static str, &str) -> bool,
    ) -> bool {
        let mut segments = imported.split("::");
        let first = segments.next().unwrap_or_default();
        let second = segments.next().unwrap_or_default();
        match first {
            "engine" | "face" => true,
            "place" => module.package == CORE_PACKAGE || fills(module.package, second),
            "generated" => match second {
                "pieces" => segments.next() == Some(own),
                "targets" | "composite" => true,
                _ => second == own,
            },
            _ => first == own,
        }
    }

    pub(crate) fn number(&mut self) -> Vec<(String, String, &'static str, Ty)> {
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
            (core(first.0), first.0, first.1, first.2).cmp(&(core(second.0), second.0, second.1, second.2))
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
                listed.push((key.0, key.1, package, declared.ty));
            }
        }
        listed.sort_by_key(|(path, name, _, _)| self.numbers.get(&(path.clone(), name.clone())).copied());
        self.numbered = true;
        self.composed.clear();
        listed
    }

    pub(crate) fn binding_of(&self, path: &str, name: &str) -> Option<u32> {
        self.numbers.get(&(path.to_owned(), name.to_owned())).copied()
    }

    pub(crate) fn compose(&mut self, entry: Entry) -> Result<Arc<Composed>, Error> {
        let (path, name) = path_of((entry.module(), entry.name()));
        self.compose_named(path, &name)
    }

    pub(crate) fn compose_named(&mut self, module: &str, entry: &str) -> Result<Arc<Composed>, Error> {
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
        let mut resolver = wesl::resolver::VirtualResolver::new();
        for source in &self.modules {
            let text = self.numbered_text(source);
            resolver.add_module(
                module_path(&source.path).map_err(|error| failed(&error))?,
                Cow::Owned(text),
            );
        }
        let mut features = wesl::Features::new();
        features.default = wesl::Feature::Error;
        let options = wesl::CompileOptions {
            imports: true,
            condcomp: true,
            visibility: true,
            generics: false,
            strip: true,
            lower: false,
            validate: false,
            sourcemap: false,
            mangler: wesl::ManglerKind::Escape,
            mangle_main: false,
            keep: kept.map(|entry| vec![entry.to_owned()]),
            keep_main: kept.is_none(),
            features,
            ..Default::default()
        };
        let main = module_path(module).map_err(|error| failed(&error))?;
        let mut result = wesl::compile(&main, &options, &resolver).map_err(|error| failed(&error))?;
        let order = |path: &wesl::syntax::ModulePath| {
            if *path == main {
                return 0;
            }
            let named = path.to_string();
            1 + self
                .modules
                .iter()
                .position(|source| format!("package::{}", source.path) == named)
                .unwrap_or(self.modules.len())
        };
        result.modules.sort_by_key(|module| order(&module.path));
        let text = wesl::pass::link(&result.modules, Some(&result.used_items)).to_string();
        let switches = switch_ids(&text);
        let mut overrides = Vec::new();
        for source in &self.modules {
            for name in declared_overrides(&source.text) {
                let mangled = if source.path == module {
                    name.clone()
                } else {
                    let path = module_path(&source.path).map_err(|error| failed(&error))?;
                    wesl::mangler::EscapeMangler.mangle(&path, &name)
                };
                if declares_override(&text, &mangled) {
                    overrides.push((name, mangled));
                }
            }
        }
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let composed = Arc::new(Composed {
            module: module.to_owned(),
            entry: entry.to_owned(),
            hash: hasher.finish(),
            text,
            switches,
            overrides,
        });
        self.composed.insert(key, Arc::clone(&composed));
        Ok(composed)
    }

    fn numbered_text(&self, module: &Module) -> String {
        let mut placed = String::with_capacity(module.text.len());
        for line in module.text.split_inclusive('\n') {
            match declared_binding(line) {
                Some((indent, 0, name, rest)) => {
                    let number = self
                        .numbers
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

    pub(crate) fn reaches(&self, composed: &Composed, path: &str, name: &str) -> bool {
        let mangled = if path == composed.module {
            name.to_owned()
        } else {
            match module_path(path) {
                Ok(module) => wesl::mangler::EscapeMangler.mangle(&module, name),
                Err(_) => return false,
            }
        };
        declares_var(&composed.text, &mangled)
    }

    pub(crate) fn reached(&self, composed: &Composed) -> Vec<(String, Declaration)> {
        let mut found = Vec::new();
        for module in &self.modules {
            for declared in &module.declared {
                if self.reaches(composed, &module.path, &declared.name) {
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

    pub(crate) fn defines(&self, path: &str, function: &str) -> bool {
        self.module(path)
            .is_some_and(|module| defines_function(&module.text, function))
    }

    pub(crate) fn provides(&self, path: &str, name: &str) -> bool {
        use wesl::syntax::{GlobalDeclaration, TranslationUnit};
        let Some(module) = self.module(path) else {
            return false;
        };
        let Ok(unit) = module.text.parse::<TranslationUnit>() else {
            return false;
        };
        let declared = unit.global_declarations.iter().any(|declaration| {
            matches!(declaration.node(), GlobalDeclaration::Declaration(item) if item.ident.to_string() == name)
        });
        declared || unit.imports.iter().any(|import| imports_name(&import.content, name))
    }

    pub(crate) fn unused(&mut self, composed: &[Arc<Composed>], roots: &[(String, String)]) -> Vec<Error> {
        let mut texts: Vec<(String, String)> = composed
            .iter()
            .map(|composed| (composed.module.clone(), composed.text.clone()))
            .collect();
        for (module, name) in roots {
            if mentioned(&texts, module, name) || !self.defines(module, name) {
                continue;
            }
            if let Ok(composed) = self.compose_named(module, name) {
                texts.push((composed.module.clone(), composed.text.clone()));
            }
        }
        let mut problems = Vec::new();
        for module in self.modules.iter().filter(|module| !module.generated && module.package != CORE_PACKAGE) {
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

fn imports_name(content: &wesl::syntax::ImportContent, name: &str) -> bool {
    match content {
        wesl::syntax::ImportContent::Item(item) => {
            item.rename.as_ref().unwrap_or(&item.ident).to_string() == name
        }
        wesl::syntax::ImportContent::Collection(imports) => {
            imports.iter().any(|import| imports_name(&import.content, name))
        }
    }
}

pub(crate) fn declares_struct(text: &str, name: &str) -> bool {
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
    let Ok(module) = module_path(path) else {
        return false;
    };
    let mangled = wesl::mangler::EscapeMangler.mangle(&module, name);
    texts.iter().any(|(main, text)| {
        let ident = if main == path { name } else { mangled.as_str() };
        mentions(text, ident)
    })
}

fn mentions(text: &str, ident: &str) -> bool {
    let word = |character: char| character.is_ascii_alphanumeric() || character == '_';
    text.match_indices(ident).any(|(at, _)| {
        let before = text[..at].chars().next_back().is_none_or(|character| !word(character));
        let after = text[at + ident.len()..].chars().next().is_none_or(|character| !word(character));
        before && after
    })
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
            GlobalDeclaration::Void | GlobalDeclaration::ConstAssert(_) | GlobalDeclaration::Compound(_) => None,
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
        at == 0 || !text[..at].ends_with(|character: char| character.is_ascii_alphanumeric() || character == '_')
    })
}

fn declares_var(text: &str, name: &str) -> bool {
    text.lines().any(|line| {
        declared_binding(line).is_some_and(|(_, _, declared, _)| declared == name)
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

fn declares_override(text: &str, name: &str) -> bool {
    let declaration = format!("override {name}");
    text.match_indices(&declaration).any(|(at, _)| {
        !text[at + declaration.len()..]
            .starts_with(|character: char| character.is_ascii_alphanumeric() || character == '_')
    })
}

fn switch_ids(text: &str) -> Vec<u32> {
    let mut ids = Vec::new();
    for (at, _) in text.match_indices("@id(") {
        let digits: String = text[at + 4..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        if let Ok(id) = digits.parse::<u32>()
            && !ids.contains(&id)
        {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids
}

fn imports(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    for line in text.lines() {
        let Some(path) = line.trim_start().strip_prefix("import package::") else {
            continue;
        };
        let path: String = path
            .chars()
            .take_while(|character| character.is_ascii_alphanumeric() || *character == '_' || *character == ':')
            .collect();
        let path = path.trim_end_matches(':').to_owned();
        if !path.is_empty() && !found.contains(&path) {
            found.push(path);
        }
    }
    found
}

fn declarations(text: &str) -> Vec<Declaration> {
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
