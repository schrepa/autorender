use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Arc;

use engine_core::collections::HashMap;
use engine_core::{Error, ErrorCode, Source};
use engine_registry::Registry;
use wesl::Mangler as _;

use super::{CORE_PACKAGE, Composed, Declaration, ShaderModule, Shaders, driver, facts};
use crate::shaders::Handed;

const CORE_BINDINGS: [&str; 1] = ["camera"];
const GENERATED_ROOT: &str = "generated::";

pub(crate) struct ShaderInputs<'a> {
    sources: Vec<&'static Source>,
    handed: Handed,
    edited: &'a BTreeMap<String, String>,
}

impl<'a> ShaderInputs<'a> {
    pub(crate) fn of(registry: &Registry, edited: &'a BTreeMap<String, String>) -> Self {
        Self {
            sources: std::iter::once(crate::shared::core_source())
                .chain(super::sources(registry))
                .collect(),
            handed: crate::shaders::handed(registry),
            edited,
        }
    }
}

struct Module {
    path: String,
    package: &'static str,
    text: Cow<'static, str>,
    declared: Vec<Declaration>,
    overrides: Vec<String>,
    functions: Vec<(String, Option<wgpu::ShaderStages>)>,
    items: Vec<String>,
    generated: bool,
}

impl Module {
    fn of(path: String, package: &'static str, text: Cow<'static, str>, generated: bool) -> Self {
        let facts = facts::facts(&text);
        Self {
            path,
            package,
            text,
            declared: facts.declared,
            overrides: facts.overrides,
            functions: facts.functions,
            items: facts.items,
            generated,
        }
    }
}

#[derive(Default)]
struct Parses {
    parsed: driver::Parsed,
    wholes: HashMap<String, driver::Slot>,
}

impl Shaders {
    pub(crate) fn link(inputs: ShaderInputs<'_>, unused: bool) -> Result<Self, Vec<Error>> {
        let mut modules: Vec<Module> = Vec::new();
        for source in &inputs.sources {
            for module in source.modules() {
                let text = (inputs.edited.get(module.path()))
                    .map_or(Cow::Borrowed(module.text()), |text| Cow::Owned(text.clone()));
                let generated = module.path().starts_with(GENERATED_ROOT);
                let path = module.path().to_owned();
                place(&mut modules, Module::of(path, source.package(), text, generated));
            }
        }
        let generated = crate::shaders::hashed(&inputs.handed);
        for (path, package, parts) in inputs.handed {
            let base = (modules.iter())
                .find(|module| module.path == path)
                .map(|module| module.text.to_string());
            let text = handed_text(base.as_deref(), &parts);
            place(&mut modules, Module::of(path, package, Cow::Owned(text), true));
        }
        let mut problems = hex_problems(&modules);
        let numbers = number(&mut modules);
        let prepared = prepare(&modules, &numbers)
            .map_err(|error| vec![Error::new(ErrorCode::ShaderLink, "$.shaders").got(error)])?;
        let mut parses = Parses::default();
        let mut linked = Vec::with_capacity(modules.len());
        for module in &modules {
            let mut entries = Vec::new();
            for (name, stage) in &module.functions {
                let Some(stage) = *stage else {
                    continue;
                };
                match compose(&modules, (&prepared, &mut parses), (&module.path, name, stage)) {
                    Ok(composed) => entries.push(Arc::new(composed)),
                    Err(error) => problems.push(error),
                }
            }
            linked.push(ShaderModule {
                path: module.path.clone(),
                core: module.package == CORE_PACKAGE,
                declared: module.declared.clone(),
                entries,
            });
        }
        if unused {
            problems.extend(unused_items(&modules, &linked));
        }
        if !problems.is_empty() {
            return Err(problems);
        }
        let sources = (inputs.sources.iter())
            .map(|source| (source.package().to_owned(), source.hash()))
            .collect();
        Ok(Self {
            sources,
            generated,
            modules: linked,
        })
    }
}

fn place(modules: &mut Vec<Module>, module: Module) {
    match modules.iter().position(|known| known.path == module.path) {
        Some(at) => modules[at] = module,
        None => modules.push(module),
    }
}

fn hex_problems(modules: &[Module]) -> Vec<Error> {
    let mut problems = Vec::new();
    for module in modules.iter().filter(|module| !module.generated) {
        let path = format!("$.shaders.{}", module.path);
        let found = hex_floats(&module.text);
        if !found.is_empty() {
            problems.push(Error::new(ErrorCode::ShaderLink, path).got(found));
        }
    }
    problems
}

fn number(modules: &mut [Module]) -> BTreeMap<(String, String), u32> {
    let mut ordered: Vec<(&'static str, usize, usize)> = Vec::new();
    for (index, module) in modules.iter().enumerate() {
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
    let mut numbers = BTreeMap::new();
    let mut next = CORE_BINDINGS.len() as u32;
    for (package, index, at) in ordered {
        let module = &modules[index];
        let declared = &module.declared[at];
        let number = match CORE_BINDINGS.iter().position(|name| *name == declared.name) {
            Some(core) if package == CORE_PACKAGE => core as u32,
            Some(_) | None => {
                next += 1;
                next - 1
            }
        };
        numbers.insert((module.path.clone(), declared.name.clone()), number);
    }
    for module in modules.iter_mut() {
        for declared in module.declared.iter_mut().filter(|declared| declared.group == 0) {
            let key = (module.path.clone(), declared.name.clone());
            if let Some(&number) = numbers.get(&key) {
                declared.binding = number;
            }
        }
    }
    numbers
}

fn compose(
    modules: &[Module],
    (prepared, parses): (&driver::Prepared, &mut Parses),
    (module, entry, stage): (&str, &str, wgpu::ShaderStages),
) -> Result<Composed, Error> {
    let path = format!("$.shaders.{module}.{entry}");
    let failed = |error: &dyn std::fmt::Display| {
        Error::new(ErrorCode::ShaderLink, path.clone()).got(error.to_string())
    };
    let main = module_path(module).map_err(|error| failed(&error))?;
    let whole = driver::entry(
        &mut parses.wholes,
        (module.to_owned(), entry),
        (&main, prepared, &mut parses.parsed),
    );
    let linked = match whole {
        Some(linked) => linked,
        None => driver::linked_entry(&main, entry, prepared, &mut parses.parsed)
            .map_err(|error| failed(&error))?,
    };
    let mut overrides = Vec::new();
    for (source, names) in modules.iter().zip(&prepared.overrides) {
        for (name, other) in names {
            let mangled = if source.path == module { name } else { other };
            if linked.overrides.contains(mangled) {
                overrides.push((name.clone(), mangled.clone()));
            }
        }
    }
    let mut reached = Vec::new();
    for (source, mangled) in modules.iter().zip(&prepared.declared) {
        for (declared, other) in source.declared.iter().zip(mangled) {
            let name = if source.path == module {
                &declared.name
            } else {
                other
            };
            if linked.bindings.contains(name) {
                reached.push((source.path.clone(), declared.clone()));
            }
        }
    }
    let fragment = facts::fragment(&linked.text, entry);
    Ok(Composed {
        module: module.to_owned(),
        entry: entry.to_owned(),
        stage,
        text: linked.text,
        overrides,
        fragment,
        reached,
    })
}

fn unused_items(modules: &[Module], linked: &[ShaderModule]) -> Vec<Error> {
    let texts: Vec<(&str, &str)> = (linked.iter())
        .flat_map(|module| &module.entries)
        .map(|composed| (composed.module.as_str(), composed.text.as_str()))
        .collect();
    let mut problems = Vec::new();
    let checked = modules.iter().filter(|module| !module.generated);
    for module in checked.filter(|module| module.package != CORE_PACKAGE) {
        for item in &module.items {
            let root = (module.functions.iter())
                .any(|(name, stage)| stage.is_some() && name == item);
            if !root && !mentioned(&texts, &module.path, item) {
                problems.push(Error::new(
                    ErrorCode::WgslUnused,
                    format!("$.shaders.{}.{item}", module.path),
                ));
            }
        }
    }
    problems
}

fn prepare(
    modules: &[Module],
    numbers: &BTreeMap<(String, String), u32>,
) -> Result<driver::Prepared, String> {
    let mut paths = Vec::with_capacity(modules.len());
    for source in modules {
        paths.push(module_path(&source.path)?);
    }
    let mut at = HashMap::default();
    let mut named = HashMap::default();
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
                (source.overrides.iter())
                    .map(|name| (name.clone(), mangle(path, name)))
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

fn handed_text(base: Option<&str>, parts: &[Cow<'static, str>]) -> String {
    let mut text = base.unwrap_or_default().to_owned();
    for part in parts.iter().map(AsRef::as_ref) {
        let listed = facts::facts(part).items;
        let known = facts::facts(&text).items;
        let declared = text.contains(part)
            || (listed.is_empty() && base.is_some())
            || (!listed.is_empty() && listed.iter().all(|item| known.contains(item)));
        if !declared {
            text.push_str(part);
            if !part.ends_with('\n') {
                text.push('\n');
            }
        }
    }
    text
}

fn mentioned(texts: &[(&str, &str)], path: &str, name: &str) -> bool {
    let Ok(module) = module_path(path) else {
        return false;
    };
    let mangled = wesl::mangler::EscapeMangler.mangle(&module, name);
    texts.iter().any(|&(main, text)| {
        let ident = if main == path { name } else { mangled.as_str() };
        text.match_indices(ident).any(|(at, _)| {
            let before = text[..at]
                .chars()
                .next_back()
                .is_none_or(|character| !word(character));
            let after = text[at + ident.len()..]
                .chars()
                .next()
                .is_none_or(|character| !word(character));
            before && after
        })
    })
}

fn word(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '_'
}

fn module_path(path: &str) -> Result<wesl::syntax::ModulePath, String> {
    format!("package::{path}")
        .parse::<wesl::syntax::ModulePath>()
        .map_err(|_| path.to_owned())
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
