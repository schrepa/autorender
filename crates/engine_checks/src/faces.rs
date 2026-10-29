pub(crate) mod parse;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::json;

use crate::Repository;
use parse::{Impl, Item, Kind, Parsed, joined};

pub const FILE: &str = "faces.json";

const DEPTH: usize = 16;

#[derive(Default)]
pub struct Face {
    pub items: BTreeSet<String>,
    pub wgsl: BTreeSet<String>,
    modules: BTreeSet<String>,
    names: BTreeSet<String>,
}

impl Face {
    pub fn reaches(&self, segments: &[String]) -> bool {
        let mut path = String::new();
        for segment in segments {
            if segment == "self" || segment == "*" {
                return true;
            }
            path = joined(&path, segment);
            if self.names.contains(&path) {
                return true;
            }
            if !self.modules.contains(&path) {
                return false;
            }
        }
        true
    }
}

struct Module {
    parent: Option<usize>,
    parsed: Parsed,
    children: Vec<(String, usize)>,
}

impl Module {
    fn child(&self, name: &str) -> Option<usize> {
        self.children
            .iter()
            .find(|(child, _)| child == name)
            .map(|&(_, index)| index)
    }
}

enum Target {
    Module(usize),
    Item(usize, usize),
    Outside,
}

pub fn face(folder: &Path) -> Result<Face, String> {
    let mut modules = Vec::new();
    let lib = folder.join("src").join("lib.rs");
    let mut face = Face::default();
    if lib.is_file() {
        let code = parse::code(&read(&lib)?);
        load(&mut modules, None, code, folder.join("src"), 0)?;
        let mut impls: BTreeMap<&str, Vec<&Impl>> = BTreeMap::new();
        for module in &modules {
            for implementation in &module.parsed.impls {
                impls
                    .entry(implementation.target.as_str())
                    .or_default()
                    .push(implementation);
            }
        }
        let mut builder = Builder {
            face: &mut face,
            impls: &impls,
        };
        expose(&modules, 0, "", &mut builder, 0);
        for module in &modules {
            for name in &module.parsed.macros {
                face.items.insert(format!("macro {name}"));
                face.names.insert(name.clone());
            }
        }
    }
    face.wgsl = wgsl(folder)?;
    Ok(face)
}

fn read(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn load(
    modules: &mut Vec<Module>,
    parent: Option<usize>,
    code: Vec<char>,
    children: PathBuf,
    depth: usize,
) -> Result<usize, String> {
    let index = modules.len();
    modules.push(Module {
        parent,
        parsed: parse::parse(&code),
        children: Vec::new(),
    });
    if depth > DEPTH {
        return Ok(index);
    }
    let declared: Vec<(String, Option<String>)> = modules[index]
        .parsed
        .items
        .iter()
        .filter(|item| item.kind == Kind::Module)
        .map(|item| (item.name.clone(), item.body.clone()))
        .collect();
    for (name, body) in declared {
        if modules[index].child(&name).is_some() {
            continue;
        }
        let folder = children.join(&name);
        let code = match body {
            Some(body) => body.chars().collect(),
            None => {
                let Some(file) = [children.join(format!("{name}.rs")), folder.join("mod.rs")]
                    .into_iter()
                    .find(|file| file.is_file())
                else {
                    continue;
                };
                parse::code(&read(&file)?)
            }
        };
        let child = load(modules, Some(index), code, folder, depth + 1)?;
        modules[index].children.push((name, child));
    }
    Ok(index)
}

struct Builder<'a> {
    face: &'a mut Face,
    impls: &'a BTreeMap<&'a str, Vec<&'a Impl>>,
}

impl Builder<'_> {
    fn module(&mut self, path: &str) {
        self.face.items.insert(format!("mod {path}"));
        self.face.modules.insert(path.to_owned());
    }

    fn external(&mut self, path: &str, source: &str) {
        self.face.items.insert(format!("use {path} = {source}"));
        self.face.names.insert(path.to_owned());
    }

    fn item(&mut self, item: &Item, path: &str) {
        self.face.names.insert(path.to_owned());
        let lines = &mut self.face.items;
        match item.kind {
            Kind::Function => {
                lines.insert(format!("{}fn {path}{}", item.qualifiers, item.header));
            }
            Kind::Constant => {
                lines.insert(format!("const {path}{}", item.header));
            }
            Kind::Static => {
                lines.insert(format!("static {}{path}{}", item.qualifiers, item.header));
            }
            Kind::Module => {}
            Kind::Struct | Kind::Enum | Kind::Union | Kind::Trait | Kind::Alias => {
                lines.insert(format!("{} {path}{}", item.kind.word(), item.header));
                for member in &item.members {
                    let separator = if member.word == "field" { "." } else { "::" };
                    lines.insert(format!(
                        "{} {path}{separator}{}{}",
                        member.word, member.name, member.rest
                    ));
                }
                for derive in &item.derives {
                    lines.insert(format!("impl {derive} for {path}"));
                }
                for implementation in self.impls.get(item.name.as_str()).into_iter().flatten() {
                    match &implementation.trait_name {
                        Some(name) => {
                            lines.insert(format!("impl {name} for {path}"));
                        }
                        None => {
                            for member in &implementation.members {
                                lines.insert(format!(
                                    "{} {path}::{}{}",
                                    member.word, member.name, member.rest
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
}

fn expose(modules: &[Module], index: usize, prefix: &str, builder: &mut Builder<'_>, depth: usize) {
    if depth > DEPTH {
        return;
    }
    let module = &modules[index];
    for item in module.parsed.items.iter().filter(|item| item.public) {
        let path = joined(prefix, &item.name);
        if item.kind == Kind::Module {
            builder.module(&path);
            if let Some(child) = module.child(&item.name) {
                expose(modules, child, &path, builder, depth + 1);
            }
        } else {
            builder.item(item, &path);
        }
    }
    for import in module.parsed.uses.iter().filter(|import| import.public) {
        let path = joined(prefix, &import.alias);
        match resolve(modules, index, &import.path, 0) {
            Target::Module(child) if import.glob => {
                expose(modules, child, prefix, builder, depth + 1);
            }
            Target::Module(child) => {
                builder.module(&path);
                expose(modules, child, &path, builder, depth + 1);
            }
            Target::Item(owner, item) => builder.item(&modules[owner].parsed.items[item], &path),
            Target::Outside if import.glob => {
                builder.external(
                    &joined(prefix, "*"),
                    &format!("{}::*", import.path.join("::")),
                );
            }
            Target::Outside => builder.external(&path, &import.path.join("::")),
        }
    }
}

fn lookup(modules: &[Module], module: usize, name: &str, depth: usize) -> Option<Target> {
    if depth > DEPTH {
        return None;
    }
    let found = &modules[module];
    if let Some(child) = found.child(name) {
        return Some(Target::Module(child));
    }
    if let Some(item) = found
        .parsed
        .items
        .iter()
        .position(|item| item.name == name && item.kind != Kind::Module)
    {
        return Some(Target::Item(module, item));
    }
    if let Some(import) = found
        .parsed
        .uses
        .iter()
        .find(|import| !import.glob && import.alias == name)
    {
        return Some(resolve(modules, module, &import.path, depth + 1));
    }
    for import in found.parsed.uses.iter().filter(|import| import.glob) {
        if let Target::Module(target) = resolve(modules, module, &import.path, depth + 1)
            && target != module
            && let Some(target) = lookup(modules, target, name, depth + 1)
        {
            return Some(target);
        }
    }
    None
}

fn resolve(modules: &[Module], from: usize, segments: &[String], depth: usize) -> Target {
    let Some((first, mut rest)) = segments.split_first() else {
        return Target::Outside;
    };
    if depth > DEPTH {
        return Target::Outside;
    }
    let mut current = match first.as_str() {
        "crate" => 0,
        "self" => from,
        "super" => {
            let Some(mut parent) = modules[from].parent else {
                return Target::Outside;
            };
            while rest.first().is_some_and(|segment| segment == "super") {
                let Some(above) = modules[parent].parent else {
                    return Target::Outside;
                };
                parent = above;
                rest = &rest[1..];
            }
            parent
        }
        _ => match lookup(modules, from, first, depth) {
            Some(Target::Module(child)) => child,
            Some(target) if rest.is_empty() => return target,
            Some(_) | None => return Target::Outside,
        },
    };
    for (position, segment) in rest.iter().enumerate() {
        let last = position + 1 == rest.len();
        match lookup(modules, current, segment, depth) {
            Some(Target::Module(child)) => current = child,
            Some(target) if last => return target,
            Some(_) | None => return Target::Outside,
        }
    }
    Target::Module(current)
}

fn wgsl(folder: &Path) -> Result<BTreeSet<String>, String> {
    let mut files = Vec::new();
    shaders(folder, &mut files);
    files.sort();
    let mut functions = BTreeSet::new();
    for file in files {
        let source = read(&file)?;
        let shown = file
            .strip_prefix(folder)
            .map(crate::workspace::slashed)
            .unwrap_or_default();
        let code: Vec<char> = uncommented(&source).chars().collect();
        let mut depth = 0_usize;
        let mut at = 0;
        while at < code.len() {
            match code[at] {
                '{' => depth += 1,
                '}' => depth = depth.saturating_sub(1),
                'f' if depth == 0
                    && code.get(at + 1) == Some(&'n')
                    && code.get(at + 2).is_some_and(|c| c.is_whitespace())
                    && (at == 0 || !parse::identifier(code[at - 1])) =>
                {
                    let start = at + 2;
                    let stop = (start..code.len())
                        .find(|&index| code[index] == '{' || code[index] == ';')
                        .unwrap_or(code.len());
                    let signature = parse::normalize(&code[start..stop]);
                    functions.insert(format!("wgsl {shown}::{signature}"));
                    at = stop;
                    continue;
                }
                _ => {}
            }
            at += 1;
        }
    }
    Ok(functions)
}

fn shaders(folder: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(folder).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name != "target") {
                shaders(&path, files);
            }
        } else if path
            .extension()
            .is_some_and(|extension| extension == "wgsl")
        {
            files.push(path);
        }
    }
}

fn uncommented(source: &str) -> String {
    let characters: Vec<char> = source.chars().collect();
    let mut out = String::with_capacity(source.len());
    let mut at = 0;
    let mut depth = 0_usize;
    while at < characters.len() {
        let pair = (characters[at], characters.get(at + 1).copied());
        if pair == ('/', Some('*')) {
            depth += 1;
            at += 2;
        } else if depth > 0 && pair == ('*', Some('/')) {
            depth -= 1;
            at += 2;
        } else if depth > 0 {
            at += 1;
        } else if pair == ('/', Some('/')) {
            while at < characters.len() && characters[at] != '\n' {
                at += 1;
            }
        } else {
            out.push(characters[at]);
            at += 1;
        }
    }
    out
}

pub fn faces(
    repository: &Repository,
    crates: &BTreeSet<String>,
) -> Result<BTreeMap<String, Face>, String> {
    let mut faces = BTreeMap::new();
    for name in crates {
        if let Some(package) = repository.of_crate(name) {
            faces.insert(name.clone(), face(&package.folder)?);
        }
    }
    Ok(faces)
}

pub fn files(
    repository: &Repository,
    registered: &[&str],
) -> Result<Vec<(String, String)>, String> {
    let crates: BTreeSet<String> =
        crate::bricks::brick_crates(repository, registered.iter().copied())
            .into_iter()
            .filter(|name| {
                repository
                    .of_crate(name)
                    .is_some_and(|package| package.own())
            })
            .collect();
    let listed: Vec<serde_json::Value> = faces(repository, &crates)?
        .into_iter()
        .map(|(name, face)| {
            json!({
                "crate": name,
                "items": face.items,
                "wgsl": face.wgsl,
            })
        })
        .collect();
    let text = serde_json::to_string_pretty(&json!({ "faces": listed }))
        .map_err(|error| format!("cannot write {FILE}: {error}"))?;
    Ok(vec![(FILE.to_owned(), text + "\n")])
}

pub fn check(repository: &Repository) -> Result<(), String> {
    let mut problems = Vec::new();
    let mut compared = 0;
    for package in repository.own() {
        let Some(replaced) = package.replaces() else {
            continue;
        };
        let Some(original) = repository
            .of_crate(replaced)
            .or_else(|| repository.package(replaced))
        else {
            problems.push(
                json!({"rule": "replaces_unknown", "crate": package.name, "replaces": replaced})
                    .to_string(),
            );
            continue;
        };
        let offered = face(&package.folder)?;
        let wanted = face(&original.folder)?;
        let missing: Vec<&String> = wanted
            .items
            .difference(&offered.items)
            .chain(wanted.wgsl.difference(&offered.wgsl))
            .collect();
        compared += 1;
        if !missing.is_empty() {
            problems.push(
                json!({
                    "rule": "face_missing",
                    "crate": package.name,
                    "replaces": original.name,
                    "missing": missing,
                })
                .to_string(),
            );
        }
    }
    if problems.is_empty() {
        println!(
            "{}",
            json!({"event": "faces_offered", "replacements": compared})
        );
        return Ok(());
    }
    Err(problems.join("\n"))
}
