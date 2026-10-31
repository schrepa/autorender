pub(crate) mod parse;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use engine::bricks::{AddBrick, BrickRegistry};
use engine_core::{Error, ErrorCode, Expected};
use engine_render::linker::ShaderModule;
use engine_render::places::{Answer, Contract, contract, place_module};
use engine_render::render_bricks::RenderBrick;
use serde_json::json;

use crate::Repository;
use crate::policy::rust;
use parse::{Impl, Item, Kind, Parsed, joined};

pub const FILE: &str = "faces.json";

const DEPTH: usize = 16;

const FACES: &str = "faces";

const BRICK_TRAIT: &str = "Brick";

const SIDES: [&str; 3] = ["cpu", "gpu", "wgsl"];

const ITEM_WORDS: [&str; 17] = [
    "pub", "const", "unsafe", "async", "extern", "fn", "struct", "enum", "union", "trait", "type",
    "static", "mut", "field", "variant", "mod", "use",
];

#[derive(Default)]
pub struct Face {
    pub items: BTreeSet<String>,
    pub wgsl: BTreeSet<String>,
    pub faces: BTreeMap<String, Vec<String>>,
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
    face.faces = signatures_in(&folder.join("src").join(FACES))?;
    Ok(face)
}

fn signatures_in(folder: &Path) -> Result<BTreeMap<String, Vec<String>>, String> {
    let mut found = BTreeMap::new();
    let mut files: Vec<PathBuf> = fs::read_dir(folder)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "wgsl")
        })
        .collect();
    files.sort();
    for file in files {
        let Some(name) = file.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        found.insert(name.to_owned(), signatures(&read(&file)?));
    }
    Ok(found)
}

pub fn signatures(source: &str) -> Vec<String> {
    let code: Vec<char> = uncommented(source).chars().collect();
    let mut found = Vec::new();
    let mut depth = 0_usize;
    let mut at = 0;
    while at < code.len() {
        match code[at] {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ if depth == 0 && (at == 0 || !parse::identifier(code[at - 1])) => {
                if let Some((signature, stop)) = declaration(&code, at) {
                    found.push(signature);
                    at = stop;
                    continue;
                }
            }
            _ => {}
        }
        at += 1;
    }
    found
}

fn declaration(code: &[char], at: usize) -> Option<(String, usize)> {
    let word: String = code[at..]
        .iter()
        .take_while(|&&character| parse::identifier(character))
        .collect();
    let after = at + word.chars().count();
    if !code
        .get(after)
        .is_some_and(|character| character.is_whitespace())
    {
        return None;
    }
    match word.as_str() {
        "fn" => {
            let stop = (after..code.len())
                .find(|&index| code[index] == '{' || code[index] == ';')
                .unwrap_or(code.len());
            Some((format!("fn {}", parse::normalize(&code[after..stop])), stop))
        }
        "struct" => {
            let open = (after..code.len()).find(|&index| code[index] == '{')?;
            let close = parse::closing(code, open).min(code.len());
            Some((
                format!("struct {}", parse::normalize(&code[after..close])),
                close,
            ))
        }
        "const" | "override" | "alias" => {
            let stop = (after..code.len())
                .find(|&index| code[index] == ';')
                .unwrap_or(code.len());
            let text = parse::normalize(&code[after..stop]);
            let kept = match word.as_str() {
                "alias" => text.as_str(),
                _ => text
                    .split_once(" =")
                    .map_or(text.as_str(), |(head, _)| head),
            };
            Some((format!("{word} {kept}"), stop))
        }
        _ => None,
    }
}

fn brick_types(items: &BTreeSet<String>) -> Vec<&str> {
    items
        .iter()
        .filter_map(|line| line.strip_prefix("impl ")?.rsplit_once(" for "))
        .filter(|(trait_name, _)| trait_name.rsplit("::").next() == Some(BRICK_TRAIT))
        .map(|(_, target)| target)
        .collect()
}

fn about(line: &str, brick: &str) -> bool {
    if let Some(rest) = line.strip_prefix("impl ") {
        return rest
            .rsplit_once(" for ")
            .is_some_and(|(_, target)| target == brick);
    }
    line.split_whitespace()
        .find(|word| !ITEM_WORDS.contains(word))
        .and_then(|subject| subject.strip_prefix(brick))
        .is_some_and(|rest| !rest.starts_with(parse::identifier))
}

fn without_bricks(items: &BTreeSet<String>) -> BTreeSet<&String> {
    let bricks = brick_types(items);
    items
        .iter()
        .filter(|line| !bricks.iter().any(|brick| about(line, brick)))
        .collect()
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
    let mut crates: BTreeSet<String> =
        crate::bricks::brick_crates(repository, registered.iter().copied())
            .into_iter()
            .filter(|name| {
                repository
                    .of_crate(name)
                    .is_some_and(|package| package.own())
            })
            .collect();
    let declared = crate::gpu::contracts::catalog();
    crates.extend(
        declared
            .iter()
            .map(|place| place.declared_in().to_owned())
            .filter(|name| {
                repository
                    .of_crate(name)
                    .is_some_and(|package| package.own() && !package.core())
            }),
    );
    let listed: Vec<serde_json::Value> = faces(repository, &crates)?
        .into_iter()
        .map(|(name, face)| {
            let mut listed = json!({
                "crate": name,
                "items": face.items,
                "wgsl": face.wgsl,
                "faces": face.faces,
            });
            let places: serde_json::Map<String, serde_json::Value> = declared
                .iter()
                .filter(|place| place.declared_in() == name)
                .map(|place| (place.name.to_owned(), crate::bricks::place_face(place)))
                .collect();
            if !places.is_empty() {
                listed["places"] = serde_json::Value::Object(places);
            }
            listed
        })
        .collect();
    let text = serde_json::to_string_pretty(&json!({ "faces": listed }))
        .map_err(|error| format!("cannot write {FILE}: {error}"))?;
    Ok(vec![(FILE.to_owned(), text + "\n")])
}

struct Registered {
    kind: &'static str,
    place: String,
    sides: BTreeSet<&'static str>,
}

fn after_space(code: &[char], mut at: usize) -> usize {
    while code.get(at).is_some_and(|character| character.is_whitespace()) {
        at += 1;
    }
    at
}

fn angle_close(code: &[char], open: usize) -> usize {
    let mut depth = 0_usize;
    for (at, &character) in code.iter().enumerate().skip(open) {
        match character {
            '<' => depth += 1,
            '>' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return at;
                }
            }
            _ => {}
        }
    }
    code.len()
}

fn chained(code: &[char], mut at: usize) -> (Vec<String>, usize) {
    let mut calls = Vec::new();
    loop {
        let mut next = after_space(code, at);
        if code.get(next) == Some(&'.') {
            next += 1;
        } else if code.get(next) == Some(&':') && code.get(next + 1) == Some(&':') {
            next += 2;
        } else {
            return (calls, at);
        }
        let name: String = code[next..]
            .iter()
            .take_while(|&&character| parse::identifier(character))
            .collect();
        let open = after_space(code, next + name.chars().count());
        if name.is_empty() || code.get(open) != Some(&'(') {
            return (calls, at);
        }
        calls.push(name);
        at = parse::closing(code, open);
    }
}

fn registered_in(code: &[char]) -> Vec<Registered> {
    let mut found = Vec::new();
    for (kind, word) in [("fill", "Fill::<"), ("join", "Join::<")] {
        let needle: Vec<char> = word.chars().collect();
        let mut at = 0;
        while at + needle.len() <= code.len() {
            let starts = code[at..at + needle.len()] == needle[..]
                && (at == 0 || !parse::identifier(code[at - 1]));
            if !starts {
                at += 1;
                continue;
            }
            let open = at + needle.len() - 1;
            let close = angle_close(code, open);
            let named: String = code[open + 1..close].iter().collect();
            let place = named.rsplit("::").next().unwrap_or_default().trim();
            let (calls, end) = chained(code, close + 1);
            found.push(Registered {
                kind,
                place: place.to_owned(),
                sides: SIDES
                    .into_iter()
                    .filter(|side| calls.iter().any(|call| call == side))
                    .collect(),
            });
            at = end.max(at + 1);
        }
    }
    found
}

fn registrations(folder: &Path) -> Result<Vec<Registered>, String> {
    let mut files = Vec::new();
    crate::dependencies::rust_files(&folder.join("src"), &mut files);
    files.sort();
    let mut found = Vec::new();
    for file in files {
        found.extend(registered_in(&parse::code(&read(&file)?)));
    }
    Ok(found)
}

fn missing_places(wanted: &[Registered], offered: &[Registered]) -> Vec<String> {
    let mut missing = Vec::new();
    for registered in wanted {
        let shown = format!("{} {}", registered.kind, registered.place);
        let found = offered
            .iter()
            .find(|other| other.kind == registered.kind && other.place == registered.place);
        match found {
            None => missing.push(shown),
            Some(other) => missing.extend(
                registered
                    .sides
                    .difference(&other.sides)
                    .map(|side| format!("{shown} {side}")),
            ),
        }
    }
    missing
}

fn imported_from(module: &Contract, place: &str) -> BTreeSet<String> {
    let from = format!("package::{}::", place_module(place));
    module
        .imports
        .iter()
        .filter_map(|import| import.split_once("import ").map(|(_, rest)| rest))
        .flat_map(|rest| parse::expand(rest.trim_end().trim_end_matches(';')))
        .filter_map(|(path, _)| path.strip_prefix(from.as_str()).map(str::to_owned))
        .collect()
}

fn same(wanted: &Answer, found: &Answer) -> bool {
    wanted.constant == found.constant
        && wanted.returns == found.returns
        && wanted.parameters.len() == found.parameters.len()
        && wanted
            .parameters
            .iter()
            .zip(&found.parameters)
            .all(|((_, wanted), (_, found))| wanted == found)
}

fn unanswered(
    place: &str,
    wanted: &Contract,
    source: &str,
    functions: Option<&[&str]>,
) -> Vec<String> {
    let found = match contract(source) {
        Ok(found) => found,
        Err(error) => return vec![error],
    };
    let mut problems = Vec::new();
    let imported = imported_from(&found, place);
    let code = uncommented(source);
    let defined = found.names();
    let types = wanted
        .types
        .iter()
        .map(String::as_str)
        .chain(wanted.shared.iter().map(|answer| answer.name.as_str()));
    for ty in types {
        if defined.contains(&ty) {
            problems.push(format!(
                "defines {ty}, a type of place::{place}, where it imports it from there"
            ));
        } else if !imported.contains(ty) && rust::has_word(&code, ty) {
            problems.push(format!("uses {ty} without importing it from place::{place}"));
        }
    }
    let asked: Vec<&Answer> = match functions {
        None => wanted.answers.iter().collect(),
        Some(functions) => {
            for function in functions {
                let declared = wanted
                    .answers
                    .iter()
                    .any(|answer| !answer.constant && answer.name == *function);
                if !declared {
                    problems.push(format!(
                        "names {function}, which place::{place} does not declare"
                    ));
                }
            }
            wanted
                .answers
                .iter()
                .filter(|answer| functions.contains(&answer.name.as_str()))
                .collect()
        }
    };
    for answer in asked {
        let defines = found
            .shared
            .iter()
            .chain(&found.answers)
            .find(|other| other.name == answer.name);
        match defines {
            None => problems.push(format!("does not define {}", answer.signature())),
            Some(other) if !same(answer, other) => problems.push(format!(
                "defines {} where place::{place} declares {}",
                other.signature(),
                answer.signature()
            )),
            Some(_) => {}
        }
    }
    problems
}

fn answered_files(folder: &Path, offered: &[Registered]) -> Result<Vec<String>, String> {
    let declared = crate::gpu::contracts::catalog();
    let mut missing = Vec::new();
    for registered in offered
        .iter()
        .filter(|registered| registered.kind == "fill" && registered.sides.contains("wgsl"))
    {
        let found = declared.iter().find_map(|place| {
            let source = place.face.wgsl?;
            let named = (place.declared)();
            let short = named.rsplit("::").next().unwrap_or(named);
            (short == registered.place).then_some((place.name, source))
        });
        let Some((place, source)) = found else {
            continue;
        };
        let file = folder.join(format!("src/{FACES}/{place}.wgsl"));
        if !file.is_file() {
            missing.push(format!("wgsl src/{FACES}/{place}.wgsl"));
            continue;
        }
        let wanted = contract(source).map_err(|error| format!("place {place}: {error}"))?;
        missing.extend(
            unanswered(place, &wanted, &read(&file)?, None)
                .into_iter()
                .map(|problem| format!("wgsl {place}: {problem}")),
        );
    }
    Ok(missing)
}

fn replacements(repository: &Repository, problems: &mut Vec<Error>) -> Result<usize, String> {
    let mut compared = 0;
    for package in repository.own() {
        let Some(replaced) = package.replaces() else {
            continue;
        };
        let file = repository.relative(&package.manifest).unwrap_or_default();
        let Some(original) = repository
            .of_crate(replaced)
            .or_else(|| repository.package(replaced))
        else {
            problems.push(Error::new(ErrorCode::ReplacesUnknown, file).got(replaced));
            continue;
        };
        let offered = face(&package.folder)?;
        let wanted = face(&original.folder)?;
        let filled = registrations(&package.folder)?;
        let mut missing = missing_places(&registrations(&original.folder)?, &filled);
        missing.extend(answered_files(&package.folder, &filled)?);
        missing.extend(
            without_bricks(&wanted.items)
                .into_iter()
                .filter(|line| !offered.items.contains(*line))
                .cloned(),
        );
        compared += 1;
        if !missing.is_empty() {
            problems.push(
                Error::new(ErrorCode::FaceMissing, file)
                    .got(missing)
                    .nearest(&original.name),
            );
        }
    }
    Ok(compared)
}

fn answers(problems: &mut Vec<Error>) -> Result<usize, String> {
    let mut engine = engine::Engine::default();
    engine.app.add_bricks(engine::bricks::standard());
    let registry = engine
        .app
        .world()
        .get_resource::<BrickRegistry>()
        .ok_or("the standard engine registers no brick")?;
    let mut compared = 0;
    for brick in registry.bricks() {
        let modules: Vec<&ShaderModule> = brick
            .extension::<RenderBrick>()
            .into_iter()
            .flat_map(RenderBrick::modules)
            .collect();
        let mut answering: Vec<(&str, &str, Option<&[&str]>)> = Vec::new();
        for filled in &brick.fills {
            if let Some(module) = filled.wgsl {
                answering.push((filled.place, module, None));
            }
        }
        for joined in &brick.joins {
            if let Some(member) = joined.wgsl {
                answering.push((joined.place, member.module, Some(member.functions)));
            }
        }
        for (place, module, functions) in answering {
            compared += 1;
            let wanted = registry.place(place).and_then(|known| known.face.wgsl);
            let source = modules
                .iter()
                .find(|known| known.path == module)
                .map(|known| known.source);
            let missing = match (wanted, source) {
                (None, _) => vec![format!("place {place} has no WGSL side")],
                (_, None) => vec![format!("the brick registers no shader module {module}")],
                (Some(wanted), Some(source)) => match contract(wanted) {
                    Ok(wanted) => unanswered(place, &wanted, source, functions),
                    Err(error) => vec![error],
                },
            };
            if !missing.is_empty() {
                let got = json!({
                    "brick": brick.name,
                    "module": module,
                    "unanswered": missing,
                });
                problems.push(
                    Error::new(ErrorCode::FaceMissing, format!("$.places.{place}"))
                        .expected(Expected::Matching)
                        .got(got)
                        .nearest(place_module(place)),
                );
            }
        }
    }
    Ok(compared)
}

pub fn check(repository: &Repository) -> Result<(), String> {
    let mut problems = Vec::new();
    let compared = replacements(repository, &mut problems)?;
    let answered = answers(&mut problems)?;
    if problems.is_empty() {
        println!(
            "{}",
            json!({"event": "faces_offered", "replacements": compared, "answered": answered})
        );
        return Ok(());
    }
    crate::report(&problems)
}
