#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};
use toml_edit::{Document, Item};

const RESERVED: &[&str] = &["engine", "generated", "face", "place"];

const KEYWORDS: &[&str] = &[
    "as", "async", "await", "box", "break", "const", "continue", "dyn", "else", "enum", "extern",
    "false", "fn", "for", "gen", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut",
    "pub", "ref", "return", "static", "struct", "trait", "true", "try", "type", "unsafe", "use",
    "where", "while", "yield",
];

#[derive(Default)]
struct Node {
    module: Option<u32>,
    entries: Vec<String>,
    variables: Vec<String>,
    children: BTreeMap<String, Node>,
}

impl Node {
    fn at(&mut self, path: &[String]) -> &mut Node {
        let mut node = self;
        for segment in path {
            node = node.children.entry(segment.clone()).or_default();
        }
        node
    }

    fn write(&self, out: &mut String, depth: usize) {
        let indent = "    ".repeat(depth);
        if let Some(index) = self.module {
            out.push_str(&format!(
                "{indent}pub const MODULE: ::engine_core::Module = \
                 ::engine_core::Module::generated(&crate::SOURCE, {index});\n"
            ));
            let mut named = vec!["MODULE".to_owned()];
            for (names, kind) in [(&self.entries, "Entry"), (&self.variables, "Var")] {
                for name in names {
                    let constant = name.to_uppercase();
                    if named.contains(&constant) {
                        continue;
                    }
                    out.push_str(&format!(
                        "{indent}pub const {constant}: ::engine_core::{kind} = \
                         ::engine_core::{kind}::generated(MODULE, {name:?});\n"
                    ));
                    named.push(constant);
                }
            }
        }
        for (name, child) in &self.children {
            let name = if KEYWORDS.contains(&name.as_str()) {
                format!("r#{name}")
            } else {
                name.clone()
            };
            out.push_str(&format!("{indent}pub mod {name} {{\n"));
            child.write(out, depth + 1);
            out.push_str(&format!("{indent}}}\n"));
        }
    }
}

pub fn emit() {
    let manifest = PathBuf::from(var("CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(var("OUT_DIR"));
    let package = var("CARGO_PKG_NAME");
    let shaders = manifest.join("src").join("shaders");
    let prefix = prefix(&shaders);
    let root = workspace_root(&manifest);
    let hash = digest(&package, &manifest, &root, false);

    let mut wgsl = Vec::new();
    if prefix.is_some() {
        list(&shaders, &mut wgsl);
    }
    wgsl.retain(|file| file.extension().is_some_and(|extension| extension == "wgsl"));
    wgsl.sort();

    let mut tree = Node::default();
    let mut texts = String::new();
    for (index, file) in wgsl.iter().enumerate() {
        let inside = file.strip_prefix(&shaders).expect("under src/shaders");
        let segments: Vec<String> = inside
            .with_extension("")
            .components()
            .map(|component| component.as_os_str().to_string_lossy().into_owned())
            .collect();
        let path = segments.join("::");
        let text = fs::read_to_string(file).expect("a WGSL file");
        texts.push_str(&format!(
            "    ::engine_core::ShaderText::generated({path:?}, include_str!({:?})),\n",
            file.display().to_string()
        ));
        let node = tree.at(&segments);
        node.module = Some(index as u32);
        node.entries = entries(&text);
        node.variables = variables(&text);
    }
    if let Some(prefix) = &prefix {
        let generated = generated_wgsl(&manifest);
        if !generated.is_empty() {
            texts.push_str(&format!(
                "    ::engine_core::ShaderText::generated({:?}, {generated:?}),\n",
                format!("generated::{prefix}")
            ));
        }
    }

    let mut code = String::new();
    code.push_str("extern crate engine_ecs as bevy_ecs;\n");
    code.push_str(&format!(
        "pub(crate) static SOURCE: ::engine_core::Source = \
         ::engine_core::Source::generated({package:?}, {hash:?}, &[\n{texts}]);\n"
    ));
    code.push_str("pub(crate) mod shaders {\n");
    code.push_str(
        "    #![allow(dead_code, non_upper_case_globals, \
         reason = \"every entry and variable of the brick's modules is named\")]\n",
    );
    tree.write(&mut code, 1);
    code.push_str("}\n");
    fs::write(out.join("brick.rs"), code).expect("OUT_DIR is writable");
}

pub fn emit_of(named: &[(&str, &str)]) {
    let manifest = PathBuf::from(var("CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(var("OUT_DIR"));
    let root = workspace_root(&manifest);
    let workspace = read(&root.join("Cargo.toml"));
    let mut code = String::new();
    for &(constant, package) in named {
        let path = workspace
            .as_ref()
            .and_then(|item| item.get("workspace")?.get("dependencies")?.get(package))
            .and_then(|entry| entry.get("path"))
            .and_then(Item::as_str)
            .unwrap_or_else(|| panic!("the workspace names no path for {package}"));
        let folder = normal(&root.join(path));
        let hash = digest(package, &folder, &root, true);
        code.push_str(&format!("pub(crate) const {constant}: [u8; 32] = {hash:?};\n"));
    }
    fs::write(out.join("sources.rs"), code).expect("OUT_DIR is writable");
}

fn digest(package: &str, manifest: &Path, root: &Path, every: bool) -> [u8; 32] {
    let mut crates = BTreeSet::new();
    crates.insert(normal(manifest));
    reach(manifest, root, &mut crates, every);

    let mut hash = Sha256::new();
    hash.update(package.as_bytes());
    hash.update([0]);
    for dir in &crates {
        let mut files = Vec::new();
        list(&dir.join("src"), &mut files);
        files.push(dir.join("Cargo.toml"));
        files.sort();
        println!("cargo:rerun-if-changed={}", dir.join("src").display());
        println!("cargo:rerun-if-changed={}", dir.join("Cargo.toml").display());
        for file in &files {
            hash.update(relative(root, file).as_bytes());
            hash.update([0]);
            hash.update(text_bytes(file));
            hash.update([0]);
        }
    }
    let lock = root.join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    hash.update(text_bytes(&lock));
    hash.finalize().into()
}

fn var(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is set by cargo"))
}

fn prefix(shaders: &Path) -> Option<String> {
    let read = fs::read_dir(shaders).ok()?;
    let mut tops = Vec::new();
    for entry in read.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if scratch(&name) {
            continue;
        }
        if !entry.path().is_dir() {
            panic!("src/shaders/{name}: a brick's WGSL lives in its one folder under src/shaders/");
        }
        tops.push(name);
    }
    tops.sort();
    match tops.as_slice() {
        [] => None,
        [one] if RESERVED.contains(&one.as_str()) => {
            panic!("src/shaders/{one}: the prefix {one} is core's")
        }
        [one] => Some(one.clone()),
        many => panic!(
            "src/shaders/ holds {}: a brick's modules share one prefix folder",
            many.join(", ")
        ),
    }
}

fn read(path: &Path) -> Option<Item> {
    let text = fs::read_to_string(path).ok()?;
    Document::parse(text).ok().map(Document::into_item)
}

fn workspace_root(start: &Path) -> PathBuf {
    let mut at = Some(start);
    while let Some(dir) = at {
        if read(&dir.join("Cargo.toml")).is_some_and(|manifest| manifest.get("workspace").is_some())
        {
            return normal(dir);
        }
        at = dir.parent();
    }
    normal(start)
}

fn kind(manifest: &Item) -> Option<&str> {
    manifest
        .get("package")?
        .get("metadata")?
        .get("engine")?
        .get("kind")?
        .as_str()
}

fn dependency_tables(manifest: &Item) -> Vec<&Item> {
    let mut tables: Vec<&Item> = manifest.get("dependencies").into_iter().collect();
    if let Some(targets) = manifest.get("target").and_then(Item::as_table_like) {
        tables.extend(targets.iter().filter_map(|(_, target)| target.get("dependencies")));
    }
    tables
}

fn reach(dir: &Path, root: &Path, found: &mut BTreeSet<PathBuf>, every: bool) {
    let Some(manifest) = read(&dir.join("Cargo.toml")) else {
        return;
    };
    let workspace = read(&root.join("Cargo.toml"));
    for table in dependency_tables(&manifest) {
        let Some(dependencies) = table.as_table_like() else {
            continue;
        };
        for (key, entry) in dependencies.iter() {
            let inherited = entry.get("workspace").and_then(Item::as_bool) == Some(true);
            let (source, base) = if inherited {
                let declared = workspace
                    .as_ref()
                    .and_then(|item| item.get("workspace")?.get("dependencies")?.get(key));
                (declared, root)
            } else {
                (Some(entry), dir)
            };
            let Some(path) = source
                .and_then(|item| item.get("path"))
                .and_then(Item::as_str)
            else {
                continue;
            };
            let target = normal(&base.join(path));
            let Some(reached) = read(&target.join("Cargo.toml")) else {
                continue;
            };
            let followed = every || matches!(kind(&reached), Some("support" | "place"));
            if followed && found.insert(target.clone()) {
                reach(&target, root, found, every);
            }
        }
    }
}

fn normal(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                out.push(component.as_os_str());
            }
        }
    }
    out
}

fn text_bytes(file: &Path) -> Vec<u8> {
    let bytes = fs::read(file).unwrap_or_default();
    let mut out = Vec::with_capacity(bytes.len());
    for (at, byte) in bytes.iter().enumerate() {
        if *byte == b'\r' && bytes.get(at + 1) == Some(&b'\n') {
            continue;
        }
        out.push(*byte);
    }
    out
}

fn scratch(name: &str) -> bool {
    name.starts_with('.')
        || name.ends_with('~')
        || [".swp", ".tmp", ".orig", ".rej"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
}

fn test_code(name: &str) -> bool {
    let stem = name.strip_suffix(".rs").unwrap_or(name);
    ["smoke", "smokes", "tests"].contains(&stem)
        || stem.ends_with("_smoke")
        || stem.ends_with("_smokes")
}

fn list(folder: &Path, files: &mut Vec<PathBuf>) {
    let Ok(read) = fs::read_dir(folder) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if scratch(&name) || test_code(&name) {
            continue;
        }
        if path.is_dir() {
            list(&path, files);
        } else {
            files.push(path);
        }
    }
}

fn relative(root: &Path, file: &Path) -> String {
    normal(file)
        .strip_prefix(root)
        .unwrap_or(file)
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

fn identifier(text: &str) -> String {
    text.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

fn header(text: &str, at: usize) -> &str {
    let start = text[..at]
        .rfind([';', '{', '}'])
        .map_or(0, |found| found + 1);
    &text[start..at]
}

fn words(text: &str, word: &str) -> Vec<usize> {
    let bytes = text.as_bytes();
    text.match_indices(word)
        .map(|(at, _)| at)
        .filter(|&at| {
            let before =
                at == 0 || !(bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_');
            let after = bytes
                .get(at + word.len())
                .is_none_or(|c| !(c.is_ascii_alphanumeric() || *c == b'_'));
            before && after
        })
        .collect()
}

fn entries(text: &str) -> Vec<String> {
    words(text, "fn")
        .into_iter()
        .filter(|&at| {
            let head = header(text, at);
            head.contains("@vertex") || head.contains("@fragment") || head.contains("@compute")
        })
        .map(|at| identifier(text[at + 2..].trim_start()))
        .filter(|name| !name.is_empty())
        .collect()
}

fn variables(text: &str) -> Vec<String> {
    words(text, "var")
        .into_iter()
        .filter(|&at| header(text, at).contains("@group("))
        .map(|at| {
            let mut rest = text[at + 3..].trim_start();
            if rest.starts_with('<') {
                rest = rest
                    .find('>')
                    .map_or("", |close| &rest[close + 1..])
                    .trim_start();
            }
            identifier(rest)
        })
        .filter(|name| !name.is_empty())
        .collect()
}

fn generated_wgsl(manifest: &Path) -> String {
    let mut files = Vec::new();
    list(&manifest.join("src"), &mut files);
    files.retain(|file| file.extension().is_some_and(|extension| extension == "rs"));
    files.sort();
    let mut wgsl = String::new();
    for file in &files {
        let text = fs::read_to_string(file).expect("a Rust file");
        let mut found: Vec<(usize, &str)> = ["gpu_struct", "gpu_consts"]
            .into_iter()
            .flat_map(|name| words(&text, name).into_iter().map(move |at| (at, name)))
            .collect();
        found.sort_unstable();
        for (at, name) in found {
            let Some(body) = text[at + name.len()..]
                .trim_start()
                .strip_prefix('!')
                .and_then(braced)
            else {
                continue;
            };
            let declared = if name == "gpu_struct" {
                struct_wgsl(body)
            } else {
                consts_wgsl(body)
            };
            wgsl.push_str(&declared.unwrap_or_else(|| {
                panic!("{}: the generated module cannot read this {name}!", file.display())
            }));
        }
    }
    wgsl
}

fn braced(text: &str) -> Option<&str> {
    let text = text.trim_start();
    let open = text.chars().next()?;
    let close = match open {
        '{' => '}',
        '(' => ')',
        '[' => ']',
        _ => return None,
    };
    let inner = &text[1..];
    let mut depth = 0usize;
    for (at, character) in inner.char_indices() {
        if character == open {
            depth += 1;
        } else if character == close {
            if depth == 0 {
                return Some(&inner[..at]);
            }
            depth -= 1;
        }
    }
    None
}

fn without_attributes(text: &str) -> &str {
    let mut rest = text.trim_start();
    while let Some(attribute) = rest.strip_prefix('#') {
        let Some(inner) = braced(attribute) else {
            return rest;
        };
        let start = attribute.len() - attribute.trim_start().len();
        rest = attribute[start + inner.len() + 2..].trim_start();
    }
    rest
}

fn without_visibility(text: &str) -> &str {
    let text = text.trim_start();
    let Some(rest) = text.strip_prefix("pub") else {
        return text;
    };
    let rest = rest.trim_start();
    match rest.strip_prefix('(') {
        Some(scoped) => scoped
            .find(')')
            .map_or(rest, |close| scoped[close + 1..].trim_start()),
        None => rest,
    }
}

fn struct_wgsl(body: &str) -> Option<String> {
    let rest = without_visibility(without_attributes(body))
        .strip_prefix("struct")?
        .trim_start();
    let name = identifier(rest);
    let fields = braced(&rest[name.len()..])?;
    let mut wgsl = format!("struct {name} {{");
    for field in fields.split(',') {
        let field = without_visibility(field);
        if field.is_empty() {
            continue;
        }
        let (field, ty) = field.split_once(':')?;
        let ty: String = ty.chars().filter(|c| !c.is_whitespace()).collect();
        wgsl.push_str(&format!(" {}: {},", field.trim(), wgsl_type(&ty)?));
    }
    wgsl.push_str(" }\n");
    Some(wgsl)
}

fn wgsl_type(ty: &str) -> Option<String> {
    if matches!(ty, "f32" | "u32" | "i32") {
        return Some(ty.to_owned());
    }
    if ty == "[[f32;4];4]" {
        return Some("mat4x4<f32>".to_owned());
    }
    if named(ty) {
        return Some(ty.to_owned());
    }
    let (element, count) = ty.strip_prefix('[')?.strip_suffix(']')?.rsplit_once(';')?;
    match element {
        "f32" | "u32" | "i32" if matches!(count, "2" | "3" | "4") => {
            Some(format!("vec{count}<{element}>"))
        }
        "[[f32;4];4]" => Some(format!("array<mat4x4<f32>, {count}>")),
        "[f32;4]" | "[u32;4]" | "[i32;4]" => {
            Some(format!("array<vec4<{}>, {count}>", &element[1..4]))
        }
        _ if named(element) => Some(format!("array<{element}, {count}>")),
        _ => None,
    }
}

fn named(ty: &str) -> bool {
    ty.chars().next().is_some_and(|first| first.is_alphabetic() || first == '_')
        && ty.chars().all(|c| c.is_alphanumeric() || c == '_')
}

fn consts_wgsl(body: &str) -> Option<String> {
    let mut wgsl = String::new();
    for declaration in body.split(';') {
        let declaration = without_visibility(declaration);
        if declaration.is_empty() {
            continue;
        }
        let rest = declaration.strip_prefix("const")?;
        let (name, rest) = rest.split_once(':')?;
        let (ty, value) = rest.split_once('=')?;
        let mut value = value.split_whitespace().collect::<Vec<_>>().join(" ");
        for (rust, written) in LIMITS {
            value = value.replace(rust, written);
        }
        if value.contains("::") {
            return None;
        }
        wgsl.push_str(&format!("const {}: {} = {value};\n", name.trim(), ty.trim()));
    }
    Some(wgsl)
}

const LIMITS: [(&str, &str); 4] = [
    ("u32::MAX", "4294967295"),
    ("u32::MIN", "0"),
    ("i32::MAX", "2147483647"),
    ("i32::MIN", "(-2147483648)"),
];
