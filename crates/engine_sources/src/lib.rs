#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};
use toml_edit::{Document, Item};
use wgsl_parse::syntax::TranslationUnit;

const RESERVED: &[&str] = &["engine", "generated"];

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

    fn write(&self, out: &mut String, depth: usize, source: &str) {
        let indent = "    ".repeat(depth);
        if let Some(index) = self.module {
            out.push_str(&format!(
                "{indent}pub const MODULE: ::engine_core::Module = \
                 ::engine_core::Module::generated(&{source}, {index});\n"
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
            child.write(out, depth + 1, source);
            out.push_str(&format!("{indent}}}\n"));
        }
    }
}

pub fn emit() {
    let manifest = PathBuf::from(var("CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(var("OUT_DIR"));
    let package = var("CARGO_PKG_NAME");
    let root = workspace_root(&manifest);
    let src = manifest.join("src");
    let bricks = src.join("bricks");
    let cargo_toml = manifest.join("Cargo.toml");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-changed={}", cargo_toml.display());
    let game = read(&cargo_toml).is_some_and(|item| kind(&item) == Some("game"));
    let mut code = String::new();
    code.push_str("extern crate engine_ecs as bevy_ecs;\n");
    code.push_str(&source_code(
        &package,
        game,
        &src,
        Some(&bricks),
        "crate::SOURCE",
    ));
    fs::write(out.join("brick.rs"), code).expect("OUT_DIR is writable");

    let mut modules: Vec<String> = fs::read_dir(&bricks)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.path().join("mod.rs").is_file())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| !scratch(name))
        .collect();
    modules.sort();
    let mut shaders = shader_files(&src.join("shaders"));
    for module in &modules {
        shaders.extend(shader_files(&bricks.join(module).join("shaders")));
    }
    parse_shaders(&root, &out, &shaders);
    if modules.is_empty() {
        return;
    }
    let folder = out.join("bricks");
    fs::create_dir_all(&folder).expect("OUT_DIR is writable");
    let crate_name = package.replace('-', "_");
    for module in modules {
        let inside = bricks.join(&module);
        let named = format!("{crate_name}_{module}");
        let code = source_code(
            &named,
            game,
            &inside,
            None,
            &format!("crate::bricks::{module}::SOURCE"),
        );
        fs::write(folder.join(format!("{module}.rs")), code).expect("OUT_DIR is writable");
    }
}

fn shader_files(shaders: &Path) -> Vec<PathBuf> {
    let mut wgsl = Vec::new();
    list(shaders, &mut wgsl);
    wgsl.retain(|file| {
        file.extension()
            .is_some_and(|extension| extension == "wgsl")
    });
    wgsl.sort();
    wgsl
}

fn parse_shaders(root: &Path, out: &Path, files: &[PathBuf]) {
    let texts: Vec<(String, String)> = files
        .iter()
        .map(|file| {
            let text = fs::read_to_string(file).expect("a WGSL file");
            (relative(root, file), text)
        })
        .collect();
    let mut hash = Sha256::new();
    for (path, text) in &texts {
        hash.update(path.as_bytes());
        hash.update([0]);
        hash.update(text.as_bytes());
        hash.update([0]);
    }
    let hash: [u8; 32] = hash.finalize().into();
    let stamp = out.join("shaders.parsed");
    if fs::read(&stamp).is_ok_and(|parsed| parsed == hash) {
        return;
    }
    let mut parsed = true;
    for (path, text) in &texts {
        if let Err(error) = text.parse::<TranslationUnit>() {
            let (line, column) = line_column(text, error.span.start);
            println!("cargo::error={path}:{line}:{column}: {}", error.error);
            parsed = false;
        }
    }
    if parsed {
        fs::write(stamp, hash).expect("OUT_DIR is writable");
    }
}

fn line_column(text: &str, at: usize) -> (usize, usize) {
    let before = &text.as_bytes()[..at.min(text.len())];
    let lines = before.iter().filter(|byte| **byte == b'\n').count();
    let start = before
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |found| found + 1);
    (lines + 1, before.len() - start + 1)
}

fn source_code(
    package: &str,
    game: bool,
    folder: &Path,
    skipped: Option<&Path>,
    source: &str,
) -> String {
    let shaders = folder.join("shaders");
    let prefix = prefix(&shaders);
    let wgsl = shader_files(&shaders);

    let mut tree = Node::default();
    let mut texts = String::new();
    for (index, file) in wgsl.iter().enumerate() {
        let inside = file.strip_prefix(&shaders).expect("under shaders");
        let segments: Vec<String> = inside
            .with_extension("")
            .components()
            .map(|component| component.as_os_str().to_string_lossy().into_owned())
            .collect();
        let path = segments.join("::");
        let text = fs::read_to_string(file).expect("a WGSL file");
        let shown = file.display().to_string();
        texts.push_str(&format!(
            "    ::engine_core::ShaderText::generated({path:?}, include_str!({shown:?}), {shown:?}),\n"
        ));
        let node = tree.at(&segments);
        node.module = Some(index as u32);
        node.entries = entries(&text);
        node.variables = variables(&text);
    }
    if let Some(prefix) = prefix.as_ref().filter(|_| !wgsl.is_empty()) {
        let generated = generated_wgsl(folder, skipped);
        if !generated.is_empty() {
            texts.push_str(&format!(
                "    ::engine_core::ShaderText::generated({:?}, {generated:?}, \"\"),\n",
                format!("generated::{prefix}")
            ));
        }
    }

    let mut code = String::new();
    code.push_str(&format!(
        "pub(crate) static SOURCE: ::engine_core::Source = \
         ::engine_core::Source::generated({package:?}, {game}, &[\n{texts}]);\n"
    ));
    code.push_str("pub(crate) mod shaders {\n");
    code.push_str(
        "    #![allow(dead_code, non_upper_case_globals, \
         reason = \"every entry and variable of the brick's modules is named\")]\n",
    );
    tree.write(&mut code, 1, source);
    code.push_str("}\n");
    code
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

fn scratch(name: &str) -> bool {
    name.starts_with('.')
        || name.ends_with('~')
        || [".swp", ".tmp", ".orig", ".rej"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
}

fn list(folder: &Path, files: &mut Vec<PathBuf>) {
    let Ok(read) = fs::read_dir(folder) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if scratch(&name) {
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

fn generated_wgsl(folder: &Path, skipped: Option<&Path>) -> String {
    let mut files = Vec::new();
    list(folder, &mut files);
    files.retain(|file| {
        file.extension().is_some_and(|extension| extension == "rs")
            && skipped.is_none_or(|skipped| !file.starts_with(skipped))
    });
    files.sort();
    let mut wgsl = String::new();
    for file in &files {
        let text = fs::read_to_string(file).expect("a Rust file");
        for at in words(&text, "gpu_struct") {
            let Some(body) = text[at + "gpu_struct".len()..]
                .trim_start()
                .strip_prefix('!')
                .and_then(braced)
            else {
                continue;
            };
            wgsl.push_str(&struct_wgsl(body).unwrap_or_else(|| {
                panic!(
                    "{}: the generated module cannot read this gpu_struct!",
                    file.display()
                )
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
    ty.chars()
        .next()
        .is_some_and(|first| first.is_alphabetic() || first == '_')
        && ty.chars().all(|c| c.is_alphanumeric() || c == '_')
}
