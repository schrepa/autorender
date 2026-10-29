use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use engine_render::bindings::{Kind, ResolvedBinding, ResolvedLayout, Stages};
use engine_render::faces::face_of;
use engine_render::gpu_layouts::{Constant, Layout};
use engine_render::linker::{Linker, ShaderModule, declared_bindings, unmangle};
use engine_render::shaders::{ENTRIES, MODULES, Stage};
use naga::valid::{Capabilities, ModuleInfo, ValidationFlags, Validator};

mod bindings;
mod eye_reads;
mod fades;
mod handovers;
mod installed;
mod inter_stage;

use installed::{Installed, core, generated_path, installed};

pub struct Checked {
    pub name: String,
    pub source: String,
    pub text: String,
    pub module: naga::Module,
    pub info: naga::valid::ModuleInfo,
}

pub(crate) fn plain(name: &str) -> String {
    unmangle(name).map_or_else(|| name.to_owned(), |(_, item)| item)
}

fn own(name: &str) -> bool {
    unmangle(name).is_none()
}

fn placement(sources: &[ShaderModule]) -> ResolvedLayout {
    let mut names: Vec<&'static str> = Vec::new();
    for module in sources {
        for name in declared_bindings(module.source) {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    ResolvedLayout {
        label: "engine/checks",
        groups: 1,
        bindings: names
            .into_iter()
            .zip(0..)
            .map(|(name, number)| ResolvedBinding {
                group: 0,
                number,
                name,
                kind: Kind::Write,
                stages: Stages::VERTEX_FRAGMENT_COMPUTE,
                owner: "engine/checks",
            })
            .collect(),
    }
}

fn validated(text: &str) -> Result<(naga::Module, ModuleInfo), String> {
    let module =
        naga::front::wgsl::parse_str(text).map_err(|error| error.emit_to_string(text))?;
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let info = validator
        .validate(&module)
        .map_err(|error| error.emit_to_string(text))?;
    Ok((module, info))
}

pub fn modules() -> Result<Vec<Checked>, String> {
    checked(&mut installed()?)
}

fn checked(installed: &mut Installed) -> Result<Vec<Checked>, String> {
    let layout = placement(&installed.sources);
    let mut checked = Vec::new();
    let mut failures = Vec::new();
    for source in installed.sources.clone() {
        let name = source.path.to_owned();
        let text = match installed.linker.compose_module(&name, &layout) {
            Ok(composed) => composed.text.clone(),
            Err(error) => {
                failures.push(format!("{name}: {error}"));
                continue;
            }
        };
        match validated(&text) {
            Ok((module, info)) => checked.push(Checked {
                name,
                source: source.source.to_owned(),
                text,
                module,
                info,
            }),
            Err(error) => failures.push(format!("{name}: {error}")),
        }
    }
    if failures.is_empty() {
        Ok(checked)
    } else {
        Err(failures.join("\n"))
    }
}

fn check_defaults(installed: &Installed) -> Result<usize, String> {
    let mut linker = Linker::new(MODULES);
    linker
        .add_modules(&installed.defaults)
        .map_err(|error| error.to_string())?;
    let mut sources = MODULES.to_vec();
    sources.extend(installed.defaults.iter().copied());
    let layout = placement(&sources);
    let mut failures = Vec::new();
    for default in &installed.defaults {
        let linked = linker
            .compose_module(default.path, &layout)
            .map_err(|error| error.to_string())
            .and_then(|composed| validated(&composed.text));
        if let Err(error) = linked {
            failures.push(format!("the default {}: {error}", default.path));
        }
    }
    if failures.is_empty() {
        Ok(installed.defaults.len())
    } else {
        Err(failures.join("\n"))
    }
}

fn struct_size(module: &naga::Module, name: &str) -> Option<u64> {
    module.types.iter().find_map(|(_, ty)| {
        let named = ty.name.as_deref() == Some(name);
        (named && matches!(ty.inner, naga::TypeInner::Struct { .. }))
            .then(|| u64::from(ty.inner.size(module.to_ctx())))
    })
}

fn check_pieces(installed: &Installed, modules: &[Checked]) -> Result<usize, String> {
    let mut errors = Vec::new();
    for &(brick, piece) in &installed.pieces {
        let size = modules
            .iter()
            .filter(|checked| checked.name == piece.module)
            .find_map(|checked| struct_size(&checked.module, piece.ty));
        match size {
            None => errors.push(format!(
                "{brick}'s piece {}: {} declares no struct {}",
                piece.name, piece.module, piece.ty
            )),
            Some(size) if size != piece.size => errors.push(format!(
                "{brick}'s piece {}: the struct {} spans {size} bytes where its PieceDecl says {}",
                piece.name, piece.ty, piece.size
            )),
            Some(_) => {}
        }
        let twinned = installed.mirrors.iter().any(|layout| {
            layout.mirrors.iter().any(|mirror| {
                mirror.name == piece.ty && mirror.module.is_none_or(|module| module == piece.module)
            })
        });
        if !twinned {
            errors.push(format!(
                "{brick}'s piece {}: no Rust twin mirrors the struct {} (gpu_structs!)",
                piece.name, piece.ty
            ));
        }
    }
    if errors.is_empty() {
        Ok(installed.pieces.len())
    } else {
        Err(errors.join("\n"))
    }
}

fn wgsl_struct(module: &naga::Module, name: &str) -> Option<Vec<(String, u32, u32)>> {
    module.types.iter().find_map(|(_, ty)| {
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            return None;
        };
        (ty.name.as_deref() == Some(name)).then(|| {
            members
                .iter()
                .map(|member| {
                    (
                        member.name.clone().unwrap_or_default(),
                        member.offset,
                        module.types[member.ty].inner.size(module.to_ctx()),
                    )
                })
                .collect()
        })
    })
}

fn compare(
    layout: &Layout,
    mirror: &str,
    module: &str,
    members: &[(String, u32, u32)],
) -> Vec<String> {
    let at = format!("{module} `{mirror}` mirroring {}", layout.rust);
    let mut errors = Vec::new();
    if members.len() > layout.fields.len() {
        errors.push(format!(
            "{at}: {} members beyond the Rust struct's {} fields",
            members.len(),
            layout.fields.len()
        ));
    }
    for ((name, offset, size), field) in members.iter().zip(layout.fields) {
        if name != field.name || *offset as usize != field.offset || *size as usize != field.size {
            errors.push(format!(
                "{at}: member `{name}` at {offset} ({size} bytes) where Rust has `{}` at {} ({} bytes)",
                field.name, field.offset, field.size
            ));
        }
    }
    errors
}

fn check_layouts(modules: &[Checked], layouts: &[&'static Layout]) -> Result<usize, String> {
    let mut errors = Vec::new();
    let mut mirrors = 0;
    for layout in layouts {
        for mirror in layout.mirrors {
            let mut found = false;
            for checked in modules {
                if mirror.module.is_some_and(|module| module != checked.name) {
                    continue;
                }
                if let Some(members) = wgsl_struct(&checked.module, mirror.name) {
                    found = true;
                    mirrors += 1;
                    errors.extend(compare(layout, mirror.name, &checked.name, &members));
                }
            }
            if !found {
                errors.push(format!(
                    "{}: no shader module {} declares its mirror `{}`",
                    layout.rust,
                    mirror.module.unwrap_or("of engine_render::shaders"),
                    mirror.name
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(mirrors)
    } else {
        Err(errors.join("\n"))
    }
}

fn stage_of(stage: naga::ShaderStage) -> Stages {
    match stage {
        naga::ShaderStage::Vertex => Stages::VERTEX,
        naga::ShaderStage::Fragment => Stages::FRAGMENT,
        _ => Stages::COMPUTE,
    }
}

fn declared_stage(stage: Stage) -> Stages {
    match stage {
        Stage::Vertex => Stages::VERTEX,
        Stage::Fragment => Stages::FRAGMENT,
        Stage::Compute => Stages::COMPUTE,
    }
}

fn check_entries(modules: &[Checked]) -> Result<usize, String> {
    let mut errors = Vec::new();
    let mut distinct = 0;
    for (index, entry) in ENTRIES.iter().enumerate() {
        if let Some(earlier) = ENTRIES[..index]
            .iter()
            .find(|earlier| earlier.module == entry.module && earlier.name == entry.name)
        {
            if earlier.stage != entry.stage {
                errors.push(format!(
                    "{} `{}` is declared {:?} and {:?} in entry_points!",
                    entry.module, entry.name, earlier.stage, entry.stage
                ));
            }
            continue;
        }
        distinct += 1;
        let Some(checked) = modules.iter().find(|checked| checked.name == entry.module) else {
            errors.push(format!(
                "entry point `{}`: {} is not a shader module",
                entry.name, entry.module
            ));
            continue;
        };
        match checked
            .module
            .entry_points
            .iter()
            .find(|point| point.name == entry.name)
        {
            None => errors.push(format!(
                "{} has no entry point `{}` (entry_points! declares one)",
                entry.module, entry.name
            )),
            Some(point) if stage_of(point.stage) != declared_stage(entry.stage) => {
                errors.push(format!(
                    "{} `{}` is a {:?} stage, declared {:?} in entry_points!",
                    entry.module, entry.name, point.stage, entry.stage
                ));
            }
            Some(_) => {}
        }
    }
    for checked in modules.iter().filter(|checked| core(&checked.name)) {
        for point in &checked.module.entry_points {
            if !ENTRIES
                .iter()
                .any(|entry| entry.module == checked.name && entry.name == point.name)
            {
                errors.push(format!(
                    "{} `{}` is in no entry_points! list (crates/engine_render/src/shaders.rs)",
                    checked.name, point.name
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(distinct)
    } else {
        Err(errors.join("\n"))
    }
}

fn spelled_out(word: &str) -> Option<[String; 4]> {
    let scalar = |suffix: u8| match suffix {
        b'f' => Some("f32"),
        b'i' => Some("i32"),
        b'u' => Some("u32"),
        b'h' => Some("f16"),
        _ => None,
    };
    let (generic, suffix) = match word.as_bytes() {
        [b'v', b'e', b'c', b'2'..=b'4', suffix] => (&word[..4], *suffix),
        [b'm', b'a', b't', b'2'..=b'4', b'x', b'2'..=b'4', suffix] => (&word[..6], *suffix),
        _ => return None,
    };
    Some([
        generic.to_string(),
        "<".to_string(),
        scalar(suffix)?.to_string(),
        ">".to_string(),
    ])
}

pub(crate) fn wgsl_spans(source: &str) -> Vec<(String, usize)> {
    const OPERATORS: [&str; 18] = [
        "->", "<<=", ">>=", "<=", ">=", "==", "!=", "&&", "||", "<<", ">>", "+=", "-=", "*=", "/=",
        "%=", "&=", "|=",
    ];
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let rest = &source[at..];
        let byte = bytes[at];
        if byte.is_ascii_whitespace() {
            at += 1;
        } else if rest.starts_with("//") {
            at += rest.find('\n').unwrap_or(rest.len());
        } else if rest.starts_with("/*") {
            let mut depth = 0;
            while at < bytes.len() {
                if source[at..].starts_with("/*") {
                    depth += 1;
                    at += 2;
                } else if source[at..].starts_with("*/") {
                    depth -= 1;
                    at += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    at += 1;
                }
            }
        } else if byte.is_ascii_alphanumeric()
            || byte == b'_'
            || (byte == b'.' && bytes.get(at + 1).is_some_and(u8::is_ascii_digit))
        {
            let number = byte.is_ascii_digit() || byte == b'.';
            let mut end = at + 1;
            while let Some(&next) = bytes.get(end) {
                let sign = number
                    && matches!(next, b'+' | b'-')
                    && matches!(bytes[end - 1], b'e' | b'E' | b'p' | b'P');
                if next.is_ascii_alphanumeric() || next == b'_' || (number && next == b'.') || sign
                {
                    end += 1;
                } else {
                    break;
                }
            }
            let word = &source[at..end];
            match spelled_out(word) {
                Some(spelled) => tokens.extend(spelled.map(|token| (token, at))),
                None => tokens.push((word.to_string(), at)),
            }
            at = end;
        } else {
            let length = OPERATORS
                .iter()
                .find(|operator| rest.starts_with(**operator))
                .map_or_else(
                    || rest.chars().next().map_or(1, char::len_utf8),
                    |operator| operator.len(),
                );
            tokens.push((rest[..length].to_string(), at));
            at += length;
        }
    }
    tokens
}

fn wgsl_tokens(source: &str) -> Vec<String> {
    wgsl_spans(source)
        .into_iter()
        .map(|(token, _)| token)
        .collect()
}

fn wgsl_functions(tokens: &[String]) -> Vec<(String, Vec<String>)> {
    let identifier = |token: &str| {
        token
            .chars()
            .next()
            .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
    };
    let mut functions = Vec::new();
    let mut at = 0;
    while at + 1 < tokens.len() {
        if tokens[at] != "fn" {
            at += 1;
            continue;
        }
        let name = tokens[at + 1].clone();
        let Some(open) = tokens[at..].iter().position(|token| token == "{") else {
            break;
        };
        let code_start = at + 2;
        let open = at + open;
        let mut depth = 0usize;
        let mut close = tokens.len() - 1;
        for (index, token) in tokens.iter().enumerate().skip(open) {
            match token.as_str() {
                "{" => depth += 1,
                "}" => {
                    depth -= 1;
                    if depth == 0 {
                        close = index;
                        break;
                    }
                }
                _ => {}
            }
        }
        let code = &tokens[code_start..=close];
        let signature = open - code_start;
        let mut renamed: Vec<(&str, String)> = Vec::new();
        let mut parentheses = 0usize;
        for (index, token) in code.iter().enumerate() {
            if index < signature {
                match token.as_str() {
                    "(" => parentheses += 1,
                    ")" => parentheses = parentheses.saturating_sub(1),
                    _ => {}
                }
                if parentheses == 1
                    && code.get(index + 1).is_some_and(|next| next == ":")
                    && identifier(token)
                {
                    renamed.push((token.as_str(), format!("$p{}", renamed.len())));
                }
            } else if matches!(token.as_str(), "let" | "var" | "const") {
                let mut local = index + 1;
                if code.get(local).is_some_and(|next| next == "<") {
                    while code.get(local).is_some_and(|next| next != ">") {
                        local += 1;
                    }
                    local += 1;
                }
                if let Some(name) = code.get(local).filter(|name| identifier(name)) {
                    renamed.push((name.as_str(), format!("$l{}", renamed.len())));
                }
            }
        }
        let normalized = code
            .iter()
            .enumerate()
            .map(|(index, token)| {
                let field = index > 0 && code[index - 1] == ".";
                renamed
                    .iter()
                    .find(|(name, _)| !field && *name == token.as_str())
                    .map_or_else(|| token.clone(), |(_, renaming)| renaming.clone())
            })
            .collect();
        functions.push((name, normalized));
        at = close + 1;
    }
    functions
}

fn check_duplicate_bodies(installed: &Installed) -> Result<usize, String> {
    let mut functions: Vec<(&str, String, Vec<String>)> = Vec::new();
    let mut errors = Vec::new();
    for module in installed.sources.iter().filter(|module| {
        !generated_path(module.path)
            && !installed
                .defaults
                .iter()
                .any(|default| default.path == module.path && default.brick == module.brick)
    }) {
        for (name, code) in wgsl_functions(&wgsl_tokens(module.source)) {
            if let Some((other_module, other, _)) = functions
                .iter()
                .find(|(_, _, other_code)| *other_code == code)
            {
                errors.push(format!(
                    "{}'s `{name}` is {other_module}'s `{other}` written again: keep one, in a module the others import (engine_render::shaders)",
                    module.path
                ));
            }
            functions.push((module.path, name, code));
        }
    }
    if errors.is_empty() {
        Ok(functions.len())
    } else {
        Err(errors.join("\n"))
    }
}

fn wgsl_constants(module: &naga::Module) -> Vec<(&str, f64)> {
    module
        .constants
        .iter()
        .filter_map(|(_, constant)| {
            let value = match module.global_expressions[constant.init] {
                naga::Expression::Literal(literal) => match literal {
                    naga::Literal::F64(value) | naga::Literal::AbstractFloat(value) => value,
                    naga::Literal::F32(value) => f64::from(value),
                    naga::Literal::U32(value) => f64::from(value),
                    naga::Literal::I32(value) => f64::from(value),
                    naga::Literal::U16(value) => f64::from(value),
                    naga::Literal::I16(value) => f64::from(value),
                    naga::Literal::AbstractInt(value) | naga::Literal::I64(value) => value as f64,
                    naga::Literal::U64(value) => value as f64,
                    _ => return None,
                },
                _ => return None,
            };
            Some((constant.name.as_deref()?, value))
        })
        .collect()
}

fn mirrored(checked: &Checked, mirror: &str) -> Option<f64> {
    if let Some(entry) = mirror.strip_prefix('@') {
        let entry = checked
            .module
            .entry_points
            .iter()
            .find(|point| point.name == entry)?;
        let side = entry.workgroup_size[0];
        let square = entry
            .workgroup_size
            .iter()
            .all(|&size| size == side || size == 1);
        return Some(if square { f64::from(side) } else { f64::NAN });
    }
    if let Some(name) = mirror.strip_prefix("size_of ") {
        return checked.module.types.iter().find_map(|(_, ty)| {
            (ty.name.as_deref() == Some(name))
                .then(|| f64::from(ty.inner.size(checked.module.to_ctx())))
        });
    }
    wgsl_constants(&checked.module)
        .into_iter()
        .find_map(|(name, value)| (name == mirror).then_some(value))
}

fn check_mirrored_constants(
    modules: &[Checked],
    constants: &[&'static Constant],
) -> (usize, Vec<String>) {
    let mut errors = Vec::new();
    let mut mirrors = 0;
    for constant in constants {
        for mirror in constant.mirrors {
            let mut found = false;
            for checked in modules {
                if mirror.module.is_some_and(|name| name != checked.name) {
                    continue;
                }
                let Some(value) = mirrored(checked, mirror.name) else {
                    continue;
                };
                found = true;
                mirrors += 1;
                if value != constant.value {
                    errors.push(format!(
                        "{} `{}` is {value} where its Rust mirror {} is {}",
                        checked.name, mirror.name, constant.rust, constant.value
                    ));
                }
            }
            if !found {
                errors.push(format!(
                    "{}: no shader module {} declares its mirror `{}`",
                    constant.rust,
                    mirror.module.unwrap_or("of engine_render::shaders"),
                    mirror.name
                ));
            }
        }
    }
    (mirrors, errors)
}

fn rust_constants(folders: &[PathBuf]) -> Vec<(PathBuf, usize, String)> {
    fn walk(directory: &Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(directory).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name != "gpu_layouts") {
                    walk(&path, files);
                }
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
    let mut files = Vec::new();
    for folder in folders {
        walk(folder, &mut files);
    }
    let mut found = Vec::new();
    for file in files {
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        for (index, line) in text.lines().enumerate() {
            let item = match line.strip_prefix("pub") {
                Some(rest) if rest.starts_with('(') => {
                    rest.split_once(") ").map_or(line, |(_, item)| item)
                }
                Some(rest) => rest.strip_prefix(' ').unwrap_or(line),
                None => line,
            };
            let Some(rest) = item.strip_prefix("const ") else {
                continue;
            };
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if rest[name.len()..].trim_start().starts_with(':') {
                found.push((file.clone(), index + 1, name));
            }
        }
    }
    found
}

fn check_constants(
    root: &Path,
    folders: &[PathBuf],
    modules: &[Checked],
    constants: &[&'static Constant],
) -> Result<usize, String> {
    let (mirrors, mut errors) = check_mirrored_constants(modules, constants);
    let mut by_name: BTreeMap<&str, Vec<(&str, f64)>> = BTreeMap::new();
    for checked in modules {
        for (name, value) in wgsl_constants(&checked.module) {
            if own(name) {
                by_name
                    .entry(name)
                    .or_default()
                    .push((checked.name.as_str(), value));
            }
        }
    }
    for (name, uses) in &by_name {
        if uses.iter().any(|&(_, value)| value != uses[0].1) {
            let spelled: Vec<String> = uses
                .iter()
                .map(|(module, value)| format!("{value} in {module}"))
                .collect();
            errors.push(format!(
                "WGSL `{name}` differs between modules ({}); name differing values apart",
                spelled.join(", ")
            ));
        }
    }
    for (file, line, name) in rust_constants(folders) {
        if by_name.contains_key(name.as_str()) {
            errors.push(format!(
                "{}:{line}: `{name}` shares its name with a WGSL constant; declare it once in \
                 engine_render::gpu_layouts::constants with its mirrors, or name it apart",
                file.strip_prefix(root).unwrap_or(&file).display(),
            ));
        }
    }
    if errors.is_empty() {
        Ok(mirrors)
    } else {
        Err(errors.join("\n"))
    }
}

fn word_counts(source: &str) -> engine_core::collections::HashMap<&str, usize> {
    let mut counts = engine_core::collections::HashMap::default();
    for line in source.lines() {
        let code = line.split("//").next().unwrap_or_default();
        for word in code.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
            if !word.is_empty() {
                *counts.entry(word).or_default() += 1;
            }
        }
    }
    counts
}

fn items(module: &naga::Module) -> Vec<(&'static str, &str)> {
    let mut items = Vec::new();
    items.extend(
        module
            .functions
            .iter()
            .filter_map(|(_, function)| Some(("function", function.name.as_deref()?))),
    );
    items.extend(
        module
            .global_variables
            .iter()
            .filter_map(|(_, global)| Some(("variable", global.name.as_deref()?))),
    );
    items.extend(
        module
            .overrides
            .iter()
            .filter_map(|(_, value)| Some(("override", value.name.as_deref()?))),
    );
    items.extend(module.types.iter().filter_map(|(_, ty)| match ty.inner {
        naga::TypeInner::Struct { .. } => Some(("struct", ty.name.as_deref()?)),
        _ => None,
    }));
    items
}

fn check_unused(modules: &[Checked]) -> Result<(), String> {
    let mut held: BTreeSet<(String, &'static str, String)> = BTreeSet::new();
    let mut used: BTreeSet<(String, &'static str, String)> = BTreeSet::new();
    let words: Vec<_> = modules
        .iter()
        .map(|checked| word_counts(&checked.source))
        .collect();
    for (index, checked) in modules.iter().enumerate() {
        let home =
            |name: &str| unmangle(name).unwrap_or_else(|| (checked.name.clone(), name.to_owned()));
        let published = face_of(&checked.name).is_some();
        for (kind, name) in items(&checked.module) {
            if own(name) && !published {
                held.insert((checked.name.clone(), kind, name.to_owned()));
            }
        }
        let mut compacted = checked.module.clone();
        naga::compact::compact(&mut compacted, naga::compact::KeepUnused::No);
        for (kind, name) in items(&compacted) {
            let (module, item) = home(name);
            used.insert((module, kind, item));
        }
        let imported = format!("package::{}::", checked.name);
        for name in checked
            .module
            .constants
            .iter()
            .filter_map(|(_, constant)| constant.name.as_deref())
            .filter(|name| own(name))
        {
            let item = (checked.name.clone(), "constant", name.to_owned());
            let here = words[index].get(name).copied().unwrap_or_default() > 1;
            let elsewhere = modules.iter().zip(&words).any(|(other, counts)| {
                other.name != checked.name
                    && other.source.contains(&imported)
                    && counts.contains_key(name)
            });
            if here || elsewhere {
                used.insert(item.clone());
            }
            if !published {
                held.insert(item);
            }
        }
    }
    let unused: Vec<String> = held
        .difference(&used)
        .map(|(module, kind, name)| {
            format!("{kind} `{name}` of {module} is used nowhere; remove it")
        })
        .collect();
    if unused.is_empty() {
        Ok(())
    } else {
        Err(unused.join("\n"))
    }
}

fn check_upload_structs(folders: &[PathBuf]) -> Result<(), String> {
    fn walk(directory: &Path, found: &mut Vec<String>) {
        for entry in std::fs::read_dir(directory).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name != "gpu_layouts") {
                    walk(&path, found);
                }
            } else if path.extension().is_some_and(|extension| extension == "rs")
                && path.file_stem().is_some_and(|stem| stem != "gpu_layouts")
                && std::fs::read_to_string(&path).is_ok_and(|text| {
                    text.contains("Pod") && text.contains("derive(") && derives_pod(&text)
                })
            {
                found.push(path.display().to_string());
            }
        }
    }
    fn derives_pod(text: &str) -> bool {
        text.split("#[derive(").skip(1).any(|rest| {
            rest.split(')')
                .next()
                .is_some_and(|list| list.split(',').any(|item| item.trim() == "Pod"))
        })
    }
    let mut found = Vec::new();
    for folder in folders {
        walk(folder, &mut found);
    }
    if found.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "upload structs belong in a gpu_layouts module whose layouts the crate registers, \
             where the GPU check sees them: {}",
            found.join(", ")
        ))
    }
}

fn folders(repository: &crate::Repository, installed: &Installed) -> Vec<PathBuf> {
    let mut folders = vec![repository.root.join("crates/engine_render/src")];
    for name in &installed.crates {
        if let Some(package) = repository.of_crate(name).filter(|package| package.own()) {
            let folder = package.folder.join("src");
            if !folders.contains(&folder) {
                folders.push(folder);
            }
        }
    }
    folders
}

pub fn check(repository: &crate::Repository) -> Result<(), String> {
    let root = repository.root.as_path();
    let mut installed = installed()?;
    let modules = checked(&mut installed)?;
    let folders = folders(repository, &installed);
    let imports = installed.import_rules();
    let imports = if imports.is_empty() {
        Ok(())
    } else {
        Err(imports.join("\n"))
    };
    let (resolved, unresolved) = bindings::resolve(&installed);
    let mirrors = check_layouts(&modules, &installed.mirrors);
    let pieces = check_pieces(&installed, &modules);
    let defaults = check_defaults(&installed);
    let entry_points = bindings::check(&modules, &resolved, &installed);
    let entries = check_entries(&modules);
    let inter_stage = inter_stage::check(&modules);
    let pipeline_layouts = bindings::check_layouts(&resolved, unresolved, &installed);
    let functions = check_duplicate_bodies(&installed);
    let wgsl_constants = check_constants(root, &folders, &modules, &installed.constants);
    let unused = check_unused(&modules);
    let structs = check_upload_structs(&folders);
    let handovers = handovers::check(&modules);
    let fades = fades::check(&modules);
    let eye_reads = eye_reads::check(repository);
    let errors: Vec<String> = [
        imports.as_ref().err(),
        pieces.as_ref().err(),
        defaults.as_ref().err(),
        mirrors.as_ref().err(),
        entry_points.as_ref().err(),
        entries.as_ref().err(),
        inter_stage.as_ref().err(),
        pipeline_layouts.as_ref().err(),
        functions.as_ref().err(),
        wgsl_constants.as_ref().err(),
        unused.as_ref().err(),
        structs.as_ref().err(),
        handovers.as_ref().err(),
        fades.as_ref().err(),
        eye_reads.as_ref().err(),
    ]
    .into_iter()
    .flatten()
    .cloned()
    .collect();
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    let record = serde_json::json!({
        "event": "shaders_checked",
        "modules": modules.len(),
        "defaulted": installed.defaulted,
        "defaults": defaults?,
        "pieces": pieces?,
        "mirrors": mirrors?,
        "upload_structs": installed.mirrors.len(),
        "binding_tables": installed.tables.len(),
        "entry_points": entry_points?,
        "staged_entries": entries?,
        "fragment_stages": inter_stage?,
        "shared_bindings": installed
            .shared
            .groups()
            .map(|(_, bindings)| bindings.len())
            .sum::<usize>(),
        "pipeline_layouts": pipeline_layouts?,
        "stage_sums": bindings::sums(&resolved),
        "functions": functions?,
        "wgsl_constants": wgsl_constants?,
        "rust_constants": installed.constants.len(),
        "shader_decisions": handovers?,
        "handovers": fades?,
        "eye_reads": eye_reads?,
    });
    println!("{record}");
    Ok(())
}
