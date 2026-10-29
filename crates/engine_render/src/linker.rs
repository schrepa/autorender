use std::borrow::Cow;
use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use engine_core::{Error, ErrorCode, Expected};
use wesl::Mangler as _;

use crate::bindings::ResolvedLayout;
use crate::shaders::Entry;

pub(crate) const UNPLACED_GROUP: u32 = 1_000_000;

#[derive(Clone, Copy, Debug)]
pub struct ShaderModule {
    pub path: &'static str,
    pub brick: &'static str,
    pub source: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub struct ComposeKey {
    pub module: String,
    pub entry: String,
    pub layout: String,
    pub features: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Composed {
    pub key: ComposeKey,
    pub text: String,
    pub hash: u64,
    pub overrides: Vec<(String, String)>,
}

impl Composed {
    pub fn constants<'a>(&'a self, values: &[(&str, f64)]) -> Vec<(&'a str, f64)> {
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

struct Source {
    path: String,
    brick: &'static str,
    text: Cow<'static, str>,
    overrides: Vec<String>,
    hex_floats: Vec<String>,
}

#[derive(Default)]
pub struct Linker {
    sources: Vec<Source>,
    features: Vec<(&'static str, bool)>,
    features_hash: u64,
    composed: BTreeMap<ComposeKey, Arc<Composed>>,
}

impl Linker {
    pub fn new(modules: &[ShaderModule]) -> Self {
        let mut linker = Self::default();
        for module in modules {
            linker.add(module.path.to_owned(), module.brick, Cow::Borrowed(module.source));
        }
        linker
    }

    pub fn add_generated(&mut self, path: String, brick: &'static str, source: String) {
        self.sources.retain(|known| known.path != path);
        self.composed.retain(|key, _| key.module != path);
        self.add(path, brick, Cow::Owned(source));
    }

    fn add(&mut self, path: String, brick: &'static str, text: Cow<'static, str>) {
        let overrides = declared_overrides(&text);
        let hex_floats = hex_floats(&text);
        self.sources.push(Source {
            path,
            brick,
            text,
            overrides,
            hex_floats,
        });
    }

    pub fn set_features(&mut self, features: Vec<(&'static str, bool)>) {
        let mut hasher = DefaultHasher::new();
        features.hash(&mut hasher);
        let hash = hasher.finish();
        if hash != self.features_hash || self.features != features {
            self.composed.clear();
        }
        self.features = features;
        self.features_hash = hash;
    }

    pub fn module_brick(&self, path: &str) -> Option<&'static str> {
        self.sources
            .iter()
            .find(|source| source.path == path)
            .map(|source| source.brick)
    }

    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.sources.iter().map(|source| source.path.as_str())
    }

    pub fn key(&self, entry: Entry, layout: &ResolvedLayout) -> ComposeKey {
        ComposeKey {
            module: entry.module.to_owned(),
            entry: entry.name.to_owned(),
            layout: layout.label.to_owned(),
            features: self.features_hash,
        }
    }

    pub fn preload(&mut self, composed: Vec<Composed>) {
        for composed in composed {
            if composed.key.features == self.features_hash {
                self.composed
                    .insert(composed.key.clone(), Arc::new(composed));
            }
        }
    }

    pub fn composed(&self) -> impl Iterator<Item = &Composed> {
        self.composed.values().map(|composed| &**composed)
    }

    pub fn compose(
        &mut self,
        entry: Entry,
        layout: &ResolvedLayout,
    ) -> Result<Arc<Composed>, Error> {
        let key = self.key(entry, layout);
        if let Some(composed) = self.composed.get(&key) {
            return Ok(Arc::clone(composed));
        }
        let path = format!("$.shaders.{}.{}", entry.module, entry.name);
        if !self.sources.iter().any(|source| source.path == entry.module) {
            return Err(Error::new(ErrorCode::UnknownModule, path.clone())
                .got(entry.module)
                .variants(self.paths().map(str::to_owned).collect::<Vec<_>>()));
        }
        if let Some(source) = self.sources.iter().find(|source| !source.hex_floats.is_empty()) {
            return Err(Error::new(ErrorCode::ShaderLink, format!("$.shaders.{}", source.path))
                .expected(Expected::Supported)
                .got(&source.hex_floats));
        }
        let failed = |error: &dyn std::fmt::Display| {
            Error::new(ErrorCode::ShaderLink, path.clone()).got(error.to_string())
        };
        let mut resolver = wesl::resolver::VirtualResolver::new();
        let mut unplaced = Vec::new();
        for source in &self.sources {
            let (text, missing) = place_bindings(&source.text, layout);
            unplaced.extend(missing);
            resolver.add_module(module_path(&source.path).map_err(|error| failed(&error))?, Cow::Owned(text));
        }
        let mut features = wesl::Features::new();
        features.default = wesl::Feature::Error;
        for &(name, enabled) in &self.features {
            features.set(name, enabled);
        }
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
            keep: Some(vec![entry.name.to_owned()]),
            keep_main: false,
            features,
            ..Default::default()
        };
        let main = module_path(entry.module).map_err(|error| failed(&error))?;
        let mut result =
            wesl::compile(&main, &options, &resolver).map_err(|error| failed(&error))?;
        let order = |module: &wesl::syntax::ModulePath| {
            let named = module.to_string();
            if *module == main {
                return 0;
            }
            1 + self
                .sources
                .iter()
                .position(|source| format!("package::{}", source.path) == named)
                .unwrap_or(self.sources.len())
        };
        result.modules.sort_by_key(|module| order(&module.path));
        let text = wesl::pass::link(&result.modules, Some(&result.used_items)).to_string();
        if text.contains(&format!("@group({UNPLACED_GROUP})")) {
            let used: Vec<String> = unplaced
                .into_iter()
                .filter(|name: &String| text.contains(name.as_str()))
                .collect();
            return Err(Error::new(ErrorCode::UnknownBinding, path.clone())
                .expected(Expected::Present)
                .got(used)
                .variants(layout.bindings.iter().map(|binding| binding.name)));
        }
        let mut overrides = Vec::new();
        for source in &self.sources {
            for name in &source.overrides {
                let mangled = if source.path == entry.module {
                    name.clone()
                } else {
                    let module = module_path(&source.path).map_err(|error| failed(&error))?;
                    wesl::mangler::EscapeMangler.mangle(&module, name)
                };
                if declares_override(&text, &mangled) {
                    overrides.push((name.clone(), mangled));
                }
            }
        }
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let composed = Arc::new(Composed {
            key: key.clone(),
            hash: hasher.finish(),
            text,
            overrides,
        });
        self.composed.insert(key, Arc::clone(&composed));
        Ok(composed)
    }
}

fn module_path(path: &str) -> Result<wesl::syntax::ModulePath, String> {
    format!("package::{path}")
        .parse::<wesl::syntax::ModulePath>()
        .map_err(|_| path.to_owned())
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

fn place_bindings(text: &str, layout: &ResolvedLayout) -> (String, Vec<String>) {
    let mut placed = String::with_capacity(text.len());
    let mut unplaced = Vec::new();
    for line in text.split_inclusive('\n') {
        match declared_binding(line) {
            Some((indent, name, rest)) => {
                let (group, number) = match layout.binding(&name) {
                    Some(binding) => (binding.group, binding.number),
                    None => {
                        unplaced.push(name);
                        (UNPLACED_GROUP, unplaced.len() as u32)
                    }
                };
                placed.push_str(indent);
                placed.push_str(&format!("@group({group}) @binding({number}) "));
                placed.push_str(rest);
            }
            None => placed.push_str(line),
        }
    }
    (placed, unplaced)
}

fn declared_binding(line: &str) -> Option<(&str, String, &str)> {
    let trimmed = line.trim_start();
    let indent = &line[..line.len() - trimmed.len()];
    let after_group = trimmed.strip_prefix("@group(")?;
    let after_group = &after_group[after_group.find(')')? + 1..];
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
    (!name.is_empty()).then_some((indent, name, rest))
}
