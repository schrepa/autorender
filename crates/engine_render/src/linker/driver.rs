use engine_core::collections::HashMap;
use std::borrow::Cow;

use wesl::error::{Error, ResolveError, UsageError};
use wesl::pass::{self, CompileResult, CompilerDriver, Module, UsedItems, Visit};
use wesl::syntax::{
    Attribute, DeclarationKind, GlobalDeclaration, Ident, ModulePath, TranslationUnit,
    TypeExpression,
};
use wesl::{Feature, Features, Mangler as _, Resolver};

pub(super) struct Prepared {
    pub(super) at: HashMap<ModulePath, usize>,
    pub(super) texts: Vec<String>,
    pub(super) writes: Vec<bool>,
    pub(super) named: HashMap<String, usize>,
    pub(super) overrides: Vec<Vec<(String, String)>>,
    pub(super) declared: Vec<Vec<String>>,
}

#[derive(Default)]
pub(super) struct Parsed {
    stamp: u64,
    units: HashMap<(ModulePath, Option<String>), Unit>,
}

struct Unit {
    source: String,
    stamp: u64,
    syntax: TranslationUnit,
}

impl Parsed {
    pub(super) fn changed(&mut self) {
        self.stamp += 1;
    }
}

#[derive(Default)]
struct Names(Vec<(Ident, String)>);

impl Drop for Names {
    fn drop(&mut self) {
        for (ident, name) in &mut self.0 {
            ident.rename(std::mem::take(name));
        }
    }
}

struct Texts<'a> {
    prepared: &'a Prepared,
    writes: Option<&'a str>,
}

impl Texts<'_> {
    fn written(&self, at: usize) -> Option<&str> {
        self.writes
            .filter(|_| self.prepared.writes.get(at).copied().unwrap_or(false))
    }
}

impl Resolver for Texts<'_> {
    fn resolve_source<'b>(&'b self, path: &ModulePath) -> Result<Cow<'b, str>, ResolveError> {
        let at = self.prepared.at.get(path).copied().ok_or_else(|| {
            ResolveError::ModuleNotFound(path.clone(), "virtual module".to_string())
        })?;
        Ok(Cow::Borrowed(
            self.written(at)
                .unwrap_or_else(|| self.prepared.texts[at].as_str()),
        ))
    }
}

struct Linking<'a> {
    main: &'a ModulePath,
    kept: Option<&'a str>,
    texts: Texts<'a>,
    parsed: &'a mut Parsed,
    features: Features,
    names: Names,
}

fn declared(declaration: &GlobalDeclaration) -> Option<&Ident> {
    match declaration {
        GlobalDeclaration::Declaration(item) => Some(&item.ident),
        GlobalDeclaration::TypeAlias(item) => Some(&item.ident),
        GlobalDeclaration::Struct(item) => Some(&item.ident),
        GlobalDeclaration::Function(item) => Some(&item.ident),
        GlobalDeclaration::Void
        | GlobalDeclaration::ConstAssert(_)
        | GlobalDeclaration::Compound(_) => None,
    }
}

impl CompilerDriver for Linking<'_> {
    fn main_path(&self) -> &ModulePath {
        self.main
    }

    #[expect(
        clippy::disallowed_types,
        reason = "the wesl driver trait returns this set"
    )]
    fn main_entry_points(
        &self,
        main_module: &TranslationUnit,
    ) -> Result<std::collections::HashSet<Ident>, Error> {
        let mut idents = main_module
            .global_declarations
            .iter()
            .filter_map(|declaration| declared(declaration.node()).cloned());
        match self.kept {
            None => Ok(idents.collect()),
            Some(name) => idents
                .find(|ident| *ident.name() == name)
                .map(|ident| std::collections::HashSet::from([ident]))
                .ok_or_else(|| UsageError::NotFound(self.main.clone(), name.to_string()).into()),
        }
    }

    fn load_module(&mut self, path: &ModulePath) -> Result<TranslationUnit, Error> {
        let Some(at) = self.texts.prepared.at.get(path).copied() else {
            return pass::load_module(path, &self.texts);
        };
        let written = self.texts.written(at);
        let source = written.unwrap_or(&self.texts.prepared.texts[at]);
        let key = (path.clone(), written.map(str::to_owned));
        let stamp = self.parsed.stamp;
        let kept = match self.parsed.units.get_mut(&key) {
            Some(unit) if unit.stamp == stamp => true,
            Some(unit) if unit.source == source => {
                unit.stamp = stamp;
                true
            }
            Some(_) | None => false,
        };
        if !kept {
            let mut syntax = pass::load_module(path, &self.texts)?;
            pass::condcomp(&mut syntax, &self.features)?;
            pass::retarget_idents(&mut syntax);
            let source = source.to_owned();
            let unit = Unit {
                source,
                stamp,
                syntax,
            };
            self.parsed.units.insert(key.clone(), unit);
        }
        let module = self.parsed.units[&key].syntax.clone();
        for declaration in &module.global_declarations {
            if let Some(ident) = declared(declaration.node()) {
                self.names.0.push((ident.clone(), ident.name().clone()));
            }
        }
        Ok(module)
    }

    fn link(
        &self,
        modules: &mut Vec<Module>,
        used_items: &UsedItems,
    ) -> Result<TranslationUnit, Error> {
        pass::retarget_modules(modules, used_items, &self.texts);
        for module in modules.iter_mut() {
            if module.path != *self.main {
                pass::mangle(
                    &mut module.syntax,
                    &module.path,
                    &wesl::mangler::EscapeMangler,
                );
            }
        }
        Ok(TranslationUnit::default())
    }
}

fn sort(modules: &mut [Module], main: &ModulePath, prepared: &Prepared) {
    modules.sort_by_key(|module| {
        if module.path == *main {
            return 0;
        }
        1 + prepared
            .named
            .get(&module.path.to_string())
            .copied()
            .unwrap_or(prepared.texts.len())
    });
}

#[derive(Default)]
pub(super) struct Linked {
    pub(super) text: String,
    pub(super) bindings: Vec<String>,
    pub(super) overrides: Vec<String>,
    pub(super) functions: Vec<String>,
}

impl Linked {
    fn note(&mut self, declaration: &GlobalDeclaration, name: &str) {
        let list = match declaration {
            GlobalDeclaration::Declaration(item) if item.kind == DeclarationKind::Override => {
                &mut self.overrides
            }
            GlobalDeclaration::Declaration(item)
                if item
                    .attributes
                    .iter()
                    .any(|attribute| matches!(attribute.node(), Attribute::Binding(_))) =>
            {
                &mut self.bindings
            }
            GlobalDeclaration::Function(_) => &mut self.functions,
            GlobalDeclaration::Void
            | GlobalDeclaration::Declaration(_)
            | GlobalDeclaration::TypeAlias(_)
            | GlobalDeclaration::Struct(_)
            | GlobalDeclaration::ConstAssert(_)
            | GlobalDeclaration::Compound(_) => return,
        };
        list.push(name.to_owned());
    }
}

fn linked(unit: &TranslationUnit) -> Linked {
    let mut linked = Linked {
        text: unit.to_string(),
        ..Linked::default()
    };
    for declaration in &unit.global_declarations {
        if let Some(ident) = declared(declaration.node()) {
            linked.note(declaration.node(), &ident.name());
        }
    }
    linked
}

fn assemble(modules: &mut [Module], used: Option<&UsedItems>) -> TranslationUnit {
    let mut unit = TranslationUnit::default();
    for module in modules {
        let declarations = std::mem::take(&mut module.syntax.global_declarations);
        let path = &module.path;
        unit.global_declarations
            .extend(declarations.into_iter().filter(|declaration| {
                used.is_none_or(|used| {
                    declaration.is_const_assert()
                        || declared(declaration.node())
                            .is_some_and(|ident| used.get_ident(path, ident).is_some())
                })
            }));
        unit.global_directives
            .append(&mut module.syntax.global_directives);
    }
    pass::strip_visibility(&mut unit);
    unit.global_directives.dedup();
    unit
}

pub(super) fn linked_entry(
    main: &ModulePath,
    kept: Option<&str>,
    texts: (&Prepared, Option<&str>),
    parsed: &mut Parsed,
) -> Result<Linked, Error> {
    let (result, _names) = compile(main, kept, texts, parsed);
    let mut result = result?;
    sort(&mut result.modules, main, texts.0);
    Ok(linked(&assemble(
        &mut result.modules,
        Some(&result.used_items),
    )))
}

pub(super) fn bindings(
    main: &ModulePath,
    kept: &str,
    texts: (&Prepared, Option<&str>),
    parsed: &mut Parsed,
) -> Result<Vec<String>, Error> {
    let (result, _names) = compile(main, Some(kept), texts, parsed);
    let result = result?;
    let mut linked = Linked::default();
    for module in &result.modules {
        for declaration in &module.syntax.global_declarations {
            if let Some(ident) = declared(declaration.node())
                && result.used_items.get_ident(&module.path, ident).is_some()
            {
                linked.note(declaration.node(), &ident.name());
            }
        }
    }
    Ok(linked.bindings)
}

pub(super) enum Slot {
    Seen,
    Whole(Whole),
    Refused,
}

pub(super) struct Whole {
    declarations: Vec<(String, Option<String>)>,
    roles: Vec<Linked>,
    uses: Vec<Vec<usize>>,
    qualified: Vec<bool>,
    main: usize,
}

pub(super) fn entry(
    slots: &mut HashMap<(String, String), Slot>,
    (key, name): ((String, String), &str),
    (main, texts, parsed): (&ModulePath, (&Prepared, Option<&str>), &mut Parsed),
) -> Option<Linked> {
    match slots.get_mut(&key) {
        None => {
            slots.insert(key, Slot::Seen);
            return None;
        }
        Some(slot @ Slot::Seen) => {
            *slot = whole(main, texts, parsed).map_or(Slot::Refused, Slot::Whole);
        }
        Some(Slot::Whole(_) | Slot::Refused) => {}
    }
    match slots.get(&key)? {
        Slot::Whole(whole) => whole.entry(name),
        Slot::Seen | Slot::Refused => None,
    }
}

#[expect(
    clippy::mutable_key_type,
    reason = "identifiers hash and compare by address, as the wesl passes key them"
)]
fn whole(
    main: &ModulePath,
    texts: (&Prepared, Option<&str>),
    parsed: &mut Parsed,
) -> Option<Whole> {
    let (result, _names) = compile(main, None, texts, parsed);
    let mut result = result.ok()?;
    let plain = result.modules.iter().all(|module| {
        module.syntax.global_directives.is_empty()
            && !module
                .syntax
                .global_declarations
                .iter()
                .any(|declaration| declaration.is_const_assert())
    });
    if !plain {
        return None;
    }
    sort(&mut result.modules, main, texts.0);
    let main = result.modules.first()?.syntax.global_declarations.len();
    let unit = assemble(&mut result.modules, None);
    let mut at: HashMap<Ident, usize> = HashMap::default();
    let mut names = Vec::with_capacity(unit.global_declarations.len());
    let mut roles = Vec::with_capacity(unit.global_declarations.len());
    for (index, declaration) in unit.global_declarations.iter().enumerate() {
        let ident = declared(declaration.node());
        let mut role = Linked::default();
        if let Some(ident) = ident {
            at.entry(ident.clone()).or_insert(index);
            role.note(declaration.node(), &ident.name());
        }
        names.push(ident.map(|ident| ident.name().clone()));
        roles.push(role);
    }
    let mut uses = Vec::with_capacity(unit.global_declarations.len());
    let mut qualified = Vec::with_capacity(unit.global_declarations.len());
    for declaration in &unit.global_declarations {
        let (mut used, mut pathed) = (Vec::new(), false);
        Visit::<TypeExpression>::visit_rec(declaration.node(), &mut |ty: &TypeExpression| {
            pathed |= ty.path.is_some();
            if let Some(&index) = at.get(&ty.ident) {
                used.push(index);
            }
        });
        uses.push(used);
        qualified.push(pathed);
    }
    let reached = reached(&uses, 0..main);
    let declarations = unit
        .global_declarations
        .iter()
        .zip(names)
        .enumerate()
        .map(|(index, (declaration, name))| {
            let text = match &name {
                Some(_) if reached[index] => declaration.to_string(),
                Some(_) | None => String::new(),
            };
            (text, name)
        })
        .collect();
    Some(Whole {
        declarations,
        roles,
        uses,
        qualified,
        main,
    })
}

fn reached(uses: &[Vec<usize>], starts: std::ops::Range<usize>) -> Vec<bool> {
    let mut reached = vec![false; uses.len()];
    let mut stack: Vec<usize> = starts.collect();
    while let Some(index) = stack.pop() {
        if !std::mem::replace(&mut reached[index], true) {
            stack.extend(&uses[index]);
        }
    }
    reached
}

impl Whole {
    fn entry(&self, name: &str) -> Option<Linked> {
        let start =
            (0..self.main).find(|&index| self.declarations[index].1.as_deref() == Some(name))?;
        let mut used = vec![false; self.declarations.len()];
        let mut stack = vec![start];
        while let Some(index) = stack.pop() {
            if std::mem::replace(&mut used[index], true) {
                continue;
            }
            if self.qualified[index] {
                return None;
            }
            stack.extend(&self.uses[index]);
        }
        let mut linked = Linked::default();
        for (index, _) in used.iter().enumerate().filter(|(_, used)| **used) {
            if !linked.text.is_empty() {
                linked.text.push_str("\n\n");
            }
            linked.text.push_str(&self.declarations[index].0);
            let role = &self.roles[index];
            linked.bindings.extend(role.bindings.iter().cloned());
            linked.overrides.extend(role.overrides.iter().cloned());
            linked.functions.extend(role.functions.iter().cloned());
        }
        linked.text.push('\n');
        Some(linked)
    }
}

fn compile(
    main: &ModulePath,
    kept: Option<&str>,
    (prepared, writes): (&Prepared, Option<&str>),
    parsed: &mut Parsed,
) -> (Result<CompileResult, Error>, Names) {
    let mut features = Features::new();
    features.default = Feature::Error;
    let mut linking = Linking {
        main,
        kept,
        texts: Texts { prepared, writes },
        parsed,
        features,
        names: Names::default(),
    };
    let result = linking.compile();
    (result, linking.names)
}

pub(super) fn mangled(path: &ModulePath, name: &str) -> String {
    wesl::mangler::EscapeMangler.mangle(path, name)
}
