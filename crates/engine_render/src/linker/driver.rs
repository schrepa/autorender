use engine_core::collections::HashMap;
use std::borrow::Cow;

use wesl::error::{Error, ResolveError, UsageError};
use wesl::pass::{self, CompileResult, CompilerDriver, Module, UsedItems};
use wesl::syntax::{GlobalDeclaration, Ident, ModulePath, TranslationUnit};
use wesl::{Feature, Features, Mangler as _, Resolver};

pub(super) struct Prepared {
    pub(super) at: HashMap<ModulePath, usize>,
    pub(super) texts: Vec<String>,
    pub(super) writes: Vec<bool>,
    pub(super) named: HashMap<String, usize>,
    pub(super) overrides: Vec<Vec<(String, String)>>,
    pub(super) declared: Vec<Vec<String>>,
    pub(super) pieces: Vec<(String, String, String)>,
}

pub(super) type Parsed = HashMap<(usize, Option<String>), TranslationUnit>;

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

fn declared_mut(declaration: &mut GlobalDeclaration) -> Option<&mut Ident> {
    match declaration {
        GlobalDeclaration::Declaration(item) => Some(&mut item.ident),
        GlobalDeclaration::TypeAlias(item) => Some(&mut item.ident),
        GlobalDeclaration::Struct(item) => Some(&mut item.ident),
        GlobalDeclaration::Function(item) => Some(&mut item.ident),
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
        let key = (at, self.texts.written(at).map(str::to_owned));
        if !self.parsed.contains_key(&key) {
            let mut module = pass::load_module(path, &self.texts)?;
            pass::condcomp(&mut module, &self.features)?;
            self.parsed.insert(key.clone(), module);
        }
        let mut module = self.parsed[&key].clone();
        for declaration in &mut module.global_declarations {
            if let Some(ident) = declared_mut(declaration.node_mut()) {
                let name = ident.name().clone();
                *ident = Ident::new(name);
            }
        }
        pass::retarget_idents(&mut module);
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
        Ok(pass::link(modules, Some(used_items)))
    }
}

pub(super) fn compile(
    main: &ModulePath,
    kept: Option<&str>,
    (prepared, writes): (&Prepared, Option<&str>),
    parsed: &mut Parsed,
) -> Result<CompileResult, Error> {
    let mut features = Features::new();
    features.default = Feature::Error;
    let mut linking = Linking {
        main,
        kept,
        texts: Texts { prepared, writes },
        parsed,
        features,
    };
    linking.compile()
}

pub(super) fn mangled(path: &ModulePath, name: &str) -> String {
    wesl::mangler::EscapeMangler.mangle(path, name)
}
