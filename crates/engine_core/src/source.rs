#[derive(Debug)]
pub struct Source {
    package: &'static str,
    game: bool,
    hash: [u8; 32],
    modules: &'static [ShaderText],
}

impl Source {
    #[doc(hidden)]
    pub const fn generated(
        package: &'static str,
        game: bool,
        hash: [u8; 32],
        modules: &'static [ShaderText],
    ) -> Self {
        Self {
            package,
            game,
            hash,
            modules,
        }
    }

    #[doc(hidden)]
    pub const fn package(&self) -> &'static str {
        self.package
    }

    #[doc(hidden)]
    pub const fn game(&self) -> bool {
        self.game
    }

    #[doc(hidden)]
    pub const fn hash(&self) -> [u8; 32] {
        self.hash
    }

    #[doc(hidden)]
    pub fn modules(&self) -> &'static [ShaderText] {
        self.modules
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct ShaderText {
    path: &'static str,
    text: &'static str,
}

impl ShaderText {
    #[doc(hidden)]
    pub const fn generated(path: &'static str, text: &'static str) -> Self {
        Self { path, text }
    }

    #[doc(hidden)]
    pub fn path(&self) -> &'static str {
        self.path
    }

    #[doc(hidden)]
    pub fn text(&self) -> &'static str {
        self.text
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Module {
    source: &'static Source,
    index: u32,
}

impl Module {
    #[doc(hidden)]
    pub const fn generated(source: &'static Source, index: u32) -> Self {
        Self { source, index }
    }

    #[doc(hidden)]
    pub fn source(&self) -> &'static Source {
        self.source
    }

    #[doc(hidden)]
    pub fn index(&self) -> u32 {
        self.index
    }
}

impl PartialEq for Module {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.source, other.source) && self.index == other.index
    }
}

impl Eq for Module {}

impl std::hash::Hash for Module {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(self.source, state);
        self.index.hash(state);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Entry {
    module: Module,
    name: &'static str,
}

impl Entry {
    #[doc(hidden)]
    pub const fn generated(module: Module, name: &'static str) -> Self {
        Self { module, name }
    }

    #[doc(hidden)]
    pub fn module(&self) -> Module {
        self.module
    }

    #[doc(hidden)]
    pub fn name(&self) -> &'static str {
        self.name
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Var {
    module: Module,
    name: &'static str,
}

impl Var {
    #[doc(hidden)]
    pub const fn generated(module: Module, name: &'static str) -> Self {
        Self { module, name }
    }

    #[doc(hidden)]
    pub fn module(&self) -> Module {
        self.module
    }

    #[doc(hidden)]
    pub fn name(&self) -> &'static str {
        self.name
    }
}
