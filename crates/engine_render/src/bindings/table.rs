//! What a bind group table is made of: shader stages, the kinds of
//! bindings and their texture, sampler and storage formats, and the
//! constructors the tables declare their bindings with.

/// Shader stages, as bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stages(pub(super) u8);

impl Stages {
    pub const VERTEX: Self = Self(1);
    pub const FRAGMENT: Self = Self(2);
    pub const COMPUTE: Self = Self(4);
    pub const VERTEX_FRAGMENT: Self = Self(3);

    pub fn vertex(self) -> bool {
        self.0 & 1 != 0
    }

    pub fn fragment(self) -> bool {
        self.0 & 2 != 0
    }

    pub fn compute(self) -> bool {
        self.0 & 4 != 0
    }
}

/// How shaders read a texture's texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sample {
    /// Floats, filtered by samplers.
    Float,
    /// Floats shaders only load.
    Unfilterable,
    Depth,
    Uint,
}

/// A texture view's dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

/// How a sampler filters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampling {
    Filtering,
    NonFiltering,
    Comparison,
}

/// A storage texture's texel format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    R32Float,
    R32Uint,
    Rgba8Snorm,
    Rgba8Unorm,
    Rgba16Float,
}

/// What a binding holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A uniform buffer of at least `min_size` bytes (0 leaves it to
    /// draws), bound at a dynamic offset when `dynamic`.
    Uniform {
        min_size: u64,
        dynamic: bool,
    },
    /// A storage buffer shaders only read, of at least `min_size` bytes.
    Read {
        min_size: u64,
    },
    /// A storage buffer shaders write.
    Write,
    Texture {
        sample: Sample,
        view: View,
    },
    /// A 2D texture of this format shaders write.
    Storage(Format),
    /// A 2D array texture of this format shaders write.
    StorageArray(Format),
    Sampler(Sampling),
}

/// One binding of a layout.
#[derive(Clone, Copy, Debug)]
pub struct Binding {
    pub binding: u32,
    pub name: &'static str,
    pub kind: Kind,
    pub stages: Stages,
}

/// A bind group slot a table serves: group `group` of shader module
/// `module` (`crate::shaders`).
#[derive(Debug)]
pub struct Serves {
    pub module: &'static str,
    pub group: u32,
}

/// A bind group layout: its label, the shader groups it serves and its
/// bindings. Tables are declared with `tables!` alone, which registers
/// each in [`TABLES`](super::TABLES) for the GPU check.
#[derive(Debug)]
pub struct Table {
    pub label: &'static str,
    pub serves: &'static [Serves],
    pub bindings: &'static [Binding],
}

impl Table {
    /// The binding named `name`; a name the table lacks is a programming
    /// error.
    pub(crate) fn binding(&self, name: &str) -> &Binding {
        self.bindings
            .iter()
            .find(|binding| binding.name == name)
            .unwrap_or_else(|| panic!("{} has no binding named {name}", self.label))
    }
}

pub(super) const fn bind(binding: u32, name: &'static str, kind: Kind, stages: Stages) -> Binding {
    Binding {
        binding,
        name,
        kind,
        stages,
    }
}

pub(super) const fn uniform(binding: u32, name: &'static str, stages: Stages) -> Binding {
    bind(
        binding,
        name,
        Kind::Uniform {
            min_size: 0,
            dynamic: false,
        },
        stages,
    )
}

pub(super) const fn uniform_of(
    binding: u32,
    name: &'static str,
    size: usize,
    stages: Stages,
) -> Binding {
    bind(
        binding,
        name,
        Kind::Uniform {
            min_size: size as u64,
            dynamic: false,
        },
        stages,
    )
}

pub(super) const fn dynamic_uniform(
    binding: u32,
    name: &'static str,
    size: usize,
    stages: Stages,
) -> Binding {
    bind(
        binding,
        name,
        Kind::Uniform {
            min_size: size as u64,
            dynamic: true,
        },
        stages,
    )
}

pub(super) const fn read(binding: u32, name: &'static str, stages: Stages) -> Binding {
    bind(binding, name, Kind::Read { min_size: 0 }, stages)
}

pub(super) const fn read_of(
    binding: u32,
    name: &'static str,
    size: usize,
    stages: Stages,
) -> Binding {
    bind(
        binding,
        name,
        Kind::Read {
            min_size: size as u64,
        },
        stages,
    )
}

pub(super) const fn write(binding: u32, name: &'static str, stages: Stages) -> Binding {
    bind(binding, name, Kind::Write, stages)
}

pub(super) const fn texture(
    binding: u32,
    name: &'static str,
    sample: Sample,
    view: View,
    stages: Stages,
) -> Binding {
    bind(binding, name, Kind::Texture { sample, view }, stages)
}

pub(super) const fn sampler(
    binding: u32,
    name: &'static str,
    sampling: Sampling,
    stages: Stages,
) -> Binding {
    bind(binding, name, Kind::Sampler(sampling), stages)
}
