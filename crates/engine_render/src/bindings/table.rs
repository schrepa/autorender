#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stages(pub(super) u8);

impl Stages {
    pub const VERTEX: Self = Self(1);
    pub const FRAGMENT: Self = Self(2);
    pub const COMPUTE: Self = Self(4);
    pub const VERTEX_FRAGMENT: Self = Self(3);
    pub const VERTEX_COMPUTE: Self = Self(5);
    pub const FRAGMENT_COMPUTE: Self = Self(6);
    pub const VERTEX_FRAGMENT_COMPUTE: Self = Self(7);

    pub fn vertex(self) -> bool {
        self.0 & 1 != 0
    }

    pub fn fragment(self) -> bool {
        self.0 & 2 != 0
    }

    pub fn compute(self) -> bool {
        self.0 & 4 != 0
    }

    pub fn covers(self, stage: Stages) -> bool {
        self.0 & stage.0 != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sample {
    Float,
    Unfilterable,
    Depth,
    Uint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampling {
    Filtering,
    NonFiltering,
    Comparison,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    R32Float,
    R32Uint,
    Rgba8Snorm,
    Rgba8Unorm,
    Rgba16Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Uniform { min_size: u64, dynamic: bool },
    Read { min_size: u64 },
    Write,
    Texture { sample: Sample, view: View },
    Storage(Format),
    StorageArray(Format),
    Sampler(Sampling),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub name: &'static str,
    pub kind: Kind,
    pub stages: Stages,
}

#[derive(Debug)]
pub struct Table {
    pub label: &'static str,
    pub bindings: &'static [Binding],
}

pub const fn bind(name: &'static str, kind: Kind, stages: Stages) -> Binding {
    Binding { name, kind, stages }
}

pub const fn uniform(name: &'static str, stages: Stages) -> Binding {
    bind(
        name,
        Kind::Uniform {
            min_size: 0,
            dynamic: false,
        },
        stages,
    )
}

pub const fn uniform_of(name: &'static str, size: usize, stages: Stages) -> Binding {
    bind(
        name,
        Kind::Uniform {
            min_size: size as u64,
            dynamic: false,
        },
        stages,
    )
}

pub const fn dynamic_uniform(name: &'static str, size: usize, stages: Stages) -> Binding {
    bind(
        name,
        Kind::Uniform {
            min_size: size as u64,
            dynamic: true,
        },
        stages,
    )
}

pub const fn read(name: &'static str, stages: Stages) -> Binding {
    bind(name, Kind::Read { min_size: 0 }, stages)
}

pub const fn read_of(name: &'static str, size: usize, stages: Stages) -> Binding {
    bind(
        name,
        Kind::Read {
            min_size: size as u64,
        },
        stages,
    )
}

pub const fn write(name: &'static str, stages: Stages) -> Binding {
    bind(name, Kind::Write, stages)
}

pub const fn texture(name: &'static str, sample: Sample, view: View, stages: Stages) -> Binding {
    bind(name, Kind::Texture { sample, view }, stages)
}

pub const fn sampler(name: &'static str, sampling: Sampling, stages: Stages) -> Binding {
    bind(name, Kind::Sampler(sampling), stages)
}
