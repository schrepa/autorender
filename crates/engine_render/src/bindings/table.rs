#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stages(pub(crate) u8);

impl Stages {
    pub const VERTEX: Self = Self(1);
    pub const FRAGMENT: Self = Self(2);
    pub const COMPUTE: Self = Self(4);
    pub const VERTEX_FRAGMENT: Self = Self(3);
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
    Depth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    D2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampling {
    Filtering,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Uniform { min_size: u64, dynamic: bool },
    Texture { sample: Sample, view: View },
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

pub const fn texture(name: &'static str, sample: Sample, view: View, stages: Stages) -> Binding {
    bind(name, Kind::Texture { sample, view }, stages)
}

pub const fn sampler(name: &'static str, sampling: Sampling, stages: Stages) -> Binding {
    bind(name, Kind::Sampler(sampling), stages)
}
