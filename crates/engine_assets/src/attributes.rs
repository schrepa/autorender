use engine_codec::{Error, ErrorCode};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Attribute {
    Position,
    Normal,
    Tangent,
    Uv0,
    Uv1,
    Color,
    Joints,
    Weights,
}

impl Attribute {
    pub const ALL: [Self; 8] = [
        Self::Position,
        Self::Normal,
        Self::Tangent,
        Self::Uv0,
        Self::Uv1,
        Self::Color,
        Self::Joints,
        Self::Weights,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Position => "position",
            Self::Normal => "normal",
            Self::Tangent => "tangent",
            Self::Uv0 => "uv0",
            Self::Uv1 => "uv1",
            Self::Color => "color",
            Self::Joints => "joints",
            Self::Weights => "weights",
        }
    }

    pub const fn wgsl_type(self) -> &'static str {
        match self {
            Self::Position | Self::Normal => "vec3<f32>",
            Self::Tangent | Self::Color | Self::Weights => "vec4<f32>",
            Self::Uv0 | Self::Uv1 => "vec2<f32>",
            Self::Joints => "vec4<u32>",
        }
    }

    pub const fn location(self) -> u32 {
        self as u32
    }

    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Attributes(u8);

impl Attributes {
    pub const NONE: Self = Self(0);
    pub const BASE: Self = Self::NONE
        .with(Attribute::Position)
        .with(Attribute::Normal)
        .with(Attribute::Uv0);
    pub const POSITION: Self = Self::NONE.with(Attribute::Position);

    #[must_use]
    pub const fn with(self, attribute: Attribute) -> Self {
        Self(self.0 | attribute.bit())
    }

    pub const fn has(self, attribute: Attribute) -> bool {
        self.0 & attribute.bit() != 0
    }

    #[must_use]
    pub const fn missing(self, needed: Self) -> Self {
        Self(needed.0 & !self.0)
    }

    pub fn iter(self) -> impl Iterator<Item = Attribute> {
        Attribute::ALL
            .into_iter()
            .filter(move |&attribute| self.has(attribute))
    }

    pub fn check(self, needed: Self, path: impl Into<String>) -> Result<(), Error> {
        let missing = self.missing(needed);
        if missing == Self::NONE {
            return Ok(());
        }
        let names: Vec<&str> = missing.iter().map(Attribute::name).collect();
        Err(Error::new(ErrorCode::InvalidRenderable, path)
            .got(names)
            .variants(self.iter().map(Attribute::name)))
    }
}

pub fn attributes_wgsl() -> String {
    let mut text = String::new();
    for attribute in Attribute::ALL {
        let upper = attribute.name().to_ascii_uppercase();
        text.push_str(&format!(
            "const ATTRIBUTE_{upper}: u32 = {}u;\n",
            attribute.location()
        ));
    }
    text
}
