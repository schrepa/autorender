use engine_codec::{Error, ErrorCode};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Attribute {
    Position,
    Normal,
    Tangent,
    Uv0,
    Uv1,
    Color,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AttributeFormat {
    Float32x2,
    Float32x3,
    Float32x4,
    Snorm16x2,
    Unorm16x4,
    Uint32x4,
}

impl AttributeFormat {
    pub const ALL: [Self; 6] = [
        Self::Float32x2,
        Self::Float32x3,
        Self::Float32x4,
        Self::Snorm16x2,
        Self::Unorm16x4,
        Self::Uint32x4,
    ];

    pub const fn code(self) -> u32 {
        match self {
            Self::Float32x2 => 0,
            Self::Float32x3 => 1,
            Self::Float32x4 => 2,
            Self::Snorm16x2 => 3,
            Self::Unorm16x4 => 4,
            Self::Uint32x4 => 5,
        }
    }

    pub fn of_code(code: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|format| format.code() == code)
    }

    pub const fn float(self) -> bool {
        match self {
            Self::Float32x2 | Self::Float32x3 | Self::Float32x4 => true,
            Self::Snorm16x2 | Self::Unorm16x4 | Self::Uint32x4 => false,
        }
    }

    pub const fn bytes(self) -> u64 {
        match self {
            Self::Snorm16x2 => 4,
            Self::Float32x2 | Self::Unorm16x4 => 8,
            Self::Float32x3 => 12,
            Self::Float32x4 | Self::Uint32x4 => 16,
        }
    }

    pub const fn wgsl_type(self) -> &'static str {
        match self {
            Self::Float32x2 | Self::Snorm16x2 => "vec2<f32>",
            Self::Float32x3 => "vec3<f32>",
            Self::Float32x4 | Self::Unorm16x4 => "vec4<f32>",
            Self::Uint32x4 => "vec4<u32>",
        }
    }
}

impl Attribute {
    pub const ALL: [Self; 6] = [
        Self::Position,
        Self::Normal,
        Self::Tangent,
        Self::Uv0,
        Self::Uv1,
        Self::Color,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Position => "position",
            Self::Normal => "normal",
            Self::Tangent => "tangent",
            Self::Uv0 => "uv0",
            Self::Uv1 => "uv1",
            Self::Color => "color",
        }
    }

    pub const fn stored(self) -> Option<AttributeFormat> {
        match self {
            Self::Position => Some(AttributeFormat::Float32x3),
            Self::Normal => Some(AttributeFormat::Snorm16x2),
            Self::Tangent => None,
            Self::Uv0 | Self::Uv1 => Some(AttributeFormat::Float32x2),
            Self::Color => Some(AttributeFormat::Unorm16x4),
        }
    }

    pub const fn location(self) -> u32 {
        self as u32
    }

    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamShape {
    pub name: &'static str,
    pub format: AttributeFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GeometryStream {
    Vertex,
    Extra,
    Named { shape: StreamShape, location: u32 },
}

impl GeometryStream {
    pub const FIXED: [Self; 2] = [Self::Vertex, Self::Extra];

    pub const fn attributes(self) -> &'static [Attribute] {
        match self {
            Self::Vertex => &[Attribute::Position, Attribute::Normal, Attribute::Uv0],
            Self::Extra => &[Attribute::Color, Attribute::Uv1],
            Self::Named { .. } => &[],
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Vertex => "Vertex",
            Self::Extra => "Extra",
            Self::Named { shape, .. } => shape.name,
        }
    }

    pub fn placed(self) -> impl Iterator<Item = (Attribute, AttributeFormat, u64)> {
        let mut offset = 0;
        self.attributes().iter().filter_map(move |&attribute| {
            let stored = attribute.stored()?;
            let at = offset;
            offset += stored.bytes();
            Some((attribute, stored, at))
        })
    }

    pub fn inputs(self) -> Vec<(u32, AttributeFormat, u64)> {
        match self {
            Self::Vertex | Self::Extra => self
                .placed()
                .map(|(attribute, stored, offset)| (attribute.location(), stored, offset))
                .collect(),
            Self::Named { shape, location } => vec![(location, shape.format, 0)],
        }
    }

    pub fn stride(self) -> u64 {
        match self {
            Self::Vertex | Self::Extra => self.placed().map(|(_, stored, _)| stored.bytes()).sum(),
            Self::Named { shape, .. } => shape.format.bytes(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InstanceInput {
    Slot,
    Batch,
    Draw,
}

impl InstanceInput {
    pub const ALL: [Self; 3] = [Self::Slot, Self::Batch, Self::Draw];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Slot => "slot",
            Self::Batch => "batch",
            Self::Draw => "draw",
        }
    }

    pub const fn location(self) -> u32 {
        Attribute::ALL.len() as u32 + self as u32
    }
}

pub const FIRST_NAMED_LOCATION: u32 = Attribute::ALL.len() as u32 + InstanceInput::ALL.len() as u32;

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
    for input in InstanceInput::ALL {
        let upper = input.name().to_ascii_uppercase();
        text.push_str(&format!(
            "const INPUT_{upper}: u32 = {}u;\n",
            input.location()
        ));
    }
    for stream in GeometryStream::FIXED {
        text.push_str(&format!("struct Geometry{} {{\n", stream.name()));
        for (attribute, stored, _) in stream.placed() {
            text.push_str(&format!(
                "    @location({}) {}: {},\n",
                attribute.location(),
                attribute.name(),
                stored.wgsl_type()
            ));
        }
        text.push_str("}\n");
    }
    text
}
