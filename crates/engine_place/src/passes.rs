use std::fmt::Write as _;

use bevy_ecs::resource::Resource;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{DrawKind, Error, ErrorCode, KindDecl, gpu_bits};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum AttachPoint {
    Depth,
    Inputs,
    Opaque,
    LateOpaque,
    Transparent,
    Post,
    Display,
    Overlay,
}

impl AttachPoint {
    pub const ALL: [Self; 8] = [
        Self::Depth,
        Self::Inputs,
        Self::Opaque,
        Self::LateOpaque,
        Self::Transparent,
        Self::Post,
        Self::Display,
        Self::Overlay,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Depth => "depth",
            Self::Inputs => "inputs",
            Self::Opaque => "opaque",
            Self::LateOpaque => "late_opaque",
            Self::Transparent => "transparent",
            Self::Post => "post",
            Self::Display => "display",
            Self::Overlay => "overlay",
        }
    }

    pub const fn scene(self) -> bool {
        match self {
            Self::Depth | Self::Opaque | Self::LateOpaque | Self::Transparent => true,
            Self::Inputs | Self::Post | Self::Display | Self::Overlay => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct TargetName(pub &'static str);

impl TargetName {
    pub const COLOR: Self = Self("color");
    pub const DEPTH: Self = Self("depth");
    pub const TRANSPARENCY: Self = Self("transparency");
    pub const ACCUMULATED: Self = Self("accumulated");
    pub const CORE: [Self; 4] = [
        Self::COLOR,
        Self::DEPTH,
        Self::TRANSPARENCY,
        Self::ACCUMULATED,
    ];

    pub const fn name(self) -> &'static str {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Writes(pub &'static [TargetName]);

impl Writes {
    pub const LIT: Self = Self(&[TargetName::COLOR, TargetName::DEPTH]);
    pub const DEPTH: Self = Self(&[TargetName::DEPTH]);
    pub const NOTHING: Self = Self(&[]);

    pub const fn targets(self) -> &'static [TargetName] {
        self.0
    }

    pub fn contains(self, target: TargetName) -> bool {
        self.0.contains(&target)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum PassViews {
    Every,
    Main,
    Owned,
}

#[gpu_bits(crate = crate)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SlotSet {
    pub position: bool,
    pub base_color: bool,
    pub opacity: bool,
    pub normal: bool,
    pub emissive: bool,
    pub lighting: bool,
    pub output: bool,
}

impl SlotSet {
    pub const NONE: Self = Self {
        position: false,
        base_color: false,
        opacity: false,
        normal: false,
        emissive: false,
        lighting: false,
        output: false,
    };
    pub const POSITION: Self = Self {
        position: true,
        ..Self::NONE
    };
    pub const MASK: Self = Self {
        position: true,
        opacity: true,
        ..Self::NONE
    };
    pub const ALL: Self = Self {
        position: true,
        base_color: true,
        opacity: true,
        normal: true,
        emissive: true,
        lighting: true,
        output: true,
    };

    pub fn covers(self, other: Self) -> bool {
        let (held, wanted) = (
            <Self as crate::GpuBits>::pack(self),
            <Self as crate::GpuBits>::pack(other),
        );
        held & wanted == wanted
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PassPurpose {
    Scene,
    Debug,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PassRole {
    Lit,
    Depth,
    Blended,
    Shown,
}

#[derive(Clone, Copy, Debug)]
pub struct Pass {
    pub writes: Writes,
    pub reads: SlotSet,
    pub views: PassViews,
    pub at: AttachPoint,
    pub purpose: PassPurpose,
    pub takes: fn(&KindDecl) -> bool,
}

impl Pass {
    pub const fn scene(&self) -> bool {
        self.reads.position
    }

    pub const fn role(&self) -> PassRole {
        if matches!(self.purpose, PassPurpose::Debug) {
            PassRole::Shown
        } else if matches!(self.at, AttachPoint::Transparent) {
            PassRole::Blended
        } else if self.reads.lighting {
            PassRole::Lit
        } else {
            PassRole::Depth
        }
    }
}

pub fn every(_: &KindDecl) -> bool {
    true
}

pub fn never(_: &KindDecl) -> bool {
    false
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassCull {
    Material,
    None,
    Front,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PassRaster {
    pub bias: i32,
    pub slope: f32,
    pub clamp_near: bool,
    pub cull: PassCull,
}

impl PassRaster {
    pub const PLAIN: Self = Self {
        bias: 0,
        slope: 0.0,
        clamp_near: false,
        cull: PassCull::Material,
    };
}

pub trait PassName: 'static {
    const NAME: &'static str;
    const PASS: Pass;
    const RASTER: PassRaster = PassRaster::PLAIN;
}

pub struct ColorPass;

impl PassName for ColorPass {
    const NAME: &'static str = "color";
    const PASS: Pass = Pass {
        writes: Writes::LIT,
        reads: SlotSet::ALL,
        views: PassViews::Every,
        at: AttachPoint::Opaque,
        purpose: PassPurpose::Scene,
        takes: every,
    };
}

pub struct BlendedPass;

impl PassName for BlendedPass {
    const NAME: &'static str = "blended";
    const PASS: Pass = Pass {
        writes: Writes(&[
            TargetName::COLOR,
            TargetName::TRANSPARENCY,
            TargetName::ACCUMULATED,
        ]),
        reads: SlotSet::ALL,
        views: PassViews::Every,
        at: AttachPoint::Transparent,
        purpose: PassPurpose::Scene,
        takes: every,
    };
}

#[derive(Clone, Copy, Debug)]
pub struct PassEntry {
    pub name: &'static str,
    pub pass: Pass,
    pub raster: PassRaster,
}

impl PassEntry {
    pub const fn of<N: PassName>() -> Self {
        Self {
            name: N::NAME,
            pass: N::PASS,
            raster: N::RASTER,
        }
    }
}

pub const PASS_BITS: u32 = u32::BITS;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct PassBit(u8);

impl PassBit {
    pub const fn index(self) -> u32 {
        self.0 as u32
    }

    pub const fn mask(self) -> PassMask {
        PassMask(1 << self.0)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PassMask(pub u32);

impl PassMask {
    pub const NONE: Self = Self(0);

    pub const fn has(self, bit: PassBit) -> bool {
        self.0 & (1 << bit.0) != 0
    }

    #[must_use]
    pub const fn set(self, bit: PassBit, on: bool) -> Self {
        if on {
            Self(self.0 | (1 << bit.0))
        } else {
            Self(self.0 & !(1 << bit.0))
        }
    }

    pub const fn word(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ScenePass {
    pub name: &'static str,
    pub brick: &'static str,
    pub pass: Pass,
    pub raster: PassRaster,
    pub bit: Option<PassBit>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(tag = "join", rename_all = "snake_case")]
pub enum KindJoin {
    Joins,
    Skips { reason: &'static str },
    NotTaken,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct KindCell {
    pub kind: &'static str,
    pub pass: &'static str,
    pub brick: &'static str,
    #[serde(flatten)]
    pub join: KindJoin,
}

#[derive(Resource, Clone, Debug, Default)]
pub struct ScenePasses {
    listed: Vec<ScenePass>,
    kinds: Vec<KindDecl>,
}

pub const CORE_PASSES: [PassEntry; 2] =
    [PassEntry::of::<ColorPass>(), PassEntry::of::<BlendedPass>()];

pub const CORE_BRICK: &str = "engine_render";

pub fn wgsl_name(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn unknown_skips(listed: &[ScenePass], kinds: &[KindDecl], left: &[&'static str]) -> Vec<Error> {
    let known = |name: &str| listed.iter().any(|known| known.name == name) || left.contains(&name);
    let mut unknown = Vec::new();
    for decl in kinds {
        for skip in decl.skips.iter().filter(|skip| !known(skip.pass)) {
            unknown.push(
                Error::new(
                    ErrorCode::UnknownVariant,
                    format!("$.draw_kinds.{}.skips.{}", decl.kind.name(), skip.pass),
                )
                .got(skip.pass)
                .variants(
                    listed
                        .iter()
                        .map(|known| known.name)
                        .chain(left.iter().copied()),
                ),
            );
        }
    }
    unknown
}

impl ScenePasses {
    pub fn of(
        registered: impl IntoIterator<Item = (&'static str, PassEntry)>,
        kinds: Vec<KindDecl>,
        left: &[&'static str],
    ) -> (Self, Vec<Error>) {
        let entries: Vec<(&'static str, PassEntry)> = registered.into_iter().collect();
        let mut listed: Vec<ScenePass> = Vec::with_capacity(entries.len() + CORE_PASSES.len());
        let mut problems = Vec::new();
        let mut next = 0_u32;
        let core = CORE_PASSES.iter().map(|entry| (CORE_BRICK, *entry));
        for (brick, entry) in core.chain(entries) {
            let path = format!("$.passes.{}", entry.name);
            if let Some(known) = listed.iter().find(|known| known.name == entry.name) {
                problems.push(Error::new(ErrorCode::NameTwice, path).got(known.brick));
                continue;
            }
            let bit = if !entry.pass.scene() {
                None
            } else if next < PASS_BITS {
                next += 1;
                Some(PassBit((next - 1) as u8))
            } else {
                problems
                    .push(Error::new(ErrorCode::OutOfRange, format!("{path}.bit")).got(PASS_BITS));
                None
            };
            listed.push(ScenePass {
                name: entry.name,
                brick,
                pass: entry.pass,
                raster: entry.raster,
                bit,
            });
        }
        problems.extend(unknown_skips(&listed, &kinds, left));
        (Self { listed, kinds }, problems)
    }

    pub fn listed(&self) -> &[ScenePass] {
        &self.listed
    }

    pub fn kinds(&self) -> &[KindDecl] {
        &self.kinds
    }

    pub fn kind(&self, kind: DrawKind) -> Option<&KindDecl> {
        self.kinds.iter().find(|decl| decl.kind == kind)
    }

    pub fn kind_table(&self) -> Vec<KindCell> {
        let mut cells = Vec::new();
        for decl in &self.kinds {
            for listed in self.listed.iter().filter(|listed| listed.bit.is_some()) {
                let join = match decl.skip(listed.name) {
                    Some(skip) => KindJoin::Skips {
                        reason: skip.reason,
                    },
                    None if (listed.pass.takes)(decl) => KindJoin::Joins,
                    None => KindJoin::NotTaken,
                };
                cells.push(KindCell {
                    kind: decl.kind.name(),
                    pass: listed.name,
                    brick: listed.brick,
                    join,
                });
            }
        }
        cells
    }

    pub fn takes(&self, listed: &ScenePass, kind: DrawKind) -> bool {
        self.kind(kind)
            .is_some_and(|decl| decl.joins(listed.name, &listed.pass))
    }

    pub fn get(&self, name: &str) -> Option<&ScenePass> {
        self.listed.iter().find(|listed| listed.name == name)
    }

    pub fn bit(&self, name: &str) -> Option<PassBit> {
        self.get(name).and_then(|listed| listed.bit)
    }

    pub fn bit_of<N: PassName>(&self) -> Option<PassBit> {
        self.bit(N::NAME)
    }

    pub fn at(&self, point: AttachPoint) -> impl Iterator<Item = &ScenePass> + '_ {
        self.listed
            .iter()
            .filter(move |listed| listed.pass.at == point)
    }

    pub fn defaults(&self, kind: DrawKind) -> PassMask {
        self.listed
            .iter()
            .filter_map(|listed| listed.bit.filter(|_| self.takes(listed, kind)))
            .fold(PassMask::NONE, |mask, bit| mask.set(bit, true))
    }

    pub fn decode(&self, mask: PassMask) -> Value {
        let mut decoded = serde_json::Map::new();
        for listed in &self.listed {
            if let Some(bit) = listed.bit {
                decoded.insert(listed.name.to_owned(), Value::Bool(mask.has(bit)));
            }
        }
        Value::Object(decoded)
    }

    pub fn encode(&self, value: &Value, base: PassMask, path: &str) -> Result<PassMask, Error> {
        let Value::Object(named) = value else {
            return Err(Error::new(ErrorCode::InvalidType, path.to_owned()).got(value.clone()));
        };
        let mut mask = base;
        for (name, on) in named {
            let at = format!("{path}.{name}");
            let Some(bit) = self.bit(name) else {
                let known: Vec<&str> = self
                    .listed
                    .iter()
                    .filter(|listed| listed.bit.is_some())
                    .map(|listed| listed.name)
                    .collect();
                return Err(Error::new(ErrorCode::UnknownField, at)
                    .got(name)
                    .variants(known));
            };
            let Value::Bool(on) = on else {
                return Err(Error::new(ErrorCode::InvalidType, at).got(on.clone()));
            };
            mask = mask.set(bit, *on);
        }
        Ok(mask)
    }

    pub fn wgsl(&self) -> String {
        let mut text = String::new();
        for listed in &self.listed {
            if let Some(bit) = listed.bit {
                let _ = writeln!(
                    text,
                    "const PASS_{}: u32 = {}u;",
                    wgsl_name(listed.name),
                    1_u32 << bit.index()
                );
            }
        }
        text
    }
}
