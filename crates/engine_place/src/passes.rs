use bevy_ecs::resource::Resource;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{DrawKind, Error, ErrorCode, KindDecl};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct TargetName(pub &'static str);

impl TargetName {
    pub const CANVAS: Self = Self("canvas");

    pub const fn name(self) -> &'static str {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Writes(pub &'static [TargetName]);

impl Writes {
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
}

#[derive(Clone, Copy, Debug)]
pub struct Pass {
    pub writes: Writes,
    pub views: PassViews,
}

pub trait PassName: 'static {
    const NAME: &'static str;
    const PASS: Pass;
}

#[derive(Clone, Copy, Debug)]
pub struct PassEntry {
    pub name: &'static str,
    pub pass: Pass,
}

impl PassEntry {
    pub const fn of<N: PassName>() -> Self {
        Self {
            name: N::NAME,
            pass: N::PASS,
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
    pub bit: Option<PassBit>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(tag = "join", rename_all = "snake_case")]
pub enum KindJoin {
    Joins,
    NotAsked,
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

impl ScenePasses {
    pub fn of(
        registered: impl IntoIterator<Item = (&'static str, PassEntry)>,
        kinds: Vec<KindDecl>,
    ) -> (Self, Vec<Error>) {
        let entries: Vec<(&'static str, PassEntry)> = registered.into_iter().collect();
        let mut listed: Vec<ScenePass> = Vec::with_capacity(entries.len());
        let mut problems = Vec::new();
        let mut next = 0_u32;
        for (brick, entry) in entries {
            let path = format!("$.passes.{}", entry.name);
            if let Some(known) = listed.iter().find(|known| known.name == entry.name) {
                problems.push(Error::new(ErrorCode::NameTwice, path).got(known.brick));
                continue;
            }
            let bit = if next < PASS_BITS {
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
                bit,
            });
        }
        let scene: Vec<&'static str> = listed
            .iter()
            .filter(|listed| listed.bit.is_some())
            .map(|listed| listed.name)
            .collect();
        for decl in &kinds {
            for &asked in decl.passes.iter().filter(|asked| !scene.contains(asked)) {
                let path = format!("$.draw_kinds.{}.passes", decl.kind.name());
                problems.push(
                    Error::new(ErrorCode::UnknownField, path)
                        .got(asked)
                        .variants(scene.iter().copied()),
                );
            }
        }
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
                let join = if decl.joins(listed.name) {
                    KindJoin::Joins
                } else {
                    KindJoin::NotAsked
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
        self.kind(kind).is_some_and(|decl| decl.joins(listed.name))
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
}
