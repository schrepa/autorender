use std::fmt::Write as _;

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

pub struct BlendedPass;

impl PassName for BlendedPass {
    const NAME: &'static str = "blended";
    const PASS: Pass = Pass {
        writes: Writes(&[TargetName::COLOR]),
        reads: SlotSet::ALL,
        views: PassViews::Every,
    };
}

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

impl ScenePasses {
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
