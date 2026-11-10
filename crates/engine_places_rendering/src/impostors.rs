use bevy_ecs::prelude::{Component, Resource};
use engine_place::{GpuBits, MaterialHandle, GeometryHandle, Slot, gpu_bits, gpu_struct};

pub const IMPOSTOR_VERTICES: u32 = 6;
pub const DEBUG_CARD: u32 = 1 << 13;

#[gpu_bits(crate = engine_place)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImpostorFlags {
    #[bits(16)]
    pub texels: u32,
    pub hemisphere: bool,
    pub thin: bool,
    #[bits(5, at = 19)]
    pub frames: u32,
}

gpu_struct! {
    pub struct ImpostorRecord {
        pub sphere: [f32; 4],
        pub extent: [f32; 3],
        pub layer: u32,
        pub emissive: [f32; 3],
        pub flags: u32,
    }
}

impl ImpostorRecord {
    fn texels(&self) -> u32 {
        ImpostorFlags::unpack(self.flags).map_or(0, |flags| flags.texels)
    }

    pub fn error(&self) -> f32 {
        let texels = self.texels();
        if texels == 0 {
            return 0.0;
        }
        2.0 * self.sphere[3] / texels as f32
    }

    pub fn framed(&self) -> bool {
        self.texels() != 0 && self.sphere[3] > 0.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImpostorAsk {
    pub mesh: GeometryHandle,
    pub material: Option<MaterialHandle>,
    pub parts: Vec<(GeometryHandle, MaterialHandle)>,
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct ImpostorAsks {
    pub generation: u64,
    pub hemisphere: bool,
    pub asks: Vec<Option<ImpostorAsk>>,
}

#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct ImpostorRecords {
    pub generation: u64,
    pub records: Vec<ImpostorRecord>,
    pub asked: u32,
    pub refused: u32,
    pub live: u32,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Impostors {
    pub pixels: f32,
}

impl Default for Impostors {
    fn default() -> Self {
        Self { pixels: 128.0 }
    }
}

impl Slot for Impostors {
    const NAME: &'static str = "impostors";
    const WGSL: Option<&'static str> = Some(engine_places::contract!(
        "places/impostors.wgsl",
        IMPOSTOR_VERTICES,
        DEBUG_CARD,
    ));
}
