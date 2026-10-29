use bytemuck::{Pod, Zeroable};

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct DebugViewUniform as ["DebugViewUniform"] {
        pub view: [u32; 4],
        pub sun: [f32; 4],
    }
}

engine_render::gpu_consts! {
    pub const DEBUG_VIEW_KINDS: u32 = 1 => ["DEBUG_VIEW_KINDS" in "debug::view"];
    pub const DEBUG_VIEW_LEVELS: u32 = 2 => ["DEBUG_VIEW_LEVELS" in "debug::view"];
    pub const DEBUG_VIEW_CARDS: u32 = 3 => ["DEBUG_VIEW_CARDS" in "debug::view"];
    pub const DEBUG_VIEW_SHADOWS: u32 = 4 => ["DEBUG_VIEW_SHADOWS" in "debug::view"];
    pub const DEBUG_VIEW_CASTERS: u32 = 5 => ["DEBUG_VIEW_CASTERS" in "debug::view"];
    pub const DEBUG_VIEW_OVERDRAW: u32 = 6 => ["DEBUG_VIEW_OVERDRAW" in "debug::view"];
    pub const DEBUG_VIEW_NORMALS: u32 = 7 => ["DEBUG_VIEW_NORMALS" in "debug::view"];
    pub const DEBUG_VIEW_TILES: u32 = 8 => ["DEBUG_VIEW_TILES" in "debug::view"];
    pub const DEBUG_VIEW_MATERIALS: u32 = 9 => ["DEBUG_VIEW_MATERIALS" in "debug::view"];
    pub const DEBUG_VIEW_SKY: u32 = 10 => ["DEBUG_VIEW_SKY" in "debug::view"];
    pub const DEBUG_VIEW_DETAIL: u32 = 11 => ["DEBUG_VIEW_DETAIL" in "debug::view"];
    pub const DEBUG_VIEW_WATER_DEPTH: u32 = 12 => ["DEBUG_VIEW_WATER_DEPTH" in "debug::view"];
    pub const DEBUG_VIEW_WATER_SPEED: u32 = 13 => ["DEBUG_VIEW_WATER_SPEED" in "debug::view"];
    pub const DEBUG_VIEW_WATER_FROUDE: u32 = 14 => ["DEBUG_VIEW_WATER_FROUDE" in "debug::view"];
    pub const DEBUG_VIEW_WATER_TURBULENCE: u32 = 15 => ["DEBUG_VIEW_WATER_TURBULENCE" in "debug::view"];
    pub const DEBUG_VIEW_WATER_WHITE: u32 = 16 => ["DEBUG_VIEW_WATER_WHITE" in "debug::view"];
    pub const DEBUG_VIEW_WATER_REACH: u32 = 17 => ["DEBUG_VIEW_WATER_REACH" in "debug::view"];
    pub const DEBUG_VIEW_WATER_SPAN: u32 = 19 => ["DEBUG_VIEW_WATER_SPAN" in "debug::view"];
    pub const DEBUG_VIEW_WATER_EDDIES: u32 = 20 => ["DEBUG_VIEW_WATER_EDDIES" in "debug::view"];
    pub const DEBUG_VIEW_WATER_ROUGHNESS: u32 = 21 => ["DEBUG_VIEW_WATER_ROUGHNESS" in "debug::view"];
}
