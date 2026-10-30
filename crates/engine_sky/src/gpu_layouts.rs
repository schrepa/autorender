use bytemuck::{Pod, Zeroable};

pub(crate) const CLOUD_SHADOW_BYTES: usize = (SHADOW_CELLS * SHADOW_CELLS * 2) as usize;
pub(crate) const CLOUD_LIGHT_BYTES: usize =
    (LIGHT_TEXELS * LIGHT_TEXELS * (LIGHT_HEIGHTS - 1) * 4) as usize;

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct CloudUniform as ["CloudUniform"] {
        pub eye: [f32; 4],
        pub mirror: [f32; 4],
        pub layer: [f32; 4],
        pub drift: [f32; 4],
        pub dome: [u32; 4],
        pub sun: [f32; 4],
        pub sun_color: [f32; 4],
        pub zenith: [f32; 4],
        pub horizon: [f32; 4],
        pub shadow: [f32; 4],
        pub shadow_rows: [u32; 4],
        pub light: [f32; 4],
        pub light_rows: [u32; 4],
    }

    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct ViewUniform as ["CloudView"] {
        pub inverse_view_projection: [[f32; 4]; 4],
        pub pixels: [u32; 4],
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub(crate) struct SkyPiece as ["SkyPiece" in "sky::piece"] {
        pub zenith: [f32; 4],
        pub horizon: [f32; 4],
        pub ground: [f32; 4],
        pub fog: [f32; 4],
        pub aerial: [f32; 4],
        pub aerial_color: [f32; 4],
        pub cloud_shadow: [f32; 4],
        pub cloud_layer: [f32; 4],
        pub night: [f32; 4],
        pub star_turn: [f32; 4],
        pub moon: [f32; 4],
        pub night_shape: [f32; 4],
        pub haze_march: [f32; 4],
        pub sun_light: [f32; 4],
        pub state: [f32; 4],
        pub air_rayleigh: [f32; 4],
        pub air_mie: [f32; 4],
        pub air_ozone: [f32; 4],
        pub air_ground: [f32; 4],
        pub air_radii: [f32; 4],
        pub air_sun: [f32; 4],
        pub air_frame: [f32; 4],
        pub air_moon: [f32; 4],
        pub domes_older: [f32; 4],
        pub domes_newer: [f32; 4],
        pub domes_older_mirror: [f32; 4],
        pub domes_newer_mirror: [f32; 4],
        pub domes_eye: [f32; 4],
        pub domes_view: [f32; 4],
        pub haze: [f32; 4],
    }
}

engine_render::gpu_consts! {
    pub(crate) const SHADOW_CELLS: u32 = 128 => ["CLOUD_SHADOW_CELLS"];
    pub(crate) const LIGHT_TEXELS: u32 = 384 => ["LIGHT_TEXELS" in "sky::clouds"];
    pub(crate) const LIGHT_HEIGHTS: u32 = 16 => ["LIGHT_HEIGHTS" in "sky::clouds"];
    pub(crate) const EYE_LAYER: u32 = 0 => ["EYE_LAYER"];
    pub(crate) const MIRROR_LAYER: u32 = 1 => ["MIRROR_LAYER"];

    pub(crate) const NOON_LIGHT: f32 = engine_faces::environment::NOON_LIGHT => ["NOON_LIGHT" in "sky::tables"];
    pub(crate) const ADAPTATION_POWER: f32 = engine_faces::environment::ADAPTATION_POWER => ["ADAPTATION_POWER" in "sky::tables"];
    pub(crate) const ADAPTATION_MOST: f32 = engine_faces::environment::ADAPTATION_MOST => ["ADAPTATION_MOST" in "sky::tables"];

    pub(crate) const CLOUD_GROUP: u32 = 8 => ["@march_view" in "sky::clouds", "@march_clouds" in "sky::clouds", "@shadow_clouds" in "sky::clouds", "@light_clouds" in "sky::clouds"];
    pub(crate) const SKY_TABLE_GROUP: u32 = 8 => ["@build_sky_view" in "sky::tables"];

    pub(crate) const CLOUD_HORIZON: f32 = 0.05 => ["HORIZON" in "sky::clouds", "CLOUD_HORIZON"];
    pub(crate) const CLOUD_REACH: f32 = 60000.0 => ["REACH" in "sky::clouds", "CLOUD_PARALLAX_REACH"];
}

engine_render::gpu_mirrors! {
    engine_core::table_data::sky::MULTIPLE_SIDE => ["SKY_MULTIPLE_SIZE"];
}
