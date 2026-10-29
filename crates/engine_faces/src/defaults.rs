use engine_render::linker::ShaderModule;

pub const BRICK: &str = "faces";

pub const SHADOWS: ShaderModule = ShaderModule {
    path: "face::shadows",
    brick: BRICK,
    source: include_str!("defaults/shadows.wgsl"),
};

pub const SKY: ShaderModule = ShaderModule {
    path: "face::sky",
    brick: BRICK,
    source: include_str!("defaults/sky.wgsl"),
};

pub const LIGHTS: ShaderModule = ShaderModule {
    path: "face::lights",
    brick: BRICK,
    source: include_str!("defaults/lights.wgsl"),
};

pub const PROBES: ShaderModule = ShaderModule {
    path: "face::probes",
    brick: BRICK,
    source: include_str!("defaults/probes.wgsl"),
};

pub const OCCUPANCY: ShaderModule = ShaderModule {
    path: "face::occupancy",
    brick: BRICK,
    source: include_str!("defaults/occupancy.wgsl"),
};

pub const HORIZONS: ShaderModule = ShaderModule {
    path: "face::horizons",
    brick: BRICK,
    source: include_str!("defaults/horizons.wgsl"),
};

pub const TONEMAP: ShaderModule = ShaderModule {
    path: "face::tonemap",
    brick: BRICK,
    source: include_str!("defaults/tonemap.wgsl"),
};

pub const SUN: ShaderModule = ShaderModule {
    path: "face::sun",
    brick: BRICK,
    source: include_str!("defaults/sun.wgsl"),
};

pub const BLOOM: ShaderModule = ShaderModule {
    path: "face::bloom",
    brick: BRICK,
    source: include_str!("defaults/bloom.wgsl"),
};

pub const REFLECTIONS: ShaderModule = ShaderModule {
    path: "face::reflections",
    brick: BRICK,
    source: include_str!("defaults/reflections.wgsl"),
};

pub const DITHER: ShaderModule = ShaderModule {
    path: "face::dither",
    brick: BRICK,
    source: include_str!("defaults/dither.wgsl"),
};

pub const SURFACE_POINT: ShaderModule = ShaderModule {
    path: "face::surface_point",
    brick: BRICK,
    source: include_str!("defaults/surface_point.wgsl"),
};

pub static MODULES: &[ShaderModule] = &[
    SHADOWS,
    SKY,
    LIGHTS,
    PROBES,
    OCCUPANCY,
    HORIZONS,
    TONEMAP,
    SUN,
    BLOOM,
    REFLECTIONS,
    DITHER,
    SURFACE_POINT,
];
