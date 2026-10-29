engine_render::shader_modules! {
    "face::lighting" in "lighting_model" = "faces/lighting.wgsl";
    "lighting::surface" in "lighting_model" = "surface.wgsl";
    "lighting::local" in "lighting_model" = "local.wgsl";
    "lighting::debug" in "lighting_model" = "debug.wgsl";
    "lighting::piece" in "lighting_model" = "piece.wgsl";
}
