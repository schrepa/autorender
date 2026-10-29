engine_render::shader_modules! {
    "lights::clusters" in "lights" = "clusters.wgsl";
    "lights::binning" in "lights" = "binning.wgsl";
    "face::lights" in "lights" = "faces/lights.wgsl";
}

engine_render::entry_points! {
    lights {
        compute: [
            count_lights @ "lights::binning",
            bin_lights @ "lights::binning",
            scan_clusters @ "lights::binning",
        ],
    }
}
