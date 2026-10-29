use engine_sources::Set;

const FILES: &[&str] = &[
    "src/bursts.wgsl",
    "src/flow.wgsl",
    "src/foam.wgsl",
    "src/footprint.wgsl",
    "src/particles.wgsl",
    "src/refraction.wgsl",
    "src/setup.wgsl",
    "src/sim.wgsl",
    "src/sim_shared.wgsl",
    "src/surface.wgsl",
    "../engine_render/src/color.wgsl",
    "../engine_render/src/frame/view.wgsl",
    "../engine_render/src/math.wgsl",
    "../engine_render/src/meshes/declarations.wgsl",
    "../engine_render/src/meshes/surface.wgsl",
    "../engine_render/src/meshes/vertex.wgsl",
    "../engine_render/src/noise.wgsl",
    "../engine_render/src/wind.wgsl",
    "../engine_faces/src/defaults/",
    "../engine_bloom/src/",
    "../engine_dither/src/",
    "../engine_lighting/src/",
    "../engine_lights/src/",
    "../engine_probes/src/",
    "../engine_reflections/src/",
    "../engine_scatter/src/",
    "../engine_shadows/src/",
    "../engine_sky/src/",
    "../engine_sun/src/",
    "../engine_terrain_shading/src/",
    "../engine_tonemap/src/",
];

fn main() {
    for file in FILES.iter().filter(|file| file.starts_with("../")) {
        println!("cargo:rerun-if-changed={file}");
    }
    engine_sources::emit_sets(&[Set {
        name: "WATER",
        roots: &["water_settle", "water_kernels"],
        stop: &[
            "bindings",
            "gpu_layouts",
            "pipeline_layouts",
            "shaders",
            "water_sim",
        ],
        files: FILES,
        crates: Some(&[]),
    }]);
}
