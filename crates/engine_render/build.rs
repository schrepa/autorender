use engine_sources::Set;

fn main() {
    engine_sources::emit_sets(&[Set {
        name: "WATER",
        roots: &["water_settle", "water_kernels"],
        stop: &[
            "bindings",
            "buffers",
            "gpu_data",
            "gpu_layouts",
            "pipeline_layouts",
            "pipelines",
            "platform",
            "profiling",
            "shaders",
            "terrain",
            "water_sim",
        ],
        files: &[
            "src/water_bursts.wgsl",
            "src/water_sim.wgsl",
            "src/meshes/water_setup.wgsl",
            "src/meshes/terrain.wgsl",
        ],
        crates: Some(&[]),
    }]);
}
