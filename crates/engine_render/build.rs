//! Declares `SOURCE`, the hash of this crate's code, which keys the records
//! it makes: impostors' layers and the scene's pipeline list; and in
//! `sources`, `WATER`, the hash of the code settling water on the GPU
//! alone, the shaders raising its bed by its materials' relief among it,
//! which settled water's records are keyed by with the terrain's code
//! preparing it ([`engine_sources`]).

use engine_sources::Set;

fn main() {
    engine_sources::emit_sets(&[Set {
        name: "WATER",
        roots: &["water_settle", "water_kernels"],
        // The plumbing of buffers, bindings and pipelines, and the terrain
        // group and the setup's passes the raise of its bed binds through:
        // the scheme's shader, and the shaders the raise is written in, are
        // hashed as files of their own.
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
