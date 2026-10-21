//! Declares `SOURCE`, the hash of this crate's code, which keys the records
//! it makes: impostors' layers, the runtime virtual texture's far pages and
//! the scene's pipeline list; and in `sources`, `WATER`, the hash of the
//! code settling water on the GPU alone, which settled water's records are
//! keyed by with the terrain's code preparing it ([`engine_sources`]).

use engine_sources::Set;

fn main() {
    engine_sources::emit_sets(&[Set {
        name: "WATER",
        roots: &["water_settle", "water_kernels"],
        // The plumbing of buffers, bindings and pipelines: the scheme's
        // shader is hashed as a file of its own.
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
        ],
        files: &["src/water_sim.wgsl"],
        crates: Some(&[]),
    }]);
}
