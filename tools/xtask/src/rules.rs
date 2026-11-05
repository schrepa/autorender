use std::path::Path;
use std::sync::OnceLock;

use engine_checks::{Errors, Exceptions, Repository, Rules};

use crate::scaffold::TEMPLATE_CRATE;

pub const RULES: Rules = Rules {
    exceptions: Exceptions {
        long_files: &[
            ("crates/engine_water/src/shaders/water/sim.wgsl", 1645),
            ("crates/engine_water/src/steady_water.rs", 2360),
            ("tools/lodbuild/src/simplify.rs", 2009),
        ],
        wide_structs: &[
            (
                "crates/engine_core/src/interpolation/gpu.rs",
                "GpuInputs",
                35,
            ),
            ("crates/engine_scatter/src/gpu.rs", "ScatterGpu", 34),
        ],
        unsafe_files: &[
            "crates/engine_profile/src/allocator.rs",
            "crates/engine_profiler/src/counters.rs",
            "crates/engine_profiler/src/load.rs",
            "crates/engine_profiler/src/pix.rs",
            "crates/engine_profiler/src/renderdoc.rs",
            "crates/engine_profiler/src/sampler/linux.rs",
            "crates/engine_profiler/src/sampler/macos.rs",
            "crates/engine_profiler/src/sampler/mod.rs",
            "crates/engine_profiler/src/sampler/windows.rs",
            "crates/engine_profiler/src/stack.rs",
            "crates/engine_render/src/renderer/native.rs",
        ],
    },
    errors: Some(Errors {
        within: "crates/",
        file: "crates/engine_codec/src/error.rs",
    }),
};

pub fn repository(root: &Path) -> Result<&'static Repository, String> {
    static REPOSITORY: OnceLock<Result<Repository, String>> = OnceLock::new();
    REPOSITORY
        .get_or_init(|| Repository::read(root))
        .as_ref()
        .map_err(Clone::clone)
}

pub fn warnings_allowed(root: &Path) -> Result<Vec<&'static str>, String> {
    let repository = repository(root)?;
    Ok(repository
        .own()
        .map(|package| package.name.as_str())
        .filter(|name| *name != TEMPLATE_CRATE && repository.free(name))
        .collect())
}
