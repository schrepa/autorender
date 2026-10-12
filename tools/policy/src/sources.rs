//! Source rules: unsafe code stays in the files the list names.

use crate::{Problems, rust};

/// The files that may contain unsafe code, each for a reason. Every other
/// Rust file has none (the workspace denies `unsafe_code`, and a file
/// allowing it must be listed here), and every listed file still has some.
pub const UNSAFE_FILES: &[(&str, &str)] = &[
    (
        "crates/engine_profile/src/allocator.rs",
        "implements the standard allocator's unsafe contract",
    ),
    (
        "crates/engine_profiler/src/counters.rs",
        "Vulkan's layer entry points and calls",
    ),
    (
        "crates/engine_profiler/src/load.rs",
        "Windows' process, processor time and GPU counter calls",
    ),
    (
        "crates/engine_profiler/src/pix.rs",
        "loads PIX's capturer and calls its capture functions",
    ),
    (
        "crates/engine_profiler/src/renderdoc.rs",
        "calls RenderDoc's capture functions",
    ),
    (
        "crates/engine_profiler/src/sampler/linux.rs",
        "signals threads and reads their registers",
    ),
    (
        "crates/engine_profiler/src/sampler/macos.rs",
        "suspends threads and reads their registers",
    ),
    (
        "crates/engine_profiler/src/sampler/mod.rs",
        "follows frame pointers within a thread's stack and reads sampled code",
    ),
    (
        "crates/engine_profiler/src/sampler/windows.rs",
        "suspends threads, reads their registers and unwinds them",
    ),
    (
        "crates/engine_profiler/src/stack.rs",
        "queries the calling thread's stack limits and committed pages",
    ),
    (
        "crates/engine_render/src/browser.rs",
        "starts and stops a graphics debugger's capture",
    ),
    (
        "tools/vgbuild/src/ffi.rs",
        "calls the vendored C++ meshoptimizer",
    ),
];

/// Checks the Rust file `path` (relative, `/`-separated) with text
/// `source`.
pub fn check(path: &str, source: &str, problems: &mut Problems) {
    let split = rust::split(source);
    let listed = UNSAFE_FILES.iter().any(|(file, _)| *file == path);
    let first = split
        .code
        .iter()
        .position(|code| rust::has_word(code, "unsafe") || rust::has_word(code, "unsafe_code"));
    if !listed && let Some(line) = first {
        problems.at(
            path,
            line + 1,
            "unsafe code outside the files tools/policy/src/sources.rs lists (UNSAFE_FILES); \
             keep it in a listed file, or list this one with its reason",
        );
    }
    if listed && !split.code.iter().any(|code| rust::has_word(code, "unsafe")) {
        problems.at(
            path,
            1,
            "listed in UNSAFE_FILES (tools/policy/src/sources.rs) but has no unsafe code: \
             remove it from the list",
        );
    }
}
