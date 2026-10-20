//! Upload structs of `crate::debug_view`.

use bytemuck::{Pod, Zeroable};

gpu_structs! {
    /// WGSL `DebugViewUniform` (`debug_view.wgsl`).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct DebugViewUniform as ["DebugViewUniform"] {
        pub view: [u32; 4],
        pub sun: [f32; 4],
    }
}
