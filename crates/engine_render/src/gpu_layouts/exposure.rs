//! Upload structs of `crate::exposure`.

use bytemuck::{Pod, Zeroable};

gpu_structs! {
    /// WGSL `ExposureUniform` (`exposure.wgsl`).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ExposureUniform as ["ExposureUniform"] {
        /// The authored exposure, the seconds since the last metered frame,
        /// how far exposure follows the view and the stops added after.
        pub frame: [f32; 4],
        /// The most stops the view raises and lowers exposure, and the
        /// seconds of easing toward a darker and a brighter view.
        pub reach: [f32; 4],
        /// The shares of the darkest and brightest samples left out, the
        /// middle's weight and the sky's.
        pub meter: [f32; 4],
        /// The physically based sky's light at noon (0 without it), the
        /// power of its light's ratio exposure rises by and the most stops.
        pub day: [f32; 4],
    }

    /// WGSL `ExposureState` (`exposure.wgsl`): the exposure frames show,
    /// its stops from the authored one, the meter's last average, whether
    /// these hold a metered frame, how much brighter the view is than its
    /// light gives a view, and the time of day's stops.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ExposureState as ["ExposureState"] {
        pub exposure: f32,
        pub stops: f32,
        pub metered: f32,
        pub held: f32,
        pub view: f32,
        pub day: f32,
    }
}
