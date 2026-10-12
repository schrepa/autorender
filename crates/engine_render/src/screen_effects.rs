//! The screen effects, one entry each: whether a frame runs it (its gate,
//! from the frame's environment and the direction toward the sun), its
//! fields of the effects' uniform, and what the effects pass reads for it.
//! The frame's targets, the effects pass, its stats and the uniform's
//! "runs" flags all follow the same gates; the post composite and the
//! temporal resolve apply the effects alike (`effect_shade` and
//! `effect_light` in effects.wgsl).

use engine_core::Environment;
use engine_core::glam::Vec3;

use crate::render_graph::{NodeBuilder, Resource};

/// What the effects' uniform fields read beyond the settings: the size of
/// a half-resolution texel in texture coordinates, and which pixel of each
/// 2x2 block reflections trace this frame.
pub(crate) struct UniformFrame {
    pub(crate) half_texel: [f32; 2],
    pub(crate) trace_index: u32,
}

/// Builds [`Effects`], which effects a frame runs, from one entry per
/// effect: its gate, its uniform field (`on` is 1 when it runs) and the
/// resources the effects pass reads for it.
macro_rules! screen_effects {
    ($(
        $(#[$doc:meta])*
        $effect:ident {
            gate: |$environment:ident, $sun:pat_param| $gate:expr,
            uniform: |$settings:ident, $on:ident, $frame:pat_param| $uniform:expr,
            reads: [$($read:ident),*],
        },
    )*) => {
        /// The screen effects a frame runs.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub(crate) struct Effects {
            $($(#[$doc])* pub(crate) $effect: bool,)*
        }

        /// Each effect's field of the effects' uniform.
        pub(crate) struct EffectUniforms {
            $(pub(crate) $effect: [f32; 4],)*
        }

        impl Effects {
            /// The effects `environment` runs.
            pub(crate) fn of(environment: &Environment) -> Self {
                let toward_sun: Vec3 = environment
                    .atmosphere
                    .sanitized()
                    .sun_direction
                    .normalize_or_zero();
                Self {
                    $($effect: {
                        let ($environment, $sun) = (environment, toward_sun);
                        $gate
                    },)*
                }
            }

            /// Whether any runs.
            pub(crate) fn any(self) -> bool {
                false $(|| self.$effect)*
            }

            /// The effects pass `node` reading what the effects that run
            /// read.
            pub(crate) fn reads(self, mut node: NodeBuilder<'_>) -> NodeBuilder<'_> {
                $(if self.$effect {
                    $(node = node.read(Resource::$read);)*
                })*
                node
            }

            /// Each effect's uniform field under `environment`'s settings
            /// this `frame`, flagged where it runs.
            pub(crate) fn uniforms(
                self,
                environment: &Environment,
                frame: &UniformFrame,
            ) -> EffectUniforms {
                EffectUniforms {
                    $($effect: {
                        let ($settings, $on, $frame) = (
                            environment,
                            if self.$effect { 1.0 } else { 0.0 },
                            frame,
                        );
                        $uniform
                    },)*
                }
            }
        }
    };
}

screen_effects! {
    /// Ambient occlusion: the depth buffer sampled at half resolution.
    occlusion {
        gate: |environment, _| environment.ambient_occlusion.sanitized().enabled(),
        uniform: |settings, on, _| {
            let occlusion = settings.ambient_occlusion.sanitized();
            [occlusion.radius, occlusion.intensity, 0.0, on]
        },
        reads: [Depth],
    },
    /// Bloom: the glowing light filtered down a chain of half-resolution
    /// levels and back up.
    bloom {
        gate: |environment, _| environment.bloom.sanitized().enabled(),
        uniform: |settings, on, frame| {
            let [x, y] = frame.half_texel;
            [settings.bloom.sanitized().intensity, on, x, y]
        },
        reads: [Scene, Glow],
    },
    /// Screen-space reflections: the surface target's rays walked over a
    /// pyramid of the depth buffer's nearest depths.
    reflections {
        gate: |environment, _| environment.reflections.sanitized().enabled(),
        uniform: |settings, on, frame| {
            let reflections = settings.reflections.sanitized();
            [reflections.distance, reflections.intensity, on, frame.trace_index as f32]
        },
        reads: [Depth, Scene, Glow, Surface],
    },
    /// Sun shafts: light gathered through the shadow maps, which they
    /// need.
    shafts {
        gate: |environment, _| {
            environment.sun_shafts.sanitized().enabled() && environment.shadows.enabled
        },
        uniform: |settings, on, _| {
            let shafts = settings.sun_shafts.sanitized();
            [shafts.intensity, shafts.anisotropy, shafts.distance, on]
        },
        reads: [Depth, ShadowMaps],
    },
    /// Contact shadows, traced with the occlusion toward a sun above the
    /// horizon.
    contact {
        gate: |environment, toward_sun| {
            environment.contact_shadows.sanitized().traces() && toward_sun.y > 0.0
        },
        uniform: |settings, on, _| {
            let contact = settings.contact_shadows.sanitized();
            [contact.length, contact.strength, contact.distance, on]
        },
        reads: [Depth],
    },
}
