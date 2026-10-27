//! The screen effects, one entry each: whether a frame runs it (its gate,
//! from the frame's environment), its field of the effects' uniform, and
//! what the effects pass reads for it.
//! The frame's targets, the effects pass, its stats and the uniform's
//! "runs" flags all follow the same gates; the post composite and the
//! temporal resolve apply the effects alike (`effect_shade`, `reflected`
//! and `effect_light` in effects.wgsl), the resolve ambient occlusion and
//! reflections before its blend, so they enter the history, and the haze
//! and bloom after it.

use engine_core::Environment;

use crate::render_graph::{NodeBuilder, Resource};

/// What the effects' uniform fields read beyond the settings: the turn of
/// the haze's jitter this frame, and the pixels across and down the haze's
/// history holds (none without one).
pub(crate) struct UniformFrame {
    pub(crate) haze_turn: f32,
    pub(crate) haze_history: [f32; 2],
}

/// Whether the haze's march runs under `environment`: haze on, with a
/// reach, through air that scatters: the haze's own, as thick as its
/// strength makes it, or the fog, which the physically based sky's
/// lighting zeroes. Shadows and the clouds' shadows shade it where they
/// are on; without them the sun lights all of its air. Terrains' bounce
/// light, where it is on, hides from its air the sky their probes do not
/// see; without it the air sees the whole sky.
pub(crate) fn haze_runs(environment: &Environment) -> bool {
    let haze = environment.haze.sanitized();
    let atmosphere = environment.atmosphere.sanitized();
    let fog = !environment.sky.sanitized().enabled
        && atmosphere.fog_density > 0.0
        && atmosphere.fog_max > 0.0;
    haze.enabled && haze.reach > 0.0 && (haze.strength > 0.0 || fog)
}

/// Whether frames under `environment` draw into the screen effects'
/// targets for what reads the scene there: an effect, or auto exposure's
/// meter.
pub(crate) fn targets_read(environment: &Environment) -> bool {
    Effects::of(environment).any() || environment.auto_exposure.enabled
}

/// Builds [`Effects`], which effects a frame runs, from one entry per
/// effect: its gate, its uniform field (`on` is 1 when it runs) and the
/// resources the effects pass reads for it.
macro_rules! screen_effects {
    ($(
        $(#[$doc:meta])*
        $effect:ident {
            gate: |$environment:ident| $gate:expr,
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
                Self {
                    $($effect: {
                        let $environment = environment;
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
    /// Ambient occlusion: the depth buffer sampled at every pixel, handing
    /// over by its radius's span on screen as the occupancy maps' texels
    /// do.
    occlusion {
        gate: |environment| environment.ambient_occlusion.sanitized().enabled(),
        uniform: |settings, on, _| {
            let occlusion = settings.ambient_occlusion.sanitized();
            let share = crate::occupancy::texel_share(&settings.occupancy.sanitized());
            [occlusion.radius, occlusion.intensity, share, on]
        },
        reads: [Depth],
    },
    /// Bloom: the glowing light filtered down a chain of levels from half
    /// the render size, each half the one before, and back up.
    bloom {
        gate: |environment| environment.bloom.sanitized().enabled(),
        uniform: |settings, on, _| [settings.bloom.sanitized().intensity, on, 0.0, 0.0],
        reads: [Scene, Glow],
    },
    /// Screen-space reflections: every pixel's ray, by its own shading
    /// normal and roughness from the surface target, walked over a pyramid
    /// of the depth buffer's nearest depths every frame, on past the sheets
    /// and objects it passes behind by what the surface target says lies
    /// behind each pixel.
    reflections {
        gate: |environment| environment.reflections.sanitized().enabled(),
        uniform: |settings, on, _| {
            let reflections = settings.reflections.sanitized();
            [reflections.distance, reflections.intensity, on, 0.0]
        },
        reads: [Depth, Scene, Glow, Surface],
    },
    /// The haze's march ([`haze_runs`]): the sunlight the near air scatters
    /// through the shadow clipmap and the clouds' shadows, and the sky's
    /// light from the share of the sky terrains' bounce probes see, at every
    /// pixel, filtered over time with the history the last frame left.
    haze {
        gate: |environment| haze_runs(environment),
        uniform: |_settings, on, frame| {
            let [across, down] = frame.haze_history;
            [frame.haze_turn, across, down, on]
        },
        reads: [Depth, ShadowMaps, CloudDome, Bounce, Haze],
    },
    /// Sky occlusion: sky light fading where still meshes hide the sky, from
    /// the shadow pass's sky maps (`shadows::sky`), which need shadows.
    sky {
        gate: |environment| {
            environment.sky_occlusion.sanitized().darkens() && environment.shadows.enabled
        },
        uniform: |settings, on, _| {
            let sky = settings.sky_occlusion.sanitized();
            [sky.strength, sky.floor, 0.0, on]
        },
        reads: [Depth, ShadowMaps],
    },
}
