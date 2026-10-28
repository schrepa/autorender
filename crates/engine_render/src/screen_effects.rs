use engine_core::Environment;

use crate::render_graph::{NodeBuilder, Resource};

pub(crate) struct UniformFrame {
    pub(crate) haze_turn: f32,
    pub(crate) haze_history: [f32; 2],
    pub(crate) world_terrains: u32,
}

pub(crate) fn haze_runs(environment: &Environment) -> bool {
    let haze = environment.haze.sanitized();
    let atmosphere = environment.atmosphere.sanitized();
    let fog = !environment.sky.sanitized().enabled
        && atmosphere.fog_density > 0.0
        && atmosphere.fog_max > 0.0;
    haze.enabled && haze.reach > 0.0 && (haze.strength > 0.0 || fog)
}

pub(crate) fn targets_read(environment: &Environment) -> bool {
    Effects::of(environment).any() || environment.auto_exposure.enabled
}

macro_rules! screen_effects {
    ($(
        $(#[$doc:meta])*
        $effect:ident {
            gate: |$environment:ident| $gate:expr,
            uniform: |$settings:ident, $on:ident, $frame:pat_param| $uniform:expr,
            reads: [$($read:ident),*],
        },
    )*) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub(crate) struct Effects {
            $($(#[$doc])* pub(crate) $effect: bool,)*
        }

        pub(crate) struct EffectUniforms {
            $(pub(crate) $effect: [f32; 4],)*
        }

        impl Effects {
            pub(crate) fn of(environment: &Environment) -> Self {
                Self {
                    $($effect: {
                        let $environment = environment;
                        $gate
                    },)*
                }
            }

            pub(crate) fn any(self) -> bool {
                false $(|| self.$effect)*
            }

            pub(crate) fn reads(self, mut node: NodeBuilder<'_>) -> NodeBuilder<'_> {
                $(if self.$effect {
                    $(node = node.read(Resource::$read);)*
                })*
                node
            }

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
    occlusion {
        gate: |environment| environment.ambient_occlusion.sanitized().enabled(),
        uniform: |settings, on, _| {
            let occlusion = settings.ambient_occlusion.sanitized();
            let share = crate::occupancy::texel_share(&settings.occupancy.sanitized());
            [occlusion.radius, occlusion.intensity, share, on]
        },
        reads: [Depth],
    },
    bloom {
        gate: |environment| environment.bloom.sanitized().enabled(),
        uniform: |settings, on, _| [settings.bloom.sanitized().intensity, on, 0.0, 0.0],
        reads: [Scene, Glow],
    },
    reflections {
        gate: |environment| environment.reflections.sanitized().enabled(),
        uniform: |settings, on, frame| {
            let reflections = settings.reflections.sanitized();
            [
                reflections.distance,
                reflections.intensity,
                on,
                frame.world_terrains as f32,
            ]
        },
        reads: [Depth, Scene, Glow, Surface],
    },
    haze {
        gate: |environment| haze_runs(environment),
        uniform: |_settings, on, frame| {
            let [across, down] = frame.haze_history;
            [frame.haze_turn, across, down, on]
        },
        reads: [Depth, ShadowMaps, CloudDome, Bounce, Haze],
    },
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
