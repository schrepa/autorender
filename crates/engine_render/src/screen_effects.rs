use engine_core::Environment;

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
            uniform: |$settings:ident, $on:ident| $uniform:expr,
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

            pub(crate) fn uniforms(self, environment: &Environment) -> EffectUniforms {
                EffectUniforms {
                    $($effect: {
                        let ($settings, $on) = (
                            environment,
                            if self.$effect { 1.0 } else { 0.0 },
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
        uniform: |settings, on| {
            let occlusion = settings.ambient_occlusion.sanitized();
            [occlusion.radius, occlusion.intensity, 0.0, on]
        },
    },
    bloom {
        gate: |environment| environment.bloom.sanitized().enabled(),
        uniform: |settings, on| [settings.bloom.sanitized().intensity, on, 0.0, 0.0],
    },
    reflections {
        gate: |environment| environment.reflections.sanitized().enabled(),
        uniform: |settings, on| {
            let reflections = settings.reflections.sanitized();
            [reflections.distance, reflections.intensity, on, 0.0]
        },
    },
    haze {
        gate: |environment| haze_runs(environment),
        uniform: |_settings, on| [0.0, 0.0, 0.0, on],
    },
    sky {
        gate: |environment| {
            environment.sky_occlusion.sanitized().darkens() && environment.shadows.enabled
        },
        uniform: |settings, on| {
            let sky = settings.sky_occlusion.sanitized();
            [sky.strength, sky.floor, 0.0, on]
        },
    },
}
