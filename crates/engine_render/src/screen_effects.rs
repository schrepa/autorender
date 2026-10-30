use engine_core::{
    AmbientOcclusion, AutoExposure, Bloom, Haze, Reflections, RenderSnapshot, Shadows, Sky,
    SkyLighting, SkyOcclusion,
};

pub fn haze_runs(snapshot: &RenderSnapshot) -> bool {
    let haze = snapshot.section::<Haze>();
    let sky_light = snapshot.section::<SkyLighting>().values();
    let fog = !snapshot.section::<Sky>().enabled
        && sky_light.fog_density > 0.0
        && sky_light.fog_max > 0.0;
    haze.enabled && haze.reach > 0.0 && (haze.strength > 0.0 || fog)
}

pub(crate) fn targets_read(snapshot: &RenderSnapshot) -> bool {
    Effects::of(snapshot).any() || snapshot.section::<AutoExposure>().enabled
}

macro_rules! screen_effects {
    ($(
        $(#[$doc:meta])*
        $effect:ident {
            gate: |$snapshot:ident| $gate:expr,
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
            pub(crate) fn of(snapshot: &RenderSnapshot) -> Self {
                Self {
                    $($effect: {
                        let $snapshot = snapshot;
                        $gate
                    },)*
                }
            }

            pub(crate) fn any(self) -> bool {
                false $(|| self.$effect)*
            }

            pub(crate) fn uniforms(self, snapshot: &RenderSnapshot) -> EffectUniforms {
                EffectUniforms {
                    $($effect: {
                        let ($settings, $on) = (
                            snapshot,
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
        gate: |snapshot| snapshot.section::<AmbientOcclusion>().enabled(),
        uniform: |snapshot, on| {
            let occlusion = snapshot.section::<AmbientOcclusion>();
            [occlusion.radius, occlusion.intensity, 0.0, on]
        },
    },
    bloom {
        gate: |snapshot| snapshot.section::<Bloom>().enabled(),
        uniform: |snapshot, on| [snapshot.section::<Bloom>().intensity, on, 0.0, 0.0],
    },
    reflections {
        gate: |snapshot| snapshot.section::<Reflections>().enabled(),
        uniform: |snapshot, on| {
            let reflections = snapshot.section::<Reflections>();
            [reflections.distance, reflections.intensity, on, 0.0]
        },
    },
    haze {
        gate: |snapshot| haze_runs(snapshot),
        uniform: |_snapshot, on| [0.0, 0.0, 0.0, on],
    },
    sky {
        gate: |snapshot| {
            snapshot.section::<SkyOcclusion>().darkens() && snapshot.section::<Shadows>().enabled
        },
        uniform: |snapshot, on| {
            let sky = snapshot.section::<SkyOcclusion>();
            [sky.strength, sky.floor, 0.0, on]
        },
    },
}
