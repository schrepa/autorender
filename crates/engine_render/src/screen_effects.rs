use engine_core::{AmbientOcclusion, RenderSnapshot, Shadows, SkyOcclusion};

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
