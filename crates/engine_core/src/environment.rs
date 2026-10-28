use bevy_ecs::prelude::World;

use crate::{Atmosphere, EnvironmentSection};

pub use engine_settings::InvalidSetting;

macro_rules! render_environment {
    ($($key:ident: $settings:ident,)*) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct Environment {
            pub atmosphere: Atmosphere,
            $(pub $key: crate::$settings,)*
        }

        impl Environment {
            pub fn current(world: &World) -> Self {
                Self {
                    atmosphere: world.get_resource::<Atmosphere>().copied().unwrap_or_default(),
                    $($key: world
                        .get_resource::<crate::$settings>()
                        .copied()
                        .unwrap_or_else(<crate::$settings as EnvironmentSection>::section_off),)*
                }
            }

            #[must_use]
            pub fn sanitized(self) -> Self {
                Self {
                    atmosphere: self.atmosphere.sanitized(),
                    $($key: self.$key.sanitized(),)*
                }
            }
        }
    };
}

render_environment! {
    shadows: Shadows,
    ambient_occlusion: AmbientOcclusion,
    sky_occlusion: SkyOcclusion,
    bloom: Bloom,
    dither: Dither,
    auto_exposure: AutoExposure,
    local_lights: LocalLightSettings,
    highlights: Highlights,
    reflections: Reflections,
    particles: ParticleSettings,
    reflection_probes: ReflectionProbeSettings,
    level_of_detail: LevelOfDetail,
    foliage: Foliage,
    clouds: Clouds,
    sky: Sky,
    haze: Haze,
    night_sky: NightSky,
    occlusion_culling: OcclusionCulling,
    subpixel_culling: SubpixelCulling,
    temporal_antialiasing: TemporalAntialiasing,
    resolution: Resolution,
    dynamic_resolution: DynamicResolution,
    debug_view: DebugView,
}
