use bevy_ecs::prelude::{Resource, World};

use crate::Atmosphere;

pub use engine_settings::InvalidSetting;

pub trait EnvironmentSection:
    Resource + Copy + Default + PartialEq + serde::Serialize + std::fmt::Debug
{
    const KEY: &'static str;
    const NAME: &'static str;
    const OPT_IN: bool;
    const FEATURE: Option<crate::Feature>;
    fn section_off() -> Self;
    fn section_check(&self) -> Result<(), InvalidSetting>;
    #[must_use]
    fn section_sanitized(self) -> Self;
    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema;
}

#[macro_export]
macro_rules! environment_sections {
    ($then:ident) => {
        $then! {
            shadows: Shadows { opt_in: false, feature: Core },
            ambient_occlusion: AmbientOcclusion { opt_in: false, feature: Core },
            sky_occlusion: SkyOcclusion { opt_in: false, feature: Core },
            bloom: Bloom { opt_in: false, feature: Core },
            dither: Dither { opt_in: false, feature: Core },
            auto_exposure: AutoExposure { opt_in: true, feature: Core },
            local_lights: LocalLightSettings { opt_in: false, feature: Core },
            highlights: Highlights { opt_in: false, feature: Core },
            reflections: Reflections { opt_in: true, feature: Core },
            particles: ParticleSettings { opt_in: false, feature: Core },
            reflection_probes: ReflectionProbeSettings { opt_in: false, feature: Core },
            virtual_geometry: VirtualGeometry { opt_in: false, feature: Core },
            level_of_detail: LevelOfDetail { opt_in: false, feature: Core },
            terrain: TerrainSettings { opt_in: false, feature: Terrain },
            ground_cover: GroundCover { opt_in: false, feature: Terrain },
            wind: Wind { opt_in: false, feature: Terrain },
            foliage: Foliage { opt_in: false, feature: Core },
            occupancy: Occupancy { opt_in: false, feature: Terrain },
            water: Water { opt_in: false, feature: Terrain },
            ocean: Ocean { opt_in: false, feature: Ocean },
            deformation: Deformation { opt_in: false, feature: Deformation },
            clouds: Clouds { opt_in: true, feature: Core },
            world_streaming: WorldStreaming { opt_in: false, feature: WorldStreaming },
            sky: Sky { opt_in: true, feature: Core },
            haze: Haze { opt_in: false, feature: Core },
            night_sky: NightSky { opt_in: false, feature: Core },
            time_of_day: TimeOfDay { opt_in: true, feature: TimeOfDay },
            navigation: Navigation { opt_in: false, feature: Navigation },
            physics: Physics { opt_in: false, feature: Physics },
            occlusion_culling: OcclusionCulling { opt_in: false, feature: Core },
            subpixel_culling: SubpixelCulling { opt_in: false, feature: Core },
            temporal_antialiasing: TemporalAntialiasing { opt_in: false, feature: Core },
            resolution: Resolution { opt_in: false, feature: Core },
            dynamic_resolution: DynamicResolution { opt_in: true, feature: Core },
            debug_view: DebugView { opt_in: true, feature: Core },
        }
    };
}

macro_rules! section_feature {
    (Core) => {
        None
    };
    ($feature:ident) => {
        Some(crate::Feature::$feature)
    };
}

macro_rules! core_sections {
    ($($key:ident: $settings:ident { opt_in: $opt_in:literal, feature: $feature:ident }),* $(,)?) => {
        $(
            impl EnvironmentSection for crate::$settings {
                const KEY: &'static str = stringify!($key);
                const NAME: &'static str = stringify!($settings);
                const OPT_IN: bool = $opt_in;
                const FEATURE: Option<crate::Feature> = section_feature!($feature);
                fn section_off() -> Self {
                    crate::$settings::off()
                }
                fn section_check(&self) -> Result<(), InvalidSetting> {
                    self.validate()
                }
                fn section_sanitized(self) -> Self {
                    self.sanitized()
                }
                #[cfg(feature = "schema")]
                fn schema() -> schemars::Schema {
                    schemars::schema_for!(crate::$settings)
                }
            }
        )*

        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct Environment {
            pub atmosphere: Atmosphere,
            $(pub $key: crate::$settings,)*
        }

        impl Environment {
            pub const KEYS: &'static [&'static str] = &[$(stringify!($key)),*];

            pub fn current(world: &World) -> Self {
                let mut environment = Self {
                    atmosphere: world.get_resource::<Atmosphere>().copied().unwrap_or_default(),
                    $($key: world
                        .get_resource::<crate::$settings>()
                        .copied()
                        .unwrap_or_else(<crate::$settings as EnvironmentSection>::section_off),)*
                };
                if environment.time_of_day.enabled {
                    environment.atmosphere.sun_direction =
                        environment.time_of_day.sanitized().sun_direction();
                }
                environment
            }

            #[must_use]
            pub fn sanitized(self) -> Self {
                Self {
                    atmosphere: self.atmosphere.sanitized(),
                    $($key: self.$key.sanitized(),)*
                }
            }
        }

        pub(crate) fn init_sections(world: &mut World, feature: Option<crate::Feature>) {
            $(if !$opt_in && section_feature!($feature) == feature {
                world.init_resource::<crate::$settings>();
            })*
        }
    };
}

environment_sections!(core_sections);
