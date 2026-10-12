//! The look as data: a world's atmosphere and the settings of
//! every rendering effect. Each settings type is a resource, a section of a
//! scene's `environment` in the same JSON shape (the atmosphere's section is
//! a preset with overrides), and checks itself with `validate`; frames carry
//! them all sanitized as [`crate::RenderSnapshot::environment`].

use bevy_ecs::prelude::{Resource, World};

use crate::Atmosphere;

pub use engine_settings::InvalidSetting;

/// An environment section: the settings of one engine feature, a resource
/// that is section [`Self::KEY`] of a scene `environment` (from scene version
/// [`Self::SINCE`]), of the environment control and of inspection, in the
/// same JSON shape. A world without the resource reads as
/// [`Self::section_off`]; opt-in sections are off unless a scene names them,
/// the others start at their defaults. [`environment_sections`] implements it
/// for every type it lists, from each type's own `off`, `validate` and
/// `sanitized`.
pub trait EnvironmentSection:
    Resource + Copy + Default + PartialEq + serde::Serialize + std::fmt::Debug
{
    /// The section's key in a scene environment, the environment control and
    /// inspection.
    const KEY: &'static str;
    /// The settings type's name.
    const NAME: &'static str;
    /// The scene version that first authors the section.
    const SINCE: u32;
    /// Whether the section is off unless a scene names it.
    const OPT_IN: bool;
    /// The feature whose plugin adds the section; None for the renderer's
    /// own, always present.
    const FEATURE: Option<crate::Feature>;
    /// What a world without the resource reads as: the feature off.
    fn section_off() -> Self;
    /// The first invalid field and the rule it breaks, if any.
    fn section_check(&self) -> Result<(), InvalidSetting>;
    /// Every value usable: out-of-range values clamp, unusable ones fall back.
    #[must_use]
    fn section_sanitized(self) -> Self;
    /// The section's JSON schema.
    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema;
}

/// Every environment section in authored order: the one list of them. Each
/// entry is `key: Settings { since: VERSION, opt_in: BOOL, feature: FEATURE }`,
/// the section's key, its settings type in this crate (which has `off`,
/// `validate` and `sanitized`), the scene version that first authors it,
/// whether it is off unless a scene names it, and the [`crate::Feature`] whose
/// plugin adds it (`Core` for the renderer's own, always present). Code listing sections passes a macro that
/// receives this list (`environment_sections!(callback)`) instead of
/// writing them out, so a section added here reaches the environment
/// snapshot, resource registration, scenes, the environment control and
/// inspection at once.
#[macro_export]
macro_rules! environment_sections {
    ($then:ident) => {
        $then! {
            shadows: Shadows { since: 5, opt_in: false, feature: Core },
            ambient_occlusion: AmbientOcclusion { since: 5, opt_in: false, feature: Core },
            bloom: Bloom { since: 5, opt_in: false, feature: Core },
            local_lights: LocalLightSettings { since: 6, opt_in: false, feature: Core },
            sun_shafts: SunShafts { since: 6, opt_in: true, feature: Core },
            reflections: Reflections { since: 6, opt_in: true, feature: Core },
            particles: ParticleSettings { since: 6, opt_in: false, feature: Core },
            refraction: Refraction { since: 8, opt_in: true, feature: Core },
            decals: DecalSettings { since: 8, opt_in: false, feature: Core },
            reflection_probes: ReflectionProbeSettings { since: 8, opt_in: false, feature: Core },
            virtual_geometry: VirtualGeometry { since: 8, opt_in: false, feature: Core },
            level_of_detail: LevelOfDetail { since: 8, opt_in: false, feature: Core },
            terrain: TerrainSettings { since: 9, opt_in: false, feature: Terrain },
            ground_cover: GroundCover { since: 9, opt_in: false, feature: Terrain },
            wind: Wind { since: 9, opt_in: false, feature: Terrain },
            occupancy: Occupancy { since: 9, opt_in: false, feature: Terrain },
            contact_shadows: ContactShadows { since: 9, opt_in: false, feature: Core },
            water: Water { since: 9, opt_in: false, feature: Terrain },
            ocean: Ocean { since: 9, opt_in: false, feature: Ocean },
            deformation: Deformation { since: 9, opt_in: false, feature: Deformation },
            clouds: Clouds { since: 9, opt_in: true, feature: Core },
            world_streaming: WorldStreaming { since: 9, opt_in: false, feature: WorldStreaming },
            sky: Sky { since: 9, opt_in: true, feature: Core },
            time_of_day: TimeOfDay { since: 9, opt_in: true, feature: TimeOfDay },
            navigation: Navigation { since: 9, opt_in: false, feature: Navigation },
            occlusion_culling: OcclusionCulling { since: 9, opt_in: false, feature: Core },
            subpixel_culling: SubpixelCulling { since: 9, opt_in: false, feature: Core },
            temporal_antialiasing: TemporalAntialiasing { since: 9, opt_in: false, feature: Core },
        }
    };
}

/// A section's feature: None for the renderer's own (`Core`).
macro_rules! section_feature {
    (Core) => {
        None
    };
    ($feature:ident) => {
        Some(crate::Feature::$feature)
    };
}

/// engine_core's part: each section's [`EnvironmentSection`], the
/// [`Environment`] snapshot and the registration of default settings.
macro_rules! core_sections {
    ($($key:ident: $settings:ident { since: $since:literal, opt_in: $opt_in:literal, feature: $feature:ident }),* $(,)?) => {
        $(
            impl EnvironmentSection for crate::$settings {
                const KEY: &'static str = stringify!($key);
                const NAME: &'static str = stringify!($settings);
                const SINCE: u32 = $since;
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

        /// Every look setting of a world: its atmosphere and each section's
        /// settings, by section key.
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct Environment {
            pub atmosphere: Atmosphere,
            $(pub $key: crate::$settings,)*
        }

        impl Environment {
            /// Every section's key, in order.
            pub const KEYS: &'static [&'static str] = &[$(stringify!($key)),*];

            /// `world`'s settings as its resources hold them. A missing
            /// atmosphere reads as the default one, and a missing section
            /// resource as its feature off. An enabled time of day sets the
            /// atmosphere's sun direction.
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

            /// Every section sanitized: usable for rendering whatever it held.
            #[must_use]
            pub fn sanitized(self) -> Self {
                Self {
                    atmosphere: self.atmosphere.sanitized(),
                    $($key: self.$key.sanitized(),)*
                }
            }
        }

        /// Inserts the default settings of every section of `feature` (the
        /// renderer's own for None) that is on unless a scene turns it off:
        /// all but the opt-in ones. Feature plugins insert theirs as they
        /// build.
        pub(crate) fn init_sections(world: &mut World, feature: Option<crate::Feature>) {
            $(if !$opt_in && section_feature!($feature) == feature {
                world.init_resource::<crate::$settings>();
            })*
        }
    };
}

environment_sections!(core_sections);
