//! The features of a frame's scene that shading looks up, as `scene.wgsl`'s
//! override constants: the pipelines shading the scene are specialized for
//! them, so lookups of features the scene's settings switch off compile
//! out. A feature is on whenever its settings let the frame's uniforms and
//! bindings hold it, so specialized pipelines shade exactly as unspecialized
//! ones. Settings alone decide, which scenes carry from their first frame:
//! terrains and probes that come later (a cave's interior opening once its
//! bakes are done) find their lookups compiled in, and the set changes,
//! those pipelines being created anew with their next draws, only when the
//! scene's settings do.

use engine_core::RenderSnapshot;

/// Which of the scene's features shading looks up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SceneShading(u8);

impl SceneShading {
    const SHADOW_MAPS: u8 = 1;
    const KEPT_LEVELS: u8 = 2;
    const TERRAIN_HORIZONS: u8 = 4;
    const CLOUD_SHADOWS: u8 = 8;
    const OCCUPANCY: u8 = 16;
    const PROBES: u8 = 32;

    /// Every feature, as pipelines given none of the constants look up.
    pub(crate) const ALL: Self = Self(63);

    /// The features `snapshot`'s settings let its frame hold, read as the
    /// passes filling their uniforms and bindings read them: shadows off
    /// zero the shadow lookups' uniform, and kept levels follow its
    /// settings; horizon maps exist for casting terrains with shadows on;
    /// clouds give shadows while they cover the sky; covers darken with a
    /// strength; the probe grid lists probes while they are on.
    pub(crate) fn of(snapshot: &RenderSnapshot) -> Self {
        let environment = &snapshot.environment;
        let shadows = environment.shadows.sanitized();
        let terrain = environment.terrain.sanitized();
        let clouds = environment.clouds.sanitized();
        let occupancy = environment.occupancy;
        let features = [
            (Self::SHADOW_MAPS, shadows.enabled),
            (Self::KEPT_LEVELS, shadows.enabled && shadows.levels > 0),
            (
                Self::TERRAIN_HORIZONS,
                environment.shadows.enabled && terrain.enabled && terrain.sun_shadows,
            ),
            (
                Self::CLOUD_SHADOWS,
                clouds.enabled && clouds.coverage > 0.0 && clouds.shadows > 0.0,
            ),
            (
                Self::OCCUPANCY,
                occupancy.enabled && occupancy.strength > 0.0,
            ),
            (Self::PROBES, environment.reflection_probes.enabled),
        ];
        Self(
            features
                .into_iter()
                .filter(|&(_, on)| on)
                .fold(0, |bits, (bit, _)| bits | bit),
        )
    }

    /// The override constants of pipelines shading the scene.
    pub(crate) fn constants(self) -> [(&'static str, f64); 6] {
        let set = |bit: u8| if self.0 & bit != 0 { 1.0 } else { 0.0 };
        [
            ("SHADOW_MAPS", set(Self::SHADOW_MAPS)),
            ("KEPT_LEVELS", set(Self::KEPT_LEVELS)),
            ("TERRAIN_HORIZONS", set(Self::TERRAIN_HORIZONS)),
            ("CLOUD_SHADOWS", set(Self::CLOUD_SHADOWS)),
            ("OCCUPANCY", set(Self::OCCUPANCY)),
            ("PROBES", set(Self::PROBES)),
        ]
    }

    /// `constants` and then this one's.
    pub(crate) fn with(self, constants: &[(&'static str, f64)]) -> Vec<(&'static str, f64)> {
        constants.iter().copied().chain(self.constants()).collect()
    }
}
