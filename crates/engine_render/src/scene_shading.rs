use engine_core::RenderSnapshot;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SceneShading(u8);

impl SceneShading {
    const SHADOW_MAPS: u8 = 1;
    const TERRAIN_HORIZONS: u8 = 2;
    const CLOUD_SHADOWS: u8 = 4;
    const OCCUPANCY: u8 = 8;
    const PROBES: u8 = 16;
    const BOUNCE: u8 = 32;

    pub(crate) const ALL: Self = Self(63);

    pub(crate) fn of(snapshot: &RenderSnapshot) -> Self {
        let environment = &snapshot.environment;
        let shadows = environment.shadows.sanitized();
        let terrain = environment.terrain.sanitized();
        let clouds = environment.clouds.sanitized();
        let occupancy = environment.occupancy;
        let horizons = environment.shadows.enabled && terrain.enabled && terrain.sun_shadows;
        let features = [
            (Self::SHADOW_MAPS, shadows.enabled),
            (Self::TERRAIN_HORIZONS, horizons),
            (
                Self::CLOUD_SHADOWS,
                clouds.enabled && clouds.coverage > 0.0 && clouds.shadows > 0.0,
            ),
            (
                Self::OCCUPANCY,
                occupancy.enabled && occupancy.strength > 0.0,
            ),
            (Self::PROBES, environment.reflection_probes.enabled),
            (Self::BOUNCE, horizons && terrain.bounce),
        ];
        Self(
            features
                .into_iter()
                .filter(|&(_, on)| on)
                .fold(0, |bits, (bit, _)| bits | bit),
        )
    }

    pub(crate) fn constants(self) -> [(&'static str, f64); 6] {
        let set = |bit: u8| if self.0 & bit != 0 { 1.0 } else { 0.0 };
        [
            ("SHADOW_MAPS", set(Self::SHADOW_MAPS)),
            ("TERRAIN_HORIZONS", set(Self::TERRAIN_HORIZONS)),
            ("CLOUD_SHADOWS", set(Self::CLOUD_SHADOWS)),
            ("OCCUPANCY", set(Self::OCCUPANCY)),
            ("PROBES", set(Self::PROBES)),
            ("BOUNCE", set(Self::BOUNCE)),
        ]
    }

    pub(crate) fn with(self, constants: &[(&'static str, f64)]) -> Vec<(&'static str, f64)> {
        constants.iter().copied().chain(self.constants()).collect()
    }
}
