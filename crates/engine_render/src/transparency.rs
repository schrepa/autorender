engine_core::settings! {
    #[section("transparency")]
    #[check(sorted_weights)]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    pub struct TransparencySettings {
        pub near_distance: f32 = 5.0 => above(0.0),
        pub far_distance: f32 = 200.0 => above(0.0),
        pub least_weight: f32 = 0.01 => at_least(0.0),
        pub most_weight: f32 = 3000.0 => above(0.0),
    }
}

fn sorted_weights(settings: &TransparencySettings) -> Result<(), engine_core::InvalidSetting> {
    if settings.least_weight <= settings.most_weight {
        return Ok(());
    }
    let unsorted = engine_core::InvalidSetting::new("least_weight", engine_core::Expected::Sorted);
    Err(unsorted
        .got(settings.least_weight)
        .nearest(settings.most_weight))
}

impl TransparencySettings {
    pub fn weighting(&self) -> [f32; 4] {
        [
            self.near_distance,
            self.far_distance,
            self.least_weight,
            self.most_weight,
        ]
    }
}
