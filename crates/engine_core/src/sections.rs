macro_rules! environment_section {
    ($settings:ident, $key:literal, opt_in: $opt_in:literal) => {
        impl $crate::bricks::EnvironmentSection for $settings {
            const KEY: &'static str = $key;
            const NAME: &'static str = stringify!($settings);
            const OPT_IN: bool = $opt_in;
            fn section_off() -> Self {
                Self::off()
            }
            fn section_check(&self) -> Result<(), engine_settings::InvalidSetting> {
                self.validate()
            }
            fn section_sanitized(self) -> Self {
                self.sanitized()
            }
            #[cfg(feature = "schema")]
            fn schema() -> schemars::Schema {
                schemars::schema_for!($settings)
            }
        }
    };
}

pub(crate) use environment_section;
