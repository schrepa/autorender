macro_rules! policy {
    ($(pub const $name:ident: $ty:ty = $value:expr;)*) => {
        $(pub const $name: $ty = $value;)*

        #[cfg(all(feature = "schema", not(target_arch = "wasm32")))]
        pub(crate) fn constants() -> Vec<(&'static str, serde_json::Value)> {
            vec![$((stringify!($name), serde_json::json!($name))),*]
        }
    };
}

policy! {
    pub const FRAME_INTERVAL_MS: f64 = 1000.0 / 60.0;
    pub const WHEEL_PIXELS: f32 = 100.0;
}
