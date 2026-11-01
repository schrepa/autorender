use serde::Serialize;

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StreamedIdView {
    pub id: String,
}
