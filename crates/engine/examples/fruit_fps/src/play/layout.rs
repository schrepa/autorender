use serde::Deserialize;

use crate::bots::MapLayout;

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Layout {
    #[serde(flatten)]
    pub map: MapLayout,
    pub collision: Vec<Collider>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Collider {
    pub name: String,
    #[serde(default)]
    pub surface: Option<String>,
}
