use std::sync::Arc;

use engine_core::{Error, Pyramid};
use engine_terrain::Layers;
use serde_json::Value;

pub fn resolve_values(
    _value: &mut Value,
    _named: &[(String, Arc<Layers>)],
    _pyramid: &Pyramid,
    _path: &str,
) -> Result<(), Error> {
    Ok(())
}
