use engine_core::Error;

use crate::composite::{PixelEffect, SCENE_MODULE, composites, scene_source};
use crate::parts::PartEntry;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generated {
    pub path: String,
    pub source: String,
}

pub fn generated(parts: &[&'static PartEntry]) -> Result<Vec<Generated>, Error> {
    let effects: Vec<PixelEffect> = parts
        .iter()
        .flat_map(|part| part.decl.effects.iter().copied())
        .collect();
    let mut made: Vec<Generated> = composites(&effects)?
        .into_iter()
        .map(|(path, _, source)| Generated { path, source })
        .collect();
    made.push(Generated {
        path: SCENE_MODULE.to_owned(),
        source: scene_source(&effects),
    });
    Ok(made)
}
