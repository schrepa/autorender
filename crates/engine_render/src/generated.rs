use engine_core::Error;

use crate::composite::{composites, resolves};
use crate::parts::PartEntry;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generated {
    pub path: String,
    pub source: String,
}

pub fn generated(parts: &[&'static PartEntry]) -> Result<Vec<Generated>, Error> {
    let resolves = resolves(parts.iter().filter_map(|part| part.decl.resolve));
    Ok(composites(&resolves)
        .into_iter()
        .map(|(path, _, source)| Generated { path, source })
        .collect())
}
