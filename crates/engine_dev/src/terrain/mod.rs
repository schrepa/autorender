//! Terrain for agents: read-only queries ([`queries`]: heights at points,
//! a sweep comparing what terrain queries, collision rays and generated
//! terrains' layers say the ground is, routes for paths, walks and tracks).

mod queries;

use crate::coordinate;
use crate::features::{feature, missing};

pub(crate) use queries::execute;
