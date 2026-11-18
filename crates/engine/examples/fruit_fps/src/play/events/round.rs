use engine::prelude::*;

use crate::rules::Event;

#[derive(Resource, Default, Clone, Debug)]
pub struct RoundEvents {
    pub tick: u64,
    pub events: Vec<Event>,
}
