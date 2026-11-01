use bevy_ecs::prelude::{Resource, World};
use engine_core::{Error, ErrorCode, Expected, section_in};
use engine_places::Ground;

use super::Water;

#[derive(Resource)]
pub(crate) struct LiveWaterChecked;

pub(crate) fn check_live_water(world: &mut World) {
    if world.contains_resource::<LiveWaterChecked>() {
        return;
    }
    world.insert_resource(LiveWaterChecked);
    let live = section_in::<Water>(world).enabled;
    let heightfield = world
        .get_resource::<Ground>()
        .is_some_and(|ground| ground.face().heightfield());
    if live && !heightfield {
        let error = Error::new(ErrorCode::MissingPlugin, "$.environment.water")
            .expected(Expected::Present)
            .got("heightfield");
        eprintln!(
            "{}",
            serde_json::json!({"event": "live_water_refused", "error": error})
        );
    }
}
