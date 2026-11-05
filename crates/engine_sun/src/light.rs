use bevy_ecs::world::World;
use engine_core::{Error, ErrorCode, Expected, InvalidSetting, members_in, settings};
use engine_places::sun::SunClocks;
use glam::Vec3;

const UNAUTHORED: Vec3 = Vec3::new(0.55, 0.75, 0.35);

settings! {
    #[section("sun")]
    #[check(check_direction)]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneSun")]
    pub struct Sunlight {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub sun_direction: Option<[f32; 3]> = None => finite,
    }
}

fn check_direction(sun: &Sunlight) -> Result<(), InvalidSetting> {
    match sun.sun_direction {
        Some(direction) if Vec3::from_array(direction).length_squared() == 0.0 => {
            Err(InvalidSetting::new("sun_direction", Expected::Nonzero).got(direction))
        }
        Some(_) | None => Ok(()),
    }
}

impl Sunlight {
    pub(crate) fn direction(&self) -> Vec3 {
        self.sun_direction.map_or(UNAUTHORED, Vec3::from_array)
    }
}

pub(crate) fn direction_without_clock(world: &World) -> Result<(), Error> {
    let authored = world
        .get_resource::<Sunlight>()
        .is_some_and(|sun| sun.sun_direction.is_some());
    let clocks = members_in::<SunClocks>(world);
    if authored && clocks.iter().any(|runs| runs(world)) {
        return Err(
            Error::new(ErrorCode::InvalidEnvironment, "$.sun.sun_direction")
                .expected(Expected::Absent),
        );
    }
    Ok(())
}
