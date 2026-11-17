use engine_core::{Error, ErrorCode, Expected, Slot};
use engine_places_simulation::ground::{Ground, GroundPoint, GroundRead};
use glam::Vec2;

pub(crate) fn unplaced(read: GroundRead<Option<GroundPoint>>, position: Vec2) -> Error {
    match read {
        GroundRead::Off => Error::new(ErrorCode::PlaceNotFilled, "$.grounded")
            .expected(Expected::Present)
            .got(Ground::NAME),
        GroundRead::Ready(_) | GroundRead::NotLoaded => {
            Error::new(ErrorCode::InvalidPosition, "$.grounded").got(position.to_array())
        }
    }
}
