use bevy_ecs::resource::Resource;

use crate::{Error, ErrorCode};

#[derive(Resource, Default, Debug)]
pub struct Closing {
    offered: bool,
    asked: bool,
}

impl Closing {
    pub fn offered(&self) -> bool {
        self.offered
    }

    pub fn ask(&mut self) -> Result<(), Error> {
        if !self.offered {
            return Err(Error::new(ErrorCode::UnsupportedFeature, "$.close")
                .got("this host cannot close itself"));
        }
        self.asked = true;
        Ok(())
    }

    pub fn offer(&mut self) {
        self.offered = true;
    }

    pub fn asked(&self) -> bool {
        self.asked
    }
}
