use bevy_ecs::resource::Resource;
use glam::Vec3;

use crate::{Cursor, DrawnSettings, Feed, Since};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DrawnArea {
    Box { min: Vec3, max: Vec3, spacing: f32 },
    All,
}

#[derive(Resource, Clone, Debug)]
pub struct DrawnAreas {
    feed: Feed<DrawnArea>,
}

impl Default for DrawnAreas {
    fn default() -> Self {
        Self {
            feed: Feed::new(DrawnSettings::default().changes as usize),
        }
    }
}

impl DrawnAreas {
    pub fn push(&mut self, area: DrawnArea) {
        self.feed.push(area);
    }

    pub fn resize(&mut self, size: usize) {
        self.feed.resize(size);
    }

    pub fn since<'a>(&'a self, cursor: &mut Cursor) -> Since<'a, DrawnArea> {
        self.feed.since(cursor)
    }
}
