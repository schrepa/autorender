use engine_core::ErrorCode;
use engine_registry::Registry;

use super::AppHost;
use crate::{Error, Result};

impl AppHost {
    pub(super) fn claim_passes(&mut self, ahead: Option<engine_render::Ahead>) -> Result<()> {
        let world = self.engine.app.world_mut();
        if !world.contains_resource::<Registry>() {
            return Ok(());
        }
        match ahead {
            Some(ahead) => self.renderer.install_ahead(world, ahead),
            None => self.renderer.install(world),
        }
        .map_err(|mut problems| match problems.len() {
            1 => problems.remove(0),
            _ => Error::new(ErrorCode::RenderFailed, "$.bricks").got(&problems),
        })
    }
}
