use engine::prelude::*;
use serde::{Deserialize, Serialize};

/// The world a workload builds from its configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum StressScene {
    /// A grid of towers (the original workload).
    #[default]
    Field,
    /// A procedural city of diverse buildings, traffic and machines.
    City,
}

/// Configuration changes rebuild the owned workload outside the frame loop.
/// Configurations without `scene` build the tower field.
#[derive(Resource, Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct StressConfig {
    pub seed: u32,
    pub renderables: u32,
    pub moving_percent: u32,
    pub hierarchy: bool,
    pub interpolation: bool,
    pub colliders: u32,
    pub sensor_percent: u32,
    pub dense_colliders: bool,
    pub queries_per_tick: u32,
    pub mesh_variety: u32,
    pub textured: bool,
    pub culling: bool,
    pub scene: StressScene,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            seed: 1337,
            renderables: 10_000,
            moving_percent: 25,
            hierarchy: true,
            interpolation: true,
            colliders: 1_000,
            sensor_percent: 10,
            dense_colliders: false,
            queries_per_tick: 32,
            mesh_variety: 8,
            textured: true,
            culling: true,
            scene: StressScene::Field,
        }
    }
}

impl StressConfig {
    pub fn from_json(source: &str) -> Result<Self, String> {
        if source.len() > 4096 {
            return Err("configuration exceeds 4096 bytes".into());
        }
        let config: Self = serde_json::from_str(source).map_err(|error| error.to_string())?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(1..=250_000).contains(&self.renderables) {
            return Err("renderables must be in 1..=250000".into());
        }
        if self.moving_percent > 100 || self.sensor_percent > 100 {
            return Err("moving_percent and sensor_percent must be in 0..=100".into());
        }
        if self.colliders > 10_000 || self.colliders > self.renderables {
            return Err("colliders must be in 0..=min(renderables,10000)".into());
        }
        if self.queries_per_tick > 512 {
            return Err("queries_per_tick must be in 0..=512".into());
        }
        if ![1, 8, 32, 64].contains(&self.mesh_variety) {
            return Err("mesh_variety must be 1, 8, 32 or 64".into());
        }
        Ok(())
    }
}
