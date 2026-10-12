//! The Stress Lab's content as digests: each of its five scenes built as
//! its native host builds it (the dense and highlands scenes from the
//! generated files they read), run for 120 fixed ticks, then digested by
//! the development tools' `digest` query: every component's views and
//! every asset, hashed so neither entity ids nor spawn order count. Equal
//! digests between two builds mean the same assets, the same spawned
//! components and the same transforms.

use std::collections::HashMap;
use std::path::Path;

use engine_dev::{Context, DevTools};
use stress_lab::{StressConfig, StressScene, create_workload_with, native};

use crate::Report;

const TICKS: u32 = 120;

pub fn run(assets: Option<&Path>, report: &mut Report) -> Result<(), String> {
    let tools = DevTools::default();
    for scene in StressScene::ALL {
        let name = format!("content {}", format!("{scene:?}").to_lowercase());
        let files = match (scene, assets) {
            (StressScene::Dense | StressScene::Highlands, Some(folder)) => {
                native::stress_files(folder, scene, false)
                    .map_err(|error| format!("{name}: {}", error.message))?
            }
            (StressScene::Dense | StressScene::Highlands, None) => {
                report.record(name, "not built: no generated files (--assets)");
                continue;
            }
            _ => HashMap::new(),
        };
        let config = StressConfig {
            scene,
            ..StressConfig::default()
        };
        let mut engine =
            create_workload_with(config, &files).map_err(|error| format!("{name}: {error}"))?;
        for _ in 0..TICKS {
            engine
                .step_fixed(1)
                .map_err(|error| format!("{name}: {error}"))?;
        }
        engine.refresh_transforms();
        report.record(name, tools.digest(engine.app.world(), Context::default()));
    }
    Ok(())
}
