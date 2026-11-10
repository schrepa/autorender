#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, Expected, FrameTiming, InvalidSetting, RenderScale, Schedule, Source, settings,
    stats,
};

settings! {
    #[section("dynamic_resolution")]
    #[check(sorted_scales)]
    #[switch(off)]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneDynamicResolution")]
    pub struct DynamicResolution {
        pub min_scale: f32 = 0.5 => range(0.25, 1.0),
        pub max_scale: f32 = 1.0 => range(0.25, 1.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub budget_ms: Option<f32> = None => range(1.0, 1000.0),
        pub step: f64 = 1.0 / 32.0 => above_to(0.0, 1.0),
        pub over: f64 = 1.05 => above(0.0),
        pub under: f64 = 0.85 => above(0.0),
        pub aim: f64 = 0.9 => above(0.0),
        pub lower_frames: u32 = 3 => at_least(1),
        pub raise_frames: u32 = 30 => at_least(1),
        pub probe_frames: u32 = 120 => at_least(1),
        pub interval_limit_ms: f64 = 100.0 => above(0.0),
    }
}

fn sorted_scales(resolution: &DynamicResolution) -> Result<(), InvalidSetting> {
    if resolution.min_scale > resolution.max_scale {
        return Err(InvalidSetting::new("min_scale", Expected::Sorted)
            .got(resolution.min_scale)
            .nearest(resolution.max_scale));
    }
    Ok(())
}

stats! {
    #[section("dynamic_resolution_stats")]
    #[derive(Copy)]
    pub struct DynamicResolutionStats {
        pub active: bool,
        pub scale: f32,
        pub budget_ms: f32,
    }
}

#[derive(Resource, Clone, Copy, Debug)]
struct DynamicScale {
    scale: f64,
    over: u32,
    under: u32,
    suspends: u64,
}

impl Default for DynamicScale {
    fn default() -> Self {
        Self {
            scale: 1.0,
            over: 0,
            under: 0,
            suspends: 0,
        }
    }
}

impl DynamicScale {
    fn scale(&self, settings: &DynamicResolution) -> f64 {
        self.scale
            .clamp(f64::from(settings.min_scale), f64::from(settings.max_scale))
    }

    fn reset(&mut self, suspends: u64) {
        *self = Self {
            suspends,
            ..Self::default()
        };
    }

    fn frame(&mut self, settings: &DynamicResolution, ms: f64, budget: f64, timed: bool) {
        let (low, high) = (f64::from(settings.min_scale), f64::from(settings.max_scale));
        let scale = self.scale.clamp(low, high);
        let (under, frames) = if timed {
            (settings.under, settings.raise_frames)
        } else {
            (settings.over, settings.probe_frames)
        };
        if ms > budget * settings.over {
            self.over += 1;
            self.under = 0;
            if self.over >= settings.lower_frames {
                let aimed = scale * (budget * settings.aim / ms).sqrt();
                let stepped =
                    ((aimed / settings.step).floor() * settings.step).min(scale - settings.step);
                self.scale = stepped.clamp(low, high);
                self.over = 0;
            }
        } else if ms < budget * under {
            self.under += 1;
            self.over = 0;
            if self.under >= frames {
                self.scale = (scale + settings.step).min(high);
                self.under = 0;
            }
        } else {
            self.over = 0;
            self.under = 0;
        }
    }
}

fn budget_ms(settings: &DynamicResolution, timing: &FrameTiming) -> Option<f64> {
    settings.budget_ms.map(f64::from).or(timing.refresh_ms)
}

fn adapt_resolution(
    settings: Res<DynamicResolution>,
    timing: Res<FrameTiming>,
    mut dynamic: ResMut<DynamicScale>,
    mut scale: ResMut<RenderScale>,
    mut stats: ResMut<DynamicResolutionStats>,
) {
    let Some(budget) = budget_ms(&settings, &timing) else {
        dynamic.reset(timing.suspends);
        scale.set_if_neq(RenderScale::default());
        stats.set_if_neq(DynamicResolutionStats {
            active: false,
            scale: 1.0,
            budget_ms: 0.0,
        });
        return;
    };
    if timing.is_changed() {
        if dynamic.suspends != timing.suspends {
            dynamic.reset(timing.suspends);
        }
        if timing.gpu_timed {
            if let Some(gpu) = timing.gpu
                && Some(gpu.render_size) == timing.render_size
            {
                dynamic.frame(&settings, gpu.ms, budget, true);
            }
        } else if let Some(interval) = timing
            .interval_ms
            .filter(|&ms| ms > 0.0 && ms < settings.interval_limit_ms)
        {
            dynamic.frame(&settings, interval, budget, false);
        }
    }
    let current = dynamic.scale(&settings);
    scale.set_if_neq(RenderScale(current));
    stats.set_if_neq(DynamicResolutionStats {
        active: true,
        scale: current as f32,
        budget_ms: budget as f32,
    });
}

pub struct DynamicResolutionBrick;

impl Brick for DynamicResolutionBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .switch::<DynamicResolution>()
            .stats::<DynamicResolutionStats>()
            .resource::<DynamicScale>()
            .fill::<RenderScale>(None)
            .systems(Schedule::FrameStart, adapt_resolution);
    }
}
