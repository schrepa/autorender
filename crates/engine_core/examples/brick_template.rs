#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

extern crate engine_ecs as bevy_ecs;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Schedule, Source, settings, stats};

static SOURCE: Source = Source::generated("brick_template", [0; 32], &[]);

settings! {
    #[section("template")]
    pub struct Template {
        pub enabled: bool = true => any,
    }
}

stats! {
    #[section("template_stats")]
    pub struct TemplateStats {
        pub frames: u32,
    }
}

fn enabled(settings: Res<Template>) -> bool {
    settings.enabled
}

fn count_frames(mut stats: ResMut<TemplateStats>) {
    stats.frames = stats.frames.wrapping_add(1);
}

pub struct TemplateBrick;

impl Brick for TemplateBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(enabled)
            .settings::<Template>()
            .stats::<TemplateStats>()
            .systems(Schedule::Presentation, count_frames);
    }
}
