#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod highlights;
mod light;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Schedule, Source, gpu_struct};
use engine_places::lighting::Lighting;
use engine_places::sky::SkyLight;

pub use highlights::Highlights;
pub use light::SurfaceLight;

gpu_struct! {
    #[derive(Resource)]
    pub struct LightingPiece {
        pub model: [f32; 4],
        pub highlights: [f32; 4],
    }
}

pub struct LightingModelBrick;

impl Brick for LightingModelBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<SurfaceLight>()
            .settings::<Highlights>()
            .piece::<LightingPiece>()
            .fill::<Lighting>(Some(shaders::lighting_model::faces::lighting::MODULE))
            .systems(Schedule::ViewSync, (publish_lighting, publish_piece));
    }
}

fn publish_lighting(highlights: Res<Highlights>, mut lighting: ResMut<Lighting>) {
    lighting.set_if_neq(Lighting {
        highlights: highlights.enabled,
    });
}

fn publish_piece(
    surface: Res<SurfaceLight>,
    highlights: Res<Highlights>,
    sky: Res<SkyLight>,
    mut piece: ResMut<LightingPiece>,
) {
    piece.set_if_neq(LightingPiece {
        model: [surface.rim, surface.emission, surface.specular, 0.0],
        highlights: sized_highlights(&highlights, &sky),
    });
}

fn sized_highlights(highlights: &Highlights, sky: &SkyLight) -> [f32; 4] {
    let cap = if highlights.capped {
        highlights.cap
    } else {
        0.0
    };
    if !highlights.enabled {
        return if cap > 0.0 {
            [1.0, 0.0, cap, 0.0]
        } else {
            [0.0; 4]
        };
    }
    let (sine, cosine) = engine_math::sin_cos(sky.sun_size.to_radians());
    [cosine, sine, cap, highlights.light_radius]
}
