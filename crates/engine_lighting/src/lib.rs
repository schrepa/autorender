#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod highlights;
mod light;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, MAPPED, Module, Rule, Schedule, Source, gpu_struct};
use engine_places::surface::{
    METALLIC_ROUGHNESS_MAP, METALNESS, OCCLUSION, OCCLUSION_MAP, OCCLUSION_STRENGTH, ROUGHNESS,
    SurfaceValues, TRANSMISSION,
};
use engine_places_rendering::lighting::Lighting;
use engine_places_rendering::sky::SkyLight;
use engine_render::{
    Lit, MaterialValue, OpenSlot, RenderBricks, SlotModule, SurfaceKind, SurfaceTerms,
};

pub use highlights::Highlights;
pub use light::SurfaceLight;

gpu_struct! {
    #[derive(Resource)]
    pub struct LightingPiece {
        pub model: [f32; 4],
        pub highlights: [f32; 4],
        pub widening: [f32; 4],
    }
}

const UNIT: Rule = Rule::range(0.0, 1.0);

const OPEN_SLOTS: [(&str, &str, &[MaterialValue]); 4] = [
    (ROUGHNESS, "1.0", &[]),
    (METALNESS, "0.0", &[]),
    (
        OCCLUSION,
        "1.0",
        &[MaterialValue {
            name: OCCLUSION_STRENGTH,
            standard: SurfaceValues::MATTE.occlusion_strength,
            rule: UNIT,
        }],
    ),
    (
        TRANSMISSION,
        "0.0",
        &[MaterialValue {
            name: TRANSMISSION,
            standard: SurfaceValues::MATTE.transmission,
            rule: UNIT,
        }],
    ),
];

const TERMS: SurfaceTerms = SurfaceTerms {
    fields: &[
        ("alpha", "f32"),
        ("f0", "vec3<f32>"),
        ("diffuse_color", "vec3<f32>"),
    ],
    derive: shaders::lighting_model::terms::MODULE,
};

const MAPPED_SLOTS: [(&str, &[&str]); 3] = [
    (ROUGHNESS, &[METALLIC_ROUGHNESS_MAP]),
    (METALNESS, &[METALLIC_ROUGHNESS_MAP]),
    (OCCLUSION, &[OCCLUSION_MAP]),
];

const STANDARD: Module = shaders::lighting_model::slots::standard::MODULE;

pub struct LightingModelBrick;

impl Brick for LightingModelBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<SurfaceLight>()
            .settings::<Highlights>()
            .within(
                |highlights: Res<Highlights>| highlights.enabled,
                |bricks| {
                    bricks.resource::<SizedHighlights>();
                },
            )
            .piece::<LightingPiece>()
            .surface_kind(SurfaceKind { name: "thin" })
            .fill::<Lighting>(Some(shaders::lighting_model::faces::lighting::MODULE))
            .fill::<Lit>(Some(shaders::lighting_model::faces::lighting::MODULE))
            .systems(Schedule::ViewSync, (publish_lighting, publish_piece));
        for (name, flat, values) in OPEN_SLOTS {
            bricks.open_slot(OpenSlot {
                name,
                returns: "f32",
                standard: STANDARD,
                flat,
                values,
            });
        }
        bricks.surface_terms(TERMS);
        for (slot, maps) in MAPPED_SLOTS {
            bricks.material_slot(SlotModule {
                slot,
                name: MAPPED,
                module: shaders::lighting_model::slots::mapped::MODULE,
                maps,
                values: &[],
                reach: 0.0,
            });
        }
    }
}

#[derive(Resource, Default)]
struct SizedHighlights;

fn publish_lighting(sized: Option<Res<SizedHighlights>>, mut lighting: ResMut<Lighting>) {
    lighting.set_if_neq(Lighting {
        highlights: sized.is_some(),
    });
}

fn publish_piece(
    (surface, highlights, sized): (
        Res<SurfaceLight>,
        Res<Highlights>,
        Option<Res<SizedHighlights>>,
    ),
    sky: Res<SkyLight>,
    mut piece: ResMut<LightingPiece>,
) {
    piece.set_if_neq(LightingPiece {
        model: [surface.rim, surface.emission, surface.specular, 0.0],
        highlights: sized_highlights(&highlights, &sky, sized.is_some()),
        widening: [surface.specular_variance, surface.specular_limit, 0.0, 0.0],
    });
}

fn sized_highlights(highlights: &Highlights, sky: &SkyLight, sized: bool) -> [f32; 4] {
    let cap = if highlights.capped {
        highlights.cap
    } else {
        0.0
    };
    if !sized {
        return if cap > 0.0 {
            [1.0, 0.0, cap, 0.0]
        } else {
            [0.0; 4]
        };
    }
    let (sine, cosine) = engine_math::sin_cos(sky.sun_size.to_radians());
    [cosine, sine, cap, highlights.light_radius]
}
