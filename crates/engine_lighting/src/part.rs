use engine_core::{Highlights, Sky, SurfaceLight};
use engine_render::parts::{Begin, BeginOut, Create, GpuPart, PartDecl, PartEntry};
use engine_render::pieces::PieceDecl;

use crate::gpu_layouts::LightingPiece;

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "lighting",
        brick: "lighting_model",
        passes: &[],
        resources: &[],
        hooks: &[],
        contributes: &[],
        resolve: None,
        labels: &[],
        lazy: false,
        needs: &[],
        piece: Some(PieceDecl {
            name: "lighting",
            module: "lighting::piece",
            ty: "LightingPiece",
            size: size_of::<LightingPiece>() as u64,
        }),
    },
    create,
};

struct LightingPart {
    piece: LightingPiece,
}

fn create(_cx: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(LightingPart {
        piece: LightingPiece {
            model: [0.0; 4],
            highlights: [0.0; 4],
        },
    })
}

impl GpuPart for LightingPart {
    fn begin(&mut self, cx: &mut Begin<'_>, _out: &mut BeginOut) {
        let snapshot = cx.snapshot;
        let surface = snapshot.section::<SurfaceLight>().values();
        let sky = snapshot.section::<Sky>();
        let highlights = snapshot.section::<Highlights>();
        self.piece = LightingPiece {
            model: [surface.rim, surface.emission, surface.specular, 0.0],
            highlights: sized_highlights(&highlights, &sky),
        };
    }

    fn piece(&self) -> Option<&[u8]> {
        Some(bytemuck::bytes_of(&self.piece))
    }
}

fn sized_highlights(highlights: &engine_core::Highlights, sky: &engine_core::Sky) -> [f32; 4] {
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
    let radius = if sky.enabled {
        sky.sun_size
    } else {
        highlights.sun_size
    };
    let (sine, cosine) = engine_core::math::sin_cos(radius.to_radians());
    [cosine, sine, cap, highlights.light_radius]
}
