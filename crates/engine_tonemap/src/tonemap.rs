use std::mem::offset_of;

use engine_core::glam::Vec3;
use engine_core::{Grading, GradingValues};
use engine_faces::display::MeteredExposure;
use engine_render::composite::{PixelEffect, PostStage};
use engine_render::parts::{Create, GpuPart, PartDecl, PartEntry, Prelude, PreludeStage, Step};
use engine_render::pieces::{PieceAt, PieceDecl};

use crate::gpu_layouts::DisplayPiece;

pub(crate) const TONEMAP: &str = "face::tonemap";
pub(crate) const PIECE: &str = "tonemap::piece";

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "tonemap",
        brick: "tonemap",
        passes: &[],
        resources: &[],
        hooks: &[],
        contributes: &[],
        effects: &[PixelEffect {
            stage: PostStage::Tonemap,
            module: TONEMAP,
        }],
        labels: &[],
        lazy: false,
        faces: &[],
        needs: &[],
        piece: Some(PieceDecl {
            name: "display",
            module: PIECE,
            ty: "DisplayPiece",
            size: size_of::<DisplayPiece>() as u64,
        }),
    },
    create,
};

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(TonemapPass::default())
}

fn contrast_weight(contrast: f32) -> f32 {
    engine_core::math::clamp(2.0 * (contrast - 1.0), -2.0, 1.0)
}

fn unit_luminance(tint: Vec3) -> [f32; 4] {
    let luminance = tint.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    if luminance > 1.0e-6 {
        (tint / luminance).extend(0.0).to_array()
    } else {
        [1.0, 1.0, 1.0, 0.0]
    }
}

impl DisplayPiece {
    fn new(grading: &GradingValues, metered: bool) -> Self {
        Self {
            grade: [
                contrast_weight(grading.contrast),
                grading.vibrance,
                grading.saturation,
                grading.exposure,
            ],
            shadow_tint: unit_luminance(grading.shadow_tint),
            highlight_tint: unit_luminance(grading.highlight_tint),
            exposure: [0.0, if metered { 1.0 } else { 0.0 }, 0.0, 0.0],
        }
    }
}

#[derive(Default)]
struct TonemapPass {
    piece: DisplayPiece,
    at: PieceAt,
}

impl GpuPart for TonemapPass {
    fn prepare(&mut self, cx: &mut Step<'_>, prelude: &mut Prelude) {
        let metered = cx
            .parts
            .face::<dyn MeteredExposure>()
            .and_then(MeteredExposure::metered);
        let grading = cx.frame.snapshot.section::<Grading>().values();
        self.piece = DisplayPiece::new(&grading, metered.is_some());
        let Some((state, offset)) = metered else {
            return;
        };
        let core = &*cx.core;
        let encoder = prelude.encoder(
            &core.device,
            PreludeStage::Inputs,
            "pre_inputs",
            "engine/exposure-copy",
        );
        encoder.copy_buffer_to_buffer(
            state,
            offset,
            core.pieces_buffer(),
            self.at.offset + offset_of!(DisplayPiece, exposure) as u64,
            4,
        );
    }

    fn place_piece(&mut self, at: PieceAt) {
        self.at = at;
    }

    fn piece(&self) -> Option<&[u8]> {
        Some(bytemuck::bytes_of(&self.piece))
    }
}
