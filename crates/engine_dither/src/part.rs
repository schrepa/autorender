use engine_render::composite::{PixelEffect, PostStage};
use engine_render::parts::{Create, GpuPart, PartDecl, PartEntry, Prelude, Step};
use engine_render::pieces::PieceDecl;

use crate::piece::DitherPiece;

pub static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "dither",
        brick: "dither",
        passes: &[],
        resources: &[],
        hooks: &[],
        contributes: &[],
        effects: &[PixelEffect {
            stage: PostStage::Display,
            module: "dither::apply",
        }],
        labels: &[],
        lazy: false,
        faces: &[],
        needs: &[],
        piece: Some(PieceDecl {
            name: "dither",
            module: "dither::piece",
            ty: "DitherPiece",
            size: size_of::<DitherPiece>() as u64,
        }),
    },
    create,
};

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(DitherPart::default())
}

#[derive(Default)]
struct DitherPart {
    piece: DitherPiece,
}

impl GpuPart for DitherPart {
    fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
        let enabled = cx.frame.snapshot.environment.dither.enabled;
        self.piece.flags = [f32::from(u8::from(enabled)), 0.0, 0.0, 0.0];
    }

    fn piece(&self) -> Option<&[u8]> {
        Some(bytemuck::bytes_of(&self.piece))
    }
}
