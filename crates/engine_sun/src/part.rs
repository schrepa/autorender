use engine_faces::sun::{Sun, SunFace};
use engine_render::parts::{Begin, BeginOut, Create, GpuPart, PartDecl, PartEntry};
use engine_render::pieces::PieceDecl;

use crate::gpu_layouts::SunPiece;

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "sun",
        brick: "sun",
        passes: &[],
        resources: &[],
        hooks: &[],
        contributes: &[],
        effects: &[],
        labels: &[],
        lazy: false,
        faces: &[engine_render::offer_face!(SunPart => dyn SunFace)],
        needs: &[],
        piece: Some(PieceDecl {
            name: "sun",
            module: "sun::piece",
            ty: "SunPiece",
            size: size_of::<SunPiece>() as u64,
        }),
    },
    create,
};

struct SunPart {
    piece: SunPiece,
}

fn create(_cx: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(SunPart {
        piece: SunPiece {
            direction: Sun::NONE.direction,
            color: Sun::NONE.color,
        },
    })
}

impl SunFace for SunPart {
    fn sun(&self) -> Sun {
        Sun {
            direction: self.piece.direction,
            color: self.piece.color,
        }
    }
}

impl GpuPart for SunPart {
    fn begin(&mut self, cx: &mut Begin<'_>, _out: &mut BeginOut) {
        let atmosphere = cx.snapshot.environment.atmosphere.sanitized();
        self.piece = SunPiece {
            direction: atmosphere.sun_direction.normalize().extend(0.0).to_array(),
            color: atmosphere.sun_color.extend(0.0).to_array(),
        };
    }

    fn piece(&self) -> Option<&[u8]> {
        Some(bytemuck::bytes_of(&self.piece))
    }
}
