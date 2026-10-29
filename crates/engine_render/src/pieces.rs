use std::fmt::Write as _;

use engine_core::{Error, ErrorCode, Expected};

pub const PIECES_MODULE: &str = "generated::pieces";
pub const PIECES_BINDING: &str = "pieces";
pub const PIECES_STRUCT: &str = "Pieces";
pub const PIECE_ALIGN: u64 = 16;
pub const PIECES_MOST: u64 = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PieceDecl {
    pub name: &'static str,
    pub module: &'static str,
    pub ty: &'static str,
    pub size: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieceAt {
    pub offset: u64,
    pub size: u64,
}

pub fn place(pieces: &[(&'static str, PieceDecl)]) -> Result<(Vec<PieceAt>, u64), Error> {
    let mut placed = Vec::with_capacity(pieces.len());
    let mut end = 0_u64;
    for (index, &(brick, piece)) in pieces.iter().enumerate() {
        if let Some(&(first, _)) = pieces[..index]
            .iter()
            .find(|(_, earlier)| earlier.name == piece.name)
        {
            return Err(Error::new(
                ErrorCode::DuplicateBrick,
                format!("$.pieces.{}", piece.name),
            )
            .expected(Expected::Unique)
            .got(brick)
            .nearest(first));
        }
        if piece.size == 0 || !piece.size.is_multiple_of(PIECE_ALIGN) {
            return Err(Error::new(
                ErrorCode::InvalidArguments,
                format!("$.pieces.{}.size", piece.name),
            )
            .expected(Expected::Multiple)
            .got(piece.size));
        }
        let offset = end.next_multiple_of(PIECE_ALIGN);
        placed.push(PieceAt {
            offset,
            size: piece.size,
        });
        end = offset + piece.size;
    }
    let total = end.max(PIECE_ALIGN);
    if total > PIECES_MOST {
        return Err(Error::new(ErrorCode::BindingLimit, "$.pieces")
            .at_most(PIECES_MOST)
            .got(total));
    }
    Ok((placed, total))
}

pub fn pieces_source(pieces: &[PieceDecl]) -> String {
    let mut source = String::new();
    for piece in pieces {
        let _ = writeln!(source, "import package::{}::{};", piece.module, piece.ty);
    }
    let _ = writeln!(source, "\nstruct {PIECES_STRUCT} {{");
    if pieces.is_empty() {
        let _ = writeln!(source, "    none: vec4<f32>,");
    }
    for piece in pieces {
        let _ = writeln!(source, "    {}: {},", piece.name, piece.ty);
    }
    let _ = writeln!(source, "}}");
    let _ = writeln!(
        source,
        "\n@group(0) @binding(0) var<uniform> {PIECES_BINDING}: {PIECES_STRUCT};"
    );
    source
}
