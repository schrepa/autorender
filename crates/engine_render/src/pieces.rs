use std::fmt::Write as _;

use engine_core::{Error, ErrorCode};

pub const PIECES_MODULE: &str = "generated::pieces";
pub const PIECES_BINDING: &str = "pieces";
pub const PIECES_STRUCT: &str = "Pieces";
pub const PIECE_ALIGN: u64 = 16;
pub const PIECES_BYTES: u64 = 16_384;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub brick: &'static str,
    pub prefix: &'static str,
    pub ty: &'static str,
    pub size: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieceAt {
    pub offset: u64,
    pub size: u64,
}

pub fn place(pieces: &[Piece], most: u64) -> Result<(Vec<PieceAt>, u64), Error> {
    let mut placed = Vec::with_capacity(pieces.len());
    let mut end = 0_u64;
    for (index, piece) in pieces.iter().enumerate() {
        if pieces[..index]
            .iter()
            .any(|earlier| earlier.brick == piece.brick || earlier.prefix == piece.prefix)
        {
            return Err(Error::new(
                ErrorCode::NameTwice,
                format!("$.pieces.{}", piece.prefix),
            ));
        }
        if piece.size == 0 || !piece.size.is_multiple_of(PIECE_ALIGN) {
            return Err(Error::new(
                ErrorCode::InvalidValue,
                format!("$.pieces.{}.size", piece.prefix),
            ));
        }
        let offset = end.next_multiple_of(PIECE_ALIGN);
        placed.push(PieceAt {
            offset,
            size: piece.size,
        });
        end = offset + piece.size;
    }
    let total = end.max(PIECE_ALIGN);
    if total > most {
        return Err(Error::new(ErrorCode::PiecesLimit, "$.pieces"));
    }
    Ok((placed, total))
}

pub fn pieces_source(pieces: &[Piece]) -> String {
    let mut source = String::new();
    for piece in pieces {
        let _ = writeln!(
            source,
            "import package::generated::{}::{};",
            piece.prefix, piece.ty
        );
    }
    let _ = writeln!(source, "\nstruct {PIECES_STRUCT} {{");
    if pieces.is_empty() {
        let _ = writeln!(source, "    none: vec4<f32>,");
    }
    for piece in pieces {
        let _ = writeln!(source, "    {}: {},", piece.prefix, piece.ty);
    }
    let _ = writeln!(source, "}}");
    let _ = writeln!(
        source,
        "\n@group(0) @binding(0) var<uniform> {PIECES_BINDING}: {PIECES_STRUCT};"
    );
    source
}

pub fn piece_path(piece: &Piece) -> String {
    format!("{PIECES_MODULE}::{}", piece.prefix)
}

pub fn piece_source(piece: &Piece) -> String {
    format!(
        "import package::{PIECES_MODULE}::{PIECES_BINDING};\n\
         import package::generated::{prefix}::{ty};\n\n\
         fn piece() -> {ty} {{\n    return {PIECES_BINDING}.{prefix};\n}}\n",
        prefix = piece.prefix,
        ty = piece.ty,
    )
}
