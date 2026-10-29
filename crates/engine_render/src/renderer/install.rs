use engine_core::BrickRegistry;

use super::*;
use crate::linker::Linker;
use crate::parts::{CORE_BRICK, CULLING_BRICK, core_head, shared_of};
use crate::pieces::{PIECES_MODULE, PieceAt, PieceDecl, pieces_source, place};
use crate::pipeline_layouts::PipelineLayout;
use crate::render_bricks::render_bricks;

type Layouts = Vec<(&'static str, &'static PipelineLayout)>;

struct Pieces {
    placed: Vec<(usize, PieceAt)>,
    size: u64,
    source: String,
}

impl Renderer {
    pub fn install(&mut self, bricks: &BrickRegistry) -> Result<(), Error> {
        let entries = installed_parts(bricks);
        let pieces = pieces_of(&entries)?;
        let linker = link(bricks, &entries, pieces.source)?;
        let shared = shared_of(&entries)?;
        let layouts = brick_layouts(bricks);
        let device = self.core.device.clone();
        crate::pipelines::set_linker(&device, linker);
        crate::pipelines::install(&device, &shared, &layouts);
        self.core.camera.install(&device, &layouts);
        self.core
            .camera
            .place_pieces(&device, pieces.placed.clone(), pieces.size);
        self.core.shared = shared;
        let scope = crate::gpu_calls::validation_scope(&device);
        let installed = self.parts.install(&entries, &self.core);
        if installed.is_ok() {
            for &(index, at) in &pieces.placed {
                let _owner = crate::memory::owned(self.parts.decl(index).brick);
                self.parts
                    .with_part(index, |part, _| part.place_piece(at));
            }
            self.core.mesh_entries = self.parts.mesh_entries();
        }
        let core = &mut self.core;
        core.camera
            .refresh(&core.device, &core.shared, &self.parts, None);
        self.installing = Some(Box::pin(scope.pop()));
        self.table = installed?;
        self.graph.clear();
        Ok(())
    }
}

fn installed_parts(bricks: &BrickRegistry) -> Vec<&'static PartEntry> {
    let mut entries = core_head(bricks.has(CULLING_BRICK));
    for (_, render) in render_bricks(bricks) {
        entries.extend(render.parts.iter().copied());
    }
    entries
}

fn pieces_of(entries: &[&'static PartEntry]) -> Result<Pieces, Error> {
    let owned: Vec<(usize, &'static str, PieceDecl)> = entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| Some((index, entry.decl.brick, entry.decl.piece?)))
        .collect();
    let declared: Vec<(&'static str, PieceDecl)> = owned
        .iter()
        .map(|&(_, brick, piece)| (brick, piece))
        .collect();
    let (at, size) = place(&declared)?;
    let placed = owned
        .iter()
        .zip(at)
        .map(|(&(index, _, _), at)| (index, at))
        .collect();
    let decls: Vec<PieceDecl> = declared.iter().map(|&(_, piece)| piece).collect();
    Ok(Pieces {
        placed,
        size,
        source: pieces_source(&decls),
    })
}

fn link(
    bricks: &BrickRegistry,
    entries: &[&'static PartEntry],
    pieces: String,
) -> Result<Linker, Error> {
    let mut linker = Linker::new(crate::shaders::MODULES);
    for (_, render) in render_bricks(bricks) {
        for modules in &render.modules {
            linker.add_modules(modules)?;
        }
    }
    for (_, render) in render_bricks(bricks) {
        for defaults in &render.defaults {
            linker.add_defaults(defaults)?;
        }
    }
    linker.add_generated(PIECES_MODULE.to_owned(), CORE_BRICK, pieces);
    check_needs(bricks, entries, &linker)?;
    linker
        .resolve_faces()
        .map_err(|need| bricks.face_unanswered(&need.face, need.brick))?;
    Ok(linker)
}

fn check_needs(
    bricks: &BrickRegistry,
    entries: &[&'static PartEntry],
    linker: &Linker,
) -> Result<(), Error> {
    let offered = linker.faces_offered();
    for entry in entries {
        for &need in entry.decl.needs {
            let answered = bricks.filler(need).is_some()
                || offered.iter().any(|&(face, _)| face == need)
                || entries.iter().any(|other| {
                    other
                        .decl
                        .faces
                        .iter()
                        .any(|offer| offer.name() == need)
                });
            if !answered {
                return Err(bricks.face_unanswered(need, entry.decl.brick));
            }
        }
    }
    Ok(())
}

fn brick_layouts(bricks: &BrickRegistry) -> Layouts {
    render_bricks(bricks)
        .flat_map(|(_, render)| {
            render
                .layouts
                .iter()
                .flat_map(|layouts| layouts.iter().copied())
        })
        .collect()
}
