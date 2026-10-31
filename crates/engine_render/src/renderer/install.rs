use engine_core::{BrickRegistry, LinkedStats};

use super::*;
use crate::generated::generated;
use crate::linker::Linker;
use crate::parts::{CORE_BRICK, CULLING_BRICK, core_head, shared_of};
use crate::pieces::{PIECES_MODULE, PieceAt, PieceDecl, pieces_source, place};
use crate::pipeline_layouts::PipelineLayout;
use crate::places::{Answered, LinkedPlace, places_of};
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
        let (linker, linked) = link(bricks, &entries, pieces.source)?;
        let shared = shared_of(&entries)?;
        let layouts = brick_layouts(bricks);
        let device = self.core.device.clone();
        crate::pipelines::link(&device, linker);
        let lists: Vec<crate::pipelines::Layouts> = render_bricks(bricks)
            .flat_map(|(_, render)| render.layouts.iter().copied())
            .collect();
        crate::pipelines::install(&device, &shared, lists);
        self.core.camera.install(&device, &layouts);
        self.core
            .camera
            .place_pieces(&device, pieces.placed.clone(), pieces.size);
        self.core.shared = shared;
        let scope = crate::gpu_calls::validation_scope(&device);
        let installed = self.parts.install(&entries, &self.core, bricks, linked);
        if installed.is_ok() {
            for &(index, at) in &pieces.placed {
                let _owner = crate::memory::owned(self.parts.decl(index).brick);
                self.parts.with_part(index, |part, _| part.place_piece(at));
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

pub fn installed_parts(bricks: &BrickRegistry) -> Vec<&'static PartEntry> {
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
) -> Result<(Linker, LinkedStats), Error> {
    let mut linker = Linker::new(crate::shaders::MODULES);
    for (_, render) in render_bricks(bricks) {
        for modules in &render.modules {
            linker.add_modules(modules)?;
        }
    }
    let places = places_of(bricks)?;
    linker.add_places(&places)?;
    linker.add_generated(PIECES_MODULE.to_owned(), CORE_BRICK, pieces);
    for module in generated(entries)? {
        linker.add_generated(module.path, CORE_BRICK, module.source);
    }
    check_needs(bricks, entries)?;
    if let Some(need) = linker.unanswered() {
        return Err(bricks.face_unanswered(&need.face, need.brick));
    }
    Ok((linker, linked_stats(&places)))
}

fn linked_stats(places: &[LinkedPlace]) -> LinkedStats {
    let mut faces: Vec<(&'static str, &'static str)> = Vec::new();
    let mut nobody: Vec<&'static str> = Vec::new();
    let mut lists: Vec<(&'static str, &'static [&'static str])> = Vec::new();
    for linked in places {
        match &linked.answered {
            Some(Answered::Filler { brick, .. }) => faces.push((linked.place.name, *brick)),
            Some(Answered::Nobody) => nobody.push(linked.place.name),
            Some(Answered::Members(members)) => {
                let bricks: Vec<&'static str> = members.iter().map(|&(brick, _)| brick).collect();
                lists.push((linked.place.name, Box::leak(bricks.into_boxed_slice())));
            }
            None => {}
        }
    }
    faces.sort_unstable();
    nobody.sort_unstable();
    lists.sort_unstable_by_key(|&(place, _)| place);
    LinkedStats {
        faces: Box::leak(faces.into_boxed_slice()),
        nobody: Box::leak(nobody.into_boxed_slice()),
        lists: Box::leak(lists.into_boxed_slice()),
    }
}

fn check_needs(bricks: &BrickRegistry, entries: &[&'static PartEntry]) -> Result<(), Error> {
    for entry in entries {
        for &need in entry.decl.needs {
            if bricks.filled(need).is_none() {
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
