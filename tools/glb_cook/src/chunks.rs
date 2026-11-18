use std::collections::BTreeMap;

use engine_assets::MeshVertex;
use serde_json::{Value, json};

use crate::config::ChunkPlan;

pub struct Piece {
    pub cell: (i32, i32),
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub uv1: Option<Vec<[f32; 2]>>,
    pub colors: Option<Vec<[f32; 4]>>,
    origins: Vec<(u32, u32)>,
}

pub struct Source<'a> {
    pub vertices: &'a [MeshVertex],
    pub indices: &'a [u32],
    pub uv1: Option<&'a [[f32; 2]]>,
    pub colors: Option<&'a [[f32; 4]]>,
}

impl Piece {
    fn new(cell: (i32, i32), uv1: bool, colors: bool) -> Self {
        Self {
            cell,
            vertices: Vec::new(),
            indices: Vec::new(),
            uv1: uv1.then(Vec::new),
            colors: colors.then(Vec::new),
            origins: Vec::new(),
        }
    }

    fn push(&mut self, from: &Source<'_>, origin: (u32, u32), at: &mut BTreeMap<(u32, u32), u32>) {
        let index = *at.entry(origin).or_insert_with(|| {
            let vertex = origin.1 as usize;
            self.vertices.push(from.vertices[vertex]);
            if let (Some(out), Some(held)) = (&mut self.uv1, from.uv1) {
                out.push(held[vertex]);
            }
            if let (Some(out), Some(held)) = (&mut self.colors, from.colors) {
                out.push(held[vertex]);
            }
            self.origins.push(origin);
            u32::try_from(self.vertices.len() - 1).expect("chunk_vertices_fit_u32")
        });
        self.indices.push(index);
    }

    fn absorb(&mut self, other: Piece) {
        let mut at: BTreeMap<(u32, u32), u32> = self
            .origins
            .iter()
            .enumerate()
            .map(|(index, origin)| {
                (
                    *origin,
                    u32::try_from(index).expect("chunk_vertices_fit_u32"),
                )
            })
            .collect();
        let held = Source {
            vertices: &other.vertices,
            indices: &other.indices,
            uv1: other.uv1.as_deref(),
            colors: other.colors.as_deref(),
        };
        for &index in held.indices {
            let origin = other.origins[index as usize];
            let index = *at.entry(origin).or_insert_with(|| {
                let vertex = index as usize;
                self.vertices.push(held.vertices[vertex]);
                if let (Some(out), Some(from)) = (&mut self.uv1, held.uv1) {
                    out.push(from[vertex]);
                }
                if let (Some(out), Some(from)) = (&mut self.colors, held.colors) {
                    out.push(from[vertex]);
                }
                self.origins.push(origin);
                u32::try_from(self.vertices.len() - 1).expect("chunk_vertices_fit_u32")
            });
            self.indices.push(index);
        }
    }

    pub fn triangles(&self) -> usize {
        self.indices.len() / 3
    }
}

fn cell_of(plan: &ChunkPlan, corners: [[f32; 3]; 3]) -> (i32, i32) {
    let x = (corners[0][0] + corners[1][0] + corners[2][0]) / 3.0;
    let z = (corners[0][2] + corners[1][2] + corners[2][2]) / 3.0;
    (
        (x / plan.cell_m).floor() as i32,
        (z / plan.cell_m).floor() as i32,
    )
}

pub fn split(plan: &ChunkPlan, sources: &[Source<'_>]) -> (Vec<Piece>, usize) {
    let uv1 = sources.first().is_some_and(|source| source.uv1.is_some());
    let colors = sources
        .first()
        .is_some_and(|source| source.colors.is_some());
    let mut cells: BTreeMap<(i32, i32), Piece> = BTreeMap::new();
    for (part, source) in sources.iter().enumerate() {
        let part = u32::try_from(part).expect("parts_fit_u32");
        let mut at: BTreeMap<(i32, i32), BTreeMap<(u32, u32), u32>> = BTreeMap::new();
        for triangle in source.indices.as_chunks::<3>().0 {
            let corners =
                [0, 1, 2].map(|corner| source.vertices[triangle[corner] as usize].position);
            let cell = cell_of(plan, corners);
            let piece = cells
                .entry(cell)
                .or_insert_with(|| Piece::new(cell, uv1, colors));
            let map = at.entry(cell).or_default();
            for &vertex in triangle {
                piece.push(source, (part, vertex), map);
            }
        }
    }
    let mut pieces: Vec<Piece> = cells.into_values().collect();
    let mut folded = 0;
    while pieces.len() > 1 {
        let Some((small, _)) = pieces
            .iter()
            .enumerate()
            .filter(|(_, piece)| piece.triangles() < plan.fold_triangles)
            .min_by_key(|(at, piece)| (piece.triangles(), *at))
        else {
            break;
        };
        let taken = pieces.remove(small);
        let distance = |piece: &Piece| {
            let dx = i64::from(piece.cell.0 - taken.cell.0);
            let dz = i64::from(piece.cell.1 - taken.cell.1);
            dx * dx + dz * dz
        };
        let near = pieces
            .iter()
            .enumerate()
            .min_by_key(|(at, piece)| (distance(piece), *at))
            .map(|(at, _)| at)
            .expect("another_piece");
        pieces[near].absorb(taken);
        folded += 1;
    }
    (pieces, folded)
}

pub fn duplicated(pieces: &[Piece]) -> usize {
    let all: usize = pieces.iter().map(|piece| piece.origins.len()).sum();
    let mut distinct: Vec<(u32, u32)> = pieces
        .iter()
        .flat_map(|piece| piece.origins.iter().copied())
        .collect();
    distinct.sort_unstable();
    distinct.dedup();
    all - distinct.len()
}

pub fn summary(
    plan: &ChunkPlan,
    before: (usize, usize),
    rows: &[(String, usize, usize)],
    extra: (usize, usize),
) -> Value {
    let (vertices, indices) = before;
    let (duplicated, folded) = extra;
    let largest = rows.iter().max_by_key(|row| (row.1, row.0.clone()));
    let smallest = rows.iter().min_by_key(|row| (row.1, row.0.clone()));
    let row = |held: Option<&(String, usize, usize)>| {
        held.map(|(stem, triangles, vertices)| json!({"mesh": stem, "triangles": triangles, "vertices": vertices}))
    };
    json!({
        "cell_m": plan.cell_m,
        "fold_triangles": plan.fold_triangles,
        "chunks": rows.len(),
        "folded": folded,
        "largest": row(largest),
        "smallest": row(smallest),
        "vertices_before": vertices,
        "vertices_after": rows.iter().map(|row| row.2).sum::<usize>(),
        "border_vertices_duplicated": duplicated,
        "indices_before": indices,
        "indices_after": rows.iter().map(|row| row.1 * 3).sum::<usize>(),
    })
}
