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

fn cell_of(plan: &ChunkPlan, origin: [f32; 2], corners: [[f32; 3]; 3]) -> (i32, i32) {
    let x = (corners[0][0] + corners[1][0] + corners[2][0]) / 3.0;
    let z = (corners[0][2] + corners[1][2] + corners[2][2]) / 3.0;
    (
        ((x - origin[0]) / plan.cell_m).floor() as i32,
        ((z - origin[1]) / plan.cell_m).floor() as i32,
    )
}

pub fn split(
    plan: &ChunkPlan,
    origin: [f32; 2],
    sources: &[Source<'_>],
) -> (Vec<Piece>, usize, usize) {
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
            let cell = cell_of(plan, origin, corners);
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
    let under = pieces
        .iter()
        .filter(|piece| piece.triangles() < plan.fold_triangles)
        .count();
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
    (pieces, folded, under)
}

fn root(parent: &mut [usize], mut at: usize) -> usize {
    while parent[at] != at {
        parent[at] = parent[parent[at]];
        at = parent[at];
    }
    at
}

pub struct Sized {
    pub class: usize,
    pub largest_m: f32,
    pub pieces: usize,
    pub piece: Piece,
}

pub fn by_size(piece: Piece, bounds: &[f32]) -> Vec<Sized> {
    let mut welded: BTreeMap<[i64; 3], usize> = BTreeMap::new();
    let weld: Vec<usize> = piece
        .vertices
        .iter()
        .map(|vertex| {
            let key = vertex
                .position
                .map(|axis| (f64::from(axis) * 1000.0).round() as i64);
            let next = welded.len();
            *welded.entry(key).or_insert(next)
        })
        .collect();
    let mut parent: Vec<usize> = (0..welded.len()).collect();
    let triangles = piece.indices.as_chunks::<3>().0;
    for triangle in triangles {
        let first = root(&mut parent, weld[triangle[0] as usize]);
        for &corner in &triangle[1..] {
            let other = root(&mut parent, weld[corner as usize]);
            parent[other] = first;
        }
    }
    let mut boxes: BTreeMap<usize, ([f32; 3], [f32; 3])> = BTreeMap::new();
    let roots: Vec<usize> = triangles
        .iter()
        .map(|triangle| {
            let held = root(&mut parent, weld[triangle[0] as usize]);
            let (low, high) = boxes
                .entry(held)
                .or_insert(([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]));
            for &corner in triangle {
                let position = piece.vertices[corner as usize].position;
                for axis in 0..3 {
                    if position[axis] < low[axis] {
                        low[axis] = position[axis];
                    }
                    if position[axis] > high[axis] {
                        high[axis] = position[axis];
                    }
                }
            }
            held
        })
        .collect();
    let extents: BTreeMap<usize, (usize, f32)> = boxes
        .into_iter()
        .map(|(held, (low, high))| {
            let extent = (0..3)
                .map(|axis| high[axis] - low[axis])
                .fold(0.0, |most, next| if next > most { next } else { most });
            let class = bounds.iter().filter(|bound| **bound <= extent).count();
            (held, (class, extent))
        })
        .collect();
    let mut classes: Vec<Sized> = (0..=bounds.len())
        .map(|class| Sized {
            class,
            largest_m: 0.0,
            pieces: 0,
            piece: Piece::new(piece.cell, piece.uv1.is_some(), piece.colors.is_some()),
        })
        .collect();
    for &(class, extent) in extents.values() {
        let held = &mut classes[class];
        held.pieces += 1;
        if extent > held.largest_m {
            held.largest_m = extent;
        }
    }
    let source = Source {
        vertices: &piece.vertices,
        indices: &piece.indices,
        uv1: piece.uv1.as_deref(),
        colors: piece.colors.as_deref(),
    };
    let mut at: Vec<BTreeMap<(u32, u32), u32>> = vec![BTreeMap::new(); classes.len()];
    for (triangle, held) in triangles.iter().zip(&roots) {
        let class = extents[held].0;
        for &corner in triangle {
            let origin = piece.origins[corner as usize];
            let target = &mut classes[class].piece;
            let index = *at[class].entry(origin).or_insert_with(|| {
                let vertex = corner as usize;
                target.vertices.push(source.vertices[vertex]);
                if let (Some(out), Some(from)) = (&mut target.uv1, source.uv1) {
                    out.push(from[vertex]);
                }
                if let (Some(out), Some(from)) = (&mut target.colors, source.colors) {
                    out.push(from[vertex]);
                }
                target.origins.push(origin);
                u32::try_from(target.vertices.len() - 1).expect("chunk_vertices_fit_u32")
            });
            target.indices.push(index);
        }
    }
    classes.retain(|held| held.pieces != 0);
    classes
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
    extra: (usize, usize, usize),
) -> Value {
    let (vertices, indices) = before;
    let (duplicated, folded, under) = extra;
    let largest = rows.iter().max_by_key(|row| (row.1, row.0.clone()));
    let smallest = rows.iter().min_by_key(|row| (row.1, row.0.clone()));
    let row = |held: Option<&(String, usize, usize)>| {
        held.map(|(stem, triangles, vertices)| json!({"mesh": stem, "triangles": triangles, "vertices": vertices}))
    };
    json!({
        "cell_m": plan.cell_m,
        "fold_triangles": plan.fold_triangles,
        "chunks": rows.len(),
        "cells_under_fold": under,
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
