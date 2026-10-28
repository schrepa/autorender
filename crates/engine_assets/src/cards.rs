use std::collections::BTreeMap;

use crate::{AssetError, MAX_TEXTURE_DIMENSION, Mesh, MeshVertex};

#[derive(Debug, Clone, Copy)]
pub struct AlphaImage<'a> {
    pub width: u32,
    pub height: u32,
    pub rgba: &'a [u8],
}

const CORNERS: usize = 8;
const GROWTH: f64 = 1.5;
const SAVING: f64 = 0.1;
const SNAP: f32 = 1.0 / 64.0;
const SLACK: f64 = 1.0e-6;

pub fn trim_cards(mesh: &Mesh, alpha: AlphaImage<'_>, cutoff: f32) -> Result<Mesh, AssetError> {
    let coverage = Coverage::new(alpha, cutoff)?;
    if mesh.lods().is_some() {
        return Err(AssetError::InvalidLevels {
            reason: "cards are trimmed before their mesh's levels of detail are made",
        });
    }
    if mesh.virtual_geometry().is_some() {
        return Err(AssetError::InvalidVirtualMesh {
            reason: "cards are trimmed before their mesh's virtual geometry is made",
        });
    }
    let cards = find_cards(mesh, coverage.snap);
    let mut fitted = BTreeMap::new();
    let mut fits = Vec::new();
    let card_fits: Vec<usize> = cards
        .iter()
        .map(|card| {
            *fitted
                .entry(card.rect.map(f32::to_bits))
                .or_insert_with(|| {
                    fits.push(coverage.fit(card.rect));
                    fits.len() - 1
                })
        })
        .collect();
    let mut roles = vec![Role::Kept; mesh.indices().len() / 3];
    let mut trimmed = Trimmed::new(mesh);
    for (card, &fit) in cards.iter().zip(&card_fits) {
        let [first, second] = card.triangles;
        match &fits[fit] {
            Fit::Keep => continue,
            Fit::Drop => roles[first] = Role::Gone,
            Fit::Trim(polygon) => roles[first] = Role::Polygon(card, polygon),
        }
        roles[second] = Role::Gone;
        trimmed.replace(card.corners);
    }
    for (triangle, role) in roles.into_iter().enumerate() {
        match role {
            Role::Kept => trimmed.copy(triangle),
            Role::Gone => {}
            Role::Polygon(card, polygon) => trimmed.polygon(card, polygon),
        }
    }
    trimmed.finish()
}

#[derive(Clone, Copy)]
enum Role<'a> {
    Kept,
    Gone,
    Polygon(&'a Card, &'a [[f64; 2]]),
}

enum Fit {
    Keep,
    Drop,
    Trim(Vec<[f64; 2]>),
}

struct Card {
    triangles: [usize; 2],
    corners: [u32; 4],
    rect: [f32; 4],
    clockwise: bool,
}

#[derive(Clone, Copy)]
struct Half {
    corner: u32,
    along_v: u32,
    along_u: u32,
    diagonal: [u32; 2],
}

impl Half {
    fn of(triangle: [u32; 3], vertices: &[MeshVertex], snap: [f32; 2]) -> Option<Self> {
        let uv = |vertex: u32| vertices[vertex as usize].uv;
        let near = |a: f32, b: f32, axis: usize| (a - b).abs() <= snap[axis];
        (0..3).find_map(|at| {
            let [corner, next, last] = [0, 1, 2].map(|step| triangle[(at + step) % 3]);
            let [c, n, l] = [corner, next, last].map(uv);
            let [along_v, along_u] = if near(n[0], c[0], 0) && near(l[1], c[1], 1) {
                [next, last]
            } else if near(n[1], c[1], 1) && near(l[0], c[0], 0) {
                [last, next]
            } else {
                return None;
            };
            let [v_end, u_end] = [along_v, along_u].map(uv);
            (!near(v_end[1], c[1], 1) && !near(u_end[0], c[0], 0)).then_some(Self {
                corner,
                along_v,
                along_u,
                diagonal: [next, last],
            })
        })
    }

    fn completes(self, vertex: u32, vertices: &[MeshVertex], snap: [f32; 2]) -> bool {
        let uv = |vertex: u32| vertices[vertex as usize].uv;
        let [fourth, u_end, v_end] = [vertex, self.along_u, self.along_v].map(uv);
        (fourth[0] - u_end[0]).abs() <= snap[0] && (fourth[1] - v_end[1]).abs() <= snap[1]
    }

    fn card(self, fourth: u32, triangles: [usize; 2], vertices: &[MeshVertex]) -> Card {
        let uv = |vertex: u32| vertices[vertex as usize].uv;
        let [right, u_end, v_end] = [self.corner, self.along_u, self.along_v].map(uv);
        let [s, t] = [right[0] > u_end[0], right[1] > v_end[1]].map(usize::from);
        let mut corners = [0; 4];
        corners[s + 2 * t] = self.corner;
        corners[s + 2 * (1 - t)] = self.along_v;
        corners[1 - s + 2 * t] = self.along_u;
        corners[1 - s + 2 * (1 - t)] = fourth;
        let [c00, c10, c01, c11] = corners.map(uv);
        let rect = [
            engine_math::min(c00[0], c01[0]),
            engine_math::max(c10[0], c11[0]),
            engine_math::min(c00[1], c10[1]),
            engine_math::max(c01[1], c11[1]),
        ];
        let exact = |vertex: u32| uv(vertex).map(f64::from);
        let [next, last] = self.diagonal;
        let [first, second] = triangles;
        Card {
            triangles: [first.min(second), first.max(second)],
            corners,
            rect,
            clockwise: turn(exact(self.corner), exact(next), exact(last)) < 0.0,
        }
    }
}

fn find_cards(mesh: &Mesh, snap: [f32; 2]) -> Vec<Card> {
    let (vertices, indices) = (mesh.vertices(), mesh.indices());
    let triangle = |number: usize| -> [u32; 3] {
        [
            indices[number * 3],
            indices[number * 3 + 1],
            indices[number * 3 + 2],
        ]
    };
    let count = indices.len() / 3;
    let mut edges: Vec<([u32; 2], usize)> = (0..count)
        .flat_map(|number| {
            let [a, b, c] = triangle(number);
            [([a, b], number), ([b, c], number), ([c, a], number)]
        })
        .collect();
    edges.sort_unstable();
    let mut paired = vec![false; count];
    let mut cards = Vec::new();
    for first in 0..count {
        if paired[first] {
            continue;
        }
        let Some(half) = Half::of(triangle(first), vertices, snap) else {
            continue;
        };
        let [from, to] = half.diagonal;
        let start = edges.partition_point(|&(edge, _)| edge < [to, from]);
        let partner = edges[start..]
            .iter()
            .take_while(|&&(edge, _)| edge == [to, from])
            .find_map(|&(_, second)| {
                let fourth = across(triangle(second), [to, from])?;
                (second != first && !paired[second] && half.completes(fourth, vertices, snap))
                    .then_some((second, fourth))
            });
        if let Some((second, fourth)) = partner {
            paired[first] = true;
            paired[second] = true;
            cards.push(half.card(fourth, [first, second], vertices));
        }
    }
    cards
}

fn across(triangle: [u32; 3], edge: [u32; 2]) -> Option<u32> {
    (0..3)
        .find(|&at| [triangle[at], triangle[(at + 1) % 3]] == edge)
        .map(|at| triangle[(at + 2) % 3])
}

struct Coverage<'a> {
    width: u32,
    height: u32,
    rgba: &'a [u8],
    keeps: [bool; 256],
    snap: [f32; 2],
}

impl<'a> Coverage<'a> {
    fn new(image: AlphaImage<'a>, cutoff: f32) -> Result<Self, AssetError> {
        let AlphaImage {
            width,
            height,
            rgba,
        } = image;
        let sides = 1..=MAX_TEXTURE_DIMENSION;
        if !sides.contains(&width) || !sides.contains(&height) {
            return Err(AssetError::InvalidTextureDimensions {
                width,
                height,
                limit: MAX_TEXTURE_DIMENSION,
            });
        }
        let expected = width as usize * height as usize * 4;
        if rgba.len() != expected {
            return Err(AssetError::InvalidTextureLength {
                expected,
                actual: rgba.len(),
            });
        }
        if !(0.0..=1.0).contains(&cutoff) {
            return Err(AssetError::InvalidMaterialMap {
                field: "alpha_cutoff",
            });
        }
        Ok(Self {
            width,
            height,
            rgba,
            keeps: std::array::from_fn(|alpha| alpha as f32 / 255.0 >= cutoff),
            snap: [SNAP / width as f32, SNAP / height as f32],
        })
    }

    fn fit(&self, rect: [f32; 4]) -> Fit {
        let [u0, u1, v0, v1] = rect;
        let [snap_u, snap_v] = self.snap;
        if u0 < -snap_u || u1 > 1.0 + snap_u || v0 < -snap_v || v1 > 1.0 + snap_v {
            return Fit::Keep;
        }
        let [width, height] = [self.width, self.height].map(f64::from);
        let bounds = [
            f64::from(u0) * width,
            f64::from(u1) * width,
            f64::from(v0) * height,
            f64::from(v1) * height,
        ];
        let Some(points) = self.outline(bounds) else {
            return Fit::Drop;
        };
        let mut polygon = hull(points);
        if polygon.len() < 3 || !reduce(&mut polygon, bounds) {
            return Fit::Keep;
        }
        let [x0, x1, y0, y1] = bounds;
        let [wide, tall] = [x1 - x0, y1 - y0];
        if area(&polygon) > (1.0 - SAVING) * wide * tall {
            return Fit::Keep;
        }
        Fit::Trim(
            polygon
                .into_iter()
                .map(|[x, y]| {
                    [
                        ((x - x0) / wide).clamp(0.0, 1.0),
                        ((y - y0) / tall).clamp(0.0, 1.0),
                    ]
                })
                .collect(),
        )
    }

    fn outline(&self, bounds: [f64; 4]) -> Option<Vec<[f64; 2]>> {
        let [x0, x1, y0, y1] = bounds;
        let columns = reach(x0, x1, self.width)?;
        let rows = reach(y0, y1, self.height)?;
        let mut points = Vec::new();
        for row in rows[0]..=rows[1] {
            let Some(run) = self.run(row, columns) else {
                continue;
            };
            let [top, bottom] = grown([row, row], y0, y1);
            let [left, right] = grown(run, x0, x1);
            points.extend([[left, top], [right, top], [right, bottom], [left, bottom]]);
        }
        (!points.is_empty()).then_some(points)
    }

    fn run(&self, row: u32, columns: [u32; 2]) -> Option<[u32; 2]> {
        let start = row as usize * self.width as usize;
        let kept =
            |column: &u32| self.keeps[usize::from(self.rgba[(start + *column as usize) * 4 + 3])];
        let first = (columns[0]..=columns[1]).find(kept)?;
        let last = (first..=columns[1]).rev().find(kept)?;
        Some([first, last])
    }
}

fn reach(low: f64, high: f64, size: u32) -> Option<[u32; 2]> {
    let first = ((low - 1.0 - GROWTH).floor() + 1.0).max(0.0);
    let last = ((high + GROWTH).ceil() - 1.0).min(f64::from(size) - 1.0);
    (first <= last).then_some([first as u32, last as u32])
}

fn grown(run: [u32; 2], low: f64, high: f64) -> [f64; 2] {
    [
        (f64::from(run[0]) - GROWTH).max(low),
        (f64::from(run[1]) + 1.0 + GROWTH).min(high),
    ]
}

fn hull(mut points: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    points.dedup();
    if points.len() < 3 {
        return points;
    }
    let mut around = chain(points.iter().copied());
    around.extend(chain(points.iter().rev().copied()));
    around
}

fn chain(points: impl Iterator<Item = [f64; 2]>) -> Vec<[f64; 2]> {
    let mut side: Vec<[f64; 2]> = Vec::new();
    for point in points {
        while let [.., a, b] = side[..] {
            if turn(a, b, point) > 0.0 {
                break;
            }
            side.pop();
        }
        side.push(point);
    }
    side.pop();
    side
}

fn turn(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn area(polygon: &[[f64; 2]]) -> f64 {
    let first = polygon[0];
    let twice: f64 = polygon
        .windows(2)
        .map(|pair| turn(first, pair[0], pair[1]))
        .sum();
    0.5 * twice
}

fn reduce(polygon: &mut Vec<[f64; 2]>, bounds: [f64; 4]) -> bool {
    while polygon.len() > CORNERS {
        let count = polygon.len();
        let mut best: Option<(f64, usize, [f64; 2])> = None;
        for edge in 0..count {
            let at = |step: usize| polygon[(edge + step) % count];
            let Some((added, point)) = meeting([at(count - 1), at(0), at(1), at(2)], bounds) else {
                continue;
            };
            if best.is_none_or(|(least, _, _)| added < least) {
                best = Some((added, edge, point));
            }
        }
        let Some((_, edge, point)) = best else {
            return false;
        };
        polygon[edge] = point;
        polygon.remove((edge + 1) % count);
    }
    true
}

fn meeting(
    [before, start, end, after]: [[f64; 2]; 4],
    bounds: [f64; 4],
) -> Option<(f64, [f64; 2])> {
    let cross = |a: [f64; 2], b: [f64; 2]| a[0] * b[1] - a[1] * b[0];
    let incoming = [start[0] - before[0], start[1] - before[1]];
    let edge = [end[0] - start[0], end[1] - start[1]];
    let outgoing = [after[0] - end[0], after[1] - end[1]];
    let converging = cross(incoming, outgoing);
    let (at_start, at_end) = (cross(incoming, edge), cross(edge, outgoing));
    if converging <= 0.0 || at_start < 0.0 || at_end < 0.0 {
        return None;
    }
    let along = at_end / converging;
    let point = [
        start[0] + incoming[0] * along,
        start[1] + incoming[1] * along,
    ];
    let [x0, x1, y0, y1] = bounds;
    let within = |value: f64, low: f64, high: f64| (low - SLACK..=high + SLACK).contains(&value);
    if !(within(point[0], x0, x1) && within(point[1], y0, y1)) {
        return None;
    }
    let added = 0.5 * along * at_start;
    Some((added, [point[0].clamp(x0, x1), point[1].clamp(y0, y1)]))
}

struct Trimmed<'a> {
    mesh: &'a Mesh,
    indices: Vec<usize>,
    added: Vec<Added>,
    places: BTreeMap<([u32; 4], [u64; 2]), usize>,
    replaced: Vec<bool>,
}

impl<'a> Trimmed<'a> {
    fn new(mesh: &'a Mesh) -> Self {
        Self {
            mesh,
            indices: Vec::with_capacity(mesh.indices().len()),
            added: Vec::new(),
            places: BTreeMap::new(),
            replaced: vec![false; mesh.vertices().len()],
        }
    }

    fn copy(&mut self, triangle: usize) {
        let mesh = self.mesh;
        let corners = &mesh.indices()[triangle * 3..triangle * 3 + 3];
        self.indices
            .extend(corners.iter().map(|&index| index as usize));
    }

    fn replace(&mut self, corners: [u32; 4]) {
        for corner in corners {
            self.replaced[corner as usize] = true;
        }
    }

    fn polygon(&mut self, card: &Card, polygon: &[[f64; 2]]) {
        let corners: Vec<usize> = polygon.iter().map(|&at| self.vertex(card, at)).collect();
        for pair in corners[1..].windows(2) {
            let [b, c] = if card.clockwise {
                [pair[1], pair[0]]
            } else {
                [pair[0], pair[1]]
            };
            self.indices.extend([corners[0], b, c]);
        }
    }

    fn vertex(&mut self, card: &Card, at: [f64; 2]) -> usize {
        let [s, t] = at;
        if (s == 0.0 || s == 1.0) && (t == 0.0 || t == 1.0) {
            return card.corners[usize::from(s == 1.0) + 2 * usize::from(t == 1.0)] as usize;
        }
        let next = self.mesh.vertices().len() + self.added.len();
        *self
            .places
            .entry((card.corners, at.map(f64::to_bits)))
            .or_insert_with(|| {
                self.added.push(Added {
                    corners: card.corners,
                    at,
                });
                next
            })
    }

    fn finish(self) -> Result<Mesh, AssetError> {
        let mesh = self.mesh;
        let own = mesh.vertices().len();
        let mut used = vec![false; own];
        for &index in &self.indices {
            if index < own {
                used[index] = true;
            }
        }
        let kept: Vec<usize> = (0..own)
            .filter(|&vertex| used[vertex] || !self.replaced[vertex])
            .collect();
        let mut moved = vec![0; own];
        for (place, &vertex) in kept.iter().enumerate() {
            moved[vertex] = place;
        }
        let indices = self
            .indices
            .iter()
            .map(|&index| {
                let index = if index < own {
                    moved[index]
                } else {
                    kept.len() + index - own
                };
                u32::try_from(index).map_err(|_| AssetError::MeshCapacity {
                    limit: u32::MAX as usize,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if indices.is_empty() {
            return Err(AssetError::EmptyMesh);
        }
        let (vertices, added) = (mesh.vertices(), &self.added);
        let blended = added.iter().map(|vertex| MeshVertex {
            position: vertex.blend(|corner| vertices[corner].position),
            normal: vertex.blend(|corner| vertices[corner].normal),
            uv: vertex.blend(|corner| vertices[corner].uv),
        });
        let mut trimmed = Mesh::new(joined(vertices, &kept, blended), indices)?;
        if let Some(colors) = mesh.colors() {
            let blended = added.iter().map(|vertex| {
                vertex
                    .blend(|corner| colors[corner])
                    .map(|value| engine_math::clamp(value, 0.0, 1.0))
            });
            trimmed = trimmed.with_colors(joined(colors, &kept, blended))?;
        }
        if let Some(uv1) = mesh.uv1() {
            let blended = added
                .iter()
                .map(|vertex| vertex.blend(|corner| uv1[corner]));
            trimmed = trimmed.with_uv1(joined(uv1, &kept, blended))?;
        }
        if let (Some(joints), Some(weights)) = (mesh.joints(), mesh.weights()) {
            let (blended_joints, blended_weights): (Vec<_>, Vec<_>) = added
                .iter()
                .map(|vertex| vertex.skin(joints, weights))
                .unzip();
            trimmed = trimmed.with_skin(
                joined(joints, &kept, blended_joints),
                joined(weights, &kept, blended_weights),
            )?;
        }
        Ok(trimmed)
    }
}

fn joined<T: Copy>(values: &[T], kept: &[usize], added: impl IntoIterator<Item = T>) -> Vec<T> {
    kept.iter()
        .map(|&vertex| values[vertex])
        .chain(added)
        .collect()
}

#[derive(Clone, Copy)]
struct Added {
    corners: [u32; 4],
    at: [f64; 2],
}

impl Added {
    fn shares(self) -> [f64; 4] {
        let [s, t] = self.at;
        [(1.0 - s) * (1.0 - t), s * (1.0 - t), (1.0 - s) * t, s * t]
    }

    fn blend<const N: usize>(self, of: impl Fn(usize) -> [f32; N]) -> [f32; N] {
        let mut blended = [0.0_f64; N];
        for (corner, share) in self.corners.into_iter().zip(self.shares()) {
            for (sum, value) in blended.iter_mut().zip(of(corner as usize)) {
                *sum += share * f64::from(value);
            }
        }
        blended.map(|sum| sum as f32)
    }

    fn skin(self, joints: &[[u16; 4]], weights: &[[f32; 4]]) -> ([u16; 4], [f32; 4]) {
        let mut sums: Vec<(u16, f64)> = Vec::with_capacity(16);
        for (corner, share) in self.corners.into_iter().zip(self.shares()) {
            let corner = corner as usize;
            for (joint, weight) in joints[corner].into_iter().zip(weights[corner]) {
                let weight = share * f64::from(weight);
                if weight <= 0.0 {
                    continue;
                }
                match sums.iter_mut().find(|(other, _)| *other == joint) {
                    Some((_, sum)) => *sum += weight,
                    None => sums.push((joint, weight)),
                }
            }
        }
        sums.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut skin = ([0; 4], [0.0; 4]);
        for (slot, (joint, weight)) in sums.into_iter().take(4).enumerate() {
            skin.0[slot] = joint;
            skin.1[slot] = weight as f32;
        }
        skin
    }
}
