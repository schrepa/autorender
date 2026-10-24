//! Cards trimmed to their alpha: each quad of a masked card (leaves,
//! needles, grass clumps, fences) cut to a convex polygon around the texels
//! its texture keeps, so alpha testing shades fewer of the pixels it
//! discards.
//!
//! A card is two triangles sharing an edge, the diagonal of the
//! axis-aligned rectangle their four vertices' texture coordinates make; a
//! card bent into several quads is several cards, each covering part of its
//! rectangle. A card's polygon holds the texels in and beside its rectangle
//! whose alpha reaches the cutoff, each grown by a texel and a half on every
//! side so filtered mip levels and hashed cutoffs never cut into them: their
//! convex hull within the rectangle, which then gives up corners until it
//! has at most eight, each time trading the edge whose neighbours, extended
//! until they meet within the rectangle, add the least area for the point
//! where they meet. A card where no edge can give way keeps its quad.

use std::collections::BTreeMap;

use crate::{AssetError, MAX_TEXTURE_DIMENSION, Mesh, MeshVertex};

/// RGBA8 pixels, `width` x `height`, rows top to bottom.
#[derive(Debug, Clone, Copy)]
pub struct AlphaImage<'a> {
    pub width: u32,
    pub height: u32,
    pub rgba: &'a [u8],
}

/// Corners a card's polygon has at most.
const CORNERS: usize = 8;
/// Texels a kept texel grows by on every side.
const GROWTH: f64 = 1.5;
/// The least share of its area a card's polygon saves, or the card keeps
/// its quad.
const SAVING: f64 = 0.1;
/// Texture coordinates nearer than this fraction of a texel are the same.
const SNAP: f32 = 1.0 / 64.0;
/// Texels by which the point where extended edges meet may lie outside its
/// card, from rounding, and still be moved onto it.
const SLACK: f64 = 1.0e-6;

/// `mesh` with each card cut to the texels `alpha` keeps at `cutoff` (0 to 1).
///
/// `alpha` is the image the mesh's first texture coordinates sample, as the
/// masked material drawing its cards does. Each card becomes its polygon: a
/// fan wound as the card was, over vertices whose position, normal, texture
/// coordinates, colors, second texture coordinates and joints blend the
/// card's corners' bilinearly by their places on it. A card keeping no
/// texel goes; one whose polygon would save less than a tenth of its area,
/// or reaching past the image, where its texels are the sampler's, keeps
/// its quad. Every other triangle stays as it was, in order; vertices only
/// replaced cards used go. The same input gives the same bytes on every
/// host.
///
/// Fails for an image whose pixels do not match its size or with a side
/// outside 1 to [`MAX_TEXTURE_DIMENSION`], for a cutoff outside 0 to 1, and
/// for meshes with levels of detail or virtual geometry, which are made
/// from trimmed cards.
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
    // Cards over one rectangle share its fit.
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

/// What becomes of one of a mesh's triangles.
#[derive(Clone, Copy)]
enum Role<'a> {
    /// It stays as it is.
    Kept,
    /// It goes: its card goes, or its card's other triangle draws its
    /// polygon.
    Gone,
    /// Its card's polygon, places on the card, is drawn where it was.
    Polygon(&'a Card, &'a [[f64; 2]]),
}

/// What a card becomes.
enum Fit {
    /// It keeps its quad.
    Keep,
    /// It keeps no texel and goes.
    Drop,
    /// Its polygon: corners counterclockwise with u right and v up, each its
    /// place on the card from 0 to 1 along u, then along v.
    Trim(Vec<[f64; 2]>),
}

/// Two triangles of a mesh making a card.
struct Card {
    /// Their places in the mesh's triangles, the earlier first.
    triangles: [usize; 2],
    /// Its vertices at (u0, v0), (u1, v0), (u0, v1) and (u1, v1).
    corners: [u32; 4],
    /// Its texture coordinates' span: u0, u1, v0 and v1.
    rect: [f32; 4],
    /// Whether its triangles wind clockwise with u right and v up.
    clockwise: bool,
}

/// A triangle that is half a card: the vertex at its right angle, whose
/// sides run along u and v, the vertex the side along v reaches and the one
/// the side along u reaches, and the diagonal across the right angle as the
/// triangle runs it.
#[derive(Clone, Copy)]
struct Half {
    corner: u32,
    along_v: u32,
    along_u: u32,
    diagonal: [u32; 2],
}

impl Half {
    /// `triangle` as half a card, where one of its corners is a right angle
    /// whose sides run along u and v, each longer than `snap` (in u, in v).
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

    /// Whether `vertex` is the rectangle's fourth corner: at the u of
    /// `along_u` and the v of `along_v`.
    fn completes(self, vertex: u32, vertices: &[MeshVertex], snap: [f32; 2]) -> bool {
        let uv = |vertex: u32| vertices[vertex as usize].uv;
        let [fourth, u_end, v_end] = [vertex, self.along_u, self.along_v].map(uv);
        (fourth[0] - u_end[0]).abs() <= snap[0] && (fourth[1] - v_end[1]).abs() <= snap[1]
    }

    /// The card this half makes with the triangle at `triangles[1]`, whose
    /// vertex across the diagonal is `fourth`.
    fn card(self, fourth: u32, triangles: [usize; 2], vertices: &[MeshVertex]) -> Card {
        let uv = |vertex: u32| vertices[vertex as usize].uv;
        let [right, u_end, v_end] = [self.corner, self.along_u, self.along_v].map(uv);
        // Whether the right angle is on the card's u1 side, and its v1 side.
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
        // Differences and products of f32 values are exact in f64.
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

/// Every card of `mesh`, texture coordinates nearer than `snap` (in u, in
/// v) counting as the same: in triangle order, each triangle that is half a
/// card with the first other triangle in no card yet that runs its diagonal
/// the other way and completes its rectangle.
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
    // Every directed edge with its triangle, sorted to look up.
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

/// The vertex of `triangle` across its directed `edge`, where it has it.
fn across(triangle: [u32; 3], edge: [u32; 2]) -> Option<u32> {
    (0..3)
        .find(|&at| [triangle[at], triangle[(at + 1) % 3]] == edge)
        .map(|at| triangle[(at + 2) % 3])
}

/// An image's alpha read against a cutoff.
struct Coverage<'a> {
    width: u32,
    height: u32,
    rgba: &'a [u8],
    /// Whether each alpha value reaches the cutoff.
    keeps: [bool; 256],
    /// How near in u, and in v, texture coordinates are the same.
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
        // The dimension limit bounds the product below usize::MAX on Wasm32.
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
            // Samplers read an alpha byte as its unorm value.
            keeps: std::array::from_fn(|alpha| alpha as f32 / 255.0 >= cutoff),
            snap: [SNAP / width as f32, SNAP / height as f32],
        })
    }

    /// What a card whose texture coordinates span `rect` (u0, u1, v0, v1)
    /// becomes.
    fn fit(&self, rect: [f32; 4]) -> Fit {
        let [u0, u1, v0, v1] = rect;
        let [snap_u, snap_v] = self.snap;
        // Past the image, the texels sampled are the sampler's.
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

    /// The corners of each row's run of kept texels, grown and cut to
    /// `bounds` (x0, x1, y0, y1 in texels, x right and y down); `None` where
    /// it keeps no texel.
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

    /// The first and last texels of `row` from `columns` (both included)
    /// that it keeps.
    fn run(&self, row: u32, columns: [u32; 2]) -> Option<[u32; 2]> {
        let start = row as usize * self.width as usize;
        let kept =
            |column: &u32| self.keeps[usize::from(self.rgba[(start + *column as usize) * 4 + 3])];
        let first = (columns[0]..=columns[1]).find(kept)?;
        let last = (first..=columns[1]).rev().find(kept)?;
        Some([first, last])
    }
}

/// The first and last of `size` texels whose grown spans overlap the open
/// span from `low` to `high`; `None` where none does.
fn reach(low: f64, high: f64, size: u32) -> Option<[u32; 2]> {
    let first = ((low - 1.0 - GROWTH).floor() + 1.0).max(0.0);
    let last = ((high + GROWTH).ceil() - 1.0).min(f64::from(size) - 1.0);
    (first <= last).then_some([first as u32, last as u32])
}

/// Texels `run[0]` to `run[1]` of a row or column grown by [`GROWTH`] on
/// both sides and cut to `low` and `high`.
fn grown(run: [u32; 2], low: f64, high: f64) -> [f64; 2] {
    [
        (f64::from(run[0]) - GROWTH).max(low),
        (f64::from(run[1]) + 1.0 + GROWTH).min(high),
    ]
}

/// The convex hull of `points`, counterclockwise with x right and y up,
/// without collinear corners.
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

/// One side of a hull: the points, in order, that turn left, but the last,
/// which starts the other side.
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

/// How far `c` lies left of the line from `a` through `b`: twice the signed
/// area of their triangle.
fn turn(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// The area of `polygon`, counterclockwise.
fn area(polygon: &[[f64; 2]]) -> f64 {
    let first = polygon[0];
    let twice: f64 = polygon
        .windows(2)
        .map(|pair| turn(first, pair[0], pair[1]))
        .sum();
    0.5 * twice
}

/// Cuts `polygon`, convex and counterclockwise, to at most [`CORNERS`]
/// corners within `bounds` (x0, x1, y0, y1): each time the edge whose
/// neighbours, extended until they meet within `bounds`, add the least area
/// gives way to the point where they meet. False where no edge can.
fn reduce(polygon: &mut Vec<[f64; 2]>, bounds: [f64; 4]) -> bool {
    while polygon.len() > CORNERS {
        let count = polygon.len();
        let mut best: Option<(f64, usize, [f64; 2])> = None;
        for edge in 0..count {
            let at = |step: usize| polygon[(edge + step) % count];
            // `count - 1` steps on is one back.
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

/// Where the lines along `before` to `start` and along `end` to `after`
/// meet past the edge from `start` to `end`, moved onto `bounds` from within
/// rounding of it, and the area its triangle with the edge adds; `None`
/// where they meet behind the edge, never, or outside `bounds`.
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

/// The mesh being made: its triangles, over the mesh's own vertices and
/// then those its polygons add, and which of its own vertices the cards it
/// replaces used.
struct Trimmed<'a> {
    mesh: &'a Mesh,
    /// The mesh's own vertices below its vertex count, then the added ones.
    indices: Vec<usize>,
    added: Vec<Added>,
    /// Each added vertex by its card's corners and its place on the card.
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

    /// Copies the mesh's triangle `triangle`.
    fn copy(&mut self, triangle: usize) {
        let mesh = self.mesh;
        let corners = &mesh.indices()[triangle * 3..triangle * 3 + 3];
        self.indices
            .extend(corners.iter().map(|&index| index as usize));
    }

    /// Marks the corners of a card that is trimmed or goes.
    fn replace(&mut self, corners: [u32; 4]) {
        for corner in corners {
            self.replaced[corner as usize] = true;
        }
    }

    /// The fan of `polygon`, places on `card`, wound as the card's
    /// triangles.
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

    /// The vertex at `at` on `card`: its corner where it is one, otherwise
    /// the one every polygon of the card adds there.
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

    /// The trimmed mesh: the mesh's own vertices but those only replaced
    /// cards used, in order, then the added ones, with every attribute the
    /// mesh has.
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
                // Indices name at most u32::MAX + 1 vertices.
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
            // Blending rounds, so it may step past the range a little.
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

/// The `values` of a mesh's own vertices `kept`, then the `added` ones.
fn joined<T: Copy>(values: &[T], kept: &[usize], added: impl IntoIterator<Item = T>) -> Vec<T> {
    kept.iter()
        .map(|&vertex| values[vertex])
        .chain(added)
        .collect()
}

/// A vertex a polygon adds: its card's corners and its place on the card.
#[derive(Clone, Copy)]
struct Added {
    corners: [u32; 4],
    at: [f64; 2],
}

impl Added {
    /// Each corner's share of it, bilinear in its place.
    fn shares(self) -> [f64; 4] {
        let [s, t] = self.at;
        [(1.0 - s) * (1.0 - t), s * (1.0 - t), (1.0 - s) * t, s * t]
    }

    /// The corners' values `of` blended by their shares.
    fn blend<const N: usize>(self, of: impl Fn(usize) -> [f32; N]) -> [f32; N] {
        let mut blended = [0.0_f64; N];
        for (corner, share) in self.corners.into_iter().zip(self.shares()) {
            for (sum, value) in blended.iter_mut().zip(of(corner as usize)) {
                *sum += share * f64::from(value);
            }
        }
        blended.map(|sum| sum as f32)
    }

    /// The corners' joints blended by their shares: each joint's weight
    /// summed over them, the four heaviest kept.
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
        // The heaviest first; of equal weights, the lower joint.
        sums.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut skin = ([0; 4], [0.0; 4]);
        for (slot, (joint, weight)) in sums.into_iter().take(4).enumerate() {
            skin.0[slot] = joint;
            skin.1[slot] = weight as f32;
        }
        skin
    }
}
