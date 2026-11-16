use engine_core::{Error, ErrorCode, Mesh, MeshVertex, math};
use glam::{Vec2, Vec3};

use super::random::{golden_hash, unit};

const CELL_WORDS: usize = 6;

const SMALLEST: f32 = 0.55;

const SOFT: f32 = 0.72;

const TIP_OPEN: f32 = 0.35;

const TIP_HOLD: f32 = 0.8;

const KEPT: [f32; 4] = [1.0, 0.46, 0.19, 0.075];

const GROWN: [f32; 4] = [1.0, 1.4, 2.0, 2.9];

pub(crate) const DETAILS: usize = KEPT.len();

#[derive(Clone, Copy)]
struct Cell {
    corners: [Vec2; 4],
    uvs: [[f32; 2]; 4],
    reach: f32,
}

pub(crate) struct Atlas {
    size: [f32; 2],
    cells: Vec<[f32; CELL_WORDS]>,
}

impl Atlas {
    pub(crate) fn new(size: [f32; 2], cells: impl Iterator<Item = [f32; CELL_WORDS]>) -> Self {
        Self {
            size,
            cells: cells.collect(),
        }
    }

    pub(crate) fn write(&self, out: &mut Vec<u8>) {
        out.extend(self.size.iter().flat_map(|value| value.to_le_bytes()));
        out.extend(self.cells.iter().flatten().flat_map(|value| value.to_le_bytes()));
    }

    pub(crate) fn read(input: &[u8], at: &str) -> Result<Self, Error> {
        let values: Vec<f32> = input
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&bytes| f32::from_le_bytes(bytes))
            .collect();
        let sized = values.len() > 2
            && (values.len() - 2).is_multiple_of(CELL_WORDS)
            && input.len().is_multiple_of(4);
        if !sized || values.iter().any(|value| !value.is_finite()) {
            return Err(
                Error::new(ErrorCode::InvalidRecord, format!("{at}.cells")).got(input.len())
            );
        }
        Ok(Self {
            size: [values[0], values[1]],
            cells: values[2..]
                .as_chunks::<CELL_WORDS>()
                .0
                .to_vec(),
        })
    }

    fn cells(&self) -> Vec<Cell> {
        let size = Vec2::from(self.size).max(Vec2::ONE);
        self.cells
            .iter()
            .map(|&[x, y, w, h, root_x, root_y]| {
                let root = Vec2::new(root_x, root_y);
                let low = Vec2::new(x, y);
                let span = Vec2::new(w, h).max(Vec2::ONE);
                let along = (low + span * 0.5 - root).normalize_or(Vec2::NEG_Y);
                let across = Vec2::new(-along.y, along.x);
                let pixels = [
                    low,
                    low + Vec2::new(span.x, 0.0),
                    low + Vec2::new(0.0, span.y),
                    low + span,
                ];
                let corners = pixels.map(|pixel| {
                    let from_root = pixel - root;
                    Vec2::new(from_root.dot(along), from_root.dot(across))
                });
                Cell {
                    corners,
                    uvs: pixels.map(|pixel| (pixel / size).to_array()),
                    reach: corners
                        .iter()
                        .fold(1.0, |reach, corner| math::max(reach, corner.x)),
                }
            })
            .collect()
    }
}

pub(crate) struct Sprig {
    pub(crate) at: Vec3,
    pub(crate) along: Vec3,
    pub(crate) across: Vec3,
    pub(crate) length: f32,
    pub(crate) heart: Vec3,
    pub(crate) open: f32,
    pub(crate) tint: [f32; 3],
    pub(crate) key: u32,
    pub(crate) hold: f32,
}

pub(crate) struct Foliage {
    cells: Vec<Cell>,
    longest: f32,
    kept: f32,
    grown: f32,
    vertices: Vec<MeshVertex>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl Foliage {
    pub(crate) fn new(atlas: &Atlas, detail: usize) -> Self {
        let cells = atlas.cells();
        let longest = cells
            .iter()
            .fold(1.0, |longest, cell| math::max(longest, cell.reach));
        Self {
            cells,
            longest,
            kept: KEPT[detail],
            grown: GROWN[detail],
            vertices: Vec::new(),
            colors: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub(crate) fn sprig(&mut self, sprig: &Sprig) {
        let rank = unit(golden_hash(sprig.key ^ 0x51ed_270b)) * sprig.hold;
        if rank >= self.kept || self.cells.is_empty() {
            return;
        }
        let cell = self.cells[golden_hash(sprig.key) as usize % self.cells.len()];
        let length = sprig.length * math::max(cell.reach / self.longest, SMALLEST);
        let scale = length * self.grown / cell.reach;
        let at = sprig.at - sprig.along * (length * (self.grown - 1.0) * TIP_HOLD);
        let mirror = match golden_hash(sprig.key ^ 0x2545_f491) & 1 {
            0 => 1.0,
            _ => -1.0,
        };
        let face = sprig.across.cross(sprig.along).normalize_or(Vec3::Y);
        let up = match face.dot(sprig.at - sprig.heart) < 0.0 {
            true => -face,
            false => face,
        };
        let first = self.vertices.len() as u32;
        for (corner, uv) in cell.corners.iter().zip(cell.uvs) {
            let position =
                at + (sprig.along * corner.x + sprig.across * (corner.y * mirror)) * scale;
            let soft = (position - sprig.heart).normalize_or(up);
            let tip = math::clamp(corner.x / cell.reach, 0.0, 1.0);
            let open = sprig.open + (1.0 - sprig.open) * TIP_OPEN * tip;
            self.vertices.push(MeshVertex {
                position: position.to_array(),
                normal: (soft * SOFT + up * (1.0 - SOFT)).normalize_or(up).to_array(),
                uv,
            });
            let [red, green, blue] = sprig.tint.map(|channel| math::clamp(channel, 0.0, 1.0));
            self.colors
                .push([red, green, blue, math::clamp(open, 0.0, 1.0)]);
        }
        self.indices
            .extend([0, 1, 2, 1, 3, 2].map(|corner| first + corner));
    }

    pub(crate) fn mesh(self) -> Result<Mesh, Error> {
        Mesh::new(self.vertices, self.indices).and_then(|mesh| mesh.with_colors(self.colors))
    }
}
