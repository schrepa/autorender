//! Deterministic procedural high-poly meshes for the Stress Lab's dense
//! scene: a displaced boulder, a fluted column and a terrain
//! tile, each hundreds of thousands of triangles.

use engine_math::collections::HashMap;

use glam::Vec3;

/// Triangles with per-vertex normals and optional texture coordinates.
pub struct Geometry {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Option<Vec<[f32; 2]>>,
    pub indices: Vec<u32>,
}

impl Geometry {
    pub fn source(&self) -> crate::Source<'_> {
        crate::Source {
            positions: &self.positions,
            normals: &self.normals,
            uvs: self.uvs.as_deref(),
            indices: &self.indices,
        }
    }

    /// Area-weighted vertex normals of the triangles, shared by vertices at
    /// the same position (seams stay smooth).
    fn smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
        let mut first = HashMap::default();
        let welded: Vec<usize> = positions
            .iter()
            .enumerate()
            .map(|(vertex, position)| *first.entry(position.map(f32::to_bits)).or_insert(vertex))
            .collect();
        let mut sums = vec![Vec3::ZERO; positions.len()];
        for triangle in indices.as_chunks::<3>().0 {
            let [a, b, c] = triangle.map(|index| Vec3::from_array(positions[index as usize]));
            let normal = (b - a).cross(c - a);
            for &index in triangle {
                sums[welded[index as usize]] += normal;
            }
        }
        welded
            .iter()
            .map(|&vertex| sums[vertex].try_normalize().unwrap_or(Vec3::Y).to_array())
            .collect()
    }
}

fn hash(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    let mut value = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f)
        ^ seed.wrapping_mul(0x1656_67b1);
    value ^= value >> 13;
    value = value.wrapping_mul(0x5bd1_e995);
    value ^= value >> 15;
    (value & 0x00ff_ffff) as f32 / 0x00ff_ffff as f32 * 2.0 - 1.0
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Value noise in [-1, 1].
fn noise(point: Vec3, seed: u32) -> f32 {
    let cell = point.floor();
    let fraction = point - cell;
    let [x, y, z] = cell.to_array().map(|value| value as i32);
    let (u, v, w) = (smooth(fraction.x), smooth(fraction.y), smooth(fraction.z));
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let corner = |dx, dy, dz| hash(x + dx, y + dy, z + dz, seed);
    lerp(
        lerp(
            lerp(corner(0, 0, 0), corner(1, 0, 0), u),
            lerp(corner(0, 1, 0), corner(1, 1, 0), u),
            v,
        ),
        lerp(
            lerp(corner(0, 0, 1), corner(1, 0, 1), u),
            lerp(corner(0, 1, 1), corner(1, 1, 1), u),
            v,
        ),
        w,
    )
}

/// Fractal noise of `octaves` octaves, roughly in [-1, 1].
fn fractal(point: Vec3, octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut amplitude, mut frequency, mut total) = (0.0, 1.0, 1.0, 0.0);
    for octave in 0..octaves {
        sum += noise(point * frequency, seed.wrapping_add(octave)) * amplitude;
        total += amplitude;
        amplitude *= 0.5;
        frequency *= 2.03;
    }
    sum / total
}

/// A boulder: an icosahedron subdivided `levels` times (20 * 4^levels
/// triangles) and displaced by fractal noise, about a unit across.
pub fn boulder(levels: u32, seed: u32) -> Geometry {
    let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let mut positions: Vec<Vec3> = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ]
    .iter()
    .map(|point| Vec3::from_array(*point).normalize())
    .collect();
    let mut indices: Vec<u32> = vec![
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7,
        1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9,
        8, 1,
    ];
    for _ in 0..levels {
        let mut midpoints: HashMap<(u32, u32), u32> = HashMap::default();
        let mut midpoint = |a: u32, b: u32, positions: &mut Vec<Vec3>| {
            let key = (a.min(b), a.max(b));
            *midpoints.entry(key).or_insert_with(|| {
                positions.push(((positions[a as usize] + positions[b as usize]) * 0.5).normalize());
                (positions.len() - 1) as u32
            })
        };
        let mut next = Vec::with_capacity(indices.len() * 4);
        for &[a, b, c] in indices.as_chunks::<3>().0 {
            let ab = midpoint(a, b, &mut positions);
            let bc = midpoint(b, c, &mut positions);
            let ca = midpoint(c, a, &mut positions);
            next.extend([a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca]);
        }
        indices = next;
    }
    let displaced: Vec<[f32; 3]> = positions
        .iter()
        .map(|direction| {
            // Broad lumps, then sharp detail, squashed a little.
            let lumps = fractal(*direction * 1.7, 4, seed);
            let detail = fractal(*direction * 9.0, 4, seed.wrapping_add(17));
            let radius = 0.5 * (1.0 + 0.32 * lumps + 0.06 * detail);
            (*direction * radius * Vec3::new(1.0, 0.72, 1.0)).to_array()
        })
        .collect();
    let normals = Geometry::smooth_normals(&displaced, &indices);
    Geometry {
        positions: displaced,
        normals,
        uvs: None,
        indices,
    }
}

/// A fluted column: `segments` around, `rings` up, six units tall, with a
/// base and a capital.
pub fn column(segments: u32, rings: u32) -> Geometry {
    let (mut positions, mut uvs, mut indices) = (Vec::new(), Vec::new(), Vec::new());
    let height = 6.0;
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let y = v * height;
        // Profile: base and capital blocks, an entasis in between.
        let profile = if v < 0.06 {
            0.62
        } else if v > 0.93 {
            0.66 - (v - 0.93) * 0.8
        } else {
            0.5 + 0.03 * engine_math::sin(v * std::f32::consts::PI)
        };
        let seam = positions.len();
        for segment in 0..=segments {
            let u = segment as f32 / segments as f32;
            uvs.push([u, v]);
            // The seam's second column of vertices shares the first's
            // positions: the column stays closed.
            if segment == segments {
                positions.push(positions[seam]);
                continue;
            }
            let angle = u * std::f32::consts::TAU;
            let (sin, cos) = engine_math::sin_cos(angle);
            // Noise sampled around the cylinder repeats with the angle.
            let around = |circumference: f32, height: f32| {
                let radius = circumference / std::f32::consts::TAU;
                Vec3::new(cos * radius, sin * radius, v * height)
            };
            let fluted = if (0.06..=0.93).contains(&v) {
                // Twenty-four concave flutes with fine chisel noise.
                let flute = engine_math::cos(angle * 24.0);
                profile - 0.018 * (1.0 - flute * flute) + 0.002 * noise(around(180.0, 90.0), 11)
            } else {
                profile + 0.004 * noise(around(60.0, 40.0), 13)
            };
            positions.push([cos * fluted, y, sin * fluted]);
        }
    }
    let row = segments + 1;
    for ring in 0..rings {
        for segment in 0..segments {
            let a = ring * row + segment;
            let (b, c, d) = (a + 1, a + row, a + row + 1);
            // Counterclockwise seen from outside.
            indices.extend([a, c, b, b, c, d]);
        }
    }
    let mut normals = Geometry::smooth_normals(&positions, &indices);
    // Caps close the ends (a closed mesh has no open borders, which stay at
    // full detail at every level), over rims of their own for crisp edges.
    for (ring, top) in [(0, false), (rings, true)] {
        let center = positions.len() as u32;
        let y = ring as f32 / rings as f32 * height;
        let normal = [0.0, if top { 1.0 } else { -1.0 }, 0.0];
        positions.push([0.0, y, 0.0]);
        normals.push(normal);
        uvs.push([0.5, 0.5]);
        for segment in 0..=segments {
            let rim = positions[(ring * row + segment) as usize];
            positions.push(rim);
            normals.push(normal);
            uvs.push([0.5 + rim[0] * 0.5, 0.5 + rim[2] * 0.5]);
        }
        for segment in 0..segments {
            let a = center + 1 + segment;
            if top {
                indices.extend([center, a + 1, a]);
            } else {
                indices.extend([center, a, a + 1]);
            }
        }
    }
    Geometry {
        positions,
        normals,
        uvs: Some(uvs),
        indices,
    }
}

/// Value noise in [-1, 1] over the plane, repeating every `period` lattice
/// cells in x and z.
fn periodic_noise(x: f32, z: f32, period: i32, seed: u32) -> f32 {
    let (cell_x, cell_z) = (x.floor(), z.floor());
    let (u, v) = (smooth(x - cell_x), smooth(z - cell_z));
    let wrap = |value: i32| value.rem_euclid(period);
    let (x0, z0) = (cell_x as i32, cell_z as i32);
    let corner = |dx: i32, dz: i32| hash(wrap(x0 + dx), 0, wrap(z0 + dz), seed);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    lerp(
        lerp(corner(0, 0), corner(1, 0), u),
        lerp(corner(0, 1), corner(1, 1), u),
        v,
    )
}

/// Fractal [`periodic_noise`] over a tile spanning `[0, 1)` in x and z:
/// every octave repeats across the tile, so copies of it meet seamlessly.
fn tileable(x: f32, z: f32, cells: i32, octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut amplitude, mut period, mut total) = (0.0, 1.0, cells, 0.0);
    for octave in 0..octaves {
        let scale = period as f32;
        sum += periodic_noise(x * scale, z * scale, period, seed.wrapping_add(octave)) * amplitude;
        total += amplitude;
        amplitude *= 0.5;
        period *= 2;
    }
    sum / total
}

/// A terrain tile `size` units across of `cells` x `cells` quads, rolling
/// within about a unit of zero height; copies side by side meet exactly.
pub fn terrain(cells: u32, size: f32, seed: u32) -> Geometry {
    let (mut positions, mut normals, mut uvs, mut indices) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let step = size / cells as f32;
    // Heights repeat every `cells` samples, so the last row and column equal
    // the first.
    let heights: Vec<f32> = (0..cells * cells)
        .map(|index| {
            let (u, v) = (
                (index % cells) as f32 / cells as f32,
                (index / cells) as f32 / cells as f32,
            );
            0.9 * tileable(u, v, 4, 5, seed) + 0.08 * tileable(u, v, 64, 3, seed.wrapping_add(5))
        })
        .collect();
    let height = |column: u32, row: u32| heights[((row % cells) * cells + column % cells) as usize];
    for row in 0..=cells {
        for column in 0..=cells {
            let (x, z) = (
                column as f32 * step - size * 0.5,
                row as f32 * step - size * 0.5,
            );
            positions.push([x, height(column, row), z]);
            // Central differences across the wrap: a border vertex gets the
            // normal its neighbor tile's copy gets.
            let (left, right) = (height(column + cells - 1, row), height(column + 1, row));
            let (back, front) = (height(column, row + cells - 1), height(column, row + 1));
            normals.push(
                Vec3::new(left - right, 2.0 * step, back - front)
                    .normalize()
                    .to_array(),
            );
            uvs.push([column as f32 / cells as f32, row as f32 / cells as f32]);
        }
    }
    let row = cells + 1;
    for z in 0..cells {
        for x in 0..cells {
            let a = z * row + x;
            let (b, c, d) = (a + 1, a + row, a + row + 1);
            // Counterclockwise seen from above.
            indices.extend([a, c, b, b, c, d]);
        }
    }
    Geometry {
        positions,
        normals,
        uvs: Some(uvs),
        indices,
    }
}
