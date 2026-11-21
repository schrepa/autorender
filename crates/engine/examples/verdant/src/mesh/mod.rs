pub mod flora;
pub mod shrine;
pub mod terrain;

use glam::{Vec2, Vec3};

pub use crate::bricks::world::MeshVertex;

pub const MAT_TERRAIN: f32 = 0.0;
pub const MAT_WATER: f32 = 1.0;
pub const MAT_STONE: f32 = 2.0;
pub const MAT_VERMILION: f32 = 3.0;
pub const MAT_TIMBER: f32 = 4.0;
pub const MAT_GLOW: f32 = 5.0;
pub const MAT_ROPE: f32 = 6.0;
pub const MAT_BARK: f32 = 7.0;
pub const MAT_BLOSSOM: f32 = 8.0;
pub const MAT_PINE: f32 = 9.0;
pub const MAT_GOLD: f32 = 10.0;

#[derive(Clone, Copy)]
pub struct SurfaceStyle {
    pub mat: f32,
    pub ao: f32,
    pub moss: f32,
    pub tint: [f32; 4],
}

pub struct WorldMeshes {
    pub vertices: Vec<MeshVertex>,
    pub opaque_indices: Vec<u32>,
    pub foliage_indices: Vec<u32>,
    pub terrain_index_count: u32,
}

pub struct MeshBuilder {
    pub vertices: Vec<MeshVertex>,
    pub opaque_indices: Vec<u32>,
    pub foliage_indices: Vec<u32>,
}

impl Default for MeshBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MeshBuilder {
    pub fn new() -> Self {
        Self {
            vertices: Vec::with_capacity(140_000),
            opaque_indices: Vec::with_capacity(320_000),
            foliage_indices: Vec::with_capacity(140_000),
        }
    }

    pub fn push_vertex(
        &mut self,
        pos: Vec3,
        normal: Vec3,
        uv: Vec2,
        extra: Vec2,
        style: SurfaceStyle,
    ) -> u32 {
        let index = self.vertices.len() as u32;
        let n = normal.normalize_or_zero();
        self.vertices.push(MeshVertex {
            pos_mat: [pos.x, pos.y, pos.z, style.mat],
            norm_ao: [n.x, n.y, n.z, style.ao],
            uv_extra: [uv.x, uv.y, extra.x, extra.y],
            tint: style.tint,
        });
        index
    }

    pub fn push_tri(&mut self, a: u32, b: u32, c: u32, foliage: bool) {
        let target = if foliage {
            &mut self.foliage_indices
        } else {
            &mut self.opaque_indices
        };
        target.push(a);
        target.push(b);
        target.push(c);
    }

    pub fn push_quad(&mut self, a: u32, b: u32, c: u32, d: u32, foliage: bool) {
        self.push_tri(a, b, c, foliage);
        self.push_tri(a, c, d, foliage);
    }

    pub fn push_box(&mut self, center: Vec3, half: Vec3, yaw: f32, style: SurfaceStyle) {
        let (sy, cy) = engine::math::sin_cos(yaw);
        let rot = |v: Vec3| Vec3::new(v.x * cy + v.z * sy, v.y, -v.x * sy + v.z * cy);
        let faces: [(Vec3, Vec3, Vec3); 6] = [
            (Vec3::X, Vec3::NEG_Z, Vec3::Y),
            (Vec3::NEG_X, Vec3::Z, Vec3::Y),
            (Vec3::Y, Vec3::X, Vec3::NEG_Z),
            (Vec3::NEG_Y, Vec3::X, Vec3::Z),
            (Vec3::Z, Vec3::X, Vec3::Y),
            (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y),
        ];
        for (n_local, u_axis, v_axis) in faces {
            let n = rot(n_local);
            let du = u_axis * (u_axis.abs().dot(half));
            let dv = v_axis * (v_axis.abs().dot(half));
            let dn = n_local * (n_local.abs().dot(half));
            let corners = [dn - du - dv, dn + du - dv, dn + du + dv, dn - du + dv];
            let uvs = [
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(0.0, 1.0),
            ];
            let mut idx = [0u32; 4];
            for i in 0..4 {
                let p = center + rot(corners[i]);
                let ao = engine::math::clamp(
                    style.ao + (corners[i].y / (half.y + 0.001)) * 0.14,
                    0.3,
                    1.0,
                );
                let s = SurfaceStyle { ao, ..style };
                idx[i] = self.push_vertex(p, n, uvs[i], Vec2::new(0.0, style.moss), s);
            }
            self.push_quad(idx[0], idx[1], idx[2], idx[3], false);
        }
    }

    pub fn push_tube(
        &mut self,
        points: &[(Vec3, f32)],
        sides: u32,
        style: SurfaceStyle,
        twist: f32,
    ) {
        if points.len() < 2 || sides < 3 {
            return;
        }
        let mut rings: Vec<Vec<u32>> = Vec::with_capacity(points.len());
        let mut prev_right = Vec3::X;
        for (i, &(pos, radius)) in points.iter().enumerate() {
            let tang = if i == 0 {
                (points[1].0 - pos).normalize_or_zero()
            } else if i + 1 == points.len() {
                (pos - points[i - 1].0).normalize_or_zero()
            } else {
                (points[i + 1].0 - points[i - 1].0).normalize_or_zero()
            };
            let ref_up = if tang.y.abs() > 0.92 { Vec3::Z } else { Vec3::Y };
            let mut right = tang.cross(ref_up).normalize_or_zero();
            if i > 0 && right.dot(prev_right) < 0.0 {
                right = -right;
            }
            prev_right = right;
            let up = right.cross(tang).normalize_or_zero();
            let v_coord = i as f32 / (points.len() - 1) as f32;
            let ao = engine::math::clamp(0.62 + 0.38 * v_coord, 0.45, 1.0);
            let s_ring = SurfaceStyle { ao, ..style };
            let mut ring = Vec::with_capacity((sides + 1) as usize);
            for s in 0..=sides {
                let u_coord = s as f32 / sides as f32;
                let angle = u_coord * core::f32::consts::TAU + v_coord * twist;
                let (sa, ca) = engine::math::sin_cos(angle);
                let dir = (right * ca + up * sa).normalize_or_zero();
                let r_mod = if twist > 0.01 {
                    radius * (1.0 + 0.14 * engine::math::sin(angle * 3.0))
                } else {
                    radius
                };
                let v = self.push_vertex(
                    pos + dir * r_mod,
                    dir,
                    Vec2::new(u_coord * 2.0, v_coord * 4.0),
                    Vec2::new(v_coord, style.moss),
                    s_ring,
                );
                ring.push(v);
            }
            rings.push(ring);
        }
        for i in 0..rings.len() - 1 {
            for s in 0..sides as usize {
                let a = rings[i][s];
                let b = rings[i][s + 1];
                let c = rings[i + 1][s + 1];
                let d = rings[i + 1][s];
                self.push_quad(a, d, c, b, false);
            }
        }
    }

    pub fn push_lathe(
        &mut self,
        origin: Vec3,
        profile: &[(f32, f32)],
        sides: u32,
        yaw_offset: f32,
        style: SurfaceStyle,
    ) {
        if profile.len() < 2 || sides < 3 {
            return;
        }
        let mut rings: Vec<Vec<u32>> = Vec::with_capacity(profile.len());
        let y_min = profile[0].1;
        let y_span = engine::math::max(profile[profile.len() - 1].1 - y_min, 0.01);
        for (i, &(radius, height)) in profile.iter().enumerate() {
            let (dr, dy) = if i == 0 {
                (profile[1].0 - radius, profile[1].1 - height)
            } else if i + 1 == profile.len() {
                (radius - profile[i - 1].0, height - profile[i - 1].1)
            } else {
                (profile[i + 1].0 - profile[i - 1].0, profile[i + 1].1 - profile[i - 1].1)
            };
            let t = (height - y_min) / y_span;
            let ao = engine::math::clamp(0.55 + 0.45 * t, 0.4, 1.0);
            let s_ring = SurfaceStyle { ao, ..style };
            let mut ring = Vec::with_capacity((sides + 1) as usize);
            for s in 0..=sides {
                let u = s as f32 / sides as f32;
                let angle = u * core::f32::consts::TAU + yaw_offset;
                let (sa, ca) = engine::math::sin_cos(angle);
                let radial = Vec3::new(ca, 0.0, sa);
                let normal = (radial * dy - Vec3::Y * dr).normalize_or_zero();
                let pos = origin + Vec3::new(ca * radius, height, sa * radius);
                let v = self.push_vertex(
                    pos,
                    normal,
                    Vec2::new(u * 2.0, t * 2.0),
                    Vec2::new(t, style.moss),
                    s_ring,
                );
                ring.push(v);
            }
            rings.push(ring);
        }
        for i in 0..rings.len() - 1 {
            for s in 0..sides as usize {
                let a = rings[i][s];
                let b = rings[i][s + 1];
                let c = rings[i + 1][s + 1];
                let d = rings[i + 1][s];
                self.push_quad(a, b, c, d, false);
            }
        }
    }

    pub fn build(self, terrain_index_count: u32) -> WorldMeshes {
        WorldMeshes {
            vertices: self.vertices,
            opaque_indices: self.opaque_indices,
            foliage_indices: self.foliage_indices,
            terrain_index_count,
        }
    }
}

pub fn build_world_meshes() -> WorldMeshes {
    let mut builder = MeshBuilder::new();
    terrain::build_terrain(&mut builder);
    let terrain_index_count = builder.opaque_indices.len() as u32;
    shrine::build_shrine_complex(&mut builder);
    flora::build_flora_and_rocks(&mut builder);
    builder.build(terrain_index_count)
}
