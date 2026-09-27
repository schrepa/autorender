use bevy_ecs::prelude::{Component, Resource};
use engine_assets::{MaterialHandle, Mesh, MeshHandle, Texture, TextureHandle};
use glam::{BVec4A, Mat4, Quat, Vec3, Vec4, Vec4Swizzles};

use crate::NodePose;
use std::sync::Arc;

/// Local translation, normalized rotation, and scale relative to [`crate::Parent`].
/// A root's local transform is also its world transform. The derived
/// [`crate::GlobalTransform`] preserves composed matrices, including shear.
/// Edit through ordinary mutable Bevy queries or [`crate::edit_transform`].
/// Normal change detection tracks writes; do not bypass it for transform data.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
#[component(
    on_add = crate::hierarchy::tracking::topology_changed,
    on_remove = crate::hierarchy::tracking::topology_changed)]
pub struct Transform {
    pub translation: Vec3,
    /// Keep this quaternion normalized.
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    /// The pose in upload layout, `[t.x, t.y, t.z, s.x, r.x, r.y, r.z, r.w,
    /// s.y, s.z]`: the hierarchy's cached copy doubles as GPU input.
    #[inline]
    pub(crate) fn to_pose(self) -> NodePose {
        let (t, r, s) = (self.translation, self.rotation, self.scale);
        [t.x, t.y, t.z, s.x, r.x, r.y, r.z, r.w, s.y, s.z]
    }

    #[inline]
    pub(crate) fn from_pose(pose: &NodePose) -> Self {
        let [tx, ty, tz, sx, rx, ry, rz, rw, sy, sz] = *pose;
        Self {
            translation: Vec3::new(tx, ty, tz),
            rotation: Quat::from_xyzw(rx, ry, rz, rw),
            scale: Vec3::new(sx, sy, sz),
        }
    }

    /// `Mat4::from_scale_rotation_translation(scale, rotation, translation)`,
    /// bit for bit except NaN payloads.
    ///
    /// Every lane performs the same IEEE operation as glam's SIMD
    /// `quat_to_axes` followed by each axis times its scale component, without
    /// extracting and reinserting scalar lanes (costly in browser Wasm). Pinned
    /// glam 0.30.9; revisit on math-library upgrades.
    #[cfg_attr(not(feature = "attribution"), inline)]
    #[cfg_attr(feature = "attribution", inline(never))]
    pub fn matrix(&self) -> Mat4 {
        const X: BVec4A = BVec4A::new(true, false, false, false);
        const Y: BVec4A = BVec4A::new(false, true, false, false);
        const Z: BVec4A = BVec4A::new(false, false, true, false);
        let q = Vec4::from(self.rotation);
        let q2 = q + q;
        // (xx, yy, zz), (xy, xz, yz) and (wx, wy, wz), with xx = x * (x + x).
        let squares = q * q2;
        let mixed = q.xxyw() * q2.yzzw();
        let w = (q.wwww() * q2).zyxw();
        // 1 - (yy + zz), 1 - (xx + zz), 1 - (xx + yy).
        let diagonal = Vec4::ONE - (squares.yxxw() + squares.zzyw());
        // (xy + wz, xz + wy, yz + wx) and (xy - wz, xz - wy, yz - wx).
        let sum = mixed + w;
        let difference = mixed - w;
        let axis = |x: Vec4, y: Vec4, z: Vec4| {
            Vec4::select(X, x, Vec4::select(Y, y, Vec4::select(Z, z, Vec4::ZERO)))
        };
        Mat4::from_cols(
            axis(diagonal, sum.xxzw(), difference.xxyw()) * self.scale.x,
            axis(difference, diagonal, sum) * self.scale.y,
            axis(sum.yyyy(), difference.zzzz(), diagonal) * self.scale.z,
            self.translation.extend(1.0),
        )
    }
}

/// A centered unit cube; its transform supplies position and dimensions.
#[derive(Component, Debug, Clone, Copy)]
pub struct Cube {
    /// Linear RGBA color. The initial renderer draws opaque cubes.
    pub color: [f32; 4],
}

/// An instance of immutable CPU geometry and an opaque material.
///
/// Handles resolve in the world's `AssetLibrary`. A missing asset skips this
/// entity during extraction. This component takes precedence over `Cube`.
#[derive(Component, Debug, Clone, Copy)]
pub struct Mesh3d {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}

/// The scene's single perspective camera, in a right-handed Y-up world.
#[derive(Resource, Debug, Clone, Copy)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub vertical_fov_radians: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(24.0, 19.0, 28.0),
            target: Vec3::ZERO,
            vertical_fov_radians: 55.0_f32.to_radians(),
            near: 0.1,
            far: 500.0,
        }
    }
}

impl Camera {
    /// The position [`Self::view_projection`] looks from: `position`, or
    /// the default position when it is not finite.
    pub fn eye(&self) -> Vec3 {
        if self.position.is_finite() {
            self.position
        } else {
            Self::default().position
        }
    }

    /// Builds a WebGPU-compatible projection (depth in [0, 1]).
    /// Invalid aspect/lens values fall back to sensible defaults; a coincident
    /// target looks down -Z and a vertical direction uses Z as the up axis.
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        let defaults = Self::default();
        let aspect = if aspect.is_finite() && aspect > 0.0 {
            aspect
        } else {
            1.0
        };
        let fov = if self.vertical_fov_radians.is_finite()
            && self.vertical_fov_radians > 0.0
            && self.vertical_fov_radians < std::f32::consts::PI
        {
            self.vertical_fov_radians
        } else {
            defaults.vertical_fov_radians
        };
        let near = if self.near.is_finite() && self.near > 0.0 {
            self.near
        } else {
            defaults.near
        };
        let far = if self.far.is_finite() && self.far > near {
            self.far
        } else {
            near + defaults.far
        };
        let position = if self.position.is_finite() {
            self.position
        } else {
            defaults.position
        };
        let direction = (self.target - position)
            .try_normalize()
            .unwrap_or(Vec3::NEG_Z);
        // `eye` repeats this fallback.
        let up = if direction.dot(Vec3::Y).abs() > 0.999 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        Mat4::perspective_rh(fov, aspect, near, far) * Mat4::look_to_rh(position, direction, up)
    }
}

/// One coarse host input batch. Movement axes are held in [-1, 1]; look values
/// are pointer deltas in CSS pixels, consumed in `Presentation`, not once per tick.
/// The host overwrites this resource before every `Engine::advance` call.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct InputState {
    pub move_forward: f32,
    pub move_right: f32,
    pub move_up: f32,
    pub look_yaw: f32,
    pub look_pitch: f32,
}

/// CPU data for one instance; this is deliberately not a GPU buffer layout.
#[derive(Debug, Clone, Copy)]
pub struct RenderInstance {
    pub mesh: MeshHandle,
    /// Optional opaque base color texture; absence samples renderer white.
    pub texture: Option<TextureHandle>,
    pub model: Mat4,
    pub color: [f32; 4],
}

/// One referenced mesh, retained independently of its library's lifetime.
#[derive(Debug, Clone)]
pub struct RenderMesh {
    pub handle: MeshHandle,
    pub mesh: Arc<Mesh>,
}

/// One referenced texture, retained independently of its library's lifetime.
#[derive(Debug, Clone)]
pub struct RenderTexture {
    pub handle: TextureHandle,
    pub texture: Arc<Texture>,
}

/// Owned render data, reusable across frames without borrowing the ECS world.
#[derive(Debug, Default)]
pub struct RenderSnapshot {
    pub view_projection: Mat4,
    /// The camera's position ([`Camera::eye`]), for view-dependent lighting.
    pub camera_position: Vec3,
    /// The scene's lighting, sanitized ([`crate::Atmosphere::sanitized`]).
    pub atmosphere: crate::Atmosphere,
    pub instances: Vec<RenderInstance>,
    /// Unique sources used by emitted or frustum-culled valid instances.
    /// Culled sources allow the renderer to retain already resident resources;
    /// they do not request uploads by themselves.
    pub meshes: Vec<RenderMesh>,
    /// Unique textures used by emitted or frustum-culled valid instances.
    pub textures: Vec<RenderTexture>,
    /// Every entity carrying Cube or Mesh3d, before transform/asset checks.
    pub render_candidates: usize,
    /// Valid instances surviving culling, before the output-instance budget.
    pub visible_instances: usize,
    /// Valid instances whose mesh bounds are proven outside the frustum.
    pub culled_instances: usize,
    pub dropped_instances: usize,
    /// Renderable entities skipped because a mesh, material, or texture is absent.
    pub missing_assets: usize,
    /// Renderable entities skipped because their derived transform is invalid.
    pub invalid_transforms: usize,
}

#[cfg(test)]
mod matrix_tests {
    use super::*;

    fn same_lanes(left: Mat4, right: Mat4) -> bool {
        // NaN payloads are unspecified; every other lane matches exactly.
        left.to_cols_array()
            .iter()
            .zip(right.to_cols_array())
            .all(|(a, b)| (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits())
    }

    #[test]
    fn lane_parallel_matrix_matches_glam_bit_for_bit() {
        let mut state = 0x2545_f491_u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let special = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.0e-40,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            3.0e19,
        ];
        let mut value = |magnitude: f32| {
            let bits = next();
            if bits % 16 == 0 {
                special[(bits >> 8) as usize % special.len()]
            } else {
                ((bits >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0) * magnitude
            }
        };
        for round in 0..200_000 {
            let magnitude = [1.0, 30.0, 1.0e6, 1.0e-3][round % 4];
            let rotation = Quat::from_xyzw(value(1.0), value(1.0), value(1.0), value(1.0));
            let rotation = if round % 3 == 0 {
                rotation.normalize()
            } else {
                rotation
            };
            let transform = Transform {
                translation: Vec3::new(value(magnitude), value(magnitude), value(magnitude)),
                rotation,
                scale: Vec3::new(value(4.0), value(4.0), value(4.0)),
            };
            let expected = Mat4::from_scale_rotation_translation(
                transform.scale,
                transform.rotation,
                transform.translation,
            );
            assert!(same_lanes(transform.matrix(), expected), "{transform:?}");
        }
    }
}
