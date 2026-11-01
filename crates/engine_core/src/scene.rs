use bevy_ecs::prelude::{Component, Resource};
use engine_assets::{
    AssetLibrary, Material, MaterialHandle, Mesh, MeshHandle, MeshLods, Surface, Texture,
    TextureHandle,
};
use glam::{BVec4A, Mat4, Quat, Vec3, Vec4, Vec4Swizzles};
use serde::Serialize;

use crate::{Error, ErrorCode, NodePose};
use std::sync::Arc;

#[derive(Component, Debug, Clone, Copy, PartialEq)]
#[component(
    on_add = crate::hierarchy::tracking::topology_changed,
    on_remove = crate::hierarchy::tracking::topology_changed)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    #[must_use]
    pub const fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            ..Self::IDENTITY
        }
    }

    #[must_use]
    pub const fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self::from_translation(Vec3::new(x, y, z))
    }

    #[must_use]
    pub const fn from_rotation(rotation: Quat) -> Self {
        Self {
            rotation,
            ..Self::IDENTITY
        }
    }

    #[must_use]
    pub const fn from_scale(scale: Vec3) -> Self {
        Self {
            scale,
            ..Self::IDENTITY
        }
    }

    #[must_use]
    pub const fn with_translation(mut self, translation: Vec3) -> Self {
        self.translation = translation;
        self
    }

    #[must_use]
    pub const fn with_rotation(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    #[must_use]
    pub const fn with_scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

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

    #[cfg_attr(not(feature = "attribution"), inline)]
    #[cfg_attr(feature = "attribution", inline(never))]
    pub fn matrix(&self) -> Mat4 {
        const X: BVec4A = BVec4A::new(true, false, false, false);
        const Y: BVec4A = BVec4A::new(false, true, false, false);
        const Z: BVec4A = BVec4A::new(false, false, true, false);
        let q = Vec4::from(self.rotation);
        let q2 = q + q;
        let squares = q * q2;
        let mixed = q.xxyw() * q2.yzzw();
        let w = (q.wwww() * q2).zyxw();
        let diagonal = Vec4::ONE - (squares.yxxw() + squares.zzyw());
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

#[derive(Component, Debug, Clone, Copy)]
pub struct Cube {
    pub color: [f32; 4],
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub near: f32,
    pub far: f32,
    pub projection: Projection,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, serde::Deserialize)]
pub enum Projection {
    Perspective { vertical_fov_radians: f32 },
    Orthographic { height: f32 },
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(24.0, 19.0, 28.0),
            target: Vec3::ZERO,
            near: 0.1,
            far: 500.0,
            projection: Projection::Perspective {
                vertical_fov_radians: default_fov(),
            },
        }
    }
}

fn default_fov() -> f32 {
    55.0_f32.to_radians()
}

fn kept(value: f32, min: f32, max: f32, default: f32) -> f32 {
    if value.is_finite() {
        crate::math::clamp(value, min, max)
    } else {
        default
    }
}

fn kept_point(value: Vec3, default: Vec3) -> Vec3 {
    let reach = Camera::REACH;
    Vec3::new(
        kept(value.x, -reach, reach, default.x),
        kept(value.y, -reach, reach, default.y),
        kept(value.z, -reach, reach, default.z),
    )
}

fn within(value: f32, min: f32, max: f32) -> bool {
    (min..=max).contains(&value)
}

impl Camera {
    pub const REACH: f32 = 1.0e6;
    pub const MIN_DEPTH: f32 = 0.001;
    pub const MIN_FOV_DEGREES: f32 = 1.0;
    pub const MAX_FOV_DEGREES: f32 = 179.0;

    #[doc(hidden)]
    pub fn check(&self) -> Result<(), Error> {
        let invalid = |field: &str| Err(Error::new(ErrorCode::InvalidValue, format!("$.{field}")));
        let reach = Self::REACH;
        if !self
            .position
            .to_array()
            .iter()
            .all(|&axis| within(axis, -reach, reach))
        {
            return invalid("position");
        }
        if !self
            .target
            .to_array()
            .iter()
            .all(|&axis| within(axis, -reach, reach))
        {
            return invalid("target");
        }
        if !within(self.near, Self::MIN_DEPTH, reach) {
            return invalid("near");
        }
        if !within(self.far, Self::MIN_DEPTH, reach) {
            return invalid("far");
        }
        match self.projection {
            Projection::Perspective {
                vertical_fov_radians,
            } if !within(
                vertical_fov_radians,
                Self::MIN_FOV_DEGREES.to_radians(),
                Self::MAX_FOV_DEGREES.to_radians(),
            ) =>
            {
                return invalid("projection.vertical_fov_radians");
            }
            Projection::Orthographic { height } if !within(height, Self::MIN_DEPTH, reach) => {
                return invalid("projection.height");
            }
            Projection::Perspective { .. } | Projection::Orthographic { .. } => {}
        }
        if (self.target - self.position).length_squared() < Self::MIN_DEPTH * Self::MIN_DEPTH {
            return invalid("target");
        }
        if self.far - self.near < Self::MIN_DEPTH {
            return invalid("far");
        }
        Ok(())
    }

    #[doc(hidden)]
    #[must_use]
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        let reach = Self::REACH;
        let projection = match self.projection {
            Projection::Perspective {
                vertical_fov_radians,
            } => Projection::Perspective {
                vertical_fov_radians: kept(
                    vertical_fov_radians,
                    Self::MIN_FOV_DEGREES.to_radians(),
                    Self::MAX_FOV_DEGREES.to_radians(),
                    default_fov(),
                ),
            },
            Projection::Orthographic { height } => Projection::Orthographic {
                height: kept(height, Self::MIN_DEPTH, reach, 1.0),
            },
        };
        let mut sanitized = Self {
            position: kept_point(self.position, defaults.position),
            target: kept_point(self.target, defaults.target),
            near: kept(self.near, Self::MIN_DEPTH, reach, defaults.near),
            far: kept(self.far, Self::MIN_DEPTH, reach, defaults.far),
            projection,
        };
        if (sanitized.target - sanitized.position).length_squared()
            < Self::MIN_DEPTH * Self::MIN_DEPTH
        {
            let step = if sanitized.position.z - 1.0 >= -reach {
                Vec3::NEG_Z
            } else {
                Vec3::Z
            };
            sanitized.target = sanitized.position + step;
        }
        if sanitized.far - sanitized.near < Self::MIN_DEPTH {
            let depth = defaults.far - defaults.near;
            sanitized.far = crate::math::min(sanitized.near + depth, reach);
            sanitized.near = crate::math::min(sanitized.near, sanitized.far - depth);
        }
        sanitized
    }

    pub fn eye(&self) -> Vec3 {
        self.sanitized().position
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        let (projection, position, direction, up) = self.lens(aspect);
        projection * Mat4::look_to_rh(position, direction, up)
    }

    pub fn relative_view_projection(&self, aspect: f32) -> Mat4 {
        let (projection, _, direction, up) = self.lens(aspect);
        projection * Mat4::look_to_rh(Vec3::ZERO, direction, up)
    }

    fn lens(&self, aspect: f32) -> (Mat4, Vec3, Vec3, Vec3) {
        let aspect = if aspect.is_finite() && aspect > 0.0 {
            aspect
        } else {
            1.0
        };
        let Self {
            position,
            target,
            near,
            far,
            projection,
        } = self.sanitized();
        let direction = (target - position).try_normalize().unwrap_or(Vec3::NEG_Z);
        let up = if direction.dot(Vec3::Y).abs() > 0.999 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let projection = match projection {
            Projection::Perspective {
                vertical_fov_radians,
            } => Mat4::perspective_rh(vertical_fov_radians, aspect, far, near),
            Projection::Orthographic { height } => {
                let (half_width, half_height) = (height * aspect * 0.5, height * 0.5);
                Mat4::orthographic_rh(
                    -half_width,
                    half_width,
                    -half_height,
                    half_height,
                    far,
                    near,
                )
            }
        };
        (projection, position, direction, up)
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
        }
    }
}

#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct InputState {
    pub move_forward: f32,
    pub move_right: f32,
    pub move_up: f32,
    pub look_yaw: f32,
    pub look_pitch: f32,
}

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Tint(pub [f32; 4]);

impl Tint {
    pub const WHITE: Self = Self([1.0; 4]);

    pub fn apply(&self, color: [f32; 4]) -> [f32; 4] {
        let mut tinted = color;
        for (channel, factor) in tinted.iter_mut().zip(self.0) {
            let factor = if factor.is_finite() {
                crate::math::clamp(factor, 0.0, 1.0)
            } else {
                1.0
            };
            *channel *= factor;
        }
        tinted
    }
}

impl Default for Tint {
    fn default() -> Self {
        Self::WHITE
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RenderInstance {
    pub mesh: MeshHandle,
    pub material: Option<MaterialHandle>,
    pub model: Mat4,
    pub color: [f32; 4],
    pub surface: Surface,
}

#[derive(Debug, Clone)]
pub struct RenderMesh {
    pub handle: MeshHandle,
    pub mesh: Arc<Mesh>,
    pub lods: Option<Arc<MeshLods>>,
}

impl RenderMesh {
    pub fn of(assets: &AssetLibrary, handle: MeshHandle) -> Option<Self> {
        let mesh = assets.mesh(handle)?;
        let lods = assets.mesh_lods(handle)?;
        Some(Self {
            handle,
            mesh: Arc::clone(mesh),
            lods: lods.cloned(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct RenderTexture {
    pub handle: TextureHandle,
    pub texture: Arc<Texture>,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderMaterial {
    pub handle: MaterialHandle,
    pub material: Material,
    pub version: u64,
}

#[derive(Debug, Default)]
pub struct RenderSnapshot {
    pub view_projection: Mat4,
    pub camera_position: Vec3,
    pub detail_eye: Vec3,
    pub elapsed_seconds: f32,
    pub presented_seconds: f64,
    pub wind: [f32; 4],
    pub lights: crate::RenderLights,
    pub particles: crate::RenderParticles,
    pub reflection_probes: crate::RenderReflectionProbes,
    pub tables: crate::RenderTables,
    pub instances: Vec<RenderInstance>,
    pub(crate) instance_slots: crate::retained_render::InstanceSlots,
    pub meshes: Vec<RenderMesh>,
    pub materials: Vec<RenderMaterial>,
    pub textures: Vec<RenderTexture>,
    pub render_candidates: usize,
    pub visible_instances: usize,
    pub dropped_instances: usize,
    pub missing_assets: usize,
    pub invalid_transforms: usize,
}
