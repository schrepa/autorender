use bevy_ecs::prelude::{Component, Entity, Resource, World};
use engine_assets::{
    AssetLibrary, Material, MaterialHandle, Mesh, MeshHandle, MeshLods, Texture, TextureHandle,
};
use glam::{Affine3A, BVec4A, Mat4, Quat, Vec3, Vec4, Vec4Swizzles};
use serde::Serialize;

use crate::{
    Clock, Error, ErrorCode, GlobalTransforms, Mobility, Node, NodePose, Parent, View,
};
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

#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Camera {
    pub near: f32,
    pub far: f32,
    pub projection: Projection,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Projection {
    Perspective { vertical_fov_radians: f32 },
    Orthographic { height: f32 },
    Fitted { clip: [[f32; 4]; 4] },
}

impl Default for Camera {
    fn default() -> Self {
        Self {
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

fn within(value: f32, min: f32, max: f32) -> bool {
    (min..=max).contains(&value)
}

impl Camera {
    pub const REACH: f32 = 1.0e6;
    pub const MIN_DEPTH: f32 = 0.001;
    pub const MIN_FOV_DEGREES: f32 = 1.0;
    pub const MAX_FOV_DEGREES: f32 = 179.0;
    pub const EYE: Vec3 = Vec3::new(24.0, 19.0, 28.0);

    pub fn check(&self) -> Result<(), Error> {
        let invalid = |field: &str| Err(Error::new(ErrorCode::InvalidCamera, format!("$.{field}")));
        let reach = Self::REACH;
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
            Projection::Fitted { clip } if !clip.iter().flatten().all(|value| value.is_finite()) => {
                return invalid("projection.clip");
            }
            Projection::Perspective { .. }
            | Projection::Orthographic { .. }
            | Projection::Fitted { .. } => {}
        }
        if self.far - self.near < Self::MIN_DEPTH {
            return invalid("far");
        }
        Ok(())
    }

    pub fn vertical_fov(&self) -> Option<f32> {
        match self.projection {
            Projection::Perspective {
                vertical_fov_radians,
            } => Some(vertical_fov_radians),
            Projection::Orthographic { .. } | Projection::Fitted { .. } => None,
        }
    }

    pub fn clip(&self, aspect: f32) -> Mat4 {
        let aspect = if aspect.is_finite() && aspect > 0.0 {
            aspect
        } else {
            1.0
        };
        match self.projection {
            Projection::Perspective {
                vertical_fov_radians,
            } => Mat4::perspective_rh(vertical_fov_radians, aspect, self.far, self.near),
            Projection::Orthographic { height } => {
                let (half_width, half_height) = (height * aspect * 0.5, height * 0.5);
                Mat4::orthographic_rh(
                    -half_width,
                    half_width,
                    -half_height,
                    half_height,
                    self.far,
                    self.near,
                )
            }
            Projection::Fitted { clip } => Mat4::from_cols_array_2d(&clip),
        }
    }

    pub fn fitted(clip: Mat4, origin: Vec3) -> Self {
        Self {
            projection: Projection::Fitted {
                clip: (clip * Mat4::from_translation(origin)).to_cols_array_2d(),
            },
            ..Self::default()
        }
    }

    pub fn view_projection(&self, eye: Vec3, forward: Vec3, aspect: f32) -> Mat4 {
        let direction = forward.try_normalize().unwrap_or(Vec3::NEG_Z);
        let up = if direction.dot(Vec3::Y).abs() > 0.999 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        self.clip(aspect) * Mat4::look_to_rh(eye, direction, up)
    }
}

impl Transform {
    pub fn looking_at(position: Vec3, target: Vec3) -> Self {
        let direction = (target - position).try_normalize().unwrap_or(Vec3::NEG_Z);
        let yaw = crate::math::atan2(-direction.x, -direction.z);
        let pitch = crate::math::clamp(direction.y, -1.0, 1.0).asin();
        Self::from_translation(position)
            .with_rotation(Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch))
    }
}

pub(crate) fn spawn_main_view(world: &mut World) {
    let camera = world
        .spawn((
            Transform::looking_at(Camera::EYE, Vec3::ZERO),
            Camera::default(),
            Node {
                clock: Some(Clock::Real),
                mobility: Some(Mobility::Moving),
                ..Node::default()
            },
        ))
        .id();
    world.spawn(View::new(camera));
}

pub fn place_camera(world: &mut World, camera: Entity, at: Transform, lens: Camera) {
    let parent = world
        .get::<Parent>(camera)
        .map(Parent::parent)
        .and_then(|parent| world.get_resource::<GlobalTransforms>()?.pose(parent));
    let to = match parent {
        Some(pose) => {
            let local =
                pose.inverse() * Affine3A::from_rotation_translation(at.rotation, at.translation);
            let (scale, rotation, translation) = local.to_scale_rotation_translation();
            Transform {
                translation,
                rotation,
                scale,
            }
        }
        None => at,
    };
    if let Ok(mut held) = world.get_entity_mut(camera) {
        held.insert(lens);
    }
    crate::send_ask(world, crate::Teleport { node: camera, to });
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

#[derive(Debug, Clone)]
pub struct RenderMaterial {
    pub handle: MaterialHandle,
    pub material: Material,
    pub version: u64,
}
