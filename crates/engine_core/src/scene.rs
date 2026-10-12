use bevy_ecs::prelude::{Component, Resource};
use engine_assets::{Material, MaterialHandle, Mesh, MeshHandle, Surface, Texture, TextureHandle};
use engine_settings::{ExtraRules, InvalidSetting, settings};
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
        Self::IDENTITY
    }
}

impl Transform {
    /// No translation, rotation or scaling.
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// At `translation`, unrotated and unscaled.
    #[must_use]
    pub const fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            ..Self::IDENTITY
        }
    }

    /// At `(x, y, z)`, unrotated and unscaled.
    #[must_use]
    pub const fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self::from_translation(Vec3::new(x, y, z))
    }

    /// Rotated by the normalized `rotation`, at the origin and unscaled.
    #[must_use]
    pub const fn from_rotation(rotation: Quat) -> Self {
        Self {
            rotation,
            ..Self::IDENTITY
        }
    }

    /// Scaled by `scale`, at the origin and unrotated.
    #[must_use]
    pub const fn from_scale(scale: Vec3) -> Self {
        Self {
            scale,
            ..Self::IDENTITY
        }
    }

    /// This transform at `translation`.
    #[must_use]
    pub const fn with_translation(mut self, translation: Vec3) -> Self {
        self.translation = translation;
        self
    }

    /// This transform rotated by the normalized `rotation` instead.
    #[must_use]
    pub const fn with_rotation(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    /// This transform scaled by `scale` instead.
    #[must_use]
    pub const fn with_scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

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
    /// Linear RGB color; cubes are always opaque and ignore alpha. Draw
    /// transparent geometry with a [`Mesh3d`] whose material has an opacity.
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

settings! {
    unauthored;
    extra_rules;
    /// The scene's single perspective camera, in a right-handed Y-up world.
    /// Its rules are the ones scenes author and development controls
    /// request; frames draw a camera breaking them as made to keep them
    /// ([`Self::sanitized`]).
    #[derive(Resource, Debug, Clone, Copy)]
    pub struct Camera {
        /// Where it stands.
        pub position: Vec3 = Vec3::new(24.0, 19.0, 28.0) => range(-Camera::REACH, Camera::REACH),
        /// What it looks at, at least [`Camera::MIN_DEPTH`] from its position.
        pub target: Vec3 = Vec3::ZERO => range(-Camera::REACH, Camera::REACH),
        /// The vertical field of view, radians.
        pub vertical_fov_radians: f32 = 55.0_f32.to_radians()
            => range(Camera::MIN_FOV_DEGREES.to_radians(), Camera::MAX_FOV_DEGREES.to_radians()),
        /// The near plane's distance.
        pub near: f32 = 0.1 => range(Camera::MIN_DEPTH, Camera::REACH),
        /// The far plane's distance, at least [`Camera::MIN_DEPTH`] beyond
        /// the near plane's.
        pub far: f32 = 500.0 => range(Camera::MIN_DEPTH, Camera::REACH),
    }
}

impl ExtraRules for Camera {
    /// It looks somewhere, through some depth.
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if (self.target - self.position).length_squared() < Self::MIN_DEPTH * Self::MIN_DEPTH {
            return Err(InvalidSetting::new(
                "target",
                format!(
                    "must be at least {} units from the position",
                    Self::MIN_DEPTH
                ),
            ));
        }
        if self.far - self.near < Self::MIN_DEPTH {
            return Err(InvalidSetting::new(
                "far",
                format!("must be at least {} beyond near", Self::MIN_DEPTH),
            ));
        }
        Ok(())
    }

    /// A target on the position moves a unit down -Z (up it at the edge of
    /// reach); a far plane short of the near one lies the default's depth
    /// beyond it, the near plane brought in where that would leave reach.
    fn sanitize_extra(mut self) -> Self {
        if (self.target - self.position).length_squared() < Self::MIN_DEPTH * Self::MIN_DEPTH {
            let step = if self.position.z - 1.0 >= -Self::REACH {
                Vec3::NEG_Z
            } else {
                Vec3::Z
            };
            self.target = self.position + step;
        }
        if self.far - self.near < Self::MIN_DEPTH {
            let defaults = Self::default();
            let depth = defaults.far - defaults.near;
            self.far = crate::math::min(self.near + depth, Self::REACH);
            self.near = crate::math::min(self.near, self.far - depth);
        }
        self
    }
}

impl Camera {
    /// How far from the world's origin cameras stand and look, and their
    /// planes reach: world units, on each axis.
    pub const REACH: f32 = 1.0e6;
    /// The nearest a near plane comes, and the least a far plane lies
    /// beyond it and a target from its position: world units.
    pub const MIN_DEPTH: f32 = 0.001;
    /// The narrowest and widest vertical fields of view, in degrees.
    pub const MIN_FOV_DEGREES: f32 = 1.0;
    pub const MAX_FOV_DEGREES: f32 = 179.0;

    /// The position [`Self::view_projection`] looks from: `position`, made
    /// to keep its rule.
    pub fn eye(&self) -> Vec3 {
        self.sanitized().position
    }

    /// Builds a WebGPU-compatible projection with reversed depth:
    /// the near plane maps to depth 1 and the far plane to 0, which a
    /// floating-point depth buffer resolves about evenly at every distance.
    /// A camera breaking its rules draws as made to keep them
    /// ([`Self::sanitized`]), an invalid aspect as 1; a vertical direction
    /// uses Z as the up axis.
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        let (projection, position, direction, up) = self.lens(aspect);
        projection * Mat4::look_to_rh(position, direction, up)
    }

    /// [`Self::view_projection`] seen from the origin: the view and
    /// projection of positions relative to the eye (camera-relative
    /// rendering), exact however far the eye is from the world's origin.
    pub fn relative_view_projection(&self, aspect: f32) -> Mat4 {
        let (projection, _, direction, up) = self.lens(aspect);
        projection * Mat4::look_to_rh(Vec3::ZERO, direction, up)
    }

    /// The projection, eye, view direction and up axis of
    /// [`Self::view_projection`], of the camera made to keep its rules.
    fn lens(&self, aspect: f32) -> (Mat4, Vec3, Vec3, Vec3) {
        let aspect = if aspect.is_finite() && aspect > 0.0 {
            aspect
        } else {
            1.0
        };
        let Self {
            position,
            target,
            vertical_fov_radians: fov,
            near,
            far,
        } = self.sanitized();
        let direction = (target - position).try_normalize().unwrap_or(Vec3::NEG_Z);
        let up = if direction.dot(Vec3::Y).abs() > 0.999 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        // Swapping the planes reverses depth.
        (
            Mat4::perspective_rh(fov, aspect, far, near),
            position,
            direction,
            up,
        )
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

/// Multiplies one object's color channel by channel: its
/// material's base color and opacity, or its cube's color. Factors are in
/// `[0, 1]`; others read as the nearest bound, and non-finite ones as 1.
/// Objects keep their material's draw groups, and changing a tint uploads
/// only that object's color: use it for per-object and per-frame colors
/// instead of a material per color.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Tint(pub [f32; 4]);

impl Tint {
    /// No change.
    pub const WHITE: Self = Self([1.0; 4]);

    /// `color` multiplied by this tint's sanitized factors.
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

/// CPU data for one instance; this is deliberately not a GPU buffer layout.
#[derive(Debug, Clone, Copy)]
pub struct RenderInstance {
    pub mesh: MeshHandle,
    /// For materials with maps ([`Material::has_maps`]), the
    /// material whose maps the renderer binds: the first extracted material
    /// with an equal [`Material::binding`], listed in
    /// [`RenderSnapshot::materials`], so materials differing only in color
    /// and surface draw together. `None` draws from color and surface alone.
    pub material: Option<MaterialHandle>,
    pub model: Mat4,
    /// Linear color; alpha is the opacity, 1 for opaque.
    /// Without a material, an opacity below 1 draws transparent; a
    /// material's alpha mode decides otherwise.
    pub color: [f32; 4],
    /// The material's surface response; cubes are [`Surface::MATTE`].
    pub surface: Surface,
    /// Skinned instances: the offset in
    /// [`RenderSnapshot::joints`] of the first joint of their palette, which
    /// then positions their vertices instead of `model`. `None` draws rigidly.
    pub skin: Option<u32>,
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

/// One referenced material with maps, copied so it outlives
/// its library entry; its textures are in [`RenderSnapshot::textures`].
/// `version` is its [`engine_assets::AssetLibrary::material_version`]: renderers
/// keep what they built from a handle while its version stands.
#[derive(Debug, Clone, Copy)]
pub struct RenderMaterial {
    pub handle: MaterialHandle,
    pub material: Material,
    pub version: u64,
}

impl RenderInstance {
    /// Whether this instance draws in the transparent pass: its material's
    /// alpha mode decides, `material` being the instance's listed material;
    /// without one, an opacity below 1.
    pub fn is_transparent(&self, material: Option<&Material>) -> bool {
        match material {
            Some(material) => material.is_transparent(),
            None => self.color[3] < 1.0,
        }
    }
}

/// Owned render data, reusable across frames without borrowing the ECS world.
#[derive(Debug, Default)]
pub struct RenderSnapshot {
    pub view_projection: Mat4,
    /// The camera's position ([`Camera::eye`]), for view-dependent lighting.
    pub camera_position: Vec3,
    /// Presentation time ([`crate::FrameTime::elapsed_seconds`]) for animated
    /// surfaces such as ripples.
    pub elapsed_seconds: f32,
    /// The simulation time the frame presents ([`crate::PresentedTime`]),
    /// for what must line up with simulated objects, such as the sea's
    /// waves.
    pub presented_seconds: f64,
    /// The scene's atmosphere and every effect's settings,
    /// sanitized ([`crate::Environment::current`]): a missing atmosphere
    /// reads as the default one and a missing effect resource as off.
    pub environment: crate::Environment,
    /// This frame's local lights.
    pub lights: crate::RenderLights,
    /// This frame's particle emitters.
    pub particles: crate::RenderParticles,
    /// This frame's decals and what they draw with.
    pub decals: crate::RenderDecals,
    /// This frame's reflection probes.
    pub reflection_probes: crate::RenderReflectionProbes,
    /// This frame's terrains: their resident tiles and the
    /// patches the camera draws.
    pub terrain: crate::RenderTerrains,
    /// This frame's deformation map and the stamps it presses.
    pub deformation: crate::RenderDeformation,
    /// World streaming's state at this frame.
    pub cells: crate::CellStats,
    /// Every instance's `color[3]` is its opacity; see
    /// [`RenderInstance::is_transparent`].
    pub instances: Vec<RenderInstance>,
    /// Joint palettes of skinned instances, each a skin's
    /// joints in order from its instance's [`RenderInstance::skin`] offset,
    /// followed by decals' poses.
    pub joints: Vec<crate::RenderJoint>,
    /// Unique sources used by emitted or frustum-culled valid instances.
    /// Culled sources allow the renderer to retain already resident resources;
    /// they do not request uploads by themselves.
    pub meshes: Vec<RenderMesh>,
    /// Unique materials with maps used by emitted or frustum-culled valid
    /// instances.
    pub materials: Vec<RenderMaterial>,
    /// Unique textures of those materials.
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

/// One joint of a skinned instance's palette, in
/// [`crate::RenderSnapshot::joints`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderJoint {
    /// The joint's hierarchy node in the index space of
    /// [`crate::GpuPresentation`] records, or `u32::MAX` without a valid
    /// pose. Set on GPU-presented frames, which compose the joint's
    /// displayed pose as they do slots'.
    pub node: u32,
    /// The joint's inverse bind matrix, in upload layout.
    pub inverse_bind: crate::PoseRows,
    /// CPU-prepared frames: the joint's displayed world matrix times its
    /// inverse bind matrix, in upload layout. GPU-presented frames leave it
    /// unset.
    pub matrix: crate::PoseRows,
}
