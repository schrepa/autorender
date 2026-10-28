use crate::prelude::*;

#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CameraSystems;

#[derive(Resource, Debug)]
pub struct FlyCameraController {
    yaw: f32,
    pitch: f32,
    speed: f32,
    sensitivity: f32,
}

impl FlyCameraController {
    pub fn new(yaw: f32, pitch: f32) -> Self {
        let yaw = if yaw.is_finite() {
            yaw % std::f32::consts::TAU
        } else {
            0.0
        };
        let pitch = if pitch.is_finite() {
            math::clamp(pitch, -1.48, 1.48)
        } else {
            0.0
        };
        Self {
            yaw,
            pitch,
            speed: 22.0,
            sensitivity: 0.003,
        }
    }

    pub fn from_camera(camera: &Camera) -> Self {
        let direction = (camera.target - camera.position).normalize_or_zero();
        Self::new((-direction.x).atan2(-direction.z), direction.y.asin())
    }

    pub fn align_to(&mut self, camera: &Camera) {
        let aligned = Self::from_camera(camera);
        self.yaw = aligned.yaw;
        self.pitch = aligned.pitch;
    }

    #[must_use]
    pub fn with_speed(mut self, units_per_second: f32) -> Self {
        if units_per_second.is_finite() {
            self.speed = math::clamp(units_per_second, 0.0, 1000.0);
        }
        self
    }
}

pub struct FlyCameraPlugin;

impl Plugin for FlyCameraPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<FlyCameraController>() {
            let camera = app
                .world()
                .get_resource::<Camera>()
                .copied()
                .unwrap_or_default();
            app.insert_resource(FlyCameraController::from_camera(&camera));
        }
        app.add_systems(Presentation, fly_camera.in_set(CameraSystems));
    }
}

fn fly_camera(
    time: Res<FrameTime>,
    input: Res<InputState>,
    mut control: ResMut<FlyCameraController>,
    mut camera: ResMut<Camera>,
) {
    control.yaw = (control.yaw - input.look_yaw * control.sensitivity) % std::f32::consts::TAU;
    control.pitch = math::clamp(
        control.pitch - input.look_pitch * control.sensitivity,
        -1.48,
        1.48,
    );
    let rotation = Quat::from_rotation_y(control.yaw) * Quat::from_rotation_x(control.pitch);
    let forward = rotation * -Vec3::Z;
    let right = rotation * Vec3::X;
    let movement =
        forward * input.move_forward + right * input.move_right + Vec3::Y * input.move_up;
    camera.position += movement.normalize_or_zero() * time.delta_seconds * control.speed;
    camera.target = camera.position + forward;
}
