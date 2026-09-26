use bevy::{
    asset::{AssetMetaCheck, AssetPlugin, RenderAssetUsages},
    color::LinearRgba,
    input::mouse::AccumulatedMouseMotion,
    mesh::Indices,
    prelude::*,
    reflect::TypePath,
    render::render_resource::{
        AsBindGroup, Extent3d, PrimitiveTopology, TextureDimension, TextureFormat,
    },
    shader::ShaderRef,
    time::{Fixed, Virtual},
    transform::TransformSystems,
    window::WindowPlugin,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const GROUP_SIZE: u32 = 32;
const SPACING: f32 = 2.4;
const COLORS: [[f32; 4]; 8] = [
    [0.09, 0.88, 0.78, 1.0],
    [0.10, 0.49, 1.0, 1.0],
    [0.55, 0.20, 1.0, 1.0],
    [1.0, 0.22, 0.52, 1.0],
    [1.0, 0.61, 0.13, 1.0],
    [0.25, 0.93, 0.54, 1.0],
    [0.45, 0.72, 1.0, 1.0],
    [0.92, 0.37, 0.97, 1.0],
];

/// Shares the source scene's configuration vocabulary so its copied links work.
#[derive(Clone, Resource, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct StressConfig {
    seed: u32,
    renderables: u32,
    moving_percent: u32,
    hierarchy: bool,
    interpolation: bool,
    colliders: u32,
    sensor_percent: u32,
    dense_colliders: bool,
    queries_per_tick: u32,
    mesh_variety: u32,
    textured: bool,
    culling: bool,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            seed: 1337,
            renderables: 90_000,
            moving_percent: 0,
            hierarchy: false,
            interpolation: false,
            colliders: 0,
            sensor_percent: 0,
            dense_colliders: false,
            queries_per_tick: 0,
            mesh_variety: 32,
            textured: true,
            culling: true,
        }
    }
}

impl StressConfig {
    fn validate(&self) -> Result<(), String> {
        if !(1..=90_000).contains(&self.renderables)
            || self.moving_percent > 100
            || ![1, 8, 32].contains(&self.mesh_variety)
        {
            return Err(
                "renderables must be 1..90000, moving_percent 0..100, mesh_variety 1/8/32".into(),
            );
        }
        if self.colliders != 0 || self.queries_per_tick != 0 || self.dense_colliders {
            return Err("Bevy-only comparison supports render and transform workloads; set colliders=0, queries_per_tick=0, dense_colliders=false".into());
        }
        Ok(())
    }
}

#[derive(Component)]
struct Motion {
    anchor: Vec3,
    rotation: Quat,
    phase: f32,
    speed: f32,
    previous: Transform,
    current: Transform,
}

#[derive(Component)]
struct ComparisonCamera;

#[derive(Resource)]
struct CameraSpeed(f32);

/// Bevy owns drawing and material binding; the fragment math matches our Stress Lab.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct StressMaterial {
    #[uniform(0)]
    color: LinearRgba,
    #[texture(1)]
    #[sampler(2)]
    color_texture: Option<Handle<Image>>,
}

impl Material for StressMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/stress_material.wgsl".into()
    }
}

fn main() {
    let config = load_config().unwrap_or_else(|error| panic!("invalid comparison config: {error}"));
    let moving = config.moving_percent > 0;
    let interpolate = moving && config.interpolation;
    let span = (config.renderables as f32).sqrt().ceil() * SPACING;
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Bevy Stress Comparison".into(),
                    canvas: Some("#bevy-canvas".into()),
                    fit_canvas_to_parent: true,
                    ..default()
                }),
                ..default()
            })
            .set(AssetPlugin {
                meta_check: AssetMetaCheck::Never,
                ..default()
            }),
        MaterialPlugin::<StressMaterial>::default(),
    ));
    app.insert_resource(Time::<Fixed>::from_duration(Duration::from_secs_f64(
        1.0 / 60.0,
    )));
    app.insert_resource(Time::<Virtual>::from_max_delta(Duration::from_secs_f64(
        5.0 / 60.0,
    )));
    app.insert_resource(ClearColor(Color::linear_rgb(0.009, 0.015, 0.025)));
    app.insert_resource(CameraSpeed((span * 0.3).clamp(22.0, 220.0)));
    app.insert_resource(config);
    app.add_systems(Startup, setup);
    app.add_systems(Update, move_camera);
    if moving {
        app.add_systems(FixedUpdate, animate);
    }
    if interpolate {
        app.add_systems(
            PostUpdate,
            interpolate_motion.before(TransformSystems::Propagate),
        );
    }
    app.run();
}

fn load_config() -> Result<StressConfig, String> {
    #[cfg(target_arch = "wasm32")]
    {
        let search = web_sys::window()
            .ok_or("missing browser window")?
            .location()
            .search()
            .map_err(|_| "cannot read location.search")?;
        let params = web_sys::UrlSearchParams::new_with_str(&search)
            .map_err(|_| "cannot parse query string")?;
        let config = if let Some(value) = params.get("config") {
            if value.len() > 4096 {
                return Err("configuration exceeds 4096 bytes".into());
            }
            serde_json::from_str(&value).map_err(|error| error.to_string())?
        } else {
            StressConfig::default()
        };
        config.validate()?;
        Ok(config)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let config = match std::env::var("BEVY_STRESS_CONFIG") {
            Ok(value) => serde_json::from_str(&value).map_err(|error| error.to_string())?,
            Err(_) => StressConfig::default(),
        };
        config.validate()?;
        Ok(config)
    }
}

fn setup(world: &mut World) {
    let config = world.resource::<StressConfig>().clone();
    let variants = make_variants(world, &config);
    let mut parent = None;
    let mut parent_inverse = Mat4::IDENTITY;

    for index in 0..config.renderables {
        let position = position_for_index(index, &config);
        if config.hierarchy && index % GROUP_SIZE == 0 {
            let end = (index + GROUP_SIZE - 1).min(config.renderables - 1);
            let anchor = (position + position_for_index(end, &config)) * 0.5;
            let transform = Transform {
                translation: anchor,
                rotation: Quat::from_rotation_y(0.25 + (index % 5) as f32 * 0.1),
                scale: Vec3::new(1.2, 0.9, 0.8),
            };
            parent_inverse = transform.to_matrix().inverse();
            parent = Some(world.spawn(transform).id());
        }
        let random = hash(index ^ config.seed);
        let tall = 1.0 + ((random >> 8) % 70) as f32 * 0.11;
        let transform = Transform {
            translation: if parent.is_some() {
                parent_inverse.transform_point3(position)
            } else {
                position
            },
            rotation: Quat::from_rotation_y((random % 8) as f32 * 0.16),
            scale: Vec3::new(
                0.7 + (random % 5) as f32 * 0.08,
                tall,
                0.7 + ((random >> 4) % 5) as f32 * 0.08,
            ),
        };
        let (mesh, material) = &variants[index as usize % variants.len()];
        let mut entity = world.spawn((
            Mesh3d(mesh.clone()),
            MeshMaterial3d(material.clone()),
            transform,
        ));
        if index % 100 < config.moving_percent {
            entity.insert(Motion {
                anchor: transform.translation,
                rotation: transform.rotation,
                phase: (random % 10_000) as f32 * 0.001,
                speed: 0.3 + ((random >> 12) % 15) as f32 * 0.05,
                previous: transform,
                current: transform,
            });
        }
        if !config.culling {
            entity.insert(bevy::camera::visibility::NoFrustumCulling);
        }
        if let Some(parent) = parent {
            entity.insert(ChildOf(parent));
        }
    }

    let span = (config.renderables as f32).sqrt().ceil() * SPACING;
    let (camera_position, camera_target) = match load_view().as_str() {
        "close" => (Vec3::new(15.0, 15.0, 23.0), Vec3::new(0.0, 3.0, 0.0)),
        "away" => {
            let position = Vec3::new(0.0, span, span * 1.4);
            (position, position + Vec3::Z)
        }
        _ => (Vec3::new(span * 0.46, span * 0.62, span * 0.8), Vec3::ZERO),
    };
    world.spawn((
        Camera3d::default(),
        bevy::camera::CompositingSpace::Srgb,
        ComparisonCamera,
        bevy::render::prelude::Msaa::Off,
        Projection::Perspective(PerspectiveProjection {
            fov: 55.0_f32.to_radians(),
            far: 5000.0,
            ..default()
        }),
        Transform::from_translation(camera_position).looking_at(camera_target, Vec3::Y),
        bevy::core_pipeline::tonemapping::Tonemapping::None,
    ));
}

fn move_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    speed: Res<CameraSpeed>,
    mut camera: Single<&mut Transform, With<ComparisonCamera>>,
) {
    if mouse_buttons.pressed(MouseButton::Left) && mouse_motion.delta != Vec2::ZERO {
        let (yaw, pitch, _) = camera.rotation.to_euler(EulerRot::YXZ);
        camera.rotation = Quat::from_rotation_y(yaw - mouse_motion.delta.x * 0.003)
            * Quat::from_rotation_x((pitch - mouse_motion.delta.y * 0.003).clamp(-1.48, 1.48));
    }

    let forward = i32::from(keys.any_pressed([KeyCode::KeyW, KeyCode::ArrowUp]))
        - i32::from(keys.any_pressed([KeyCode::KeyS, KeyCode::ArrowDown]));
    let right = i32::from(keys.any_pressed([KeyCode::KeyD, KeyCode::ArrowRight]))
        - i32::from(keys.any_pressed([KeyCode::KeyA, KeyCode::ArrowLeft]));
    let up = i32::from(keys.pressed(KeyCode::KeyE)) - i32::from(keys.pressed(KeyCode::KeyQ));
    if forward == 0 && right == 0 && up == 0 {
        return;
    }
    let forward_axis = camera.rotation * Vec3::NEG_Z;
    let right_axis = camera.rotation * Vec3::X;
    let movement = forward_axis * forward as f32 + right_axis * right as f32 + Vec3::Y * up as f32;
    camera.translation += movement.normalize_or_zero() * speed.0 * time.delta_secs();
}

fn load_view() -> String {
    #[cfg(target_arch = "wasm32")]
    {
        let search = web_sys::window()
            .and_then(|window| window.location().search().ok())
            .unwrap_or_default();
        web_sys::UrlSearchParams::new_with_str(&search)
            .ok()
            .and_then(|params| params.get("view"))
            .unwrap_or_else(|| "overview".into())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var("BEVY_STRESS_VIEW").unwrap_or_else(|_| "overview".into())
    }
}

fn make_variants(
    world: &mut World,
    config: &StressConfig,
) -> Vec<(Handle<Mesh>, Handle<StressMaterial>)> {
    let mut variants = Vec::with_capacity(config.mesh_variety as usize);
    for index in 0..config.mesh_variety {
        let mesh = world.resource_mut::<Assets<Mesh>>().add(make_mesh(index));
        let texture = if config.textured {
            Some(
                world
                    .resource_mut::<Assets<Image>>()
                    .add(make_texture(index)),
            )
        } else {
            None
        };
        let color = COLORS[index as usize % COLORS.len()];
        let material = world
            .resource_mut::<Assets<StressMaterial>>()
            .add(StressMaterial {
                color: LinearRgba::new(color[0], color[1], color[2], color[3]),
                color_texture: texture,
            });
        variants.push((mesh, material));
    }
    variants
}

fn make_mesh(index: u32) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    if index % 3 == 2 {
        for y in [-1.0_f32, 1.0] {
            for (a, b) in [
                (Vec3::X, Vec3::Z),
                (Vec3::Z, Vec3::NEG_X),
                (Vec3::NEG_X, Vec3::NEG_Z),
                (Vec3::NEG_Z, Vec3::X),
            ] {
                let tip = Vec3::Y * y;
                let (a, b) = if y > 0.0 { (b, a) } else { (a, b) };
                let normal = (a - tip).cross(b - tip).normalize();
                for (position, uv) in
                    [tip, a, b]
                        .into_iter()
                        .zip([[0.5, 0.0], [0.0, 1.0], [1.0, 1.0]])
                {
                    positions.push((position * 0.5).to_array());
                    normals.push(normal.to_array());
                    uvs.push(uv);
                    indices.push(indices.len() as u32);
                }
            }
        }
    } else {
        for normal in [
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
        ] {
            let tangent = if normal.y.abs() == 1.0 {
                Vec3::X
            } else {
                Vec3::Y.cross(normal)
            };
            let bitangent = normal.cross(tangent);
            let offset = positions.len() as u32;
            for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                positions.push(((normal + tangent * x + bitangent * y) * 0.5).to_array());
                normals.push(normal.to_array());
                uvs.push([(x + 1.0) * 0.5, (1.0 - y) * 0.5]);
            }
            indices.extend([
                offset,
                offset + 1,
                offset + 2,
                offset,
                offset + 2,
                offset + 3,
            ]);
        }
    }
    let sx = 0.85 + (index % 4) as f32 * 0.05;
    let sz = 0.85 + (index % 7) as f32 * 0.025;
    for position in &mut positions {
        position[0] *= sx;
        position[2] *= sz;
    }
    for normal in &mut normals {
        normal[0] /= sx;
        normal[2] /= sz;
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices))
}

fn make_texture(index: u32) -> Image {
    let mut pixels = Vec::with_capacity(64 * 64 * 4);
    for y in 0..64_u32 {
        for x in 0..64_u32 {
            let stripe = (x / 4 + y / 4 + index) % 2 == 0;
            let value = if x % 8 == 0 || y % 8 == 0 {
                90
            } else if stripe {
                255
            } else {
                195
            };
            pixels.extend_from_slice(&[value, value, value, 255]);
        }
    }
    Image::new(
        Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

fn position_for_index(index: u32, config: &StressConfig) -> Vec3 {
    let side = (config.renderables as f32).sqrt().ceil() as u32;
    let center = (side - 1) as f32 * SPACING * 0.5;
    let random = hash(index ^ config.seed);
    let height = 1.0 + ((random >> 8) % 70) as f32 * 0.11;
    Vec3::new(
        (index % side) as f32 * SPACING - center,
        height * 0.5,
        (index / side) as f32 * SPACING - center,
    )
}

fn hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn animate(time: Res<Time<Fixed>>, mut objects: Query<(&mut Motion, &mut Transform)>) {
    let t = (time.elapsed_secs_f64() % 10_000.0) as f32;
    for (mut motion, mut transform) in &mut objects {
        let angle = t * motion.speed + motion.phase;
        let next = Transform {
            translation: motion.anchor + Vec3::Y * (angle.sin() * 0.6),
            rotation: motion.rotation
                * Quat::from_rotation_y(angle * 0.7)
                * Quat::from_rotation_z(angle.sin() * 0.12),
            scale: transform.scale,
        };
        motion.previous = motion.current;
        motion.current = next;
        *transform = next;
    }
}

fn interpolate_motion(time: Res<Time<Fixed>>, mut objects: Query<(&Motion, &mut Transform)>) {
    let alpha = time.overstep_fraction();
    for (motion, mut transform) in &mut objects {
        *transform = Transform {
            translation: motion
                .previous
                .translation
                .lerp(motion.current.translation, alpha),
            rotation: motion
                .previous
                .rotation
                .slerp(motion.current.rotation, alpha),
            scale: motion.current.scale,
        };
    }
}
