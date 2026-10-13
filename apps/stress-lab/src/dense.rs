//! The dense scene: boulders and fluted columns strewn over
//! terrain tiles, every mesh a virtual mesh of hundreds of thousands of
//! triangles built offline (`cargo run -p engine_vgbuild --release --bin
//! dense_assets`) and fetched by the page.

use engine::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use engine::Error;
use engine::VirtualMesh;
use engine::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use engine_native::files;

use crate::kit::field::{Motion, animate};
use crate::kit::motion::Animation;
use crate::kit::random::hash;
use crate::kit::{StressConfig, StressFile, StressWorld, Workload};

/// The files the page fetches for this scene, from `stress/dense/`: the ones
/// `tools/vgbuild`'s `dense_assets` writes (it lists them again, with their
/// generators, and never depends on this game).
pub const FILES: [&str; 4] = [
    "boulder-a.evg",
    "boulder-b.evg",
    "column.evg",
    "terrain.evg",
];
/// Units each terrain tile spans.
const TILE: f32 = 64.0;
/// Ground each object stands on, on average.
const AREA_PER_OBJECT: f32 = 36.0;

/// Builds the scene from `files`, returning its moving nodes and bounds.
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub(crate) fn populate(
    world: &mut World,
    config: &StressConfig,
    files: &HashMap<String, StressFile>,
) -> engine::Result<(u32, (Vec3, Vec3))> {
    let mut assets = world.resource_mut::<AssetLibrary>();
    let mut mesh = |name: &str| -> engine::Result<MeshHandle> {
        let file = files.get(name).ok_or_else(|| {
            engine::Error::new(
                engine::ErrorCode::MissingFile,
                name,
                format!(
                    "the dense scene needs {name}: run `cargo run -p engine_vgbuild --release --bin dense_assets`"
                ),
            )
        })?;
        // A prefix streams its other pages from the file's URL.
        let geometry = VirtualMesh::decode(&file.bytes)
            .map_err(|error| engine::Error::from(error).context(name))?;
        let geometry = Arc::new(match &file.url {
            Some(url) => geometry.with_source(url),
            None => geometry,
        });
        let mesh = Mesh::from_virtual(geometry, engine::DEFAULT_FALLBACK_TRIANGLES)
            .map_err(|error| engine::Error::from(error).context(name))?;
        assets.insert_mesh(mesh).map_err(engine::Error::from)
    };
    let boulders = [mesh(FILES[0])?, mesh(FILES[1])?];
    let column = mesh(FILES[2])?;
    let terrain = mesh(FILES[3])?;
    let mut material =
        |color: [f32; 4], roughness: f32, metalness: f32| -> engine::Result<MaterialHandle> {
            let material = Material::new(color).and_then(|material| {
                material.with_surface(Surface {
                    roughness,
                    metalness,
                    emission: 0.0,
                    ripples: 0.0,
                })
            })?;
            assets
                .insert_material(material)
                .map_err(engine::Error::from)
        };
    let rock = material([0.46, 0.43, 0.40, 1.0], 0.85, 0.0)?;
    let dark_rock = material([0.30, 0.29, 0.30, 1.0], 0.7, 0.0)?;
    let marble = material([0.86, 0.84, 0.80, 1.0], 0.35, 0.0)?;
    let bronze = material([0.70, 0.48, 0.28, 1.0], 0.3, 0.9)?;
    let ground = material([0.36, 0.42, 0.26, 1.0], 0.9, 0.0)?;

    let side = math::max((config.renderables as f32 * AREA_PER_OBJECT).sqrt(), TILE);
    let tiles = (side / TILE).ceil() as i32;
    let half = tiles as f32 * TILE * 0.5;
    for x in 0..tiles {
        for z in 0..tiles {
            let center = Vec3::new(
                (x as f32 + 0.5) * TILE - half,
                0.0,
                (z as f32 + 0.5) * TILE - half,
            );
            world.spawn((
                Transform {
                    translation: center,
                    ..Transform::default()
                },
                Mesh3d {
                    mesh: terrain,
                    material: ground,
                },
            ));
        }
    }
    let mut moving_nodes = 0;
    for index in 0..config.renderables {
        let random = hash(index ^ config.seed);
        let spread =
            |shift: u32| ((hash(random ^ shift) % 10_000) as f32 / 10_000.0 - 0.5) * half * 1.9;
        let (x, z) = (spread(0x9e37), spread(0x7f4a));
        // One in five is a column, the rest boulders from pebbles to crags.
        let columnar = random.is_multiple_of(5);
        let turn = Quat::from_rotation_y((random % 628) as f32 * 0.01);
        let transform = if columnar {
            Transform {
                translation: Vec3::new(x, -0.4, z),
                rotation: turn,
                scale: Vec3::splat(0.8 + ((random >> 8) % 60) as f32 * 0.01),
            }
        } else {
            let size = 0.4 + ((random >> 8) % 1000) as f32 * 0.0035;
            Transform {
                translation: Vec3::new(x, size * 0.12, z),
                rotation: turn * Quat::from_rotation_x(((random >> 20) % 100) as f32 * 0.004),
                scale: Vec3::splat(size),
            }
        };
        let (mesh, material) = if columnar {
            (
                column,
                if random.is_multiple_of(3) {
                    bronze
                } else {
                    marble
                },
            )
        } else {
            (
                boulders[(random >> 4) as usize % 2],
                if random.is_multiple_of(7) {
                    dark_rock
                } else {
                    rock
                },
            )
        };
        let mut entity = world.spawn((transform, Mesh3d { mesh, material }));
        if index % 100 < config.moving_percent {
            entity.insert(Motion {
                anchor: transform.translation + Vec3::Y * 1.5,
                rotation: transform.rotation,
                phase: (random % 10_000) as f32 * 0.001,
                speed: 0.3 + ((random >> 12) % 15) as f32 * 0.05,
            });
            if config.interpolation {
                entity.insert(InterpolateTransform);
            }
            moving_nodes += 1;
        }
    }
    world.insert_resource(Workload {
        transform_nodes: config.renderables + (tiles * tiles) as u32,
        moving_nodes,
        sensor_nodes: 0,
        probes: Vec::new(),
        overlap_hits: Vec::new(),
    });
    let bounds = (Vec3::new(-half, -2.0, -half), Vec3::new(half, 12.0, half));
    Ok((moving_nodes, bounds))
}

pub(crate) struct DenseWorld;

impl StressWorld for DenseWorld {
    fn name(&self) -> &'static str {
        "dense"
    }

    fn label(&self) -> &'static str {
        "Dense geometry"
    }

    fn look(&self) -> &'static str {
        "day"
    }

    fn populate(
        &self,
        world: &mut World,
        config: &StressConfig,
        files: &HashMap<String, StressFile>,
    ) -> engine::Result<(u32, (Vec3, Vec3))> {
        populate(world, config, files)
    }

    fn add_systems(&self, app: &mut App, moving_nodes: u32) {
        // Its moving objects bob and turn as the tower field's do.
        if moving_nodes > 0 {
            app.add_systems(FixedSimulation, animate.in_set(Animation));
        }
    }

    /// Its virtual meshes' prefixes, streamed from their files.
    #[cfg(not(target_arch = "wasm32"))]
    fn files(&self, assets: &Path, _rocks: bool) -> engine::Result<HashMap<String, StressFile>> {
        let mut found = HashMap::default();
        for name in FILES {
            let path = assets.join("stress/dense").join(name);
            let bytes = files::read_prefix(&path, 128, engine::VirtualMesh::prefix_bytes).map_err(
                |error| {
                    Error::new(
                        error.code,
                        error.path,
                        format!(
                            "{} Build the files with `cargo run -p engine_vgbuild --release --bin dense_assets`.",
                            error.message
                        ),
                    )
                },
            )?;
            let url = Some(path.to_string_lossy().into_owned());
            found.insert(
                name.to_owned(),
                StressFile {
                    bytes,
                    url,
                    size: None,
                },
            );
        }
        Ok(found)
    }
}
