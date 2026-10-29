mod bodies;
mod caves;
mod ground;
mod hiker;
mod looks;
mod rocks;
mod trees;

use engine::collections::HashMap;
use engine::host::Image;
use std::sync::Arc;

use engine::prelude::*;
use engine::terrain::*;

use crate::records;
use looks::{LOOKS, Look};

pub(crate) const SEED: u32 = 1337;

pub type RockMaps = HashMap<String, Image>;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) const ROCK_FILES: [&str; 6] = [
    "boulder_albedo.ktx2",
    "boulder_normal.ktx2",
    "boulder_orm.ktx2",
    "cliff_albedo.ktx2",
    "cliff_normal.ktx2",
    "cliff_orm.ktx2",
];

const BOULDER_SCAN_TILING: f32 = 1.8;
const WALL_SCAN_TILING: f32 = 2.5;

const TILES: u32 = 32;
const SPACING: f32 = 1.0;
const SIZE: f32 = TILES as f32 * 256.0 * SPACING;
const SURROUNDINGS: f32 = 28_672.0;
const PLATEAU: [f32; 2] = [0.0, 0.0];

const EAST_ROAD: [[f32; 2]; 3] = [[150.0, 55.0], [376.0, 137.0], [1128.0, 410.0]];

const WEST_ROAD: [[f32; 2]; 181] = [
    [3968.0, 4136.0],
    [4015.7, 4265.48],
    [4021.21, 4275.72],
    [4025.34, 4282.22],
    [4029.93, 4288.37],
    [4034.96, 4294.19],
    [4043.2, 4302.4],
    [4065.6, 4319.2],
    [4071.32, 4324.15],
    [4076.38, 4329.25],
    [4080.75, 4334.5],
    [4084.45, 4339.9],
    [4087.48, 4345.45],
    [4089.82, 4351.15],
    [4091.5, 4357.0],
    [4093.0, 4366.0],
    [4093.5, 4377.5],
    [4092.75, 4385.75],
    [4091.88, 4391.13],
    [4090.63, 4396.38],
    [4089.0, 4401.5],
    [4086.0, 4409.0],
    [4081.0, 4418.0],
    [4076.5, 4424.0],
    [4073.25, 4427.75],
    [4069.75, 4431.25],
    [4065.0, 4435.25],
    [4056.0, 4442.0],
    [4045.5, 4448.0],
    [4039.0, 4451.0],
    [4034.5, 4452.5],
    [4031.5, 4453.25],
    [4028.5, 4453.75],
    [4025.5, 4454.0],
    [4022.5, 4454.0],
    [4019.5, 4453.75],
    [4016.5, 4453.25],
    [4013.5, 4452.5],
    [4010.5, 4451.5],
    [4007.13, 4450.88],
    [4003.38, 4450.63],
    [3999.25, 4450.75],
    [3992.5, 4451.5],
    [3982.0, 4454.0],
    [3970.0, 4458.0],
    [3957.71, 4463.28],
    [3938.84, 4473.11],
    [3861.16, 4526.89],
    [3833.02, 4541.62],
    [3810.22, 4550.17],
    [3803.35, 4553.57],
    [3797.1, 4557.58],
    [3791.47, 4562.18],
    [3786.46, 4567.38],
    [3782.07, 4573.18],
    [3778.3, 4579.57],
    [3775.15, 4586.56],
    [3767.59, 4609.23],
    [3760.78, 4623.25],
    [3752.23, 4636.11],
    [3741.94, 4647.8],
    [3714.06, 4672.2],
    [3703.53, 4683.6],
    [3694.51, 4695.87],
    [3687.0, 4709.0],
    [3681.0, 4723.0],
    [3675.0, 4734.5],
    [3670.5, 4741.25],
    [3667.5, 4745.13],
    [3664.5, 4748.38],
    [3661.5, 4751.0],
    [3658.5, 4753.0],
    [3655.25, 4754.5],
    [3651.75, 4755.5],
    [3648.0, 4756.0],
    [3644.0, 4756.0],
    [3639.75, 4755.5],
    [3635.25, 4754.5],
    [3630.5, 4753.0],
    [3623.0, 4750.0],
    [3614.0, 4745.0],
    [3608.0, 4740.5],
    [3604.25, 4737.25],
    [3600.75, 4733.75],
    [3597.5, 4730.0],
    [3594.5, 4726.0],
    [3590.93, 4722.18],
    [3586.78, 4718.55],
    [3582.06, 4715.1],
    [3574.12, 4710.21],
    [3561.23, 4704.41],
    [3534.77, 4695.59],
    [3520.11, 4689.5],
    [3505.96, 4682.38],
    [3492.34, 4674.24],
    [3451.66, 4645.76],
    [3438.04, 4637.62],
    [3423.89, 4630.5],
    [3409.23, 4624.41],
    [3375.18, 4613.06],
    [3359.29, 4609.26],
    [3353.07, 4608.32],
    [3349.5, 4608.0],
    [3345.0, 4608.0],
    [3337.5, 4609.0],
    [3324.0, 4612.0],
    [3300.0, 4620.0],
    [3284.0, 4626.0],
    [3279.5, 4628.25],
    [3276.75, 4630.0],
    [3273.0, 4633.0],
    [3264.0, 4642.0],
    [3254.5, 4653.88],
    [3249.58, 4659.27],
    [3242.32, 4666.08],
    [3230.97, 4674.23],
    [3217.94, 4681.45],
    [3153.74, 4708.97],
    [3146.43, 4711.23],
    [3139.18, 4712.6],
    [3131.98, 4713.08],
    [3124.83, 4712.68],
    [3117.73, 4711.38],
    [3110.69, 4709.19],
    [3103.69, 4706.11],
    [3029.89, 4663.94],
    [3004.0, 4652.0],
    [2980.0, 4644.0],
    [2966.5, 4638.0],
    [2959.0, 4634.0],
    [2954.5, 4631.0],
    [2951.38, 4628.5],
    [2946.5, 4624.0],
    [2936.0, 4612.0],
    [2920.0, 4588.0],
    [2909.5, 4576.0],
    [2904.63, 4571.5],
    [2901.5, 4569.0],
    [2898.5, 4567.0],
    [2895.38, 4565.38],
    [2892.13, 4564.13],
    [2888.75, 4563.25],
    [2885.25, 4562.75],
    [2881.63, 4562.63],
    [2877.88, 4562.88],
    [2874.0, 4563.5],
    [2868.0, 4565.0],
    [2860.5, 4566.0],
    [2855.25, 4566.0],
    [2851.88, 4565.75],
    [2848.63, 4565.25],
    [2845.5, 4564.5],
    [2842.5, 4563.5],
    [2839.5, 4562.25],
    [2835.0, 4560.0],
    [2829.0, 4556.0],
    [2823.0, 4551.0],
    [2817.0, 4545.0],
    [2811.0, 4540.0],
    [2805.0, 4536.0],
    [2800.5, 4533.75],
    [2797.5, 4532.5],
    [2794.5, 4531.5],
    [2790.93, 4530.94],
    [2786.78, 4530.82],
    [2782.06, 4531.15],
    [2774.12, 4532.29],
    [2761.23, 4535.59],
    [2727.18, 4546.94],
    [2712.79, 4550.74],
    [2707.7, 4551.68],
    [2705.0, 4552.0],
    [2703.0, 4552.0],
    [2700.44, 4551.42],
    [2697.31, 4550.26],
    [2692.55, 4547.93],
    [2682.98, 4542.7],
    [2653.02, 4521.3],
    [2635.75, 4511.33],
    [2622.57, 4505.03],
    [2600.0, 4496.0],
];

const ROAD: [[f32; 2]; 74] = [
    [4208.0, 4000.0],
    [4216.21, 3975.38],
    [4217.93, 3968.03],
    [4218.86, 3960.92],
    [4218.99, 3954.05],
    [4218.31, 3947.42],
    [4216.96, 3940.63],
    [4214.95, 3933.68],
    [4212.27, 3926.58],
    [4187.73, 3873.42],
    [4185.09, 3866.11],
    [4183.15, 3858.75],
    [4181.92, 3851.33],
    [4181.39, 3843.87],
    [4181.57, 3836.36],
    [4182.45, 3828.79],
    [4184.05, 3821.18],
    [4193.1, 3790.99],
    [4198.87, 3776.23],
    [4205.8, 3762.05],
    [4213.89, 3748.43],
    [4362.11, 3539.57],
    [4372.45, 3527.67],
    [4383.86, 3516.9],
    [4396.35, 3507.28],
    [4544.26, 3414.84],
    [4550.94, 3411.5],
    [4557.53, 3409.07],
    [4564.02, 3407.55],
    [4570.41, 3406.93],
    [4576.71, 3407.22],
    [4582.9, 3408.41],
    [4589.0, 3410.5],
    [4598.0, 3415.0],
    [4609.66, 3422.49],
    [4620.97, 3431.47],
    [4631.94, 3441.94],
    [4669.37, 3484.04],
    [4692.0, 3504.0],
    [4716.0, 3520.0],
    [4729.5, 3527.5],
    [4737.0, 3531.0],
    [4743.0, 3533.0],
    [4750.87, 3536.65],
    [4765.5, 3544.59],
    [4907.75, 3635.11],
    [4918.88, 3641.06],
    [4925.44, 3643.53],
    [4929.22, 3644.51],
    [4932.41, 3644.84],
    [4935.0, 3644.5],
    [4937.0, 3643.5],
    [4938.5, 3642.13],
    [4939.5, 3640.38],
    [4940.0, 3638.25],
    [4940.0, 3635.75],
    [4939.5, 3632.88],
    [4938.5, 3629.63],
    [4937.0, 3626.0],
    [4934.0, 3620.0],
    [4929.0, 3612.5],
    [4923.0, 3605.5],
    [4916.0, 3599.0],
    [4908.0, 3593.0],
    [4899.5, 3585.5],
    [4886.0, 3572.0],
    [4866.0, 3548.0],
    [4848.84, 3521.69],
    [4800.0, 3424.0],
    [4760.88, 3365.31],
    [4745.85, 3337.23],
    [4712.0, 3256.0],
    [4664.85, 3114.56],
    [4663.09, 3107.03],
];

#[derive(Resource, Clone, Copy)]
pub(crate) struct Plateau(pub f32);

pub(crate) fn populate(
    world: &mut World,
    bodies: u32,
    hamlet: f32,
    rocks: &RockMaps,
) -> engine::Result<f32> {
    let half = SIZE * 0.5;
    let local = |point: [f32; 2]| [point[0] + half, point[1] + half];
    let plateau = local(PLATEAU);
    let pyramid = Pyramid::new([TILES, TILES], SPACING)?;
    let mountains = Layer::Noise(NoiseLayer {
        seed: SEED,
        height: 1500.0,
        wavelength: 3200.0,
        ridges: 0.5,
        warp: 0.4,
        erosion: 0.9,
        smoothing: 0.6,
        valleys: 0.7,
        roughness: 0.3,
        ..NoiseLayer::default()
    });
    let level = Layers::new(vec![mountains.clone()])?.height_at(&pyramid, Vec2::from(plateau));
    world.insert_resource(Plateau(level));
    let plateau_stamp = Layer::Stamp(Stamp {
        center: plateau,
        radius: hamlet,
        falloff: 220.0,
        op: StampOp::Flatten(level),
    });
    let cave_mouth = {
        let shaped = Layers::new(vec![mountains.clone(), plateau_stamp.clone()])?;
        steepest_hillside(&shaped, &pyramid, Vec2::from(plateau), hamlet)
    };
    let layers = Layers::new(vec![
        mountains,
        plateau_stamp.clone(),
        Layer::Erosion(ErosionStage {
            layer: ErosionLayer {
                level: 3,
                rounds: 20,
                incision: 0.003,
                debris: 0.006,
                creep: 6.0,
            },
            baked: None,
        }),
        Layer::Stamp(Stamp {
            center: local([1500.0, -1300.0]),
            radius: 520.0,
            falloff: 700.0,
            op: StampOp::Flatten(60.0),
        }),
        plateau_stamp,
        Layer::Hydrology(HydrologyStage {
            layer: HydrologyLayer {
                level: 0,
                discharge: 3.2,
            },
            baked: None,
        }),
        Layer::Path(PathStage {
            layer: Arc::new(PathLayer {
                points: ROAD.to_vec(),
                width: 6.0,
                shoulder: 6.0,
                smoothing: 80.0,
                bridge_fill: 4.0,
                tunnel_cut: 25.0,
            }),
            baked: None,
        }),
        Layer::Path(PathStage {
            layer: Arc::new(PathLayer {
                points: EAST_ROAD.map(local).to_vec(),
                width: 6.0,
                shoulder: 6.0,
                smoothing: 600.0,
                bridge_fill: 4.0,
                tunnel_cut: 25.0,
            }),
            baked: None,
        }),
        Layer::Path(PathStage {
            layer: Arc::new(PathLayer {
                points: WEST_ROAD.to_vec(),
                width: 6.0,
                shoulder: 6.0,
                smoothing: 80.0,
                bridge_fill: 4.0,
                tunnel_cut: 25.0,
            }),
            baked: None,
        }),
        Layer::Stamp(Stamp {
            center: cave_mouth,
            radius: CAVE_MOUTH,
            falloff: 0.0,
            op: StampOp::Hole,
        }),
    ])?;
    world.insert_resource(CaveMouth(Vec2::new(
        cave_mouth[0] - half,
        cave_mouth[1] - half,
    )));
    let roads = [
        ROAD.to_vec(),
        WEST_ROAD.to_vec(),
        EAST_ROAD.map(local).to_vec(),
    ];
    let surface = surface(world, rocks, plateau_ring(plateau, hamlet - 8.0, &roads))?;
    world.spawn(
        Terrain::generated(Vec3::new(-half, 0.0, -half), pyramid, layers)
            .with_surface(surface)
            .with_surroundings(Surroundings {
                reach: SURROUNDINGS,
                blend: 256.0,
            }),
    );
    lanterns(world, &ROAD, half)?;
    hiker::hiker(world, &ROAD, half)?;
    bodies::request(world, bodies)?;
    Ok(level)
}

const MOSS: TopLayer = TopLayer {
    color: [0.055, 0.07, 0.03],
    coverage: 0.36,
    roughness: 0.9,
};
const LICHEN: TopLayer = TopLayer {
    color: [0.11, 0.12, 0.05],
    coverage: 0.34,
    roughness: 0.85,
};

const SOIL: Grounding = Grounding { height: 0.12 };
const ROOTS: Grounding = Grounding { height: 0.25 };

const LANTERN_SPACING: f32 = 40.0;
const LANTERN_SIDE: f32 = 5.0;
const LANTERN_LIGHT: ([f32; 3], f32, f32) = ([1.0, 0.66, 0.34], 14.0, 12.0);

struct Lantern {
    position: Vec3,
    yaw: f32,
    cube: MeshHandle,
    wood: MaterialHandle,
    glow: MaterialHandle,
}

impl Streamed for Lantern {
    fn entities(&self) -> u32 {
        3
    }

    fn spawn(&self, world: &mut World, spawned: &mut Vec<Entity>) {
        let root = world
            .spawn((
                Transform {
                    translation: self.position,
                    rotation: Quat::from_rotation_y(self.yaw),
                    ..Transform::default()
                },
                Grounded::default(),
            ))
            .id();
        let post = world
            .spawn((
                Transform {
                    translation: Vec3::new(0.0, 1.6, 0.0),
                    scale: Vec3::new(0.14, 3.2, 0.14),
                    ..Transform::default()
                },
                Mesh3d {
                    mesh: self.cube,
                    material: self.wood,
                },
            ))
            .id();
        let ([red, green, blue], intensity, range) = LANTERN_LIGHT;
        let lamp = world
            .spawn((
                Transform {
                    translation: Vec3::new(0.0, 3.35, 0.0),
                    scale: Vec3::splat(0.3),
                    ..Transform::default()
                },
                Mesh3d {
                    mesh: self.cube,
                    material: self.glow,
                },
                LocalLight::point(Vec3::new(red, green, blue), intensity, range),
            ))
            .id();
        if set_parents(world, &[(post, Some(root)), (lamp, Some(root))]).is_err() {
            for entity in [root, post, lamp] {
                let _ = world.try_despawn(entity);
            }
            return;
        }
        spawned.extend([root, post, lamp]);
    }
}

fn lanterns(world: &mut World, road: &[[f32; 2]], half: f32) -> engine::Result<()> {
    let mut assets = world.resource_mut::<AssetLibrary>();
    let cube = assets.insert_mesh(Mesh::cube())?;
    let wood = assets.insert_material(Material::new([0.09, 0.06, 0.04, 1.0])?)?;
    let glow =
        assets.insert_material(Material::new([1.0, 0.72, 0.42, 1.0]).and_then(|material| {
            material.with_surface(Surface {
                emission: 3.0,
                ..Surface::MATTE
            })
        })?)?;
    let mut items = Vec::new();
    let mut next = LANTERN_SPACING * 0.5;
    let mut walked = 0.0;
    for pair in road.windows(2) {
        let (from, to) = (Vec2::from(pair[0]), Vec2::from(pair[1]));
        let length = from.distance(to);
        if length <= 0.0 {
            continue;
        }
        let along = (to - from) / length;
        while next <= walked + length {
            let point = from + along * (next - walked) + along.perp() * LANTERN_SIDE;
            items.push(StreamedItem {
                id: Some(format!("highlands/lantern{}", items.len()).into()),
                position: Vec3::new(point.x - half, 0.0, point.y - half),
                reach: LANTERN_LIGHT.2,
                content: Arc::new(Lantern {
                    position: Vec3::new(point.x - half, 0.0, point.y - half),
                    yaw: -along.y.atan2(along.x),
                    cube,
                    wood,
                    glow,
                }),
            });
            next += LANTERN_SPACING;
        }
        walked += length;
    }
    world.resource_mut::<WorldCells>().add(items);
    Ok(())
}

fn plateau_ring(center: [f32; 2], radius: f32, roads: &[Vec<[f32; 2]>]) -> Vec<AuthoredPlacement> {
    let distance = |point: Vec2, from: Vec2, to: Vec2| {
        let along = to - from;
        let t = math::clamp(
            (point - from).dot(along) / math::max(along.length_squared(), 1.0e-6),
            0.0,
            1.0,
        );
        point.distance(from + along * t)
    };
    let count = (std::f32::consts::TAU * radius / 7.0).round() as usize;
    (0..count)
        .filter_map(|index| {
            let angle = index as f32 / count as f32 * std::f32::consts::TAU;
            let (sin, cos) = math::sin_cos(angle);
            let point = Vec2::new(center[0] + cos * radius, center[1] + sin * radius);
            let near_road = roads.iter().any(|road| {
                road.windows(2)
                    .any(|pair| distance(point, Vec2::from(pair[0]), Vec2::from(pair[1])) < 12.0)
            });
            (!near_road).then(|| AuthoredPlacement {
                asset: format!("boulder{}", index % BOULDERS),
                position: point.to_array(),
                yaw: 90.0 - angle.to_degrees(),
                scale: 2.0 + 1.2 * ((index * 7919) % 13) as f32 / 12.0,
                sink: 0.0,
                align: 0.3,
                level: 0,
            })
        })
        .collect()
}

fn surface(
    world: &mut World,
    files: &RockMaps,
    authored: Vec<AuthoredPlacement>,
) -> engine::Result<TerrainSurface> {
    let boulders = (0..BOULDERS as u32)
        .map(|seed| rocks::boulder_record(world, seed))
        .collect::<engine::Result<Vec<_>>>()?;
    let atlas = trees::atlas_record(world)?;
    let needles = {
        let material = trees::material(atlas)?.with_grounding(ROOTS)?;
        let mut assets = world.resource_mut::<AssetLibrary>();
        assets.insert_material(material)?
    };
    let spruces = (0..SPRUCES as u32)
        .map(|seed| trees::spruce(world, seed, needles))
        .collect::<engine::Result<Vec<_>>>()?;
    let materials = world.resource_scope(|world, mut records: Mut<Records>| {
        let mut assets = world.resource_mut::<AssetLibrary>();
        let mut loaded = HashMap::default();
        LOOKS
            .iter()
            .map(|look| {
                Ok(SurfaceMaterial {
                    material: Some(maps((&mut records, &mut assets), look, &mut loaded)?),
                    color: look.color,
                    roughness: look.roughness,
                    tiling: look.tiling,
                    height_contrast: look.height_contrast,
                    projection: look.projection,
                    cover: look.cover,
                    depth: look.depth,
                })
            })
            .collect::<engine::Result<Vec<_>>>()
    })?;
    let mut assets = world.resource_mut::<AssetLibrary>();
    let terrain_rock = materials[4]
        .material
        .and_then(|rock| assets.material(rock).copied());
    let boulder_scan = scanned(&mut assets, files, "boulder", BOULDER_SCAN_TILING, MOSS)?;
    let material = match (boulder_scan, terrain_rock) {
        (Some(scan), _) => scan,
        (None, Some(terrain_rock)) => {
            in_terrain_rock(terrain_rock, ([0.8, 0.77, 0.72], 0.85), 3.0, MOSS)?
        }
        (None, None) => Material::new([0.3, 0.28, 0.26, 1.0])
            .and_then(|material| {
                material.with_surface(Surface {
                    roughness: 0.85,
                    ..Surface::MATTE
                })
            })
            .and_then(|material| material.with_top_layer(MOSS))?,
    };
    let rock = material
        .with_grounding(SOIL)
        .and_then(|material| assets.insert_material(material))?;
    let mut placed = Vec::with_capacity(BOULDERS + SPRUCES);
    for &mesh in &boulders {
        placed.push(PlacementAsset {
            mesh,
            material: rock,
            lods: Vec::new(),
            parts: Vec::new(),
            sway: 0.0,
            collider: Some(PlacedCollider::new(
                PlacedShape::Sphere { radius: 0.42 },
                [0.0; 3],
            )),
            shadow: None,
        });
    }
    for spruce in spruces {
        placed.push(PlacementAsset {
            sway: 0.35,
            collider: Some(PlacedCollider::new(
                PlacedShape::Cuboid {
                    half_extents: [0.28, 2.0, 0.28],
                },
                [0.0, 1.5, 0.0],
            )),
            ..spruce
        });
    }
    let wall_scan = scanned(&mut assets, files, "cliff", WALL_SCAN_TILING, LICHEN)?;
    let wall_rock = match (wall_scan, terrain_rock) {
        (Some(scan), _) => assets.insert_material(scan)?,
        (None, Some(terrain_rock)) => {
            let material = in_terrain_rock(terrain_rock, (LOOKS[4].tint, 0.9), 4.0, LICHEN)?;
            assets.insert_material(material)?
        }
        (None, None) => rock,
    };
    let wall = assets
        .material(wall_rock)
        .copied()
        .map(|material| material.with_double_sided(true))
        .ok_or_else(|| engine::Error::game_rule("the cave's rock is resident"))?;
    let wall = assets.insert_material(wall)?;
    world.insert_resource(CaveRock {
        boulders,
        material: wall_rock,
        wall,
    });
    TerrainSurface::new(
        &Biomes {
            placed: authored,
            ..rules()
        },
        materials,
    )
    .and_then(|surface| surface.with_assets(placed))
    .map(|surface| TerrainSurface {
        bridge: Some(rock),
        ..surface
    })
    .map_err(engine::Error::from)
}

#[derive(Resource)]
struct CaveMouth(Vec2);
const CAVE_MOUTH: f32 = 3.5;

#[derive(Resource, Clone)]
struct CaveRock {
    boulders: Vec<MeshHandle>,
    material: MaterialHandle,
    wall: MaterialHandle,
}

#[derive(Resource)]
struct CaveOpened;

const CAVE_RADIUS: [f32; 2] = [6.0, 4.5];

const CAVE_MARGIN: f32 = 1.5;

fn steepest_hillside(layers: &Layers, pyramid: &Pyramid, center: Vec2, inner: f32) -> [f32; 2] {
    let height = |point: Vec2| layers.height_at(pyramid, point);
    let mut best = (f32::INFINITY, center + Vec2::X * (inner + 210.0));
    for step in 0..36 {
        let (sin, cos) = math::sin_cos(step as f32 * std::f32::consts::TAU / 36.0);
        for ring in 0..16 {
            let point = center + Vec2::new(cos, sin) * (inner + 150.0 + 30.0 * ring as f32);
            let gradient = Vec2::new(
                height(point + Vec2::X * 20.0) - height(point - Vec2::X * 20.0),
                height(point + Vec2::Y * 20.0) - height(point - Vec2::Y * 20.0),
            ) / 40.0;
            let miss = (gradient.length() - 0.58).abs();
            if miss < best.0 {
                best = (miss, point);
            }
        }
    }
    best.1.to_array()
}

pub(crate) fn add_systems(app: &mut App) {
    app.add_systems(Presentation, (open_cave, bodies::place))
        .add_systems(FixedSimulation, hiker::hike.after(NavSystems::Walk));
}

fn open_cave(world: &mut World) {
    if world.contains_resource::<CaveOpened>() {
        return;
    }
    let (Some(mouth), Some(rock)) = (
        world.get_resource::<CaveMouth>().map(|mouth| mouth.0),
        world.get_resource::<CaveRock>().cloned(),
    ) else {
        return;
    };
    let Some((terrain, layers, bake)) = ({
        let mut query = world.query::<(Entity, &Terrain)>();
        let terrains = world.resource::<Terrains>();
        query.iter(world).find_map(|(entity, terrain)| {
            Some((
                terrain.clone(),
                terrains.layers(entity)?,
                terrains.bake_record(entity)?,
            ))
        })
    }) else {
        return;
    };
    if layers.hydrology().is_none() {
        return;
    }
    let height = |point: Vec2| {
        terrain.origin.y
            + layers.height_at(
                &terrain.pyramid,
                Vec2::new(point.x - terrain.origin.x, point.y - terrain.origin.z),
            )
    };
    let gradient = Vec2::new(
        height(mouth + Vec2::X * 20.0) - height(mouth - Vec2::X * 20.0),
        height(mouth + Vec2::Y * 20.0) - height(mouth - Vec2::Y * 20.0),
    ) / 40.0;
    let slope = gradient.length();
    let uphill = if slope > 0.02 {
        gradient / slope
    } else {
        Vec2::X
    };
    let side = Vec2::new(-uphill.y, uphill.x);
    let start = mouth - uphill * (CAVE_MOUTH + 1.0);
    let center = height(start) - 0.5 + 0.55 * CAVE_RADIUS[1] * 1.15;
    let fall = math::max(0.75 - slope, 0.25);
    let at = |along: f32, across: f32, drop: f32| {
        let point = mouth + uphill * along + side * across;
        Vec3::new(point.x, center - drop, point.y)
    };
    let points = [
        at(-(CAVE_MOUTH + 1.0), 0.0, 0.0),
        at(12.0, 0.0, fall * 18.0),
        at(28.0, 6.0, fall * 18.0 + fall * 8.0),
        at(44.0, -5.0, fall * 18.0 + fall * 16.0),
        at(58.0, 0.0, fall * 18.0 + fall * 22.0),
    ];
    let facing = Quat::from_rotation_y((-uphill.x).atan2(-uphill.y));
    let framing: Vec<(Transform, MeshHandle)> = [
        (4.0, 0.0, 1.5, 10.0, 0),
        (-1.0, 10.0, -1.5, 7.0, 1),
        (-1.0, -10.0, -1.5, 7.0, 2),
    ]
    .into_iter()
    .filter_map(
        |(along, across, rise, scale, index): (f32, f32, f32, f32, usize)| {
            let point = mouth + uphill * along + side * across;
            let mesh = *rock.boulders.get(index % rock.boulders.len().max(1))?;
            Some((
                Transform {
                    translation: Vec3::new(point.x, height(point) + rise, point.y),
                    rotation: facing,
                    scale: Vec3::splat(scale),
                },
                mesh,
            ))
        },
    )
    .collect();
    let inward = Vec3::new(uphill.x, 0.0, uphill.y);
    let mut identity = bake.hash.to_vec();
    identity.extend_from_slice(&mouth.x.to_bits().to_le_bytes());
    identity.extend_from_slice(&mouth.y.to_bits().to_le_bytes());
    world.insert_resource(CaveOpened);
    let spawn = move |world: &mut World, mesh: MeshHandle, built: &Mesh| {
        spawn_cave(world, (mesh, built), &points, inward, (&rock, framing));
        Ok(())
    };
    let made = records::mesh_then(
        world,
        "highlands_cave",
        &identity,
        move || caves::cave(&points, CAVE_RADIUS, 79),
        spawn,
    );
    if let Err(error) = made {
        report_failure("highlands_cave_failed", &error);
    }
}

fn report_failure(event: &str, error: &engine::Error) {
    let record = serde_json::json!({
        "event": event,
        "code": error.code,
        "path": error.path,
        "message": error.message,
    });
    eprintln!("{record}");
}

fn spawn_cave(
    world: &mut World,
    (mesh, built): (MeshHandle, &Mesh),
    points: &[Vec3; 5],
    inward: Vec3,
    (rock, framing): (&CaveRock, Vec<(Transform, MeshHandle)>),
) {
    let (low, high) = cave_interior(built, points, inward);
    world.spawn((
        Transform {
            translation: points[0],
            ..Transform::default()
        },
        Mesh3d {
            mesh,
            material: rock.wall,
        },
    ));
    let chamber = points[4] - inward * 6.0 + Vec3::Y * 2.0;
    world.spawn((
        Transform {
            translation: chamber,
            ..Transform::default()
        },
        LocalLight::point(Vec3::new(1.0, 0.62, 0.3), 40.0, 26.0).with_region(LightRegion::Interior),
    ));
    world.spawn((
        Transform {
            translation: (low + high) * 0.5,
            ..Transform::default()
        },
        ReflectionProbe::new((high - low) * 0.5)
            .with_blend(CAVE_MARGIN)
            .with_interior(Interior { sky: 1.0, sun: 1.0 }),
    ));
    for (transform, mesh) in framing {
        world.spawn((
            transform,
            Mesh3d {
                mesh,
                material: rock.material,
            },
        ));
    }
}

fn cave_interior(mesh: &Mesh, points: &[Vec3], inward: Vec3) -> (Vec3, Vec3) {
    let bend = (points[1] - points[0]).dot(inward);
    let (low, high) = mesh
        .vertices()
        .iter()
        .map(|vertex| points[0] + Vec3::from(vertex.position))
        .filter(|&point| (point - points[0]).dot(inward) >= bend)
        .fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(low, high), point| (low.min(point), high.max(point)),
        );
    (
        low - Vec3::splat(CAVE_MARGIN),
        high + Vec3::splat(CAVE_MARGIN),
    )
}

const BOULDERS: usize = 5;
const SPRUCES: usize = 3;

fn maps(
    (records, assets): (&mut Records, &mut AssetLibrary),
    look: &Look,
    loaded: &mut HashMap<&'static str, Vec<TextureHandle>>,
) -> engine::Result<MaterialHandle> {
    let name = look.maps;
    let textures = match loaded.get(name) {
        Some(textures) => textures.clone(),
        None => {
            let maps = [
                ("albedo", TextureUsage::Color),
                ("normal", TextureUsage::Normal),
                ("orm", TextureUsage::Data),
            ];
            let set = format!("highlands/{name}");
            let textures =
                crate::records::textures((records, &mut *assets), set, &maps, move || {
                    ground::sources(name)
                })?;
            loaded.insert(name, textures.clone());
            textures
        }
    };
    let [red, green, blue] = look.tint;
    let material = Material::new([red, green, blue, 1.0])
        .and_then(|material| {
            material.with_surface(Surface {
                roughness: 1.0,
                ..Surface::MATTE
            })
        })
        .and_then(|material| material.with_base_color_texture(TextureRef::new(textures[0])))
        .and_then(|material| material.with_normal_texture(TextureRef::new(textures[1]), 1.0))
        .and_then(|material| material.with_metallic_roughness_texture(TextureRef::new(textures[2])))
        .and_then(|material| material.with_occlusion_texture(TextureRef::new(textures[2]), 1.0))?;
    assets
        .insert_material(material)
        .map_err(engine::Error::from)
}

fn scanned(
    assets: &mut AssetLibrary,
    files: &RockMaps,
    name: &str,
    tiling: f32,
    top: TopLayer,
) -> engine::Result<Option<Material>> {
    let mut textures = Vec::with_capacity(3);
    for map in ["albedo", "normal", "orm"] {
        let file_name = format!("rocks/{name}_{map}.ktx2");
        let Some(file) = files.get(&file_name) else {
            return Ok(None);
        };
        let texture = Texture::from_rgba(file.width, file.height, file.pixels.clone())
            .map_err(|error| engine::Error::from(error).context(&file_name))?;
        textures.push(assets.insert_texture(texture)?);
    }
    Material::new([1.0; 4])
        .and_then(|material| {
            material.with_surface(Surface {
                roughness: 1.0,
                ..Surface::MATTE
            })
        })
        .and_then(|material| material.with_base_color_texture(TextureRef::new(textures[0])))
        .and_then(|material| material.with_normal_texture(TextureRef::new(textures[1]), 1.0))
        .and_then(|material| material.with_metallic_roughness_texture(TextureRef::new(textures[2])))
        .and_then(|material| material.with_occlusion_texture(TextureRef::new(textures[2]), 1.0))
        .and_then(|material| material.with_world_mapping(tiling))
        .and_then(|material| material.with_top_layer(top))
        .map(Some)
        .map_err(engine::Error::from)
}

fn in_terrain_rock(
    terrain_rock: Material,
    (color, roughness): ([f32; 3], f32),
    tiling: f32,
    top: TopLayer,
) -> engine::Result<Material> {
    let [red, green, blue] = color;
    let mut material = Material::new([red, green, blue, 1.0]).and_then(|material| {
        material.with_surface(Surface {
            roughness,
            ..Surface::MATTE
        })
    })?;
    if let Some(map) = terrain_rock.base_color_texture() {
        material = material.with_base_color_texture(map)?;
    }
    if let Some(map) = terrain_rock.normal_texture() {
        material = material.with_normal_texture(map, 0.8)?;
    }
    if let Some(map) = terrain_rock.metallic_roughness_texture() {
        material = material
            .with_metallic_roughness_texture(map)
            .and_then(|material| material.with_occlusion_texture(map, 1.0))?;
    }
    material
        .with_world_mapping(tiling)
        .and_then(|material| material.with_top_layer(top))
        .map_err(engine::Error::from)
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn rules() -> Biomes {
    let forest = || {
        Mask::Multiply(vec![
            remap(
                Mask::Add(vec![
                    scaled(noise(320.0, 31, 3), 0.75),
                    scaled(noise(45.0, 32, 2), 0.25),
                ]),
                0.45,
                0.62,
            ),
            Mask::Invert(Box::new(Mask::Height([560.0, 640.0]))),
            Mask::Invert(Box::new(Mask::Slope([24.0, 34.0]))),
            Mask::Invert(Box::new(Mask::Flow([4.95, 5.15]))),
            Mask::Invert(Box::new(Mask::Lake([0.1, 0.6]))),
            Mask::Invert(Box::new(Mask::Path([10.0, 4.0]))),
        ])
    };
    Biomes {
        assets: (0..BOULDERS)
            .map(|index| format!("boulder{index}"))
            .chain((0..SPRUCES).map(|index| format!("spruce{index}")))
            .collect(),
        placed: Vec::new(),
        placements: [
            PlacementLayer {
                id: "spruces".into(),
                assets: (0..SPRUCES)
                    .map(|index| PlacementChoice {
                        asset: format!("spruce{index}"),
                        weight: 1.0,
                    })
                    .collect(),
                density: 420.0,
                footprint: 3.5,
                materials: vec![
                    "grass".into(),
                    "dry_grass".into(),
                    "dirt".into(),
                    "forest_floor".into(),
                ],
                mask: forest(),
                scale: [0.65, 1.15],
                align: 0.04,
                downhill: 0.0,
                sink: 0.3,
                bed: 0.0,
                level: 3,
                far: Some(FarPlacement { level: 4 }),
            },
            PlacementLayer {
                id: "boulders".into(),
                assets: boulder_choices(),
                density: 18.0,
                footprint: 7.0,
                materials: vec!["grass".into(), "dry_grass".into(), "dirt".into()],
                mask: Mask::Multiply(vec![
                    remap(noise(90.0, 21, 2), 0.45, 0.75),
                    Mask::Invert(Box::new(Mask::Path([6.0, 2.0]))),
                ]),
                scale: [0.6, 2.4],
                align: 0.3,
                downhill: 0.0,
                sink: 0.1,
                bed: 0.45,
                level: 0,
                far: Some(FarPlacement { level: 2 }),
            },
            PlacementLayer {
                id: "stones".into(),
                assets: boulder_choices(),
                density: 450.0,
                footprint: 1.5,
                materials: vec!["rock".into()],
                mask: Mask::one(),
                scale: [0.12, 0.45],
                align: 0.8,
                downhill: 0.0,
                sink: 0.08,
                bed: 0.0,
                level: 0,
                far: Some(FarPlacement { level: 2 }),
            },
            PlacementLayer {
                id: "outcrops".into(),
                assets: boulder_choices(),
                density: 25.0,
                footprint: 8.0,
                materials: vec!["rock".into()],
                mask: Mask::one(),
                scale: [2.0, 5.0],
                align: 0.6,
                downhill: 0.0,
                sink: 0.5,
                bed: 0.0,
                level: 0,
                far: Some(FarPlacement { level: 2 }),
            },
        ]
        .into_iter()
        .chain(stream_rocks())
        .collect(),
        materials: LOOKS.iter().map(|look| look.name.to_owned()).collect(),
        biomes: vec![Biome {
            id: "highlands".into(),
            fitness: Mask::one(),
            layers: [
                layer("grass", Mask::one()),
                layer(
                    "dry_grass",
                    remap(
                        Mask::Add(vec![
                            scaled(noise(160.0, 11, 3), 0.6),
                            scaled(noise(28.0, 12, 2), 0.25),
                            scaled(noise(5.0, 16, 1), 0.15),
                        ]),
                        0.48,
                        0.72,
                    ),
                ),
                layer(
                    "dirt",
                    remap(
                        Mask::Add(vec![
                            scaled(noise(60.0, 13, 2), 0.6),
                            scaled(noise(9.0, 14, 2), 0.4),
                        ]),
                        0.6,
                        0.74,
                    ),
                ),
                layer("forest_floor", remap(forest(), 0.12, 0.42)),
                layer(
                    "wet_grass",
                    remap(
                        Mask::Add(vec![
                            Mask::Wetness([10.0, 13.5]),
                            scaled(noise(24.0, 41, 2), 0.35),
                            Mask::Constant(-0.15),
                        ]),
                        0.35,
                        0.6,
                    ),
                ),
                layer(
                    "silt",
                    remap(
                        Mask::Add(vec![
                            Mask::Multiply(vec![Mask::Deposit(LAID), Mask::River(FLOODED)]),
                            scaled(noise(4.0, 85, 1), 0.4),
                            Mask::Constant(-0.2),
                        ]),
                        0.45,
                        0.55,
                    ),
                ),
            ]
            .into_iter()
            .chain(stream_beds())
            .chain([
                layer("road", Mask::Path([1.0, -0.5])),
                layer(
                    "snow",
                    remap(
                        Mask::Add(vec![
                            Mask::Height([700.0, 900.0]),
                            scaled(noise(60.0, 3, 2), 0.4),
                            Mask::Constant(-0.2),
                        ]),
                        0.4,
                        0.6,
                    ),
                ),
            ])
            .collect(),
            wetness: Some(wetness()),
        }],
        steep: Some(engine::terrain::data::SteepLayer {
            material: "rock".into(),
            slope: [35.0, 40.0],
        }),
        macro_colors: engine::terrain::data::MacroCauses::default(),
    }
}

const GRAINS: [f32; 3] = [0.002, 0.064, 0.256];
const GRAIN_BLEND: f32 = 0.2;
const BOULDER_POWER: [f32; 2] = [150.0, 400.0];
const SPRAY_POWER: [f32; 2] = [200.0, 2000.0];
const SATURATED: [f32; 2] = [11.5, 14.0];
const DAMP: f32 = 0.5;
const LAID: [f32; 2] = [0.5, 2.0];
const FLOODED: [f32; 2] = [6.0, 1.0];

fn stream_beds() -> [BiomeLayer; 4] {
    let bed = || {
        remap(
            Mask::Add(vec![
                Mask::River([1.5, 0.0]),
                scaled(noise(12.0, 42, 2), 0.3),
            ]),
            0.55,
            0.75,
        )
    };
    let short_of = |grain: f32| {
        Mask::Invert(Box::new(Mask::Grain([
            grain * (1.0 + GRAIN_BLEND),
            grain * (1.0 + 2.0 * GRAIN_BLEND),
        ])))
    };
    let [gravel, cobbles, boulders] = GRAINS;
    [
        layer("sand", Mask::Multiply(vec![bed(), short_of(gravel)])),
        layer(
            "gravel",
            Mask::Multiply(vec![bed(), coarser_than(gravel), short_of(cobbles)]),
        ),
        layer(
            "cobble",
            Mask::Multiply(vec![bed(), coarser_than(cobbles), short_of(boulders)]),
        ),
        layer("rock", Mask::Multiply(vec![bed(), coarser_than(boulders)])),
    ]
}

fn coarser_than(grain: f32) -> Mask {
    Mask::Grain([grain * (1.0 - GRAIN_BLEND), grain * (1.0 + GRAIN_BLEND)])
}

fn wetness() -> Mask {
    Mask::Max(vec![
        scaled(
            remap(
                Mask::Add(vec![
                    Mask::Wetness(SATURATED),
                    scaled(noise(6.0, 71, 2), 0.3),
                    Mask::Constant(-0.15),
                ]),
                0.2,
                0.8,
            ),
            DAMP,
        ),
        remap(
            Mask::Add(vec![
                Mask::Power(SPRAY_POWER),
                Mask::River([5.0, 0.0]),
                scaled(noise(2.0, 72, 2), 0.25),
                Mask::Constant(-0.12),
            ]),
            1.1,
            1.4,
        ),
    ])
}

fn stream_rocks() -> [PlacementLayer; 2] {
    [
        PlacementLayer {
            id: "channel_boulders".into(),
            assets: boulder_choices(),
            density: 300.0,
            footprint: 3.0,
            materials: vec!["gravel".into(), "cobble".into(), "rock".into()],
            mask: Mask::Multiply(vec![
                Mask::River([-0.5, -2.0]),
                Mask::Power(BOULDER_POWER),
                remap(noise(14.0, 61, 2), 0.35, 0.7),
            ]),
            scale: [0.6, 1.6],
            align: 0.7,
            downhill: 0.0,
            sink: 0.3,
            bed: 0.3,
            level: 0,
            far: Some(FarPlacement { level: 2 }),
        },
        PlacementLayer {
            id: "channel_stones".into(),
            assets: boulder_choices(),
            density: 1200.0,
            footprint: 1.5,
            materials: vec!["cobble".into(), "rock".into()],
            mask: Mask::Multiply(vec![
                Mask::River([0.0, -0.75]),
                coarser_than(GRAINS[1]),
                remap(noise(8.0, 63, 2), 0.3, 0.65),
            ]),
            scale: [0.3, 1.0],
            align: 0.7,
            downhill: 0.0,
            sink: 0.1,
            bed: 0.3,
            level: 0,
            far: Some(FarPlacement { level: 2 }),
        },
    ]
}

fn boulder_choices() -> Vec<PlacementChoice> {
    (0..BOULDERS)
        .map(|index| PlacementChoice {
            asset: format!("boulder{index}"),
            weight: 1.0,
        })
        .collect()
}

fn noise(wavelength: f32, seed: u32, octaves: u32) -> Mask {
    Mask::Noise(NoiseMask {
        wavelength,
        seed,
        octaves,
    })
}

fn scaled(mask: Mask, by: f32) -> Mask {
    Mask::Multiply(vec![mask, Mask::Constant(by)])
}

fn remap(mask: Mask, from: f32, to: f32) -> Mask {
    Mask::Remap(Remap {
        mask: Box::new(mask),
        from,
        to,
    })
}

fn layer(material: &str, mask: Mask) -> BiomeLayer {
    BiomeLayer {
        material: material.into(),
        mask,
        opacity: 1.0,
    }
}
