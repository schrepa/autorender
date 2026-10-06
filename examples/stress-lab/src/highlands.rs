//! Highlands (decision 051): an 8 km terrain the engine generates from a
//! layer stack (eroded mountains, then hand-placed stamps: a plateau
//! leveled for a hamlet, a flat valley floor and a cave mouth cut as a
//! hole), streamed around the camera, with the scene's renderables standing
//! on the plateau as the tower field does. Its ground (decision 054) is
//! meadow, dry grass, dirt, scree, rock and snow, each textured by maps
//! built offline (`cargo run -p engine_assetc --release --bin
//! terrain_materials`) and fetched by the page, or in plain colors without
//! them. Boulders lie on its meadows, stones on its scree and outcrops on
//! its rock, and spruce forests cover the lower slopes (decision 057).
//! Gravel roads (decision 069) wind from the plateau down to the crater
//! lake, west over a river on a stone bridge (decision 072) and east
//! through a ridge in a tunnel (decision 073).

use std::collections::HashMap;
use std::sync::Arc;

use engine::prelude::*;

use crate::{StressConfig, StressFile};

/// The images the page fetches for this scene, from `stress/terrain/`:
/// each surface material's base color, normal, and occlusion, roughness,
/// metalness and height maps, which `tools/assetc`'s `terrain_materials`
/// writes.
pub const FILES: [&str; 18] = [
    "grass_albedo.ktx2",
    "grass_normal.ktx2",
    "grass_orm.ktx2",
    "dry_grass_albedo.ktx2",
    "dry_grass_normal.ktx2",
    "dry_grass_orm.ktx2",
    "dirt_albedo.ktx2",
    "dirt_normal.ktx2",
    "dirt_orm.ktx2",
    "scree_albedo.ktx2",
    "scree_normal.ktx2",
    "scree_orm.ktx2",
    "rock_albedo.ktx2",
    "rock_normal.ktx2",
    "rock_orm.ktx2",
    "snow_albedo.ktx2",
    "snow_normal.ktx2",
    "snow_orm.ktx2",
];

/// How one surface material looks: its name; the material whose maps it
/// uses (whose files start with that name, its own unless it borrows
/// another's); its linear color and roughness without maps, and the color
/// its maps' colors are multiplied by; the meters its maps repeat over; how
/// sharply its heights win where it blends; how it lies on steep ground;
/// and the cover growing where it lies (decision 056).
struct Look {
    name: &'static str,
    maps: &'static str,
    color: [f32; 3],
    roughness: f32,
    tint: [f32; 3],
    tiling: f32,
    height_contrast: f32,
    projection: SurfaceProjection,
    cover: Option<Cover>,
}

const LOOKS: [Look; 10] = [
    Look {
        name: "grass",
        maps: "grass",
        color: [0.105, 0.135, 0.045],
        roughness: 0.92,
        tint: [1.0, 1.0, 1.0],
        tiling: 3.0,
        height_contrast: 0.35,
        projection: SurfaceProjection::Top,
        cover: Some(Cover {
            density: 140.0,
            height: [0.1, 0.34],
            width: 0.035,
            color: [0.16, 0.23, 0.06],
            sway: 0.6,
        }),
    },
    Look {
        name: "dry_grass",
        maps: "dry_grass",
        color: [0.2, 0.16, 0.08],
        roughness: 0.92,
        tint: [0.9, 0.9, 0.9],
        tiling: 3.5,
        height_contrast: 0.35,
        projection: SurfaceProjection::Top,
        cover: Some(Cover {
            density: 70.0,
            height: [0.14, 0.46],
            width: 0.025,
            color: [0.34, 0.27, 0.12],
            sway: 0.7,
        }),
    },
    Look {
        name: "dirt",
        maps: "dirt",
        color: [0.21, 0.165, 0.11],
        roughness: 0.92,
        tint: [1.0, 1.0, 1.0],
        tiling: 3.0,
        height_contrast: 0.5,
        projection: SurfaceProjection::Top,
        cover: None,
    },
    Look {
        name: "scree",
        maps: "scree",
        color: [0.2, 0.19, 0.18],
        roughness: 0.82,
        tint: [0.78, 0.74, 0.68],
        tiling: 4.0,
        height_contrast: 0.8,
        projection: SurfaceProjection::Top,
        cover: None,
    },
    Look {
        name: "rock",
        maps: "rock",
        color: [0.25, 0.235, 0.22],
        roughness: 0.82,
        tint: [0.74, 0.7, 0.64],
        tiling: 8.0,
        height_contrast: 0.7,
        projection: SurfaceProjection::Biplanar,
        cover: None,
    },
    Look {
        name: "snow",
        maps: "snow",
        color: [0.82, 0.84, 0.88],
        roughness: 0.55,
        tint: [1.0, 1.0, 1.0],
        tiling: 6.0,
        height_contrast: 0.4,
        projection: SurfaceProjection::Top,
        cover: None,
    },
    // Needle litter under the spruces: the dirt's maps, darkened toward
    // olive, so forests read as dark stands from afar.
    Look {
        name: "forest_floor",
        maps: "dirt",
        color: [0.03, 0.036, 0.022],
        roughness: 0.95,
        tint: [0.18, 0.23, 0.15],
        tiling: 3.0,
        height_contrast: 0.5,
        projection: SurfaceProjection::Top,
        cover: None,
    },
    // Grass where water gathers (decision 066): deeper green, taller and
    // denser, in hollows and along streams.
    Look {
        name: "wet_grass",
        maps: "grass",
        color: [0.07, 0.13, 0.04],
        roughness: 0.85,
        tint: [0.72, 0.95, 0.68],
        tiling: 3.0,
        height_contrast: 0.35,
        projection: SurfaceProjection::Top,
        cover: Some(Cover {
            density: 170.0,
            height: [0.16, 0.5],
            width: 0.035,
            color: [0.1, 0.2, 0.05],
            sway: 0.6,
        }),
    },
    // Stream and river beds: the scree's stones, washed darker and wetter.
    Look {
        name: "riverbed",
        maps: "scree",
        color: [0.13, 0.13, 0.12],
        roughness: 0.55,
        tint: [0.55, 0.56, 0.54],
        tiling: 3.0,
        height_contrast: 0.9,
        projection: SurfaceProjection::Top,
        cover: None,
    },
    // Roads (decision 069): packed gravel, the scree's stones finer and
    // warmer, where no grass grows.
    Look {
        name: "road",
        maps: "scree",
        color: [0.27, 0.24, 0.2],
        roughness: 0.9,
        tint: [0.92, 0.85, 0.74],
        tiling: 1.6,
        height_contrast: 0.6,
        projection: SurfaceProjection::Top,
        cover: None,
    },
];

/// Level-0 tiles along each axis, and their sample spacing: 8,192 meters.
const TILES: u32 = 32;
const SPACING: f32 = 1.0;
const SIZE: f32 = TILES as f32 * 256.0 * SPACING;
/// The plateau's center, where the hamlet stands.
const PLATEAU: [f32; 2] = [0.0, 0.0];

/// The road east from the plateau's rim (world meters): straight on a
/// highway's grade, over the valley beyond on a viaduct (decision 072) and
/// through the ridge after it in a tunnel (decision 073).
const EAST_ROAD: [[f32; 2]; 3] = [[150.0, 55.0], [376.0, 137.0], [1128.0, 410.0]];

/// The road west from the plateau's rim into the western valleys
/// (decision 069), terrain-local: the route `terrain_route` found over the
/// default seed's ground from world (-130, 40) to (-1500, 400) with its
/// default costs, 2.1 kilometers. It crosses a river four meters deep,
/// which it bridges (decision 072).
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

/// The road from the plateau's rim down to the lake in the valley (decision
/// 069), terrain-local: the route `terrain_route` found over the default
/// seed's ground from world (113, -98) toward the valley's center (1500,
/// -1300) with a grade cost of 100, up to where it reaches the lake's
/// shore and a few meters into the water: 1.8 kilometers with a hairpin,
/// every grade within 12% but its hairpin's.
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

/// The height the hamlet's plateau is leveled to.
#[derive(Resource, Clone, Copy)]
pub(crate) struct Plateau(pub f32);

/// Spawns the terrain, its materials' maps read from `files` where the page
/// supplied them; returns the region shadows cover and the height the
/// hamlet stands at.
pub(crate) fn populate(
    world: &mut World,
    config: &StressConfig,
    hamlet: f32,
    files: &HashMap<String, StressFile>,
) -> Result<((Vec3, Vec3), f32), String> {
    let half = SIZE * 0.5;
    // Terrain-local: the terrain's first corner is at -half.
    let local = |point: [f32; 2]| [point[0] + half, point[1] + half];
    let plateau = local(PLATEAU);
    let pyramid = Pyramid::new([TILES, TILES], SPACING).map_err(|error| error.to_string())?;
    let mountains = Layer::Noise(NoiseLayer {
        seed: config.seed,
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
    // Leveled where the mountains already stand there.
    let level = Layers::new(vec![mountains.clone()])
        .map_err(|error| error.to_string())?
        .height_at(&pyramid, Vec2::from(plateau));
    world.insert_resource(Plateau(level));
    // The hamlet's plateau, wide enough for the field's grid: leveled
    // before erosion, which then wears around it, and again after, so it
    // stays level whatever runs over it.
    let plateau_stamp = Layer::Stamp(Stamp {
        center: plateau,
        radius: hamlet,
        falloff: 220.0,
        op: StampOp::Flatten(level),
    });
    let layers = Layers::new(vec![
        mountains,
        plateau_stamp.clone(),
        // Rivers wear valleys into the mountains and slopes creep toward
        // them (decision 068), baked over the sixteen-meter tiles, so most
        // basins drain.
        Layer::Erosion(ErosionStage {
            layer: ErosionLayer {
                level: 4,
                rounds: 40,
                incision: 0.0015,
                creep: 3.0,
            },
            baked: None,
        }),
        // A wide valley floor to the north-east, leveled low.
        Layer::Stamp(Stamp {
            center: local([1500.0, -1300.0]),
            radius: 520.0,
            falloff: 700.0,
            op: StampOp::Flatten(60.0),
        }),
        // Streams and rivers (decision 066) carved where water gathers off
        // the mountains, baked over the sixteen-meter tiles erosion shaped.
        Layer::Hydrology(HydrologyStage {
            layer: HydrologyLayer {
                level: 4,
                channel_area: 150_000.0,
                depth: 1.2,
                width: 7.0,
                max_depth: 7.0,
                max_width: 70.0,
            },
            baked: None,
        }),
        plateau_stamp,
        // The road down into the valley (decision 069), leveled to its own
        // profile across six meters, its shoulders easing into the ground.
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
        // The road east, straight through the ridge beyond the plateau in a
        // tunnel (decision 073).
        Layer::Path(PathStage {
            layer: Arc::new(PathLayer {
                points: EAST_ROAD.map(local).to_vec(),
                width: 6.0,
                shoulder: 6.0,
                // A highway's grade: viaducts over the valleys, the tunnel
                // through the ridge.
                smoothing: 600.0,
                bridge_fill: 4.0,
                tunnel_cut: 25.0,
            }),
            baked: None,
        }),
        // The road west, bridging the river it crosses (decision 072).
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
        // A cave mouth in the plateau's slope.
        Layer::Stamp(Stamp {
            center: [plateau[0] + hamlet + 120.0, plateau[1]],
            radius: 7.0,
            falloff: 0.0,
            op: StampOp::Hole,
        }),
    ])
    .map_err(|error| error.to_string())?;
    let surface = surface(world, files)?;
    world.spawn(
        Terrain::generated(Vec3::new(-half, 0.0, -half), pyramid, layers).with_surface(surface),
    );
    Ok((
        (
            Vec3::new(-half, -50.0, -half),
            Vec3::new(half, 1600.0, half),
        ),
        level,
    ))
}

/// The highlands' surface: its rules over [`LOOKS`], each material textured
/// by its maps when `files` holds them.
fn surface(
    world: &mut World,
    files: &HashMap<String, StressFile>,
) -> Result<TerrainSurface, String> {
    let mut assets = world.resource_mut::<AssetLibrary>();
    let mut materials = Vec::with_capacity(LOOKS.len());
    let mut loaded = HashMap::new();
    for look in &LOOKS {
        materials.push(SurfaceMaterial {
            material: maps(&mut assets, files, look, &mut loaded)?,
            color: look.color,
            roughness: look.roughness,
            tiling: look.tiling,
            height_contrast: look.height_contrast,
            projection: look.projection,
            cover: look.cover,
        });
    }
    // Boulders of a few shapes, in the rock's maps where the page supplied
    // them: lighter than the cliffs' and gentler in relief, as weathered
    // stones are.
    let base = Material::new([0.8, 0.77, 0.72, 1.0])
        .and_then(|material| {
            material.with_surface(Surface {
                roughness: 0.85,
                ..Surface::MATTE
            })
        })
        .map_err(|error| error.to_string())?;
    let error = |error: AssetError| error.to_string();
    let material = match materials[4]
        .material
        .and_then(|rock| assets.material(rock).copied())
    {
        Some(rock) => {
            let mut material = base;
            if let Some(map) = rock.base_color_texture() {
                material = material.with_base_color_texture(map).map_err(error)?;
            }
            if let Some(map) = rock.normal_texture() {
                material = material.with_normal_texture(map, 0.5).map_err(error)?;
            }
            if let Some(map) = rock.metallic_roughness_texture() {
                material = material
                    .with_metallic_roughness_texture(map)
                    .and_then(|material| material.with_occlusion_texture(map, 1.0))
                    .map_err(error)?;
            }
            material
        }
        None => Material::new([0.3, 0.28, 0.26, 1.0])
            .and_then(|material| material.with_surface(base.surface()))
            .map_err(error)?,
    };
    let rock = assets
        .insert_material(material)
        .map_err(|error| error.to_string())?;
    let mut placed = Vec::with_capacity(BOULDERS + SPRUCES);
    for seed in 0..BOULDERS as u32 {
        let mesh = assets
            .insert_mesh(crate::rocks::boulder(seed)?)
            .map_err(|error| error.to_string())?;
        // Boulders collide as spheres a little inside their faces
        // (decision 064).
        placed.push(PlacementAsset {
            mesh,
            material: rock,
            lods: Vec::new(),
            sway: 0.0,
            collider: Some(PlacedCollider::new(
                PlacedShape::Sphere { radius: 0.42 },
                [0.0; 3],
            )),
        });
    }
    // Spruces of a few shapes over one bark and needle atlas, their needles
    // cut out by its alpha.
    let atlas = Texture::with_alpha(
        crate::trees::ATLAS,
        crate::trees::ATLAS,
        crate::trees::atlas(),
    )
    .and_then(|texture| assets.insert_texture(texture))
    .map_err(error)?;
    let needles = Material::new([1.0, 1.0, 1.0, 1.0])
        .and_then(|material| {
            material.with_surface(Surface {
                roughness: 0.9,
                ..Surface::MATTE
            })
        })
        .and_then(|material| material.with_base_color_texture(TextureRef::new(atlas)))
        .and_then(|material| material.with_alpha_mode(AlphaMode::Mask { cutoff: 0.5 }))
        .and_then(|material| assets.insert_material(material))
        .map_err(error)?;
    for seed in 0..SPRUCES as u32 {
        let mut mesh = |detail| {
            crate::trees::conifer(seed, detail)
                .and_then(|mesh| assets.insert_mesh(mesh).map_err(error))
        };
        let full = mesh(0)?;
        let mut lods = Vec::with_capacity(crate::trees::CONIFER_ERRORS.len());
        for (detail, error) in crate::trees::CONIFER_ERRORS.into_iter().enumerate() {
            lods.push(PlacementLod {
                mesh: mesh(detail + 1)?,
                error,
            });
        }
        // Spruces bend in the wind (decision 058): a third of a meter at ten
        // meters up in full wind.
        // Their trunks collide, up to where the lowest whorls spread.
        placed.push(PlacementAsset {
            mesh: full,
            material: needles,
            lods,
            sway: 0.35,
            collider: Some(PlacedCollider::new(
                PlacedShape::Cuboid {
                    half_extents: [0.28, 2.0, 0.28],
                },
                [0.0, 1.5, 0.0],
            )),
        });
    }
    // Bridges (decision 072) are built of the boulders' weathered stone.
    TerrainSurface::new(&rules(), materials)
        .and_then(|surface| surface.with_assets(placed))
        .map(|surface| TerrainSurface {
            bridge: Some(rock),
            ..surface
        })
        .map_err(|error| error.to_string())
}

/// Boulder and spruce shapes the placement layers choose from.
const BOULDERS: usize = 5;
const SPRUCES: usize = 3;

/// `look`'s maps from `files` as an engine material tinted by its tint,
/// when the page supplied all three: base color, normal, and occlusion
/// (red), roughness (green), metalness (blue) and height (alpha) in one.
/// Looks borrowing the same maps share their textures through `loaded`.
fn maps(
    assets: &mut AssetLibrary,
    files: &HashMap<String, StressFile>,
    look: &Look,
    loaded: &mut HashMap<String, TextureHandle>,
) -> Result<Option<MaterialHandle>, String> {
    let mut textures = Vec::with_capacity(3);
    for map in ["albedo", "normal", "orm"] {
        let file_name = format!("{}_{map}.ktx2", look.maps);
        if let Some(&texture) = loaded.get(&file_name) {
            textures.push(texture);
            continue;
        }
        let Some(file) = files.get(&file_name) else {
            return Ok(None);
        };
        let Some([width, height]) = file.size else {
            return Ok(None);
        };
        let texture = Texture::with_alpha(width, height, file.bytes.clone())
            .map_err(|error| format!("{file_name}: {error}"))?;
        let texture = assets
            .insert_texture(texture)
            .map_err(|error| error.to_string())?;
        loaded.insert(file_name, texture);
        textures.push(texture);
    }
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
        .and_then(|material| material.with_occlusion_texture(TextureRef::new(textures[2]), 1.0))
        .map_err(|error| error.to_string())?;
    assets
        .insert_material(material)
        .map(Some)
        .map_err(|error| error.to_string())
}

/// Where each material lies: meadow everywhere, dry grass in broad patches
/// with ragged edges, dirt where slopes thin the grass, scree on steep
/// ground below the cliffs, rock on the cliffs and snow on the peaks'
/// gentler slopes.
fn rules() -> Biomes {
    let noise = |wavelength: f32, seed: u32, octaves: u32| {
        Mask::Noise(NoiseMask {
            wavelength,
            seed,
            octaves,
        })
    };
    let scaled = |mask: Mask, by: f32| Mask::Multiply(vec![mask, Mask::Constant(by)]);
    let remap = |mask: Mask, from: f32, to: f32| {
        Mask::Remap(Remap {
            mask: Box::new(mask),
            from,
            to,
        })
    };
    let layer = |material: &str, mask: Mask| BiomeLayer {
        material: material.into(),
        mask,
        opacity: 1.0,
    };
    // Streams and rivers (decision 066): where enough water gathers, but
    // not across the pits it would fill as lakes.
    let channels = || {
        Mask::Multiply(vec![
            Mask::Flow([5.12, 5.3]),
            Mask::Invert(Box::new(Mask::Lake([0.5, 2.0]))),
        ])
    };
    // Where spruces stand in broad stands below the tree line on gentle
    // ground, thinning at their edges and kept out of stream beds; their
    // needles litter the ground there.
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
            // Nor where lakes stand (decision 067), nor on roads and their
            // verges (decision 069).
            Mask::Invert(Box::new(Mask::Lake([0.1, 0.6]))),
            Mask::Invert(Box::new(Mask::Path([10.0, 4.0]))),
        ])
    };
    let boulders = |weight: f32| {
        (0..BOULDERS)
            .map(|index| PlacementChoice {
                asset: format!("boulder{index}"),
                weight,
            })
            .collect::<Vec<_>>()
    };
    Biomes {
        assets: (0..BOULDERS)
            .map(|index| format!("boulder{index}"))
            .chain((0..SPRUCES).map(|index| format!("spruce{index}")))
            .collect(),
        // Boulders in clumps over the meadows, stones strewn on scree,
        // outcrops on rock, and spruce forests in broad stands below the
        // tree line on gentle ground, thinning at their edges: placed on
        // two-kilometer tiles, which stay resident farther away.
        placements: vec![
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
                sink: 0.3,
                level: 3,
                // A third of them stand on the pinned four-kilometer tiles too,
                // so forests reach the horizon (decision 061).
                far: Some(FarPlacement {
                    level: 4,
                    share: 0.3,
                }),
            },
            PlacementLayer {
                id: "boulders".into(),
                assets: boulders(1.0),
                density: 18.0,
                footprint: 7.0,
                materials: vec!["grass".into(), "dry_grass".into(), "dirt".into()],
                mask: Mask::Multiply(vec![
                    remap(noise(90.0, 21, 2), 0.45, 0.75),
                    Mask::Invert(Box::new(Mask::Path([6.0, 2.0]))),
                ]),
                scale: [0.6, 2.4],
                align: 0.3,
                sink: 0.2,
                level: 0,
                far: None,
            },
            PlacementLayer {
                id: "stones".into(),
                assets: boulders(1.0),
                density: 450.0,
                footprint: 1.5,
                materials: vec!["scree".into()],
                mask: Mask::one(),
                scale: [0.12, 0.45],
                align: 0.8,
                sink: 0.08,
                level: 0,
                far: None,
            },
            PlacementLayer {
                id: "outcrops".into(),
                assets: boulders(1.0),
                density: 25.0,
                footprint: 8.0,
                materials: vec!["rock".into()],
                mask: Mask::one(),
                scale: [2.0, 5.0],
                align: 0.6,
                sink: 0.5,
                level: 0,
                far: None,
            },
        ],
        materials: LOOKS.iter().map(|look| look.name.to_owned()).collect(),
        biomes: vec![Biome {
            id: "highlands".into(),
            fitness: Mask::one(),
            layers: vec![
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
                            scaled(noise(60.0, 13, 2), 0.45),
                            scaled(noise(9.0, 14, 2), 0.25),
                            scaled(Mask::Slope([18.0, 34.0]), 0.5),
                        ]),
                        0.66,
                        0.8,
                    ),
                ),
                layer("forest_floor", remap(forest(), 0.12, 0.42)),
                // Lush grass in wet hollows and along streams.
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
                    "scree",
                    remap(
                        Mask::Add(vec![
                            scaled(Mask::Slope([27.0, 38.0]), 0.8),
                            scaled(noise(40.0, 15, 2), 0.4),
                            scaled(noise(8.0, 17, 1), 0.15),
                        ]),
                        0.62,
                        0.8,
                    ),
                ),
                // Stream and river beds, with ragged edges.
                layer(
                    "riverbed",
                    remap(
                        Mask::Add(vec![channels(), scaled(noise(12.0, 42, 2), 0.3)]),
                        0.55,
                        0.75,
                    ),
                ),
                // Roads (decision 069), their edges soft over a meter and a
                // half and ragged where the gravel's stones win.
                layer("road", Mask::Path([1.0, -0.5])),
                layer(
                    "rock",
                    remap(
                        Mask::Add(vec![
                            Mask::Slope([33.0, 45.0]),
                            scaled(noise(70.0, 1, 2), 0.3),
                        ]),
                        0.55,
                        0.75,
                    ),
                ),
                layer(
                    "snow",
                    Mask::Multiply(vec![
                        remap(
                            Mask::Add(vec![
                                Mask::Height([700.0, 900.0]),
                                scaled(noise(60.0, 3, 2), 0.4),
                                Mask::Constant(-0.2),
                            ]),
                            0.4,
                            0.6,
                        ),
                        Mask::Invert(Box::new(Mask::Slope([35.0, 50.0]))),
                    ]),
                ),
            ],
        }],
    }
}

/// The plateau's radius for the hamlet of `config`'s renderables, the tower
/// field's grid with a margin.
pub(crate) fn hamlet_radius(config: &StressConfig) -> f32 {
    let side = (config.renderables as f32).sqrt().ceil() * crate::scene::SPACING;
    side * 0.75 + 30.0
}

/// The ground's world height under world `(x, z)` of `world`'s terrain,
/// from its layers wherever tiles are resident or not.
fn ground(world: &World, x: f32, z: f32) -> f32 {
    let mut terrains = world
        .try_query::<&Terrain>()
        .expect("terrain is registered");
    terrains
        .iter(world)
        .find_map(|terrain| match &terrain.source {
            TerrainSource::Generated(layers) => Some(
                terrain.origin.y
                    + layers.height_at(
                        &terrain.pyramid,
                        Vec2::new(x - terrain.origin.x, z - terrain.origin.z),
                    ),
            ),
            TerrainSource::File { .. } => None,
        })
        .unwrap_or_else(|| world.resource::<Plateau>().0)
}

/// Horizontal unit directions around the plateau, every 45 degrees.
fn directions() -> impl Iterator<Item = Vec2> {
    (0..8).map(|step| {
        let (sin, cos) = math::sin_cos(step as f32 * std::f32::consts::FRAC_PI_4);
        Vec2::new(cos, sin)
    })
}

/// The direction from the plateau whose ground `reach` meters out is
/// lowest: the widest view over the valleys.
fn lowest(world: &World, reach: f32) -> Vec2 {
    directions()
        .map(|direction| {
            let point = direction * reach;
            (ground(world, point.x, point.y), direction)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map_or(Vec2::Y, |(_, direction)| direction)
}

/// A camera `distance` meters from `target` along horizontal `direction`,
/// `rise` meters up per meter out, raised until its line of sight clears
/// the ground by 30 meters.
fn clear_view(world: &World, target: Vec3, direction: Vec2, distance: f32, rise: f32) -> Vec3 {
    let mut height = distance * rise;
    for step in 1..=24 {
        let along = step as f32 / 24.0;
        let point = Vec2::new(target.x, target.z) + direction * distance * along;
        let needed = (ground(world, point.x, point.y) + 30.0 - target.y) / along;
        height = math::max(height, needed);
    }
    let point = Vec2::new(target.x, target.z) + direction * distance;
    Vec3::new(point.x, target.y + height, point.y)
}

/// The overview: high over the plateau from its lowest side, looking back
/// across the hamlet into the mountains.
pub(crate) fn overview(world: &World, config: &StressConfig) -> (Vec3, Vec3) {
    let target = Vec3::new(0.0, world.resource::<Plateau>().0, 0.0);
    let direction = lowest(world, 1200.0);
    let distance = hamlet_radius(config) + 700.0;
    (clear_view(world, target, direction, distance, 0.45), target)
}

/// Inside the plateau's edge, looking out over the lowest valley beyond it.
pub(crate) fn close_view(world: &World, config: &StressConfig) -> (Vec3, Vec3) {
    let direction = lowest(world, 1500.0);
    let edge = direction * hamlet_radius(config) * 0.85;
    let far = direction * 1500.0;
    (
        Vec3::new(edge.x, ground(world, edge.x, edge.y) + 10.0, edge.y),
        Vec3::new(far.x, ground(world, far.x, far.y) + 60.0, far.y),
    )
}

/// From high above the plateau across the whole terrain to the north.
pub(crate) fn away_view(world: &World) -> (Vec3, Vec3) {
    let height = ground(world, 0.0, 0.0) + 700.0;
    (
        Vec3::new(0.0, height, 600.0),
        Vec3::new(0.0, height - 500.0, -3000.0),
    )
}
