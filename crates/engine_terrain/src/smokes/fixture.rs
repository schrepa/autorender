use std::sync::Arc;

use bevy_ecs::prelude::{Entity, World};
use engine_assets::{AssetLibrary, Material, MaterialHandle, Mesh, MeshHandle, MeshVertex};
use engine_core::{
    AuthoredId, Interior, OperationScene, ReflectionProbe, Transform, propagate_transforms,
};
use glam::{Vec2, Vec3};

use crate::{
    AuthoredPlacement, Biome, BiomeLayer, Biomes, Blend, Cover, ErosionLayer, ErosionStage,
    FarPlacement, HeightImage, HeightmapLayer, HydrologyLayer, HydrologyStage, Layer, Layers,
    MacroCauses, Mask, NoiseLayer, PathLayer, PathStage, PlacedCollider, PlacedShape,
    PlacementAsset, PlacementChoice, PlacementLayer, PlacementLod, Pyramid, Stamp, StampOp,
    SurfaceMaterial, Surroundings, Terrain, TerrainSurface,
};

pub const FIXTURE: &str = "terrain-fixture";

const DOME_SIDE: u32 = 65;

const ICOSAHEDRON: [[u32; 3]; 20] = [
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [4, 9, 5],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1],
];

#[derive(Clone, Copy, Debug)]
pub struct Fixture {
    pub ground: Entity,
    pub probe: Entity,
    pub hall: Entity,
    pub cube: MeshHandle,
    pub plain: MaterialHandle,
}

pub fn fixture(world: &mut World) -> Fixture {
    let (cube, ball, plain, grain) = {
        let mut assets = world.resource_mut::<AssetLibrary>();
        let cube = assets.insert_mesh(Mesh::cube()).expect("fixture_fits");
        let ball = assets.insert_mesh(ball()).expect("fixture_fits");
        let plain = Material::new([0.8, 0.3, 0.2, 1.0]).expect("plain_valid");
        let grain = Material::new([1.0; 4]).expect("grain_valid");
        let plain = assets.insert_material(plain).expect("fixture_fits");
        let grain = assets.insert_material(grain).expect("fixture_fits");
        (cube, ball, plain, grain)
    };
    let mut surface = TerrainSurface::new(&biomes(), looks(grain)).expect("fixture_surface_valid");
    surface.assets = vec![
        PlacementAsset {
            mesh: cube,
            material: plain,
            lods: vec![PlacementLod {
                mesh: cube,
                error: 0.0625,
            }],
            parts: Vec::new(),
            shadow: None,
            sway: 0.5,
            collider: Some(PlacedCollider::new(
                PlacedShape::Cuboid {
                    half_extents: [0.5; 3],
                },
                [0.0; 3],
            )),
        },
        PlacementAsset {
            mesh: ball,
            material: plain,
            lods: Vec::new(),
            parts: Vec::new(),
            shadow: None,
            sway: 0.0,
            collider: Some(PlacedCollider::new(
                PlacedShape::Sphere { radius: 1.0 },
                [0.0; 3],
            )),
        },
    ];
    let terrain = Terrain {
        origin: Vec3::new(-512.0, 0.0, -512.0),
        pyramid: Pyramid::new([4, 4], 1.0).expect("fixture_tiles_valid"),
        layers: Arc::new(Layers::new(layers()).expect("fixture_layers_valid")),
        surface,
        collision: true,
        edits: Vec::new().into(),
        rivers: Vec::new().into(),
        surroundings: Some(Surroundings {
            reach: 3000.0,
            blend: 128.0,
        }),
    };
    let ground = world.spawn((Transform::IDENTITY, terrain)).id();
    let probe = world
        .spawn((
            placed(Vec3::new(0.0, 45.0, 0.0)),
            ReflectionProbe::new(Vec3::new(60.0, 40.0, 60.0)),
        ))
        .id();
    let hall = world
        .spawn((
            placed(Vec3::new(-300.0, 40.0, 300.0)),
            ReflectionProbe::new(Vec3::new(20.0, 200.0, 20.0))
                .with_blend(0.5)
                .with_interior(Interior { sky: 0.1, sun: 0.1 }),
        ))
        .id();
    propagate_transforms(world);
    for (id, entity) in [("ground", ground), ("probe", probe), ("hall", hall)] {
        name(world, id, entity);
    }
    Fixture {
        ground,
        probe,
        hall,
        cube,
        plain,
    }
}

pub fn name(world: &mut World, id: &str, entity: Entity) {
    let mut entities = world
        .get_resource::<OperationScene>()
        .map(|scene| (*scene.entities).clone())
        .unwrap_or_default();
    entities.insert(id.to_owned(), entity);
    world.insert_resource(OperationScene {
        id: FIXTURE.to_owned(),
        entities: Arc::new(entities),
        streamed: false,
        authored,
    });
}

fn authored(world: &World, entity: Entity) -> Option<AuthoredId> {
    let scene = world.get_resource::<OperationScene>()?;
    let (id, _) = scene.entities.iter().find(|&(_, &named)| named == entity)?;
    Some(AuthoredId {
        scene: scene.id.clone(),
        id: id.clone(),
        name: id.clone(),
    })
}

fn placed(translation: Vec3) -> Transform {
    Transform {
        translation,
        ..Transform::IDENTITY
    }
}

fn layers() -> Vec<Layer> {
    let stamp = |center: [f32; 2], radius: f32, falloff: f32, op: StampOp| {
        Layer::Stamp(Stamp {
            center,
            radius,
            falloff,
            op,
        })
    };
    let path = |points: Vec<[f32; 2]>, smoothing: f32, tunnel_cut: f32| {
        Layer::Path(PathStage {
            layer: Arc::new(PathLayer {
                points,
                width: 6.0,
                shoulder: 6.0,
                smoothing,
                bridge_fill: 4.0,
                tunnel_cut,
            }),
            baked: None,
        })
    };
    vec![
        Layer::Noise(NoiseLayer {
            seed: 3,
            height: 140.0,
            wavelength: 600.0,
            ..NoiseLayer::default()
        }),
        Layer::Erosion(ErosionStage {
            layer: ErosionLayer {
                level: 1,
                rounds: 6,
                incision: 0.001_953_125,
                creep: 2.0,
                ..ErosionLayer::default()
            },
            baked: None,
        }),
        Layer::Hydrology(HydrologyStage {
            layer: HydrologyLayer {
                level: 1,
                ..HydrologyLayer::default()
            },
            baked: None,
        }),
        stamp([430.0, 96.0], 10.0, 10.0, StampOp::Lower(12.0)),
        path(
            vec![
                [100.0, 300.0],
                [250.0, 260.0],
                [400.0, 150.0],
                [450.0, 60.0],
            ],
            60.0,
            25.0,
        ),
        stamp([750.0, 200.0], 30.0, 40.0, StampOp::Raise(40.0)),
        path(vec![[600.0, 200.0], [900.0, 200.0]], 400.0, 15.0),
        stamp([512.0, 512.0], 40.0, 60.0, StampOp::Flatten(30.0)),
        stamp([560.0, 512.0], 4.0, 0.0, StampOp::Hole),
        Layer::Heightmap(HeightmapLayer {
            image: Arc::new(HeightImage::new(DOME_SIDE, DOME_SIDE, dome()).expect("dome_fits")),
            min: Vec2::new(700.0, 700.0),
            size: Vec2::new(128.0, 128.0),
            low: 0.0,
            high: 200.0,
            falloff: 0.0,
            blend: Blend::Max,
            uri: None,
            source: None,
            attribution: None,
        }),
    ]
}

fn dome() -> Vec<u16> {
    let middle = f64::from(DOME_SIDE / 2);
    let mut samples = Vec::with_capacity((DOME_SIDE * DOME_SIDE) as usize);
    for z in 0..DOME_SIDE {
        for x in 0..DOME_SIDE {
            let (dx, dz) = (f64::from(x) - middle, f64::from(z) - middle);
            let rise = (1.0 - (dx * dx + dz * dz) / (middle * middle)).max(0.0);
            samples.push((f64::from(u16::MAX) * rise).round() as u16);
        }
    }
    samples
}

fn looks(grain: MaterialHandle) -> Vec<SurfaceMaterial> {
    vec![
        SurfaceMaterial {
            color: [0.12, 0.2, 0.06],
            cover: Some(Cover {
                density: 60.0,
                ..Cover::default()
            }),
            ..SurfaceMaterial::default()
        },
        SurfaceMaterial {
            color: [0.3, 0.28, 0.26],
            roughness: 0.8,
            ..SurfaceMaterial::default()
        },
        SurfaceMaterial {
            color: [0.85, 0.87, 0.9],
            roughness: 0.5,
            ..SurfaceMaterial::default()
        },
        SurfaceMaterial {
            material: Some(grain),
            color: [0.36, 0.33, 0.28],
            roughness: 0.9,
            depth: 0.1,
            ..SurfaceMaterial::default()
        },
    ]
}

fn biomes() -> Biomes {
    let layer = |material: &str, mask: Mask| BiomeLayer {
        material: material.to_owned(),
        mask,
        opacity: 1.0,
    };
    let choice = |asset: &str| PlacementChoice {
        asset: asset.to_owned(),
        weight: 1.0,
    };
    Biomes {
        materials: ["meadow", "rock", "snow", "gravel"]
            .map(String::from)
            .to_vec(),
        biomes: vec![
            Biome {
                id: "lowland".to_owned(),
                fitness: Mask::one(),
                layers: vec![
                    layer("meadow", Mask::one()),
                    layer("rock", Mask::Slope([25.0, 35.0])),
                    layer("gravel", Mask::Path([1.0, 0.0])),
                ],
                wetness: None,
            },
            Biome {
                id: "summit".to_owned(),
                fitness: Mask::Height([110.0, 130.0]),
                layers: vec![layer("snow", Mask::one())],
                wetness: None,
            },
        ],
        steep: None,
        assets: vec!["crate".to_owned(), "ball".to_owned()],
        placements: vec![PlacementLayer {
            id: "crates".to_owned(),
            assets: vec![choice("crate"), choice("ball")],
            density: 100.0,
            footprint: 4.0,
            materials: vec!["meadow".to_owned()],
            scale: [0.5, 1.0],
            far: Some(FarPlacement { level: 2 }),
            ..PlacementLayer::default()
        }],
        placed: vec![
            AuthoredPlacement {
                asset: "crate".to_owned(),
                position: [530.0, 500.0],
                yaw: 30.0,
                scale: 3.0,
                ..AuthoredPlacement::default()
            },
            AuthoredPlacement {
                asset: "ball".to_owned(),
                position: [500.0, 530.0],
                scale: 2.0,
                level: 1,
                ..AuthoredPlacement::default()
            },
        ],
        macro_colors: MacroCauses::default(),
    }
}

fn ball() -> Mesh {
    let golden = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let mut points: Vec<Vec3> = [
        [-1.0, golden, 0.0],
        [1.0, golden, 0.0],
        [-1.0, -golden, 0.0],
        [1.0, -golden, 0.0],
        [0.0, -1.0, golden],
        [0.0, 1.0, golden],
        [0.0, -1.0, -golden],
        [0.0, 1.0, -golden],
        [golden, 0.0, -1.0],
        [golden, 0.0, 1.0],
        [-golden, 0.0, -1.0],
        [-golden, 0.0, 1.0],
    ]
    .into_iter()
    .map(|point| Vec3::from_array(point).normalize())
    .collect();
    let mut middles: Vec<([u32; 2], u32)> = Vec::new();
    let mut indices = Vec::with_capacity(ICOSAHEDRON.len() * 12);
    for [a, b, c] in ICOSAHEDRON {
        let mut middle = |from: u32, to: u32| {
            let edge = [from.min(to), from.max(to)];
            if let Some(&(_, index)) = middles.iter().find(|(known, _)| *known == edge) {
                return index;
            }
            let index = points.len() as u32;
            points.push((points[from as usize] + points[to as usize]).normalize());
            middles.push((edge, index));
            index
        };
        let (ab, bc, ca) = (middle(a, b), middle(b, c), middle(c, a));
        indices.extend([a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca]);
    }
    let vertices = points
        .iter()
        .map(|point| MeshVertex {
            position: point.to_array(),
            normal: point.to_array(),
            uv: [0.0; 2],
        })
        .collect();
    Mesh::new(vertices, indices).expect("ball_valid")
}
