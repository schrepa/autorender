use engine_core::SmokeSession;
use glam::Vec3;
use serde_json::{Value, json};

pub(crate) const GROUND: &str = "ground";

const DOME_SIDE: u32 = 65;

const DOME: &str = "dome";

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

pub(crate) fn fixture(session: &mut dyn SmokeSession, hiker: bool) {
    session.file(DOME, &dome());
    session.source(&document(hiker).to_string());
}

pub(crate) fn prefetched(session: &mut dyn SmokeSession, spawn: [f64; 3]) {
    let mut placed = document(false);
    if let Some(entities) = placed["entities"].as_array_mut() {
        entities.push(
            json!({"id": "spawn-point", "prefetch": {}, "transform": {"translation": spawn}}),
        );
    }
    session.file(DOME, &dome());
    session.source(&placed.to_string());
}

fn document(hiker: bool) -> Value {
    let mut entities = vec![
        json!({"id": GROUND, "terrain": terrain()}),
        json!({"id": "probe", "reflection_probe": {"extent": [60.0, 40.0, 60.0]},
            "transform": {"translation": [0.0, 45.0, 0.0]}}),
        json!({"id": "hall", "reflection_probe": {"extent": [20.0, 200.0, 20.0], "blend": 0.5,
            "interior": {"sky": 0.1, "sun": 0.1}}, "transform": {"translation": [-300.0, 40.0, 300.0]}}),
    ];
    if hiker {
        entities.push(json!({"id": "hiker", "mesh": "cube", "material": "plain",
            "nav_agent": {"speed": 6.0, "offset": 0.5}, "transform": {"translation": [20.0, 31.0, 20.0]}}));
    }
    json!({
        "id": "terrain-fixture", "revision": 1,
        "camera": {"position": [0.0, 80.0, 140.0], "target": [0.0, 30.0, 0.0], "fov_degrees": 60, "near": 0.5, "far": 4000.0},
        "materials": [
            {"id": "plain", "color": [0.8, 0.3, 0.2, 1.0]},
            {"id": "grain", "color": [1.0, 1.0, 1.0, 1.0]},
        ],
        "meshes": [{"id": "cube", "source": {"kind": "cube"}}, ball()],
        "environment": {
            "ground_cover": {"looks": {"meadow": {"density": 60.0}}},
            "scatter": {"looks": [{"mesh": "cube", "lods": [{"mesh": "cube", "error": 0.0625}], "sway": 0.5}]},
        },
        "entities": entities,
    })
}

fn terrain() -> Value {
    json!({
        "origin": [-512.0, 0.0, -512.0], "tiles": [4, 4], "spacing": 1.0,
        "layers": layers(),
        "surface": surface(),
        "surroundings": {"reach": 3000.0, "blend": 128.0},
    })
}

fn layers() -> Vec<Value> {
    let stamp = |center: [f32; 2], radius: f32, falloff: f32, op: Value| json!({"type": "stamp", "center": center, "radius": radius, "falloff": falloff, "op": op});
    let path = |points: Value, smoothing: f32, tunnel_cut: f32| {
        json!({
            "type": "path",
            "points": points,
            "width": 6.0,
            "shoulder": 6.0,
            "smoothing": smoothing,
            "bridge_fill": 4.0,
            "tunnel_cut": tunnel_cut,
        })
    };
    vec![
        json!({"type": "noise", "seed": 3, "height": 140.0, "wavelength": 600.0}),
        json!({
            "type": "erosion",
            "level": 1,
            "rounds": 6,
            "incision": 0.001_953_125,
            "creep": 2.0,
        }),
        json!({"type": "hydrology", "level": 1}),
        stamp([430.0, 96.0], 10.0, 10.0, json!({"lower": 12.0})),
        path(
            json!([
                [100.0, 300.0],
                [250.0, 260.0],
                [400.0, 150.0],
                [450.0, 60.0]
            ]),
            60.0,
            25.0,
        ),
        stamp([750.0, 200.0], 30.0, 40.0, json!({"raise": 40.0})),
        path(json!([[600.0, 200.0], [900.0, 200.0]]), 400.0, 15.0),
        stamp([512.0, 512.0], 40.0, 60.0, json!({"flatten": 30.0})),
        stamp([560.0, 512.0], 4.0, 0.0, json!("hole")),
        json!({
            "type": "heightmap",
            "uri": DOME,
            "width": DOME_SIDE,
            "height": DOME_SIDE,
            "min": [700.0, 700.0],
            "size": [128.0, 128.0],
            "low": 0.0,
            "high": 200.0,
            "blend": "max",
        }),
    ]
}

fn surface() -> Value {
    let ramp =
        |map: &str, from: f32, to: f32| json!({"ramp": {"map": map, "from": from, "to": to}});
    json!({
        "materials": [
            {"id": "meadow", "color": [0.12, 0.2, 0.06]},
            {"id": "rock", "color": [0.3, 0.28, 0.26], "roughness": 0.8},
            {"id": "snow", "color": [0.85, 0.87, 0.9], "roughness": 0.5},
            {"id": "gravel", "material": "grain", "color": [0.36, 0.33, 0.28], "roughness": 0.9, "depth": 0.1},
        ],
        "biomes": [
            {"id": "lowland", "layers": [
                {"material": "meadow"},
                {"material": "rock", "mask": ramp("slope", 25.0, 35.0)},
                {"material": "gravel", "mask": ramp("path", 1.0, 0.0)},
            ]},
            {"id": "summit", "fitness": ramp("height", 110.0, 130.0), "layers": [{"material": "snow"}]},
        ],
        "assets": [
            {"id": "crate", "mesh": "cube", "material": "plain",
             "collider": {"shape": {"kind": "cuboid", "half_extents": [0.5, 0.5, 0.5]}}},
            {"id": "ball", "mesh": "ball", "material": "plain",
             "collider": {"shape": {"kind": "sphere", "radius": 1.0}}},
        ],
        "placements": [{
            "id": "crates",
            "assets": [{"asset": "crate", "weight": 1.0}, {"asset": "ball", "weight": 1.0}],
            "density": 100.0, "footprint": 4.0, "materials": ["meadow"], "scale": [0.5, 1.0],
            "far": {"level": 2},
        }],
        "placed": [
            {"asset": "crate", "position": [530.0, 500.0], "yaw": 30.0, "scale": 3.0},
            {"asset": "ball", "position": [500.0, 530.0], "scale": 2.0, "level": 1},
        ],
    })
}

fn dome() -> Vec<u8> {
    let middle = f64::from(DOME_SIDE / 2);
    let mut samples = Vec::with_capacity((DOME_SIDE * DOME_SIDE * 2) as usize);
    for z in 0..DOME_SIDE {
        for x in 0..DOME_SIDE {
            let (dx, dz) = (f64::from(x) - middle, f64::from(z) - middle);
            let rise = (1.0 - (dx * dx + dz * dz) / (middle * middle)).max(0.0);
            let sample = (f64::from(u16::MAX) * rise).round() as u16;
            samples.extend(sample.to_le_bytes());
        }
    }
    samples
}

fn ball() -> Value {
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
    let vertices: Vec<Value> = points
        .iter()
        .map(|point| json!({"position": point.to_array(), "normal": point.to_array()}))
        .collect();
    json!({"id": "ball", "source": {"kind": "inline", "vertices": vertices, "indices": indices}})
}
