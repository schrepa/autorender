use brick_nanite::NaniteWorld;
use engine::prelude::*;

pub const HERO_ASSETS: [&str; 5] = [
    "colossus_sentinel",
    "fluted_column",
    "vaulted_arch",
    "canyon_monolith",
    "cypress_cluster",
];

pub const MESH_SENTINEL: u32 = 0;
pub const MESH_COLUMN: u32 = 1;
pub const MESH_ARCH: u32 = 2;
pub const MESH_CANYON: u32 = 3;
pub const MESH_CYPRESS: u32 = 4;

fn hash_unit(n: i32) -> f32 {
    let raw = ((n as f64) * 12.9898).sin() * 43758.5453;
    (raw - raw.floor()) as f32
}

fn inst_rot(yaw_deg: f32, roll_deg: f32) -> Quat {
    Quat::from_rotation_y(yaw_deg.to_radians()) * Quat::from_rotation_z(roll_deg.to_radians())
}

fn add_colonnade(world: &mut NaniteWorld) {
    let rows = [-20.0_f32, -14.0, 14.0, 20.0];
    for x in rows {
        for i in 0..49_i32 {
            let z = 141.0 - 6.0 * (i as f32);
            let yaw = 90.0 * ((i % 4) as f32);
            world.add_instance(MESH_COLUMN, Vec3::new(x, 0.0, z), inst_rot(yaw, 0.0), 1.0);
        }
    }
}

fn add_arches_and_sentinels(world: &mut NaniteWorld) {
    world.add_instance(MESH_ARCH, Vec3::new(0.0, 0.0, 42.0), inst_rot(0.0, 0.0), 1.2);
    world.add_instance(MESH_ARCH, Vec3::new(0.0, 0.0, -66.0), inst_rot(0.0, 0.0), 1.2);

    world.add_instance(
        MESH_SENTINEL,
        Vec3::new(0.0, -3.6, -215.0),
        inst_rot(0.0, 0.0),
        1.6,
    );
    world.add_instance(
        MESH_SENTINEL,
        Vec3::new(-22.0, -3.6, -196.0),
        inst_rot(20.0, 0.0),
        0.85,
    );
    world.add_instance(
        MESH_SENTINEL,
        Vec3::new(22.0, -3.6, -196.0),
        inst_rot(-20.0, 0.0),
        0.85,
    );
}

fn add_plaza_ring_and_ruins(world: &mut NaniteWorld) {
    for k in 0..18_i32 {
        if k == 8 || k == 9 {
            continue;
        }
        let deg = 10.0 + 20.0 * (k as f32);
        let rad = deg.to_radians();
        let (s, c) = engine::math::sin_cos(rad);
        let pos = Vec3::new(40.0 * s, -1.2, -205.0 - 40.0 * c);
        world.add_instance(MESH_COLUMN, pos, inst_rot(-deg, 0.0), 1.0);
    }

    let fallen: [([f32; 3], f32); 10] = [
        ([-30.0, 0.95, 95.0], 20.0),
        ([31.0, 0.95, 18.0], 160.0),
        ([-34.0, 0.95, -40.0], 75.0),
        ([36.0, 0.95, -102.0], -30.0),
        ([-45.0, -0.25, -176.0], 10.0),
        ([47.0, -0.25, -230.0], 200.0),
        ([-18.0, -2.65, -186.0], 40.0),
        ([20.0, -2.65, -238.0], -120.0),
        ([62.0, 0.95, -128.0], 95.0),
        ([-60.0, 0.95, 130.0], -15.0),
    ];
    for (p, yaw) in fallen {
        world.add_instance(MESH_COLUMN, Vec3::from_array(p), inst_rot(yaw, 90.0), 1.0);
    }
}

fn add_canyon_megaliths(world: &mut NaniteWorld) {
    let rocks: [([f32; 3], f32, f32); 16] = [
        ([0.0, -4.0, -340.0], 90.0, 3.0),
        ([-38.0, -2.0, -290.0], 15.0, 2.2),
        ([30.0, -2.0, -300.0], -160.0, 2.6),
        ([-85.0, -2.0, -270.0], 60.0, 2.0),
        ([85.0, -2.0, -265.0], -70.0, 2.1),
        ([-95.0, 0.0, -200.0], 175.0, 1.8),
        ([95.0, 0.0, -190.0], -5.0, 1.9),
        ([-110.0, 0.0, -130.0], 200.0, 1.6),
        ([112.0, 0.0, -120.0], 15.0, 1.7),
        ([-120.0, 0.0, -30.0], 165.0, 1.5),
        ([125.0, 0.0, -10.0], -20.0, 1.4),
        ([-130.0, 0.0, 70.0], 190.0, 1.3),
        ([135.0, 0.0, 95.0], 10.0, 1.6),
        ([-200.0, -6.0, -420.0], 30.0, 4.0),
        ([180.0, -6.0, -440.0], -20.0, 3.6),
        ([0.0, -8.0, -520.0], 0.0, 5.0),
    ];
    for (p, yaw, scale) in rocks {
        world.add_instance(MESH_CANYON, Vec3::from_array(p), inst_rot(yaw, 0.0), scale);
    }
}

fn add_cypresses_and_arcade(world: &mut NaniteWorld) {
    let ave_rows = [-32.0_f32, -27.0, 27.0, 32.0];
    for (r, &x) in ave_rows.iter().enumerate() {
        for j in 0..36_i32 {
            let n = (r as i32) * 36 + j;
            let z = 138.0 - 8.0 * (j as f32);
            let px = x + (hash_unit(n) - 0.5) * 1.2;
            let pz = z + (hash_unit(n + 7) - 0.5) * 2.0;
            let yaw = 360.0 * hash_unit(n + 3);
            let scale = 0.85 + 0.35 * hash_unit(n + 5);
            world.add_instance(MESH_CYPRESS, Vec3::new(px, 0.0, pz), inst_rot(yaw, 0.0), scale);
        }
    }

    for k in 0..24_i32 {
        let deg = 20.0 + (320.0 / 23.0) * (k as f32);
        let rad = deg.to_radians();
        let (s, c) = engine::math::sin_cos(rad);
        let pos = Vec3::new(58.0 * s, 0.0, -205.0 - 58.0 * c);
        let yaw = 360.0 * hash_unit(200 + k);
        let scale = 1.05 + 0.4 * hash_unit(240 + k);
        world.add_instance(MESH_CYPRESS, pos, inst_rot(yaw, 0.0), scale);
    }

    for &x in &[50.0_f32, 70.0] {
        for m in 0..16_i32 {
            let z = 40.0 - 10.0 * (m as f32);
            let yaw = 360.0 * hash_unit(300 + m);
            let scale = 0.95 + 0.3 * hash_unit(330 + m);
            world.add_instance(MESH_CYPRESS, Vec3::new(x, 0.0, z), inst_rot(yaw, 0.0), scale);
        }
    }

    for m in 0..14_i32 {
        let z = 30.0 - 10.8 * (m as f32);
        world.add_instance(MESH_ARCH, Vec3::new(60.0, 0.0, z), inst_rot(0.0, 0.0), 1.0);
    }
}

pub fn populate_sanctuary(world: &mut NaniteWorld) {
    world.instances.clear();
    world.total_virtual_tris = 0;
    add_colonnade(world);
    add_arches_and_sentinels(world);
    add_plaza_ring_and_ruins(world);
    add_canyon_megaliths(world);
    add_cypresses_and_arcade(world);
}
