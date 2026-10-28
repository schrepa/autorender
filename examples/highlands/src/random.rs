use engine::prelude::*;

pub(crate) fn hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

pub(crate) fn golden_hash(value: u32) -> u32 {
    let mut h = value.wrapping_mul(0x9e37_79b1);
    h = (h ^ (h >> 16)).wrapping_mul(0x7feb_352d);
    h = (h ^ (h >> 15)).wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

pub(crate) fn unit(value: u32) -> f32 {
    (value >> 8) as f32 / 16_777_216.0
}

pub(crate) fn noise(point: Vec3) -> f32 {
    let cell = point.floor();
    let f = point - cell;
    let u = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let corner = |dx: i32, dy: i32, dz: i32| {
        let key = (cell.x as i32 + dx) as u32
            ^ ((cell.y as i32 + dy) as u32).wrapping_mul(0x8da6_b343)
            ^ ((cell.z as i32 + dz) as u32).wrapping_mul(0xd816_3841);
        unit(golden_hash(key))
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), u.x);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), u.x);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), u.x);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), u.x);
    lerp(lerp(x00, x10, u.y), lerp(x01, x11, u.y), u.z)
}

pub(crate) struct Rng(u32);

impl Rng {
    pub(crate) fn new(seed: u32, a: i32, b: i32) -> Self {
        Self(hash(
            seed ^ hash((a as u32).wrapping_mul(73_856_093) ^ (b as u32).wrapping_mul(19_349_663)),
        ))
    }
    pub(crate) fn next(&mut self) -> u32 {
        self.0 = hash(self.0.wrapping_add(0x9e37_79b9));
        self.0
    }
    pub(crate) fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / 16_777_216.0
    }
    pub(crate) fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }
}
