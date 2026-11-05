use glam::Vec4;

pub use engine_heightfield::hydraulics::GRAVITY;
pub use engine_math::{REDUCTION_LIMIT, atan2, clamp, cos, exp, ln, max, min, pow, sin, sin_cos};

pub fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * fraction * 2.0_f32.powi(-24),
        31 => sign * f32::INFINITY,
        _ => sign * (1.0 + fraction / 1024.0) * 2.0_f32.powi(exponent - 15),
    }
}

#[inline]
pub fn sin_cos4(x: Vec4) -> (Vec4, Vec4) {
    use engine_math::lanes::*;
    if !x.abs().cmple(Vec4::splat(REDUCTION_LIMIT)).all() {
        let lanes = x.to_array().map(sin_cos);
        return (
            Vec4::from_array(lanes.map(|(s, _)| s)),
            Vec4::from_array(lanes.map(|(_, c)| c)),
        );
    }
    let shifted = x * FRAC_2_PI + SHIFTER;
    let j = shifted - SHIFTER;
    let y = ((x - j * PIO2_A) - j * PIO2_B) - j * PIO2_C;
    let z = y * y;
    let s = (y + y * z * (S1 + z * (S2 + z * S3))).copysign(y);
    let c = 1.0 - 0.5 * z + z * z * (C1 + z * (C2 + z * C3));
    let quadrant = j - (j * 0.25).floor() * 4.0;
    let first = quadrant.cmpeq(Vec4::ONE);
    let swap = first | quadrant.cmpeq(Vec4::splat(3.0));
    let (sin, cos) = (Vec4::select(swap, c, s), Vec4::select(swap, s, c));
    let negate_sin = quadrant.cmpge(Vec4::splat(2.0));
    let negate_cos = first | quadrant.cmpeq(Vec4::splat(2.0));
    (
        Vec4::select(negate_sin, -sin, sin),
        Vec4::select(negate_cos, -cos, cos),
    )
}

pub struct Lanes4<T> {
    items: [Option<T>; 4],
}

impl<T> Lanes4<T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.items.iter().filter(|item| item.is_some()).count()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items[0].is_none()
    }

    #[inline]
    pub fn load(&self, field: impl Fn(&T) -> f32) -> Vec4 {
        Vec4::from_array(
            self.items
                .each_ref()
                .map(|item| item.as_ref().map_or(0.0, &field)),
        )
    }

    #[inline]
    pub fn for_each(&mut self, mut visit: impl FnMut(usize, &mut T)) {
        for (lane, item) in self.items.iter_mut().enumerate() {
            if let Some(item) = item {
                visit(lane, item);
            }
        }
    }
}

#[inline]
pub fn for_each4<I: IntoIterator>(items: I, mut kernel: impl FnMut(&mut Lanes4<I::Item>)) {
    let mut lanes = Lanes4 {
        items: [None, None, None, None],
    };
    let mut len = 0;
    for item in items {
        lanes.items[len] = Some(item);
        len += 1;
        if len == 4 {
            kernel(&mut lanes);
            lanes.items = [None, None, None, None];
            len = 0;
        }
    }
    if len != 0 {
        kernel(&mut lanes);
    }
}
