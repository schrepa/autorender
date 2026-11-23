use glam::DVec3;

#[derive(Clone, Copy, Default)]
pub(crate) struct Quadric {
    pub a00: f64,
    pub a01: f64,
    pub a02: f64,
    pub a03: f64,
    pub a11: f64,
    pub a12: f64,
    pub a13: f64,
    pub a22: f64,
    pub a23: f64,
    pub a33: f64,
    pub weight: f64,
}

impl Quadric {
    pub(crate) fn from_plane(n: DVec3, d: f64, weight: f64) -> Self {
        let (a, b, c) = (n.x, n.y, n.z);
        Self {
            a00: a * a * weight,
            a01: a * b * weight,
            a02: a * c * weight,
            a03: a * d * weight,
            a11: b * b * weight,
            a12: b * c * weight,
            a13: b * d * weight,
            a22: c * c * weight,
            a23: c * d * weight,
            a33: d * d * weight,
            weight,
        }
    }

    pub(crate) fn add(&mut self, other: &Self) {
        self.a00 += other.a00;
        self.a01 += other.a01;
        self.a02 += other.a02;
        self.a03 += other.a03;
        self.a11 += other.a11;
        self.a12 += other.a12;
        self.a13 += other.a13;
        self.a22 += other.a22;
        self.a23 += other.a23;
        self.a33 += other.a33;
        self.weight += other.weight;
    }

    pub(crate) fn sum(&self, other: &Self) -> Self {
        let mut out = *self;
        out.add(other);
        out
    }

    pub(crate) fn eval(&self, p: DVec3) -> f64 {
        let (x, y, z) = (p.x, p.y, p.z);
        let value = self.a00 * x * x
            + 2.0 * self.a01 * x * y
            + 2.0 * self.a02 * x * z
            + 2.0 * self.a03 * x
            + self.a11 * y * y
            + 2.0 * self.a12 * y * z
            + 2.0 * self.a13 * y
            + self.a22 * z * z
            + 2.0 * self.a23 * z
            + self.a33;
        if value > 0.0 { value } else { 0.0 }
    }

    pub(crate) fn solve_optimal(&self) -> Option<DVec3> {
        let c00 = self.a11 * self.a22 - self.a12 * self.a12;
        let c01 = self.a02 * self.a12 - self.a01 * self.a22;
        let c02 = self.a01 * self.a12 - self.a02 * self.a11;
        let c11 = self.a00 * self.a22 - self.a02 * self.a02;
        let c12 = self.a01 * self.a02 - self.a00 * self.a12;
        let c22 = self.a00 * self.a11 - self.a01 * self.a01;
        let det = self.a00 * c00 + self.a01 * c01 + self.a02 * c02;
        let trace = self.a00 + self.a11 + self.a22;
        let scale = trace * trace * trace;
        if !det.is_finite() || det.abs() <= 1.0e-10 * scale.max(1.0e-12) {
            return None;
        }
        let inv_det = -1.0 / det;
        let x = (c00 * self.a03 + c01 * self.a13 + c02 * self.a23) * inv_det;
        let y = (c01 * self.a03 + c11 * self.a13 + c12 * self.a23) * inv_det;
        let z = (c02 * self.a03 + c12 * self.a13 + c22 * self.a23) * inv_det;
        let p = DVec3::new(x, y, z);
        p.is_finite().then_some(p)
    }
}
