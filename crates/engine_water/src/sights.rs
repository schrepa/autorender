use crate::{Pool, RiverFlow};

impl RiverFlow {
    pub fn pool_place(&self, pool: &Pool) -> ([f32; 2], f32) {
        let river = &self.base().rivers()[pool.river as usize];
        let point = pool.point as usize;
        (river.points[point], pool.level - river.surface[point])
    }
}
