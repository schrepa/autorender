use engine::prelude::*;

use crate::types::{NaniteCamera, NaniteConfig};
use crate::world::NaniteWorld;

pub(crate) struct ShadowTrack {
    pub last_c0_center: Vec3,
    pub last_c1_center: Vec3,
    pub last_sun_dir: Vec3,
    pub last_dim: u32,
    pub last_c0_res: u64,
    pub last_c1_res: u64,
    pub last_geom_rev: u64,
    pub last_inst_rev: u64,
    pub dirty_c0: bool,
    pub dirty_c1: bool,
    pub fresh_tex: bool,
    pub enabled: bool,
}

impl Default for ShadowTrack {
    fn default() -> Self {
        Self {
            last_c0_center: Vec3::splat(1.0e9),
            last_c1_center: Vec3::splat(1.0e9),
            last_sun_dir: Vec3::ZERO,
            last_dim: 0,
            last_c0_res: u64::MAX,
            last_c1_res: u64::MAX,
            last_geom_rev: u64::MAX,
            last_inst_rev: u64::MAX,
            dirty_c0: false,
            dirty_c1: false,
            fresh_tex: true,
            enabled: false,
        }
    }
}

pub(crate) struct ShadowFrame {
    pub vp0: Mat4,
    pub vp1: Mat4,
    pub center0: [f32; 4],
    pub center1: [f32; 4],
    pub dim: u32,
}

fn shadow_res_tier(residency_ver: u64, total_pages: u32) -> u64 {
    let cap = u64::from(total_pages.max(1));
    if residency_ver == 0 {
        0
    } else if residency_ver < cap.min(64) {
        1
    } else if residency_ver < cap.min(256) {
        2
    } else {
        3 + (residency_ver / 512)
    }
}

fn ortho_shadow_vp(center: Vec3, half_extent: f32, depth_half: f32, sun_dir: Vec3, dim: u32) -> Mat4 {
    let dir = sun_dir.normalize_or_zero();
    let up = if dir.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
    let eye = center + dir * depth_half;
    let view = Mat4::look_at_rh(eye, center, up);
    let texel = (half_extent * 2.0) / (dim as f32);
    let snapped_c = view.transform_point3(center);
    let sx = (snapped_c.x / texel).floor() * texel - snapped_c.x;
    let sy = (snapped_c.y / texel).floor() * texel - snapped_c.y;
    let shift = Mat4::from_translation(Vec3::new(sx, sy, 0.0));
    let ortho = Mat4::orthographic_rh(
        -half_extent,
        half_extent,
        -half_extent,
        half_extent,
        depth_half * 2.0,
        0.0,
    );
    ortho * shift * view
}

pub(crate) fn build_shadow_jobs(world: &NaniteWorld) -> (Vec<[u32; 2]>, u32, u32) {
    let mut jobs = Vec::new();
    for (inst_id, inst) in world.instances.iter().enumerate() {
        let Some(mesh) = world.meshes.get(inst.info[3] as usize) else {
            continue;
        };
        let [start, count] = mesh.shadow_c0_range;
        for &cid in world
            .shadow_cut_clusters
            .iter()
            .skip(start as usize)
            .take(count as usize)
        {
            jobs.push([inst_id as u32, cid]);
        }
    }
    let c0_count = jobs.len() as u32;
    for (inst_id, inst) in world.instances.iter().enumerate() {
        let Some(mesh) = world.meshes.get(inst.info[3] as usize) else {
            continue;
        };
        let [start, count] = mesh.shadow_c1_range;
        for &cid in world
            .shadow_cut_clusters
            .iter()
            .skip(start as usize)
            .take(count as usize)
        {
            jobs.push([inst_id as u32, cid]);
        }
    }
    let c1_count = (jobs.len() as u32).saturating_sub(c0_count);
    (jobs, c0_count, c1_count)
}

impl ShadowTrack {
    pub(crate) fn update(
        &mut self,
        world: &NaniteWorld,
        cam: &NaniteCamera,
        cfg: &NaniteConfig,
        instance_count: u32,
        residency_ver: u64,
    ) -> ShadowFrame {
        self.enabled = cfg.shadow.enabled && instance_count > 0;
        if !cfg.shadow.enabled {
            self.dirty_c0 = false;
            self.dirty_c1 = false;
            return ShadowFrame {
                vp0: Mat4::IDENTITY,
                vp1: Mat4::IDENTITY,
                center0: [0.0; 4],
                center1: [0.0; 4],
                dim: 1,
            };
        }
        let dim = cfg.shadow.effective_dim();
        let c0_half = engine::math::max(cfg.shadow.c0_half, 2.0);
        let c1_half = engine::math::max(cfg.shadow.c1_half, 4.0);
        let vm = cam.render_view.to_cols_array_2d();
        let fwd_xz = Vec3::new(-vm[0][2], 0.0, -vm[2][2]).normalize_or_zero();
        let c0_y = match cfg.shadow.c0_y_bounds {
            Some([lo, hi]) => engine::math::clamp(cam.render_pos.y, lo, hi),
            None => cam.render_pos.y,
        };
        let desired_c0 = Vec3::new(
            cam.render_pos.x + fwd_xz.x * cfg.shadow.c0_forward,
            c0_y,
            cam.render_pos.z + fwd_xz.z * cfg.shadow.c0_forward,
        );
        let desired_c1 = cfg.shadow.c1_center.unwrap_or(cam.render_pos);

        let sun_dir = cam.sun_dir.normalize_or_zero();
        let sun_changed = (sun_dir - self.last_sun_dir).length_squared() > 1.0e-6;
        let dim_changed = dim != self.last_dim;
        let geom_changed = self.last_geom_rev != world.geometry_revision
            || self.last_inst_rev != world.instance_revision;
        if geom_changed || self.fresh_tex || sun_changed || dim_changed {
            self.last_geom_rev = world.geometry_revision;
            self.last_inst_rev = world.instance_revision;
            self.last_sun_dir = sun_dir;
            self.last_dim = dim;
            self.last_c0_res = u64::MAX;
            self.last_c1_res = u64::MAX;
        }
        let c0_step = engine::math::max(c0_half * 0.25, 1.5);
        let c0_moved = (desired_c0 - self.last_c0_center).length_squared() > c0_step * c0_step;
        let c1_step = engine::math::max(c1_half * 0.25, 4.0);
        let c1_moved = (desired_c1 - self.last_c1_center).length_squared() > c1_step * c1_step;
        let tier = shadow_res_tier(residency_ver, world.total_pages);
        let c0_res_changed =
            self.last_c0_res == u64::MAX || (tier > 0 && tier != self.last_c0_res);
        let c1_res_changed =
            self.last_c1_res == u64::MAX || (tier > 0 && tier != self.last_c1_res);
        self.dirty_c0 = (self.enabled && (c0_moved || c0_res_changed)) || self.fresh_tex;
        self.dirty_c1 = (self.enabled && (c1_moved || c1_res_changed)) || self.fresh_tex;
        if self.dirty_c0 && self.enabled {
            self.last_c0_center = desired_c0;
            self.last_c0_res = tier;
        }
        if self.dirty_c1 && self.enabled {
            self.last_c1_center = desired_c1;
            self.last_c1_res = tier;
        }
        let c0_center = if self.last_c0_center.length_squared() < 1.0e14 {
            self.last_c0_center
        } else {
            desired_c0
        };
        let c1_center = if self.last_c1_center.length_squared() < 1.0e14 {
            self.last_c1_center
        } else {
            desired_c1
        };
        let vp0 = ortho_shadow_vp(
            c0_center,
            c0_half,
            engine::math::max(cfg.shadow.c0_depth, 20.0),
            sun_dir,
            dim,
        );
        let vp1 = ortho_shadow_vp(
            c1_center,
            c1_half,
            engine::math::max(cfg.shadow.c1_depth, 40.0),
            sun_dir,
            dim,
        );
        ShadowFrame {
            vp0,
            vp1,
            center0: [c0_center.x, c0_center.y, c0_center.z, c0_half],
            center1: [c1_center.x, c1_center.y, c1_center.z, c1_half],
            dim,
        }
    }
}
