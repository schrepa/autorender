use engine::prelude::*;
use engine_core::{GpuStruct, gpu_struct};

pub const MAX_HEAP_PAGES: u32 = (128 * 1024 * 1024 / engine_meshlet::PAGE_SIZE_BYTES) as u32;
pub const MAX_PHASE_CLUSTERS: u32 = 131_072;
pub const MAX_VIS_CLUSTERS: u32 = 524_288;
pub const MAX_OCC_INSTANCES: u32 = 4096;
pub const MAX_OCC_CLUSTERS: u32 = 262_144;
pub const DEDUP_SLOTS: u32 = 262_144;
pub const BVH_QUEUE_CAP: u32 = 262_144;
pub const CAND_CAP: u32 = 524_288;
pub const MAX_HIZ_MIPS: usize = 11;
pub const MAX_HIZ_DIM: u32 = 512;
pub const SHADOW_C0_DIM: u32 = 2048;
pub const SHADOW_C1_DIM: u32 = 2048;
pub const SHADOW_LIST_CAP: u32 = 131_072;
pub const UNMAPPED_PAGE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ViewMode {
    #[default]
    Lit,
    Meshlets,
    Triangles,
    Lod,
    HwSw,
    Pages,
}

impl ViewMode {
    pub const ALL: [Self; 6] = [
        Self::Lit,
        Self::Meshlets,
        Self::Triangles,
        Self::Lod,
        Self::HwSw,
        Self::Pages,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Lit => "LIT",
            Self::Meshlets => "MESHLETS",
            Self::Triangles => "TRIS",
            Self::Lod => "LOD",
            Self::HwSw => "HW/SW",
            Self::Pages => "PAGES",
        }
    }

    #[must_use]
    pub const fn index(self) -> u32 {
        match self {
            Self::Lit => 0,
            Self::Meshlets => 1,
            Self::Triangles => 2,
            Self::Lod => 3,
            Self::HwSw => 4,
            Self::Pages => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HeapBudget {
    Mb16,
    Mb32,
    Mb64,
    #[default]
    Mb128,
}

impl HeapBudget {
    #[must_use]
    pub const fn pages(self) -> u32 {
        (self.mb() * 1024 * 1024) / (engine_meshlet::PAGE_SIZE_BYTES as u32)
    }

    #[must_use]
    pub const fn mb(self) -> u32 {
        match self {
            Self::Mb16 => 16,
            Self::Mb32 => 32,
            Self::Mb64 => 64,
            Self::Mb128 => 128,
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Mb16 => "HEAP 16M",
            Self::Mb32 => "HEAP 32M",
            Self::Mb64 => "HEAP 64M",
            Self::Mb128 => "HEAP 128M",
        }
    }

    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Mb16 => Self::Mb32,
            Self::Mb32 => Self::Mb64,
            Self::Mb64 => Self::Mb128,
            Self::Mb128 => Self::Mb16,
        }
    }

    #[must_use]
    pub const fn next_within(self, max_pages: u32) -> Self {
        let cand = self.next();
        if cand.pages() <= max_pages {
            cand
        } else {
            Self::Mb16
        }
    }

    #[must_use]
    pub const fn clamp_to_pages(self, max_pages: u32) -> Self {
        if self.pages() <= max_pages {
            self
        } else if Self::Mb64.pages() <= max_pages {
            Self::Mb64
        } else if Self::Mb32.pages() <= max_pages {
            Self::Mb32
        } else {
            Self::Mb16
        }
    }
}

#[must_use]
pub const fn safe_heap_pages(storage_room: u64) -> u32 {
    let page_bytes = engine_meshlet::PAGE_SIZE_BYTES as u64;
    let max_bytes = (MAX_HEAP_PAGES as u64) * page_bytes;
    let room = if storage_room == 0 {
        max_bytes
    } else if cfg!(target_arch = "wasm32") && storage_room <= 128 * 1024 * 1024 {
        storage_room / 2
    } else {
        storage_room
    };
    let pages = room / page_bytes;
    let hard_cap = if storage_room == 0 {
        MAX_HEAP_PAGES as u64
    } else {
        storage_room / page_bytes
    };
    let clamped = if pages > MAX_HEAP_PAGES as u64 {
        MAX_HEAP_PAGES as u64
    } else if pages < 64 {
        64
    } else {
        pages
    };
    if clamped > hard_cap && hard_cap >= 64 {
        hard_cap as u32
    } else {
        clamped as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LodPreset {
    Px1,
    #[default]
    Px2,
    Px4,
}

impl LodPreset {
    #[must_use]
    pub const fn threshold_px(self) -> f32 {
        match self {
            Self::Px1 => 1.0,
            Self::Px2 => {
                #[cfg(target_arch = "wasm32")]
                {
                    3.0
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    2.0
                }
            }
            Self::Px4 => {
                #[cfg(target_arch = "wasm32")]
                {
                    5.0
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    4.0
                }
            }
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Px1 => "LOD 1PX",
            Self::Px2 => "LOD 2PX",
            Self::Px4 => "LOD 4PX",
        }
    }

    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Px1 => Self::Px2,
            Self::Px2 => Self::Px4,
            Self::Px4 => Self::Px1,
        }
    }
}

gpu_struct! {
    pub struct NaniteUniforms {
        pub render_view_proj: [[f32; 4]; 4],
        pub cull_view_proj: [[f32; 4]; 4],
        pub cull_view: [[f32; 4]; 4],
        pub cull_proj: [[f32; 4]; 4],
        pub inv_render_vp: [[f32; 4]; 4],
        pub cull_pos_near: [f32; 4],
        pub render_pos_time: [f32; 4],
        pub viewport_lod: [f32; 4],
        pub hiz_params: [f32; 4],
        pub counts: [u32; 4],
        pub flags: [u32; 4],
        pub plane0: [f32; 4],
        pub plane1: [f32; 4],
        pub plane2: [f32; 4],
        pub plane3: [f32; 4],
        pub plane4: [f32; 4],
        pub plane5: [f32; 4],
        pub shadow_vp0: [[f32; 4]; 4],
        pub shadow_vp1: [[f32; 4]; 4],
        pub shadow_info: [f32; 4],
        pub shadow_center0: [f32; 4],
        pub shadow_center1: [f32; 4],
        pub extra: [u32; 4],
        pub prev_cull_view: [[f32; 4]; 4],
        pub prev_cull_proj: [[f32; 4]; 4],
    }
}

gpu_struct! {
    pub struct NaniteInstance {
        pub pos_scale: [f32; 4],
        pub rot_quat: [f32; 4],
        pub bound_sphere: [f32; 4],
        pub info: [u32; 4],
    }
}

gpu_struct! {
    pub struct GpuClusterHeader {
        pub bound_sphere: [f32; 4],
        pub lod_sphere: [f32; 4],
        pub parent_lod_sphere: [f32; 4],
        pub errors_and_cone: [u32; 4],
        pub page_and_fallback: [u32; 4],
    }
}

gpu_struct! {
    pub struct GpuBvhNode {
        pub sphere: [f32; 4],
        pub child_or_cluster: [u32; 4],
        pub max_parent_error: [f32; 4],
        pub min_self_error: [f32; 4],
        pub info: [u32; 4],
    }
}

gpu_struct! {
    pub struct VisibleCluster {
        pub instance_id: u32,
        pub cluster_id: u32,
        pub heap_slot: u32,
        pub vis_slot: u32,
    }
}

gpu_struct! {
    pub struct HizBuildInfo {
        pub src_size: [u32; 2],
        pub mip0_size: [u32; 2],
    }
}

pub const GENERATED: &[&str] = &[
    <NaniteUniforms as GpuStruct>::WGSL,
    <NaniteInstance as GpuStruct>::WGSL,
    <GpuClusterHeader as GpuStruct>::WGSL,
    <GpuBvhNode as GpuStruct>::WGSL,
    <VisibleCluster as GpuStruct>::WGSL,
    <HizBuildInfo as GpuStruct>::WGSL,
];

#[derive(Resource, Clone, Copy, Debug)]
pub struct NaniteCamera {
    pub render_pos: Vec3,
    pub render_view: Mat4,
    pub render_proj: Mat4,
    pub cull_pos: Vec3,
    pub cull_view: Mat4,
    pub cull_proj: Mat4,
    pub near_z: f32,
    pub sim_time: f32,
    pub sun_dir: Vec3,
}

impl Default for NaniteCamera {
    fn default() -> Self {
        let pos = Vec3::new(0.0, 7.5, 160.0);
        let view = Mat4::look_at_rh(pos, Vec3::new(0.0, 14.0, -215.0), Vec3::Y);
        let proj = reverse_z_infinite_proj(42.0_f32.to_radians(), 16.0 / 9.0, 0.1);
        Self {
            render_pos: pos,
            render_view: view,
            render_proj: proj,
            cull_pos: pos,
            cull_view: view,
            cull_proj: proj,
            near_z: 0.1,
            sim_time: 0.0,
            sun_dir: Vec3::new(0.28, 0.86, 0.42).normalize(),
        }
    }
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct NaniteConfig {
    pub view_mode: ViewMode,
    pub heap_budget: HeapBudget,
    pub lod_preset: LodPreset,
    pub freeze_culling: bool,
    pub hw_only: bool,
    pub sw_max_px: f32,
    pub max_pages_per_frame: usize,
    pub shadows_enabled: bool,
    pub render_scale: f32,
    pub shadow_c0_half: f32,
    pub shadow_c0_forward: f32,
    pub shadow_c0_depth: f32,
    pub shadow_c1_half: f32,
    pub shadow_c1_center: Vec3,
    pub shadow_c1_depth: f32,
    pub double_sided_materials: u32,
}

impl Default for NaniteConfig {
    fn default() -> Self {
        Self {
            view_mode: ViewMode::Lit,
            heap_budget: HeapBudget::Mb128,
            lod_preset: LodPreset::Px2,
            freeze_culling: false,
            hw_only: true,
            sw_max_px: 2.0,
            max_pages_per_frame: 96,
            shadows_enabled: true,
            render_scale: 1.0,
            shadow_c0_half: 14.0,
            shadow_c0_forward: 6.0,
            shadow_c0_depth: 48.0,
            shadow_c1_half: 32.0,
            shadow_c1_center: Vec3::new(1.0, 6.0, 0.0),
            shadow_c1_depth: 64.0,
            double_sided_materials: 0,
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct NaniteTelemetry {
    pub total_virtual_tris: u64,
    pub total_pages: u32,
    pub resident_pages: u32,
    pub pinned_pages: u32,
    pub budget_pages: u32,
    pub pending_fetches: u32,
    pub pending_requests: u32,
    pub ready_pages: u32,
    pub inst_p1_vis: u32,
    pub inst_p1_occ: u32,
    pub inst_p2_vis: u32,
    pub clusters_tested: u32,
    pub hw_clusters: u32,
    pub sw_clusters: u32,
    pub tris_hw: u32,
    pub tris_sw: u32,
    pub page_faults: u32,
    pub fallback_draws: u32,
    pub cone_culled: u32,
    pub frustum_culled: u32,
    pub hiz_culled: u32,
}

impl NaniteTelemetry {
    #[must_use]
    pub fn visible_tris(&self) -> u32 {
        self.tris_hw.saturating_add(self.tris_sw)
    }

    #[must_use]
    pub fn hw_percent(&self) -> u32 {
        let total = self.visible_tris();
        if total == 0 {
            return 0;
        }
        ((u64::from(self.tris_hw) * 100 + u64::from(total) / 2) / u64::from(total)) as u32
    }

    #[must_use]
    pub fn sw_percent(&self) -> u32 {
        let total = self.visible_tris();
        if total == 0 {
            return 0;
        }
        100_u32.saturating_sub(self.hw_percent())
    }
}

#[must_use]
pub fn reverse_z_infinite_proj(fov_y_rad: f32, aspect: f32, near: f32) -> Mat4 {
    let f = 1.0 / (fov_y_rad * 0.5).tan();
    let safe_aspect = if aspect.is_finite() && aspect > 1.0e-4 {
        aspect
    } else {
        1.0
    };
    Mat4::from_cols(
        Vec4::new(f / safe_aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, -1.0),
        Vec4::new(0.0, 0.0, near, 0.0),
    )
}

#[must_use]
pub fn extract_frustum_planes(vp: Mat4) -> [[f32; 4]; 6] {
    let cols = vp.to_cols_array_2d();
    let r0 = Vec4::new(cols[0][0], cols[1][0], cols[2][0], cols[3][0]);
    let r1 = Vec4::new(cols[0][1], cols[1][1], cols[2][1], cols[3][1]);
    let r2 = Vec4::new(cols[0][2], cols[1][2], cols[2][2], cols[3][2]);
    let r3 = Vec4::new(cols[0][3], cols[1][3], cols[2][3], cols[3][3]);
    let raw = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2, r2];
    let mut out = [[0.0_f32; 4]; 6];
    for (i, plane) in raw.into_iter().enumerate() {
        let len = plane.truncate().length();
        let norm = if len > 1.0e-6 { plane / len } else { plane };
        out[i] = norm.to_array();
    }
    out
}

#[must_use]
pub const fn prev_pow2(value: u32) -> u32 {
    if value <= 1 {
        1
    } else {
        1_u32 << (31 - value.leading_zeros())
    }
}

#[must_use]
pub const fn hiz_dimensions(src_w: u32, src_h: u32) -> (u32, u32, u32) {
    let sw = if src_w > 0 { src_w } else { 1 };
    let sh = if src_h > 0 { src_h } else { 1 };
    let mut w = prev_pow2(sw);
    let mut h = prev_pow2(sh);
    while w > MAX_HIZ_DIM || h > MAX_HIZ_DIM {
        if w > 1 {
            w >>= 1;
        }
        if h > 1 {
            h >>= 1;
        }
    }
    let max_d = if w >= h { w } else { h };
    let raw_mips = max_d.ilog2() + 1;
    let max_mips = MAX_HIZ_MIPS as u32;
    let mips = if raw_mips > max_mips { max_mips } else { raw_mips };
    (w, h, mips)
}

#[must_use]
pub const fn mip_extent(mip0_w: u32, mip0_h: u32, mip: u32) -> (u32, u32) {
    let shift = if mip > 31 { 31 } else { mip };
    let w = mip0_w >> shift;
    let h = mip0_h >> shift;
    (if w > 0 { w } else { 1 }, if h > 0 { h } else { 1 })
}
