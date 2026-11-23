use bevy_ecs::system::SystemParamItem;
use engine::prelude::*;
use engine_core::{DeviceLimits, Entry, Error, PassViews, Var, ViewFacts};
use engine_meshlet::PAGE_SIZE_BYTES;
use engine_render::{
    Access, Allocation, Bind, Buffer, ComputePipeline, DepthUse, Encode, GpuPart, Install,
    Lifetime, Own, Pass, PerBatch, Pipeline, Prepare, Readback, RenderPipeline, SharedTarget,
    TargetClear, Texture, wgpu,
};

use crate::bridge::NaniteBridgeHandle;
use crate::shaders::nanite::{cull, frame, hiz, hw, shadow, sw};
use crate::types::{
    BVH_QUEUE_CAP, CAND_CAP, DEDUP_SLOTS, GpuBvhNode, GpuClusterHeader, HizBuildInfo,
    MAX_HEAP_PAGES, MAX_HIZ_MIPS, MAX_OCC_CLUSTERS, MAX_OCC_INSTANCES, MAX_PHASE_CLUSTERS,
    MAX_VIS_CLUSTERS, NaniteCamera, NaniteConfig, NaniteInstance, NaniteUniforms, SHADOW_C0_DIM,
    SHADOW_C1_DIM, SHADOW_LIST_CAP, VisibleCluster, extract_frustum_planes, hiz_dimensions,
    mip_extent, safe_heap_pages,
};
use crate::world::NaniteWorld;
use crate::NANITE_VIS_HW;

const HIZ_WG: u32 = 8;
const DEDUP_BASE_WORDS: u64 = 32 + MAX_OCC_INSTANCES as u64 + (MAX_OCC_CLUSTERS as u64) * 2;
const CULL_STATE_WORDS: u64 = DEDUP_BASE_WORDS + DEDUP_SLOTS as u64;
const BVH_SCRATCH_WORDS: u64 = (BVH_QUEUE_CAP as u64) * 4 + (CAND_CAP as u64) * 2;
const INDIRECT_BYTES: u64 = 256;

struct NaniteBuffers {
    uniforms: Buffer<NaniteGpu>,
    hiz_info: Buffer<NaniteGpu>,
    instances: Buffer<NaniteGpu>,
    clusters: Buffer<NaniteGpu>,
    bvh: Buffer<NaniteGpu>,
    page_table: Buffer<NaniteGpu>,
    page_heap: Buffer<NaniteGpu>,
    page_requests: Buffer<NaniteGpu>,
    vis_clusters: Buffer<NaniteGpu>,
    p1_draw_lists: Buffer<NaniteGpu>,
    p2_draw_lists: Buffer<NaniteGpu>,
    bvh_scratch: Buffer<NaniteGpu>,
    cull_state: Buffer<NaniteGpu>,
    indirect_p1: Buffer<NaniteGpu>,
    indirect_p2: Buffer<NaniteGpu>,
    sw_depth: Buffer<NaniteGpu>,
    sw_vis: Buffer<NaniteGpu>,
    shadow_list: Buffer<NaniteGpu>,
    shadow_indirect: Buffer<NaniteGpu>,
    shadow_jobs: Buffer<NaniteGpu>,
    depth_tex: Texture<NaniteGpu>,
    shadow_c0: Texture<NaniteGpu>,
    shadow_c1: Texture<NaniteGpu>,
    hiz_tex: Texture<NaniteGpu>,
    stats_rb: Readback<NaniteGpu>,
    req_rb: Readback<NaniteGpu>,
}

struct CullPipes {
    reset: ComputePipeline<NaniteGpu>,
    inst_p1: ComputePipeline<NaniteGpu>,
    prep_bvh_a_p1: ComputePipeline<NaniteGpu>,
    prep_bvh_b_p1: ComputePipeline<NaniteGpu>,
    trav_ab_p1: ComputePipeline<NaniteGpu>,
    trav_ba_p1: ComputePipeline<NaniteGpu>,
    dfs_cut_p1: ComputePipeline<NaniteGpu>,
    prep_cand_p1: ComputePipeline<NaniteGpu>,
    clus_p1: ComputePipeline<NaniteGpu>,
    prep_draws_p1: ComputePipeline<NaniteGpu>,
    inst_p2: ComputePipeline<NaniteGpu>,
    prep_bvh_a_p2: ComputePipeline<NaniteGpu>,
    prep_bvh_b_p2: ComputePipeline<NaniteGpu>,
    trav_ab_p2: ComputePipeline<NaniteGpu>,
    trav_ba_p2: ComputePipeline<NaniteGpu>,
    dfs_cut_p2: ComputePipeline<NaniteGpu>,
    prep_cand_p2: ComputePipeline<NaniteGpu>,
    clus_p2: ComputePipeline<NaniteGpu>,
    occ_clus_p2: ComputePipeline<NaniteGpu>,
    prep_draws_p2: ComputePipeline<NaniteGpu>,
}

struct RasterPipes {
    hw_p1: Pipeline<NaniteGpu, Own>,
    hw_ds_p1: Pipeline<NaniteGpu, Own>,
    hw_p2: Pipeline<NaniteGpu, Own>,
    hw_ds_p2: Pipeline<NaniteGpu, Own>,
    sw_seed_p1: ComputePipeline<NaniteGpu>,
    sw_merge_p2: ComputePipeline<NaniteGpu>,
    sw_depth_p1: ComputePipeline<NaniteGpu>,
    sw_vis_p1: ComputePipeline<NaniteGpu>,
    sw_depth_p2: ComputePipeline<NaniteGpu>,
    sw_vis_p2: ComputePipeline<NaniteGpu>,
    hiz_p1_hw_mip0: ComputePipeline<NaniteGpu>,
    hiz_p1_sw_mip0: ComputePipeline<NaniteGpu>,
    hiz_p1_down: Vec<ComputePipeline<NaniteGpu>>,
    hiz_final_hw_mip0: ComputePipeline<NaniteGpu>,
    hiz_final_sw_mip0: ComputePipeline<NaniteGpu>,
    hiz_final_clear: ComputePipeline<NaniteGpu>,
    hiz_final_down: Vec<ComputePipeline<NaniteGpu>>,
    shadow_collect_c0: ComputePipeline<NaniteGpu>,
    shadow_collect_c1: ComputePipeline<NaniteGpu>,
    shadow_prep: ComputePipeline<NaniteGpu>,
    shadow_draw_c0: Pipeline<NaniteGpu, Own>,
    shadow_draw_c1: Pipeline<NaniteGpu, Own>,
}

struct NanitePasses {
    shadow_cull: Pass<NaniteGpu>,
    shadow_c0: Pass<NaniteGpu>,
    shadow_c1: Pass<NaniteGpu>,
    cull_p1: Pass<NaniteGpu>,
    hw_p1: Pass<NaniteGpu>,
    sw_p1: Pass<NaniteGpu>,
    hiz_p1: Pass<NaniteGpu>,
    cull_p2: Pass<NaniteGpu>,
    hw_p2: Pass<NaniteGpu>,
    sw_p2: Pass<NaniteGpu>,
    hiz_final: Pass<NaniteGpu>,
}

struct ShadowTrack {
    last_c0_center: Vec3,
    last_c0_res: u64,
    last_c1_res: u64,
    last_geom_rev: u64,
    last_inst_rev: u64,
    dirty_c0: bool,
    dirty_c1: bool,
    fresh_tex: bool,
    enabled: bool,
}

struct HizTrack {
    prev_view: Mat4,
    prev_proj: Mat4,
    curr_view: Mat4,
    curr_proj: Mat4,
}

pub(crate) struct NaniteGpu {
    buf: NaniteBuffers,
    passes: NanitePasses,
    cull: CullPipes,
    raster: RasterPipes,
    shadow: ShadowTrack,
    hiz: HizTrack,
    geom_rev: u64,
    inst_rev: u64,
    page_count: u32,
    instance_count: u32,
    cluster_count: u32,
    shadow_c0_count: u32,
    shadow_c1_count: u32,
    last_page_table: Vec<u32>,
    canvas: (u32, u32),
    hiz_dims: (u32, u32, u32),
    frame_gen: u32,
    hiz_ready: bool,
    freeze: bool,
    hw_only: bool,
    skip_p2: bool,
    static_frames: u32,
    last_cull_vp: [[f32; 4]; 4],
    last_lod_bits: u32,
    last_hw_only: bool,
    last_freeze: bool,
    last_residency_ver: u64,
    residency_ver: u64,
    heap_cap_pages: u32,
}

fn pick(
    cx: &mut Install<'_, NaniteGpu>,
    entries: &[Entry],
    all: &[(Var, Bind<NaniteGpu>)],
) -> Vec<(Var, Bind<NaniteGpu>)> {
    all.iter()
        .filter(|(var, _)| entries.iter().any(|entry| cx.reaches(*entry, *var)))
        .map(|(var, bind)| (*var, bind.clone()))
        .collect()
}

fn alloc_buffers(cx: &mut Install<'_, NaniteGpu>) -> NaniteBuffers {
    NaniteBuffers {
        uniforms: cx.buffer("nanite_uniforms", Lifetime::Kept),
        hiz_info: cx.buffer("nanite_hiz_info", Lifetime::Kept),
        instances: cx.buffer("nanite_instances", Lifetime::Kept),
        clusters: cx.buffer("nanite_clusters", Lifetime::Kept),
        bvh: cx.buffer("nanite_bvh", Lifetime::Kept),
        page_table: cx.buffer("nanite_page_table", Lifetime::Kept),
        page_heap: cx.buffer("nanite_page_heap", Lifetime::Kept),
        page_requests: cx.buffer("nanite_page_requests", Lifetime::Kept),
        vis_clusters: cx.buffer("nanite_vis_clusters", Lifetime::Kept),
        p1_draw_lists: cx.buffer("nanite_p1_draw_lists", Lifetime::Kept),
        p2_draw_lists: cx.buffer("nanite_p2_draw_lists", Lifetime::Kept),
        bvh_scratch: cx.buffer("nanite_bvh_scratch", Lifetime::Kept),
        cull_state: cx.buffer("nanite_cull_state", Lifetime::Kept),
        indirect_p1: cx.indirect_buffer("nanite_indirect_p1", Lifetime::Kept),
        indirect_p2: cx.indirect_buffer("nanite_indirect_p2", Lifetime::Kept),
        sw_depth: cx.buffer("nanite_sw_depth", Lifetime::Kept),
        sw_vis: cx.buffer("nanite_sw_vis", Lifetime::Kept),
        shadow_list: cx.buffer("nanite_shadow_list", Lifetime::Kept),
        shadow_indirect: cx.indirect_buffer("nanite_shadow_indirect", Lifetime::Kept),
        shadow_jobs: cx.buffer("nanite_shadow_jobs", Lifetime::Kept),
        depth_tex: cx.texture(
            "nanite_depth_tex",
            wgpu::TextureFormat::Depth32Float,
            Lifetime::Kept,
        ),
        shadow_c0: cx.texture(
            "nanite_shadow_c0",
            wgpu::TextureFormat::Depth32Float,
            Lifetime::Kept,
        ),
        shadow_c1: cx.texture(
            "nanite_shadow_c1",
            wgpu::TextureFormat::Depth32Float,
            Lifetime::Kept,
        ),
        hiz_tex: cx.texture("nanite_hiz", wgpu::TextureFormat::R32Float, Lifetime::Kept),
        stats_rb: cx.readback("nanite_stats_rb"),
        req_rb: cx.readback("nanite_req_rb"),
    }
}

fn declare_targets_and_frame(
    cx: &mut Install<'_, NaniteGpu>,
    buf: &NaniteBuffers,
) -> Result<(), Error> {
    cx.shared_target(SharedTarget {
        name: NANITE_VIS_HW,
        format: wgpu::TextureFormat::R32Uint,
        samples: 1,
        clear: TargetClear::Color([0.0, 0.0, 0.0, 0.0]),
    });
    cx.frame(frame::NANITE_FRAME_UNIFORMS, Bind::Buffer(buf.uniforms))?;
    cx.frame(frame::NANITE_FRAME_INSTANCES, Bind::Buffer(buf.instances))?;
    cx.frame(frame::NANITE_FRAME_CLUSTERS, Bind::Buffer(buf.clusters))?;
    cx.frame(frame::NANITE_FRAME_PAGE_HEAP, Bind::Buffer(buf.page_heap))?;
    cx.frame(frame::NANITE_FRAME_VIS_CLUSTERS, Bind::Buffer(buf.vis_clusters))?;
    cx.frame(frame::NANITE_FRAME_SW_DEPTH, Bind::Buffer(buf.sw_depth))?;
    cx.frame(frame::NANITE_FRAME_SW_VIS, Bind::Buffer(buf.sw_vis))?;
    cx.frame(frame::NANITE_FRAME_DEPTH, Bind::Texture(buf.depth_tex))?;
    cx.frame(frame::NANITE_FRAME_SHADOW_C0, Bind::Texture(buf.shadow_c0))?;
    cx.frame(frame::NANITE_FRAME_SHADOW_C1, Bind::Texture(buf.shadow_c1))?;
    Ok(())
}

fn install_passes(cx: &mut Install<'_, NaniteGpu>, buf: &NaniteBuffers) -> NanitePasses {
    let shadow_cull = cx
        .pass("nanite_shadow_cull")
        .label("nanite/shadow_cull")
        .span("nanite_shadow")
        .id();
    let shadow_c0 = cx
        .pass("nanite_shadow_c0")
        .label("nanite/shadow_c0")
        .span("nanite_shadow")
        .depth_texture(buf.shadow_c0, Access::Clear, 0.0)
        .id();
    let shadow_c1 = cx
        .pass("nanite_shadow_c1")
        .label("nanite/shadow_c1")
        .span("nanite_shadow")
        .depth_texture(buf.shadow_c1, Access::Clear, 0.0)
        .id();
    let cull_p1 = cx
        .pass("nanite_cull_p1")
        .label("nanite/cull_p1")
        .span("nanite_cull_p1")
        .id();
    let hw_p1 = cx
        .pass("nanite_hw_p1")
        .label("nanite/hw_p1")
        .span("nanite_hw_p1")
        .views(PassViews::Main)
        .target(NANITE_VIS_HW, Access::Clear)
        .depth_texture(buf.depth_tex, Access::Clear, 0.0)
        .id();
    let sw_p1 = cx
        .pass("nanite_sw_p1")
        .label("nanite/sw_p1")
        .span("nanite_sw_p1")
        .id();
    let hiz_p1 = cx
        .pass("nanite_hiz_p1")
        .label("nanite/hiz_p1")
        .span("nanite_hiz_p1")
        .id();
    let cull_p2 = cx
        .pass("nanite_cull_p2")
        .label("nanite/cull_p2")
        .span("nanite_cull_p2")
        .id();
    let hw_p2 = cx
        .pass("nanite_hw_p2")
        .label("nanite/hw_p2")
        .span("nanite_hw_p2")
        .views(PassViews::Main)
        .target(NANITE_VIS_HW, Access::Load)
        .depth_texture(buf.depth_tex, Access::Load, 0.0)
        .id();
    let sw_p2 = cx
        .pass("nanite_sw_p2")
        .label("nanite/sw_p2")
        .span("nanite_sw_p2")
        .id();
    let hiz_final = cx
        .pass("nanite_hiz_final")
        .label("nanite/hiz_final")
        .span("nanite_hiz_final")
        .id();

    cx.after(shadow_c0, shadow_cull);
    cx.after(shadow_c1, shadow_cull);
    cx.after(cull_p1, shadow_c0);
    cx.after(cull_p1, shadow_c1);
    cx.after(hw_p1, cull_p1);
    cx.after(sw_p1, hw_p1);
    cx.after(hiz_p1, hw_p1);
    cx.after(hiz_p1, sw_p1);
    cx.after(cull_p2, hiz_p1);
    cx.after(hw_p2, cull_p2);
    cx.after(sw_p2, hw_p2);
    cx.after(hiz_final, hw_p2);
    cx.after(hiz_final, sw_p2);

    NanitePasses {
        shadow_cull,
        shadow_c0,
        shadow_c1,
        cull_p1,
        hw_p1,
        sw_p1,
        hiz_p1,
        cull_p2,
        hw_p2,
        sw_p2,
        hiz_final,
    }
}

fn install_cull_pipes(
    cx: &mut Install<'_, NaniteGpu>,
    passes: &NanitePasses,
    buf: &NaniteBuffers,
) -> Result<CullPipes, Error> {
    let p1_binds = [
        (cull::CULL_UNIFORMS, Bind::Buffer(buf.uniforms)),
        (cull::CULL_HIZ_TEX, Bind::Texture(buf.hiz_tex)),
        (cull::CULL_INSTANCES, Bind::Buffer(buf.instances)),
        (cull::CULL_CLUSTERS, Bind::Buffer(buf.clusters)),
        (cull::CULL_BVH, Bind::Buffer(buf.bvh)),
        (cull::CULL_PAGE_TABLE, Bind::Buffer(buf.page_table)),
        (cull::CULL_BVH_SCRATCH, Bind::Buffer(buf.bvh_scratch)),
        (cull::CULL_VIS_CLUSTERS, Bind::Buffer(buf.vis_clusters)),
        (cull::CULL_DRAW_LISTS, Bind::Buffer(buf.p1_draw_lists)),
        (cull::CULL_STATE, Bind::Buffer(buf.cull_state)),
        (cull::CULL_PAGE_REQUESTS, Bind::Buffer(buf.page_requests)),
        (cull::CULL_INDIRECT, Bind::Buffer(buf.indirect_p1)),
    ];
    let p2_binds = [
        (cull::CULL_UNIFORMS, Bind::Buffer(buf.uniforms)),
        (cull::CULL_HIZ_TEX, Bind::Texture(buf.hiz_tex)),
        (cull::CULL_INSTANCES, Bind::Buffer(buf.instances)),
        (cull::CULL_CLUSTERS, Bind::Buffer(buf.clusters)),
        (cull::CULL_BVH, Bind::Buffer(buf.bvh)),
        (cull::CULL_PAGE_TABLE, Bind::Buffer(buf.page_table)),
        (cull::CULL_BVH_SCRATCH, Bind::Buffer(buf.bvh_scratch)),
        (cull::CULL_VIS_CLUSTERS, Bind::Buffer(buf.vis_clusters)),
        (cull::CULL_DRAW_LISTS, Bind::Buffer(buf.p2_draw_lists)),
        (cull::CULL_STATE, Bind::Buffer(buf.cull_state)),
        (cull::CULL_PAGE_REQUESTS, Bind::Buffer(buf.page_requests)),
        (cull::CULL_INDIRECT, Bind::Buffer(buf.indirect_p2)),
    ];
    let make = |cx: &mut Install<'_, NaniteGpu>,
                pass: Pass<NaniteGpu>,
                entry: Entry,
                all: &[(Var, Bind<NaniteGpu>)]| {
        let b = pick(cx, &[entry], all);
        cx.compute_pipeline(pass, entry, &[], &b)
    };
    Ok(CullPipes {
        reset: make(cx, passes.cull_p1, cull::CS_RESET_FRAME, &p1_binds)?,
        inst_p1: make(cx, passes.cull_p1, cull::CS_CULL_INSTANCES_P1, &p1_binds)?,
        prep_bvh_a_p1: make(cx, passes.cull_p1, cull::CS_PREP_BVH_A, &p1_binds)?,
        prep_bvh_b_p1: make(cx, passes.cull_p1, cull::CS_PREP_BVH_B, &p1_binds)?,
        trav_ab_p1: make(cx, passes.cull_p1, cull::CS_TRAVERSE_BVH_AB, &p1_binds)?,
        trav_ba_p1: make(cx, passes.cull_p1, cull::CS_TRAVERSE_BVH_BA, &p1_binds)?,
        dfs_cut_p1: make(cx, passes.cull_p1, cull::CS_BVH_DFS_CUT, &p1_binds)?,
        prep_cand_p1: make(cx, passes.cull_p1, cull::CS_PREP_CANDIDATES, &p1_binds)?,
        clus_p1: make(cx, passes.cull_p1, cull::CS_CULL_CLUSTERS_P1, &p1_binds)?,
        prep_draws_p1: make(cx, passes.cull_p1, cull::CS_PREP_DRAWS_P1, &p1_binds)?,
        inst_p2: make(cx, passes.cull_p2, cull::CS_CULL_INSTANCES_P2, &p2_binds)?,
        prep_bvh_a_p2: make(cx, passes.cull_p2, cull::CS_PREP_BVH_A, &p2_binds)?,
        prep_bvh_b_p2: make(cx, passes.cull_p2, cull::CS_PREP_BVH_B, &p2_binds)?,
        trav_ab_p2: make(cx, passes.cull_p2, cull::CS_TRAVERSE_BVH_AB, &p2_binds)?,
        trav_ba_p2: make(cx, passes.cull_p2, cull::CS_TRAVERSE_BVH_BA, &p2_binds)?,
        dfs_cut_p2: make(cx, passes.cull_p2, cull::CS_BVH_DFS_CUT, &p2_binds)?,
        prep_cand_p2: make(cx, passes.cull_p2, cull::CS_PREP_CANDIDATES, &p2_binds)?,
        clus_p2: make(cx, passes.cull_p2, cull::CS_CULL_CLUSTERS_P2, &p2_binds)?,
        occ_clus_p2: make(cx, passes.cull_p2, cull::CS_CULL_OCC_CLUSTERS_P2, &p2_binds)?,
        prep_draws_p2: make(cx, passes.cull_p2, cull::CS_PREP_DRAWS_P2, &p2_binds)?,
    })
}

fn install_hiz_downs(
    cx: &mut Install<'_, NaniteGpu>,
    pass: Pass<NaniteGpu>,
    tex: Texture<NaniteGpu>,
) -> Result<Vec<ComputePipeline<NaniteGpu>>, Error> {
    let mut out = Vec::with_capacity(MAX_HIZ_MIPS - 1);
    for mip in 0..(MAX_HIZ_MIPS - 1) as u32 {
        out.push(cx.compute_pipeline(
            pass,
            hiz::CS_HIZ_DOWN,
            &[],
            &[
                (hiz::HIZ_SRC, Bind::TextureMip(tex, mip)),
                (hiz::HIZ_DST, Bind::TextureMip(tex, mip + 1)),
            ],
        )?);
    }
    Ok(out)
}

type HwPipelines = [Pipeline<NaniteGpu, Own>; 4];

fn install_hw_pipes(
    cx: &mut Install<'_, NaniteGpu>,
    passes: &NanitePasses,
    buf: &NaniteBuffers,
) -> Result<HwPipelines, Error> {
    let prim_back = wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: Some(wgpu::Face::Back),
        ..wgpu::PrimitiveState::default()
    };
    let prim_none = wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: None,
        ..wgpu::PrimitiveState::default()
    };
    let depth = DepthUse::Compare {
        compare: wgpu::CompareFunction::GreaterEqual,
        write: true,
        bias: wgpu::DepthBiasState::default(),
    };
    let targets = [(None, wgpu::ColorWrites::ALL)];
    let hw_p1_all = [
        (hw::HW_UNIFORMS, Bind::Buffer(buf.uniforms)),
        (hw::HW_INSTANCES, Bind::Buffer(buf.instances)),
        (hw::HW_CLUSTERS, Bind::Buffer(buf.clusters)),
        (hw::HW_PAGE_HEAP, Bind::Buffer(buf.page_heap)),
        (hw::HW_DRAW_LIST, Bind::Buffer(buf.p1_draw_lists)),
    ];
    let hw_p2_all = [
        (hw::HW_UNIFORMS, Bind::Buffer(buf.uniforms)),
        (hw::HW_INSTANCES, Bind::Buffer(buf.instances)),
        (hw::HW_CLUSTERS, Bind::Buffer(buf.clusters)),
        (hw::HW_PAGE_HEAP, Bind::Buffer(buf.page_heap)),
        (hw::HW_DRAW_LIST, Bind::Buffer(buf.p2_draw_lists)),
    ];
    let make = |cx: &mut Install<'_, NaniteGpu>,
                pass: Pass<NaniteGpu>,
                vs: Entry,
                fs: Entry,
                primitive: wgpu::PrimitiveState,
                all: &[(Var, Bind<NaniteGpu>)]| {
        let b = pick(cx, &[vs, fs], all);
        cx.render_pipeline(
            pass,
            &RenderPipeline {
                vertex: vs,
                fragment: Some(fs),
                vertices: &[],
                primitive,
                depth,
                stencil: wgpu::StencilState::default(),
                samples: 1,
                targets: &targets,
                constants: &[],
                per_batch: PerBatch::NONE,
            },
            &b,
        )
    };
    Ok([
        make(cx, passes.hw_p1, hw::VS_HW_P1, hw::FS_HW_P1, prim_back, &hw_p1_all)?,
        make(cx, passes.hw_p1, hw::VS_HW_DS_P1, hw::FS_HW_P1, prim_none, &hw_p1_all)?,
        make(cx, passes.hw_p2, hw::VS_HW_P2, hw::FS_HW_P2, prim_back, &hw_p2_all)?,
        make(cx, passes.hw_p2, hw::VS_HW_DS_P2, hw::FS_HW_P2, prim_none, &hw_p2_all)?,
    ])
}

struct ShadowPipes {
    collect_c0: ComputePipeline<NaniteGpu>,
    collect_c1: ComputePipeline<NaniteGpu>,
    prep: ComputePipeline<NaniteGpu>,
    draw_c0: Pipeline<NaniteGpu, Own>,
    draw_c1: Pipeline<NaniteGpu, Own>,
}

fn install_shadow_pipes(
    cx: &mut Install<'_, NaniteGpu>,
    passes: &NanitePasses,
    buf: &NaniteBuffers,
) -> Result<ShadowPipes, Error> {
    let sh_all = [
        (shadow::SH_UNIFORMS, Bind::Buffer(buf.uniforms)),
        (shadow::SH_INSTANCES, Bind::Buffer(buf.instances)),
        (shadow::SH_CLUSTERS, Bind::Buffer(buf.clusters)),
        (shadow::SH_PAGE_TABLE, Bind::Buffer(buf.page_table)),
        (shadow::SH_PAGE_HEAP, Bind::Buffer(buf.page_heap)),
        (shadow::SH_JOBS, Bind::Buffer(buf.shadow_jobs)),
        (shadow::SH_DEDUP, Bind::Buffer(buf.cull_state)),
        (shadow::SH_LIST, Bind::Buffer(buf.shadow_list)),
        (shadow::SH_INDIRECT, Bind::Buffer(buf.shadow_indirect)),
        (shadow::SH_LIST_RO, Bind::Buffer(buf.shadow_list)),
    ];
    let make_cs = |cx: &mut Install<'_, NaniteGpu>, entry: Entry| {
        let b = pick(cx, &[entry], &sh_all);
        cx.compute_pipeline(passes.shadow_cull, entry, &[], &b)
    };
    let prim = wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: None,
        ..wgpu::PrimitiveState::default()
    };
    let make_vs = |cx: &mut Install<'_, NaniteGpu>,
                   pass: Pass<NaniteGpu>,
                   entry: Entry,
                   bias: wgpu::DepthBiasState| {
        let b = pick(cx, &[entry], &sh_all);
        cx.render_pipeline(
            pass,
            &RenderPipeline {
                vertex: entry,
                fragment: None,
                vertices: &[],
                primitive: prim,
                depth: DepthUse::Compare {
                    compare: wgpu::CompareFunction::GreaterEqual,
                    write: true,
                    bias,
                },
                stencil: wgpu::StencilState::default(),
                samples: 1,
                targets: &[],
                constants: &[],
                per_batch: PerBatch::NONE,
            },
            &b,
        )
    };
    Ok(ShadowPipes {
        collect_c0: make_cs(cx, shadow::CS_SHADOW_COLLECT_C0)?,
        collect_c1: make_cs(cx, shadow::CS_SHADOW_COLLECT_C1)?,
        prep: make_cs(cx, shadow::CS_SHADOW_PREP)?,
        draw_c0: make_vs(
            cx,
            passes.shadow_c0,
            shadow::VS_SHADOW_C0,
            wgpu::DepthBiasState {
                constant: -2,
                slope_scale: -2.0,
                clamp: 0.0,
            },
        )?,
        draw_c1: make_vs(
            cx,
            passes.shadow_c1,
            shadow::VS_SHADOW_C1,
            wgpu::DepthBiasState {
                constant: -3,
                slope_scale: -2.5,
                clamp: 0.0,
            },
        )?,
    })
}

fn install_raster_pipes(
    cx: &mut Install<'_, NaniteGpu>,
    passes: &NanitePasses,
    buf: &NaniteBuffers,
) -> Result<RasterPipes, Error> {
    let [hw_p1, hw_ds_p1, hw_p2, hw_ds_p2] = install_hw_pipes(cx, passes, buf)?;
    let sw_p1_all = [
        (sw::SW_UNIFORMS, Bind::Buffer(buf.uniforms)),
        (sw::SW_INSTANCES, Bind::Buffer(buf.instances)),
        (sw::SW_CLUSTERS, Bind::Buffer(buf.clusters)),
        (sw::SW_PAGE_HEAP, Bind::Buffer(buf.page_heap)),
        (sw::SW_DRAW_LIST, Bind::Buffer(buf.p1_draw_lists)),
        (sw::SW_HW_DEPTH_TEX, Bind::Texture(buf.depth_tex)),
        (sw::SW_DEPTH_RW, Bind::Buffer(buf.sw_depth)),
        (sw::SW_VIS_RW, Bind::Buffer(buf.sw_vis)),
    ];
    let sw_p2_all = [
        (sw::SW_UNIFORMS, Bind::Buffer(buf.uniforms)),
        (sw::SW_INSTANCES, Bind::Buffer(buf.instances)),
        (sw::SW_CLUSTERS, Bind::Buffer(buf.clusters)),
        (sw::SW_PAGE_HEAP, Bind::Buffer(buf.page_heap)),
        (sw::SW_DRAW_LIST, Bind::Buffer(buf.p2_draw_lists)),
        (sw::SW_HW_DEPTH_TEX, Bind::Texture(buf.depth_tex)),
        (sw::SW_DEPTH_RW, Bind::Buffer(buf.sw_depth)),
        (sw::SW_VIS_RW, Bind::Buffer(buf.sw_vis)),
    ];
    let make_sw = |cx: &mut Install<'_, NaniteGpu>,
                   pass: Pass<NaniteGpu>,
                   entry: Entry,
                   all: &[(Var, Bind<NaniteGpu>)]| {
        let b = pick(cx, &[entry], all);
        cx.compute_pipeline(pass, entry, &[], &b)
    };
    let sw_seed_p1 = make_sw(cx, passes.sw_p1, sw::CS_SEED_SW_DEPTH_P1, &sw_p1_all)?;
    let sw_merge_p2 = make_sw(cx, passes.sw_p2, sw::CS_MERGE_HW_DEPTH_P2, &sw_p2_all)?;
    let sw_depth_p1 = make_sw(cx, passes.sw_p1, sw::CS_SW_DEPTH, &sw_p1_all)?;
    let sw_vis_p1 = make_sw(cx, passes.sw_p1, sw::CS_SW_VIS, &sw_p1_all)?;
    let sw_depth_p2 = make_sw(cx, passes.sw_p2, sw::CS_SW_DEPTH, &sw_p2_all)?;
    let sw_vis_p2 = make_sw(cx, passes.sw_p2, sw::CS_SW_VIS, &sw_p2_all)?;

    let hiz_p1_hw_mip0 = cx.compute_pipeline(
        passes.hiz_p1,
        hiz::CS_HIZ_FROM_HW,
        &[],
        &[
            (hiz::HW_DEPTH_TEX, Bind::Texture(buf.depth_tex)),
            (hiz::HIZ_DST, Bind::TextureMip(buf.hiz_tex, 0)),
        ],
    )?;
    let hiz_p1_sw_mip0 = cx.compute_pipeline(
        passes.hiz_p1,
        hiz::CS_HIZ_FROM_SW,
        &[],
        &[
            (hiz::HIZ_BUILD_INFO, Bind::Buffer(buf.hiz_info)),
            (hiz::SW_DEPTH_RO, Bind::Buffer(buf.sw_depth)),
            (hiz::HIZ_DST, Bind::TextureMip(buf.hiz_tex, 0)),
        ],
    )?;
    let hiz_p1_down = install_hiz_downs(cx, passes.hiz_p1, buf.hiz_tex)?;

    let hiz_final_hw_mip0 = cx.compute_pipeline(
        passes.hiz_final,
        hiz::CS_HIZ_FROM_HW,
        &[],
        &[
            (hiz::HW_DEPTH_TEX, Bind::Texture(buf.depth_tex)),
            (hiz::HIZ_DST, Bind::TextureMip(buf.hiz_tex, 0)),
        ],
    )?;
    let hiz_final_sw_mip0 = cx.compute_pipeline(
        passes.hiz_final,
        hiz::CS_HIZ_FROM_SW,
        &[],
        &[
            (hiz::HIZ_BUILD_INFO, Bind::Buffer(buf.hiz_info)),
            (hiz::SW_DEPTH_RO, Bind::Buffer(buf.sw_depth)),
            (hiz::HIZ_DST, Bind::TextureMip(buf.hiz_tex, 0)),
        ],
    )?;
    let hiz_final_clear = cx.compute_pipeline(
        passes.hiz_final,
        hiz::CS_HIZ_CLEAR,
        &[],
        &[(hiz::HIZ_DST, Bind::TextureMip(buf.hiz_tex, 0))],
    )?;
    let hiz_final_down = install_hiz_downs(cx, passes.hiz_final, buf.hiz_tex)?;
    let sh = install_shadow_pipes(cx, passes, buf)?;

    Ok(RasterPipes {
        hw_p1,
        hw_ds_p1,
        hw_p2,
        hw_ds_p2,
        sw_seed_p1,
        sw_merge_p2,
        sw_depth_p1,
        sw_vis_p1,
        sw_depth_p2,
        sw_vis_p2,
        hiz_p1_hw_mip0,
        hiz_p1_sw_mip0,
        hiz_p1_down,
        hiz_final_hw_mip0,
        hiz_final_sw_mip0,
        hiz_final_clear,
        hiz_final_down,
        shadow_collect_c0: sh.collect_c0,
        shadow_collect_c1: sh.collect_c1,
        shadow_prep: sh.prep,
        shadow_draw_c0: sh.draw_c0,
        shadow_draw_c1: sh.draw_c1,
    })
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

fn mat4_bits_eq(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> bool {
    for col in 0..4 {
        for row in 0..4 {
            if a[col][row].to_bits() != b[col][row].to_bits() {
                return false;
            }
        }
    }
    true
}

fn build_shadow_jobs(world: &NaniteWorld) -> (Vec<[u32; 2]>, u32, u32) {
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

impl NaniteGpu {
    fn alloc_static_and_scratch(
        &mut self,
        world: &NaniteWorld,
        storage_room: u64,
        cx: &mut Prepare<'_, Self>,
    ) {
        let heap_pages = safe_heap_pages(storage_room);
        self.heap_cap_pages = heap_pages;
        let _ = cx.allocate_buffer(self.buf.uniforms, size_of::<NaniteUniforms>() as u64);
        let _ = cx.allocate_buffer(self.buf.hiz_info, size_of::<HizBuildInfo>() as u64);
        let _ = cx.allocate_buffer(
            self.buf.page_heap,
            u64::from(heap_pages) * (PAGE_SIZE_BYTES as u64),
        );
        let _ = cx.allocate_buffer(
            self.buf.vis_clusters,
            u64::from(MAX_VIS_CLUSTERS) * size_of::<VisibleCluster>() as u64,
        );
        let phase_bytes =
            (u64::from(MAX_PHASE_CLUSTERS) * 3 + 1) * size_of::<VisibleCluster>() as u64;
        let _ = cx.allocate_buffer(self.buf.p1_draw_lists, phase_bytes);
        let _ = cx.allocate_buffer(self.buf.p2_draw_lists, phase_bytes);
        let _ = cx.allocate_buffer(self.buf.bvh_scratch, BVH_SCRATCH_WORDS * 4);
        let _ = cx.allocate_buffer(self.buf.cull_state, CULL_STATE_WORDS * 4);
        if cx.allocate_buffer(self.buf.indirect_p1, INDIRECT_BYTES) == Allocation::Fresh {
            cx.write(self.buf.indirect_p1, 0, &[0_u32; 64]);
        }
        if cx.allocate_buffer(self.buf.indirect_p2, INDIRECT_BYTES) == Allocation::Fresh {
            cx.write(self.buf.indirect_p2, 0, &[0_u32; 64]);
        }
        let sh_c0_alloc = cx.allocate(
            self.buf.shadow_c0,
            wgpu::Extent3d {
                width: SHADOW_C0_DIM,
                height: SHADOW_C0_DIM,
                depth_or_array_layers: 1,
            },
        );
        let sh_c1_alloc = cx.allocate(
            self.buf.shadow_c1,
            wgpu::Extent3d {
                width: SHADOW_C1_DIM,
                height: SHADOW_C1_DIM,
                depth_or_array_layers: 1,
            },
        );
        self.shadow.fresh_tex =
            sh_c0_alloc == Allocation::Fresh || sh_c1_alloc == Allocation::Fresh;
        let _ = cx.allocate_buffer(
            self.buf.shadow_list,
            u64::from(SHADOW_LIST_CAP) * 2 * size_of::<VisibleCluster>() as u64,
        );
        if cx.allocate_buffer(self.buf.shadow_indirect, 64) == Allocation::Fresh {
            cx.write(self.buf.shadow_indirect, 0, &[0_u32; 16]);
        }
        self.upload_world_buffers(world, cx);
    }

    fn upload_world_buffers(&mut self, world: &NaniteWorld, cx: &mut Prepare<'_, Self>) {
        let cl_bytes = (world.clusters.len().max(1) * size_of::<GpuClusterHeader>()) as u64;
        let bvh_bytes = (world.bvh.len().max(1) * size_of::<GpuBvhNode>()) as u64;
        let fresh_cl = cx.allocate_buffer(self.buf.clusters, cl_bytes) == Allocation::Fresh;
        let fresh_bvh = cx.allocate_buffer(self.buf.bvh, bvh_bytes) == Allocation::Fresh;
        let geom_dirty = fresh_cl || fresh_bvh || self.geom_rev != world.geometry_revision;
        if geom_dirty {
            if !world.clusters.is_empty() {
                cx.write(self.buf.clusters, 0, &world.clusters);
            }
            if !world.bvh.is_empty() {
                cx.write(self.buf.bvh, 0, &world.bvh);
            }
            self.geom_rev = world.geometry_revision;
            self.static_frames = 0;
        }

        let inst_bytes = (world.instances.len().max(1) * size_of::<NaniteInstance>()) as u64;
        let fresh_inst = cx.allocate_buffer(self.buf.instances, inst_bytes) == Allocation::Fresh;
        let inst_dirty = fresh_inst || self.inst_rev != world.instance_revision;
        if inst_dirty {
            if !world.instances.is_empty() {
                cx.write(self.buf.instances, 0, &world.instances);
            }
            self.inst_rev = world.instance_revision;
            self.static_frames = 0;
        }

        let jobs_dirty = geom_dirty || inst_dirty;
        let (jobs, c0_cnt, c1_cnt) = if jobs_dirty {
            build_shadow_jobs(world)
        } else {
            (Vec::new(), self.shadow_c0_count, self.shadow_c1_count)
        };
        let total_jobs = (c0_cnt.saturating_add(c1_cnt) as usize).max(1);
        let fresh_jobs = cx.allocate_buffer(self.buf.shadow_jobs, (total_jobs * 8) as u64)
            == Allocation::Fresh;
        if jobs_dirty || fresh_jobs {
            let to_write = if jobs_dirty {
                jobs
            } else {
                build_shadow_jobs(world).0
            };
            self.shadow_c0_count = c0_cnt;
            self.shadow_c1_count = c1_cnt;
            if !to_write.is_empty() {
                cx.write(self.buf.shadow_jobs, 0, &to_write);
            }
        }

        let page_words = world.total_pages.max(1) as u64;
        let fresh_pt = cx.allocate_buffer(self.buf.page_table, page_words * 4) == Allocation::Fresh;
        let _ = cx.allocate_buffer(self.buf.page_requests, page_words * 4);
        if fresh_pt {
            let mut init = vec![crate::types::UNMAPPED_PAGE; page_words as usize];
            let keep = self.last_page_table.len().min(init.len());
            init[..keep].copy_from_slice(&self.last_page_table[..keep]);
            cx.write(self.buf.page_table, 0, &init);
        }
    }

    fn sync_bridge(&mut self, bridge: &NaniteBridgeHandle, cx: &mut Prepare<'_, Self>) {
        let stats_bytes = cx.take_readback(self.buf.stats_rb);
        let req_bytes = cx.take_readback(self.buf.req_rb);
        let mut uploads = Vec::new();
        let mut table = None;
        if let Ok(mut guard) = bridge.0.lock() {
            if let Some(bytes) = stats_bytes
                && bytes.len() >= 64
            {
                let mut s = [0_u32; 16];
                for (dst, chunk) in s.iter_mut().zip(bytes.as_chunks::<4>().0) {
                    *dst = u32::from_le_bytes(*chunk);
                }
                guard.latest_stats = Some(s);
            }
            if let Some(bytes) = req_bytes {
                let words: Vec<u32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes(*c))
                    .collect();
                guard.latest_requests = Some(words);
            }
            uploads = std::mem::take(&mut guard.pending_uploads);
            table = guard.dirty_page_table.take();
            self.residency_ver = guard.residency_version;
        }
        for up in uploads {
            if up.slot < self.heap_cap_pages {
                let byte_off = u64::from(up.slot) * (PAGE_SIZE_BYTES as u64);
                cx.write(self.buf.page_heap, byte_off, &up.words);
            }
        }
        if let Some(pt) = table
            && !pt.is_empty()
        {
            self.last_page_table = pt.clone();
            cx.write(self.buf.page_table, 0, &pt);
        }
    }

    fn encode_bvh_traversal(
        &self,
        cx: &mut Encode<'_, Self>,
        indirect: Buffer<Self>,
        is_p2: bool,
    ) {
        let (prep_a, prep_b, trav_ab, trav_ba, dfs_cut) = if is_p2 {
            (
                &self.cull.prep_bvh_a_p2,
                &self.cull.prep_bvh_b_p2,
                &self.cull.trav_ab_p2,
                &self.cull.trav_ba_p2,
                &self.cull.dfs_cut_p2,
            )
        } else {
            (
                &self.cull.prep_bvh_a_p1,
                &self.cull.prep_bvh_b_p1,
                &self.cull.trav_ab_p1,
                &self.cull.trav_ba_p1,
                &self.cull.dfs_cut_p1,
            )
        };
        for _ in 0..4 {
            cx.dispatch(prep_a, [1, 1, 1]);
            cx.dispatch_indirect(trav_ab, indirect, 32);
            cx.dispatch(prep_b, [1, 1, 1]);
            cx.dispatch_indirect(trav_ba, indirect, 48);
        }
        cx.dispatch(prep_a, [1, 1, 1]);
        cx.dispatch_indirect(dfs_cut, indirect, 32);
    }

    fn encode_hiz_down(&self, cx: &mut Encode<'_, Self>, downs: &[ComputePipeline<Self>]) {
        let (mip0_w, mip0_h, mip_count) = self.hiz_dims;
        let count = (mip_count.saturating_sub(1) as usize).min(downs.len());
        for (idx, pipe) in downs.iter().take(count).enumerate() {
            let (dw, dh) = mip_extent(mip0_w, mip0_h, (idx + 1) as u32);
            cx.dispatch(pipe, [dw.div_ceil(HIZ_WG), dh.div_ceil(HIZ_WG), 1]);
        }
    }

    fn encode_shadow_cull(&self, cx: &mut Encode<'_, Self>) {
        cx.clear_buffer(self.buf.shadow_indirect, 0..64);
        if self.instance_count == 0 || self.cluster_count == 0 {
            return;
        }
        let dedup_range = (DEDUP_BASE_WORDS * 4)..(CULL_STATE_WORDS * 4);
        if self.shadow.dirty_c0 && self.shadow_c0_count > 0 {
            cx.clear_buffer(self.buf.cull_state, dedup_range.clone());
            cx.dispatch(
                &self.raster.shadow_collect_c0,
                [self.shadow_c0_count.div_ceil(64), 1, 1],
            );
        }
        if self.shadow.dirty_c1 && self.shadow_c1_count > 0 {
            cx.clear_buffer(self.buf.cull_state, dedup_range);
            cx.dispatch(
                &self.raster.shadow_collect_c1,
                [self.shadow_c1_count.div_ceil(64), 1, 1],
            );
        }
        cx.dispatch(&self.raster.shadow_prep, [1, 1, 1]);
    }

    fn encode_cull_p1(&self, cx: &mut Encode<'_, Self>) {
        cx.clear_buffer(self.buf.cull_state, 0..(32 * 4));
        cx.clear_buffer(
            self.buf.cull_state,
            (DEDUP_BASE_WORDS * 4)..(CULL_STATE_WORDS * 4),
        );
        let req_bytes = u64::from(self.page_count.max(1)) * 4;
        cx.clear_buffer(self.buf.page_requests, 0..req_bytes);
        cx.dispatch(&self.cull.reset, [1, 1, 1]);
        if self.instance_count == 0 || self.cluster_count == 0 {
            return;
        }
        cx.dispatch(&self.cull.inst_p1, [self.instance_count.div_ceil(64), 1, 1]);
        self.encode_bvh_traversal(cx, self.buf.indirect_p1, false);
        cx.dispatch(&self.cull.prep_cand_p1, [1, 1, 1]);
        cx.dispatch_indirect(&self.cull.clus_p1, self.buf.indirect_p1, 64);
        cx.dispatch(&self.cull.prep_draws_p1, [1, 1, 1]);
    }

    fn encode_cull_p2(&self, cx: &mut Encode<'_, Self>) {
        if self.instance_count == 0 || self.cluster_count == 0 || self.freeze || self.skip_p2 {
            return;
        }
        cx.clear_buffer(self.buf.cull_state, 64..76);
        cx.dispatch_indirect(&self.cull.inst_p2, self.buf.indirect_p1, 80);
        self.encode_bvh_traversal(cx, self.buf.indirect_p2, true);
        cx.dispatch(&self.cull.prep_cand_p2, [1, 1, 1]);
        cx.dispatch_indirect(&self.cull.clus_p2, self.buf.indirect_p2, 64);
        cx.dispatch_indirect(&self.cull.occ_clus_p2, self.buf.indirect_p1, 96);
        cx.dispatch(&self.cull.prep_draws_p2, [1, 1, 1]);
    }

    fn update_shadow_track(
        &mut self,
        world: &NaniteWorld,
        cam: &NaniteCamera,
        cfg: &NaniteConfig,
    ) -> (Mat4, Mat4, [f32; 4], [f32; 4]) {
        let c0_half = engine::math::max(cfg.shadow_c0_half, 2.0);
        let c1_half = engine::math::max(cfg.shadow_c1_half, 4.0);
        let vm = cam.render_view.to_cols_array_2d();
        let fwd_xz = Vec3::new(-vm[0][2], 0.0, -vm[2][2]).normalize_or_zero();
        let desired_c0 = Vec3::new(
            cam.render_pos.x + fwd_xz.x * cfg.shadow_c0_forward,
            engine::math::clamp(cam.render_pos.y, -2.0, 26.0),
            cam.render_pos.z + fwd_xz.z * cfg.shadow_c0_forward,
        );
        let c1_center = cfg.shadow_c1_center;

        let geom_changed = self.shadow.last_geom_rev != world.geometry_revision
            || self.shadow.last_inst_rev != world.instance_revision;
        if geom_changed || self.shadow.fresh_tex {
            self.shadow.last_geom_rev = world.geometry_revision;
            self.shadow.last_inst_rev = world.instance_revision;
            self.shadow.last_c0_res = u64::MAX;
            self.shadow.last_c1_res = u64::MAX;
        }
        let move_step = engine::math::max(c0_half * 0.25, 1.5);
        let c0_moved = (desired_c0 - self.shadow.last_c0_center).length_squared()
            > move_step * move_step;
        let tier = shadow_res_tier(self.residency_ver, world.total_pages);
        let c0_res_changed = self.shadow.last_c0_res == u64::MAX
            || (tier > 0 && tier != self.shadow.last_c0_res);
        let c1_res_changed = self.shadow.last_c1_res == u64::MAX
            || (tier > 0 && tier != self.shadow.last_c1_res);
        self.shadow.enabled = cfg.shadows_enabled && self.instance_count > 0;
        let want_c0 = (self.shadow.enabled && (c0_moved || c0_res_changed)) || self.shadow.fresh_tex;
        let want_c1 = (self.shadow.enabled && c1_res_changed) || self.shadow.fresh_tex;
        self.shadow.dirty_c0 = want_c0;
        self.shadow.dirty_c1 = want_c1;
        if self.shadow.dirty_c0 && self.shadow.enabled {
            self.shadow.last_c0_center = desired_c0;
            self.shadow.last_c0_res = tier;
        }
        if self.shadow.dirty_c1 && self.shadow.enabled {
            self.shadow.last_c1_res = tier;
        }
        let c0_center = if self.shadow.last_c0_center.length_squared() < 1.0e14 {
            self.shadow.last_c0_center
        } else {
            desired_c0
        };
        let sh_vp0 = ortho_shadow_vp(
            c0_center,
            c0_half,
            engine::math::max(cfg.shadow_c0_depth, 20.0),
            cam.sun_dir,
            SHADOW_C0_DIM,
        );
        let sh_vp1 = ortho_shadow_vp(
            c1_center,
            c1_half,
            engine::math::max(cfg.shadow_c1_depth, 40.0),
            cam.sun_dir,
            SHADOW_C1_DIM,
        );
        (
            sh_vp0,
            sh_vp1,
            [c0_center.x, c0_center.y, c0_center.z, c0_half],
            [c1_center.x, c1_center.y, c1_center.z, c1_half],
        )
    }

    fn prepare_screen_targets(
        &mut self,
        cfg: &NaniteConfig,
        cx: &mut Prepare<'_, Self>,
    ) -> (u32, u32, u32, u32, u32) {
        let render_scale = engine::math::clamp(cfg.render_scale, 0.25, 1.0);
        let scale_side = |p: u32| ((f64::from(p) * f64::from(render_scale)).round() as u32).max(1);
        let (raw_w, raw_h) = cx.canvas_size();
        let (cw, ch) = (scale_side(raw_w), scale_side(raw_h));
        cx.scale_target(NANITE_VIS_HW, render_scale);
        let depth_alloc = cx.allocate(
            self.buf.depth_tex,
            wgpu::Extent3d {
                width: cw,
                height: ch,
                depth_or_array_layers: 1,
            },
        );
        if depth_alloc == Allocation::Fresh || self.canvas != (cw, ch) {
            self.hiz_ready = false;
            self.static_frames = 0;
        }
        self.canvas = (cw, ch);
        let sw_bytes = if cfg.hw_only {
            16
        } else {
            (u64::from(cw) * u64::from(ch) * 4).max(16)
        };
        let _ = cx.allocate_buffer(self.buf.sw_depth, sw_bytes);
        let _ = cx.allocate_buffer(self.buf.sw_vis, sw_bytes);
        let (mw, mh, mips) = hiz_dimensions(cw, ch);
        let hiz_alloc = cx.allocate_mips(
            self.buf.hiz_tex,
            wgpu::Extent3d {
                width: mw,
                height: mh,
                depth_or_array_layers: 1,
            },
            mips,
        );
        if hiz_alloc == Allocation::Fresh {
            self.hiz_ready = false;
            self.static_frames = 0;
        }
        self.hiz_dims = (mw, mh, mips);
        cx.write(
            self.buf.hiz_info,
            0,
            &[HizBuildInfo {
                src_size: [cw, ch],
                mip0_size: [mw, mh],
            }],
        );
        (cw, ch, mw, mh, mips)
    }

    fn encode_hiz_final(&mut self, cx: &mut Encode<'_, Self>) {
        let (mw, mh, _) = self.hiz_dims;
        if !self.freeze && !self.skip_p2 {
            let _ = cx.recorded();
            if self.instance_count > 0 && self.cluster_count > 0 {
                let mip0 = if self.hw_only {
                    &self.raster.hiz_final_hw_mip0
                } else {
                    &self.raster.hiz_final_sw_mip0
                };
                cx.dispatch(mip0, [mw.div_ceil(HIZ_WG), mh.div_ceil(HIZ_WG), 1]);
            } else {
                cx.dispatch(
                    &self.raster.hiz_final_clear,
                    [mw.div_ceil(HIZ_WG), mh.div_ceil(HIZ_WG), 1],
                );
            }
            self.encode_hiz_down(cx, &self.raster.hiz_final_down);
            if cx.recorded() == engine_render::Recorded::Exact {
                self.hiz_ready = true;
                self.hiz.prev_view = self.hiz.curr_view;
                self.hiz.prev_proj = self.hiz.curr_proj;
            }
        }
        cx.copy_to_readback(self.buf.stats_rb, self.buf.cull_state, 0..64);
        if self.page_count > 0 {
            cx.copy_to_readback(
                self.buf.req_rb,
                self.buf.page_requests,
                0..(u64::from(self.page_count) * 4),
            );
        }
    }
}

impl GpuPart for NaniteGpu {
    type Reads = (
        Res<'static, NaniteWorld>,
        Res<'static, NaniteCamera>,
        Res<'static, NaniteConfig>,
        Res<'static, NaniteBridgeHandle>,
        Res<'static, DeviceLimits>,
    );
    const NAME: &'static str = "brick_nanite";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let buf = alloc_buffers(cx);
        declare_targets_and_frame(cx, &buf)?;
        let passes = install_passes(cx, &buf);
        let cull = install_cull_pipes(cx, &passes, &buf)?;
        let raster = install_raster_pipes(cx, &passes, &buf)?;
        Ok(Self {
            buf,
            passes,
            cull,
            raster,
            shadow: ShadowTrack {
                last_c0_center: Vec3::splat(1.0e9),
                last_c0_res: u64::MAX,
                last_c1_res: u64::MAX,
                last_geom_rev: u64::MAX,
                last_inst_rev: u64::MAX,
                dirty_c0: true,
                dirty_c1: true,
                fresh_tex: true,
                enabled: true,
            },
            hiz: HizTrack {
                prev_view: Mat4::IDENTITY,
                prev_proj: Mat4::IDENTITY,
                curr_view: Mat4::IDENTITY,
                curr_proj: Mat4::IDENTITY,
            },
            geom_rev: u64::MAX,
            inst_rev: u64::MAX,
            page_count: 0,
            instance_count: 0,
            cluster_count: 0,
            shadow_c0_count: 0,
            shadow_c1_count: 0,
            last_page_table: Vec::new(),
            canvas: (0, 0),
            hiz_dims: (1, 1, 1),
            frame_gen: 0,
            hiz_ready: false,
            freeze: false,
            hw_only: true,
            skip_p2: false,
            static_frames: 0,
            last_cull_vp: [[0.0; 4]; 4],
            last_lod_bits: 0,
            last_hw_only: true,
            last_freeze: false,
            last_residency_ver: u64::MAX,
            residency_ver: 0,
            heap_cap_pages: MAX_HEAP_PAGES,
        })
    }

    fn prepare(
        &mut self,
        (world, cam, cfg, bridge, limits): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.alloc_static_and_scratch(&world, limits.storage_room, cx);
        self.sync_bridge(&bridge, cx);
        let (cw, ch, mw, mh, mips) = self.prepare_screen_targets(&cfg, cx);

        self.page_count = world.total_pages;
        self.instance_count = world.instances.len() as u32;
        self.cluster_count = world.clusters.len() as u32;
        self.freeze = cfg.freeze_culling;
        self.hw_only = cfg.hw_only;
        self.frame_gen = self.frame_gen.wrapping_add(1).max(1);

        let render_vp = cam.render_proj * cam.render_view;
        let cull_vp = cam.cull_proj * cam.cull_view;
        let cull_vp_cols = cull_vp.to_cols_array_2d();
        let lod_bits = cfg.lod_preset.threshold_px().to_bits();
        let camera_same = mat4_bits_eq(&cull_vp_cols, &self.last_cull_vp);
        let state_same = lod_bits == self.last_lod_bits
            && cfg.hw_only == self.last_hw_only
            && cfg.freeze_culling == self.last_freeze;
        let residency_same = self.residency_ver == self.last_residency_ver;
        if camera_same && state_same && residency_same && self.hiz_ready && self.instance_count > 0
        {
            self.static_frames = self.static_frames.saturating_add(1);
        } else {
            self.static_frames = 0;
            self.last_cull_vp = cull_vp_cols;
            self.last_lod_bits = lod_bits;
            self.last_hw_only = cfg.hw_only;
            self.last_freeze = cfg.freeze_culling;
            self.last_residency_ver = self.residency_ver;
        }
        self.skip_p2 = self.hw_only && self.hiz_ready && self.static_frames >= 2;

        let (sh_vp0, sh_vp1, shadow_center0, shadow_center1) =
            self.update_shadow_track(&world, &cam, &cfg);

        if !self.hiz_ready {
            self.hiz.prev_view = cam.cull_view;
            self.hiz.prev_proj = cam.cull_proj;
        }
        self.hiz.curr_view = cam.cull_view;
        self.hiz.curr_proj = cam.cull_proj;

        let planes = extract_frustum_planes(cull_vp);
        let eff_h = ch.min(1080) as f32;
        let lod_scale = (eff_h * 0.5) * cam.cull_proj.col(1).y.abs();
        let render_lod_scale = (ch as f32 * 0.5) * cam.render_proj.col(1).y.abs();
        let u = NaniteUniforms {
            render_view_proj: render_vp.to_cols_array_2d(),
            cull_view_proj: cull_vp_cols,
            cull_view: cam.cull_view.to_cols_array_2d(),
            cull_proj: cam.cull_proj.to_cols_array_2d(),
            inv_render_vp: render_vp.inverse().to_cols_array_2d(),
            cull_pos_near: [cam.cull_pos.x, cam.cull_pos.y, cam.cull_pos.z, cam.near_z],
            render_pos_time: [cam.render_pos.x, cam.render_pos.y, cam.render_pos.z, cam.sim_time],
            viewport_lod: [cw as f32, ch as f32, lod_scale, cfg.lod_preset.threshold_px()],
            hiz_params: [mw as f32, mh as f32, mips as f32, cfg.sw_max_px],
            counts: [self.instance_count, self.cluster_count, self.page_count, self.frame_gen],
            flags: [
                u32::from(cfg.freeze_culling),
                u32::from(cfg.hw_only),
                cfg.view_mode.index(),
                u32::from(self.hiz_ready),
            ],
            plane0: planes[0],
            plane1: planes[1],
            plane2: planes[2],
            plane3: planes[3],
            plane4: planes[4],
            plane5: planes[5],
            shadow_vp0: sh_vp0.to_cols_array_2d(),
            shadow_vp1: sh_vp1.to_cols_array_2d(),
            shadow_info: [SHADOW_C0_DIM as f32, SHADOW_C1_DIM as f32, render_lod_scale, 0.0],
            shadow_center0,
            shadow_center1,
            extra: [
                cfg.double_sided_materials,
                self.shadow_c0_count,
                self.shadow_c1_count,
                0,
            ],
            prev_cull_view: self.hiz.prev_view.to_cols_array_2d(),
            prev_cull_proj: self.hiz.prev_proj.to_cols_array_2d(),
        };
        cx.write(self.buf.uniforms, 0, &[u]);

        if self.shadow.dirty_c0 || self.shadow.dirty_c1 {
            cx.run(self.passes.shadow_cull);
        }
        if self.shadow.dirty_c0 {
            cx.run(self.passes.shadow_c0);
        }
        if self.shadow.dirty_c1 {
            cx.run(self.passes.shadow_c1);
        }
        cx.run(self.passes.cull_p1);
    }

    fn view(&mut self, _view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        cx.run(self.passes.hw_p1);
        if !self.hw_only {
            cx.run(self.passes.sw_p1);
        }
        if !self.freeze && !self.skip_p2 {
            cx.run(self.passes.hiz_p1);
            cx.run(self.passes.cull_p2);
            cx.run(self.passes.hw_p2);
            if !self.hw_only {
                cx.run(self.passes.sw_p2);
            }
        }
        cx.run(self.passes.hiz_final);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) {
        if pass == self.passes.shadow_cull {
            self.encode_shadow_cull(cx);
        } else if pass == self.passes.shadow_c0 {
            cx.render().draw_indirect(
                &self.raster.shadow_draw_c0,
                self.buf.shadow_indirect,
                0,
                None,
            );
        } else if pass == self.passes.shadow_c1 {
            cx.render().draw_indirect(
                &self.raster.shadow_draw_c1,
                self.buf.shadow_indirect,
                16,
                None,
            );
        } else if pass == self.passes.cull_p1 {
            self.encode_cull_p1(cx);
        } else if pass == self.passes.hw_p1 {
            let mut render = cx.render();
            render.draw_indirect(&self.raster.hw_p1, self.buf.indirect_p1, 0, None);
            render.draw_indirect(&self.raster.hw_ds_p1, self.buf.indirect_p1, 112, None);
        } else if pass == self.passes.sw_p1 {
            let (cw, ch) = self.canvas;
            if cw > 0 && ch > 0 {
                cx.dispatch(&self.raster.sw_seed_p1, [cw.div_ceil(8), ch.div_ceil(8), 1]);
            }
            if self.instance_count > 0 && self.cluster_count > 0 {
                cx.dispatch_indirect(&self.raster.sw_depth_p1, self.buf.indirect_p1, 16);
                cx.dispatch_indirect(&self.raster.sw_vis_p1, self.buf.indirect_p1, 16);
            }
        } else if pass == self.passes.hiz_p1 {
            if !self.freeze && !self.skip_p2 {
                let (mw, mh, _) = self.hiz_dims;
                let mip0 = if self.hw_only {
                    &self.raster.hiz_p1_hw_mip0
                } else {
                    &self.raster.hiz_p1_sw_mip0
                };
                cx.dispatch(mip0, [mw.div_ceil(HIZ_WG), mh.div_ceil(HIZ_WG), 1]);
                self.encode_hiz_down(cx, &self.raster.hiz_p1_down);
            }
        } else if pass == self.passes.cull_p2 {
            self.encode_cull_p2(cx);
        } else if pass == self.passes.hw_p2 {
            if self.instance_count > 0 && self.cluster_count > 0 && !self.freeze && !self.skip_p2 {
                let mut render = cx.render();
                render.draw_indirect(&self.raster.hw_p2, self.buf.indirect_p2, 0, None);
                render.draw_indirect(&self.raster.hw_ds_p2, self.buf.indirect_p2, 112, None);
            }
        } else if pass == self.passes.sw_p2 {
            if self.instance_count > 0 && self.cluster_count > 0 && !self.freeze && !self.skip_p2 {
                let (cw, ch) = self.canvas;
                if cw > 0 && ch > 0 {
                    cx.dispatch(&self.raster.sw_merge_p2, [cw.div_ceil(8), ch.div_ceil(8), 1]);
                }
                cx.dispatch_indirect(&self.raster.sw_depth_p2, self.buf.indirect_p2, 16);
                cx.dispatch_indirect(&self.raster.sw_vis_p2, self.buf.indirect_p2, 16);
            }
        } else if pass == self.passes.hiz_final {
            self.encode_hiz_final(cx);
        }
    }
}
