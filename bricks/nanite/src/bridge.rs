use std::sync::{Arc, Mutex};

use engine::prelude::*;
use engine_core::DeviceLimits;

use crate::stream::VisibleCutParams;
use crate::types::{
    NaniteCamera, NaniteConfig, NaniteTelemetry, extract_frustum_planes, safe_heap_pages,
};
use crate::world::{NaniteStreaming, NaniteWorld, PageUpload};

#[derive(Default)]
pub struct NaniteBridge {
    pub latest_stats: Option<[u32; 16]>,
    pub latest_requests: Option<Vec<u32>>,
    pub pending_uploads: Vec<PageUpload>,
    pub dirty_page_table: Option<Vec<u32>>,
    pub residency_version: u64,
}

#[derive(Resource, Clone, Default)]
pub struct NaniteBridgeHandle(pub Arc<Mutex<NaniteBridge>>);

#[derive(SystemParam)]
pub struct PumpNanite<'w> {
    world: Res<'w, NaniteWorld>,
    cam: Res<'w, NaniteCamera>,
    views: Res<'w, Views>,
    config: Res<'w, NaniteConfig>,
    limits: Res<'w, DeviceLimits>,
    bridge: Res<'w, NaniteBridgeHandle>,
    streaming: ResMut<'w, NaniteStreaming>,
    telemetry: ResMut<'w, NaniteTelemetry>,
}

pub fn pump_nanite_streaming(mut p: PumpNanite<'_>) {
    p.streaming.sync_world(&p.world);
    let heap_cap = safe_heap_pages(p.limits.storage_room);
    p.streaming
        .set_budget(p.config.heap_budget.pages().min(heap_cap));

    let (stats, reqs) = match p.bridge.0.lock() {
        Ok(mut guard) => (guard.latest_stats.take(), guard.latest_requests.take()),
        Err(_) => (None, None),
    };

    if let Some(s) = stats {
        p.telemetry.inst_p1_vis = s[0];
        p.telemetry.inst_p1_occ = s[1];
        p.telemetry.inst_p2_vis = s[2];
        p.telemetry.clusters_tested = s[3];
        p.telemetry.hw_clusters = s[4].saturating_add(s[6]);
        p.telemetry.sw_clusters = s[5].saturating_add(s[7]);
        p.telemetry.tris_hw = s[8];
        p.telemetry.tris_sw = s[9];
        p.telemetry.page_faults = s[10];
        p.telemetry.fallback_draws = s[11];
        p.telemetry.cone_culled = s[12];
        p.telemetry.frustum_culled = s[13];
        p.telemetry.hiz_culled = s[14];
    }

    let render_scale = engine::math::clamp(p.config.render_scale, 0.25, 1.0);
    let vp_h = engine::math::clamp(
        (p.views.frame().display_size.y as f32 * render_scale).round(),
        360.0,
        1080.0,
    );
    let lod_scale = (vp_h * 0.5) * p.cam.cull_proj.y_axis.y.abs();
    let max_new = (p.config.max_pages_per_frame * 8).max(1024);

    if let Some(priorities) = reqs {
        p.streaming.ingest_gpu_requests(&p.world, &priorities, max_new);
    }

    if !p.world.instances.is_empty() {
        let cull_vp = p.cam.cull_proj * p.cam.cull_view;
        p.streaming.seed_visible_cut(
            &p.world,
            VisibleCutParams {
                cam_pos: p.cam.cull_pos,
                near_z: p.cam.near_z,
                lod_scale,
                lod_thresh: p.config.lod_preset.threshold_px() * 2.0,
                frustum_planes: extract_frustum_planes(cull_vp),
                double_sided_materials: p.config.double_sided_materials,
            },
            192,
        );
    }

    if p.telemetry.page_faults == 0
        && !p.streaming.has_pending_work()
        && !p.world.instances.is_empty()
    {
        p.streaming.enqueue_prefetch_360(
            &p.world,
            p.cam.cull_pos,
            p.cam.near_z,
            lod_scale,
            p.config.lod_preset.threshold_px(),
            192,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    p.streaming.pump_native_io(p.config.max_pages_per_frame);

    p.streaming.commit_ready_pages(p.config.max_pages_per_frame);

    let uploads = p.streaming.take_uploads();
    let dirty_table = if p.streaming.take_table_dirty() {
        Some(p.streaming.page_table().to_vec())
    } else {
        None
    };
    let residency_version = p.streaming.residency_version();

    if (!uploads.is_empty() || dirty_table.is_some())
        && let Ok(mut guard) = p.bridge.0.lock()
    {
        guard.pending_uploads.extend(uploads);
        if let Some(table) = dirty_table {
            guard.dirty_page_table = Some(table);
        }
        guard.residency_version = residency_version;
    }

    p.telemetry.total_virtual_tris = p.world.total_virtual_tris;
    p.telemetry.total_pages = p.world.total_pages;
    p.telemetry.resident_pages = p.streaming.resident_count();
    p.telemetry.pinned_pages = p.streaming.pinned_count();
    p.telemetry.budget_pages = p.streaming.budget_pages();
    p.telemetry.pending_fetches = p.streaming.pending_fetch_count();
    p.telemetry.pending_requests = p.streaming.pending_request_count();
    p.telemetry.ready_pages = p.streaming.ready_page_count();
}
