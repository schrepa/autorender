use std::sync::{Arc, Mutex};

use engine::prelude::*;

use crate::types::{NaniteConfig, NaniteTelemetry};
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

pub fn pump_nanite_streaming(
    world: Res<NaniteWorld>,
    config: Res<NaniteConfig>,
    bridge: Res<NaniteBridgeHandle>,
    mut streaming: ResMut<NaniteStreaming>,
    mut telemetry: ResMut<NaniteTelemetry>,
) {
    streaming.sync_world(&world);
    streaming.set_budget(config.heap_budget.pages());

    let (stats, reqs) = match bridge.0.lock() {
        Ok(mut guard) => (guard.latest_stats.take(), guard.latest_requests.take()),
        Err(_) => (None, None),
    };

    if let Some(s) = stats {
        telemetry.inst_p1_vis = s[0];
        telemetry.inst_p1_occ = s[1];
        telemetry.inst_p2_vis = s[2];
        telemetry.clusters_tested = s[3];
        telemetry.hw_clusters = s[4].saturating_add(s[6]);
        telemetry.sw_clusters = s[5].saturating_add(s[7]);
        telemetry.tris_hw = s[8];
        telemetry.tris_sw = s[9];
        telemetry.page_faults = s[10];
        telemetry.fallback_draws = s[11];
        telemetry.cone_culled = s[12];
        telemetry.frustum_culled = s[13];
        telemetry.hiz_culled = s[14];
    }

    if let Some(priorities) = reqs {
        let max_new = (config.max_pages_per_frame * 4).max(192);
        streaming.ingest_gpu_requests(&world, &priorities, max_new);
    }

    #[cfg(not(target_arch = "wasm32"))]
    streaming.pump_native_io(config.max_pages_per_frame);

    streaming.commit_ready_pages(config.max_pages_per_frame);

    let uploads = streaming.take_uploads();
    let dirty_table = if streaming.take_table_dirty() {
        Some(streaming.page_table().to_vec())
    } else {
        None
    };
    let residency_version = streaming.residency_version();

    if (!uploads.is_empty() || dirty_table.is_some())
        && let Ok(mut guard) = bridge.0.lock()
    {
        guard.pending_uploads.extend(uploads);
        if let Some(table) = dirty_table {
            guard.dirty_page_table = Some(table);
        }
        guard.residency_version = residency_version;
    }

    telemetry.total_virtual_tris = world.total_virtual_tris;
    telemetry.total_pages = world.total_pages;
    telemetry.resident_pages = streaming.resident_count();
    telemetry.pinned_pages = streaming.pinned_count();
    telemetry.budget_pages = streaming.budget_pages();
    telemetry.pending_fetches = streaming.pending_fetch_count();
    telemetry.pending_requests = streaming.pending_request_count();
    telemetry.ready_pages = streaming.ready_page_count();
}
