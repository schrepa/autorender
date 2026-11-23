engine::brick!(sanctuary);

#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read as _, Seek as _, SeekFrom};
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

use brick_nanite::NaniteWorld;
use engine::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use engine_meshlet::MeshletManifest;

use crate::gi::GiUploadBridge;
#[cfg(not(target_arch = "wasm32"))]
use crate::gi::submit_gi_bytes;
#[cfg(not(target_arch = "wasm32"))]
use crate::layout::{HERO_ASSETS, populate_sanctuary};
use crate::pbr::{PbrTexturePack, PbrUploadBridge};
use crate::state::MonolithState;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource, Default)]
struct NativeHdSource {
    path: Option<PathBuf>,
}

pub struct SanctuaryBrick;

impl Brick for SanctuaryBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .resource::<PbrTexturePack>()
            .resource::<PbrUploadBridge>()
            .resource::<GiUploadBridge>()
            .systems(Schedule::BeforeTicks, load_sanctuary_assets);
        #[cfg(not(target_arch = "wasm32"))]
        bricks.resource::<NativeHdSource>();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn find_asset_dir() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("web/public/monolith"),
        PathBuf::from("../web/public/monolith"),
        PathBuf::from("../../web/public/monolith"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../web/public/monolith"),
    ];
    candidates
        .into_iter()
        .find(|dir| dir.join("sponza_lower_w.mlm").is_file())
}

#[cfg(not(target_arch = "wasm32"))]
fn pump_native_hd(pbr: &mut PbrTexturePack, hd_path: &PathBuf) {
    let reqs = pbr.take_hd_requests(4, true);
    if reqs.is_empty() {
        return;
    }
    let Ok(mut file) = std::fs::File::open(hd_path) else {
        return;
    };
    let mut buf = Vec::new();
    for req in reqs {
        if file.seek(SeekFrom::Start(req.byte_offset)).is_err() {
            continue;
        }
        buf.resize(req.byte_len as usize, 0);
        if file.read_exact(&mut buf).is_ok() {
            let _ = pbr.deliver_hd_blob(req, &buf);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn load_sanctuary_assets(
    mut world: ResMut<NaniteWorld>,
    mut pbr: ResMut<PbrTexturePack>,
    pbr_bridge: Res<PbrUploadBridge>,
    gi_bridge: Res<GiUploadBridge>,
    mut hd_src: ResMut<NativeHdSource>,
    mut state: ResMut<MonolithState>,
) {
    if !state.assets_loaded {
        let Some(dir) = find_asset_dir() else {
            return;
        };
        world.clear();
        for stem in HERO_ASSETS {
            let mlm_path = dir.join(format!("{stem}.mlm"));
            let mlp_path = dir.join(format!("{stem}.mlp"));
            let Ok(bytes) = std::fs::read(&mlm_path) else {
                return;
            };
            let Ok(manifest) = MeshletManifest::from_bytes(&bytes) else {
                return;
            };
            world.add_mesh(stem, &manifest, Some(mlp_path));
        }
        populate_sanctuary(&mut world);
        if let Ok(gi_bytes) = std::fs::read(dir.join("sponza_gi.bin")) {
            let _ = submit_gi_bytes(&gi_bridge, &gi_bytes);
        }
        if !pbr.pack_loaded
            && let Ok(base_bytes) = std::fs::read(dir.join("sponza_pbr_base.bin"))
        {
            let _ = pbr.load_base_bytes(base_bytes);
        }
        let hd_file = dir.join("sponza_pbr_hd.bin");
        if hd_file.is_file() {
            hd_src.path = Some(hd_file);
        }
        state.assets_loaded = true;
    }
    pbr.pump_uploads(&pbr_bridge, 14);
    if pbr.base_done
        && let Some(hd_path) = hd_src.path.as_ref()
    {
        pump_native_hd(&mut pbr, hd_path);
        pbr.pump_uploads(&pbr_bridge, 0);
    }
}

#[cfg(target_arch = "wasm32")]
const MAX_RUN_PAGES: usize = 40;
#[cfg(target_arch = "wasm32")]
const MAX_RUN_BYTES: u64 = 384 * 1024;
#[cfg(target_arch = "wasm32")]
const MAX_GAP_BYTES: u64 = 96 * 1024;
#[cfg(target_arch = "wasm32")]
const HD_IDLE_HYSTERESIS_FRAMES: u32 = 3;

#[cfg(target_arch = "wasm32")]
struct PageRunItem {
    req: Option<brick_nanite::PageFetchRequest>,
    global_page_id: u32,
    rel_offset: usize,
    byte_len: usize,
    is_requested: bool,
}

#[cfg(target_arch = "wasm32")]
struct PageRun {
    mesh_id: u32,
    mesh_name: String,
    first_local_page: u32,
    last_local_page: u32,
    byte_offset: u64,
    total_bytes: u64,
    max_priority: u32,
    items: Vec<PageRunItem>,
}

#[cfg(target_arch = "wasm32")]
fn coalesce_runs(
    mut reqs: Vec<brick_nanite::PageFetchRequest>,
    world: &NaniteWorld,
) -> Vec<PageRun> {
    reqs.sort_unstable_by(|a, b| {
        a.mesh_id
            .cmp(&b.mesh_id)
            .then_with(|| a.local_page_id.cmp(&b.local_page_id))
    });
    let mut runs: Vec<PageRun> = Vec::new();
    for req in reqs {
        if let Some(last) = runs.last_mut() {
            let expected_end = last.byte_offset.saturating_add(last.total_bytes);
            if last.mesh_id == req.mesh_id
                && req.local_page_id > last.last_local_page
                && req.byte_offset >= expected_end
            {
                let gap = req.byte_offset - expected_end;
                let next_total = req
                    .byte_offset
                    .saturating_add(req.byte_len)
                    .saturating_sub(last.byte_offset);
                let page_span = req
                    .local_page_id
                    .saturating_sub(last.first_local_page)
                    .saturating_add(1) as usize;
                if gap <= MAX_GAP_BYTES
                    && next_total <= MAX_RUN_BYTES
                    && page_span <= MAX_RUN_PAGES
                {
                    if gap > 0
                        && let Some(mesh) = world.meshes.get(req.mesh_id as usize)
                    {
                        for local_pid in (last.last_local_page + 1)..req.local_page_id {
                            if let Some(span) = mesh.page_spans.get(local_pid as usize)
                                && span.byte_len > 0
                                && span.byte_offset >= last.byte_offset
                                && span.byte_offset.saturating_add(u64::from(span.byte_len))
                                    <= req.byte_offset
                            {
                                last.items.push(PageRunItem {
                                    req: None,
                                    global_page_id: mesh.page_offset + local_pid,
                                    rel_offset: (span.byte_offset - last.byte_offset) as usize,
                                    byte_len: span.byte_len as usize,
                                    is_requested: false,
                                });
                            }
                        }
                    }
                    let rel_offset = (req.byte_offset - last.byte_offset) as usize;
                    let global_page_id = req.global_page_id;
                    let byte_len = req.byte_len as usize;
                    last.max_priority = last.max_priority.max(req.priority);
                    last.total_bytes = next_total;
                    last.last_local_page = req.local_page_id;
                    last.items.push(PageRunItem {
                        req: Some(req),
                        global_page_id,
                        rel_offset,
                        byte_len,
                        is_requested: true,
                    });
                    continue;
                }
            }
        }
        let mesh_id = req.mesh_id;
        let mesh_name = req.mesh_name.clone();
        let local_page_id = req.local_page_id;
        let global_page_id = req.global_page_id;
        let byte_offset = req.byte_offset;
        let total_bytes = req.byte_len;
        let max_priority = req.priority;
        runs.push(PageRun {
            mesh_id,
            mesh_name,
            first_local_page: local_page_id,
            last_local_page: local_page_id,
            byte_offset,
            total_bytes,
            max_priority,
            items: vec![PageRunItem {
                req: Some(req),
                global_page_id,
                rel_offset: 0,
                byte_len: total_bytes as usize,
                is_requested: true,
            }],
        });
    }
    runs.sort_unstable_by(|a, b| {
        b.max_priority
            .cmp(&a.max_priority)
            .then_with(|| a.mesh_id.cmp(&b.mesh_id))
            .then_with(|| a.byte_offset.cmp(&b.byte_offset))
    });
    runs
}

#[cfg(target_arch = "wasm32")]
fn is_geometry_busy(
    assets_loaded: bool,
    page_faults: u32,
    in_flight_requests: usize,
    streaming_pending_work: bool,
    prefetch_only: bool,
    has_evictable_slots: bool,
) -> bool {
    if !assets_loaded {
        return true;
    }
    if page_faults == 0
        && in_flight_requests <= 4
        && (prefetch_only || !streaming_pending_work)
    {
        return false;
    }
    in_flight_requests > 0
        || streaming_pending_work
        || (page_faults > 0 && has_evictable_slots)
}

#[cfg(target_arch = "wasm32")]
fn step_hd_idle_gate(hd_idle_frames: &mut u32, assets_loaded: bool, geom_busy: bool) -> bool {
    if geom_busy {
        *hd_idle_frames = 0;
    } else {
        *hd_idle_frames = hd_idle_frames.saturating_add(1);
    }
    assets_loaded && !geom_busy && *hd_idle_frames >= HD_IDLE_HYSTERESIS_FRAMES
}

#[cfg(target_arch = "wasm32")]
fn all_assets_mounted(
    mounted_manifests: usize,
    total_manifests: usize,
    gi_submitted: bool,
    pbr_pack_loaded: bool,
) -> bool {
    mounted_manifests >= total_manifests && gi_submitted && pbr_pack_loaded
}

#[cfg(target_arch = "wasm32")]
mod web_loader {
    use std::cell::RefCell;
    use std::collections::VecDeque;

    use brick_nanite::{NaniteStreaming, NaniteTelemetry, NaniteWorld};
    use engine::prelude::*;
    use engine_meshlet::MeshletManifest;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    use super::{
        PageRun, all_assets_mounted, coalesce_runs, is_geometry_busy, step_hd_idle_gate,
    };
    use crate::gi::{GiUploadBridge, submit_gi_bytes};
    use crate::layout::HERO_ASSETS;
    use crate::pbr::{PbrHdRequest, PbrTexturePack};
    use crate::state::MonolithState;

    const ASSET_COUNT: usize = HERO_ASSETS.len();
    const MAX_IN_FLIGHT_REQUESTS: usize = 10;
    const MAX_IN_FLIGHT_PBR: usize = 4;

    #[derive(Default)]
    struct WebState {
        fetches_started: bool,
        ready_manifests: VecDeque<(usize, MeshletManifest)>,
        total_mounted: usize,
        pbr_base_bytes: Option<Vec<u8>>,
        gi_bytes: Option<Vec<u8>>,
        gi_submitted: bool,
        in_flight_requests: usize,
        in_flight_pbr: usize,
        hd_idle_frames: u32,
        completed_pages: VecDeque<(u32, Option<Vec<u8>>, bool)>,
        completed_pbr: VecDeque<(PbrHdRequest, Option<Vec<u8>>)>,
    }

    thread_local! {
        static WEB_STATE: RefCell<WebState> = RefCell::new(WebState::default());
    }

    async fn fetch_bytes(url: &str, range: Option<(u64, u64)>) -> Result<Vec<u8>, ()> {
        let window = web_sys::window().ok_or(())?;
        let init = web_sys::RequestInit::new();
        init.set_method("GET");
        if let Some((start, end)) = range {
            let headers = web_sys::Headers::new().map_err(|_| ())?;
            headers
                .set("Range", &format!("bytes={start}-{end}"))
                .map_err(|_| ())?;
            init.set_headers(&headers);
        }
        let req = web_sys::Request::new_with_str_and_init(url, &init).map_err(|_| ())?;
        let resp_val = JsFuture::from(window.fetch_with_request(&req))
            .await
            .map_err(|_| ())?;
        let resp: web_sys::Response = resp_val.dyn_into().map_err(|_| ())?;
        if !resp.ok() {
            return Err(());
        }
        let ab_promise = resp.array_buffer().map_err(|_| ())?;
        let ab = JsFuture::from(ab_promise).await.map_err(|_| ())?;
        Ok(js_sys::Uint8Array::new(&ab).to_vec())
    }

    fn spawn_manifest_range(start: usize, end: usize) {
        for (idx, &stem) in HERO_ASSETS.iter().enumerate().take(end).skip(start) {
            let url = format!("/monolith/{stem}.mlm");
            wasm_bindgen_futures::spawn_local(async move {
                if let Ok(bytes) = fetch_bytes(&url, None).await
                    && let Ok(manifest) = MeshletManifest::from_bytes(&bytes)
                {
                    WEB_STATE.with(|cell| {
                        cell.borrow_mut().ready_manifests.push_back((idx, manifest));
                    });
                }
            });
        }
    }

    fn start_asset_fetches() {
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(bytes) = fetch_bytes("/monolith/sponza_gi.bin", None).await {
                WEB_STATE.with(|cell| {
                    cell.borrow_mut().gi_bytes = Some(bytes);
                });
            }
        });
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(bytes) = fetch_bytes("/monolith/sponza_pbr_base.bin", None).await {
                WEB_STATE.with(|cell| {
                    cell.borrow_mut().pbr_base_bytes = Some(bytes);
                });
            }
        });
        spawn_manifest_range(0, ASSET_COUNT);
    }

    fn start_run_fetch(run: PageRun) {
        let url = format!("/monolith/{}.mlp", run.mesh_name);
        let total_len = run.total_bytes;
        let end = run.byte_offset.saturating_add(total_len).saturating_sub(1);
        wasm_bindgen_futures::spawn_local(async move {
            let fetched = fetch_bytes(&url, Some((run.byte_offset, end))).await.ok();
            let valid_bytes = fetched.filter(|b| b.len() == total_len as usize);
            WEB_STATE.with(|cell| {
                let mut st = cell.borrow_mut();
                st.in_flight_requests = st.in_flight_requests.saturating_sub(1);
                if let Some(bytes) = valid_bytes {
                    for item in run.items {
                        let raw = bytes
                            .get(item.rel_offset..item.rel_offset + item.byte_len)
                            .map(<[u8]>::to_vec);
                        if raw.is_some() || item.is_requested {
                            st.completed_pages
                                .push_back((item.global_page_id, raw, item.is_requested));
                        }
                    }
                } else {
                    for item in run.items {
                        if item.is_requested {
                            st.completed_pages
                                .push_back((item.global_page_id, None, true));
                        }
                    }
                }
            });
        });
    }

    fn start_pbr_hd_fetch(req: PbrHdRequest) {
        let end = req
            .byte_offset
            .saturating_add(u64::from(req.byte_len))
            .saturating_sub(1);
        wasm_bindgen_futures::spawn_local(async move {
            let fetched = fetch_bytes("/monolith/sponza_pbr_hd.bin", Some((req.byte_offset, end)))
                .await
                .ok()
                .filter(|b| b.len() == req.byte_len as usize);
            WEB_STATE.with(|cell| {
                let mut st = cell.borrow_mut();
                st.in_flight_pbr = st.in_flight_pbr.saturating_sub(1);
                st.completed_pbr.push_back((req, fetched));
            });
        });
    }

    fn pump_pbr_hd(pbr: &mut PbrTexturePack, allow_mid: bool, allow_hi: bool) {
        let done: Vec<(PbrHdRequest, Option<Vec<u8>>)> = WEB_STATE.with(|cell| {
            cell.borrow_mut().completed_pbr.drain(..).collect()
        });
        for (req, bytes) in done {
            let ok = bytes.is_some_and(|data| pbr.deliver_hd_blob(req, &data));
            if !ok {
                pbr.fail_hd_request(req);
            }
        }
        if !pbr.base_done || (!allow_mid && !allow_hi) {
            return;
        }
        let slots = WEB_STATE.with(|cell| {
            let st = cell.borrow();
            let cap = if allow_hi {
                if st.in_flight_requests > 12 {
                    1
                } else {
                    MAX_IN_FLIGHT_PBR
                }
            } else {
                2
            };
            cap.saturating_sub(st.in_flight_pbr)
        });
        if slots == 0 {
            return;
        }
        let reqs = pbr.take_hd_requests(slots, allow_hi);
        if !reqs.is_empty() {
            let count = reqs.len();
            WEB_STATE.with(|cell| {
                cell.borrow_mut().in_flight_pbr += count;
            });
            for req in reqs {
                start_pbr_hd_fetch(req);
            }
        }
    }

    fn mount_ready_manifests(
        world: &mut NaniteWorld,
        streaming: &mut NaniteStreaming,
        pbr_pack_loaded: bool,
        state: &mut MonolithState,
    ) -> usize {
        let ready: Vec<(usize, MeshletManifest)> = WEB_STATE.with(|cell| {
            let mut st = cell.borrow_mut();
            if !st.fetches_started {
                st.fetches_started = true;
                start_asset_fetches();
            }
            st.ready_manifests.drain(..).collect()
        });
        if !ready.is_empty() {
            for (idx, manifest) in ready {
                let stem = HERO_ASSETS[idx];
                let mesh_id = world.add_mesh(stem, &manifest);
                world.add_instance(mesh_id, Vec3::ZERO, Quat::IDENTITY, 1.0);
                WEB_STATE.with(|cell| {
                    cell.borrow_mut().total_mounted += 1;
                });
            }
            streaming.sync_world(world);
        }
        let (total_mounted, gi_submitted) = WEB_STATE.with(|cell| {
            let st = cell.borrow();
            (st.total_mounted, st.gi_submitted)
        });
        state.assets_loaded =
            all_assets_mounted(total_mounted, ASSET_COUNT, gi_submitted, pbr_pack_loaded);
        total_mounted
    }

    fn requeue_run_items(run: PageRun, back: &mut Vec<brick_nanite::PageFetchRequest>) {
        for item in run.items {
            if item.is_requested
                && let Some(req) = item.req
            {
                back.push(req);
            }
        }
    }

    fn dispatch_page_runs(
        world: &NaniteWorld,
        streaming: &mut NaniteStreaming,
        all_manifests: bool,
        prefetch_throttled: bool,
    ) {
        let req_cap = if !all_manifests || prefetch_throttled {
            4
        } else {
            MAX_IN_FLIGHT_REQUESTS
        };
        let req_slots =
            WEB_STATE.with(|cell| req_cap.saturating_sub(cell.borrow().in_flight_requests));
        if req_slots == 0 {
            return;
        }
        let reqs = streaming.take_fetch_requests(1024);
        if reqs.is_empty() {
            return;
        }
        let mut runs = coalesce_runs(reqs, world);
        let mut back = Vec::new();
        if runs.len() > req_slots {
            for run in runs.split_off(req_slots) {
                requeue_run_items(run, &mut back);
            }
        }
        if !back.is_empty() {
            streaming.requeue_fetch_requests(back);
        }
        let run_count = runs.len();
        if run_count > 0 {
            WEB_STATE.with(|cell| {
                cell.borrow_mut().in_flight_requests += run_count;
            });
            for run in runs {
                start_run_fetch(run);
            }
        }
    }

    pub(super) fn pump(
        mut world: ResMut<NaniteWorld>,
        mut streaming: ResMut<NaniteStreaming>,
        tel: &NaniteTelemetry,
        pbr: &mut PbrTexturePack,
        gi_bridge: &GiUploadBridge,
        mut state: ResMut<MonolithState>,
    ) {
        let ready_gi = WEB_STATE.with(|cell| {
            let mut st = cell.borrow_mut();
            if !st.gi_submitted {
                st.gi_bytes.take()
            } else {
                None
            }
        });
        if let Some(bytes) = ready_gi
            && submit_gi_bytes(gi_bridge, &bytes)
        {
            WEB_STATE.with(|cell| {
                cell.borrow_mut().gi_submitted = true;
            });
        }
        if !pbr.pack_loaded {
            let ready_pbr = WEB_STATE.with(|cell| cell.borrow_mut().pbr_base_bytes.take());
            if let Some(bytes) = ready_pbr {
                let _ = pbr.load_base_bytes(bytes);
            }
        }
        let mounted =
            mount_ready_manifests(&mut world, &mut streaming, pbr.pack_loaded, &mut state);
        if mounted == 0 {
            return;
        }

        let completed: Vec<(u32, Option<Vec<u8>>, bool)> = WEB_STATE.with(|cell| {
            cell.borrow_mut().completed_pages.drain(..).collect()
        });
        for (page_id, compressed, is_requested) in completed {
            match compressed {
                Some(raw) => streaming.deliver_compressed_page(page_id, raw, is_requested),
                None if is_requested => streaming.fail_page(page_id),
                None => {}
            }
        }

        let in_flight_pbr = WEB_STATE.with(|cell| cell.borrow().in_flight_pbr);
        let pbr_hd_pending = pbr.base_done && pbr.uploaded_hi_layers < pbr.layer_count;
        let prefetch_only = streaming.is_prefetch_only();
        let prefetch_throttled =
            tel.page_faults == 0 && prefetch_only && (in_flight_pbr > 0 || pbr_hd_pending);
        dispatch_page_runs(
            &world,
            &mut streaming,
            mounted >= ASSET_COUNT,
            prefetch_throttled,
        );
        let in_flight_requests = WEB_STATE.with(|cell| cell.borrow().in_flight_requests);
        let geom_busy = is_geometry_busy(
            state.assets_loaded,
            tel.page_faults,
            in_flight_requests,
            streaming.has_pending_work(),
            prefetch_only,
            streaming.has_evictable_slots(),
        );
        let allow_hi = WEB_STATE.with(|cell| {
            let mut st = cell.borrow_mut();
            step_hd_idle_gate(&mut st.hd_idle_frames, state.assets_loaded, geom_busy)
        });
        let mid_req_ok =
            in_flight_requests <= 2 || (tel.page_faults == 0 && in_flight_requests <= 4);
        let allow_mid = allow_hi
            || (state.assets_loaded
                && pbr.base_done
                && mid_req_ok
                && tel.page_faults < 128
                && pbr.mid_tier_pending());
        pump_pbr_hd(pbr, allow_mid, allow_hi);
    }
}

#[cfg(target_arch = "wasm32")]
fn load_sanctuary_assets(
    world: ResMut<NaniteWorld>,
    streaming: ResMut<brick_nanite::NaniteStreaming>,
    tel: Res<brick_nanite::NaniteTelemetry>,
    mut pbr: ResMut<PbrTexturePack>,
    pbr_bridge: Res<PbrUploadBridge>,
    gi_bridge: Res<GiUploadBridge>,
    state: ResMut<MonolithState>,
) {
    web_loader::pump(world, streaming, &tel, &mut pbr, &gi_bridge, state);
    pbr.pump_uploads(&pbr_bridge, 14);
}
