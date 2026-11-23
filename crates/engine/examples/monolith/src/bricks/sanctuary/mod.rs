engine::brick!(sanctuary);

#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read as _, Seek as _, SeekFrom};
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

use brick_nanite::{NaniteConfig, NaniteStreaming, NaniteWorld};
use engine::prelude::*;
#[cfg(target_arch = "wasm32")]
use engine_core::DeviceLimits;
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

#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource, Default)]
struct NativePageStore {
    files: Vec<Option<std::fs::File>>,
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
        bricks
            .resource::<NativeHdSource>()
            .resource::<NativePageStore>();
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
fn pump_native_pages(
    streaming: &mut NaniteStreaming,
    store: &mut NativePageStore,
    budget: usize,
) {
    let reqs = streaming.take_fetch_requests(budget);
    for req in reqs {
        let len = req.byte_len as usize;
        if len == 0 {
            streaming.fail_page(req.global_page_id);
            continue;
        }
        let Some(Some(file)) = store.files.get_mut(req.mesh_id as usize) else {
            streaming.fail_page(req.global_page_id);
            continue;
        };
        if file.seek(SeekFrom::Start(req.byte_offset)).is_err() {
            streaming.fail_page(req.global_page_id);
            continue;
        }
        let mut buf = vec![0u8; len];
        if file.read_exact(&mut buf).is_err() {
            streaming.fail_page(req.global_page_id);
            continue;
        }
        streaming.deliver_compressed_page(req.global_page_id, buf, true);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(SystemParam)]
struct NativeSanctuary<'w> {
    world: ResMut<'w, NaniteWorld>,
    streaming: ResMut<'w, NaniteStreaming>,
    cfg: ResMut<'w, NaniteConfig>,
    pbr: ResMut<'w, PbrTexturePack>,
    pbr_bridge: Res<'w, PbrUploadBridge>,
    gi_bridge: Res<'w, GiUploadBridge>,
    hd_src: ResMut<'w, NativeHdSource>,
    page_store: ResMut<'w, NativePageStore>,
    state: ResMut<'w, MonolithState>,
}

#[cfg(not(target_arch = "wasm32"))]
fn load_sanctuary_assets(mut p: NativeSanctuary<'_>) {
    if !p.state.assets_loaded {
        let Some(dir) = find_asset_dir() else {
            return;
        };
        p.cfg.compressed_cache_mb = 192;
        p.world.clear();
        p.page_store.files.clear();
        for stem in HERO_ASSETS {
            let mlm_path = dir.join(format!("{stem}.mlm"));
            let mlp_path = dir.join(format!("{stem}.mlp"));
            let Ok(bytes) = std::fs::read(&mlm_path) else {
                return;
            };
            let Ok(manifest) = MeshletManifest::from_bytes(&bytes) else {
                return;
            };
            let mesh_id = p.world.add_mesh(stem, &manifest) as usize;
            if p.page_store.files.len() <= mesh_id {
                p.page_store.files.resize_with(mesh_id + 1, || None);
            }
            p.page_store.files[mesh_id] = std::fs::File::open(&mlp_path).ok();
        }
        populate_sanctuary(&mut p.world);
        p.streaming.sync_world(&p.world);
        if let Ok(gi_bytes) = std::fs::read(dir.join("sponza_gi.bin")) {
            let _ = submit_gi_bytes(&p.gi_bridge, &gi_bytes);
        }
        if !p.pbr.pack_loaded
            && let Ok(base_bytes) = std::fs::read(dir.join("sponza_pbr_base.bin"))
        {
            let supports_bc = p.pbr_bridge.0.lock().is_ok_and(|q| q.supports_bc);
            let _ = p.pbr.load_base_bytes(base_bytes, supports_bc);
        }
        let hd_file = dir.join("sponza_pbr_hd.bin");
        if hd_file.is_file() {
            p.hd_src.path = Some(hd_file);
        }
        p.state.assets_loaded = true;
    }
    pump_native_pages(&mut p.streaming, &mut p.page_store, p.cfg.max_pages_per_frame);
    p.pbr.pump_uploads(&p.pbr_bridge, 14);
    if p.pbr.base_done
        && let Some(hd_path) = p.hd_src.path.as_ref()
    {
        pump_native_hd(&mut p.pbr, hd_path);
        p.pbr.pump_uploads(&p.pbr_bridge, 0);
    }
}

#[cfg(target_arch = "wasm32")]
const HD_IDLE_HYSTERESIS_FRAMES: u32 = 3;

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

    use brick_nanite::{
        CoalesceConfig, NaniteStreaming, NaniteTelemetry, NaniteWorld, PageRun, coalesce_page_runs,
    };
    use engine::prelude::*;
    use engine_meshlet::MeshletManifest;
    use wasm_bindgen::{JsCast, JsValue};
    use wasm_bindgen_futures::JsFuture;

    use super::{all_assets_mounted, is_geometry_busy, step_hd_idle_gate};
    use crate::gi::{GiUploadBridge, submit_gi_bytes};
    use crate::layout::HERO_ASSETS;
    use crate::pbr::{PbrHdRequest, PbrTexturePack};
    use crate::state::MonolithState;

    const ASSET_COUNT: usize = HERO_ASSETS.len();
    const CORE_ASSET_COUNT: usize = 5;
    const ASSET_VERSION: &str = "v25";
    const CDN_ASSET_BASE: &str = "https://d15rb9vgijd23p.cloudfront.net/monolith";
    const MIN_IN_FLIGHT_REQUESTS: usize = 8;
    const MAX_IN_FLIGHT_REQUESTS: usize = 24;

    #[derive(Clone, Copy, Debug)]
    struct StreamLinkEstimator {
        rtt_ms: f64,
        bw_bps: f64,
        busy_since_ms: f64,
        accum_active_ms: f64,
        active_transfers: usize,
        window_bytes: u64,
        has_rtt: bool,
        has_bw: bool,
    }

    impl Default for StreamLinkEstimator {
        fn default() -> Self {
            Self {
                rtt_ms: 50.0,
                bw_bps: 1.5 * 1024.0 * 1024.0,
                busy_since_ms: 0.0,
                accum_active_ms: 0.0,
                active_transfers: 0,
                window_bytes: 0,
                has_rtt: false,
                has_bw: false,
            }
        }
    }

    impl StreamLinkEstimator {
        fn on_transfer_start(&mut self, now_ms: f64) {
            if !now_ms.is_finite() {
                return;
            }
            if self.active_transfers == 0 {
                self.busy_since_ms = now_ms;
            }
            self.active_transfers = self.active_transfers.saturating_add(1);
        }

        fn on_transfer_end(&mut self, now_ms: f64, bytes: u64, ttfb_ms: f64) {
            if !now_ms.is_finite() {
                self.active_transfers = self.active_transfers.saturating_sub(1);
                return;
            }
            let elapsed_segment = if self.active_transfers > 0 {
                (now_ms - self.busy_since_ms).max(0.0)
            } else {
                0.0
            };
            let active_ms = self.accum_active_ms + elapsed_segment;
            self.active_transfers = self.active_transfers.saturating_sub(1);
            if self.active_transfers == 0 {
                self.accum_active_ms = active_ms;
            }
            if bytes > 0 && ttfb_ms.is_finite() {
                let rtt_sample = ttfb_ms.clamp(1.0, 2000.0);
                if self.has_rtt {
                    self.rtt_ms = self.rtt_ms * 0.75 + rtt_sample * 0.25;
                } else {
                    self.rtt_ms = rtt_sample;
                    self.has_rtt = true;
                }
                self.window_bytes = self.window_bytes.saturating_add(bytes);
            }
            if active_ms >= 500.0 && self.window_bytes >= 64 * 1024 {
                let sample_bps = (self.window_bytes as f64 / (active_ms * 0.001))
                    .clamp(128.0 * 1024.0, 256.0e6);
                if self.has_bw {
                    self.bw_bps = self.bw_bps * 0.65 + sample_bps * 0.35;
                } else {
                    self.bw_bps = sample_bps;
                    self.has_bw = true;
                }
                self.window_bytes = 0;
                self.accum_active_ms = 0.0;
                self.busy_since_ms = now_ms;
            }
        }

        fn bandwidth_bps(&self) -> f64 {
            self.bw_bps
        }

        fn bdp_bytes(&self) -> f64 {
            self.bw_bps * (self.rtt_ms * 0.001)
        }

        fn recommended_coalesce(&self) -> CoalesceConfig {
            let bdp = self.bdp_bytes();
            let max_gap_bytes = ((bdp * 0.08) as u64).clamp(12 * 1024, 48 * 1024);
            let (max_run_pages, max_run_bytes) = if self.bw_bps >= 10.0 * 1024.0 * 1024.0 {
                (48, 512 * 1024)
            } else if self.bw_bps >= 4.0 * 1024.0 * 1024.0 {
                (36, 384 * 1024)
            } else {
                (24, 256 * 1024)
            };
            CoalesceConfig {
                max_run_pages,
                max_run_bytes,
                max_gap_bytes,
            }
        }

        fn recommended_concurrency(&self, min_cap: usize, max_cap: usize) -> usize {
            let by_bw = (self.bw_bps / (512.0 * 1024.0)).round() as usize;
            let by_bdp = (self.bdp_bytes() / (32.0 * 1024.0)).ceil() as usize;
            by_bw.max(by_bdp).clamp(min_cap, max_cap)
        }
    }

    fn asset_base() -> &'static str {
        let Some(window) = web_sys::window() else {
            return "/monolith";
        };
        let Ok(loc) = js_sys::Reflect::get(&window, &JsValue::from_str("location")) else {
            return "/monolith";
        };
        let search = js_sys::Reflect::get(&loc, &JsValue::from_str("search"))
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        if let Ok(params) = web_sys::UrlSearchParams::new_with_str(&search)
            && let Some(mode) = params.get("cdn")
        {
            if mode == "0" || mode == "origin" || mode == "off" {
                return "/monolith";
            }
            if mode == "1" || mode == "cf" || mode == "on" {
                return CDN_ASSET_BASE;
            }
        }
        let host = js_sys::Reflect::get(&loc, &JsValue::from_str("hostname"))
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        if host.is_empty() || host == "localhost" || host == "127.0.0.1" {
            return "/monolith";
        }
        CDN_ASSET_BASE
    }

    #[derive(Default)]
    struct WebState {
        config_applied: bool,
        fetches_started: bool,
        decor_fetches_started: bool,
        ready_manifests: VecDeque<(usize, MeshletManifest)>,
        total_mounted: usize,
        pbr_base_bytes: Option<Vec<u8>>,
        pbr_base_failed: bool,
        gi_bytes: Option<Vec<u8>>,
        gi_submitted: bool,
        in_flight_requests: usize,
        active_req_cap: usize,
        in_flight_pbr: usize,
        hd_idle_frames: u32,
        link: StreamLinkEstimator,
        queued_runs: VecDeque<PageRun>,
        completed_pages: VecDeque<(u32, Option<Vec<u8>>, bool)>,
        completed_pbr: VecDeque<(PbrHdRequest, Option<Vec<u8>>)>,
    }

    thread_local! {
        static WEB_STATE: RefCell<WebState> = RefCell::new(WebState::default());
    }

    async fn fetch_bytes(url: &str, range: Option<(u64, u64)>) -> Result<(Vec<u8>, f64), ()> {
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
        let t0 = js_sys::Date::now();
        let resp_val = JsFuture::from(window.fetch_with_request(&req))
            .await
            .map_err(|_| ())?;
        let ttfb_ms = (js_sys::Date::now() - t0).max(1.0);
        let resp: web_sys::Response = resp_val.dyn_into().map_err(|_| ())?;
        if !resp.ok() {
            return Err(());
        }
        let ab_promise = resp.array_buffer().map_err(|_| ())?;
        let ab = JsFuture::from(ab_promise).await.map_err(|_| ())?;
        Ok((js_sys::Uint8Array::new(&ab).to_vec(), ttfb_ms))
    }

    fn spawn_manifest_range(start: usize, end: usize) {
        let base = asset_base();
        for (idx, &stem) in HERO_ASSETS.iter().enumerate().take(end).skip(start) {
            let url = format!("{base}/{stem}.mlm?v={ASSET_VERSION}");
            let start_ms = js_sys::Date::now();
            WEB_STATE.with(|cell| cell.borrow_mut().link.on_transfer_start(start_ms));
            wasm_bindgen_futures::spawn_local(async move {
                match fetch_bytes(&url, None).await {
                    Ok((bytes, ttfb_ms)) => {
                        let wire_est = (bytes.len() as u64) / 2;
                        let now = js_sys::Date::now();
                        WEB_STATE.with(|cell| {
                            cell.borrow_mut()
                                .link
                                .on_transfer_end(now, wire_est, ttfb_ms);
                        });
                        if let Ok(manifest) = MeshletManifest::from_bytes(&bytes) {
                            WEB_STATE.with(|cell| {
                                cell.borrow_mut().ready_manifests.push_back((idx, manifest));
                            });
                        }
                    }
                    Err(()) => {
                        let now = js_sys::Date::now();
                        WEB_STATE.with(|cell| {
                            cell.borrow_mut().link.on_transfer_end(now, 0, 0.0);
                        });
                    }
                }
            });
        }
    }

    fn start_asset_fetches() {
        let base = asset_base();
        let gi_url = format!("{base}/sponza_gi.bin?v={ASSET_VERSION}");
        let start_ms = js_sys::Date::now();
        WEB_STATE.with(|cell| {
            let mut st = cell.borrow_mut();
            st.link.on_transfer_start(start_ms);
            st.link.on_transfer_start(start_ms);
        });
        wasm_bindgen_futures::spawn_local(async move {
            match fetch_bytes(&gi_url, None).await {
                Ok((bytes, ttfb_ms)) => {
                    let now = js_sys::Date::now();
                    let wire_est = bytes.len() as u64;
                    WEB_STATE.with(|cell| {
                        let mut st = cell.borrow_mut();
                        st.link.on_transfer_end(now, wire_est, ttfb_ms);
                        st.gi_bytes = Some(bytes);
                    });
                }
                Err(()) => {
                    let now = js_sys::Date::now();
                    WEB_STATE.with(|cell| {
                        cell.borrow_mut().link.on_transfer_end(now, 0, 0.0);
                    });
                }
            }
        });
        let pbr_url = format!("{base}/sponza_pbr_base.bin?v={ASSET_VERSION}");
        wasm_bindgen_futures::spawn_local(async move {
            match fetch_bytes(&pbr_url, None).await {
                Ok((bytes, ttfb_ms)) => {
                    let now = js_sys::Date::now();
                    let wire_est = bytes.len() as u64;
                    WEB_STATE.with(|cell| {
                        let mut st = cell.borrow_mut();
                        st.link.on_transfer_end(now, wire_est, ttfb_ms);
                        st.pbr_base_bytes = Some(bytes);
                    });
                }
                Err(()) => {
                    let now = js_sys::Date::now();
                    WEB_STATE.with(|cell| {
                        let mut st = cell.borrow_mut();
                        st.link.on_transfer_end(now, 0, 0.0);
                        st.pbr_base_failed = true;
                    });
                }
            }
        });
        spawn_manifest_range(0, CORE_ASSET_COUNT);
    }

    fn drain_queued_runs() {
        let to_launch: Vec<PageRun> = WEB_STATE.with(|cell| {
            let mut st = cell.borrow_mut();
            let cap = st.active_req_cap.max(1);
            let mut batch = Vec::new();
            while st.in_flight_requests < cap {
                let Some(run) = st.queued_runs.pop_front() else {
                    break;
                };
                st.in_flight_requests += 1;
                batch.push(run);
            }
            batch
        });
        for run in to_launch {
            start_run_fetch(run);
        }
    }

    fn start_run_fetch(run: PageRun) {
        let base = asset_base();
        let url = format!("{base}/{}.mlp?v={ASSET_VERSION}", run.mesh_name);
        let total_len = run.total_bytes;
        let end = run.byte_offset.saturating_add(total_len).saturating_sub(1);
        let start_ms = js_sys::Date::now();
        WEB_STATE.with(|cell| cell.borrow_mut().link.on_transfer_start(start_ms));
        wasm_bindgen_futures::spawn_local(async move {
            let fetched = fetch_bytes(&url, Some((run.byte_offset, end))).await.ok();
            let now = js_sys::Date::now();
            let valid = fetched.filter(|(b, _)| b.len() == total_len as usize);
            WEB_STATE.with(|cell| {
                let mut st = cell.borrow_mut();
                st.in_flight_requests = st.in_flight_requests.saturating_sub(1);
                if let Some((bytes, ttfb_ms)) = valid {
                    st.link.on_transfer_end(now, total_len, ttfb_ms);
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
                    st.link.on_transfer_end(now, 0, 0.0);
                    for item in run.items {
                        if item.is_requested {
                            st.completed_pages
                                .push_back((item.global_page_id, None, true));
                        }
                    }
                }
            });
            drain_queued_runs();
        });
    }

    fn start_pbr_hd_fetch(req: PbrHdRequest) {
        let base = asset_base();
        let end = req
            .byte_offset
            .saturating_add(u64::from(req.byte_len))
            .saturating_sub(1);
        let hd_url = format!("{base}/sponza_pbr_hd.bin?v={ASSET_VERSION}");
        let start_ms = js_sys::Date::now();
        WEB_STATE.with(|cell| cell.borrow_mut().link.on_transfer_start(start_ms));
        wasm_bindgen_futures::spawn_local(async move {
            let fetched = fetch_bytes(&hd_url, Some((req.byte_offset, end)))
                .await
                .ok()
                .filter(|(b, _)| b.len() == req.byte_len as usize);
            let now = js_sys::Date::now();
            WEB_STATE.with(|cell| {
                let mut st = cell.borrow_mut();
                st.in_flight_pbr = st.in_flight_pbr.saturating_sub(1);
                let data = match fetched {
                    Some((b, ttfb_ms)) => {
                        st.link
                            .on_transfer_end(now, u64::from(req.byte_len), ttfb_ms);
                        Some(b)
                    }
                    None => {
                        st.link.on_transfer_end(now, 0, 0.0);
                        None
                    }
                };
                st.completed_pbr.push_back((req, data));
            });
        });
    }

    fn pump_pbr_hd(pbr: &mut PbrTexturePack, allow_mid: bool, allow_hi: bool) {
        let max_decodes = if allow_hi { 4 } else { 2 };
        let done: Vec<(PbrHdRequest, Option<Vec<u8>>)> = WEB_STATE.with(|cell| {
            let mut st = cell.borrow_mut();
            let count = max_decodes.min(st.completed_pbr.len());
            st.completed_pbr.drain(..count).collect()
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
            let bw = st.link.bandwidth_bps();
            let cap: usize = if allow_hi {
                if bw >= 8.0 * 1024.0 * 1024.0 {
                    8
                } else {
                    4
                }
            } else if bw >= 8.0 * 1024.0 * 1024.0 {
                6
            } else if bw >= 3.0 * 1024.0 * 1024.0 {
                4
            } else {
                3
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
        let (start_initial, ready): (bool, Vec<(usize, MeshletManifest)>) = WEB_STATE.with(|cell| {
            let mut st = cell.borrow_mut();
            let start = !st.fetches_started;
            if start {
                st.fetches_started = true;
            }
            (start, st.ready_manifests.drain(..).collect())
        });
        if start_initial {
            start_asset_fetches();
        }
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
        let (total_mounted, gi_submitted, start_decor) = WEB_STATE.with(|cell| {
            let mut st = cell.borrow_mut();
            let pbr_settled = pbr_pack_loaded || st.pbr_base_failed;
            let trigger_decor = !st.decor_fetches_started
                && pbr_settled
                && st.total_mounted >= CORE_ASSET_COUNT;
            if trigger_decor {
                st.decor_fetches_started = true;
            }
            (st.total_mounted, st.gi_submitted, trigger_decor)
        });
        if start_decor {
            spawn_manifest_range(CORE_ASSET_COUNT, ASSET_COUNT);
        }
        state.assets_loaded =
            all_assets_mounted(total_mounted, ASSET_COUNT, gi_submitted, pbr_pack_loaded);
        total_mounted
    }

    struct DispatchFlags {
        roots_only: bool,
        all_manifests: bool,
        mid_pbr_active: bool,
        prefetch_throttled: bool,
    }

    fn dispatch_page_runs(
        world: &NaniteWorld,
        streaming: &mut NaniteStreaming,
        flags: DispatchFlags,
    ) {
        let stale: Vec<PageRun> = WEB_STATE.with(|cell| {
            cell.borrow_mut().queued_runs.drain(..).collect()
        });
        if !stale.is_empty() {
            let mut returned = Vec::new();
            for run in stale {
                run.into_requested(&mut returned);
            }
            if !returned.is_empty() {
                streaming.requeue_fetch_requests(returned);
            }
        }
        let (req_cap, in_flight, coalesce_cfg) = WEB_STATE.with(|cell| {
            let mut st = cell.borrow_mut();
            let link = st.link;
            let bw = link.bandwidth_bps();
            let cap = if flags.roots_only {
                if bw >= 8.0 * 1024.0 * 1024.0 { 4 } else { 2 }
            } else if flags.prefetch_throttled {
                2
            } else if !flags.all_manifests {
                if bw >= 8.0 * 1024.0 * 1024.0 { 12 } else { 4 }
            } else if flags.mid_pbr_active && bw < 8.0 * 1024.0 * 1024.0 {
                if bw >= 3.0 * 1024.0 * 1024.0 { 6 } else { 4 }
            } else {
                link.recommended_concurrency(MIN_IN_FLIGHT_REQUESTS, MAX_IN_FLIGHT_REQUESTS)
            };
            st.active_req_cap = cap;
            (cap, st.in_flight_requests, link.recommended_coalesce())
        });
        let target_buffer = if flags.roots_only || flags.prefetch_throttled {
            req_cap
        } else {
            req_cap.saturating_mul(2)
        };
        let buf_slots = target_buffer.saturating_sub(in_flight);
        if buf_slots == 0 {
            return;
        }
        let reqs = streaming.take_fetch_requests(1024);
        if reqs.is_empty() {
            return;
        }
        let mut back = Vec::new();
        let reqs = if flags.roots_only {
            let mut keep = Vec::new();
            for req in reqs {
                if req.priority >= u32::MAX - 1 {
                    keep.push(req);
                } else {
                    back.push(req);
                }
            }
            keep
        } else {
            reqs
        };
        let mut runs = coalesce_page_runs(reqs, world, coalesce_cfg);
        if runs.len() > buf_slots {
            for run in runs.split_off(buf_slots) {
                run.into_requested(&mut back);
            }
        }
        if !back.is_empty() {
            streaming.requeue_fetch_requests(back);
        }
        if !runs.is_empty() {
            WEB_STATE.with(|cell| {
                cell.borrow_mut().queued_runs.extend(runs);
            });
            drain_queued_runs();
        }
    }

    pub(super) fn pump(
        mut world: ResMut<NaniteWorld>,
        mut streaming: ResMut<NaniteStreaming>,
        tel: &NaniteTelemetry,
        pbr: &mut PbrTexturePack,
        pbr_bridge: &crate::pbr::PbrUploadBridge,
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
            let (queried, supports_bc) = pbr_bridge
                .0
                .lock()
                .map_or((false, true), |q| (q.gpu_queried, q.supports_bc));
            if queried
                && let Some(bytes) = WEB_STATE.with(|cell| cell.borrow_mut().pbr_base_bytes.take())
                && !pbr.load_base_bytes(bytes, supports_bc)
            {
                WEB_STATE.with(|cell| {
                    cell.borrow_mut().pbr_base_failed = true;
                });
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
        let pbr_hd_pending = pbr.base_done && pbr.hd_pending();
        let prefetch_only = streaming.is_prefetch_only();
        let prefetch_throttled =
            tel.page_faults == 0 && prefetch_only && (in_flight_pbr > 0 || pbr_hd_pending);
        let pbr_settled =
            pbr.pack_loaded || WEB_STATE.with(|cell| cell.borrow().pbr_base_failed);
        let roots_only = !pbr_settled || mounted < CORE_ASSET_COUNT;
        let mid_pbr_active = pbr.base_done
            && mounted >= CORE_ASSET_COUNT
            && streaming.all_roots_resident(&world)
            && pbr.mid_tier_pending();
        dispatch_page_runs(
            &world,
            &mut streaming,
            DispatchFlags {
                roots_only,
                all_manifests: mounted >= ASSET_COUNT,
                mid_pbr_active,
                prefetch_throttled,
            },
        );
        let active_requests = WEB_STATE.with(|cell| {
            let st = cell.borrow();
            st.in_flight_requests + st.queued_runs.len()
        });
        let geom_busy = is_geometry_busy(
            state.assets_loaded,
            tel.page_faults,
            active_requests,
            streaming.has_pending_work(),
            prefetch_only,
            streaming.has_evictable_slots(),
        );
        let allow_hi = WEB_STATE.with(|cell| {
            let mut st = cell.borrow_mut();
            step_hd_idle_gate(&mut st.hd_idle_frames, state.assets_loaded, geom_busy)
        });
        let allow_mid = allow_hi || mid_pbr_active;
        pump_pbr_hd(pbr, allow_mid, allow_hi);
    }

    pub(super) fn configure_nanite(
        cfg: &mut brick_nanite::NaniteConfig,
        limits: &engine_core::DeviceLimits,
    ) {
        let should_apply = WEB_STATE.with(|cell| {
            let mut st = cell.borrow_mut();
            if st.config_applied {
                return false;
            }
            st.config_applied = true;
            true
        });
        if !should_apply {
            return;
        }
        cfg.lod_error_scale = 1.5;
        cfg.compressed_cache_mb = 64;
        if limits.storage_room > 0 && limits.storage_room <= 128 * 1024 * 1024 {
            cfg.max_heap_mb = 64;
            let cap = cfg.effective_heap_cap_pages(limits.storage_room);
            cfg.heap_budget = cfg.heap_budget.clamp_to_pages(cap);
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(SystemParam)]
struct WebSanctuary<'w> {
    world: ResMut<'w, NaniteWorld>,
    streaming: ResMut<'w, NaniteStreaming>,
    cfg: ResMut<'w, NaniteConfig>,
    limits: Res<'w, DeviceLimits>,
    tel: Res<'w, brick_nanite::NaniteTelemetry>,
    pbr: ResMut<'w, PbrTexturePack>,
    pbr_bridge: Res<'w, PbrUploadBridge>,
    gi_bridge: Res<'w, GiUploadBridge>,
    state: ResMut<'w, MonolithState>,
}

#[cfg(target_arch = "wasm32")]
fn load_sanctuary_assets(mut p: WebSanctuary<'_>) {
    web_loader::configure_nanite(&mut p.cfg, &p.limits);
    web_loader::pump(
        p.world,
        p.streaming,
        &p.tel,
        &mut p.pbr,
        &p.pbr_bridge,
        &p.gi_bridge,
        p.state,
    );
    p.pbr.pump_uploads(&p.pbr_bridge, 4);
}
