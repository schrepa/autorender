engine::brick!(sanctuary);

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

use brick_nanite::NaniteWorld;
use engine::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use engine_meshlet::MeshletManifest;

#[cfg(not(target_arch = "wasm32"))]
use crate::layout::{HERO_ASSETS, populate_sanctuary};
use crate::pbr::{PbrTexturePack, PbrUploadBridge};
use crate::state::MonolithState;

pub struct SanctuaryBrick;

impl Brick for SanctuaryBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .resource::<PbrTexturePack>()
            .resource::<PbrUploadBridge>()
            .systems(Schedule::BeforeTicks, load_sanctuary_assets);
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
fn load_sanctuary_assets(
    mut world: ResMut<NaniteWorld>,
    mut pbr: ResMut<PbrTexturePack>,
    pbr_bridge: Res<PbrUploadBridge>,
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
        if !pbr.pack_loaded
            && let Ok(pack_bytes) = std::fs::read(dir.join("sponza_pbr.bin"))
        {
            let _ = pbr.load_bytes(pack_bytes);
        }
        state.assets_loaded = true;
    }
    pbr.pump_uploads(&pbr_bridge, 4);
}

#[cfg(target_arch = "wasm32")]
mod web_loader {
    use std::cell::RefCell;
    use std::collections::VecDeque;

    use brick_nanite::{NaniteStreaming, NaniteWorld, PageFetchRequest};
    use engine::prelude::*;
    use engine_meshlet::{MeshletManifest, PageSpan, decompress_page_words};
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    use crate::layout::{HERO_ASSETS, populate_sanctuary};
    use crate::pbr::PbrTexturePack;
    use crate::state::MonolithState;

    const ASSET_COUNT: usize = HERO_ASSETS.len();
    const MAX_IN_FLIGHT_REQUESTS: usize = 24;
    const MAX_RUN_PAGES: usize = 16;
    const MAX_RUN_BYTES: u64 = 256 * 1024;
    const PREFETCH_EXTRA_PAGES: u32 = 4;

    struct WebState {
        manifests_started: bool,
        manifests: [Option<MeshletManifest>; ASSET_COUNT],
        prefetched: [Vec<(u32, Vec<u32>)>; ASSET_COUNT],
        mesh_page_offsets: [u32; ASSET_COUNT],
        pbr_bytes: Option<Vec<u8>>,
        world_synced: bool,
        in_flight_requests: usize,
        completed_pages: VecDeque<(u32, Option<Vec<u32>>)>,
    }

    impl Default for WebState {
        fn default() -> Self {
            Self {
                manifests_started: false,
                manifests: std::array::from_fn(|_| None),
                prefetched: std::array::from_fn(|_| Vec::new()),
                mesh_page_offsets: [0; ASSET_COUNT],
                pbr_bytes: None,
                world_synced: false,
                in_flight_requests: 0,
                completed_pages: VecDeque::new(),
            }
        }
    }

    struct PageRunItem {
        global_page_id: u32,
        rel_offset: usize,
        byte_len: usize,
    }

    struct PageRun {
        mesh_name: String,
        byte_offset: u64,
        total_bytes: u64,
        items: Vec<PageRunItem>,
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
            let _ = js_sys::Reflect::set(
                &init,
                &wasm_bindgen::JsValue::from_str("cache"),
                &wasm_bindgen::JsValue::from_str("no-store"),
            );
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

    fn prefetch_base_pages(mesh_idx: usize, stem: &'static str, spans: Vec<PageSpan>) {
        if spans.is_empty() {
            return;
        }
        let start = spans[0].byte_offset;
        let last = spans[spans.len() - 1];
        let total = last
            .byte_offset
            .saturating_add(u64::from(last.byte_len))
            .saturating_sub(start);
        if total == 0 {
            return;
        }
        let end = start.saturating_add(total).saturating_sub(1);
        let url = format!("/monolith/{stem}.mlp");
        wasm_bindgen_futures::spawn_local(async move {
            let Ok(bytes) = fetch_bytes(&url, Some((start, end))).await else {
                return;
            };
            if bytes.len() != total as usize {
                return;
            }
            let mut decoded = Vec::with_capacity(spans.len());
            for (local_id, span) in spans.iter().enumerate() {
                let rel = span.byte_offset.saturating_sub(start) as usize;
                let len = span.byte_len as usize;
                if let Some(slice) = bytes.get(rel..rel + len)
                    && let Ok(words) = decompress_page_words(slice)
                {
                    decoded.push((local_id as u32, words));
                }
            }
            WEB_STATE.with(|cell| {
                let mut st = cell.borrow_mut();
                if st.world_synced {
                    let base = st.mesh_page_offsets[mesh_idx];
                    for (local_id, words) in decoded {
                        st.completed_pages.push_back((base + local_id, Some(words)));
                    }
                } else {
                    st.prefetched[mesh_idx] = decoded;
                }
            });
        });
    }

    fn start_manifests() {
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(bytes) = fetch_bytes("/monolith/sponza_pbr.bin", None).await {
                WEB_STATE.with(|cell| {
                    cell.borrow_mut().pbr_bytes = Some(bytes);
                });
            }
        });
        for (idx, &stem) in HERO_ASSETS.iter().enumerate() {
            let url = format!("/monolith/{stem}.mlm");
            wasm_bindgen_futures::spawn_local(async move {
                if let Ok(bytes) = fetch_bytes(&url, None).await
                    && let Ok(manifest) = MeshletManifest::from_bytes(&bytes)
                {
                    let count = (manifest.root_page_count.saturating_add(PREFETCH_EXTRA_PAGES)
                        as usize)
                        .min(manifest.page_spans.len());
                    let base_spans = manifest.page_spans[..count].to_vec();
                    WEB_STATE.with(|cell| {
                        cell.borrow_mut().manifests[idx] = Some(manifest);
                    });
                    prefetch_base_pages(idx, stem, base_spans);
                }
            });
        }
    }

    fn coalesce_runs(mut reqs: Vec<PageFetchRequest>) -> Vec<PageRun> {
        reqs.sort_unstable_by(|a, b| {
            a.mesh_name
                .cmp(&b.mesh_name)
                .then_with(|| a.byte_offset.cmp(&b.byte_offset))
        });
        let mut runs: Vec<PageRun> = Vec::new();
        for req in reqs {
            if let Some(last) = runs.last_mut() {
                let expected = last.byte_offset.saturating_add(last.total_bytes);
                let next_total = last.total_bytes.saturating_add(req.byte_len);
                if last.mesh_name == req.mesh_name
                    && req.byte_offset == expected
                    && last.items.len() < MAX_RUN_PAGES
                    && next_total <= MAX_RUN_BYTES
                {
                    last.items.push(PageRunItem {
                        global_page_id: req.global_page_id,
                        rel_offset: last.total_bytes as usize,
                        byte_len: req.byte_len as usize,
                    });
                    last.total_bytes = next_total;
                    continue;
                }
            }
            runs.push(PageRun {
                mesh_name: req.mesh_name,
                byte_offset: req.byte_offset,
                total_bytes: req.byte_len,
                items: vec![PageRunItem {
                    global_page_id: req.global_page_id,
                    rel_offset: 0,
                    byte_len: req.byte_len as usize,
                }],
            });
        }
        runs
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
                        let words = bytes
                            .get(item.rel_offset..item.rel_offset + item.byte_len)
                            .and_then(|slice| decompress_page_words(slice).ok());
                        st.completed_pages.push_back((item.global_page_id, words));
                    }
                } else {
                    for item in run.items {
                        st.completed_pages.push_back((item.global_page_id, None));
                    }
                }
            });
        });
    }

    pub(super) fn pump(
        mut world: ResMut<NaniteWorld>,
        mut streaming: ResMut<NaniteStreaming>,
        pbr: &mut PbrTexturePack,
        mut state: ResMut<MonolithState>,
    ) {
        if !pbr.pack_loaded {
            let ready_pbr = WEB_STATE.with(|cell| cell.borrow_mut().pbr_bytes.take());
            if let Some(bytes) = ready_pbr {
                let _ = pbr.load_bytes(bytes);
            }
        }
        if !state.assets_loaded {
            let ready_manifests = WEB_STATE.with(|cell| {
                let mut st = cell.borrow_mut();
                if !st.manifests_started {
                    st.manifests_started = true;
                    start_manifests();
                }
                if st.manifests.iter().all(Option::is_some) {
                    let mut out = Vec::with_capacity(ASSET_COUNT);
                    for slot in &mut st.manifests {
                        if let Some(m) = slot.take() {
                            out.push(m);
                        }
                    }
                    Some(out)
                } else {
                    None
                }
            });
            if let Some(manifests) = ready_manifests {
                world.clear();
                let mut offsets = [0_u32; ASSET_COUNT];
                for (idx, (stem, manifest)) in HERO_ASSETS.iter().zip(manifests.iter()).enumerate()
                {
                    offsets[idx] = world.total_pages;
                    world.add_mesh(*stem, manifest);
                }
                populate_sanctuary(&mut world);
                streaming.sync_world(&world);
                let pre = WEB_STATE.with(|cell| {
                    let mut st = cell.borrow_mut();
                    st.mesh_page_offsets = offsets;
                    st.world_synced = true;
                    let mut out = Vec::new();
                    for (idx, slot) in st.prefetched.iter_mut().enumerate() {
                        let base = offsets[idx];
                        for (local_id, words) in slot.drain(..) {
                            out.push((base + local_id, words));
                        }
                    }
                    out
                });
                for (global_id, words) in pre {
                    streaming.deliver_page(global_id, words);
                }
                state.assets_loaded = true;
            }
            return;
        }

        let completed: Vec<(u32, Option<Vec<u32>>)> = WEB_STATE.with(|cell| {
            cell.borrow_mut().completed_pages.drain(..).collect()
        });
        for (page_id, words) in completed {
            match words {
                Some(w) => streaming.deliver_page(page_id, w),
                None => streaming.fail_page(page_id),
            }
        }

        let req_slots = WEB_STATE
            .with(|cell| MAX_IN_FLIGHT_REQUESTS.saturating_sub(cell.borrow().in_flight_requests));
        if req_slots > 0 {
            let reqs = streaming.take_fetch_requests((req_slots * 4).min(64));
            if !reqs.is_empty() {
                let runs = coalesce_runs(reqs);
                let run_count = runs.len();
                WEB_STATE.with(|cell| {
                    cell.borrow_mut().in_flight_requests += run_count;
                });
                for run in runs {
                    start_run_fetch(run);
                }
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn load_sanctuary_assets(
    world: ResMut<NaniteWorld>,
    streaming: ResMut<brick_nanite::NaniteStreaming>,
    mut pbr: ResMut<PbrTexturePack>,
    pbr_bridge: Res<PbrUploadBridge>,
    state: ResMut<MonolithState>,
) {
    web_loader::pump(world, streaming, &mut pbr, state);
    pbr.pump_uploads(&pbr_bridge, 1);
}
