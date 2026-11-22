engine::brick!(sanctuary);

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

use brick_nanite::{NANITE_P1_PASS, NaniteWorld};
use engine::prelude::*;
use engine_core::{Error, GpuStruct, KindDecl};
#[cfg(not(target_arch = "wasm32"))]
use engine_meshlet::MeshletManifest;
use engine_render::{
    Allocation, Bind, Buffer, DepthUse, Draw, GpuPart, Install, Lifetime, PerBatch, Pipeline,
    Prepare, RenderPipeline, Scene, Shape, wgpu,
};

use crate::ground::{GroundVertex, build_ground_mesh};
#[cfg(not(target_arch = "wasm32"))]
use crate::layout::{HERO_ASSETS, populate_sanctuary};
use crate::names::GROUND_KIND;
use crate::state::MonolithState;

const GENERATED: &[&str] = &[<GroundVertex as GpuStruct>::WGSL];

pub struct SanctuaryBrick;

impl Brick for SanctuaryBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .draw_kind(KindDecl {
                kind: GROUND_KIND,
                passes: &[NANITE_P1_PASS],
            })
            .generated(GENERATED)
            .draws::<GroundGpu>(GROUND_KIND, GroundGpu::draw_ground)
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
        .find(|dir| dir.join("colossus_sentinel.mlm").is_file())
}

#[cfg(not(target_arch = "wasm32"))]
fn load_sanctuary_assets(mut world: ResMut<NaniteWorld>, mut state: ResMut<MonolithState>) {
    if state.assets_loaded {
        return;
    }
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
    state.assets_loaded = true;
}

#[cfg(target_arch = "wasm32")]
mod web_loader {
    use std::cell::RefCell;
    use std::collections::VecDeque;

    use brick_nanite::{NaniteStreaming, NaniteWorld};
    use engine::prelude::*;
    use engine_meshlet::{MeshletManifest, page_bytes_as_words};
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    use crate::layout::{HERO_ASSETS, populate_sanctuary};
    use crate::state::MonolithState;

    #[derive(Default)]
    struct WebState {
        manifests_started: bool,
        manifests: [Option<MeshletManifest>; 5],
        in_flight_pages: usize,
        completed_pages: VecDeque<(u32, Option<Vec<u32>>)>,
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

    fn start_manifests() {
        for (idx, stem) in HERO_ASSETS.iter().enumerate() {
            let url = format!("/monolith/{stem}.mlm");
            wasm_bindgen_futures::spawn_local(async move {
                if let Ok(bytes) = fetch_bytes(&url, None).await
                    && let Ok(manifest) = MeshletManifest::from_bytes(&bytes)
                {
                    WEB_STATE.with(|cell| {
                        cell.borrow_mut().manifests[idx] = Some(manifest);
                    });
                }
            });
        }
    }

    fn start_page_fetch(mesh_name: String, global_page_id: u32, offset: u64, len: u64) {
        let url = format!("/monolith/{mesh_name}.mlp");
        let end = offset.saturating_add(len).saturating_sub(1);
        wasm_bindgen_futures::spawn_local(async move {
            let words = fetch_bytes(&url, Some((offset, end)))
                .await
                .ok()
                .and_then(|bytes| page_bytes_as_words(&bytes).ok());
            WEB_STATE.with(|cell| {
                let mut st = cell.borrow_mut();
                st.in_flight_pages = st.in_flight_pages.saturating_sub(1);
                st.completed_pages.push_back((global_page_id, words));
            });
        });
    }

    pub(super) fn pump(
        mut world: ResMut<NaniteWorld>,
        mut streaming: ResMut<NaniteStreaming>,
        mut state: ResMut<MonolithState>,
    ) {
        if !state.assets_loaded {
            let ready_manifests = WEB_STATE.with(|cell| {
                let mut st = cell.borrow_mut();
                if !st.manifests_started {
                    st.manifests_started = true;
                    start_manifests();
                }
                if st.manifests.iter().all(Option::is_some) {
                    let mut out = Vec::with_capacity(5);
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
                for (stem, manifest) in HERO_ASSETS.iter().zip(manifests.iter()) {
                    world.add_mesh(*stem, manifest);
                }
                populate_sanctuary(&mut world);
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

        let capacity = WEB_STATE.with(|cell| 12_usize.saturating_sub(cell.borrow().in_flight_pages));
        if capacity > 0 {
            let reqs = streaming.take_fetch_requests(capacity);
            let count = reqs.len();
            if count > 0 {
                WEB_STATE.with(|cell| {
                    cell.borrow_mut().in_flight_pages += count;
                });
                for req in reqs {
                    start_page_fetch(req.mesh_name, req.global_page_id, req.byte_offset, req.byte_len);
                }
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn load_sanctuary_assets(
    world: ResMut<NaniteWorld>,
    streaming: ResMut<brick_nanite::NaniteStreaming>,
    state: ResMut<MonolithState>,
) {
    web_loader::pump(world, streaming, state);
}

struct GroundGpu {
    pipeline: Pipeline<Self, Scene>,
    verts: Buffer<Self>,
    mesh: Option<Vec<GroundVertex>>,
    vert_count: u32,
}

impl GroundGpu {
    fn draw_ground(&self, draw: &mut Draw<'_, Self, Scene>) {
        if self.vert_count > 0 {
            draw.draw(
                &self.pipeline,
                Shape::Pulled {
                    vertices: 0..self.vert_count,
                    instances: 0..1,
                    key: None,
                },
            );
        }
    }
}

impl GpuPart for GroundGpu {
    type Reads = ();
    const NAME: &'static str = "monolith_ground";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::sanctuary::ground;
        let verts = cx.buffer("monolith_ground_verts", Lifetime::Kept);
        let targets = [
            (None, wgpu::ColorWrites::ALL),
            (None, wgpu::ColorWrites::ALL),
        ];
        let pipeline = cx.render_pipeline(
            Scene::named(NANITE_P1_PASS),
            &RenderPipeline {
                vertex: ground::VS_GROUND,
                fragment: Some(ground::FS_GROUND),
                vertices: &[],
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..wgpu::PrimitiveState::default()
                },
                depth: DepthUse::Compare {
                    compare: wgpu::CompareFunction::GreaterEqual,
                    write: true,
                    bias: wgpu::DepthBiasState::default(),
                },
                stencil: wgpu::StencilState::default(),
                samples: 1,
                targets: &targets,
                constants: &[],
                per_batch: PerBatch::NONE,
            },
            &[(ground::GROUND_VERTS, Bind::Buffer(verts))],
        )?;
        let mesh = build_ground_mesh();
        let vert_count = mesh.len() as u32;
        Ok(Self {
            pipeline,
            verts,
            mesh: Some(mesh),
            vert_count,
        })
    }

    fn prepare(&mut self, (): (), cx: &mut Prepare<'_, Self>) {
        if let Some(mesh) = self.mesh.as_ref() {
            let bytes = (mesh.len().max(1) * size_of::<GroundVertex>()) as u64;
            if cx.allocate_buffer(self.verts, bytes) == Allocation::Fresh && !mesh.is_empty() {
                cx.write(self.verts, 0, mesh);
            }
        }
    }
}
