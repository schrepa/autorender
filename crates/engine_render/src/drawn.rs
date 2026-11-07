use bevy_ecs::system::{Res, SystemParamItem};
use engine_core::{
    Cursor, DeviceLimits, DrawnRecord, DrawnTable, Entry, Error, ErrorCode, GpuStruct, Layers,
    PassViews, ScenePasses, Since, Var, ViewFacts, Views, gpu_struct,
};

use crate::shaders::core_module;
use crate::{
    Allocation, Bind, Buffer, ComputePipeline, Count, Encode, GpuPart, Install, Lifetime, Pass,
    Prepare,
};

pub(crate) const DRAWN_MODULE: &str = "engine::drawn";
pub(crate) const DRAWN_CULL_MODULE: &str = "engine::drawn_cull";
pub(crate) const DRAWN_FADES_MODULE: &str = "engine::drawn_fades";
const CULL_GROUP: u32 = 64;
const SLOT_BITS: u32 = u32::BITS;
const FIRST_SLOTS: u64 = 64;
const VIEW_BYTES: u64 = 16;

const VARS: &str = "@group(0) @binding(0) var<storage, read> drawn_records: array<DrawnRecord>;
@group(0) @binding(0) var<storage, read> drawn_values: array<vec4<f32>>;
@group(0) @binding(0) var<storage, read> drawn_culled: array<u32>;
@group(0) @binding(0) var<uniform> drawn_view: vec4<u32>;
@group(0) @binding(0) var<storage, read> drawn_fades: array<u32>;

fn drawn_value(record: DrawnRecord, index: u32) -> f32 {
    if index >= record.floats {
        return 0.0;
    }
    return drawn_values[drawn_value_vec4(record, index)][index % 4u];
}

fn drawn_in_view(view: u32, slot: u32) -> bool {
    let words = drawn_culled[0];
    let word = slot / 32u;
    if word >= words {
        return false;
    }
    return (drawn_culled[1u + view * words + word] & (1u << (slot % 32u))) != 0u;
}

fn drawn_shown(slot: u32) -> bool {
    return drawn_in_view(drawn_view.x, slot);
}

fn drawn_fade(slot: u32) -> u32 {
    if slot >= arrayLength(&drawn_fades) {
        return 0u;
    }
    return drawn_fades[slot];
}
";

const FADES: &str = "@group(0) @binding(0) var<storage, read_write> drawn_fade_words: array<u32>;

fn set_drawn_fade(slot: u32, fade: u32) {
    if slot < arrayLength(&drawn_fade_words) {
        drawn_fade_words[slot] = fade;
    }
}
";

const CULL_IMPORTS: &str = "import package::engine::drawn::drawn_records;
";

const CULL: &str = "
@group(2) @binding(0) var<storage, read> cull_views: array<CullView>;
@group(2) @binding(1) var<storage, read_write> cull_bits: array<u32>;

@compute @workgroup_size(64)
fn drawn_cull(@builtin(global_invocation_id) id: vec3<u32>) {
    let words = cull_bits[0];
    let word = id.x;
    let view = id.y;
    if word >= words || view >= arrayLength(&cull_views) {
        return;
    }
    let wanted = cull_views[view];
    let total = arrayLength(&drawn_records);
    let first = word * 32u;
    let count = min(32u, total - min(first, total));
    var bits = 0u;
    for (var lane = 0u; lane < count; lane += 1u) {
        let record = drawn_records[first + lane];
        let member = (record.passes & wanted.passes) != 0u
            && (record.layers & wanted.layers) != 0u
            && (wanted.takes == 0u || record.mobility == wanted.takes);
        if member {
            bits |= 1u << lane;
        }
    }
    cull_bits[1u + view * words + word] = bits;
}
";

gpu_struct! {
    pub struct CullView {
        pub passes: u32,
        pub layers: u32,
        pub takes: u32,
        pub pad: u32,
    }
}

pub(crate) fn source() -> String {
    let mut source = engine_core::drawn_wgsl();
    source.push_str(VARS);
    source
}

pub(crate) fn fades_source() -> String {
    String::from(FADES)
}

pub(crate) fn cull_source() -> String {
    let mut source = String::from(CULL_IMPORTS);
    source.push_str(CullView::WGSL);
    source.push_str(CULL);
    source
}

pub(crate) struct DrawnGpu {
    records: Buffer<Self>,
    values: Buffer<Self>,
    fades: Buffer<Self>,
    culled: Buffer<Self>,
    views: Buffer<Self>,
    view: Buffer<Self>,
    cull: Pass<Self>,
    pipeline: ComputePipeline<Self>,
    seen: Cursor,
    stride: usize,
    groups: [u32; 3],
}

fn var(module: &str, name: &'static str) -> Var {
    Var::generated(core_module(module), name)
}

fn view_masks(views: &Views, passes: Option<&ScenePasses>) -> Vec<CullView> {
    let listed = passes.map_or(&[][..], ScenePasses::listed);
    let main = listed
        .iter()
        .filter(|scene| scene.pass.views != PassViews::Owned)
        .filter_map(|scene| scene.bit)
        .fold(0, |mask, bit| mask | bit.mask().word());
    let mut masks: Vec<CullView> = views
        .drawn()
        .iter()
        .map(|view| CullView {
            passes: if view.draws { view.passes.word() } else { 0 },
            layers: view.layers.0,
            takes: view.takes.map_or(0, |mobility| mobility as u32 + 1),
            pad: 0,
        })
        .collect();
    if masks.is_empty() {
        masks.push(CullView {
            passes: main,
            layers: Layers::ALL.0,
            takes: 0,
            pad: 0,
        });
    }
    masks
}

impl DrawnGpu {
    fn upload(&mut self, table: &DrawnTable, cx: &mut Prepare<'_, Self>) {
        let records = table.records();
        let stride = table.value_vec4s();
        let slots = (records.len() as u64).next_power_of_two().max(FIRST_SLOTS);
        let fresh = matches!(
            cx.allocate_buffer(self.records, slots * size_of::<DrawnRecord>() as u64),
            Allocation::Fresh
        ) | matches!(
            cx.allocate_buffer(self.values, slots * (stride.max(1) * 16) as u64),
            Allocation::Fresh
        ) | (stride != self.stride);
        self.stride = stride;
        let fades = slots * 4;
        cx.allocate_buffer(self.fades, fades);
        cx.clear_buffer_range(self.fades, 0..fades);
        let changes = table.changes_since(&mut self.seen);
        if records.is_empty() {
            return;
        }
        let range = match (fresh, changes) {
            (true, _) | (false, Since::All) => 0..records.len(),
            (false, Since::Listed(changed)) => {
                let (low, high) = changed.iter().fold((usize::MAX, 0), |(low, high), &slot| {
                    (low.min(slot as usize), high.max(slot as usize + 1))
                });
                low..high.min(records.len())
            }
            (false, Since::Same) => return,
        };
        if range.is_empty() {
            return;
        }
        cx.write(
            self.records,
            (range.start * size_of::<DrawnRecord>()) as u64,
            &records[range.clone()],
        );
        if stride > 0 {
            cx.write(
                self.values,
                (range.start * stride * 16) as u64,
                &table.values()[range.start * stride..range.end * stride],
            );
        }
    }
}

impl GpuPart for DrawnGpu {
    type Reads = (
        Res<'static, DrawnTable>,
        Res<'static, Views>,
        Option<Res<'static, ScenePasses>>,
        Res<'static, DeviceLimits>,
    );
    const NAME: &'static str = "drawn";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let records = cx.buffer("drawn_records", Lifetime::Kept);
        let values = cx.buffer("drawn_values", Lifetime::Kept);
        let fades = cx.buffer("drawn_fades", Lifetime::Kept);
        let culled = cx.buffer("drawn_culled", Lifetime::Kept);
        let views = cx.buffer("drawn_views", Lifetime::Kept);
        let view = cx.view_buffer("drawn_view", Lifetime::Kept);
        cx.frame(var(DRAWN_MODULE, "drawn_records"), Bind::Buffer(records))?;
        cx.frame(var(DRAWN_MODULE, "drawn_values"), Bind::Buffer(values))?;
        cx.frame(var(DRAWN_MODULE, "drawn_culled"), Bind::Buffer(culled))?;
        cx.frame(var(DRAWN_MODULE, "drawn_view"), Bind::Buffer(view))?;
        cx.frame(var(DRAWN_MODULE, "drawn_fades"), Bind::Buffer(fades))?;
        cx.frame(var(DRAWN_FADES_MODULE, "drawn_fade_words"), Bind::Buffer(fades))?;
        let cull = cx.pass("drawn_cull").label("engine/drawn-cull").id();
        let pipeline = cx.compute_pipeline(
            cull,
            Entry::generated(core_module(DRAWN_CULL_MODULE), "drawn_cull"),
            &[
                (var(DRAWN_CULL_MODULE, "cull_views"), Bind::Buffer(views)),
                (var(DRAWN_CULL_MODULE, "cull_bits"), Bind::Buffer(culled)),
            ],
        )?;
        Ok(Self {
            records,
            values,
            fades,
            culled,
            views,
            view,
            cull,
            pipeline,
            seen: Cursor::default(),
            stride: 0,
            groups: [0; 3],
        })
    }

    fn prepare(
        &mut self,
        (table, views, passes, limits): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.upload(&table, cx);
        let masks = view_masks(&views, passes.as_deref());
        let words = (table.records().len() as u32).div_ceil(SLOT_BITS).max(1);
        let bytes = (1 + u64::from(words) * masks.len() as u64) * 4;
        if limits.storage_room != 0 && bytes > limits.storage_room {
            cx.refuse(
                Error::new(ErrorCode::BindingLimit, "$.drawn.views")
                    .at_most(limits.storage_room)
                    .got(bytes),
            );
            return;
        }
        cx.allocate_buffer(self.views, (masks.len() * size_of::<CullView>()) as u64);
        cx.write(self.views, 0, &masks);
        cx.allocate_buffer(self.culled, bytes.next_power_of_two());
        cx.write(self.culled, 0, &[words]);
        self.groups = [words.div_ceil(CULL_GROUP), masks.len() as u32, 1];
        cx.run(self.cull);
    }

    fn view(&mut self, _: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        let index = cx.view_index();
        cx.allocate_buffer(self.view, VIEW_BYTES);
        cx.write(self.view, 0, &[index, 0, 0, 0]);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass != self.cull {
            return 0;
        }
        cx.dispatch(&self.pipeline, Count::Groups(self.groups))
    }
}
