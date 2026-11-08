use bevy_ecs::prelude::*;
use engine_core::glam::{Mat4, UVec2, Vec4};
use engine_core::math::{clamp, half, max, pow};
use engine_core::{
    AttachPoint, DrawKind, DrawnTable, Error, ErrorCode, GpuBits, Interest, Joined, Operation,
    PassMask, PassViews, Readable, Readback, ScenePasses, ViewFacts, Views, WholeFrames,
};
use engine_render::debug::DebugRecord;
use engine_render::{
    AfterRead, Allocation, Bind, Buffer, ComputePipeline, Copied, CopyRegion, Count, DEBUG_TARGETS,
    DebugModes, Encode, GpuPart, Install, Lifetime, Pass, Prepare, TargetName, wgpu,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::gpu::{PROBED_COUNT, PROBED_IDS, PROBED_WORDS};
use crate::modes::WHOLE;
use crate::shaders::debug_view::probe;

const IDS: u64 = PROBED_IDS as u64 * 4;
const COUNT: u64 = PROBED_COUNT as u64 * 4;
const SCENE: u64 = 512;
const CANVAS: u64 = 768;
const WORDS: u64 = PROBED_WORDS as u64 * 4;
const DEPTH: u64 = 1280;
const CANVAS_ORDER: u64 = 1284;
const PROBED_BYTES: u64 = 1288;
const IN_ORDER: u32 = 1;
const BLUE_FIRST: u32 = 2;
const ROW_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum Stage {
    #[default]
    Idle,
    Armed,
    Capturing,
    Waiting,
    Read(Vec<u8>),
}

#[derive(Resource, Default)]
pub(crate) struct Probe {
    stage: Stage,
    armed: u32,
    pixel: (u32, u32),
    armed_size: (u32, u32),
    depth_row: u32,
    size: (u32, u32),
    view_projection: Mat4,
    whole: Option<Interest<&'static str>>,
}

#[derive(Resource, Default)]
pub(crate) struct ProbeSteps {
    done: Vec<u32>,
}

impl Readable for ProbeSteps {
    fn arrived(&mut self, armed: u32, _: Readback<&[u8]>) {
        self.done.push(armed);
    }
}

impl Probe {
    pub(crate) fn capturing(&self) -> bool {
        self.stage == Stage::Capturing
    }

    pub(crate) fn pixel(&self) -> [u32; 4] {
        [self.pixel.0, self.pixel.1, 0, 0]
    }

    pub(crate) fn probing(&self) -> bool {
        match self.stage {
            Stage::Armed | Stage::Capturing | Stage::Waiting => true,
            Stage::Idle | Stage::Read(_) => false,
        }
    }

    fn read(&self) -> Option<&[u8]> {
        match &self.stage {
            Stage::Read(bytes) => Some(bytes),
            Stage::Idle | Stage::Armed | Stage::Capturing | Stage::Waiting => None,
        }
    }

    fn result(
        &self,
        bytes: &[u8],
        modes: &Joined<'_, DebugModes>,
        (table, passes): (Option<&DrawnTable>, Option<&ScenePasses>),
    ) -> Value {
        let record = word(bytes, IDS);
        let decoded = DebugRecord::unpack(record);
        let kind = decoded
            .zip(table)
            .and_then(|(decoded, table)| table.kind_of(decoded.kind));
        let object = word(bytes, IDS + 4);
        let detail = word(bytes, IDS + 8);
        let depth = f32::from_bits(word(bytes, DEPTH));
        let fragments = half(word(bytes, COUNT) as u16);
        let (width, height) = (self.size.0.max(1) as f32, self.size.1.max(1) as f32);
        let clip = Vec4::new(
            (self.pixel.0 as f32 + 0.5) / width * 2.0 - 1.0,
            1.0 - (self.pixel.1 as f32 + 0.5) / height * 2.0,
            depth,
            1.0,
        );
        let point = self.view_projection.inverse() * clip;
        let color = |offset: u64, blue_first: bool| {
            let texel = word(bytes, offset).to_le_bytes();
            if blue_first {
                [texel[2], texel[1], texel[0]]
            } else {
                [texel[0], texel[1], texel[2]]
            }
        };
        let canvas = word(bytes, CANVAS_ORDER);
        let drawn = kind.is_some();
        let table = table.filter(|_| matches!(kind, Some(DrawKind::MESH | DrawKind::SKIN)));
        let entity = table.and_then(|table| table.owner(object));
        let membership = table
            .and_then(|table| table.records().get(object as usize))
            .zip(passes)
            .map(|(record, passes)| passes.decode(PassMask(record.passes)));
        let probe = PixelProbe {
            pixel: [self.pixel.0, self.pixel.1],
            kind: kind.map(DrawKind::name),
            card: decoded.is_some_and(|decoded| decoded.card),
            level: decoded.map_or(0, |decoded| decoded.level),
            fading: decoded.is_some_and(|decoded| decoded.fading),
            passes: membership,
            object: drawn.then_some(object),
            detail: drawn.then_some(detail),
            normal: drawn.then(|| unpacked_normal(word(bytes, IDS + 12))),
            fragments: max(fragments.round(), 0.0) as u32,
            depth,
            world: (depth > 0.0 && point.w != 0.0).then(|| (point.truncate() / point.w).to_array()),
            shaded: Some(scene_texel(bytes, SCENE)),
            displayed: (canvas != 0).then(|| color(CANVAS, canvas == BLUE_FIRST)),
        };
        let Ok(Value::Object(mut fields)) = serde_json::to_value(probe) else {
            return Value::Null;
        };
        let ids = [record, object, detail, word(bytes, WORDS)];
        let words = [
            f32::from_bits(word(bytes, WORDS + 4)),
            fragments,
            depth,
            0.0,
        ];
        for mode in modes.all() {
            if let Value::Object(decoded) = (mode.decode)(kind, ids, words) {
                for (name, value) in decoded {
                    fields.entry(name).or_insert(value);
                }
            }
        }
        if let Some(entity) = entity {
            fields.insert(String::from("entity"), Value::from(entity.to_bits()));
        }
        Value::Object(fields)
    }
}

impl Readable for Probe {
    fn arrived(&mut self, armed: u32, read: Readback<&[u8]>) {
        if armed == self.armed && matches!(self.stage, Stage::Capturing | Stage::Waiting) {
            self.stage = Stage::Read(read.value.to_vec());
            self.whole = None;
        }
    }
}

#[derive(Debug, Serialize)]
struct PixelProbe {
    pixel: [u32; 2],
    kind: Option<&'static str>,
    card: bool,
    level: u32,
    fading: bool,
    passes: Option<Value>,
    object: Option<u32>,
    detail: Option<u32>,
    normal: Option<[f32; 3]>,
    fragments: u32,
    depth: f32,
    world: Option<[f32; 3]>,
    shaded: Option<[u8; 3]>,
    displayed: Option<[u8; 3]>,
}

fn word(bytes: &[u8], offset: u64) -> u32 {
    let start = offset as usize;
    bytes.get(start..start + 4).map_or(0, |word| {
        u32::from_le_bytes([word[0], word[1], word[2], word[3]])
    })
}

fn scene_texel(bytes: &[u8], offset: u64) -> [u8; 3] {
    let (first, second) = (word(bytes, offset), word(bytes, offset + 4));
    [first & 0xffff, first >> 16, second & 0xffff].map(|bits| encoded(half(bits as u16)))
}

fn encoded(light: f32) -> u8 {
    let light = clamp(light, 0.0, 1.0);
    let shown = if light <= 0.003_130_8 {
        light * 12.92
    } else {
        1.055 * pow(light, 1.0 / 2.4) - 0.055
    };
    (shown * 255.0).round() as u8
}

fn unpacked_normal(packed: u32) -> [f32; 3] {
    let snorm = |bits: u32| max(f32::from(bits as u16 as i16) / 32767.0, -1.0);
    let (x, y) = (snorm(packed & 0xffff), snorm(packed >> 16));
    let z = 1.0 - x.abs() - y.abs();
    let fold = max(-z, 0.0);
    let x = x - if x >= 0.0 { fold } else { -fold };
    let y = y - if y >= 0.0 { fold } else { -fold };
    let length = max((x * x + y * y + z * z).sqrt(), 1.0e-6);
    [x / length, y / length, z / length]
}

pub(crate) fn advance(views: Res<Views>, mut steps: ResMut<ProbeSteps>, mut probe: ResMut<Probe>) {
    let probe = &mut *probe;
    let done = std::mem::take(&mut steps.done);
    let next = match &probe.stage {
        Stage::Armed => Some(Stage::Capturing),
        Stage::Capturing => done.contains(&probe.armed).then_some(Stage::Waiting),
        Stage::Idle | Stage::Waiting | Stage::Read(_) => None,
    };
    if next == Some(Stage::Capturing)
        && let Some(main) = views.main()
    {
        let size = main.size();
        probe.size = (size.x, size.y);
        probe.view_projection = main.view_projection;
    }
    if let Some(next) = next {
        probe.stage = next;
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PixelRequest {
    x: u32,
    y: u32,
}

pub(crate) struct ProbePixel;

impl Operation for ProbePixel {
    const NAME: &'static str = "probe_pixel";
    type Params = (
        Res<'static, Views>,
        ResMut<'static, Probe>,
        ResMut<'static, WholeFrames>,
    );
    type Request = PixelRequest;

    fn run(
        (views, mut probe, mut frames): SystemParamItem<'_, '_, Self::Params>,
        request: PixelRequest,
    ) -> Result<Value, Error> {
        let size = views.main().map_or(UVec2::ZERO, ViewFacts::size);
        for (path, value, size) in [("$.x", request.x, size.x), ("$.y", request.y, size.y)] {
            if value >= size {
                return Err(Error::new(ErrorCode::OutOfRange, path)
                    .range(0.0, f64::from(size.saturating_sub(1)))
                    .got(value));
            }
        }
        let armed = probe.armed.wrapping_add(1);
        *probe = Probe {
            stage: Stage::Armed,
            armed,
            pixel: (request.x, request.y),
            armed_size: (size.x, size.y),
            depth_row: (size.x * 4).next_multiple_of(ROW_ALIGNMENT),
            whole: Some(frames.ask(WHOLE)),
            ..Probe::default()
        };
        Ok(json!({ "pixel": [request.x, request.y] }))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProbedRequest {}

pub(crate) struct ProbedPixel;

impl Operation for ProbedPixel {
    const NAME: &'static str = "probed_pixel";
    type Params = (
        Res<'static, Probe>,
        Joined<'static, DebugModes>,
        (Option<Res<'static, DrawnTable>>, Option<Res<'static, ScenePasses>>),
    );
    type Request = ProbedRequest;

    fn run(
        (probe, modes, (table, passes)): SystemParamItem<'_, '_, Self::Params>,
        _: ProbedRequest,
    ) -> Result<Value, Error> {
        let result = probe
            .read()
            .map(|bytes| probe.result(bytes, &modes, (table.as_deref(), passes.as_deref())));
        Ok(json!({ "probing": probe.probing(), "probe": result }))
    }
}

fn texel(at: wgpu::Origin3d, offset: u64) -> CopyRegion {
    CopyRegion::ToBuffer {
        from: at,
        to: wgpu::TexelCopyBufferLayout {
            offset,
            bytes_per_row: None,
            rows_per_image: None,
        },
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    }
}

pub(crate) struct ProbeGpu {
    probed: Buffer<Self>,
    rows: Buffer<Self>,
    pass: Pass<Self>,
    reading: ComputePipeline<Self>,
    capturing: bool,
}

impl ProbeGpu {
    fn capture(
        &self,
        probe: &Probe,
        (size, at): (UVec2, wgpu::Origin3d),
        cx: &mut Prepare<'_, Self>,
    ) {
        if (size.x, size.y) == probe.armed_size {
            let (width, height) = probe.armed_size;
            let row = u64::from(probe.depth_row);
            let (Allocation::Fresh | Allocation::Kept) =
                cx.allocate_buffer(self.rows, row * u64::from(height));
            cx.copy_at_end(
                Copied::Target(TargetName::DEPTH),
                Copied::Buffer(self.rows),
                &[CopyRegion::ToBuffer {
                    from: wgpu::Origin3d::ZERO,
                    to: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(probe.depth_row),
                        rows_per_image: Some(height),
                    },
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                }],
            );
            cx.copy_at_end(
                Copied::Buffer(self.rows),
                Copied::Buffer(self.probed),
                &[CopyRegion::Bytes {
                    from: u64::from(at.y) * row + u64::from(at.x) * 4,
                    to: DEPTH,
                    size: 4,
                }],
            );
        } else {
            cx.write(self.probed, DEPTH, &[0_u32]);
        }
        cx.copy_at_end(
            Copied::Target(TargetName::COLOR),
            Copied::Buffer(self.probed),
            &[texel(at, SCENE)],
        );
        let order = cx.canvas().map_or(0, |format| {
            if matches!(
                format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
            ) {
                BLUE_FIRST
            } else {
                IN_ORDER
            }
        });
        if order != 0 {
            cx.copy_at_end(
                Copied::Canvas,
                Copied::Buffer(self.probed),
                &[texel(at, CANVAS)],
            );
        }
        cx.write(self.probed, CANVAS_ORDER, &[order]);
        let range = 0..PROBED_BYTES;
        if cx.read_back::<Probe>(probe.armed, self.probed, range, AfterRead::Keep) {
            cx.deliver::<ProbeSteps>(probe.armed, Vec::new());
        }
    }
}

impl GpuPart for ProbeGpu {
    type Reads = (Res<'static, Probe>, Res<'static, Views>);
    const NAME: &'static str = "debug_view";
    const LAZY: bool = true;
    const LABELS: &'static [&'static str] = &["engine/debug-view"];

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let probed = cx.buffer("debug_view_probe", Lifetime::Kept);
        let pass = cx
            .pass("debug_view_probe")
            .at(AttachPoint::Post)
            .views(PassViews::Main)
            .span("debug_view")
            .id();
        let reading = cx.compute_pipeline(
            pass,
            probe::CS_DEBUG_PROBE,
            &[
                (probe::PROBED, Bind::Buffer(probed)),
                (probe::DEBUG_IDS, Bind::Target(DEBUG_TARGETS[0])),
                (probe::DEBUG_COUNT, Bind::Target(DEBUG_TARGETS[1])),
                (probe::DEBUG_WORDS, Bind::Target(DEBUG_TARGETS[2])),
            ],
        )?;
        Ok(Self {
            probed,
            rows: cx.buffer("debug_view_probe_depth", Lifetime::Kept),
            pass,
            reading,
            capturing: false,
        })
    }

    fn prepare(
        &mut self,
        (probe, views): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let size = views.main().map_or(UVec2::ZERO, ViewFacts::size);
        let at = wgpu::Origin3d {
            x: probe.pixel.0.min(size.x.saturating_sub(1)),
            y: probe.pixel.1.min(size.y.saturating_sub(1)),
            z: 0,
        };
        self.capturing = probe.capturing();
        match probe.stage {
            Stage::Capturing => {
                let (Allocation::Fresh | Allocation::Kept) =
                    cx.allocate_buffer(self.probed, PROBED_BYTES);
                self.capture(&probe, (size, at), cx);
            }
            Stage::Waiting => cx.release_buffer(self.rows),
            Stage::Idle | Stage::Armed | Stage::Read(_) => {}
        }
    }

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        if self.capturing && view.target.surface() && cx.view_index() == 0 {
            cx.run(self.pass);
        }
    }

    fn encode(&mut self, _: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        cx.dispatch(&self.reading, Count::Groups([1, 1, 1]));
        0
    }
}
