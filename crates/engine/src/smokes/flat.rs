use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession, compiled};
use engine_core::{Brick, BrickRegistry, Bricks, CpuSlot, Slot, brick_name, smoke_check};
use engine_grounding::GroundingBrick;
use engine_places::GroundSlot;
use engine_render::bindings::{self, Sample, Sampling, Stages, Table};
use engine_render::graph::{GraphTable, SCENE_CLEAR, Targets};
use engine_render::parts::{
    Create, DrawAt, GpuPart, Hook, OpaqueSlot, PartDecl, PartEntry, Step, View,
};
use engine_render::pipelines::{self, Fragment, Geometry};
use engine_render::render_bricks::AddRenderParts;
use engine_render::renderer::{Core, DEPTH_FORMAT};
use engine_render::shaders::{Entry, Stage};
use engine_render::targets::texture_2d;
use serde_json::{Map, Value, json};

use super::engine_frames::{core_only, rendered};
use super::requests::{FRAME_MS, Pixels, accepted, pixels};

const FRAMES: u32 = 8;

const UNLINKED: [&str; 4] = [
    "shader_link",
    "face_missing",
    "slot_unfilled",
    "render_failed",
];

const LIGHT: [u8; 4] = [224, 64, 32, 255];

const DARK: [u8; 4] = [32, 96, 224, 255];

const TEXELS: [[u8; 4]; 4] = [LIGHT, DARK, DARK, LIGHT];

const TOLERANCE: u8 = 8;

const SPRITE: Table = Table {
    label: "sprite/sprite-layout",
    bindings: &[
        bindings::texture(
            "checker",
            Sample::Float,
            bindings::View::D2,
            Stages::FRAGMENT,
        ),
        bindings::sampler("checker_sampler", Sampling::Filtering, Stages::FRAGMENT),
    ],
};

engine_render::pipeline_layouts! {
    SPRITE_PIPELINES = [SPRITE];
}

engine_render::shader_modules! {
    "sprite::draw" in "sprite" = "sprite.wgsl";
}

const VERTEX: Entry = Entry {
    module: "sprite::draw",
    name: "vs_sprite",
    stage: Stage::Vertex,
};

const FRAGMENT: Entry = Entry {
    module: "sprite::draw",
    name: "fs_sprite",
    stage: Stage::Fragment,
};

pub(crate) struct SpriteBrick;

impl Brick for SpriteBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .render_part(&PART)
            .shaders(MODULES)
            .pipeline_layouts(PIPELINE_LAYOUTS);
    }
}

static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "sprite",
        brick: "sprite",
        passes: &[],
        resources: &[],
        hooks: &[Hook::Opaque(OpaqueSlot::Objects)],
        contributes: &[],
        resolve: None,
        labels: &[],
        lazy: false,
        needs: &[],
        piece: None,
    },
    create,
};

fn create(cx: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(SpritePart::new(cx.core))
}

struct SpritePart {
    group: wgpu::BindGroup,
    pipeline: Option<wgpu::RenderPipeline>,
    drawn: bool,
}

impl SpritePart {
    fn new(core: &Core) -> Self {
        let device = &core.device;
        let checker = texture_2d(
            device,
            "sprite/checker",
            (2, 2),
            1,
            1,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        core.queue.write_texture(
            checker.as_image_copy(),
            TEXELS.as_flattened(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(2),
            },
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
        let view = checker.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sprite/checker"),
            ..Default::default()
        });
        let layout = pipelines::layout(device, &SPRITE);
        let group = pipelines::group(
            device,
            "sprite/checker",
            &SPRITE,
            &layout,
            &[
                ("checker", pipelines::view(&view)),
                ("checker_sampler", pipelines::sampler(&sampler)),
            ],
        );
        Self {
            group,
            pipeline: None,
            drawn: false,
        }
    }
}

fn pipeline(core: &Core) -> wgpu::RenderPipeline {
    let device = &core.device;
    let layout = pipelines::layout_of(device, &SPRITE_PIPELINES);
    let targets = Targets::Post.color_targets(core.display.format());
    pipelines::geometry(
        device,
        &Geometry {
            label: "sprite/draw",
            layout: &layout,
            vertex: VERTEX,
            vertex_constants: &[],
            buffers: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            fragment: Some(Fragment {
                entry: FRAGMENT,
                constants: &[],
                targets: &targets,
            }),
        },
    )
}

impl GpuPart for SpritePart {
    fn install(&mut self, _table: &GraphTable) {
        self.pipeline = None;
    }

    fn forget_targets(&mut self, _targets: Targets) {
        self.pipeline = None;
    }

    fn finish(&mut self, cx: &mut Step<'_>) {
        self.drawn = cx
            .frame
            .shape
            .is_some_and(|shape| shape.targets == Targets::Post);
        if self.drawn && self.pipeline.is_none() {
            self.pipeline = Some(pipeline(cx.core));
        }
    }

    fn draw(&self, _at: DrawAt<'_>, _cx: &View<'_>, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        let (true, Some(pipeline)) = (self.drawn, &self.pipeline) else {
            return 0;
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.draw(0..6, 0..1);
        1
    }
}

pub(crate) fn flat_sprites(session: &mut dyn SmokeSession) -> SmokeResult {
    let base = session.bricks();
    let found = rendered(session);
    smoke_check!(
        !found.bricks.is_empty(),
        "render_bricks_listed",
        json!(found.bricks),
        json!("the render bricks of the list")
    );
    let left_out = grounds(session);
    session.restart(
        core_only(&base, &found)
            .without(left_out)
            .with::<SpriteBrick>(),
    );
    let rendering = presented(session)?;
    groundless(session)?;
    linked(&rendering)?;
    compiled(session)?;
    let reply = accepted(session, "capture", json!({}), "captures_sprite")?;
    if let Some(image) = pixels(&reply)? {
        shown(&image)?;
    }
    Ok(())
}

fn presented(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    let pause = json!({ "name": "pause", "args": {} });
    let played = json!({ "name": "frames", "args": { "count": FRAMES, "interval_ms": FRAME_MS } });
    let reply = session
        .request("control", pause)
        .and_then(|_| session.request("control", played));
    let refused = reply
        .as_ref()
        .err()
        .map(|refused| refused["error"].clone())
        .unwrap_or_default();
    smoke_check!(
        !UNLINKED.iter().any(|code| refused["code"] == *code),
        "sprite_links_alone",
        refused,
        json!({ "none_of": UNLINKED })
    );
    smoke_check!(
        reply.is_ok(),
        "frames_present",
        refused,
        json!({ "frames": FRAMES })
    );
    Ok(reply
        .map(|answered| answered["result"]["rendering"].clone())
        .unwrap_or_default())
}

fn grounds(session: &mut dyn SmokeSession) -> Vec<&'static str> {
    let filler = session
        .app()
        .world()
        .get_resource::<BrickRegistry>()
        .and_then(|registry| registry.filled(GroundSlot::NAME))
        .map(|(brick, _)| brick);
    filler
        .into_iter()
        .chain([brick_name::<GroundingBrick>()])
        .collect()
}

fn groundless(session: &mut dyn SmokeSession) -> SmokeResult {
    let world = session.app().world();
    let registry = world.get_resource::<BrickRegistry>();
    let ground = registry
        .and_then(|registry| registry.filled(GroundSlot::NAME))
        .map(|(brick, _)| brick);
    let grounding = brick_name::<GroundingBrick>();
    let grounded = registry.is_some_and(|registry| registry.names().any(|name| name == grounding));
    let held = world.contains_resource::<<GroundSlot as CpuSlot>::Cpu>();
    smoke_check!(
        ground.is_none() && !grounded && !held,
        "installs_no_ground",
        json!({ "ground": ground, "grounding": grounded, "resource": held }),
        json!({ "ground": null, "grounding": false, "resource": false })
    );
    Ok(())
}

fn members(lists: &Value) -> usize {
    lists
        .as_object()
        .into_iter()
        .flat_map(Map::values)
        .filter_map(Value::as_array)
        .map(Vec::len)
        .sum()
}

fn linked(rendering: &Value) -> SmokeResult {
    let linked = &rendering["linked"];
    let lists = &linked["lists"];
    smoke_check!(
        linked["faces"].as_object().is_some_and(Map::is_empty)
            && linked["nobody"].as_array().is_some_and(Vec::is_empty)
            && lists.is_object()
            && members(lists) == 0,
        "links_no_face",
        linked,
        json!({ "faces": {}, "nobody": [], "lists": "no members" })
    );
    Ok(())
}

fn shown(image: &Pixels) -> SmokeResult {
    let center = color_at(image, image.width / 2, image.height / 2);
    smoke_check!(
        [LIGHT, DARK]
            .iter()
            .any(|texel| near(center, [texel[0], texel[1], texel[2]])),
        "sprite_drawn_in_the_center",
        json!(center),
        json!({ "one_of": [LIGHT, DARK] })
    );
    let corner = color_at(image, 0, 0);
    let clear = cleared();
    smoke_check!(
        near(corner, clear),
        "clear_color_in_the_corner",
        json!(corner),
        json!(clear)
    );
    Ok(())
}

fn color_at(image: &Pixels, x: u32, y: u32) -> [u8; 3] {
    let at = (y as usize * image.width as usize + x as usize) * 4;
    image
        .rgba
        .get(at..at + 3)
        .map_or([0; 3], |rgb| [rgb[0], rgb[1], rgb[2]])
}

fn near(got: [u8; 3], expected: [u8; 3]) -> bool {
    got.iter()
        .zip(expected)
        .all(|(&got, expected)| got.abs_diff(expected) <= TOLERANCE)
}

fn cleared() -> [u8; 3] {
    let [red, green, blue, _] = SCENE_CLEAR;
    [red, green, blue].map(encoded)
}

fn encoded(linear: f64) -> u8 {
    let srgb = if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}
