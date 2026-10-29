#![expect(
    clippy::expect_used,
    reason = "encoding reads the targets the frame's preparation sized"
)]

use crate::shaders::entry::post;
use engine_core::{RenderSnapshot, glam::Mat4};

use crate::bindings::{GroupRef, POST};
use crate::cached::{Bound, Versioned};
use crate::composite::{PLAIN_RESOLVE, composite_entry, composites, pixel_effects};
use crate::gpu_data::PostUniform;
use crate::graph::{
    Access, BLACK, Clear, FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl,
    ResourceId, Stage,
};
use crate::lazy::Lazy;
use crate::parts::{
    Begin, BeginOut, CORE_BRICK, Create, Encode, GpuPart, PartDecl, PartEntry, Prelude, Step,
    View,
};
use crate::pipeline_layouts::{POST_PIPELINES, PipelineLayout};
use crate::pipelines::{self, Fullscreen, fullscreen};
use crate::profiling::Span;
use crate::renderer::Core;
use crate::screen_effects::{Effects, targets_read};
use crate::targets::{FrameRegion, FrameSize, ScreenTarget};

pub const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
pub const GLOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;
pub const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Uint;
pub const OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
pub const REFLECTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
pub const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
pub const DEBUG_IDS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
pub const DEBUG_COUNT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
pub const DEBUG_TERRAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
const SCENE_LEVELS: u32 = 6;

const COMPOSITED: [&str; 5] = ["scene", "glow", "occlusion", "reflections", "surface"];
const MOTION: [(&str, &str, wgpu::TextureFormat); 1] =
    [("motion", "engine/post-motion", MOTION_FORMAT)];
const DEBUG: [(&str, &str, wgpu::TextureFormat); 3] = [
    ("debug_ids", "engine/post-debug-ids", DEBUG_IDS_FORMAT),
    ("debug_count", "engine/post-debug-count", DEBUG_COUNT_FORMAT),
    ("debug_terrain", "engine/post-debug-terrain", DEBUG_TERRAIN_FORMAT),
];

pub fn debug_color_targets() -> Vec<Option<wgpu::ColorTargetState>> {
    let add = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    vec![
        Some(DEBUG_IDS_FORMAT.into()),
        Some(wgpu::ColorTargetState {
            format: DEBUG_COUNT_FORMAT,
            blend: Some(wgpu::BlendState {
                color: add,
                alpha: add,
            }),
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(DEBUG_TERRAIN_FORMAT.into()),
    ]
}

const fn per_frame(name: &'static str) -> ResourceDecl {
    ResourceDecl {
        name,
        lifetime: Lifetime::Frame,
    }
}

pub static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "post",
        brick: CORE_BRICK,
        passes: &[PassDecl {
            name: "composite",
            stage: Stage::Composite,
        }],
        resources: &[
            per_frame("scene"),
            per_frame("glow"),
            per_frame("surface"),
            per_frame("occlusion"),
            per_frame("reflections"),
            per_frame("motion"),
            per_frame("debug_ids"),
            per_frame("debug_count"),
            per_frame("debug_terrain"),
        ],
        hooks: &[],
        contributes: &[],
        effects: &[],
        labels: &["engine/post-", "engine/post."],
        lazy: true,
        faces: &[],
        needs: &[],
        piece: None,
    },
    create,
};

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(PostPass::default())
}

#[derive(Clone, Copy, Default)]
struct Ids {
    composite_pass: Option<PassId>,
    canvas: Option<ResourceId>,
    scene: Option<ResourceId>,
    occlusion: Option<ResourceId>,
    reflections: Option<ResourceId>,
    haze: Option<ResourceId>,
    bloom: Option<ResourceId>,
}

#[derive(Default)]
pub struct PostPass {
    post: Lazy<Post>,
    frame: Option<Effects>,
    motion: ScreenTarget<Made<1>>,
    debug: ScreenTarget<Made<3>>,
    ids: Ids,
    linked: bool,
}

impl PostPass {
    pub fn active(&self) -> bool {
        self.frame.is_some()
    }

    pub fn scene(&self) -> Option<Bound<'_>> {
        self.target("scene")
    }

    pub fn scene_texture(&self) -> Option<&wgpu::Texture> {
        self.post.get()?.targets.get()?.made.textures.first()
    }

    pub fn target(&self, resource: &str) -> Option<Bound<'_>> {
        let (generation, view) = self.found(resource)?;
        Some(Bound::counted(generation, pipelines::view(view)))
    }

    pub fn target_view(&self, resource: &str) -> Option<&wgpu::TextureView> {
        Some(self.found(resource)?.1)
    }

    pub fn resolve_inputs(&self) -> Option<([Bound<'_>; 3], Bound<'_>)> {
        let post = self.post.get()?;
        Some((
            [
                self.target("scene")?,
                self.target("occlusion")?,
                self.target("reflections")?,
            ],
            Bound::fixed(post.uniform.as_entire_binding()),
        ))
    }

    #[cfg(feature = "dev")]
    pub fn debug_textures(&self) -> Option<[&wgpu::Texture; 3]> {
        Some(self.debug.get()?.textures.each_ref())
    }

    fn found(&self, resource: &str) -> Option<(u64, &wgpu::TextureView)> {
        fn within<'a, const N: usize>(
            made: Option<&'a Versioned<Made<N>>>,
            resource: &str,
        ) -> Option<(u64, &'a wgpu::TextureView)> {
            let made = made?;
            Some((made.generation(), made.view(resource)?))
        }
        within(self.motion.get(), resource)
            .or_else(|| within(self.debug.get(), resource))
            .or_else(|| {
                let targets = self.post.get()?.targets.get()?;
                Some((targets.generation(), targets.made.view(resource)?))
            })
    }
}

pub(crate) fn shared_groups<'c>(
    core: &'c Core,
    layout: &PipelineLayout,
) -> Vec<(u32, &'c wgpu::BindGroup)> {
    layout
        .groups
        .iter()
        .zip(0..)
        .filter_map(|(group, index)| match *group {
            GroupRef::Shared { group, without } => Some((index, core.shared_group(group, without))),
            GroupRef::Own(_) => None,
        })
        .collect()
}

fn composite_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let pipeline_layout = pipelines::layout_of(device, &POST_PIPELINES);
    fullscreen(
        device,
        &Fullscreen {
            label: "engine/post-composite",
            layout: &pipeline_layout,
            vertex: post::vs_fullscreen,
            fragment: composite_entry(PLAIN_RESOLVE),
            constants: &[],
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            depth: None,
        },
    )
}

fn ensure<const N: usize>(
    target: &mut ScreenTarget<Made<N>>,
    device: &wgpu::Device,
    size: FrameSize,
    (made, usage): ([(&'static str, &'static str, wgpu::TextureFormat); N], wgpu::TextureUsages),
) {
    if !target.resizes(size) {
        return;
    }
    let textures = made.map(|(_, label, format)| {
        crate::targets::texture_2d(device, label, size.pixels(), 1, 1, format, usage)
    });
    let views = textures
        .each_ref()
        .map(|texture| texture.create_view(&Default::default()));
    target.set(
        size,
        Made {
            textures,
            views,
            names: made.map(|(name, _, _)| name),
        },
    );
}

impl GpuPart for PostPass {
    fn install(&mut self, table: &GraphTable) {
        self.ids = Ids {
            composite_pass: table.pass_named("composite"),
            canvas: table.resource("canvas"),
            scene: table.resource("scene"),
            occlusion: table.resource("occlusion"),
            reflections: table.resource("reflections"),
            haze: table.resource("haze"),
            bloom: table.resource("bloom"),
        };
        self.linked = false;
    }

    fn begin(&mut self, cx: &mut Begin<'_>, _out: &mut BeginOut) {
        if self.linked {
            return;
        }
        self.linked = true;
        let device = &cx.core.device;
        if let Ok(made) = composites(&pixel_effects(cx.parts)) {
            for (path, brick, source) in made {
                pipelines::add_generated(device, path, brick, source);
            }
        }
        if let Some(post) = self.post.get_mut() {
            post.composite = composite_pipeline(device, cx.core.canvas_format());
        }
    }

    fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
        let core = &*cx.core;
        let device = &core.device;
        if cx.frame.kind.temporal {
            ensure(
                &mut self.motion,
                device,
                core.display.region.display,
                (MOTION, crate::targets::RENDER_TARGET),
            );
        }
        if cx.frame.kind.debug {
            ensure(
                &mut self.debug,
                device,
                core.display.size(),
                (
                    DEBUG,
                    crate::targets::RENDER_TARGET | wgpu::TextureUsages::COPY_SRC,
                ),
            );
            self.frame = None;
            return;
        }
        let snapshot = cx.frame.snapshot;
        let region = core.display.region;
        let needs = cx.frame.temporal.is_some()
            || targets_read(&snapshot.environment)
            || region.upscales();
        self.frame = if needs {
            let format = core.canvas_format();
            self.post
                .create(device, "engine/post-", || Post::new(device, format));
            self.post.get_mut().map(|post| {
                post.begin(device, region.display);
                Effects::of(&snapshot.environment)
            })
        } else {
            None
        };
        if !needs && let Some(post) = self.post.get_mut() {
            post.release_targets();
        }
    }

    fn finish(&mut self, cx: &mut Step<'_>) {
        let (Some(gates), Some(post)) = (self.frame, self.post.get_mut()) else {
            return;
        };
        let ids = self.ids;
        let occluded = cx.parts.produces(ids.occlusion);
        let shown = Effects {
            occlusion: gates.occlusion && occluded,
            bloom: gates.bloom && cx.parts.produces(ids.bloom),
            reflections: gates.reflections && cx.parts.produces(ids.reflections),
            haze: gates.haze && cx.parts.produces(ids.haze),
            sky: gates.sky && occluded,
        };
        post.write_uniform(
            &cx.core.queue,
            cx.frame.snapshot,
            cx.core.display.region,
            shown,
        );
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (Some(_), Some(shape)) = (self.frame, cx.frame.shape) else {
            return;
        };
        let ids = self.ids;
        if shape.targets == crate::graph::Targets::Post
            && let (Some(pass), Some(canvas)) = (ids.composite_pass, ids.canvas)
        {
            let _ = graph
                .add(cx.table, pass)
                .read(ids.scene)
                .read_written(ids.occlusion)
                .read_written(ids.reflections)
                .read_written(ids.haze)
                .read_written(ids.bloom)
                .color(canvas, Access::Clear(Clear::Color(BLACK)));
        }
    }

    fn encode(
        &mut self,
        _pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let (Some(post), Some(_)) = (self.post.get(), self.frame) else {
            return 0;
        };
        let render = cx.render_pass(
            encoder,
            "engine/post-composite",
            cx.core.profiler.span("post_composite").map(Span::render),
        );
        post.draw_composite(render, &shared_groups(cx.core, &POST_PIPELINES));
        0
    }

    fn attachment(&self, resource: &str) -> Option<wgpu::TextureView> {
        self.target_view(resource).cloned()
    }

    fn forget_targets(&mut self, targets: crate::graph::Targets) {
        if targets == crate::graph::Targets::Temporal {
            self.motion.release();
        }
    }

    fn capacity_bytes(&self) -> u64 {
        let pixels = |size: Option<FrameSize>| {
            size.map_or(0, |size| u64::from(size.width) * u64::from(size.height))
        };
        self.post.get().map_or(0, Post::capacity_bytes)
            + pixels(self.motion.size()) * 4
            + pixels(self.debug.size()) * 26
    }

    fn poll(&mut self) -> bool {
        self.post.poll()
    }

    fn disable(&mut self) -> bool {
        self.post.disable()
    }

    fn disabled(&self) -> bool {
        self.post.disabled()
    }

    #[cfg(feature = "dev")]
    fn inject(&mut self) -> bool {
        self.post.inject()
    }

    #[cfg(feature = "dev")]
    fn encode_fault(&self, encoder: &mut wgpu::CommandEncoder) {
        self.post.encode_fault(encoder);
    }
}

struct Made<const N: usize> {
    textures: [wgpu::Texture; N],
    views: [wgpu::TextureView; N],
    names: [&'static str; N],
}

impl<const N: usize> Made<N> {
    fn view(&self, resource: &str) -> Option<&wgpu::TextureView> {
        let index = self.names.iter().position(|name| *name == resource)?;
        self.views.get(index)
    }
}

impl<const N: usize> Drop for Made<N> {
    fn drop(&mut self) {
        for texture in &self.textures {
            texture.destroy();
        }
    }
}

struct Targets {
    made: Made<5>,
    composite: wgpu::BindGroup,
}

struct Post {
    layout: crate::pipelines::GroupLayout,
    composite: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    written: Option<PostUniform>,
    sampler: wgpu::Sampler,
    targets: ScreenTarget<Targets>,
}

impl Post {
    fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let uniform = crate::buffers::uniform(
            device,
            "engine/post-uniform",
            size_of::<PostUniform>() as u64,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/post-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self {
            layout: pipelines::layout(device, &POST),
            composite: composite_pipeline(device, surface_format),
            uniform,
            written: None,
            sampler,
            targets: ScreenTarget::default(),
        }
    }

    fn release_targets(&mut self) {
        self.targets.release();
    }

    fn capacity_bytes(&self) -> u64 {
        self.targets.size().map_or(0, |size| {
            let pixels = u64::from(size.width) * u64::from(size.height);
            pixels * 20 + pixels * 4 / 3
        })
    }

    fn begin(&mut self, device: &wgpu::Device, size: FrameSize) {
        if self.targets.resizes(size) {
            let targets = self.create_targets(device, size);
            self.targets.set(size, targets);
        }
    }

    fn write_uniform(
        &mut self,
        queue: &crate::gpu_calls::Queue,
        snapshot: &RenderSnapshot,
        region: FrameRegion,
        shown: Effects,
    ) {
        let size = region.display;
        let (width, height) = (region.render.width as f32, region.render.height as f32);
        let (wide, tall) = (size.width as f32, size.height as f32);
        let view_projection = snapshot.view_projection;
        let focal = view_projection.row(1).truncate().length();
        let effects = shown.uniforms(&snapshot.environment);
        let uniform = PostUniform {
            inverse_view_projection: inverse_or_identity(view_projection).to_cols_array_2d(),
            view_projection: view_projection.to_cols_array_2d(),
            eye: snapshot
                .camera_position
                .extend(focal * height * 0.5)
                .to_array(),
            size: [width, height, 1.0 / width, 1.0 / height],
            extent: [wide, tall, 1.0 / wide, 1.0 / tall],
            occlusion: effects.occlusion,
            bloom: effects.bloom,
            reflections: effects.reflections,
            haze: effects.haze,
            previous_view_projection: view_projection.to_cols_array_2d(),
            previous_eye: snapshot.camera_position.extend(0.0).to_array(),
            sky: effects.sky,
            detail: snapshot.detail_eye.extend(0.0).to_array(),
        };
        if self.written != Some(uniform) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            self.written = Some(uniform);
        }
    }

    fn draw_composite(&self, mut pass: wgpu::RenderPass<'_>, shared: &[(u32, &wgpu::BindGroup)]) {
        let targets = self.targets.get().expect("sized_in_begin");
        pass.set_pipeline(&self.composite);
        pass.set_bind_group(0, &targets.composite, &[]);
        for &(index, group) in shared {
            pass.set_bind_group(index, group, &[]);
        }
        pass.draw(0..3, 0..1);
    }

    fn create_targets(
        &self,
        device: &wgpu::Device,
        FrameSize { width, height }: FrameSize,
    ) -> Targets {
        let size = (width, height);
        let texture = |label, format| {
            crate::targets::texture_2d(
                device,
                label,
                size,
                1,
                1,
                format,
                crate::targets::RENDER_TARGET,
            )
        };
        let scene_levels = SCENE_LEVELS.min(crate::gpu_data::mip_levels(width, height) - 1);
        let textures = [
            crate::targets::texture_2d(
                device,
                "engine/post-scene",
                size,
                1,
                1 + scene_levels,
                SCENE_FORMAT,
                crate::targets::RENDER_TARGET | wgpu::TextureUsages::COPY_SRC,
            ),
            texture("engine/post-glow", GLOW_FORMAT),
            texture("engine/post-occlusion", OCCLUSION_FORMAT),
            texture("engine/post-reflections", REFLECTION_FORMAT),
            texture("engine/post-surface", SURFACE_FORMAT),
        ];
        let [scene, glow, occlusion, reflections, surface] = &textures;
        let whole = wgpu::TextureViewDescriptor::default();
        let views = [
            scene.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: 0,
                mip_level_count: Some(1),
                ..Default::default()
            }),
            glow.create_view(&whole),
            occlusion.create_view(&whole),
            reflections.create_view(&whole),
            surface.create_view(&whole),
        ];
        let made = Made {
            textures,
            views,
            names: COMPOSITED,
        };
        let view = |name| pipelines::view(made.view(name).expect("made_above"));
        let composite = pipelines::group(
            device,
            "engine/post-group",
            &POST,
            &self.layout,
            &[
                ("post", self.uniform.as_entire_binding()),
                ("scene", view("scene")),
                ("occlusion", view("occlusion")),
                ("linear", pipelines::sampler(&self.sampler)),
                ("reflections", view("reflections")),
            ],
        );
        Targets { made, composite }
    }
}

pub fn inverse_or_identity(matrix: Mat4) -> Mat4 {
    let inverse = matrix.inverse();
    if inverse.is_finite() {
        inverse
    } else {
        Mat4::IDENTITY
    }
}
