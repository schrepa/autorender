use bytemuck::Zeroable;
use engine_core::{RenderSnapshot, glam::Mat4};
use engine_render::cached::{Bound, GroupCache};
use engine_render::gpu_layouts::scene::PostUniform;
use engine_render::pipelines::{self, Fullscreen, fullscreen};
use engine_render::post::inverse_or_identity;
use engine_render::profiling::GpuProfiler;
use engine_render::renderer::Core;
use engine_render::screen_effects::haze_runs;
use engine_render::targets::{
    FrameRegion, FrameSize, RENDER_TARGET, ScreenTarget, fullscreen_pass, texture_2d,
};

use crate::bindings::{HAZE, HAZE_FILTER};
use crate::pipeline_layouts::{HAZE_FILTER_PIPELINES, HAZE_PIPELINES};
use crate::shaders::entry::haze;

const HAZE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

struct Targets {
    textures: [wgpu::Texture; 3],
    marched: wgpu::TextureView,
    haze: wgpu::TextureView,
    history: wgpu::TextureView,
}

impl Drop for Targets {
    fn drop(&mut self) {
        for texture in &self.textures {
            texture.destroy();
        }
    }
}

impl Targets {
    fn new(device: &wgpu::Device, FrameSize { width, height }: FrameSize) -> Self {
        let size = (width, height);
        let texture = |label, usage| texture_2d(device, label, size, 1, 1, HAZE_FORMAT, usage);
        let textures = [
            texture("engine/post-haze-march", RENDER_TARGET),
            texture(
                "engine/post-haze",
                RENDER_TARGET | wgpu::TextureUsages::COPY_SRC,
            ),
            texture(
                "engine/post-haze-history",
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            ),
        ];
        let [marched, haze, history] = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        Self {
            textures,
            marched,
            haze,
            history,
        }
    }
}

struct Passes {
    march: wgpu::RenderPipeline,
    march_layout: pipelines::GroupLayout,
    filter: wgpu::RenderPipeline,
    filter_layout: pipelines::GroupLayout,
    uniform: wgpu::Buffer,
}

impl Passes {
    fn new(device: &wgpu::Device) -> Self {
        let pipeline = |label, entry, layout| {
            let layout = pipelines::layout_of(device, layout);
            fullscreen(
                device,
                &Fullscreen {
                    label,
                    layout: &layout,
                    vertex: haze::vs_fullscreen,
                    fragment: entry,
                    constants: &[],
                    targets: &[Some(HAZE_FORMAT.into())],
                    depth: None,
                },
            )
        };
        Self {
            march: pipeline("engine/post-haze", haze::fs_haze, &HAZE_PIPELINES),
            march_layout: pipelines::layout(device, &HAZE),
            filter: pipeline(
                "engine/post-haze-filter",
                haze::fs_haze_filter,
                &HAZE_FILTER_PIPELINES,
            ),
            filter_layout: pipelines::layout(device, &HAZE_FILTER),
            uniform: engine_render::buffers::uniform(
                device,
                "engine/haze-uniform",
                size_of::<PostUniform>() as u64,
            ),
        }
    }
}

#[derive(Clone, Copy)]
struct History {
    view_projection: Mat4,
    eye: engine_core::glam::Vec3,
    drawn: (u32, u32),
}

#[derive(Default)]
pub(crate) struct Haze {
    passes: Option<Passes>,
    targets: ScreenTarget<Targets>,
    region: FrameRegion,
    history: Option<History>,
    kept: Option<History>,
    frames: u32,
    written: Option<PostUniform>,
    march_group: GroupCache<2>,
    filter_group: GroupCache<4>,
    active: bool,
    runs: bool,
}

impl Haze {
    pub(crate) fn runs(&self) -> bool {
        self.runs
    }

    pub(crate) fn prepare(
        &mut self,
        core: &Core,
        snapshot: &RenderSnapshot,
        (active, display, debug): (bool, bool, bool),
    ) {
        self.active = active && !debug;
        self.runs = false;
        if debug {
            return;
        }
        if !active {
            self.release();
            return;
        }
        let device = &core.device;
        let region = core.display.region;
        let size = region.display;
        let passes = self.passes.get_or_insert_with(|| Passes::new(device));
        if self.targets.resizes(size) {
            let targets = Targets::new(device, size);
            self.targets.set(size, targets);
            self.history = None;
        }
        self.region = region;
        let Some(targets) = self.targets.get() else {
            return;
        };
        let runs = haze_runs(snapshot);
        if runs {
            let post = || Bound::fixed(passes.uniform.as_entire_binding());
            let target = |view| Bound::of(targets, pipelines::view(view));
            let depth = &core.depth.view;
            self.march_group.bind(
                device,
                "engine/post-haze-group",
                (&HAZE, &passes.march_layout),
                [("post", post()), ("depth_texture", depth.bound())],
            );
            self.filter_group.bind(
                device,
                "engine/post-haze-filter-group",
                (&HAZE_FILTER, &passes.filter_layout),
                [
                    ("post", post()),
                    ("depth_texture", depth.bound()),
                    ("marched", target(&targets.marched)),
                    ("haze", target(&targets.history)),
                ],
            );
        }
        self.kept = (display && runs).then_some(History {
            view_projection: snapshot.view_projection,
            eye: snapshot.camera_position,
            drawn: self.region.render.pixels(),
        });
        let (history, turn) = if self.kept.is_none() {
            (None, 0.0)
        } else {
            let turn = (self.frames % 1024) as f32 * 0.618_034;
            (self.history, turn - turn.floor())
        };
        let (width, height) = (region.render.width as f32, region.render.height as f32);
        let view_projection = snapshot.view_projection;
        let focal = view_projection.row(1).truncate().length();
        let uniform = PostUniform {
            inverse_view_projection: inverse_or_identity(view_projection).to_cols_array_2d(),
            eye: snapshot
                .camera_position
                .extend(focal * height * 0.5)
                .to_array(),
            size: [width, height, 1.0 / width, 1.0 / height],
            haze: history.map_or([turn, 0.0, 0.0, 1.0], |last| {
                [turn, last.drawn.0 as f32, last.drawn.1 as f32, 1.0]
            }),
            previous_view_projection: history
                .map_or(view_projection, |last| last.view_projection)
                .to_cols_array_2d(),
            previous_eye: history
                .map_or(snapshot.camera_position, |last| last.eye)
                .extend(0.0)
                .to_array(),
            ..PostUniform::zeroed()
        };
        if runs && self.written != Some(uniform) {
            core.queue
                .write_buffer(&passes.uniform, 0, bytemuck::bytes_of(&uniform));
            self.written = Some(uniform);
        }
        self.runs = runs;
    }

    fn release(&mut self) {
        self.targets.release();
        self.filter_group.clear();
        self.history = None;
    }

    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera: &wgpu::BindGroup,
        profiler: &GpuProfiler,
    ) {
        let (Some(passes), Some(march), Some(filter), Some(targets)) = (
            &self.passes,
            self.march_group.get(),
            self.filter_group.get(),
            self.targets.get(),
        ) else {
            return;
        };
        let drawn = self.region.render.pixels();
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-haze",
                (&targets.marched, drawn),
                profiler
                    .span("haze")
                    .map(|span| span.render_part(true, false)),
            );
            pass.set_pipeline(&passes.march);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(1, march, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-haze-filter",
                (&targets.haze, drawn),
                profiler
                    .span("haze")
                    .map(|span| span.render_part(false, true)),
            );
            pass.set_pipeline(&passes.filter);
            pass.set_bind_group(0, filter, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_texture(
            targets.textures[1].as_image_copy(),
            targets.textures[2].as_image_copy(),
            wgpu::Extent3d {
                width: drawn.0,
                height: drawn.1,
                depth_or_array_layers: 1,
            },
        );
    }

    pub(crate) fn binding(&self) -> Option<Bound<'_>> {
        let targets = self.targets.get()?;
        Some(Bound::of(targets, pipelines::view(&targets.haze)))
    }

    pub(crate) fn after_submit(&mut self) {
        let kept = self.kept.take();
        self.history = if self.active { kept } else { None };
        if self.history.is_some() {
            self.frames = self.frames.wrapping_add(1);
        }
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        let Some(size) = self.targets.size() else {
            return 0;
        };
        u64::from(size.width) * u64::from(size.height) * 24
    }
}
