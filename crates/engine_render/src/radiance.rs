//! Main-view radiance composition and its one display-grade boundary.

#![expect(
    clippy::expect_used,
    reason = "every rendered frame sizes radiance before drawing"
)]

use std::{cell::Cell, rc::Rc};

use engine_core::{Atmosphere, RadianceStats};

use crate::bindings::GRADE;
use crate::buffers::ReadbackRing;
use crate::cached::Bound;
use crate::gpu_data::{GRADE_SHADER, GradeUniform};
use crate::shaders::entry::grade;
use crate::targets::{FrameSize, ScreenTarget};

/// Main-view light scaled by exposure / 64, with ambient share in alpha.
pub(crate) const RADIANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    group: wgpu::BindGroup,
}

impl Drop for Target {
    fn drop(&mut self) {
        self.texture.destroy();
    }
}

pub(crate) struct RadiancePass {
    layout: crate::pipelines::GroupLayout,
    pipelines: [wgpu::RenderPipeline; 2],
    uniform: wgpu::Buffer,
    written: Option<GradeUniform>,
    limits: wgpu::Buffer,
    readbacks: ReadbackRing<u64>,
    completed: Rc<Cell<RadianceStats>>,
    source_frame: u64,
    target: ScreenTarget<Target>,
}

impl RadiancePass {
    pub(crate) fn new(device: &wgpu::Device, canvas: wgpu::TextureFormat) -> Self {
        let layout = crate::pipelines::layout(device, &GRADE);
        let pipeline_layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::GRADE_PIPELINES,
            &[&layout],
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/grade.wgsl"),
            source: wgpu::ShaderSource::Wgsl(GRADE_SHADER.into()),
        });
        let pipelines =
            [(canvas, true), (crate::post::SCENE_FORMAT, false)].map(|(format, canvas)| {
                crate::pipelines::fullscreen(
                    device,
                    &crate::pipelines::Fullscreen {
                        label: "engine/grade",
                        layout: &pipeline_layout,
                        module: &module,
                        vertex: grade::vs_fullscreen,
                        fragment: grade::fs_grade,
                        constants: &[
                            ("ENCODE_OUTPUT", if format.is_srgb() { 0.0 } else { 1.0 }),
                            ("CANVAS_OUTPUT", if canvas { 1.0 } else { 0.0 }),
                        ],
                        targets: &[Some(format.into())],
                        depth: None,
                    },
                )
            });
        Self {
            layout,
            pipelines,
            uniform: crate::buffers::uniform(
                device,
                "engine/grade-uniform",
                size_of::<GradeUniform>() as u64,
            ),
            written: None,
            limits: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/radiance-limits"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            readbacks: ReadbackRing::new("engine/radiance-limits-readback"),
            completed: Rc::new(Cell::new(RadianceStats::default())),
            source_frame: 0,
            target: ScreenTarget::default(),
        }
    }

    /// Sizes the main-view target and returns the grade's uploaded bytes.
    pub(crate) fn begin(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: FrameSize,
        atmosphere: &Atmosphere,
    ) -> u64 {
        self.source_frame = self.source_frame.wrapping_add(1);
        if self.target.resizes(size) {
            let texture = crate::targets::texture_2d(
                device,
                "engine/radiance",
                size.pixels(),
                1,
                1,
                RADIANCE_FORMAT,
                crate::targets::RENDER_TARGET,
            );
            let view = texture.create_view(&Default::default());
            let group = crate::pipelines::group(
                device,
                "engine/grade-group",
                &GRADE,
                &self.layout,
                &[
                    ("radiance", crate::pipelines::view(&view)),
                    ("grade", self.uniform.as_entire_binding()),
                    ("radiance_limits", self.limits.as_entire_binding()),
                ],
            );
            self.target.set(
                size,
                Target {
                    texture,
                    view,
                    group,
                },
            );
        }
        let uniform = GradeUniform::new(atmosphere);
        if self
            .written
            .as_ref()
            .is_none_or(|old| bytemuck::bytes_of(old) != bytemuck::bytes_of(&uniform))
        {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            self.written = Some(uniform);
            size_of::<GradeUniform>() as u64
        } else {
            0
        }
    }

    pub(crate) fn view(&self) -> &wgpu::TextureView {
        &self.target.get().expect("sized before drawing").view
    }

    pub(crate) fn bound(&self) -> Bound<'_> {
        let target = self.target.get().expect("sized before drawing");
        Bound::of(target, crate::pipelines::view(&target.view))
    }

    pub(crate) fn limits(&self) -> &wgpu::Buffer {
        &self.limits
    }

    pub(crate) fn clear_limits(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.limits, 0, None);
    }

    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass<'_>, post: bool) {
        pass.set_pipeline(&self.pipelines[usize::from(post)]);
        pass.set_bind_group(
            0,
            &self.target.get().expect("sized before drawing").group,
            &[],
        );
        pass.draw(0..3, 0..1);
    }

    pub(crate) fn read_back(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.readbacks
            .record(device, encoder, &self.limits, 4, self.source_frame);
    }

    pub(crate) fn after_submit(&self) {
        let completed = Rc::clone(&self.completed);
        self.readbacks.after_submit(move |words, source_frame| {
            if completed
                .get()
                .source_frame
                .is_none_or(|last| source_frame > last)
            {
                completed.set(RadianceStats {
                    source_frame: Some(source_frame),
                    limit_mask: Some(words.get(0)),
                });
            }
        });
    }

    pub(crate) fn stats(&self) -> RadianceStats {
        self.completed.get()
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.target
            .size()
            .map_or(0, |size| u64::from(size.width) * u64::from(size.height) * 8)
            + self.uniform.size()
            + self.limits.size()
            + self.readbacks.capacity_bytes()
    }
}
