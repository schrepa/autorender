use engine_render::renderer::*;
use engine_render::cached::GroupCache;
use engine_render::gpu_data::{jitter_clip, jittered};
use engine_render::parts::{Jitter, TemporalFrame};
use engine_render::targets::{FrameRegion, ScreenTarget};

pub const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

#[derive(Clone, Copy)]
pub(crate) struct History {
    view_projection: glam::Mat4,
    size: FrameSize,
    identity: u64,
    structure: u64,
    pose_generation: u64,
}

pub struct Motion {
    texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl Drop for Motion {
    fn drop(&mut self) {
        self.texture.destroy();
    }
}

#[derive(Default)]
pub struct PreviousFrame {
    history: Option<History>,
    frames: u64,
    motion: ScreenTarget<Motion>,
    pose_layout: Option<engine_render::pipelines::GroupLayout>,
    pose_groups: [GroupCache<4>; 4],
    pub(crate) wanted: bool,
}

impl PreviousFrame {
    pub(crate) fn reset(&mut self) {
        self.history = None;
    }

    pub(crate) fn release(&mut self) {
        self.history = None;
        self.motion.release();
        for group in &mut self.pose_groups {
            group.clear();
        }
    }

    pub(crate) fn frame(
        &mut self,
        device: &wgpu::Device,
        frame: &RenderFrame<'_>,
        (region, pose_generation): (FrameRegion, u64),
        jitter: Option<Jitter>,
    ) -> TemporalFrame {
        let (size, render) = (region.display, region.render);
        if self.motion.resizes(size) {
            let motion = motion_target(device, size);
            self.motion.set(size, motion);
            self.history = None;
        }
        let snapshot = frame.snapshot();
        let view_projection = snapshot.view_projection;
        let history = History {
            view_projection,
            size,
            identity: frame.identity(),
            structure: frame.structure_revision(),
            pose_generation,
        };
        let last = self
            .history
            .filter(|last| last.size == size && last.identity == history.identity);
        let Jitter { offset, sequence } = jitter.unwrap_or(Jitter {
            offset: [0.0; 2],
            sequence: self.frames,
        });
        let clip = jitter_clip(offset, render.width, render.height);
        let previous = last.map_or(view_projection, |last| last.view_projection);
        let camera = CameraUniform::temporal(
            view_projection,
            snapshot.camera_position,
            snapshot.elapsed_seconds,
            previous,
            clip,
            render.pixels(),
            sequence,
        );
        TemporalFrame {
            parity: (sequence & 1) as usize,
            camera,
            rasterized: jittered(view_projection, clip),
            history_valid: last.is_some(),
            motion_valid: last.is_some_and(|last| {
                last.structure == history.structure && last.pose_generation == pose_generation
            }),
            history,
        }
    }

    pub(crate) fn after_submit(&mut self, frame: Option<&TemporalFrame>) {
        if let Some(frame) = frame {
            self.history = Some(frame.history);
            self.frames = self.frames.wrapping_add(1);
        }
    }

    pub(crate) fn attachment(&self, resource: &str) -> Option<wgpu::TextureView> {
        (resource == "motion")
            .then(|| self.motion.get().map(|motion| motion.view.clone()))
            .flatten()
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.motion
            .size()
            .map_or(0, |size| u64::from(size.width) * u64::from(size.height) * 4)
    }
}

fn motion_target(device: &wgpu::Device, size: FrameSize) -> Motion {
    let texture = engine_render::targets::texture_2d(
        device,
        "engine/taa-motion",
        size.pixels(),
        1,
        1,
        MOTION_FORMAT,
        engine_render::targets::RENDER_TARGET,
    );
    let view = texture.create_view(&Default::default());
    Motion { texture, view }
}

impl Core {
    pub fn motion(&self) -> Option<Bound<'_>> {
        let motion = self.previous.motion.get()?;
        Some(Bound::of(motion, engine_render::pipelines::view(&motion.view)))
    }
}
