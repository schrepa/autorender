#![expect(
    clippy::expect_used,
    reason = "temporal frames create the alternate poses their groups bind"
)]

use engine_render::renderer::*;
use engine_render::bindings::TEMPORAL_POSES;
use engine_render::parts::TemporalFrame;

impl PreviousFrame {
    pub(crate) fn prepare_pose_groups(&mut self, device: &wgpu::Device, instances: &Instances) {
        let Self {
            pose_layout,
            pose_groups,
            ..
        } = self;
        let layout =
            pose_layout.get_or_insert_with(|| engine_render::pipelines::layout(device, &TEMPORAL_POSES));
        let poses = instances.frame_poses(0);
        let both = [0, 1].map(|parity| poses.bound(parity).expect("temporal_frames_create"));
        let colors = poses.colors.bound();
        let grounds = Bound::fixed(instances.grounds().as_entire_binding());
        for (index, group) in pose_groups.iter_mut().enumerate() {
            let (current, previous) = (index / 2, index % 2);
            group.bind(
                device,
                "engine/temporal-pose-group",
                (&TEMPORAL_POSES, layout),
                [
                    ("poses", both[current].clone()),
                    ("colors", colors.clone()),
                    ("previous_poses", both[previous].clone()),
                    ("grounds", grounds.clone()),
                ],
            );
        }
    }

    pub(crate) fn pose_group(&self, frame: &TemporalFrame) -> Option<&wgpu::BindGroup> {
        let previous = if frame.motion_valid {
            1 - frame.parity
        } else {
            frame.parity
        };
        self.pose_groups[2 * frame.parity + previous].get()
    }
}

impl Core {
    pub fn temporal_pose_group(&self, frame: &TemporalFrame) -> Option<&wgpu::BindGroup> {
        self.previous.pose_group(frame)
    }
}
