//! GPU presentation's compute pipelines over their shared layouts (decision
//! 049): each layout lists every binding its passes read, by
//! presentation.wgsl's numbers, within the eight storage buffers per stage
//! WebGPU guarantees, and passes skipping some bind fillers there.

use crate::gpu_data::PRESENTATION_SHADER;
use crate::pipelines::{Fillers, Slot, filled_group, group_layout};

use Slot::{Read, Texture, Uniform, Write};

// Parent composition, and skinned slots' palettes from the composed parents.
const COMPOSE_BINDINGS: &[(u32, Slot)] = &[
    (0, Uniform),
    (1, Read),
    (2, Read),
    (3, Read),
    (4, Write),
    (5, Read),
    (15, Read),
    (16, Read),
    (17, Write),
];
const CULL_BINDINGS: &[(u32, Slot)] = &[
    (0, Uniform),
    (1, Read),
    (2, Read),
    (3, Read),
    (4, Write),
    (7, Read),
    (8, Write),
    (9, Write),
    (14, Read),
];
// The occlusion test, reflection probe culling and still frame count, and
// the previous-pose shift.
const TEST_BINDINGS: &[(u32, Slot)] = &[
    (0, Uniform),
    (2, Read),
    (7, Read),
    (8, Write),
    (9, Write),
    (13, Write),
    (14, Read),
    (18, Texture),
    (19, Write),
    (23, Write),
];
// Group sums, their scan, draw arguments and the scatter of visible slots:
// one group per compaction binds all four.
const COMPACT_BINDINGS: &[(u32, Slot)] = &[
    (0, Uniform),
    (6, Read),
    (7, Read),
    (9, Write),
    (10, Write),
    (11, Write),
    (12, Write),
    (19, Write),
    (24, Write),
];
const SKIN_BOUNDS_BINDINGS: &[(u32, Slot)] = &[
    (0, Uniform),
    (7, Read),
    (8, Write),
    (9, Write),
    (17, Write),
    (20, Read),
    (21, Read),
    (22, Write),
];

/// A shared layout: its bindings, group layout and pipeline layout.
pub(super) struct Family {
    bindings: &'static [(u32, Slot)],
    group: wgpu::BindGroupLayout,
    pipeline: wgpu::PipelineLayout,
}

impl Family {
    fn new(
        device: &wgpu::Device,
        label: &str,
        bindings: &'static [(u32, Slot)],
        level: Option<&wgpu::BindGroupLayout>,
    ) -> Self {
        let group = group_layout(device, label, wgpu::ShaderStages::COMPUTE, bindings);
        let groups = [Some(&group), level];
        let pipeline = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: if level.is_some() {
                &groups
            } else {
                &groups[..1]
            },
            immediate_size: 0,
        });
        Self {
            bindings,
            group,
            pipeline,
        }
    }
}

/// The passes' shared layouts.
pub(super) struct Families {
    pub(super) compose: Family,
    pub(super) cull: Family,
    pub(super) tests: Family,
    pub(super) compact: Family,
    pub(super) skin_bounds: Family,
}

pub(super) struct Pipelines {
    pub(super) shift_previous: wgpu::ComputePipeline,
    pub(super) compose_parents: wgpu::ComputePipeline,
    pub(super) cull: wgpu::ComputePipeline,
    pub(super) skin_palette: wgpu::ComputePipeline,
    pub(super) skin_bounds: wgpu::ComputePipeline,
    pub(super) sum_groups: wgpu::ComputePipeline,
    pub(super) scan_groups: wgpu::ComputePipeline,
    pub(super) batch_args: wgpu::ComputePipeline,
    pub(super) scatter: wgpu::ComputePipeline,
    pub(super) occlusion_test: wgpu::ComputePipeline,
    pub(super) count_still: wgpu::ComputePipeline,
    pub(super) probe_cull: wgpu::ComputePipeline,
}

/// The pipelines, their layouts and the fillers of bindings passes skip,
/// and the layout of parent composition's per-level group.
pub(super) struct Shaders {
    pub(super) pipelines: Pipelines,
    pub(super) families: Families,
    pub(super) level_layout: wgpu::BindGroupLayout,
    fillers: Fillers,
}

impl Shaders {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/presentation.wgsl"),
            source: wgpu::ShaderSource::Wgsl(PRESENTATION_SHADER.into()),
        });
        let level_layout = group_layout(
            device,
            "engine/presentation-level-layout",
            wgpu::ShaderStages::COMPUTE,
            &[(0, Uniform)],
        );
        let families = Families {
            compose: Family::new(
                device,
                "engine/presentation-compose-layout",
                COMPOSE_BINDINGS,
                Some(&level_layout),
            ),
            cull: Family::new(
                device,
                "engine/presentation-cull-layout",
                CULL_BINDINGS,
                None,
            ),
            tests: Family::new(
                device,
                "engine/presentation-test-layout",
                TEST_BINDINGS,
                None,
            ),
            compact: Family::new(
                device,
                "engine/presentation-compact-layout",
                COMPACT_BINDINGS,
                None,
            ),
            skin_bounds: Family::new(
                device,
                "engine/presentation-skin-bounds-layout",
                SKIN_BOUNDS_BINDINGS,
                None,
            ),
        };
        let pipeline = |entry: &str, family: &Family| {
            crate::pipelines::compute_entry(device, &family.pipeline, &module, entry)
        };
        let pipelines = Pipelines {
            shift_previous: pipeline("shift_previous", &families.tests),
            compose_parents: pipeline("compose_parents", &families.compose),
            cull: pipeline("cull", &families.cull),
            skin_palette: pipeline("skin_palette", &families.compose),
            skin_bounds: pipeline("skin_bounds_pass", &families.skin_bounds),
            sum_groups: pipeline("sum_groups", &families.compact),
            scan_groups: pipeline("scan_groups", &families.compact),
            batch_args: pipeline("batch_args", &families.compact),
            scatter: pipeline("scatter", &families.compact),
            occlusion_test: pipeline("occlusion_test", &families.tests),
            count_still: pipeline("count_still", &families.tests),
            probe_cull: pipeline("probe_cull", &families.tests),
        };
        Self {
            pipelines,
            families,
            level_layout,
            fillers: Fillers::new(device),
        }
    }

    /// A group of `family`'s layout holding `entries`, filled where they
    /// leave bindings out.
    pub(super) fn group(
        &self,
        device: &wgpu::Device,
        family: &Family,
        label: &str,
        entries: &[wgpu::BindGroupEntry<'_>],
    ) -> wgpu::BindGroup {
        filled_group(
            device,
            label,
            &family.group,
            family.bindings,
            entries,
            &self.fillers,
        )
    }
}
