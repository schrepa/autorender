//! GPU presentation's compute pipelines over their shared layouts (decision
//! 049; tables in `crate::bindings`): each layout lists every binding its
//! passes read within the eight storage buffers per stage WebGPU guarantees,
//! and passes skipping some bind fillers there.

use crate::bindings::{
    PRESENTATION_COMPACT, PRESENTATION_COMPOSE, PRESENTATION_CULL, PRESENTATION_LEVEL,
    PRESENTATION_SKIN_BOUNDS, PRESENTATION_TESTS, Table,
};
use crate::gpu_data::PRESENTATION_SHADER;
use crate::pipelines::{Fillers, filled_group, layout};

/// A shared layout: its table (`crate::bindings`), group layout and
/// pipeline layout.
pub(super) struct Family {
    table: &'static Table,
    group: wgpu::BindGroupLayout,
    pipeline: wgpu::PipelineLayout,
}

impl Family {
    fn new(
        device: &wgpu::Device,
        table: &'static Table,
        level: Option<&wgpu::BindGroupLayout>,
    ) -> Self {
        let group = layout(device, table);
        let groups = [Some(&group), level];
        let pipeline = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(table.label),
            bind_group_layouts: if level.is_some() {
                &groups
            } else {
                &groups[..1]
            },
            immediate_size: 0,
        });
        Self {
            table,
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
        let level_layout = layout(device, &PRESENTATION_LEVEL);
        let families = Families {
            compose: Family::new(device, &PRESENTATION_COMPOSE, Some(&level_layout)),
            cull: Family::new(device, &PRESENTATION_CULL, None),
            tests: Family::new(device, &PRESENTATION_TESTS, None),
            compact: Family::new(device, &PRESENTATION_COMPACT, None),
            skin_bounds: Family::new(device, &PRESENTATION_SKIN_BOUNDS, None),
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

    /// A group of `family`'s layout binding `resources` by name, filled
    /// where they leave bindings out.
    pub(super) fn group(
        &self,
        device: &wgpu::Device,
        family: &Family,
        label: &str,
        resources: &[(&str, wgpu::BindingResource<'_>)],
    ) -> wgpu::BindGroup {
        filled_group(
            device,
            label,
            family.table,
            &family.group,
            resources,
            &self.fillers,
        )
    }
}
