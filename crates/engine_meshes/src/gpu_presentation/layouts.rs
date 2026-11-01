use engine_render::shaders::entry::presentation;
use std::cell::OnceCell;

use engine_render::bindings::PRESENTATION_LEVEL;
use engine_render::pipeline_layouts::{
    PRESENTATION_COMPACT_PIPELINES, PRESENTATION_COMPOSE_PIPELINES, PRESENTATION_CULL_PIPELINES,
    PRESENTATION_FADE_PIPELINES, PRESENTATION_TEST_PIPELINES,
};
use engine_render::pipelines::{Family, Fillers, layout};

pub(super) struct Families {
    pub(super) compose: Family,
    pub(super) cull: Family,
    pub(super) tests: Family,
    pub(super) compact: Family,
    pub(super) fades: Family,
}

pub(super) struct Pipelines {
    pub(super) shift_previous: wgpu::ComputePipeline,
    pub(super) compose_parents: wgpu::ComputePipeline,
    pub(super) cull: wgpu::ComputePipeline,
    pub(super) sum_groups: wgpu::ComputePipeline,
    pub(super) scan_groups: wgpu::ComputePipeline,
    pub(super) batch_args: wgpu::ComputePipeline,
    pub(super) scatter: wgpu::ComputePipeline,
    pub(super) occlusion_test: wgpu::ComputePipeline,
    pub(super) fade_levels: wgpu::ComputePipeline,
    count_still: OnceCell<wgpu::ComputePipeline>,
    probe_cull: OnceCell<wgpu::ComputePipeline>,
}

pub(super) struct Shaders {
    pub(super) pipelines: Pipelines,
    pub(super) families: Families,
    pub(super) level_layout: engine_render::pipelines::GroupLayout,
    fillers: Fillers,
}

impl Shaders {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let level_layout = layout(device, &PRESENTATION_LEVEL);
        let families = Families {
            compose: Family::new(device, &PRESENTATION_COMPOSE_PIPELINES, &[&level_layout]),
            cull: Family::new(device, &PRESENTATION_CULL_PIPELINES, &[]),
            tests: Family::new(device, &PRESENTATION_TEST_PIPELINES, &[]),
            compact: Family::new(device, &PRESENTATION_COMPACT_PIPELINES, &[]),
            fades: Family::new(device, &PRESENTATION_FADE_PIPELINES, &[]),
        };
        let pipeline = |entry: engine_render::shaders::Entry, family: &Family| {
            engine_render::pipelines::compute_entry(device, &family.pipeline, entry)
        };
        let pipelines = Pipelines {
            shift_previous: pipeline(presentation::shift_previous, &families.tests),
            compose_parents: pipeline(presentation::compose_parents, &families.compose),
            cull: pipeline(presentation::cull, &families.cull),
            sum_groups: pipeline(presentation::sum_groups, &families.compact),
            scan_groups: pipeline(presentation::scan_groups, &families.compact),
            batch_args: pipeline(presentation::batch_args, &families.compact),
            scatter: pipeline(presentation::scatter, &families.compact),
            occlusion_test: pipeline(presentation::occlusion_test, &families.tests),
            fade_levels: pipeline(presentation::fade_levels, &families.fades),
            count_still: OnceCell::new(),
            probe_cull: OnceCell::new(),
        };
        Self {
            pipelines,
            families,
            level_layout,
            fillers: Fillers::new(device),
        }
    }

    fn lazy<'a>(
        &self,
        device: &wgpu::Device,
        cell: &'a OnceCell<wgpu::ComputePipeline>,
        family: &Family,
        entry: engine_render::shaders::Entry,
    ) -> &'a wgpu::ComputePipeline {
        cell.get_or_init(|| engine_render::pipelines::compute_entry(device, &family.pipeline, entry))
    }

    pub(super) fn count_still(&self, device: &wgpu::Device) -> &wgpu::ComputePipeline {
        let family = &self.families.tests;
        self.lazy(
            device,
            &self.pipelines.count_still,
            family,
            presentation::count_still,
        )
    }

    pub(super) fn probe_cull(&self, device: &wgpu::Device) -> &wgpu::ComputePipeline {
        let family = &self.families.tests;
        self.lazy(
            device,
            &self.pipelines.probe_cull,
            family,
            presentation::probe_cull,
        )
    }

    pub(super) fn release_views(&mut self) {
        self.pipelines.count_still = OnceCell::new();
        self.pipelines.probe_cull = OnceCell::new();
    }

    pub(super) fn group(
        &self,
        device: &wgpu::Device,
        family: &Family,
        label: &str,
        resources: &[(&str, wgpu::BindingResource<'_>)],
    ) -> wgpu::BindGroup {
        family.group(device, label, resources, &self.fillers)
    }
}
