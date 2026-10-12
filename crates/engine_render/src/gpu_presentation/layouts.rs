//! GPU presentation's compute pipelines over their shared layouts (tables in
//! `crate::bindings`): each layout lists every binding its passes read within
//! the eight storage buffers per stage WebGPU guarantees, and passes skipping
//! some bind fillers there.

use crate::shaders::entry::presentation;
use std::cell::OnceCell;

use crate::bindings::{
    PRESENTATION_COMPACT, PRESENTATION_COMPOSE, PRESENTATION_CULL, PRESENTATION_FADES,
    PRESENTATION_LEVEL, PRESENTATION_SKIN_BOUNDS, PRESENTATION_TESTS,
};
use crate::gpu_data::PRESENTATION_SHADER;
use crate::pipelines::{Family, Fillers, layout};

/// The passes' shared layouts; skinned slots' bounds have
/// theirs created with the first frame drawing skinned slots.
pub(super) struct Families {
    pub(super) compose: Family,
    pub(super) cull: Family,
    pub(super) tests: Family,
    pub(super) compact: Family,
    pub(super) fades: Family,
    skin_bounds: OnceCell<Family>,
}

/// The pipelines every frame runs, and those only some scenes do, created
/// with the first frame that runs them: skinned slots'
/// palettes and bounds and reflection probes' still frames
/// and culling.
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
    skin_palette: OnceCell<wgpu::ComputePipeline>,
    skin_bounds: OnceCell<wgpu::ComputePipeline>,
    count_still: OnceCell<wgpu::ComputePipeline>,
    probe_cull: OnceCell<wgpu::ComputePipeline>,
}

/// The pipelines, their layouts and the fillers of bindings passes skip,
/// and the layout of parent composition's per-level group.
pub(super) struct Shaders {
    pub(super) pipelines: Pipelines,
    pub(super) families: Families,
    pub(super) level_layout: wgpu::BindGroupLayout,
    fillers: Fillers,
    module: wgpu::ShaderModule,
}

impl Shaders {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/presentation.wgsl"),
            source: wgpu::ShaderSource::Wgsl(PRESENTATION_SHADER.into()),
        });
        let level_layout = layout(device, &PRESENTATION_LEVEL);
        let families = Families {
            compose: Family::new(device, &PRESENTATION_COMPOSE, &[&level_layout]),
            cull: Family::new(device, &PRESENTATION_CULL, &[]),
            tests: Family::new(device, &PRESENTATION_TESTS, &[]),
            compact: Family::new(device, &PRESENTATION_COMPACT, &[]),
            fades: Family::new(device, &PRESENTATION_FADES, &[]),
            skin_bounds: OnceCell::new(),
        };
        let pipeline = |entry: crate::shaders::Entry, family: &Family| {
            crate::pipelines::compute_entry(device, &family.pipeline, &module, entry)
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
            skin_palette: OnceCell::new(),
            skin_bounds: OnceCell::new(),
            count_still: OnceCell::new(),
            probe_cull: OnceCell::new(),
        };
        Self {
            pipelines,
            families,
            level_layout,
            fillers: Fillers::new(device),
            module,
        }
    }

    /// The layout of skinned slots' bounds.
    pub(super) fn skin_bounds_family(&self, device: &wgpu::Device) -> &Family {
        self.families
            .skin_bounds
            .get_or_init(|| Family::new(device, &PRESENTATION_SKIN_BOUNDS, &[]))
    }

    fn lazy<'a>(
        &self,
        device: &wgpu::Device,
        cell: &'a OnceCell<wgpu::ComputePipeline>,
        family: &Family,
        entry: crate::shaders::Entry,
    ) -> &'a wgpu::ComputePipeline {
        cell.get_or_init(|| {
            crate::pipelines::compute_entry(device, &family.pipeline, &self.module, entry)
        })
    }

    /// Skinned slots' palettes.
    pub(super) fn skin_palette(&self, device: &wgpu::Device) -> &wgpu::ComputePipeline {
        let family = &self.families.compose;
        self.lazy(
            device,
            &self.pipelines.skin_palette,
            family,
            presentation::skin_palette,
        )
    }

    /// Skinned slots' bounds from their palettes.
    pub(super) fn skin_bounds(&self, device: &wgpu::Device) -> &wgpu::ComputePipeline {
        let family = self.skin_bounds_family(device);
        self.lazy(
            device,
            &self.pipelines.skin_bounds,
            family,
            presentation::skin_bounds_pass,
        )
    }

    /// Reflection probes' count of slots' still frames.
    pub(super) fn count_still(&self, device: &wgpu::Device) -> &wgpu::ComputePipeline {
        let family = &self.families.tests;
        self.lazy(
            device,
            &self.pipelines.count_still,
            family,
            presentation::count_still,
        )
    }

    /// The culling of reflection probe faces.
    pub(super) fn probe_cull(&self, device: &wgpu::Device) -> &wgpu::ComputePipeline {
        let family = &self.families.tests;
        self.lazy(
            device,
            &self.pipelines.probe_cull,
            family,
            presentation::probe_cull,
        )
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
        family.group(device, label, resources, &self.fillers)
    }
}
