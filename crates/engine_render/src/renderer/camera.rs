#![expect(
    clippy::expect_used,
    reason = "groups are built with their bindings before captures read them"
)]

use super::*;
use crate::bindings::CAMERA;
use crate::cached::GroupCache;
use crate::terrain_horizons::Horizons;

pub(super) struct GroupSources<'a> {
    pub(super) shadows: &'a ShadowPass,
    pub(super) environment: &'a Environment,
    pub(super) probes: &'a ProbePass,
    pub(super) horizons: &'a Horizons,
    pub(super) occupancy: &'a crate::occupancy::OccupancyPass,
    pub(super) sky: crate::sky::AirBindings<'a>,
    pub(super) clouds: crate::clouds::DomeBinding<'a>,
}

const CAMERA_BINDINGS: usize = 25;
const TERRAIN_BINDINGS: usize = 16;
const WATER_BINDINGS: usize = 24;

pub(super) struct CameraBindings {
    pub(super) layout: crate::pipelines::GroupLayout,
    pub(super) buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    pub(super) cloud_shadows: wgpu::Buffer,
    lighting: Option<LightingUniform>,
    group: GroupCache<CAMERA_BINDINGS>,
    refraction_fallback: wgpu::TextureView,
    refraction_sampler: wgpu::Sampler,
    refraction_group: GroupCache<CAMERA_BINDINGS>,
    probe_groups: [GroupCache<CAMERA_BINDINGS>; FACES],
    pub(super) terrain_layout: crate::pipelines::GroupLayout,
    terrain_group: GroupCache<TERRAIN_BINDINGS>,
    terrain_probe_groups: [GroupCache<TERRAIN_BINDINGS>; FACES],
    pub(super) water_layout: crate::pipelines::GroupLayout,
    water_group: GroupCache<WATER_BINDINGS>,
}

impl CameraBindings {
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (lighting_buffer, cloud_shadows): (wgpu::Buffer, wgpu::Buffer),
        sources: &GroupSources<'_>,
    ) -> Self {
        let buffer = crate::buffers::uniform(
            device,
            "engine/camera-uniform",
            size_of::<CameraUniform>() as u64,
        );
        let layout = crate::pipelines::layout(device, &CAMERA);
        let refraction_texture = crate::targets::texel(
            device,
            queue,
            "engine/refraction-fallback",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            [255; 4],
        );
        let refraction_fallback =
            refraction_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let refraction_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/refraction-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let terrain_layout = crate::pipelines::layout(device, &crate::bindings::CAMERA_TERRAIN);
        let water_layout = crate::pipelines::layout(device, &crate::bindings::CAMERA_WATER);
        let mut bindings = Self {
            terrain_layout,
            terrain_group: GroupCache::default(),
            terrain_probe_groups: Default::default(),
            water_layout,
            water_group: GroupCache::default(),
            layout,
            buffer,
            lighting_buffer,
            cloud_shadows,
            lighting: None,
            group: GroupCache::default(),
            refraction_fallback,
            refraction_sampler,
            refraction_group: GroupCache::default(),
            probe_groups: Default::default(),
        };
        bindings.refresh(device, sources);
        bindings
    }

    pub(super) fn lighting_buffer(&self) -> &wgpu::Buffer {
        &self.lighting_buffer
    }

    pub(super) fn forget_lighting(&mut self) {
        self.lighting = None;
    }

    pub(super) fn write_lighting(&mut self, queue: &wgpu::Queue, lighting: LightingUniform) -> u64 {
        if self.lighting == Some(lighting) {
            return 0;
        }
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&lighting));
        self.lighting = Some(lighting);
        size_of::<LightingUniform>() as u64
    }

    pub(super) fn refresh(&mut self, device: &wgpu::Device, sources: &GroupSources<'_>) {
        let refraction = (
            Bound::fixed(crate::pipelines::view(&self.refraction_fallback)),
            Bound::fixed(crate::pipelines::sampler(&self.refraction_sampler)),
        );
        let uniforms = (&self.buffer, &self.lighting_buffer, &self.cloud_shadows);
        self.group.bind(
            device,
            "engine/camera-bind-group",
            (&CAMERA, &self.layout),
            camera_entries(uniforms, refraction, sources),
        );
        self.terrain_group.bind(
            device,
            "engine/camera-terrain-bind-group",
            (&crate::bindings::CAMERA_TERRAIN, &self.terrain_layout),
            terrain_entries(uniforms, sources),
        );
        self.water_group.bind(
            device,
            "engine/camera-water-bind-group",
            (&crate::bindings::CAMERA_WATER, &self.water_layout),
            water_entries(
                uniforms,
                Bound::fixed(crate::pipelines::sampler(&self.refraction_sampler)),
                sources,
            ),
        );
    }

    pub(super) fn group(&self) -> &wgpu::BindGroup {
        self.group.get().expect("built with the bindings")
    }

    pub(super) fn terrain_group(&self) -> &wgpu::BindGroup {
        self.terrain_group.get().expect("built with the bindings")
    }

    pub(super) fn water_group(&self) -> &wgpu::BindGroup {
        self.water_group.get().expect("built with the bindings")
    }

    pub(super) fn prepare_refraction(
        &mut self,
        device: &wgpu::Device,
        scene: Bound<'_>,
        sources: &GroupSources<'_>,
    ) {
        let refraction = (
            scene,
            Bound::fixed(crate::pipelines::sampler(&self.refraction_sampler)),
        );
        self.refraction_group.bind(
            device,
            "engine/camera-bind-group",
            (&CAMERA, &self.layout),
            camera_entries(
                (&self.buffer, &self.lighting_buffer, &self.cloud_shadows),
                refraction,
                sources,
            ),
        );
    }

    pub(super) fn refraction_group(&self) -> Option<&wgpu::BindGroup> {
        self.refraction_group.get()
    }

    pub(super) fn prepare_probe_groups(
        &mut self,
        device: &wgpu::Device,
        sources: &GroupSources<'_>,
    ) {
        let groups = self
            .probe_groups
            .iter_mut()
            .zip(&mut self.terrain_probe_groups);
        for (face, (group, terrain)) in groups.enumerate() {
            let refraction = (
                Bound::fixed(crate::pipelines::view(&self.refraction_fallback)),
                Bound::fixed(crate::pipelines::sampler(&self.refraction_sampler)),
            );
            let uniforms = (
                sources.probes.camera(face),
                &self.lighting_buffer,
                &self.cloud_shadows,
            );
            group.bind(
                device,
                "engine/camera-bind-group",
                (&CAMERA, &self.layout),
                camera_entries(uniforms, refraction, sources),
            );
            terrain.bind(
                device,
                "engine/camera-terrain-bind-group",
                (&crate::bindings::CAMERA_TERRAIN, &self.terrain_layout),
                terrain_entries(uniforms, sources),
            );
        }
    }

    pub(super) fn probe_group(&self, face: usize) -> &wgpu::BindGroup {
        self.probe_groups[face]
            .get()
            .expect("prepared before captures")
    }

    pub(super) fn terrain_probe_group(&self, face: usize) -> &wgpu::BindGroup {
        self.terrain_probe_groups[face]
            .get()
            .expect("prepared before captures")
    }
}

fn terrain_entries<'a>(
    (camera, lighting, cloud_shadows): (&'a wgpu::Buffer, &'a wgpu::Buffer, &'a wgpu::Buffer),
    sources: &GroupSources<'a>,
) -> [(&'static str, Bound<'a>); TERRAIN_BINDINGS] {
    let [shadow, atlas, tiles, sampler, texel_sampler] = sources.shadows.entries();
    let [horizons, horizon_records, horizon_sampler] = sources.horizons.entries();
    let (environment, environment_sampler) = sources.environment.binding();
    let (coverage, heights) = sources.occupancy.views();
    [
        ("camera", Bound::fixed(camera.as_entire_binding())),
        ("lighting", Bound::fixed(lighting.as_entire_binding())),
        shadow,
        atlas,
        tiles,
        sampler,
        texel_sampler,
        ("environment_maps", environment),
        ("environment_sampler", environment_sampler),
        ("probe_words", sources.probes.words()),
        horizons,
        horizon_records,
        horizon_sampler,
        ("occupancy_coverage", coverage),
        ("occupancy_heights", heights),
        (
            "cloud_shadows",
            Bound::fixed(cloud_shadows.as_entire_binding()),
        ),
    ]
}

fn shown_dome((generation, roles, shown): (u64, [u8; 2], bool), role: usize) -> u64 {
    (generation * 4 + u64::from(roles[role])) * 2 + u64::from(shown)
}

fn camera_entries<'a>(
    uniforms: (&'a wgpu::Buffer, &'a wgpu::Buffer, &'a wgpu::Buffer),
    (refraction, refraction_sampler): (Bound<'a>, Bound<'a>),
    sources: &GroupSources<'a>,
) -> [(&'static str, Bound<'a>); CAMERA_BINDINGS] {
    let mut entries = water_entries(uniforms, refraction_sampler, sources)
        .into_iter()
        .chain([("refraction_source", refraction)]);
    std::array::from_fn(|_| {
        entries
            .next()
            .expect("water's entries and what glass refracts fill the group")
    })
}

fn water_entries<'a>(
    (camera, lighting, cloud_shadows): (&'a wgpu::Buffer, &'a wgpu::Buffer, &'a wgpu::Buffer),
    refraction_sampler: Bound<'a>,
    sources: &GroupSources<'a>,
) -> [(&'static str, Bound<'a>); WATER_BINDINGS] {
    let [shadow, atlas, tiles, sampler, texel_sampler] = sources.shadows.entries();
    let [horizons, horizon_records, horizon_sampler] = sources.horizons.entries();
    let (environment, environment_sampler) = sources.environment.binding();
    let (coverage, heights) = sources.occupancy.views();
    let (air, view_table, _, table_sampler) = sources.sky;
    let clouds = &sources.clouds;
    [
        ("camera", Bound::fixed(camera.as_entire_binding())),
        ("lighting", Bound::fixed(lighting.as_entire_binding())),
        shadow,
        atlas,
        tiles,
        sampler,
        texel_sampler,
        ("refraction_sampler", refraction_sampler),
        ("environment_maps", environment),
        ("environment_sampler", environment_sampler),
        ("probe_words", sources.probes.words()),
        horizons,
        horizon_records,
        horizon_sampler,
        ("occupancy_coverage", coverage),
        ("occupancy_heights", heights),
        (
            "cloud_shadows",
            Bound::fixed(cloud_shadows.as_entire_binding()),
        ),
        ("sky_air", Bound::fixed(air.as_entire_binding())),
        (
            "sky_view_table",
            Bound::fixed(crate::pipelines::view(view_table)),
        ),
        (
            "sky_table_sampler",
            Bound::fixed(crate::pipelines::sampler(table_sampler)),
        ),
        (
            "sky_clouds",
            Bound::counted(
                shown_dome(clouds.key, 1),
                crate::pipelines::view(clouds.new),
            ),
        ),
        (
            "sky_clouds_old",
            Bound::counted(
                shown_dome(clouds.key, 0),
                crate::pipelines::view(clouds.old),
            ),
        ),
        (
            "sky_cloud_sampler",
            Bound::fixed(crate::pipelines::sampler(clouds.sampler)),
        ),
        (
            "sky_domes",
            Bound::fixed(clouds.uniform.as_entire_binding()),
        ),
    ]
}
