//! The main passes' first group: the camera and lighting uniforms, shadow
//! lookups, what glass refracts,
//! what surfaces reflect, terrain's horizon maps
//! and what placed objects cover, keyed by the generations of what it
//! binds, so it rebuilds when the shadow maps, environment maps, reflection
//! probe grid, horizon maps or occupancy maps are replaced.

use super::*;
use crate::bindings::CAMERA;
use crate::cached::GroupCache;
use crate::terrain_horizons::Horizons;

/// What the first group binds besides the camera and lighting uniforms.
pub(super) struct GroupSources<'a> {
    pub(super) shadows: &'a ShadowPass,
    pub(super) environment: &'a Environment,
    pub(super) probes: &'a ProbePass,
    pub(super) horizons: &'a Horizons,
    pub(super) occupancy: &'a crate::occupancy::OccupancyPass,
}

/// Bindings of the first group: [`CAMERA`]'s.
const CAMERA_BINDINGS: usize = 20;
/// Bindings of terrain's resolves' first group: `CAMERA_TERRAIN`'s.
const TERRAIN_BINDINGS: usize = 18;

pub(super) struct CameraBindings {
    pub(super) layout: crate::pipelines::GroupLayout,
    pub(super) buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    /// Clouds' shadows, copied in by the clouds' pass.
    pub(super) cloud_shadows: wgpu::Buffer,
    // The scene's atmosphere, written when it changes.
    lighting: Option<LightingUniform>,
    group: GroupCache<CAMERA_BINDINGS>,
    // Refraction through glass: the white texel bound where
    // no opaque color is, its sampler, and the group of transparent passes
    // over the opaque color.
    refraction_fallback: wgpu::TextureView,
    refraction_sampler: wgpu::Sampler,
    refraction_group: GroupCache<CAMERA_BINDINGS>,
    // Each reflection probe face's group, over the face's camera.
    probe_groups: [GroupCache<CAMERA_BINDINGS>; FACES],
    // Terrain's resolves' first group, the main one without
    // what glass refracts, and its layout.
    pub(super) terrain_layout: crate::pipelines::GroupLayout,
    terrain_group: GroupCache<TERRAIN_BINDINGS>,
}

impl CameraBindings {
    /// The layout, the camera uniform and the first group, over
    /// `lighting_buffer` and `sources`.
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lighting_buffer: wgpu::Buffer,
        sources: &GroupSources<'_>,
    ) -> Self {
        let buffer = crate::buffers::uniform(
            device,
            "engine/camera-uniform",
            size_of::<CameraUniform>() as u64,
        );
        let layout = crate::pipelines::layout(device, &CAMERA);
        let cloud_shadows = crate::clouds::shadow_uniform(device, queue);
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
        let mut bindings = Self {
            terrain_layout,
            terrain_group: GroupCache::default(),
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

    /// The lighting uniform's buffer, which the physically based sky's light
    /// is copied into.
    pub(super) fn lighting_buffer(&self) -> &wgpu::Buffer {
        &self.lighting_buffer
    }

    /// Writes `lighting` unless already written; returns the bytes written.
    pub(super) fn write_lighting(&mut self, queue: &wgpu::Queue, lighting: LightingUniform) -> u64 {
        if self.lighting == Some(lighting) {
            return 0;
        }
        queue.write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&lighting));
        self.lighting = Some(lighting);
        size_of::<LightingUniform>() as u64
    }

    /// Rebuilds the first group and terrain's when what `sources` binds was
    /// replaced.
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
    }

    /// The main passes' first group.
    pub(super) fn group(&self) -> &wgpu::BindGroup {
        self.group.get().expect("built with the bindings")
    }

    /// Terrain's resolves' first group.
    pub(super) fn terrain_group(&self) -> &wgpu::BindGroup {
        self.terrain_group.get().expect("built with the bindings")
    }

    /// The group of transparent passes when glass refracts:
    /// the main group over `scene`, the opaque color.
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

    /// The group glass refracting the opaque color binds, once prepared.
    pub(super) fn refraction_group(&self) -> Option<&wgpu::BindGroup> {
        self.refraction_group.get()
    }

    /// Each reflection probe face's group: the main group
    /// over the face's camera uniform in `probes`.
    pub(super) fn prepare_probe_groups(
        &mut self,
        device: &wgpu::Device,
        sources: &GroupSources<'_>,
    ) {
        for (face, group) in self.probe_groups.iter_mut().enumerate() {
            let refraction = (
                Bound::fixed(crate::pipelines::view(&self.refraction_fallback)),
                Bound::fixed(crate::pipelines::sampler(&self.refraction_sampler)),
            );
            group.bind(
                device,
                "engine/camera-bind-group",
                (&CAMERA, &self.layout),
                camera_entries(
                    (
                        sources.probes.camera(face),
                        &self.lighting_buffer,
                        &self.cloud_shadows,
                    ),
                    refraction,
                    sources,
                ),
            );
        }
    }

    /// Reflection probe face `face`'s group, once prepared.
    pub(super) fn probe_group(&self, face: usize) -> &wgpu::BindGroup {
        self.probe_groups[face]
            .get()
            .expect("prepared before captures")
    }
}

/// [`camera_entries`] without what glass refracts, for terrain's resolves
/// (`bindings::CAMERA_TERRAIN`).
fn terrain_entries<'a>(
    (camera, lighting, cloud_shadows): (&'a wgpu::Buffer, &'a wgpu::Buffer, &'a wgpu::Buffer),
    sources: &GroupSources<'a>,
) -> [(&'static str, Bound<'a>); TERRAIN_BINDINGS] {
    let [shadow, near, kept, moving, sampler, texel_sampler, levels] = sources.shadows.entries();
    let [horizons, horizon_records, horizon_sampler] = sources.horizons.entries();
    let (environment, environment_sampler) = sources.environment.binding();
    let (coverage, heights) = sources.occupancy.views();
    [
        ("camera", Bound::fixed(camera.as_entire_binding())),
        ("lighting", Bound::fixed(lighting.as_entire_binding())),
        shadow,
        near,
        kept,
        moving,
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
        levels,
        (
            "cloud_shadows",
            Bound::fixed(cloud_shadows.as_entire_binding()),
        ),
    ]
}

/// The main passes' first group's entries over the `camera` and
/// `lighting` uniforms: shadow lookups, what glass refracts with its
/// sampler, what surfaces reflect: the environment maps with their
/// sampler and the reflection probes' grid, and terrain's horizon maps.
fn camera_entries<'a>(
    (camera, lighting, cloud_shadows): (&'a wgpu::Buffer, &'a wgpu::Buffer, &'a wgpu::Buffer),
    (refraction, refraction_sampler): (Bound<'a>, Bound<'a>),
    sources: &GroupSources<'a>,
) -> [(&'static str, Bound<'a>); CAMERA_BINDINGS] {
    let [shadow, near, kept, moving, sampler, texel_sampler, levels] = sources.shadows.entries();
    let [horizons, horizon_records, horizon_sampler] = sources.horizons.entries();
    let (environment, environment_sampler) = sources.environment.binding();
    let (coverage, heights) = sources.occupancy.views();
    [
        ("camera", Bound::fixed(camera.as_entire_binding())),
        ("lighting", Bound::fixed(lighting.as_entire_binding())),
        shadow,
        near,
        kept,
        moving,
        sampler,
        texel_sampler,
        ("refraction_source", refraction),
        ("refraction_sampler", refraction_sampler),
        ("environment_maps", environment),
        ("environment_sampler", environment_sampler),
        ("probe_words", sources.probes.words()),
        horizons,
        horizon_records,
        horizon_sampler,
        ("occupancy_coverage", coverage),
        ("occupancy_heights", heights),
        levels,
        (
            "cloud_shadows",
            Bound::fixed(cloud_shadows.as_entire_binding()),
        ),
    ]
}
