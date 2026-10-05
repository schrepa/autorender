//! The main passes' first group: the camera and lighting uniforms, shadow
//! lookups (decisions 020 and 028), what glass refracts (decision 038),
//! what surfaces reflect (decision 043), terrain's horizon maps (decision
//! 052) and what placed objects cover (decision 059), rebuilt when the
//! shadow maps, environment maps, reflection probe grid, horizon maps or
//! occupancy map it binds are replaced.

use super::*;
use crate::bindings::CAMERA;
use crate::terrain_horizons::Horizons;

/// What the first group binds besides the camera and lighting uniforms.
pub(super) struct GroupSources<'a> {
    pub(super) shadows: &'a ShadowPass,
    pub(super) environment: &'a Environment,
    pub(super) probes: &'a ProbePass,
    pub(super) horizons: &'a Horizons,
    pub(super) occupancy: &'a crate::occupancy::OccupancyPass,
}

impl GroupSources<'_> {
    fn key(&self) -> CameraGroupsKey {
        (
            self.shadows.generation(),
            self.environment.generation(),
            self.probes.words().1,
            self.horizons.generation(),
            self.occupancy.views().2,
        )
    }
}

pub(super) struct CameraBindings {
    pub(super) layout: wgpu::BindGroupLayout,
    pub(super) buffer: wgpu::Buffer,
    lighting_buffer: wgpu::Buffer,
    // The scene's atmosphere (decision 018), written when it changes.
    lighting: Option<LightingUniform>,
    // The group, for the shadow maps, environment maps and probe grid of
    // these generations.
    pub(super) group: wgpu::BindGroup,
    pub(super) key: CameraGroupsKey,
    // Refraction through glass (decision 038): the white texel bound where
    // no opaque color is, its sampler, and the group of transparent passes
    // over the opaque color of post target generation `.0`.
    refraction_fallback: wgpu::TextureView,
    refraction_sampler: wgpu::Sampler,
    refraction_group: Cached<(u64, CameraGroupsKey), wgpu::BindGroup>,
    // Each reflection probe face's group (decision 043), built with the
    // main group's key.
    probe_groups: Vec<wgpu::BindGroup>,
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
        let group = camera_group(
            device,
            &layout,
            (&buffer, &lighting_buffer),
            (&refraction_fallback, &refraction_sampler),
            sources,
        );
        Self {
            layout,
            buffer,
            lighting_buffer,
            lighting: None,
            group,
            key: sources.key(),
            refraction_fallback,
            refraction_sampler,
            refraction_group: Cached::default(),
            probe_groups: Vec::new(),
        }
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

    /// Rebuilds the group when the maps or probe grid of `sources` it binds
    /// were replaced, and forgets the probe faces' groups.
    pub(super) fn refresh(&mut self, device: &wgpu::Device, sources: &GroupSources<'_>) {
        let key = sources.key();
        if key == self.key {
            return;
        }
        self.key = key;
        self.group = camera_group(
            device,
            &self.layout,
            (&self.buffer, &self.lighting_buffer),
            (&self.refraction_fallback, &self.refraction_sampler),
            sources,
        );
        self.probe_groups.clear();
    }

    /// The group of transparent passes when glass refracts (decision 038):
    /// the main group over `scene`, the opaque color of post target
    /// generation `generation`.
    pub(super) fn prepare_refraction(
        &mut self,
        device: &wgpu::Device,
        (scene, generation): (&wgpu::TextureView, u64),
        sources: &GroupSources<'_>,
    ) {
        let key = (generation, self.key);
        if self.refraction_group.stale(&key) {
            let group = camera_group(
                device,
                &self.layout,
                (&self.buffer, &self.lighting_buffer),
                (scene, &self.refraction_sampler),
                sources,
            );
            self.refraction_group.set(key, group);
        }
    }

    /// The group glass refracting the opaque color binds, once prepared.
    pub(super) fn refraction_group(&self) -> Option<&wgpu::BindGroup> {
        self.refraction_group.get()
    }

    /// Each reflection probe face's group (decision 043): the main group
    /// over the face's camera uniform in `probes`.
    pub(super) fn prepare_probe_groups(
        &mut self,
        device: &wgpu::Device,
        sources: &GroupSources<'_>,
    ) {
        if self.probe_groups.is_empty() {
            self.probe_groups = (0..FACES)
                .map(|face| {
                    camera_group(
                        device,
                        &self.layout,
                        (sources.probes.camera(face), &self.lighting_buffer),
                        (&self.refraction_fallback, &self.refraction_sampler),
                        sources,
                    )
                })
                .collect();
        }
    }

    /// Reflection probe face `face`'s group, once prepared.
    pub(super) fn probe_group(&self, face: usize) -> &wgpu::BindGroup {
        &self.probe_groups[face]
    }
}

/// The main passes' first group over the `camera` and `lighting` uniforms:
/// shadow lookups, what glass refracts (decision 038) with its sampler,
/// what surfaces reflect (decision 043): the environment maps with their
/// sampler and the reflection probes' grid, and terrain's horizon maps
/// (decision 052).
fn camera_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    (camera, lighting): (&wgpu::Buffer, &wgpu::Buffer),
    (refraction, refraction_sampler): (&wgpu::TextureView, &wgpu::Sampler),
    sources: &GroupSources<'_>,
) -> wgpu::BindGroup {
    let [shadow, near, kept, moving, sampler, texel_sampler, levels] = sources.shadows.entries();
    let [horizons, horizon_records, horizon_sampler] = sources.horizons.entries();
    let (environment, environment_sampler) = sources.environment.binding();
    let probe_words = sources.probes.words().0;
    let (coverage, heights, _) = sources.occupancy.views();
    crate::pipelines::group(
        device,
        "engine/camera-bind-group",
        &CAMERA,
        layout,
        &[
            ("camera", camera.as_entire_binding()),
            ("lighting", lighting.as_entire_binding()),
            shadow,
            near,
            kept,
            moving,
            sampler,
            texel_sampler,
            ("refraction_source", crate::pipelines::view(refraction)),
            (
                "refraction_sampler",
                crate::pipelines::sampler(refraction_sampler),
            ),
            ("environment_maps", crate::pipelines::view(environment)),
            (
                "environment_sampler",
                crate::pipelines::sampler(environment_sampler),
            ),
            ("probe_words", probe_words.as_entire_binding()),
            horizons,
            horizon_records,
            horizon_sampler,
            ("occupancy_coverage", crate::pipelines::view(coverage)),
            ("occupancy_heights", crate::pipelines::view(heights)),
            levels,
        ],
    )
}
