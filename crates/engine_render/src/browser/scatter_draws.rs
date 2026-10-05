//! Placed objects' draws (decision 057): what terrains' placement layers
//! stand on their level-0 tiles, laid out into the scatter pass whenever
//! tiles, assets or their meshes' residency change, culled each frame and
//! drawn after terrain in the first opaque phase.

use super::frame::FrameState;
use super::*;
use crate::gpu_data::{ColorData, pack_color, pack_surface};
use crate::scatter::{GroupLayout, LEVELS, Objects, Record};
use engine_core::glam::{Mat4, Vec3};

/// One draw group of placed objects: its mesh and material, where its
/// lists start, its objects and the levels it draws, and whether it casts
/// shadows and tests its material's texels doing so.
pub(super) struct ScatterDraw {
    mesh: MeshHandle,
    material: MaterialHandle,
    ids: u32,
    count: u32,
    levels: usize,
    casts: Option<bool>,
}

/// What the scatter pass's objects were laid out from: each terrain's
/// identity and tiles' revision, its assets, and the meshes' layout.
pub(super) type ScatterKey = (
    Vec<(u64, u64, Vec<Option<(MeshHandle, MaterialHandle, u64)>>)>,
    u64,
);

impl Renderer {
    /// Lays out the placed objects of `snapshot`'s terrains when anything
    /// they come from changed, creates their pipelines for `kind`, and
    /// plans this frame's culling; returns whether any draw.
    pub(super) fn prepare_scatter(&mut self, snapshot: &RenderSnapshot, kind: PassKind) -> bool {
        let terrains = &snapshot.terrain.terrains;
        let key: ScatterKey = (
            terrains
                .iter()
                .map(|terrain| {
                    (
                        terrain.id,
                        terrain.tiles_revision,
                        terrain
                            .assets
                            .iter()
                            .map(|asset| {
                                asset.as_ref().map(|asset| {
                                    (
                                        asset.mesh.handle,
                                        asset.material.handle,
                                        asset.material.version,
                                    )
                                })
                            })
                            .collect(),
                    )
                })
                .collect(),
            self.mesh_layout,
        );
        if self.scatter_key.as_ref() != Some(&key) {
            let (objects, draws) = self.lay_out_scatter(terrains);
            self.scatter.upload(&self.device, &self.queue, &objects);
            self.scatter_draws = draws;
            self.scatter_key = Some(key);
        }
        // Each group's pipeline variant.
        let variants: Vec<Variant> = self
            .scatter_draws
            .iter()
            .map(|draw| self.scatter_binding(draw).1)
            .collect();
        for variant in variants {
            if !self.mesh_pipelines.variants.contains_key(&(kind, variant)) {
                self.ensure_mesh_layout(kind);
                let pipeline = self.create_variant(kind, variant);
                self.mesh_pipelines
                    .variants
                    .insert((kind, variant), pipeline);
            }
        }
        let level_of_detail = snapshot.environment.level_of_detail;
        let pixel_error = level_of_detail
            .enabled
            .then_some(level_of_detail.pixel_error);
        let height = self.display.size().1 as f32;
        self.scatter.prepare(
            &self.device,
            &self.queue,
            (snapshot.view_projection, snapshot.camera_position),
            (pixel_error, height),
            (self.shadows.drawn_maps(), self.shadows.map_resolution()),
        )
    }

    /// The scatter pass's objects and draw groups from `terrains`: one group
    /// per terrain asset whose mesh is resident and material uploaded, its
    /// objects from every level-0 tile.
    fn lay_out_scatter(
        &self,
        terrains: &[engine_core::RenderTerrain],
    ) -> (Objects, Vec<ScatterDraw>) {
        let mut objects = Objects {
            poses: Vec::new(),
            colors: Vec::new(),
            records: Vec::new(),
            groups: Vec::new(),
        };
        let mut draws = Vec::new();
        let mut ids = 0;
        for terrain in terrains {
            for (asset_index, asset) in terrain.assets.iter().enumerate() {
                let Some(asset) = asset else {
                    continue;
                };
                let Some(mesh) = self.meshes.get(&asset.mesh.handle) else {
                    continue;
                };
                if mesh.skinned {
                    continue;
                }
                let group = objects.groups.len() as u32;
                let bounds = asset.mesh.mesh.bounds();
                let center = (bounds.min() + bounds.max()) * 0.5;
                let radius = (bounds.max() - bounds.min()).length() * 0.5;
                let color = ColorData {
                    color: pack_color(asset.material.material.color()),
                    surface: pack_surface(asset.material.material.surface()),
                    flags: 0,
                };
                let first = objects.records.len();
                for tile in terrain.tiles.iter().filter(|tile| tile.key().level == 0) {
                    for placed in tile.placed() {
                        if usize::from(placed.asset) != asset_index {
                            continue;
                        }
                        let model = placed_model(terrain.origin, placed);
                        objects.poses.push(pose_rows(model));
                        objects.colors.push(color);
                        objects.records.push(Record {
                            center: model.transform_point3(center).to_array(),
                            radius: radius * placed.scale,
                            group,
                            scale: placed.scale,
                            pad: [0; 2],
                        });
                    }
                }
                let count = (objects.records.len() - first) as u32;
                if count == 0 {
                    continue;
                }
                let draw = mesh.draw(&self.mesh_arena);
                let mut levels = vec![(draw.index_count, draw.first_index)];
                for &(first_index, index_count, _) in mesh.lods.iter().take(LEVELS as usize - 1) {
                    if index_count == 0 {
                        break;
                    }
                    levels.push((index_count, draw.first_index + first_index));
                }
                let error = |level: usize| mesh.lods.get(level).map_or(0.0, |lod| lod.2);
                // Blending materials cast nothing; masked ones test their
                // texels (decision 031).
                let material = &asset.material.material;
                let casts = (!material.is_transparent())
                    .then(|| matches!(material.alpha_mode(), engine_core::AlphaMode::Mask { .. }));
                draws.push(ScatterDraw {
                    mesh: asset.mesh.handle,
                    material: asset.material.handle,
                    ids,
                    count,
                    levels: levels.len(),
                    casts,
                });
                objects.groups.push(GroupLayout {
                    count,
                    levels,
                    base_vertex: draw.base_vertex,
                    errors: [error(0), error(1)],
                });
                ids += count * LEVELS;
            }
        }
        (objects, draws)
    }

    /// The material group and pipeline variant of placed group `draw`.
    fn scatter_binding(&self, draw: &ScatterDraw) -> (&wgpu::BindGroup, Variant) {
        let mesh = self.meshes.get(&draw.mesh);
        let (group, variant) = self.materials.binding(Some(draw.material));
        let variant = variant
            .with_extended(mesh.is_some_and(|mesh| mesh.extended))
            .with_skinned(false);
        (group, variant)
    }

    /// Draws this frame's placed objects into the first opaque phase; the
    /// camera and lights are bound. Returns the draws.
    pub(super) fn draw_scatter(
        &self,
        state: &FrameState<'_>,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> u32 {
        if !state.scatter {
            return 0;
        }
        let Some(groups) = self.scatter.pose_groups() else {
            return 0;
        };
        let kind = state.passes.opaque_kind;
        let temporal = state.packing.temporal_frame.is_some() && self.temporal.is_some();
        pass.set_bind_group(0, &self.camera.group, &[]);
        if kind.lit
            && let Some(lights) = self.lights.as_ref()
        {
            pass.set_bind_group(3, lights.fragment_group(), &[]);
        }
        pass.set_bind_group(2, &groups[usize::from(temporal)], &[]);
        let mut draws = 0;
        for (index, draw) in self.scatter_draws.iter().enumerate() {
            let Some(mesh) = self.meshes.get(&draw.mesh) else {
                continue;
            };
            let (group, variant) = self.scatter_binding(draw);
            let Some(pipeline) = self.mesh_pipelines.variants.get(&(kind, variant)) else {
                continue;
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, group, &[]);
            let source = mesh.draw(&self.mesh_arena);
            pass.set_vertex_buffer(0, source.vertices.slice(..));
            if let Some(extras) = source.extras.filter(|_| mesh.extended) {
                pass.set_vertex_buffer(2, extras.slice(..));
            }
            pass.set_index_buffer(source.indices.slice(..), wgpu::IndexFormat::Uint32);
            for level in 0..draw.levels {
                let (ids, args, offset) = self.scatter.draw_source(
                    0,
                    (index, draw.ids + level as u32 * draw.count),
                    level,
                );
                pass.set_vertex_buffer(1, ids);
                pass.draw_indexed_indirect(args, offset);
                draws += 1;
            }
        }
        draws
    }

    /// Draws this frame's placed objects culled for shadow map `map` with
    /// the depth passes' caster pipelines `casters`; the map's view is
    /// bound. Returns the draws.
    pub(super) fn draw_scatter_shadow(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        map: usize,
        casters: &[wgpu::RenderPipeline; 4],
    ) -> u32 {
        let (Some(view), Some(poses)) = (self.scatter.map_view(map), self.scatter.shadow_group())
        else {
            return 0;
        };
        pass.set_bind_group(1, poses, &[]);
        let mut bound = None;
        let mut draws = 0;
        for (index, draw) in self.scatter_draws.iter().enumerate() {
            let (Some(masked), Some(mesh)) = (draw.casts, self.meshes.get(&draw.mesh)) else {
                continue;
            };
            if masked {
                let Some(material) = self.materials.group(draw.material) else {
                    continue;
                };
                pass.set_bind_group(2, material, &[]);
            }
            if bound != Some(masked) {
                pass.set_pipeline(&casters[usize::from(masked)]);
                bound = Some(masked);
            }
            let source = mesh.draw(&self.mesh_arena);
            pass.set_vertex_buffer(0, source.vertices.slice(..));
            pass.set_index_buffer(source.indices.slice(..), wgpu::IndexFormat::Uint32);
            for level in 0..draw.levels {
                let (ids, args, offset) = self.scatter.draw_source(
                    view,
                    (index, draw.ids + level as u32 * draw.count),
                    level,
                );
                pass.set_vertex_buffer(1, ids);
                pass.draw_indexed_indirect(args, offset);
                draws += 1;
            }
        }
        draws
    }
}

/// The world matrix of `placed` on a terrain whose origin is `origin`: its
/// up leaning as placed, turned about it by its yaw, and scaled.
fn placed_model(origin: Vec3, placed: &engine_core::Placed) -> Mat4 {
    let [lean_x, lean_z] = placed.lean;
    let up = Vec3::new(
        lean_x,
        engine_core::math::max(1.0 - lean_x * lean_x - lean_z * lean_z, 0.01).sqrt(),
        lean_z,
    )
    .normalize();
    let (sin, cos) = engine_core::math::sin_cos(placed.yaw);
    let forward = Vec3::new(cos, 0.0, sin);
    let tangent = (forward - up * forward.dot(up)).normalize_or(Vec3::X);
    let across = tangent.cross(up);
    let scale = placed.scale;
    Mat4::from_cols(
        (tangent * scale).extend(0.0),
        (up * scale).extend(0.0),
        (across * scale).extend(0.0),
        (origin + Vec3::from_array(placed.position)).extend(1.0),
    )
}

/// The three rows of `model`'s affine part, the layout slot poses keep.
fn pose_rows(model: Mat4) -> crate::gpu_data::PoseData {
    let rows = model.transpose();
    [
        rows.x_axis.to_array(),
        rows.y_axis.to_array(),
        rows.z_axis.to_array(),
    ]
}
