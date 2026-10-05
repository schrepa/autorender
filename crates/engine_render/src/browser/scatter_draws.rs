//! Placed objects' draws (decision 057): what terrains' placement layers
//! stand on their tiles, held in the scatter pool's slots as tiles arrive
//! and leave, culled each frame and drawn after terrain in the first opaque
//! phase. Draw groups, one per terrain asset, follow the assets.

use std::collections::HashMap;
use std::sync::Arc;

use super::frame::FrameState;
use super::scatter_pool::{ScatterPool, TileObjects};
use super::*;
use crate::gpu_data::{ColorData, pack_color, pack_surface};
use crate::impostors::{BakeSource, IMPOSTOR_MIN_TRIANGLES, ImpostorRecord};
use crate::occupancy::Cover;
use crate::scatter::{
    GroupLayout, LEVELS, NEAR_LEVEL_SHIFT, Record, SnapSources, TERRAIN_INDEX, TerrainInfo,
};
use engine_core::TileKey;
use engine_core::glam::{Mat4, Vec3};

/// Levels of an automatic chain the camera draws: the mesh and its first
/// simplification. Deeper ones prune whole parts, such as foliage cards,
/// and serve only shadow maps whose texels hide their errors.
const CAMERA_LEVELS: u32 = 2;

/// Placed objects' draw groups and slots.
#[derive(Default)]
pub(super) struct ScatterDraws {
    // What the groups were built from.
    key: Option<ScatterKey>,
    groups: Vec<ScatterDraw>,
    layouts: Vec<GroupLayout>,
    // The group of each terrain's asset, and each terrain's index among the
    // frame's terrains.
    indices: HashMap<(u64, usize), u32>,
    terrains: HashMap<u64, u32>,
    pool: ScatterPool,
    // Each group's impostor (decision 048), zeroed without one, and whether
    // the groups' layouts changed since they last uploaded.
    impostors: Vec<ImpostorRecord>,
    changed: bool,
    // Each group's cover (decision 059).
    covers: Vec<Cover>,
}

impl ScatterDraws {
    /// Each draw group's impostor record in the placed objects' atlas
    /// (decision 060), zeroed without one.
    pub(super) fn impostors(&self) -> &[ImpostorRecord] {
        &self.impostors
    }

    /// Resident objects standing far away (decision 061).
    pub(super) fn far_objects(&self) -> u32 {
        self.pool.far
    }

    /// Each draw group's cover, which the occupancy map draws (decision
    /// 059).
    pub(super) fn covers(&self) -> &[Cover] {
        &self.covers
    }
}

/// One draw group of placed objects: each level's mesh, its material, its
/// objects' color and bounding sphere (mesh space), where its lists start,
/// its live objects, and whether it casts shadows and tests its material's
/// texels doing so.
struct ScatterDraw {
    meshes: Vec<MeshHandle>,
    material: MaterialHandle,
    source: engine_core::Material,
    color: ColorData,
    sphere: (Vec3, f32),
    ids: u32,
    count: u32,
    casts: Option<bool>,
}

/// An asset as its group was built: its mesh, material and version, its
/// coarser meshes, which of these meshes were resident, and its sway's
/// bits.
type AssetKey = (
    MeshHandle,
    MaterialHandle,
    u64,
    Vec<MeshHandle>,
    Vec<bool>,
    u32,
);

/// What the draw groups were built from: each terrain's identity and
/// assets, and the meshes' layout.
type ScatterKey = (Vec<(u64, Vec<Option<AssetKey>>)>, u64);

/// A placed asset's levels as drawn: each one's index count, first index
/// and base vertex, its mesh, the coarser ones' errors, and how many the
/// camera draws.
type Levels = (
    Vec<(u32, u32, i32)>,
    Vec<MeshHandle>,
    [f32; LEVELS as usize - 1],
    u32,
);

impl Renderer {
    /// Brings the scatter pool up to `snapshot`'s terrains: rebuilds the
    /// draw groups when their assets changed, gives tiles that arrived
    /// slots and frees those of tiles that left, uploads what changed,
    /// creates the groups' pipelines for `kind`, and plans this frame's
    /// culling; returns whether any object draws.
    pub(super) fn prepare_scatter(
        &mut self,
        snapshot: &RenderSnapshot,
        kind: PassKind,
        impostors: bool,
    ) -> bool {
        let terrains = &snapshot.terrain.terrains;
        let key = self.scatter_key(terrains);
        if self.scatter_draws.key.as_ref() != Some(&key) || self.scatter_draws.pool.fragmented() {
            self.build_scatter_groups(terrains);
            self.scatter_draws.key = Some(key);
        }
        for terrain in terrains {
            let draws = &mut self.scatter_draws;
            if !draws.pool.tiles_changed(terrain.id, terrain.tiles_revision) {
                continue;
            }
            let current: HashMap<TileKey, usize> = terrain
                .tiles
                .iter()
                .filter(|tile| !tile.placed().is_empty())
                .map(|tile| (tile.key(), Arc::as_ptr(tile) as usize))
                .collect();
            draws.pool.retain(terrain.id, |key, identity| {
                current.get(&key) == Some(&identity)
            });
            for tile in terrain.tiles.iter() {
                if tile.placed().is_empty() || draws.pool.contains(terrain.id, tile.key()) {
                    continue;
                }
                let objects = tile_objects(draws, terrain, tile);
                draws.pool.insert(
                    (terrain.id, tile.key()),
                    Arc::as_ptr(tile) as usize,
                    objects,
                );
            }
        }
        // Objects stand on the finest heights resident this frame.
        let infos: Vec<TerrainInfo> = terrains
            .iter()
            .enumerate()
            .map(|(index, terrain)| TerrainInfo {
                origin: terrain.origin.extend(terrain.pyramid.spacing(0)).to_array(),
                levels: [
                    self.terrain.cover_terrain(index).0,
                    terrain.pyramid.levels(),
                    0,
                    0,
                ],
            })
            .collect();
        let draws = &mut self.scatter_draws;
        let (dirty, counts_changed) = draws.pool.take_changes();
        // The kept shadow map redraws where objects came or went
        // (decision 060).
        for (min, max) in draws.pool.take_moved() {
            self.shadows.dirty_kept(min, max);
        }
        self.scatter.write_slots(
            &self.device,
            &self.queue,
            (&draws.pool.poses, &draws.pool.colors, &draws.pool.records),
            dirty,
        );
        if counts_changed || std::mem::take(&mut draws.changed) {
            let mut ids = 0;
            for (group, &count) in draws.groups.iter_mut().zip(&draws.pool.counts) {
                group.ids = ids;
                group.count = count;
                ids += count * LEVELS;
            }
            self.scatter.set_groups(
                &self.device,
                &self.queue,
                &draws.layouts,
                &draws.pool.counts,
            );
        }
        let sources = self.terrain.cover_sources().map(|sources| SnapSources {
            heights: sources.heights,
            directory: sources.directory,
            generation: sources.generation,
            revision: sources.revision,
            terrains: &infos,
        });
        self.scatter
            .prepare_snap(&self.device, &self.queue, sources);
        // Each group's pipeline variant.
        let variants: Vec<Variant> = self
            .scatter_draws
            .groups
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
        // Frames keeping still casters cast placed objects into the kept map
        // (decision 060), not the moving one.
        let keeps_still = self.shadows.keeps_still();
        let maps: Vec<(usize, engine_core::glam::Mat4)> = self
            .shadows
            .drawn_maps()
            .iter()
            .copied()
            .filter(|&(map, _)| !(keeps_still && map == crate::shadows::MOVING_MAP))
            .collect();
        self.scatter.prepare(
            &self.device,
            &self.queue,
            (snapshot.view_projection, snapshot.camera_position),
            (pixel_error, height, impostors),
            (
                &maps,
                self.shadows.kept_refresh(),
                self.shadows.map_resolution(),
            ),
        )
    }

    /// The impostors of placed draw groups (decision 057) on GPU-presented
    /// frames with impostors: each group's, baked into `encoder` after its
    /// material's levels where missing into the placed objects' atlas
    /// (decision 060), zeroed where its material blends, its mesh is too
    /// small or the budget the plan batches' leave is spent. The groups'
    /// layouts take their errors and records.
    pub(super) fn prepare_scatter_impostors(
        &mut self,
        snapshot: &RenderSnapshot,
        gpu_mode: bool,
        encoder: &mut Option<wgpu::CommandEncoder>,
    ) {
        let settings = snapshot.environment.level_of_detail;
        let enabled = gpu_mode && settings.enabled && settings.impostors;
        let (meshes, materials) = (&self.meshes, &self.materials);
        self.impostors.placed.retain(
            |(mesh, material)| {
                meshes.contains_key(mesh)
                    && material.is_none_or(|handle| materials.group(handle).is_some())
            },
            |handle| Self::placed_material(snapshot, handle),
        );
        // Plan batches' impostors come first within a budget.
        let budget = settings
            .max_impostors
            .map(|budget| budget.saturating_sub(self.impostors.atlas.live()));
        self.impostors.placed.enforce_budget(budget);
        let mut records = Vec::with_capacity(self.scatter_draws.groups.len());
        for group in &self.scatter_draws.groups {
            let mut record = ImpostorRecord::default();
            let mesh = self.meshes.get(&group.meshes[0]);
            let source = snapshot
                .terrain
                .terrains
                .iter()
                .flat_map(|terrain| terrain.assets.iter().flatten())
                .find(|asset| asset.mesh.handle == group.meshes[0])
                .map(|asset| &asset.mesh.mesh);
            let eligible = enabled
                && !group.source.is_transparent()
                && mesh.is_some_and(|mesh| !mesh.skinned)
                && source
                    .is_some_and(|source| source.indices().len() / 3 >= IMPOSTOR_MIN_TRIANGLES);
            if let (true, Some(mesh), Some(source)) = (eligible, mesh, source) {
                let key = (group.meshes[0], Some(group.material));
                if let Some(baked) = self.impostors.placed.record(key, Some(&group.source)) {
                    record = baked;
                } else {
                    let (bind_group, variant) = self.materials.binding(Some(group.material));
                    let encoder = encoder.get_or_insert_with(|| {
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("engine/texture-mips"),
                            })
                    });
                    let bounds = source.bounds();
                    let bake = BakeSource {
                        draw: mesh.draw(&self.mesh_arena),
                        extended: mesh.extended,
                        group: bind_group,
                        variant: variant.with_extended(mesh.extended).with_skinned(false),
                        bounds: (bounds.min(), bounds.max()),
                        material: Some(group.source),
                    };
                    if self.impostors.placed.bake(
                        &self.device,
                        encoder,
                        self.materials.mips(),
                        &self.mesh_shader,
                        &mut self.impostors.pipelines,
                        key,
                        &bake,
                        budget,
                    ) {
                        record = self
                            .impostors
                            .placed
                            .record(key, Some(&group.source))
                            .unwrap_or_default();
                    } else {
                        self.impostors.dropped += 1;
                    }
                }
            }
            records.push(record);
        }
        self.impostors.placed.release_targets();
        // The kept shadow map casts impostors as cards: it redraws whole
        // when they change (decision 060).
        if records != self.scatter_draws.impostors {
            self.shadows.dirty_kept_all();
        }
        let draws = &mut self.scatter_draws;
        for (index, (layout, record)) in draws.layouts.iter_mut().zip(&records).enumerate() {
            let impostor = (record.error(), index as u32);
            if layout.impostor != impostor {
                layout.impostor = impostor;
                draws.changed = true;
            }
        }
        draws.impostors = records;
    }

    /// What the draw groups of `terrains` are built from.
    fn scatter_key(&self, terrains: &[engine_core::RenderTerrain]) -> ScatterKey {
        let terrains = terrains
            .iter()
            .map(|terrain| {
                let assets = terrain
                    .assets
                    .iter()
                    .map(|asset| {
                        asset.as_ref().map(|asset| {
                            let lods: Vec<MeshHandle> =
                                asset.lods.iter().map(|(lod, _)| lod.handle).collect();
                            let resident = std::iter::once(&asset.mesh.handle)
                                .chain(&lods)
                                .map(|handle| self.meshes.contains_key(handle))
                                .collect();
                            (
                                asset.mesh.handle,
                                asset.material.handle,
                                asset.material.version,
                                lods,
                                resident,
                                asset.sway.to_bits(),
                            )
                        })
                    })
                    .collect();
                (terrain.id, assets)
            })
            .collect();
        (terrains, self.mesh_layout)
    }

    /// Builds one draw group per asset of `terrains` that draws, and empties
    /// the pool for them: every tile takes new slots.
    fn build_scatter_groups(&mut self, terrains: &[engine_core::RenderTerrain]) {
        let mut groups = Vec::new();
        let mut layouts = Vec::new();
        let mut covers = Vec::new();
        let mut indices = HashMap::new();
        for terrain in terrains {
            for (asset_index, asset) in terrain.assets.iter().enumerate() {
                let Some(asset) = asset else {
                    continue;
                };
                let Some((levels, meshes, errors, camera)) = self.scatter_levels(asset) else {
                    continue;
                };
                let bounds = asset.mesh.mesh.bounds();
                let material = &asset.material.material;
                // Blending materials cast nothing; masked ones test their
                // texels (decision 031).
                let casts = (!material.is_transparent())
                    .then(|| matches!(material.alpha_mode(), engine_core::AlphaMode::Mask { .. }));
                indices.insert((terrain.id, asset_index), groups.len() as u32);
                groups.push(ScatterDraw {
                    meshes,
                    material: asset.material.handle,
                    source: *material,
                    // Swaying objects carry their sway in the flags skinned
                    // slots keep palettes in (decision 058).
                    color: ColorData {
                        color: pack_color(material.color()),
                        surface: pack_surface(material.surface()),
                        flags: asset.sway.to_bits(),
                    },
                    sphere: (
                        (bounds.min() + bounds.max()) * 0.5,
                        (bounds.max() - bounds.min()).length() * 0.5,
                    ),
                    ids: 0,
                    count: 0,
                    casts,
                });
                layouts.push(GroupLayout {
                    levels,
                    camera,
                    errors,
                    impostor: (0.0, 0),
                });
                covers.push(Cover::of(bounds.min(), bounds.max()));
            }
        }
        let draws = &mut self.scatter_draws;
        draws.impostors = vec![ImpostorRecord::default(); groups.len()];
        draws.pool.clear(groups.len());
        draws.groups = groups;
        draws.layouts = layouts;
        draws.covers = covers;
        draws.indices = indices;
        draws.terrains = terrains
            .iter()
            .enumerate()
            .map(|(index, terrain)| (terrain.id, index as u32))
            .collect();
    }

    /// The levels placed asset `asset` draws: its coarser meshes when it has
    /// them, all of which the camera draws, as far as they are resident and
    /// stream like its mesh; otherwise its mesh's automatic levels. `None`
    /// while its mesh is not resident or is skinned.
    fn scatter_levels(&self, asset: &engine_core::RenderPlacementAsset) -> Option<Levels> {
        let mesh = self.meshes.get(&asset.mesh.handle)?;
        if mesh.skinned {
            return None;
        }
        let draw = mesh.draw(&self.mesh_arena);
        let mut levels = vec![(draw.index_count, draw.first_index, draw.base_vertex)];
        let mut meshes = vec![asset.mesh.handle];
        let mut errors = [0.0; LEVELS as usize - 1];
        if asset.lods.is_empty() {
            for &(first_index, index_count, error) in mesh.lods.iter().take(LEVELS as usize - 1) {
                if index_count == 0 {
                    break;
                }
                errors[levels.len() - 1] = error;
                levels.push((
                    index_count,
                    draw.first_index + first_index,
                    draw.base_vertex,
                ));
                meshes.push(asset.mesh.handle);
            }
            let camera = CAMERA_LEVELS.min(levels.len() as u32);
            return Some((levels, meshes, errors, camera));
        }
        for (lod, error) in asset.lods.iter().take(LEVELS as usize - 1) {
            let Some(coarser) = self.meshes.get(&lod.handle) else {
                break;
            };
            if coarser.extended != mesh.extended || coarser.skinned {
                break;
            }
            let coarse = coarser.draw(&self.mesh_arena);
            errors[levels.len() - 1] = *error;
            levels.push((coarse.index_count, coarse.first_index, coarse.base_vertex));
            meshes.push(lod.handle);
        }
        let camera = levels.len() as u32;
        Some((levels, meshes, errors, camera))
    }

    /// The material group and pipeline variant of placed group `draw`.
    fn scatter_binding(&self, draw: &ScatterDraw) -> (&wgpu::BindGroup, Variant) {
        let mesh = self.meshes.get(&draw.meshes[0]);
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
        for (index, draw) in self.scatter_draws.groups.iter().enumerate() {
            if draw.count == 0 {
                continue;
            }
            let (group, variant) = self.scatter_binding(draw);
            let Some(pipeline) = self.mesh_pipelines.variants.get(&(kind, variant)) else {
                continue;
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, group, &[]);
            draws += self.draw_scatter_levels(pass, (index, draw), 0, true);
        }
        // Impostors (decision 048): every group's cards in one draw, over
        // the poses and lights the meshes bound.
        if let Some((targets, lit)) = state.passes.impostor_pass
            && let (Some(pipeline), Some(group)) = (
                self.impostors.pipelines.pipeline(targets, lit),
                self.impostors.placed.group(),
            )
        {
            let (cards, args, offset) = self.scatter.card_source(0);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, group, &[]);
            pass.set_vertex_buffer(0, cards);
            pass.draw_indirect(args, offset);
            draws += 1;
        }
        draws
    }

    /// The engine material of handle `handle` among `snapshot`'s placed
    /// objects' (decision 057).
    pub(super) fn placed_material(
        snapshot: &RenderSnapshot,
        handle: MaterialHandle,
    ) -> Option<engine_core::Material> {
        snapshot
            .terrain
            .terrains
            .iter()
            .flat_map(|terrain| terrain.assets.iter().flatten())
            .find(|asset| asset.material.handle == handle)
            .map(|asset| asset.material.material)
    }

    /// Draws this frame's placed objects culled for shadow map `map` with
    /// the depth passes' caster pipelines `casters`, and their impostor
    /// cards with `cards` (decision 060); the map's view is bound. Returns
    /// the draws.
    pub(super) fn draw_scatter_shadow(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        map: usize,
        casters: &[wgpu::RenderPipeline; 4],
        cards: Option<&wgpu::RenderPipeline>,
    ) -> u32 {
        let (Some(view), Some(poses)) = (self.scatter.map_view(map), self.scatter.shadow_group())
        else {
            return 0;
        };
        pass.set_bind_group(1, poses, &[]);
        let mut bound = None;
        let mut draws = 0;
        for (index, draw) in self.scatter_draws.groups.iter().enumerate() {
            let Some(masked) = draw.casts.filter(|_| draw.count > 0) else {
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
            draws += self.draw_scatter_levels(pass, (index, draw), view, false);
        }
        if let (Some(pipeline), Some(group)) = (cards, self.impostors.placed.group()) {
            let (cards, args, offset) = self.scatter.card_source(view);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(2, group, &[]);
            pass.set_vertex_buffer(0, cards);
            pass.draw_indirect(args, offset);
            draws += 1;
        }
        draws
    }

    /// Draws group `draw` (the `index`th) at each of its levels from
    /// `view`'s lists, binding each level's mesh, with its extra stream
    /// when `extras`; returns the draws.
    fn draw_scatter_levels(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        (index, draw): (usize, &ScatterDraw),
        view: usize,
        extras: bool,
    ) -> u32 {
        let mut draws = 0;
        for (level, handle) in draw.meshes.iter().enumerate() {
            let Some(mesh) = self.meshes.get(handle) else {
                continue;
            };
            let source = mesh.draw(&self.mesh_arena);
            pass.set_vertex_buffer(0, source.vertices.slice(..));
            if extras && let Some(stream) = source.extras.filter(|_| mesh.extended) {
                pass.set_vertex_buffer(2, stream.slice(..));
            }
            pass.set_index_buffer(source.indices.slice(..), wgpu::IndexFormat::Uint32);
            let (ids, args, offset) = self.scatter.draw_source(
                view,
                (index, draw.ids + level as u32 * draw.count),
                level,
            );
            pass.set_vertex_buffer(1, ids);
            pass.draw_indexed_indirect(args, offset);
            draws += 1;
        }
        draws
    }
}

/// The objects terrain `terrain`'s tile `tile` holds for the draw groups of
/// `draws`: those of assets without a group are left out.
fn tile_objects(
    draws: &ScatterDraws,
    terrain: &engine_core::RenderTerrain,
    tile: &engine_core::Tile,
) -> TileObjects {
    let placed = tile.placed();
    let mut objects = (
        Vec::with_capacity(placed.len()),
        Vec::with_capacity(placed.len()),
        Vec::with_capacity(placed.len()),
    );
    // The terrain's index, and for objects standing far away the level
    // whose tiles hold them instead (decision 061).
    let index = draws
        .terrains
        .get(&terrain.id)
        .copied()
        .unwrap_or(TERRAIN_INDEX)
        .min(TERRAIN_INDEX);
    for placed in placed {
        let Some(&group) = draws.indices.get(&(terrain.id, usize::from(placed.asset))) else {
            continue;
        };
        let near = placed.near_level.map_or(0, |level| u32::from(level) + 1);
        let draw = &draws.groups[group as usize];
        let (center, radius) = draw.sphere;
        let model = placed.model(terrain.origin, placed.position[1]);
        objects.0.push(pose_rows(model));
        objects.1.push(draw.color);
        objects.2.push(Record {
            center: model.transform_point3(center).to_array(),
            radius: radius * placed.scale,
            group,
            scale: placed.scale,
            terrain: index | near << NEAR_LEVEL_SHIFT,
            sink: placed.sink,
        });
    }
    objects
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
