//! Placed objects' draws: what terrains' placement layers
//! stand on their tiles, held in the scatter pool's slots as tiles arrive
//! and leave, culled each frame and drawn after terrain in the first opaque
//! phase. Draw groups follow the assets: one per terrain asset, then one
//! per part of it drawing the same objects with its own material.

use engine_core::collections::HashMap;
use std::sync::Arc;

use super::frame::FrameState;
use super::scatter_pool::{ScatterPool, TileObjects};
use super::*;
use crate::debug_view::{DebugViewPass, DrawKind, DrawRecord, bind_record};
use crate::gpu_data::{ColorData, pack_color, pack_surface};
use crate::impostors::{
    BakeGpu, BakePart, BakeSource, IMPOSTOR_MIN_TRIANGLES, ImpostorRecord, Supplied,
};
use crate::occupancy::{Cover, OccupancyFrame};
use crate::scatter::{
    GroupLayout, LAYER_SHIFT, LEVELS, NEAR_LEVEL_SHIFT, NO_LAYER, RANK_MAX, RANK_SHIFT, Record,
    STATE_FRESH, ScatterFrame, SnapSources, TERRAIN_INDEX, TerrainInfo,
};
use engine_core::glam::{Mat4, Vec3};
use engine_core::{Tile, TileKey};

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
    // Each group's impostor, zeroed without one, and whether
    // the groups' layouts changed since they last uploaded.
    impostors: Vec<ImpostorRecord>,
    changed: bool,
    // Each group's cover.
    covers: Vec<Cover>,
}

impl ScatterDraws {
    /// Each draw group's impostor record in the placed objects' atlas,
    /// zeroed without one.
    pub(super) fn impostors(&self) -> &[ImpostorRecord] {
        &self.impostors
    }

    /// Resident objects standing far away.
    pub(super) fn far_objects(&self) -> u32 {
        self.pool.far
    }

    /// Each draw group's cover, which the occupancy map draws.
    pub(super) fn covers(&self) -> &[Cover] {
        &self.covers
    }
}

/// One draw group of placed objects: each level's mesh, its material, its
/// objects' color and bounding sphere (mesh space, around its parts' meshes
/// and its shadow proxy too), the group whose objects it draws (its own, or
/// for a part its asset's) and which of the asset's parts it draws, its
/// live objects, whether it casts shadows and tests its material's texels
/// doing so (parts cast none, and shadow proxies never test), and the mesh
/// of each level shadow maps draw of it: its shadow proxy alone where it
/// has one, else its levels'.
struct ScatterDraw {
    meshes: Vec<MeshHandle>,
    material: MaterialHandle,
    source: engine_core::Material,
    color: ColorData,
    sphere: (Vec3, f32),
    objects: u32,
    part: Option<usize>,
    count: u32,
    casts: Option<bool>,
    casters: Vec<MeshHandle>,
}

/// An asset as its groups were built: its mesh, material and version, its
/// coarser meshes, which of all its meshes were resident, its sway's bits,
/// its parts, and its shadow proxy.
type AssetKey = (
    MeshHandle,
    MaterialHandle,
    u64,
    Vec<MeshHandle>,
    Vec<bool>,
    u32,
    Vec<PartKey>,
    Option<MeshHandle>,
);

/// A part as its group was built: its material and version, and its
/// meshes.
type PartKey = (MaterialHandle, u64, Vec<MeshHandle>);

/// What the draw groups were built from: each terrain's identity and
/// assets, and the meshes' layout.
type ScatterKey = (Vec<(u64, Vec<Option<AssetKey>>)>, u64);

/// A placed asset's levels as drawn: each one's index count, first index
/// and base vertex, and those of the triangles shadow maps draw of it,
/// its mesh, the coarser ones' errors, and how many the camera draws.
type Levels = (
    Vec<(u32, u32, i32)>,
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
    /// culling, with impostors when `impostors`, advancing how the camera
    /// shows objects by this frame's fades. `None` when no object draws.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(super) fn prepare_scatter(
        &mut self,
        snapshot: &RenderSnapshot,
        kind: PassKind,
        impostors: bool,
    ) -> Option<Ready<ScatterFrame>> {
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
            let current: HashMap<TileKey, &Arc<Tile>> = terrain
                .tiles
                .iter()
                .filter(|tile| !tile.placed().is_empty())
                .map(|tile| (tile.key(), tile))
                .collect();
            // A tile replaced with the same objects (its water made again)
            // keeps them as they stand.
            draws.pool.retain(terrain.id, |key, held| {
                current.get(&key).is_some_and(|tile| {
                    Arc::ptr_eq(tile, held) || std::ptr::eq(tile.placed(), held.placed())
                })
            });
            for tile in terrain.tiles.iter() {
                if tile.placed().is_empty() || draws.pool.contains(terrain.id, tile.key()) {
                    continue;
                }
                let objects = tile_objects(draws, terrain, tile);
                draws
                    .pool
                    .insert((terrain.id, tile.key()), Arc::clone(tile), objects);
            }
        }
        // Objects stand on the ground's true surface, of the heights
        // resident this frame, and take its colors there.
        let infos: Vec<TerrainInfo> = terrains
            .iter()
            .enumerate()
            .map(|(index, terrain)| {
                let (directory, palette, _, colors) = self.terrain.cover_terrain(index);
                TerrainInfo {
                    origin: terrain.origin.extend(terrain.pyramid.spacing(0)).to_array(),
                    levels: [directory, terrain.pyramid.levels(), palette, colors],
                }
            })
            .collect();
        let thins = scatter_thinning(snapshot);
        self.features
            .scatter
            .get_mut()
            .set_thinning(&self.device, &self.queue, &thins);
        let draws = &mut self.scatter_draws;
        let (dirty, counts_changed) = draws.pool.take_changes();
        // The kept shadow map, its levels, the near cascades' caches and the
        // occupancy maps redraw where objects came or went.
        for (min, max) in draws.pool.take_moved() {
            self.shadows.dirty_kept(min, max);
            self.features.occupancy.get_mut().dirty_box(min, max);
        }
        let written = self.features.scatter.get_mut().write_slots(
            &self.device,
            &self.queue,
            (
                &draws.pool.poses,
                &draws.pool.colors,
                &draws.pool.records,
                &draws.pool.grounds,
            ),
            dirty,
        );
        if let Some(slots) = written {
            draws.pool.uploaded(slots);
        }
        if counts_changed || std::mem::take(&mut draws.changed) {
            // A part's group draws the objects of its asset's.
            let counts: Vec<u32> = draws
                .groups
                .iter()
                .map(|group| draws.pool.counts[group.objects as usize])
                .collect();
            for (group, &count) in draws.groups.iter_mut().zip(&counts) {
                group.count = count;
            }
            self.features.scatter.get_mut().set_groups(
                &self.device,
                &self.queue,
                &draws.layouts,
                &counts,
            );
        }
        // Shadows cache the objects placed near the camera while any are
        // resident.
        self.shadows
            .cache_placed(self.features.scatter.get().resident != 0);
        let sources = self.terrain.cover_sources().map(|sources| SnapSources {
            heights: sources.heights,
            control: sources.control,
            directory: sources.directory,
            palette: sources.palette,
            macro_maps: sources.macro_maps,
            generation: (
                sources.generation.0,
                sources.generation.1,
                sources.generation.3,
            ),
            revision: sources.revision,
            terrains: &infos,
        });
        // On the ground as drawn, which morphs by the distance from the
        // detail eye.
        self.features.scatter.get_mut().prepare_snap(
            &self.device,
            &self.queue,
            sources,
            snapshot.detail_eye,
        );
        // Each group's pipelines, and those its fading objects draw with:
        // masked groups' depth pre-pass too.
        let variants: Vec<Variant> = self
            .scatter_draws
            .groups
            .iter()
            .flat_map(|draw| {
                let variant = self.scatter_binding(draw).1;
                [variant, variant.with_fading(true)]
            })
            .collect();
        for variant in variants {
            self.ensure_opaque_variant(kind, variant);
        }
        let level_of_detail = snapshot.environment.level_of_detail;
        let pixel_error = level_of_detail
            .enabled
            .then_some(level_of_detail.pixel_error);
        // How much of their way the camera's switches fade this frame.
        let fade = self.switch_fades.share;
        let height = self.display.size().height as f32;
        // Frames keeping still casters cast placed objects into the kept
        // map, not the moving one, and into the near cascades' caches,
        // which the cascades are composed from, not into the cascades.
        let keeps_still = self.shadows.keeps_still();
        let caches = self.shadows.caches_near();
        let maps: Vec<(usize, engine_core::glam::Mat4, u32)> = self
            .shadows
            .drawn_maps()
            .iter()
            .filter(|&&(map, _)| {
                let moving = map == crate::shadows::MOVING_MAP;
                !(keeps_still && moving) && !(caches && map < crate::shadows::MOVING_MAP)
            })
            .map(|&(map, clip)| (map, clip, self.shadows.drawn_map_texels(map)))
            .collect();
        let impostors = impostors.then_some(level_of_detail.impostor_pixels);
        let (device, queue, shadows) = (&self.device, &self.queue, &self.shadows);
        // Objects switch how they show, and retire, by their distance from
        // the detail eye; the views cull by their own planes.
        self.features.scatter.prepare(|scatter| {
            scatter
                .prepare(
                    device,
                    queue,
                    (snapshot.view_projection, snapshot.detail_eye),
                    (pixel_error, height, impostors, fade),
                    (
                        &maps,
                        shadows.kept_refresh(),
                        shadows.level_refresh(),
                        shadows.near_refreshes(),
                    ),
                )
                .then_some(ScatterFrame)
        })
    }

    /// What placed objects cover around the camera: the maps this frame
    /// redraws, from the placed objects' slots.
    pub(super) fn prepare_occupancy(&mut self) -> Option<Ready<OccupancyFrame>> {
        let device = &self.device;
        let slots = self.features.scatter.get().slot_buffers();
        self.features
            .occupancy
            .prepare(|occupancy| occupancy.prepare_draws(device, slots))
    }

    /// The impostors of placed draw groups on GPU-presented frames with
    /// impostors: each asset's, its parts drawn into it, supplied into the
    /// placed objects' atlas where missing (baked into `encoder` after its
    /// materials' levels where no host holds its record), zeroed where its
    /// material blends, its mesh is too small, it is still coming or the
    /// budget the plan batches' leave is spent; parts' groups take none.
    /// The groups' layouts take their errors and records.
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
        let groups = &self.scatter_draws.groups;
        let mut records = Vec::with_capacity(groups.len());
        for (index, group) in groups.iter().enumerate() {
            let mut record = ImpostorRecord::default();
            let mesh = self.meshes.get(&group.meshes[0]);
            let asset = snapshot
                .terrain
                .terrains
                .iter()
                .flat_map(|terrain| terrain.assets.iter().flatten())
                .find(|asset| asset.mesh.handle == group.meshes[0]);
            let eligible = enabled
                && group.part.is_none()
                && !group.source.is_transparent()
                && mesh.is_some_and(|mesh| !mesh.skinned)
                && asset.is_some_and(|asset| {
                    asset.mesh.mesh.indices().len() / 3 >= IMPOSTOR_MIN_TRIANGLES
                });
            if let (true, Some(mesh), Some(asset)) = (eligible, mesh, asset) {
                let key = (group.meshes[0], Some(group.material));
                record = match self.impostors.placed.record(key, Some(&group.source)) {
                    Some(resident) => resident,
                    None => {
                        let textures = |handle: TextureHandle| {
                            asset
                                .textures
                                .iter()
                                .find(|texture| texture.handle == handle)
                                .map(|texture| &*texture.texture)
                        };
                        let contents = &mut self.impostors.contents;
                        let content = (
                            contents.mesh(group.meshes[0], &asset.mesh.mesh),
                            contents.material(Some(&group.source), textures),
                        );
                        let bounds = asset.mesh.mesh.bounds();
                        let (parts, content, (low, high)) = bake_parts(
                            (groups, index, asset),
                            (&self.meshes, &self.mesh_arena, &self.materials),
                            contents,
                            (content, (bounds.min(), bounds.max())),
                        );
                        let (bind_group, variant) = self.materials.binding(Some(group.material));
                        let bake = BakeSource {
                            draw: mesh.draw(&self.mesh_arena),
                            extended: mesh.extended,
                            group: bind_group,
                            variant: variant.with_extended(mesh.extended).with_skinned(false),
                            bounds: (low, high),
                            material: Some(group.source),
                            parts,
                        };
                        let gpu = BakeGpu {
                            device: &self.device,
                            queue: &self.queue,
                            encoder: &mut *encoder,
                            mips: self.materials.mips(),
                            shader: &self.mesh_shader,
                            pipelines: &mut self.impostors.pipelines,
                        };
                        let supplied = self
                            .impostors
                            .placed
                            .supply(gpu, key, &bake, content, budget);
                        self.impostors.dropped += u32::from(matches!(supplied, Supplied::Refused));
                        supplied.record()
                    }
                };
            }
            records.push(record);
        }
        self.impostors.placed.release_targets();
        // The kept shadow maps and caches cast impostors as cards: they
        // redraw whole when those change.
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
                            let resident = asset
                                .meshes()
                                .map(|mesh| self.meshes.contains_key(&mesh.handle))
                                .collect();
                            let parts = asset
                                .parts
                                .iter()
                                .map(|part| {
                                    (
                                        part.material.handle,
                                        part.material.version,
                                        part.meshes.iter().map(|mesh| mesh.handle).collect(),
                                    )
                                })
                                .collect();
                            (
                                asset.mesh.handle,
                                asset.material.handle,
                                asset.material.version,
                                lods,
                                resident,
                                asset.sway.to_bits(),
                                parts,
                                asset.shadow.as_ref().map(|proxy| proxy.handle),
                            )
                        })
                    })
                    .collect();
                (terrain.id, assets)
            })
            .collect();
        (terrains, self.mesh_layout)
    }

    /// Builds one draw group per asset of `terrains` that draws, casting
    /// its shadow proxy alone while that is resident, and one per part of
    /// it whose finest mesh is resident, right after it, and empties the
    /// pool for them: every tile takes new slots.
    fn build_scatter_groups(&mut self, terrains: &[engine_core::RenderTerrain]) {
        let mut groups = Vec::new();
        let mut layouts = Vec::new();
        let mut covers = Vec::new();
        let mut indices = HashMap::default();
        for terrain in terrains {
            for (asset_index, asset) in terrain.assets.iter().enumerate() {
                let Some(asset) = asset else {
                    continue;
                };
                let Some((levels, casters, meshes, errors, camera)) = self.scatter_levels(asset)
                else {
                    continue;
                };
                let parts: Vec<(usize, Vec<(u32, u32, i32)>, Vec<MeshHandle>)> = asset
                    .parts
                    .iter()
                    .enumerate()
                    .filter_map(|(part, rendered)| {
                        let (levels, meshes) = self.part_levels(rendered, levels.len())?;
                        Some((part, levels, meshes))
                    })
                    .collect();
                let bounds = asset.mesh.mesh.bounds();
                // Its shadow proxy, while resident, is the one level its
                // objects cast at.
                let proxy = asset
                    .shadow
                    .as_ref()
                    .and_then(|mesh| Some((mesh, self.proxy_caster(mesh)?)));
                // Its bounding sphere holds its parts' meshes and its shadow
                // proxy too.
                let (low, high) = parts
                    .iter()
                    .map(|&(part, _, _)| asset.parts[part].meshes[0].mesh.bounds())
                    .chain(proxy.map(|(mesh, _)| mesh.mesh.bounds()))
                    .fold((bounds.min(), bounds.max()), |(low, high), held| {
                        (low.min(held.min()), high.max(held.max()))
                    });
                let material = &asset.material.material;
                // Blending materials cast nothing and masked ones test their
                // texels; a shadow proxy casts depth alone in place of the
                // levels, whatever the material.
                let (casts, casters, caster_meshes) = match proxy {
                    Some((mesh, range)) => (Some(false), vec![range], vec![mesh.handle]),
                    None if material.is_transparent() => (None, Vec::new(), Vec::new()),
                    None => (
                        Some(matches!(material.alpha_mode(), engine_core::AlphaMode::Mask { .. })),
                        casters,
                        meshes.clone(),
                    ),
                };
                let objects = groups.len() as u32;
                indices.insert((terrain.id, asset_index), objects);
                // Swaying objects carry their sway in the flags skinned slots
                // keep palettes in.
                let color = ColorData {
                    color: pack_color(material.color()),
                    surface: pack_surface(material.surface()),
                    flags: asset.sway.to_bits(),
                };
                let sphere = ((low + high) * 0.5, (high - low).length() * 0.5);
                let cover = Cover::of(bounds.min(), bounds.max());
                groups.push(ScatterDraw {
                    meshes,
                    material: asset.material.handle,
                    source: *material,
                    color,
                    sphere,
                    objects,
                    part: None,
                    count: 0,
                    casts,
                    casters: caster_meshes,
                });
                layouts.push(GroupLayout {
                    levels,
                    casters,
                    camera,
                    errors,
                    impostor: (0.0, 0),
                    parts: parts.len() as u32,
                    part: false,
                });
                covers.push(cover);
                for (part, part_levels, part_meshes) in parts {
                    let rendered = &asset.parts[part];
                    layouts.push(GroupLayout {
                        casters: Vec::new(),
                        camera: part_levels.len() as u32,
                        levels: part_levels,
                        errors,
                        impostor: (0.0, groups.len() as u32),
                        parts: 0,
                        part: true,
                    });
                    groups.push(ScatterDraw {
                        meshes: part_meshes,
                        material: rendered.material.handle,
                        source: rendered.material.material,
                        color,
                        sphere,
                        objects,
                        part: Some(part),
                        count: 0,
                        casts: None,
                        casters: Vec::new(),
                    });
                    covers.push(cover);
                }
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

    /// The levels placed asset `asset` draws, all of which the camera draws:
    /// its coarser meshes when it has them, as far as they are resident and
    /// stream like its mesh; otherwise its mesh's cooked levels, whose
    /// measured errors hold even where a level prunes whole parts. `None`
    /// while its mesh is not resident or is skinned.
    fn scatter_levels(&self, asset: &engine_core::RenderPlacementAsset) -> Option<Levels> {
        let mesh = self.meshes.get(&asset.mesh.handle)?;
        if mesh.skinned {
            return None;
        }
        let draw = mesh.draw(&self.mesh_arena);
        // Level `level` of `mesh` drawn from `draw`, whose own triangles
        // are `own`, as the camera and as shadow maps draw it.
        let level = |mesh: &GpuMesh, draw: &MeshDraw<'_>, level: usize, own: (u32, u32)| {
            let (first, count) = mesh.caster_range(level, own);
            (
                (own.1, draw.first_index + own.0, draw.base_vertex),
                (count, draw.first_index + first, draw.base_vertex),
            )
        };
        let (finest, finest_casters) = level(mesh, &draw, 0, (0, draw.index_count));
        let mut levels = vec![finest];
        let mut casters = vec![finest_casters];
        let mut meshes = vec![asset.mesh.handle];
        let mut errors = [0.0; LEVELS as usize - 1];
        if asset.lods.is_empty() {
            for &(first_index, index_count, error) in mesh.lods.iter().take(LEVELS as usize - 1) {
                if index_count == 0 {
                    break;
                }
                errors[levels.len() - 1] = error;
                let (drawn, cast) = level(mesh, &draw, levels.len(), (first_index, index_count));
                levels.push(drawn);
                casters.push(cast);
                meshes.push(asset.mesh.handle);
            }
            let camera = levels.len() as u32;
            return Some((levels, casters, meshes, errors, camera));
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
            let (drawn, cast) = level(coarser, &coarse, 0, (0, coarse.index_count));
            levels.push(drawn);
            casters.push(cast);
            meshes.push(lod.handle);
        }
        let camera = levels.len() as u32;
        Some((levels, casters, meshes, errors, camera))
    }

    /// The levels part `part` of an asset drawing `levels` levels draws:
    /// each of its meshes' own triangles, finest first, as far as they are
    /// resident and stream like its first; `None` while its first is not
    /// resident or is skinned.
    fn part_levels(
        &self,
        part: &engine_core::RenderPlacementPart,
        levels: usize,
    ) -> Option<(Vec<(u32, u32, i32)>, Vec<MeshHandle>)> {
        let first = self.meshes.get(&part.meshes.first()?.handle)?;
        if first.skinned {
            return None;
        }
        let mut ranges = Vec::new();
        let mut meshes = Vec::new();
        for rendered in part.meshes.iter().take(levels) {
            let Some(mesh) = self.meshes.get(&rendered.handle) else {
                break;
            };
            if mesh.extended != first.extended || mesh.skinned {
                break;
            }
            let draw = mesh.draw(&self.mesh_arena);
            ranges.push((draw.index_count, draw.first_index, draw.base_vertex));
            meshes.push(rendered.handle);
        }
        Some((ranges, meshes))
    }

    /// The index count, first index and base vertex of the triangles shadow
    /// maps draw of shadow proxy `proxy`; `None` while it is not resident or
    /// is skinned.
    fn proxy_caster(&self, proxy: &engine_core::RenderMesh) -> Option<(u32, u32, i32)> {
        let mesh = self.meshes.get(&proxy.handle)?;
        if mesh.skinned {
            return None;
        }
        let draw = mesh.draw(&self.mesh_arena);
        let (first, count) = mesh.caster_range(0, (0, draw.index_count));
        Some((count, draw.first_index + first, draw.base_vertex))
    }

    /// The material group and pipeline variant of placed group `draw`,
    /// blending into the ground where its material says.
    fn scatter_binding(&self, draw: &ScatterDraw) -> (&wgpu::BindGroup, Variant) {
        let mesh = self.meshes.get(&draw.meshes[0]);
        let (group, variant) = self.materials.binding(Some(draw.material));
        let variant = variant
            .with_extended(mesh.is_some_and(|mesh| mesh.extended))
            .with_skinned(false)
            .with_grounding(draw.source.grounding().is_some());
        (group, variant)
    }

    /// Draws this frame's placed objects into the first opaque phase: their
    /// colors, masked groups' where their depth pre-pass left their depths,
    /// and their impostors' cards; or with `prepass` the masked groups' depth
    /// pre-pass, first in the phase's opaque pass, from the same lists, at
    /// the same levels and with the same fades as their colors. Returns the
    /// draws.
    pub(super) fn draw_scatter(
        &self,
        state: &FrameState<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        prepass: bool,
    ) -> u32 {
        if !state.scatter {
            return 0;
        }
        let Some(groups) = self.features.scatter.get().pose_groups() else {
            return 0;
        };
        let kind = state.passes.opaque_kind;
        let temporal =
            state.packing.temporal_frame.is_some() && self.features.temporal.get().is_some();
        pass.set_bind_group(0, self.camera.group(), &[]);
        // The pre-pass reads no lights.
        if kind.lit
            && !prepass
            && let Some(lights) = self.features.lights.get()
        {
            pass.set_bind_group(3, lights.fragment_group(), &[]);
        }
        pass.set_bind_group(2, &groups[usize::from(temporal)], &[]);
        // Debug views' draws read their records: placed objects cast into
        // the kept map in frames keeping still casters.
        let debug = self
            .debug_records(state)
            .map(|records| (records, state.shape.shadows, self.shadows.keeps_still()));
        let mut draws = 0;
        // The camera's lists, then those of objects fading between how the
        // camera shows them, at the levels the camera draws.
        let fading = self.features.scatter.get().fading_view();
        for (view, fades) in [(0, false), (fading, true)] {
            for (index, draw) in self.scatter_draws.groups.iter().enumerate() {
                if draw.count == 0 {
                    continue;
                }
                let (group, variant) = self.scatter_binding(draw);
                let Some(pipeline) =
                    self.opaque_pipeline(kind, variant.with_fading(fades), prepass)
                else {
                    continue;
                };
                pass.set_pipeline(pipeline);
                pass.set_bind_group(1, group, &[]);
                draws += self.draw_scatter_levels(pass, (index, draw), view, debug);
            }
        }
        if prepass {
            return draws;
        }
        // Impostors: every group's cards in one draw, over
        // the poses and lights the meshes bound.
        if let Some((targets, lit)) = state.passes.impostor_pass
            && let (Some(pipeline), Some(group)) = (
                self.impostors.pipelines.pipeline(targets, lit),
                self.impostors.placed.group(),
            )
            && let Some((cards, args, offset)) = self.features.scatter.get().card_source(0)
        {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, group, &[]);
            pass.set_vertex_buffer(0, cards);
            if let Some((records, casts, _)) = debug {
                let record = DrawRecord::of(DrawKind::PlacedCard, 0, casts);
                bind_record(pass, Some(records), record);
            }
            pass.draw_indirect(args, offset);
            draws += 1;
        }
        draws
    }

    /// The engine material of handle `handle` among `snapshot`'s placed
    /// objects' and their parts'.
    pub(super) fn placed_material(
        snapshot: &RenderSnapshot,
        handle: MaterialHandle,
    ) -> Option<engine_core::Material> {
        snapshot
            .terrain
            .terrains
            .iter()
            .flat_map(|terrain| terrain.assets.iter().flatten())
            .flat_map(engine_core::RenderPlacementAsset::materials)
            .find(|material| material.handle == handle)
            .map(|material| material.material)
    }

    /// Draws this frame's placed objects culled for shadow map `map` at the
    /// levels their groups cast at, each group's shadow proxy alone where it
    /// has one, with the depth passes' caster pipelines `casters`, and their
    /// impostor cards with `cards`; the map's view is bound. Returns the
    /// draws.
    pub(super) fn draw_scatter_shadow(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        map: usize,
        casters: &[wgpu::RenderPipeline; 4],
        cards: Option<&wgpu::RenderPipeline>,
    ) -> u32 {
        let (Some(view), Some(poses)) = (
            self.features.scatter.get().map_view(map),
            self.features.scatter.get().shadow_group(),
        ) else {
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
            draws += self.draw_scatter_casters(pass, (index, draw), view);
        }
        if let (Some(pipeline), Some(group)) = (cards, self.impostors.placed.group())
            && let Some((cards, args, offset)) = self.features.scatter.get().card_source(view)
        {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(2, group, &[]);
            pass.set_vertex_buffer(0, cards);
            pass.draw_indirect(args, offset);
            draws += 1;
        }
        draws
    }

    /// Draws group `draw` (the `index`th) at each of its levels from
    /// `view`'s lists (the camera's, or its fading objects'), binding each
    /// level's mesh with its extra stream and, in debug views, each level's
    /// record of `debug`'s records, the group casting sun shadows where they
    /// say and its casters kept; returns the draws.
    fn draw_scatter_levels(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        (index, draw): (usize, &ScatterDraw),
        view: usize,
        debug: Option<(&DebugViewPass, bool, bool)>,
    ) -> u32 {
        let mut draws = 0;
        for (level, handle) in draw.meshes.iter().enumerate() {
            let Some(mesh) = self.meshes.get(handle) else {
                continue;
            };
            let Some((ids, args, offset)) =
                self.features.scatter.get().draw_source(view, index, level)
            else {
                continue;
            };
            let source = mesh.draw(&self.mesh_arena);
            pass.set_vertex_buffer(0, source.vertices.slice(..));
            if let Some(stream) = source.extras.filter(|_| mesh.extended) {
                pass.set_vertex_buffer(2, stream.slice(..));
            }
            pass.set_index_buffer(source.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.set_vertex_buffer(1, ids);
            if let Some((records, casts, kept)) = debug {
                let casts = casts && draw.casts.is_some();
                let record = DrawRecord {
                    kept: casts && kept,
                    ..DrawRecord::of(DrawKind::Placed, level, casts)
                };
                bind_record(pass, Some(records), record);
            }
            pass.draw_indexed_indirect(args, offset);
            draws += 1;
        }
        draws
    }

    /// Draws group `draw` (the `index`th) from shadow map view `view`'s
    /// lists, at each level it casts at with that level's mesh; returns the
    /// draws.
    fn draw_scatter_casters(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        (index, draw): (usize, &ScatterDraw),
        view: usize,
    ) -> u32 {
        let mut draws = 0;
        for (level, handle) in draw.casters.iter().enumerate() {
            let Some(mesh) = self.meshes.get(handle) else {
                continue;
            };
            let Some((ids, args, offset)) =
                self.features.scatter.get().draw_source(view, index, level)
            else {
                continue;
            };
            let source = mesh.draw(&self.mesh_arena);
            pass.set_vertex_buffer(0, source.vertices.slice(..));
            pass.set_index_buffer(source.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.set_vertex_buffer(1, ids);
            pass.draw_indexed_indirect(args, offset);
            draws += 1;
        }
        draws
    }
}

/// How each of `snapshot`'s terrains' placement layers thin with distance
/// this frame, `THIN_LAYERS` entries a terrain: their tiles' reaches follow
/// the view's levels of detail, and pinned levels stay resident
/// everywhere (`scatter::thins_on`), as the ground taking on what they thin
/// reads them (`terrain::tables`).
fn scatter_thinning(snapshot: &RenderSnapshot) -> Vec<[f32; 4]> {
    let terrains = &snapshot.terrain.terrains;
    let pinned = snapshot.environment.terrain.sanitized().pinned_levels;
    let mut thins = vec![[0.0_f32; 4]; terrains.len() * crate::scatter::THIN_LAYERS];
    for (index, terrain) in terrains.iter().enumerate() {
        let levels = terrain.pyramid.levels();
        let reach = |level: u8| {
            if crate::scatter::thins_on(level, levels, pinned) {
                terrain
                    .reach
                    .get(usize::from(level))
                    .copied()
                    .unwrap_or(0.0)
            } else {
                0.0
            }
        };
        for (layer, placement) in terrain
            .layers
            .iter()
            .enumerate()
            .take(crate::scatter::THIN_LAYERS)
        {
            let (far, share) = placement.far;
            thins[index * crate::scatter::THIN_LAYERS + layer] =
                [share, reach(placement.level), reach(far), 1.0];
        }
    }
    thins
}

/// The parts `asset`, the asset of `groups`' draw group `index`, draws with
/// its objects, the groups after it, as its impostor's bake draws them over
/// its mesh: each at its finest, from `meshes` in `arena` with `materials`'
/// groups; and the impostor's `content` and `bounds` grown by theirs, whose
/// contents `contents` hashes.
fn bake_parts<'a>(
    (groups, index, asset): (&[ScatterDraw], usize, &engine_core::RenderPlacementAsset),
    (meshes, arena, materials): (
        &'a BTreeMap<MeshHandle, GpuMesh>,
        &'a MeshArena,
        &'a MaterialStore,
    ),
    contents: &mut Contents,
    (mut content, (mut low, mut high)): ((u128, u128), (Vec3, Vec3)),
) -> (Vec<BakePart<'a>>, (u128, u128), (Vec3, Vec3)) {
    let textures = |handle: TextureHandle| {
        asset
            .textures
            .iter()
            .find(|texture| texture.handle == handle)
            .map(|texture| &*texture.texture)
    };
    let mut parts = Vec::new();
    for drawn in groups[index + 1..]
        .iter()
        .take_while(|drawn| drawn.objects == index as u32)
    {
        let (Some(part), Some(mesh)) = (drawn.part, meshes.get(&drawn.meshes[0])) else {
            continue;
        };
        let Some((rendered, finest)) = asset
            .parts
            .get(part)
            .and_then(|rendered| Some((rendered, &rendered.meshes.first()?.mesh)))
        else {
            continue;
        };
        content = (
            combined(content.0, contents.mesh(drawn.meshes[0], finest)),
            combined(
                content.1,
                contents.material(Some(&rendered.material.material), textures),
            ),
        );
        let bounds = finest.bounds();
        low = low.min(bounds.min());
        high = high.max(bounds.max());
        let (group, variant) = materials.binding(Some(drawn.material));
        parts.push(BakePart {
            draw: mesh.draw(arena),
            extended: mesh.extended,
            group,
            variant: variant.with_extended(mesh.extended).with_skinned(false),
        });
    }
    (parts, content, (low, high))
}

/// The content of an impostor of what `first` and `second` hold together.
fn combined(first: u128, second: u128) -> u128 {
    let mut bytes = [0_u8; 32];
    bytes[..16].copy_from_slice(&first.to_le_bytes());
    bytes[16..].copy_from_slice(&second.to_le_bytes());
    engine_core::content_hash(&bytes)
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
        Vec::with_capacity(placed.len()),
    );
    // The terrain's index, and for objects standing far away the level
    // whose tiles hold them instead.
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
        // Its layer and its rank under the density, in ten
        // bits.
        let layer = u32::from(placed.layer).min(NO_LAYER);
        let rank = (engine_core::math::clamp(placed.rank, 0.0, 1.0) * RANK_MAX).round() as u32;
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
            terrain: index | layer << LAYER_SHIFT | near << NEAR_LEVEL_SHIFT | rank << RANK_SHIFT,
            sink: placed.sink,
            state: STATE_FRESH,
            pad: [0; 3],
        });
        objects.3.push(crate::gpu_data::ground_data(
            draw.source.grounding(),
            placed.ground,
            placed.sink,
            placed.scale,
        ));
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
