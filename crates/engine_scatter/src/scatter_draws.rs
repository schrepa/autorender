use engine_core::collections::HashMap;
use engine_core::{MaterialHandle, MeshHandle, RenderSnapshot, TextureHandle};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::gpu_layouts::Cover;
use crate::scatter::{
    GroupLayout, LAYER_SHIFT, LEVELS, NEAR_LEVEL_SHIFT, NO_LAYER, RANK_MAX, RANK_SHIFT,
    STATE_FRESH, ScatterPass, SnapSources, TERRAIN_INDEX,
};
use crate::scatter_pool::{ScatterPool, TileObjects};
use engine_core::glam::{Mat4, Vec3};
use engine_faces::heightfield::{HeightfieldFace, TerrainInfo};
use engine_faces::placed::{PlacedRecord, Placements};
use engine_faces::shadows::ShadowCastersFace;
use engine_render::debug_targets::{DrawKind, DrawRecord, RECORD_SLOT};
use engine_render::entries::PassKind;
use engine_render::gpu_data::{ColorData, Variant, pack_color, pack_surface};
use engine_render::gpu_layouts::impostors::ImpostorRecord;
use engine_render::graph::Targets;
use engine_render::impostors::{
    AtlasShape, BakeGpu, BakePart, BakeSource, Contents, IMPOSTOR_MIN_TRIANGLES, ImpostorAtlas,
    ImpostorPipelines, Supplied,
};
use engine_render::materials::MaterialStore;
use engine_render::mesh_arena::{MeshArena, MeshDraw};
use engine_render::parts::{Prelude, Step, View};
use engine_render::renderer::{Core, GpuMesh};
use engine_terrain::{Tile, TileKey};

type PartLevels = (Vec<(u32, u32, i32)>, Vec<MeshHandle>);

const PLACED_FRAMES: u32 = 12;

fn placed_shape(pixels: f32) -> AtlasShape {
    let texels = (pixels.ceil() as u32).clamp(2, 1 << 15);
    AtlasShape {
        frames: PLACED_FRAMES,
        texels: texels.next_power_of_two(),
        hemisphere: true,
    }
}

pub(crate) struct PlacedImpostors {
    pub(crate) atlas: ImpostorAtlas,
    pipelines: ImpostorPipelines,
    contents: Contents,
    pub(crate) dropped: u32,
    card: Option<wgpu::RenderPipeline>,
}

impl PlacedImpostors {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            atlas: ImpostorAtlas::new(
                device,
                placed_shape(engine_core::LevelOfDetail::default().impostor_pixels),
            ),
            pipelines: ImpostorPipelines::new(device),
            contents: Contents::default(),
            dropped: 0,
            card: None,
        }
    }

    fn prepare_cards(
        &mut self,
        core: &mut Core,
        (gpu_mode, kind): (bool, PassKind),
        records: &[ImpostorRecord],
    ) -> bool {
        self.card = None;
        let shown = gpu_mode && records.iter().any(|record| record.sphere[3] > 0.0);
        let Some(targets) = kind.targets().filter(|_| shown) else {
            return false;
        };
        if !self
            .atlas
            .prepare(&core.device, &core.queue, &self.pipelines, records)
        {
            return false;
        }
        self.card = core.ensure_card_pipeline(targets).cloned();
        self.card.is_some()
    }

    fn card(&self) -> Option<&wgpu::RenderPipeline> {
        self.card.as_ref()
    }
}

#[derive(Default)]
pub(crate) struct ScatterDraws {
    key: Option<ScatterKey>,
    groups: Vec<ScatterDraw>,
    layouts: Vec<GroupLayout>,
    indices: HashMap<(u64, usize), u32>,
    terrains: HashMap<u64, u32>,
    pool: ScatterPool,
    impostors: Vec<ImpostorRecord>,
    changed: bool,
    covers: Vec<Cover>,
    colors: Vec<[u32; 2]>,
}

impl ScatterDraws {
    pub(crate) fn impostors(&self) -> &[ImpostorRecord] {
        &self.impostors
    }

    pub(crate) fn far_objects(&self) -> u32 {
        self.pool.far
    }

    pub(crate) fn covers(&self) -> &[Cover] {
        &self.covers
    }

    pub(crate) fn placements(&self) -> Placements<'_> {
        Placements {
            tiles: self.pool.tiles(),
            records: bytemuck::cast_slice(&self.pool.records),
            poses: bytemuck::cast_slice(&self.pool.poses),
            impostors: bytemuck::cast_slice(&self.impostors),
            colors: &self.colors,
            revision: self.pool.revision(),
        }
    }
}

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

type PartKey = (MaterialHandle, u64, Vec<MeshHandle>);

type ScatterKey = (Vec<(u64, Vec<Option<AssetKey>>)>, u64);

type Levels = (
    Vec<(u32, u32, i32)>,
    Vec<(u32, u32, i32)>,
    Vec<MeshHandle>,
    [f32; LEVELS as usize - 1],
    u32,
);

impl ScatterDraws {
    #[expect(clippy::too_many_lines, reason = "one pass over the placed batches")]
    pub(crate) fn prepare(
        &mut self,
        scatter: &mut ScatterPass,
        cx: &mut Step<'_>,
        moved: &mut Vec<(Vec3, Vec3)>,
    ) -> bool {
        let Some(passes) = cx.frame.passes else {
            return false;
        };
        let snapshot = cx.frame.snapshot;
        let terrains = &engine_terrain::RenderTerrains::of(snapshot).terrains;
        if terrains.iter().any(|terrain| !terrain.assets.is_empty()) {
            scatter.warm(&cx.core.device);
        }
        let key = scatter_key(cx.core, terrains);
        if self.key.as_ref() != Some(&key) || self.pool.fragmented() {
            self.build_groups(cx.core, terrains);
            self.key = Some(key);
        }
        for terrain in terrains {
            if !self.pool.tiles_changed(terrain.id, terrain.tiles_revision) {
                continue;
            }
            let current: HashMap<TileKey, &Arc<Tile>> = terrain
                .tiles
                .iter()
                .filter(|tile| !tile.placed().is_empty())
                .map(|tile| (tile.key(), tile))
                .collect();
            self.pool.retain(terrain.id, |key, held| {
                current.get(&key).is_some_and(|tile| {
                    Arc::ptr_eq(tile, held) || std::ptr::eq(tile.placed(), held.placed())
                })
            });
            for tile in terrain.tiles.iter() {
                if tile.placed().is_empty() || self.pool.contains(terrain.id, tile.key()) {
                    continue;
                }
                let objects = tile_objects(self, terrain, tile);
                self.pool
                    .insert((terrain.id, tile.key()), Arc::clone(tile), objects);
            }
        }
        let heightfield = cx.parts.face::<dyn HeightfieldFace>();
        let infos: Vec<TerrainInfo> = heightfield.map_or_else(Vec::new, |heightfield| {
            terrains
                .iter()
                .enumerate()
                .map(|(index, terrain)| {
                    heightfield.terrain_info(index, terrain.origin, &terrain.pyramid)
                })
                .collect()
        });
        let (dirty, counts_changed) = self.pool.take_changes();
        moved.extend(self.pool.take_moved());
        let core = &mut *cx.core;
        let written = scatter.write_slots(
            &core.device,
            &core.queue,
            (
                &self.pool.poses,
                &self.pool.colors,
                &self.pool.records,
                &self.pool.grounds,
            ),
            dirty,
        );
        if let Some(slots) = written {
            self.pool.uploaded(slots);
        }
        if counts_changed || std::mem::take(&mut self.changed) {
            let counts: Vec<u32> = self
                .groups
                .iter()
                .map(|group| self.pool.counts[group.objects as usize])
                .collect();
            for (group, &count) in self.groups.iter_mut().zip(&counts) {
                group.count = count;
            }
            scatter.set_groups(&core.device, &core.queue, &self.layouts, &counts);
        }
        let sources = heightfield
            .and_then(|heightfield| heightfield.cover_sources())
            .map(|sources| SnapSources {
                heights: sources.heights,
                control: sources.control,
                directory: sources.directory,
                palette: sources.palette,
                macro_maps: sources.macro_maps,
                generation: sources.generation,
                revision: sources.revision,
                terrains: &infos,
            });
        scatter.prepare_snap(&core.device, &core.queue, sources, snapshot.detail_eye);
        let variants: Vec<Variant> = self
            .groups
            .iter()
            .flat_map(|draw| {
                let variant = scatter_binding(core, draw).1;
                [variant, variant.with_fading(true)]
            })
            .collect();
        for variant in variants {
            core.ensure_opaque_variant(passes.opaque_kind, variant);
        }
        let level_of_detail = snapshot.environment.level_of_detail;
        let pixel_error = level_of_detail
            .enabled
            .then_some(level_of_detail.pixel_error);
        let fade = core.switch_fades.share;
        let height = core.display.size().height as f32;
        let impostors = scatter
            .placed
            .prepare_cards(
                core,
                (cx.frame.kind.gpu_mode, passes.opaque_kind),
                &self.impostors,
            )
            .then_some(level_of_detail.impostor_pixels);
        let regions = cx
            .parts
            .face::<dyn ShadowCastersFace>()
            .map_or(&[][..], |shadows| shadows.region_views());
        scatter.prepare(
            &core.device,
            &core.queue,
            (snapshot.view_projection, snapshot.detail_eye),
            (pixel_error, height, impostors, fade),
            regions,
        )
    }

    pub(crate) fn prepare_impostors(
        &mut self,
        cx: &mut Step<'_>,
        prelude: &mut Prelude,
        placed: &mut PlacedImpostors,
    ) -> bool {
        let snapshot = cx.frame.snapshot;
        let settings = snapshot.environment.level_of_detail;
        let enabled = cx.frame.kind.gpu_mode && settings.enabled && settings.impostors;
        let core = &*cx.core;
        let Some(entries) = core.mesh_entries else {
            return false;
        };
        placed.atlas.fit(
            &core.device,
            placed_shape(settings.sanitized().impostor_pixels),
        );
        placed.dropped = 0;
        let (meshes, materials) = (&core.meshes, &core.materials);
        placed.contents.retain(
            |mesh| meshes.contains_key(&mesh),
            |texture| materials.is_texture_resident(texture),
        );
        placed.atlas.retain(
            |(mesh, material)| {
                meshes.contains_key(mesh)
                    && material.is_none_or(|handle| materials.group(handle).is_some())
            },
            |handle| placed_material(snapshot, handle),
        );
        let budget = settings
            .max_impostors
            .map(|budget| budget.saturating_sub(core.impostors.atlas.live()));
        placed.atlas.enforce_budget(budget);
        let groups = &self.groups;
        let mut records = Vec::with_capacity(groups.len());
        for (index, group) in groups.iter().enumerate() {
            let mut record = ImpostorRecord::default();
            let mesh = core.meshes.get(&group.meshes[0]);
            let asset = engine_terrain::RenderTerrains::of(snapshot)
                .terrains
                .iter()
                .flat_map(|terrain| terrain.assets.iter().flatten())
                .find(|asset| asset.mesh.handle == group.meshes[0]);
            let eligible = enabled
                && group.part.is_none()
                && asset.is_some_and(|asset| {
                    asset.mesh.mesh.indices().len() / 3 >= IMPOSTOR_MIN_TRIANGLES
                });
            if let (true, Some(mesh), Some(asset)) = (eligible, mesh, asset) {
                let key = (group.meshes[0], Some(group.material));
                record = match placed.atlas.record(key, Some(&group.source)) {
                    Some(resident) => resident,
                    None => {
                        let textures = |handle: TextureHandle| {
                            asset
                                .textures
                                .iter()
                                .find(|texture| texture.handle == handle)
                                .map(|texture| &*texture.texture)
                        };
                        let contents = &mut placed.contents;
                        let content = (
                            contents.mesh(group.meshes[0], &asset.mesh.mesh),
                            contents.material(Some(&group.source), textures),
                        );
                        let bounds = asset.mesh.mesh.bounds();
                        let (parts, content, (low, high)) = bake_parts(
                            (groups, index, asset),
                            (&core.meshes, &core.mesh_arena, &core.materials),
                            contents,
                            (content, (bounds.min(), bounds.max())),
                        );
                        let (bind_group, variant) = core.materials.binding(Some(group.material));
                        let bake = BakeSource {
                            draw: mesh.draw(&core.mesh_arena),
                            extended: mesh.extended,
                            group: bind_group,
                            variant: variant.with_extended(mesh.extended),
                            bounds: (low, high),
                            material: Some(group.source),
                            parts,
                        };
                        let gpu = BakeGpu {
                            device: &core.device,
                            queue: &core.queue,
                            encoder: &mut *prelude,
                            mips: core.materials.mips(),
                            pipelines: &mut placed.pipelines,
                            entries,
                        };
                        let supplied = placed.atlas.supply(gpu, key, &bake, content, budget);
                        placed.dropped += u32::from(matches!(supplied, Supplied::Refused));
                        supplied.record()
                    }
                };
            }
            records.push(record);
        }
        placed.atlas.release_targets();
        let restyled = records != self.impostors;
        for (index, (layout, record)) in self.layouts.iter_mut().zip(&records).enumerate() {
            let impostor = (record.error(), index as u32);
            if layout.impostor != impostor {
                layout.impostor = impostor;
                self.changed = true;
            }
        }
        self.impostors = records;
        restyled
    }

    fn build_groups(&mut self, core: &Core, terrains: &[engine_terrain::RenderTerrain]) {
        let mut groups = Vec::new();
        let mut layouts = Vec::new();
        let mut covers = Vec::new();
        let mut colors = Vec::new();
        let mut indices = HashMap::default();
        for terrain in terrains {
            for (asset_index, asset) in terrain.assets.iter().enumerate() {
                let Some(asset) = asset else {
                    continue;
                };
                let Some((levels, casters, meshes, errors, camera)) = scatter_levels(core, asset)
                else {
                    continue;
                };
                let parts: Vec<(usize, PartLevels)> = asset
                    .parts
                    .iter()
                    .enumerate()
                    .filter_map(|(part, rendered)| {
                        Some((part, part_levels(core, rendered, levels.len())?))
                    })
                    .collect();
                let bounds = asset.mesh.mesh.bounds();
                let proxy = asset
                    .shadow
                    .as_ref()
                    .and_then(|mesh| Some((mesh, proxy_caster(core, mesh)?)));
                let (low, high) = parts
                    .iter()
                    .map(|&(part, _)| asset.parts[part].meshes[0].mesh.bounds())
                    .chain(proxy.map(|(mesh, _)| mesh.mesh.bounds()))
                    .fold((bounds.min(), bounds.max()), |(low, high), held| {
                        (low.min(held.min()), high.max(held.max()))
                    });
                let material = &asset.material.material;
                let (casts, casters, caster_meshes) = match proxy {
                    Some((mesh, range)) => (Some(false), vec![range], vec![mesh.handle]),
                    None => (
                        Some(matches!(
                            material.alpha_mode(),
                            engine_core::AlphaMode::Mask { .. }
                        )),
                        casters,
                        meshes.clone(),
                    ),
                };
                let objects = groups.len() as u32;
                indices.insert((terrain.id, asset_index), objects);
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
                colors.push(color.color);
                for (part, (part_levels, part_meshes)) in parts {
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
                    colors.push(color.color);
                }
            }
        }
        self.impostors = vec![ImpostorRecord::default(); groups.len()];
        self.pool.clear(groups.len());
        self.groups = groups;
        self.layouts = layouts;
        self.covers = covers;
        self.colors = colors;
        self.indices = indices;
        self.terrains = terrains
            .iter()
            .enumerate()
            .map(|(index, terrain)| (terrain.id, index as u32))
            .collect();
    }

    pub(crate) fn draw(
        &self,
        scatter: &ScatterPass,
        cx: &View<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        prepass: bool,
    ) -> u32 {
        let (Some(groups), Some(passes)) = (scatter.pose_groups(), cx.frame.passes) else {
            return 0;
        };
        let kind = passes.opaque_kind;
        let temporal = cx.frame.temporal.is_some();
        pass.set_bind_group(0, cx.core.shared_group("frame", &["refraction"]), &[]);
        if !prepass {
            pass.set_bind_group(3, cx.core.shared_group("lights", &[]), &[]);
        }
        pass.set_bind_group(2, &groups[usize::from(temporal)], &[]);
        let debug = cx
            .frame
            .shape
            .is_some_and(|shape| shape.targets == Targets::Debug)
            .then(|| casts_shadows(cx));
        let mut draws = 0;
        let fading = scatter.fading_view();
        for (view, fades) in [(0, false), (fading, true)] {
            for (index, draw) in self.groups.iter().enumerate() {
                if draw.count == 0 {
                    continue;
                }
                let (group, variant) = scatter_binding(cx.core, draw);
                let Some(pipeline) =
                    cx.core
                        .opaque_pipeline(kind, variant.with_fading(fades), prepass)
                else {
                    continue;
                };
                pass.set_pipeline(pipeline);
                pass.set_bind_group(1, group, &[]);
                for (level, handle) in draw.meshes.iter().enumerate() {
                    let Some(mesh) = cx.core.meshes.get(handle) else {
                        continue;
                    };
                    let Some((ids, args, offset)) = scatter.draw_source(view, index, level) else {
                        continue;
                    };
                    let source = mesh.draw(&cx.core.mesh_arena);
                    pass.set_vertex_buffer(0, source.vertices.slice(..));
                    if let Some(stream) = source.extras.filter(|_| mesh.extended) {
                        pass.set_vertex_buffer(2, stream.slice(..));
                    }
                    pass.set_index_buffer(source.indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.set_vertex_buffer(1, ids);
                    if let Some(casts) = debug {
                        let casts = casts && draw.casts.is_some();
                        let record = DrawRecord::of(DrawKind::Placed, level, casts);
                        pass.set_vertex_buffer(RECORD_SLOT, cx.core.draw_record(record));
                    }
                    pass.draw_indexed_indirect(args, offset);
                    draws += 1;
                }
            }
        }
        if prepass {
            return draws;
        }
        if let (Some(pipeline), Some(group)) = (scatter.placed.card(), scatter.placed.atlas.group())
            && let Some((cards, args, offset)) = scatter.card_source(0)
        {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, group, &[]);
            pass.set_vertex_buffer(0, cards);
            if let Some(casts) = debug {
                let record = DrawRecord::of(DrawKind::PlacedCard, 0, casts);
                pass.set_vertex_buffer(RECORD_SLOT, cx.core.draw_record(record));
            }
            pass.draw_indirect(args, offset);
            draws += 1;
        }
        draws
    }

    pub(crate) fn draw_shadow(
        &self,
        scatter: &ScatterPass,
        cx: &View<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        (map, casters, cards): (
            usize,
            &[wgpu::RenderPipeline; 2],
            Option<&wgpu::RenderPipeline>,
        ),
    ) -> u32 {
        let (Some(view), Some(poses)) = (scatter.map_view(map), scatter.shadow_group()) else {
            return 0;
        };
        pass.set_bind_group(1, poses, &[]);
        let mut bound = None;
        let mut draws = 0;
        for (index, draw) in self.groups.iter().enumerate() {
            let Some(masked) = draw.casts.filter(|_| draw.count > 0) else {
                continue;
            };
            if masked {
                let Some(material) = cx.core.materials.group(draw.material) else {
                    continue;
                };
                pass.set_bind_group(2, material, &[]);
            }
            if bound != Some(masked) {
                pass.set_pipeline(&casters[usize::from(masked)]);
                bound = Some(masked);
            }
            draws += draw_casters(scatter, cx.core, pass, (index, draw), view);
        }
        if let (Some(pipeline), Some(group)) = (cards, scatter.placed.atlas.group())
            && let Some((cards, args, offset)) = scatter.card_source(view)
        {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(2, group, &[]);
            pass.set_vertex_buffer(0, cards);
            pass.draw_indirect(args, offset);
            draws += 1;
        }
        draws
    }
}

fn casts_shadows(cx: &View<'_>) -> bool {
    cx.parts.produces(cx.table.resource("shadow_maps"))
}

fn placed_material(
    snapshot: &RenderSnapshot,
    handle: MaterialHandle,
) -> Option<engine_core::Material> {
    engine_terrain::RenderTerrains::of(snapshot)
        .terrains
        .iter()
        .flat_map(|terrain| terrain.assets.iter().flatten())
        .flat_map(engine_terrain::RenderPlacementAsset::materials)
        .find(|material| material.handle == handle)
        .map(|material| material.material)
}

fn scatter_key(core: &Core, terrains: &[engine_terrain::RenderTerrain]) -> ScatterKey {
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
                            .map(|mesh| core.meshes.contains_key(&mesh.handle))
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
    (terrains, core.mesh_layout)
}

fn scatter_levels(core: &Core, asset: &engine_terrain::RenderPlacementAsset) -> Option<Levels> {
    let mesh = core.meshes.get(&asset.mesh.handle)?;
    let draw = mesh.draw(&core.mesh_arena);
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
        let Some(coarser) = core.meshes.get(&lod.handle) else {
            break;
        };
        if coarser.extended != mesh.extended {
            break;
        }
        let coarse = coarser.draw(&core.mesh_arena);
        errors[levels.len() - 1] = *error;
        let (drawn, cast) = level(coarser, &coarse, 0, (0, coarse.index_count));
        levels.push(drawn);
        casters.push(cast);
        meshes.push(lod.handle);
    }
    let camera = levels.len() as u32;
    Some((levels, casters, meshes, errors, camera))
}

fn part_levels(
    core: &Core,
    part: &engine_terrain::RenderPlacementPart,
    levels: usize,
) -> Option<PartLevels> {
    let first = core.meshes.get(&part.meshes.first()?.handle)?;
    let mut ranges = Vec::new();
    let mut meshes = Vec::new();
    for rendered in part.meshes.iter().take(levels) {
        let Some(mesh) = core.meshes.get(&rendered.handle) else {
            break;
        };
        if mesh.extended != first.extended {
            break;
        }
        let draw = mesh.draw(&core.mesh_arena);
        ranges.push((draw.index_count, draw.first_index, draw.base_vertex));
        meshes.push(rendered.handle);
    }
    Some((ranges, meshes))
}

fn proxy_caster(core: &Core, proxy: &engine_core::RenderMesh) -> Option<(u32, u32, i32)> {
    let mesh = core.meshes.get(&proxy.handle)?;
    let draw = mesh.draw(&core.mesh_arena);
    let (first, count) = mesh.caster_range(0, (0, draw.index_count));
    Some((count, draw.first_index + first, draw.base_vertex))
}

fn scatter_binding<'c>(core: &'c Core, draw: &ScatterDraw) -> (&'c wgpu::BindGroup, Variant) {
    let mesh = core.meshes.get(&draw.meshes[0]);
    let (group, variant) = core.materials.binding(Some(draw.material));
    let variant = variant
        .with_extended(mesh.is_some_and(|mesh| mesh.extended))
        .with_grounding(draw.source.grounding().is_some());
    (group, variant)
}

fn draw_casters(
    scatter: &ScatterPass,
    core: &Core,
    pass: &mut wgpu::RenderPass<'_>,
    (index, draw): (usize, &ScatterDraw),
    view: usize,
) -> u32 {
    let mut draws = 0;
    for (level, handle) in draw.casters.iter().enumerate() {
        let Some(mesh) = core.meshes.get(handle) else {
            continue;
        };
        let Some((ids, args, offset)) = scatter.draw_source(view, index, level) else {
            continue;
        };
        let source = mesh.draw(&core.mesh_arena);
        pass.set_vertex_buffer(0, source.vertices.slice(..));
        pass.set_index_buffer(source.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_vertex_buffer(1, ids);
        pass.draw_indexed_indirect(args, offset);
        draws += 1;
    }
    draws
}

fn bake_parts<'a>(
    (groups, index, asset): (&[ScatterDraw], usize, &engine_terrain::RenderPlacementAsset),
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
            variant: variant.with_extended(mesh.extended),
        });
    }
    (parts, content, (low, high))
}

fn combined(first: u128, second: u128) -> u128 {
    let mut bytes = [0_u8; 32];
    bytes[..16].copy_from_slice(&first.to_le_bytes());
    bytes[16..].copy_from_slice(&second.to_le_bytes());
    engine_core::content_hash(&bytes)
}

fn tile_objects(
    draws: &ScatterDraws,
    terrain: &engine_terrain::RenderTerrain,
    tile: &engine_terrain::Tile,
) -> TileObjects {
    let placed = tile.placed();
    let mut objects = (
        Vec::with_capacity(placed.len()),
        Vec::with_capacity(placed.len()),
        Vec::with_capacity(placed.len()),
        Vec::with_capacity(placed.len()),
    );
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
        let layer = u32::from(placed.layer).min(NO_LAYER);
        let rank = (engine_core::math::clamp(placed.rank, 0.0, 1.0) * RANK_MAX).round() as u32;
        let draw = &draws.groups[group as usize];
        let (center, radius) = draw.sphere;
        let model = placed.model(terrain.origin, placed.position[1]);
        objects.0.push(pose_rows(model));
        objects.1.push(draw.color);
        objects.2.push(PlacedRecord {
            center: model.transform_point3(center).to_array(),
            radius: radius * placed.scale,
            group,
            scale: placed.scale,
            terrain: index | layer << LAYER_SHIFT | near << NEAR_LEVEL_SHIFT | rank << RANK_SHIFT,
            sink: placed.sink,
            state: STATE_FRESH,
            pad: [0; 3],
        });
        objects.3.push(engine_render::gpu_data::ground_data(
            draw.source.grounding(),
            placed.ground,
            placed.sink,
            placed.scale,
        ));
    }
    objects
}

fn pose_rows(model: Mat4) -> engine_render::gpu_data::PoseData {
    let rows = model.transpose();
    [
        rows.x_axis.to_array(),
        rows.y_axis.to_array(),
        rows.z_axis.to_array(),
    ]
}
