use std::sync::Arc;

use bevy_ecs::prelude::Entity;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, Material, MaterialHandle, Mesh, MeshHandle, MeshLods,
};
use engine_math::collections::HashMap;
use engine_places::ground::{PlacedAsset, TerrainEntry, TileFace};
use glam::Vec3;

use crate::layouts::{
    Cover, LAYER_SHIFT, LEVELS, NEAR_LEVEL_SHIFT, NO_LAYER, PlacedRecord, RANK_MAX, RANK_SHIFT,
    STATE_FRESH, SlotColor, TERRAIN_INDEX, ground_data, pack_color, pack_surface, pose_rows,
};
use crate::pool::TileObjects;
use crate::settings::{PlacedLook, ScatterSettings};

type Range3 = (u32, u32, i32);

type PartLevels = (Vec<Range3>, Vec<MeshHandle>);

type Levels = (
    Vec<Range3>,
    Vec<Range3>,
    Vec<MeshHandle>,
    [f32; LEVELS as usize - 1],
    u32,
);

#[derive(Clone, Copy)]
pub(crate) struct ResolvedMesh<'a> {
    pub(crate) handle: MeshHandle,
    pub(crate) mesh: &'a Arc<Mesh>,
    pub(crate) lods: Option<&'a Arc<MeshLods>>,
}

#[derive(Clone, Copy)]
pub(crate) struct ResolvedMaterial<'a> {
    pub(crate) handle: MaterialHandle,
    pub(crate) material: &'a Material,
    pub(crate) version: u64,
}

pub(crate) struct ResolvedPart<'a> {
    pub(crate) material: ResolvedMaterial<'a>,
    pub(crate) meshes: Vec<ResolvedMesh<'a>>,
}

pub(crate) struct Resolved<'a> {
    pub(crate) mesh: ResolvedMesh<'a>,
    pub(crate) material: ResolvedMaterial<'a>,
    pub(crate) lods: Vec<(ResolvedMesh<'a>, f32)>,
    pub(crate) parts: Vec<ResolvedPart<'a>>,
    pub(crate) shadow: Option<ResolvedMesh<'a>>,
    pub(crate) sway: f32,
}

impl<'a> Resolved<'a> {
    pub(crate) fn meshes(&self) -> impl Iterator<Item = &ResolvedMesh<'a>> + '_ {
        std::iter::once(&self.mesh)
            .chain(self.lods.iter().map(|(lod, _)| lod))
            .chain(self.parts.iter().flat_map(|part| &part.meshes))
            .chain(&self.shadow)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Kept {
    pub(crate) parts: (u32, u32),
    pub(crate) lods: (u32, u32),
}

fn mesh_of(library: &AssetLibrary, handle: MeshHandle) -> Option<ResolvedMesh<'_>> {
    let mesh = library.mesh(handle)?;
    let lods = library.mesh_lods(handle)?;
    Some(ResolvedMesh {
        handle,
        mesh,
        lods,
    })
}

fn material_of(library: &AssetLibrary, handle: MaterialHandle) -> Option<ResolvedMaterial<'_>> {
    library.material(handle).map(|material| ResolvedMaterial {
        handle,
        material,
        version: library.material_version(handle).unwrap_or(0),
    })
}

pub(crate) fn resolve<'a>(
    asset: &PlacedAsset,
    look: Option<&PlacedLook>,
    library: &'a AssetLibrary,
    settings: &ScatterSettings,
    kept: &mut Kept,
) -> Option<Resolved<'a>> {
    let mesh = mesh_of(library, asset.mesh)?;
    let lods_asked = look.map_or(&[][..], |look| &look.lods);
    let parts_asked = look.map_or(&[][..], |look| &look.parts);
    let lods_kept = lods_asked.len().min(settings.lods as usize);
    let parts_kept = parts_asked.len().min(settings.parts as usize);
    kept.lods.0 += lods_asked.len() as u32;
    kept.lods.1 += lods_kept as u32;
    kept.parts.0 += parts_asked.len() as u32;
    kept.parts.1 += parts_kept as u32;
    let lods = lods_asked[..lods_kept]
        .iter()
        .map(|lod| mesh_of(library, lod.mesh).map(|mesh| (mesh, lod.error)))
        .collect::<Option<Vec<_>>>()?;
    let parts = parts_asked[..parts_kept]
        .iter()
        .map(|part| {
            Some(ResolvedPart {
                material: material_of(library, part.material)?,
                meshes: part
                    .meshes
                    .iter()
                    .map(|&mesh| mesh_of(library, mesh))
                    .collect::<Option<Vec<_>>>()?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let shadow = match look.and_then(|look| look.shadow) {
        Some(handle) => Some(mesh_of(library, handle)?),
        None => None,
    };
    let material = material_of(library, asset.material)?;
    let textures_held = std::iter::once(material.material)
        .chain(parts.iter().map(|part| part.material.material))
        .flat_map(Material::textures)
        .all(|handle| library.texture(handle).is_some());
    if !textures_held {
        return None;
    }
    Some(Resolved {
        mesh,
        material,
        lods,
        parts,
        shadow,
        sway: look.map_or(0.0, |look| look.sway),
    })
}

pub(crate) struct DrawGroup {
    pub(crate) meshes: Vec<MeshHandle>,
    pub(crate) material: MaterialHandle,
    pub(crate) source: Material,
    pub(crate) color: SlotColor,
    pub(crate) sphere: (Vec3, f32),
    pub(crate) objects: u32,
    pub(crate) part: Option<usize>,
    pub(crate) count: u32,
    pub(crate) casts: Option<bool>,
    pub(crate) casters: Vec<MeshHandle>,
}

impl DrawGroup {
    pub(crate) fn masked(&self) -> bool {
        matches!(self.source.alpha_mode(), AlphaMode::Mask { .. })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GroupLayout {
    pub(crate) levels: Vec<Range3>,
    pub(crate) casters: Vec<Range3>,
    pub(crate) camera: u32,
    pub(crate) errors: [f32; LEVELS as usize - 1],
    pub(crate) impostor: (f32, u32),
    pub(crate) parts: u32,
    pub(crate) part: bool,
}

type PartKey = (MaterialHandle, u64, Vec<MeshHandle>);

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

pub(crate) type ScatterKey = (Vec<(Entity, Vec<Option<AssetKey>>)>, u64);

pub(crate) fn scatter_key(
    residency: &AssetResidency,
    terrains: &[(Entity, Vec<Option<Resolved<'_>>>)],
) -> ScatterKey {
    let terrains = terrains
        .iter()
        .map(|(terrain, assets)| {
            let assets = assets
                .iter()
                .map(|asset| {
                    asset.as_ref().map(|asset| {
                        let lods: Vec<MeshHandle> =
                            asset.lods.iter().map(|(lod, _)| lod.handle).collect();
                        let resident = asset
                            .meshes()
                            .map(|mesh| residency.resident(mesh.handle))
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
            (*terrain, assets)
        })
        .collect();
    (terrains, residency.generation())
}

#[derive(Default)]
pub(crate) struct Groups {
    pub(crate) groups: Vec<DrawGroup>,
    pub(crate) layouts: Vec<GroupLayout>,
    pub(crate) covers: Vec<Cover>,
    pub(crate) colors: Vec<[u32; 2]>,
    indices: HashMap<(Entity, usize), u32>,
    terrains: HashMap<Entity, u32>,
}

impl Groups {
    pub(crate) fn build(
        residency: &AssetResidency,
        terrains: &[(Entity, Vec<Option<Resolved<'_>>>)],
    ) -> Self {
        let mut groups = Vec::new();
        let mut layouts = Vec::new();
        let mut covers = Vec::new();
        let mut colors = Vec::new();
        let mut indices = HashMap::default();
        for (terrain, assets) in terrains {
            for (asset_index, asset) in assets.iter().enumerate() {
                let Some(asset) = asset else {
                    continue;
                };
                let Some((levels, casters, meshes, errors, camera)) =
                    scatter_levels(residency, asset)
                else {
                    continue;
                };
                let parts: Vec<(usize, PartLevels)> = asset
                    .parts
                    .iter()
                    .enumerate()
                    .filter_map(|(part, rendered)| {
                        Some((part, part_levels(residency, rendered, levels.len())?))
                    })
                    .collect();
                let bounds = asset.mesh.mesh.bounds();
                let proxy = asset
                    .shadow
                    .as_ref()
                    .and_then(|mesh| Some((mesh, proxy_caster(residency, mesh)?)));
                let (low, high) = parts
                    .iter()
                    .map(|&(part, _)| asset.parts[part].meshes[0].mesh.bounds())
                    .chain(proxy.map(|(mesh, _)| mesh.mesh.bounds()))
                    .fold((bounds.min(), bounds.max()), |(low, high), held| {
                        (low.min(held.min()), high.max(held.max()))
                    });
                let material = asset.material.material;
                let (casts, casters, caster_meshes) = match proxy {
                    Some((mesh, range)) => (Some(false), vec![range], vec![mesh.handle]),
                    None => (
                        Some(matches!(material.alpha_mode(), AlphaMode::Mask { .. })),
                        casters,
                        meshes.clone(),
                    ),
                };
                let objects = groups.len() as u32;
                indices.insert((*terrain, asset_index), objects);
                let color = SlotColor {
                    color: pack_color(material.color()),
                    surface: pack_surface(material.surface()),
                    flags: asset.sway.to_bits(),
                };
                let sphere = ((low + high) * 0.5, (high - low).length() * 0.5);
                let cover = Cover::of(bounds.min(), bounds.max());
                groups.push(DrawGroup {
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
                    groups.push(DrawGroup {
                        meshes: part_meshes,
                        material: rendered.material.handle,
                        source: *rendered.material.material,
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
        Self {
            groups,
            layouts,
            covers,
            colors,
            indices,
            terrains: terrain_indices(terrains),
        }
    }

    pub(crate) fn tile_objects(&self, terrain: &TerrainEntry, tile: &TileFace) -> TileObjects {
        let placed = &tile.placed;
        let mut objects = (
            Vec::with_capacity(placed.len()),
            Vec::with_capacity(placed.len()),
            Vec::with_capacity(placed.len()),
            Vec::with_capacity(placed.len()),
        );
        let index = self
            .terrains
            .get(&terrain.terrain)
            .copied()
            .unwrap_or(TERRAIN_INDEX)
            .min(TERRAIN_INDEX);
        for placed in placed.iter() {
            let Some(&group) = self
                .indices
                .get(&(terrain.terrain, usize::from(placed.asset)))
            else {
                continue;
            };
            let near = placed.near_level.map_or(0, |level| u32::from(level) + 1);
            let layer = u32::from(placed.layer).min(NO_LAYER);
            let rank = (engine_math::clamp(placed.rank, 0.0, 1.0) * RANK_MAX).round() as u32;
            let draw = &self.groups[group as usize];
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
            objects.3.push(ground_data(
                draw.source.grounding(),
                placed.ground,
                placed.sink,
                placed.scale,
            ));
        }
        objects
    }
}

fn terrain_indices(terrains: &[(Entity, Vec<Option<Resolved<'_>>>)]) -> HashMap<Entity, u32> {
    terrains
        .iter()
        .enumerate()
        .map(|(index, (terrain, _))| (*terrain, index as u32))
        .collect()
}

fn extended(mesh: &Mesh) -> bool {
    mesh.colors().is_some() || mesh.uv1().is_some()
}

fn index_count(mesh: &Mesh) -> u32 {
    mesh.indices().len() as u32
}

fn caster_range(mesh: &ResolvedMesh<'_>, level: usize, own: (u32, u32)) -> (u32, u32) {
    mesh.lods
        .and_then(|lods| lods.casters.get(level).copied())
        .unwrap_or(own)
}

fn lod_levels(mesh: &ResolvedMesh<'_>) -> Vec<(u32, u32, f32)> {
    mesh.lods.map_or_else(Vec::new, |lods| {
        lods.levels
            .iter()
            .map(|level| (level.first_index, level.index_count, level.error))
            .collect()
    })
}

fn scatter_levels(residency: &AssetResidency, asset: &Resolved<'_>) -> Option<Levels> {
    let mesh = &asset.mesh;
    if !residency.resident(mesh.handle) {
        return None;
    }
    let level = |mesh: &ResolvedMesh<'_>, level: usize, own: (u32, u32)| {
        let (first, count) = caster_range(mesh, level, own);
        ((own.1, own.0, 0), (count, first, 0))
    };
    let (finest, finest_casters) = level(mesh, 0, (0, index_count(mesh.mesh)));
    let mut levels = vec![finest];
    let mut casters = vec![finest_casters];
    let mut meshes = vec![mesh.handle];
    let mut errors = [0.0; LEVELS as usize - 1];
    if asset.lods.is_empty() {
        for (first_index, count, error) in lod_levels(mesh).into_iter().take(LEVELS as usize - 1) {
            if count == 0 {
                break;
            }
            errors[levels.len() - 1] = error;
            let (drawn, cast) = level(mesh, levels.len(), (first_index, count));
            levels.push(drawn);
            casters.push(cast);
            meshes.push(mesh.handle);
        }
        let camera = levels.len() as u32;
        return Some((levels, casters, meshes, errors, camera));
    }
    for (lod, error) in asset.lods.iter().take(LEVELS as usize - 1) {
        if !residency.resident(lod.handle) {
            break;
        }
        if extended(lod.mesh) != extended(mesh.mesh) {
            break;
        }
        errors[levels.len() - 1] = *error;
        let (drawn, cast) = level(lod, 0, (0, index_count(lod.mesh)));
        levels.push(drawn);
        casters.push(cast);
        meshes.push(lod.handle);
    }
    let camera = levels.len() as u32;
    Some((levels, casters, meshes, errors, camera))
}

fn part_levels(
    residency: &AssetResidency,
    part: &ResolvedPart<'_>,
    levels: usize,
) -> Option<PartLevels> {
    let first = part.meshes.first()?;
    if !residency.resident(first.handle) {
        return None;
    }
    let mut ranges = Vec::new();
    let mut meshes = Vec::new();
    for rendered in part.meshes.iter().take(levels) {
        if !residency.resident(rendered.handle) {
            break;
        }
        if extended(rendered.mesh) != extended(first.mesh) {
            break;
        }
        ranges.push((index_count(rendered.mesh), 0, 0));
        meshes.push(rendered.handle);
    }
    Some((ranges, meshes))
}

fn proxy_caster(residency: &AssetResidency, proxy: &ResolvedMesh<'_>) -> Option<Range3> {
    if !residency.resident(proxy.handle) {
        return None;
    }
    let (first, count) = caster_range(proxy, 0, (0, index_count(proxy.mesh)));
    Some((count, first, 0))
}
