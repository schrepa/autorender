use std::sync::Arc;

use bevy_ecs::prelude::Entity;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, IMPOSTOR_MIN_TRIANGLES, ImpostorAsk, ImpostorRecord,
    Material, MaterialHandle, Mesh, MeshHandle, ResidentMesh, TextureHandle,
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
    pub(crate) textures: Vec<TextureHandle>,
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

    pub(crate) fn materials(&self) -> impl Iterator<Item = MaterialHandle> + '_ {
        std::iter::once(self.material.handle)
            .chain(self.parts.iter().map(|part| part.material.handle))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Kept {
    pub(crate) parts: (u32, u32),
    pub(crate) lods: (u32, u32),
}

fn mesh_of(library: &AssetLibrary, handle: MeshHandle) -> Option<ResolvedMesh<'_>> {
    let mesh = library.mesh(handle)?;
    library.mesh_lods(handle)?;
    Some(ResolvedMesh { handle, mesh })
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
    let textures: Vec<TextureHandle> = std::iter::once(material.material)
        .chain(parts.iter().map(|part| part.material.material))
        .flat_map(Material::textures)
        .collect();
    if !textures
        .iter()
        .all(|&handle| library.texture(handle).is_some())
    {
        return None;
    }
    Some(Resolved {
        mesh,
        material,
        textures,
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
    pub(crate) extended: bool,
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
    pub(crate) asks: Vec<Option<ImpostorAsk>>,
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
                let extended = residency
                    .mesh(asset.mesh.handle)
                    .is_some_and(ResidentMesh::extended);
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
                    extended,
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
                    let extended = residency
                        .mesh(part_meshes[0])
                        .is_some_and(ResidentMesh::extended);
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
                        extended,
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
        let asks = impostor_asks(residency, &groups, terrains);
        Self {
            groups,
            layouts,
            covers,
            colors,
            asks,
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

    pub(crate) fn take_impostors(&mut self, records: &[ImpostorRecord]) -> bool {
        let mut changed = false;
        for (index, (layout, record)) in self.layouts.iter_mut().zip(records).enumerate() {
            let impostor = (record.error(), index as u32);
            if layout.impostor != impostor {
                layout.impostor = impostor;
                changed = true;
            }
        }
        changed
    }
}

fn impostor_asks(
    residency: &AssetResidency,
    groups: &[DrawGroup],
    terrains: &[(Entity, Vec<Option<Resolved<'_>>>)],
) -> Vec<Option<ImpostorAsk>> {
    let resolved = || terrains.iter().flat_map(|(_, assets)| assets.iter().flatten());
    groups
        .iter()
        .enumerate()
        .map(|(index, group)| {
            if group.part.is_some() || !residency.resident(group.meshes[0]) {
                return None;
            }
            let asset = resolved().find(|asset| asset.mesh.handle == group.meshes[0])?;
            if asset.mesh.mesh.indices().len() / 3 < IMPOSTOR_MIN_TRIANGLES {
                return None;
            }
            let parts = groups[index + 1..]
                .iter()
                .take_while(|drawn| drawn.objects == index as u32)
                .filter(|drawn| {
                    drawn.part.is_some_and(|part| {
                        residency.resident(drawn.meshes[0])
                            && asset
                                .parts
                                .get(part)
                                .is_some_and(|rendered| !rendered.meshes.is_empty())
                    })
                })
                .map(|drawn| (drawn.meshes[0], drawn.material))
                .collect();
            Some(ImpostorAsk {
                mesh: group.meshes[0],
                material: Some(group.material),
                parts,
            })
        })
        .collect()
}

fn terrain_indices(terrains: &[(Entity, Vec<Option<Resolved<'_>>>)]) -> HashMap<Entity, u32> {
    terrains
        .iter()
        .enumerate()
        .map(|(index, (terrain, _))| (*terrain, index as u32))
        .collect()
}

fn scatter_levels(residency: &AssetResidency, asset: &Resolved<'_>) -> Option<Levels> {
    let mesh = residency.mesh(asset.mesh.handle)?;
    let level = |mesh: &ResidentMesh, level: usize, own: (u32, u32)| {
        let (first, count) = mesh.caster_range(level, own);
        (
            (own.1, mesh.first_index() + own.0, mesh.base_vertex()),
            (count, mesh.first_index() + first, mesh.base_vertex()),
        )
    };
    let (finest, finest_casters) = level(mesh, 0, (0, mesh.index_count()));
    let mut levels = vec![finest];
    let mut casters = vec![finest_casters];
    let mut meshes = vec![asset.mesh.handle];
    let mut errors = [0.0; LEVELS as usize - 1];
    if asset.lods.is_empty() {
        for &(first_index, index_count, error) in mesh.lods().iter().take(LEVELS as usize - 1) {
            if index_count == 0 {
                break;
            }
            errors[levels.len() - 1] = error;
            let (drawn, cast) = level(mesh, levels.len(), (first_index, index_count));
            levels.push(drawn);
            casters.push(cast);
            meshes.push(asset.mesh.handle);
        }
        let camera = levels.len() as u32;
        return Some((levels, casters, meshes, errors, camera));
    }
    for (lod, error) in asset.lods.iter().take(LEVELS as usize - 1) {
        let Some(coarser) = residency.mesh(lod.handle) else {
            break;
        };
        if coarser.extended() != mesh.extended() {
            break;
        }
        errors[levels.len() - 1] = *error;
        let (drawn, cast) = level(coarser, 0, (0, coarser.index_count()));
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
    let first = residency.mesh(part.meshes.first()?.handle)?;
    let mut ranges = Vec::new();
    let mut meshes = Vec::new();
    for rendered in part.meshes.iter().take(levels) {
        let Some(mesh) = residency.mesh(rendered.handle) else {
            break;
        };
        if mesh.extended() != first.extended() {
            break;
        }
        ranges.push((mesh.index_count(), mesh.first_index(), mesh.base_vertex()));
        meshes.push(rendered.handle);
    }
    Some((ranges, meshes))
}

fn proxy_caster(residency: &AssetResidency, proxy: &ResolvedMesh<'_>) -> Option<Range3> {
    let mesh = residency.mesh(proxy.handle)?;
    let (first, count) = mesh.caster_range(0, (0, mesh.index_count()));
    Some((count, mesh.first_index() + first, mesh.base_vertex()))
}
