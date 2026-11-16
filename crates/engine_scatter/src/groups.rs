use std::sync::Arc;

use bevy_ecs::prelude::Entity;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, GeometryHandle, GpuBits, Material, MaterialHandle,
    Mesh, ResidentMesh, SurfaceSlot, TextureHandle, kept_levels,
};
use engine_heightfield::tile::Placed;
use engine_math::collections::HashMap;
use engine_open_world_kit::ground::PlacedAsset;
use engine_places::surface::SurfaceValues;
use engine_places_rendering::impostors::{ImpostorAsk, ImpostorRecord};
use engine_render::{pack_color, pack_surface};
use glam::Vec3;

use crate::layouts::{
    LEVELS, NO_LAYER, PlacedRecord, PlacedValues, RANK_MAX, ScatterTerrain, SlotColor,
    TERRAIN_INDEX, WIND_REACH, pose_rows,
};
use crate::pool::TileObjects;
use crate::settings::{PlacedLook, ScatterSettings};

pub(crate) const GROUNDING_HEIGHT: &str = "grounding_height";

fn slot_color(material: &Material) -> SlotColor {
    SlotColor {
        color: pack_color(material.color()),
        surface: pack_surface(
            SurfaceValues::read(
                |name| material.surface_value(name),
                |name| material.value(name),
            )
            .units(),
            material.emission(),
        ),
        pad: 0,
    }
}

fn masked(material: &Material) -> bool {
    matches!(material.alpha_mode(), AlphaMode::Mask { .. })
}

fn reach_of(material: &Material) -> f32 {
    if material.slot(SurfaceSlot::Position.name()) == Some(crate::WIND) {
        WIND_REACH
    } else {
        0.0
    }
}

fn origin_tint(origin: Vec3) -> u32 {
    let cell = |value: f32| (value * 4.0).floor() as i32 as u32;
    let mut hash =
        cell(origin.x).wrapping_mul(0x8da6_b343) ^ cell(origin.z).wrapping_mul(0xd816_3841);
    hash = (hash ^ (hash >> 16)).wrapping_mul(0x7feb_352d);
    hash = (hash ^ (hash >> 15)).wrapping_mul(0x846c_a68b);
    (hash ^ (hash >> 16)) & 0xffff_0000
}

fn plant_look(material: &Material) -> [u32; 2] {
    let [through, softness, hue, brightness] = crate::PLANT_LOOK.map(|name| {
        let stated = material.value(name).unwrap_or(0.0);
        (engine_math::clamp(stated, 0.0, 1.0) * 65535.0).round() as u32
    });
    [through | softness << 16, hue | brightness << 16]
}

fn placed_values(
    height: Option<f32>,
    (origin, normal): (Vec3, [f32; 2]),
    (sink, scale): (f32, f32),
    (sway, look): (f32, [u32; 2]),
) -> PlacedValues {
    let mut values = PlacedValues {
        tint: origin_tint(origin),
        sway,
        look,
        ..PlacedValues::default()
    };
    let Some(height) = height else {
        return values;
    };
    let snorm = |value: f32| (engine_math::clamp(value, -1.0, 1.0) * 32767.0).round() as i16;
    let tilt = normal.map(snorm);
    let [x, z] = tilt.map(|held| engine_math::max(f32::from(held) / 32767.0, -1.0));
    let up = engine_math::max(1.0 - (x * x + z * z), 0.01).sqrt();
    values.tilt = u32::from(tilt[0] as u16) | u32::from(tilt[1] as u16) << 16;
    values.ground = origin.y + sink + (x * origin.x + z * origin.z) / up;
    values.reach = height * scale;
    values
}

type Range3 = (u32, u32, i32);

type PartLevels = (Vec<Range3>, Vec<GeometryHandle>);

type Levels = (
    Vec<Range3>,
    Vec<GeometryHandle>,
    [f32; LEVELS as usize - 1],
    u32,
);

#[derive(Clone, Copy)]
pub(crate) struct ResolvedMesh<'a> {
    pub(crate) handle: GeometryHandle,
    pub(crate) mesh: &'a Arc<Mesh>,
}

#[derive(Clone, Copy)]
pub(crate) struct ResolvedMaterial<'a> {
    pub(crate) handle: MaterialHandle,
    pub(crate) material: &'a Material,
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
    pub(crate) authored: bool,
    pub(crate) parts: Vec<ResolvedPart<'a>>,
    pub(crate) sway: f32,
}

impl<'a> Resolved<'a> {
    pub(crate) fn meshes(&self) -> impl Iterator<Item = &ResolvedMesh<'a>> + '_ {
        std::iter::once(&self.mesh)
            .chain(self.lods.iter().map(|(lod, _)| lod))
            .chain(self.parts.iter().flat_map(|part| &part.meshes))
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

fn mesh_of(library: &AssetLibrary, handle: GeometryHandle) -> Option<ResolvedMesh<'_>> {
    let mesh = library.mesh(handle)?;
    library.mesh_lods(handle)?;
    Some(ResolvedMesh { handle, mesh })
}

fn material_of(library: &AssetLibrary, handle: MaterialHandle) -> Option<ResolvedMaterial<'_>> {
    library
        .material(handle)
        .map(|material| ResolvedMaterial { handle, material })
}

pub(crate) fn resolve<'a>(
    asset: &PlacedAsset,
    (look, errors): (Option<&PlacedLook>, &[f32]),
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
        .zip(errors)
        .map(|(&lod, &error)| mesh_of(library, lod).map(|mesh| (mesh, error)))
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
        authored: !lods_asked.is_empty(),
        parts,
        sway: look.map_or(0.0, |look| look.sway),
    })
}

pub(crate) struct DrawGroup {
    pub(crate) meshes: Vec<GeometryHandle>,
    pub(crate) material: MaterialHandle,
    pub(crate) grounding: Option<f32>,
    pub(crate) masked: bool,
    pub(crate) extended: bool,
    pub(crate) color: SlotColor,
    pub(crate) sway: f32,
    pub(crate) look: [u32; 2],
    pub(crate) sphere: (Vec3, f32),
    pub(crate) objects: u32,
    pub(crate) part: Option<usize>,
    pub(crate) count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GroupLayout {
    pub(crate) levels: Vec<Range3>,
    pub(crate) camera: u32,
    pub(crate) errors: [f32; LEVELS as usize - 1],
    pub(crate) impostor: (f32, u32),
    pub(crate) parts: u32,
    pub(crate) part: bool,
}

#[derive(Default)]
pub(crate) struct Groups {
    pub(crate) groups: Vec<DrawGroup>,
    pub(crate) layouts: Vec<GroupLayout>,
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
        let mut colors = Vec::new();
        let mut indices = HashMap::default();
        for (terrain, assets) in terrains {
            for (asset_index, asset) in assets.iter().enumerate() {
                let Some(asset) = asset else {
                    continue;
                };
                let Some((levels, meshes, errors, camera)) = scatter_levels(residency, asset)
                else {
                    continue;
                };
                let extended = residency
                    .mesh(asset.mesh.handle)
                    .is_some_and(ResidentMesh::extended);
                let parts = resident_parts(residency, asset, levels.len());
                let bounds = asset.mesh.mesh.bounds();
                let (low, high) = parts
                    .iter()
                    .map(|&(part, _)| asset.parts[part].meshes[0].mesh.bounds())
                    .fold((bounds.min(), bounds.max()), |(low, high), held| {
                        (low.min(held.min()), high.max(held.max()))
                    });
                let material = asset.material.material;
                let objects = groups.len() as u32;
                indices.insert((*terrain, asset_index), objects);
                let color = slot_color(material);
                let sphere = (
                    (low + high) * 0.5,
                    (high - low).length() * 0.5 + reach_of(material),
                );
                groups.push(DrawGroup {
                    meshes,
                    material: asset.material.handle,
                    grounding: material.value(GROUNDING_HEIGHT),
                    masked: masked(material),
                    extended,
                    color,
                    sway: asset.sway,
                    look: plant_look(material),
                    sphere,
                    objects,
                    part: None,
                    count: 0,
                });
                layouts.push(GroupLayout {
                    levels,
                    camera,
                    errors,
                    impostor: (0.0, 0),
                    parts: parts.len() as u32,
                    part: false,
                });
                colors.push(color.color);
                for (part, (part_levels, part_meshes)) in parts {
                    let rendered = &asset.parts[part];
                    let extended = residency
                        .mesh(part_meshes[0])
                        .is_some_and(ResidentMesh::extended);
                    layouts.push(GroupLayout {
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
                        grounding: rendered.material.material.value(GROUNDING_HEIGHT),
                        masked: masked(rendered.material.material),
                        extended,
                        color,
                        sway: asset.sway,
                        look: plant_look(rendered.material.material),
                        sphere,
                        objects,
                        part: Some(part),
                        count: 0,
                    });
                    colors.push(color.color);
                }
            }
        }
        let asks = impostor_asks(residency, &groups, terrains);
        Self {
            groups,
            layouts,
            colors,
            asks,
            indices,
            terrains: terrain_indices(terrains),
        }
    }

    pub(crate) fn same_objects(&self, other: &Self) -> bool {
        self.indices == other.indices
            && self.terrains == other.terrains
            && self.groups.len() == other.groups.len()
            && self.groups.iter().zip(&other.groups).all(|(a, b)| {
                a.color == b.color
                    && a.sway == b.sway
                    && a.sphere == b.sphere
                    && a.objects == b.objects
                    && a.grounding == b.grounding
            })
    }

    pub(crate) fn tile_objects(
        &self,
        terrain: Entity,
        origin: Vec3,
        placed: &[Placed],
    ) -> TileObjects {
        let mut objects = (
            Vec::with_capacity(placed.len()),
            Vec::with_capacity(placed.len()),
            Vec::with_capacity(placed.len()),
            Vec::with_capacity(placed.len()),
        );
        let index = self
            .terrains
            .get(&terrain)
            .copied()
            .unwrap_or(TERRAIN_INDEX)
            .min(TERRAIN_INDEX);
        for placed in placed {
            let Some(&group) = self.indices.get(&(terrain, usize::from(placed.asset))) else {
                continue;
            };
            let near = placed.near_level.map_or(0, |level| u32::from(level) + 1);
            let layer = u32::from(placed.layer).min(NO_LAYER);
            let rank = (engine_math::clamp(placed.rank, 0.0, 1.0) * RANK_MAX).round() as u32;
            let draw = &self.groups[group as usize];
            let (center, radius) = draw.sphere;
            let model = placed.model(origin, placed.position[1]);
            objects.0.push(pose_rows(model));
            objects.1.push(draw.color);
            objects.2.push(PlacedRecord {
                center: model.transform_point3(center).to_array(),
                radius: radius * placed.scale,
                group,
                scale: placed.scale,
                terrain: ScatterTerrain {
                    index,
                    layer,
                    near,
                    rank,
                    hidden: false,
                }
                .pack(),
                sink: placed.sink,
                pad: [0; 4],
            });
            objects.3.push(placed_values(
                draw.grounding,
                (model.w_axis.truncate(), placed.ground),
                (placed.sink, placed.scale),
                (draw.sway, draw.look),
            ));
        }
        objects
    }

    pub(crate) fn tile_boxes(
        &self,
        terrain: Entity,
        origin: Vec3,
        placed: &[Placed],
        impostors: &[ImpostorRecord],
    ) -> Vec<(Vec3, Vec3)> {
        placed
            .iter()
            .filter_map(|placed| {
                let &group = self.indices.get(&(terrain, usize::from(placed.asset)))?;
                let Some(views) = impostors.get(group as usize).filter(|views| views.framed())
                else {
                    return Some((Vec3::ZERO, Vec3::ZERO));
                };
                let pose = pose_rows(placed.model(origin, placed.position[1]));
                let [x, y, z, _] = views.sphere;
                let [half_x, half_y, half_z] = views.extent;
                let middle = |row: [f32; 4]| row[0] * x + row[1] * y + row[2] * z + row[3];
                let spread = |row: [f32; 4]| {
                    row[0].abs() * half_x + row[1].abs() * half_y + row[2].abs() * half_z
                };
                Some((
                    Vec3::new(middle(pose[0]), middle(pose[1]), middle(pose[2])),
                    Vec3::new(spread(pose[0]), spread(pose[1]), spread(pose[2])),
                ))
            })
            .collect()
    }

    pub(crate) fn take_impostors(&mut self, records: &[ImpostorRecord]) {
        for (index, (layout, record)) in self.layouts.iter_mut().zip(records).enumerate() {
            layout.impostor = (record.error(), index as u32);
        }
    }
}

fn impostor_asks(
    residency: &AssetResidency,
    groups: &[DrawGroup],
    terrains: &[(Entity, Vec<Option<Resolved<'_>>>)],
) -> Vec<Option<ImpostorAsk>> {
    let resolved = || {
        terrains
            .iter()
            .flat_map(|(_, assets)| assets.iter().flatten())
    };
    groups
        .iter()
        .enumerate()
        .map(|(index, group)| {
            if group.part.is_some() || !residency.resident(group.meshes[0]) {
                return None;
            }
            let asset = resolved().find(|asset| asset.mesh.handle == group.meshes[0])?;
            let drawn_parts: Vec<_> = groups[index + 1..]
                .iter()
                .take_while(|drawn| drawn.objects == index as u32)
                .filter(|drawn| {
                    drawn.part.is_some_and(|part| {
                        asset
                            .parts
                            .get(part)
                            .is_some_and(|rendered| !rendered.meshes.is_empty())
                    })
                })
                .collect();
            if drawn_parts
                .iter()
                .any(|drawn| !residency.resident(drawn.meshes[0]))
            {
                return None;
            }
            let parts = drawn_parts
                .iter()
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
    let level = |mesh: &ResidentMesh, (first, count): (u32, u32)| {
        (count, mesh.first_index() + first, mesh.base_vertex())
    };
    let mut levels = vec![level(mesh, (0, mesh.index_count()))];
    let mut meshes = vec![asset.mesh.handle];
    let mut errors = [0.0; LEVELS as usize - 1];
    if !asset.authored {
        let lods = mesh.lods();
        let filled = lods.iter().take_while(|&&(_, count, _)| count != 0).count();
        for kept in kept_levels(filled, LEVELS as usize - 1) {
            let (first_index, index_count, error) = lods[kept];
            errors[levels.len() - 1] = error;
            levels.push(level(mesh, (first_index, index_count)));
            meshes.push(asset.mesh.handle);
        }
        let camera = levels.len() as u32;
        return Some((levels, meshes, errors, camera));
    }
    for (lod, error) in asset.lods.iter().take(LEVELS as usize - 1) {
        let Some(coarser) = residency.mesh(lod.handle) else {
            break;
        };
        if coarser.extended() != mesh.extended() {
            break;
        }
        errors[levels.len() - 1] = *error;
        levels.push(level(coarser, (0, coarser.index_count())));
        meshes.push(lod.handle);
    }
    let camera = levels.len() as u32;
    Some((levels, meshes, errors, camera))
}

fn resident_parts(
    residency: &AssetResidency,
    asset: &Resolved<'_>,
    levels: usize,
) -> Vec<(usize, PartLevels)> {
    asset
        .parts
        .iter()
        .enumerate()
        .filter_map(|(part, rendered)| Some((part, part_levels(residency, rendered, levels)?)))
        .collect()
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
