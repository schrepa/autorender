use std::collections::BTreeMap;

use cook::Refusal;
use cook_model::{Built, Part, Reading, Source, encode_mesh, node_label, placed, primitive_item, read_part};
use serde_json::{Value, json};

use crate::chunks::{self, ChunkPlan};

pub struct Merged {
    pub built: Built,
    pub meshes: Vec<Value>,
}

pub struct Drawn {
    pub built: Vec<Built>,
    pub largest_m: BTreeMap<String, f32>,
    pub widest_object: (String, usize),
    pub chunks: Option<Value>,
}

#[derive(Default)]
struct Tally {
    vertices: usize,
    indices: usize,
    duplicated: usize,
    folded: usize,
    under: usize,
    rows: Vec<(String, usize, usize)>,
    classes: BTreeMap<usize, (usize, usize, usize)>,
}

fn chunked(
    source: &Source,
    (plan, origin, classes): (&ChunkPlan, [f32; 2], Option<&[f32]>),
    stem: &str,
    group: &[Part],
    (tally, largest_m): (&mut Tally, &mut BTreeMap<String, f32>),
) -> Result<Vec<Built>, Refusal> {
    let sources: Vec<chunks::Source<'_>> = group
        .iter()
        .map(|part| chunks::Source {
            vertices: &part.vertices,
            indices: &part.indices,
            uv1: part.uv1.as_deref(),
            colors: part.colors.as_deref(),
        })
        .collect();
    tally.vertices += group.iter().map(|part| part.vertices.len()).sum::<usize>();
    tally.indices += group.iter().map(|part| part.indices.len()).sum::<usize>();
    let (pieces, folded, under) = chunks::split(plan, origin, &sources);
    tally.duplicated += chunks::duplicated(&pieces);
    tally.folded += folded;
    tally.under += under;
    let sized: Vec<(Option<(usize, f32)>, chunks::Piece)> = match classes {
        None => pieces.into_iter().map(|piece| (None, piece)).collect(),
        Some(bounds) => pieces
            .into_iter()
            .flat_map(|piece| chunks::by_size(piece, bounds))
            .map(|held| {
                let row = tally.classes.entry(held.class).or_default();
                row.0 += 1;
                row.1 += held.pieces;
                row.2 += held.piece.triangles();
                (Some((held.class, held.largest_m)), held.piece)
            })
            .collect(),
    };
    let mut built = Vec::new();
    for (at, (class, piece)) in sized.into_iter().enumerate() {
        let piece_cell = piece.cell;
        let item = format!(
            "{stem} chunk {at} (cell {}, {})",
            piece.cell.0, piece.cell.1
        );
        let part = Part {
            material: group[0].material,
            vertices: piece.vertices,
            indices: piece.indices,
            uv1: piece.uv1,
            colors: piece.colors,
            skin: None,
            influences: 0,
            normals: group[0].normals,
            uv: group[0].uv,
        };
        let mut mesh = encode_mesh(source, &item, std::slice::from_ref(&part), true)?;
        mesh.stem = format!("{stem}_{}_{}", piece_cell.0, piece_cell.1);
        if let Some((class, largest)) = class {
            mesh.stem = format!("{}_s{class}", mesh.stem);
            largest_m.insert(mesh.stem.clone(), largest);
        }
        tally
            .rows
            .push((mesh.stem.clone(), mesh.triangles, mesh.vertices));
        built.push(mesh);
    }
    Ok(built)
}

fn drawn_parts(
    source: &Source,
    (scale, tolerance): (f32, f32),
    merged: &[&str],
) -> Result<(Vec<Part>, (String, usize)), Refusal> {
    let mut parts: Vec<Part> = Vec::new();
    let mut widest = (String::new(), 0);
    for node in source.gltf.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        if merged
            .iter()
            .any(|prefix| node.name().is_some_and(|held| held.starts_with(prefix)))
        {
            continue;
        }
        let place = placed(source, &node, scale, tolerance)?;
        let mut object = 0;
        for held in mesh.primitives() {
            let item = primitive_item(&node, &held);
            let part = read_part(source, &item, &held, place, None, Reading::Drawn)?;
            if part.uv1.is_some() == part.colors.is_some() {
                return Err(source.refuse(
                    "MapMeshLight",
                    item,
                    "a drawn map mesh carries exactly one of TEXCOORD_1 (lightmapped) and COLOR_0 (vertex light)",
                ));
            }
            object += part.vertices.len();
            parts.push(part);
        }
        if object > widest.1 {
            widest = (node_label(&node), object);
        }
    }
    Ok((parts, widest))
}

pub fn drawn(
    source: &Source,
    name: &str,
    (scale, tolerance): (f32, f32),
    (merged, plan): (&[&str], Option<&ChunkPlan>),
) -> Result<Drawn, Refusal> {
    let (mut parts, widest) = drawn_parts(source, (scale, tolerance), merged)?;
    let mut groups: Vec<(usize, bool)> = parts
        .iter()
        .filter_map(|part| Some((part.material?, part.colors.is_some())))
        .collect();
    groups.sort_unstable();
    groups.dedup();
    let mut built = Vec::new();
    let mut largest_m = BTreeMap::new();
    let mut tallies = [Tally::default(), Tally::default()];
    let origin = parts.iter().flat_map(|part| part.vertices.iter()).fold(
        [f32::INFINITY; 2],
        |low, vertex| {
            let lower = |held: f32, next: f32| if next < held { next } else { held };
            [
                lower(low[0], vertex.position[0]),
                lower(low[1], vertex.position[2]),
            ]
        },
    );
    for &(material, colored) in &groups {
        let group: Vec<Part> = parts
            .extract_if(.., |part| {
                part.material == Some(material) && part.colors.is_some() == colored
            })
            .collect();
        let stem = if groups.len() == 1 {
            name.to_owned()
        } else {
            format!("{name}_{material}")
        };
        let stem = if colored {
            format!("{stem}_detail")
        } else {
            stem
        };
        if let Some(plan) = plan {
            built.extend(chunked(
                source,
                (
                    plan,
                    origin,
                    plan.detail_classes_m.as_deref().filter(|_| colored),
                ),
                &stem,
                &group,
                (&mut tallies[usize::from(colored)], &mut largest_m),
            )?);
            continue;
        }
        let item = format!("materials[{material}] group");
        let mut mesh = encode_mesh(source, &item, &group, true)?;
        mesh.stem = stem;
        built.push(mesh);
    }
    if built.is_empty() {
        return Err(source.refuse("NoMesh", "nodes", "no drawn mesh to cook"));
    }
    let chunks = plan.map(|plan| {
        let [lit, detail] = tallies.map(|tally| {
            let mut held = chunks::summary(
                plan,
                (tally.vertices, tally.indices),
                &tally.rows,
                (tally.duplicated, tally.folded, tally.under),
            );
            if !tally.classes.is_empty() {
                held["classes"] = tally
                    .classes
                    .iter()
                    .map(|(class, (meshes, pieces, triangles))| json!({"class": class, "meshes": meshes, "pieces": pieces, "triangles": triangles}))
                    .collect();
            }
            held
        });
        json!({"lightmapped": lit, "vertex_lit": detail})
    });
    Ok(Drawn {
        built,
        largest_m,
        widest_object: widest,
        chunks,
    })
}

pub fn merged(
    source: &Source,
    (prefix, output): (&str, &str),
    (scale, tolerance): (f32, f32),
) -> Result<Merged, Refusal> {
    let mut parts = Vec::new();
    let mut meshes = Vec::new();
    let (mut indices, mut vertices) = (0, 0);
    for node in source.gltf.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        let Some(node_name) = node.name().filter(|held| held.starts_with(prefix)) else {
            continue;
        };
        let place = placed(source, &node, scale, tolerance)?;
        let (first_index, first_vertex) = (indices, vertices);
        let mut sources = (Vec::new(), Vec::new());
        for held in mesh.primitives() {
            let item = primitive_item(&node, &held);
            let part = read_part(source, &item, &held, place, None, Reading::Shape)?;
            indices += part.indices.len();
            vertices += part.vertices.len();
            sources.0.push(part.normals);
            sources.1.push(part.uv);
            parts.push(part);
        }
        let said = |held: &[&'static str]| {
            if held.iter().all(|from| *from == held[0]) {
                held[0]
            } else {
                "mixed"
            }
        };
        meshes.push(json!({
            "node": node_name,
            "mesh": mesh.name(),
            "indices": [first_index, indices],
            "triangles": [first_index / 3, indices / 3],
            "vertices": [first_vertex, vertices],
            "normals": said(&sources.0),
            "uv": said(&sources.1),
            "extras": source.extras("nodes", node.index()),
            "mesh_extras": source.extras("meshes", mesh.index()),
        }));
    }
    if parts.is_empty() {
        return Err(source.refuse(
            "MergeEmpty",
            format!("merge prefix `{prefix}`"),
            "no mesh node name starts with it",
        ));
    }
    let mut built = encode_mesh(source, &format!("merge `{output}`"), &parts, false)?;
    built.stem = output.to_owned();
    Ok(Merged { built, meshes })
}
