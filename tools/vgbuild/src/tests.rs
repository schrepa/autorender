use super::*;

/// Positions, normals, texture coordinates and indices.
type Triangles = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>);

/// A bumpy sphere of `rings` x `segments` quads with texture coordinates and
/// a seam, dense enough for several levels.
pub(crate) fn bumpy_sphere(rings: u32, segments: u32) -> Triangles {
    let (mut positions, mut normals, mut uvs, mut indices) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let theta = v * std::f32::consts::PI;
        for segment in 0..=segments {
            let u = segment as f32 / segments as f32;
            let phi = u * std::f32::consts::TAU;
            let (sin_theta, cos_theta) = engine_math::sin_cos(theta);
            let (sin_phi, cos_phi) = engine_math::sin_cos(phi);
            let direction = [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi];
            let bump = 1.0 + 0.05 * ((ring * 7 + segment * 13) % 11) as f32 / 11.0;
            positions.push(direction.map(|axis| axis * bump));
            normals.push(direction);
            uvs.push([u, v]);
        }
    }
    let row = segments + 1;
    for ring in 0..rings {
        for segment in 0..segments {
            let a = ring * row + segment;
            let b = a + row;
            indices.extend([a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    (positions, normals, uvs, indices)
}

#[test]
fn hierarchies_select_fewer_triangles_farther_away() {
    let (positions, normals, uvs, indices) = bumpy_sphere(160, 320);
    let dag = build(&Source {
        positions: &positions,
        normals: &normals,
        uvs: Some(&uvs),
        indices: &indices,
    })
    .unwrap();
    assert!(dag.levels > 3, "{} levels", dag.levels);
    for cluster in &dag.clusters {
        assert!(cluster.triangles.len() / 3 <= dag::CLUSTER_TRIANGLES);
        assert!(!cluster.vertices.is_empty() && cluster.vertices.len() <= dag::CLUSTER_VERTICES);
        assert!(
            cluster
                .triangles
                .iter()
                .all(|&local| (local as usize) < cluster.vertices.len())
        );
        // A cluster's own error never exceeds its group's: a cut is either
        // the cluster or what replaces it.
        if let Some(refined) = cluster.refined {
            let own = dag.groups[refined as usize].simplified.error;
            let parent = dag.groups[cluster.group as usize].simplified.error;
            assert!(own <= parent, "{own} > {parent}");
        }
    }
    let triangles = |selected: &[u32]| {
        selected
            .iter()
            .map(|&cluster| dag.clusters[cluster as usize].triangles.len() / 3)
            .sum::<usize>()
    };
    let view = |distance: f32| View {
        eye: [0.0, 0.0, distance],
        // cot(fovy / 2) of a 60 degree field of view.
        projection: 3.0_f32.sqrt(),
        near: 0.1,
        threshold: 2.0 / 1080.0,
    };
    let full = indices.len() / 3;
    let near = triangles(&dag.select(&view(1.5)));
    let middle = triangles(&dag.select(&view(20.0)));
    let far = triangles(&dag.select(&view(400.0)));
    assert!(near <= full && near > full / 4, "near {near} of {full}");
    assert!(middle < near && far < middle, "{near} {middle} {far}");
    // Its bumped pole rings are open borders, which stay at every level.
    assert!(far <= 8 * dag::CLUSTER_TRIANGLES, "far {far}");
}

#[test]
fn packed_meshes_decode_with_their_geometry() {
    let (positions, normals, uvs, indices) = bumpy_sphere(96, 192);
    let source = Source {
        positions: &positions,
        normals: &normals,
        uvs: Some(&uvs),
        indices: &indices,
    };
    let bytes = build_virtual_mesh(&source).unwrap();
    let mesh = engine_assets::VirtualMesh::decode(&bytes).unwrap();
    let original = (indices.len() / 3) as u32;
    assert_eq!(mesh.source_triangle_count(), original);
    assert!(mesh.has_uvs() && mesh.levels() > 2 && mesh.depth() >= 2);
    // The finest cut is the original geometry; coarser cuts shrink.
    let finest = mesh.triangles_of(&mesh.cut(0.0));
    assert!(
        finest <= original && finest * 10 >= original * 9,
        "{finest} of {original}"
    );
    let sizes: Vec<u32> = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
        .iter()
        .map(|&error| mesh.triangles_of(&mesh.cut(error)))
        .collect();
    println!(
        "cut sizes {sizes:?}, {} levels, {} clusters",
        mesh.levels(),
        mesh.cluster_count()
    );
    assert!(sizes.windows(2).all(|pair| pair[1] <= pair[0]), "{sizes:?}");
    assert!(sizes[5] * 20 < finest, "{sizes:?}");
    // Decoded vertices sit on the bumpy sphere within quantization.
    let cluster = mesh.cut(0.0)[0] as usize;
    let (vertices, triangles) = mesh.cluster_geometry(cluster);
    assert!(!triangles.is_empty());
    for vertex in &vertices {
        let radius = glam::Vec3::from_array(vertex.position).length();
        assert!((0.999..=1.051).contains(&radius), "radius {radius}");
        let normal = glam::Vec3::from_array(vertex.normal);
        assert!((normal.length() - 1.0).abs() < 1.0e-3);
        assert!(normal.dot(glam::Vec3::from_array(vertex.position).normalize()) > 0.99);
        assert!(vertex.uv.iter().all(|value| (0.0..=1.0).contains(value)));
    }
    // A fallback within budget, with bounds holding every level.
    let fallback = mesh.fallback(2048).unwrap();
    assert!(fallback.indices().len() / 3 <= 2048 && !fallback.indices().is_empty());
    let geometry = std::sync::Arc::new(mesh);
    let with = engine_assets::Mesh::from_virtual(geometry.clone(), 2048).unwrap();
    assert!(with.virtual_geometry().is_some());
    assert!(with.bounds().min().cmple(geometry.bounds().min()).all());
    assert!(with.bounds().max().cmpge(geometry.bounds().max()).all());
    // Damaged files fail validation.
    let mut broken = bytes.clone();
    broken[0] = b'X';
    assert!(engine_assets::VirtualMesh::decode(&broken).is_err());
    assert!(engine_assets::VirtualMesh::decode(&bytes[..bytes.len() - 4]).is_err());
    let mut wild = bytes.clone();
    // A cluster's page far out of range.
    let cluster_page = (engine_assets::HEADER_WORDS
        + geometry.node_count() * engine_assets::NODE_WORDS
        + geometry.group_count() * engine_assets::GROUP_WORDS
        + 7)
        * 4;
    wild[cluster_page..cluster_page + 4].copy_from_slice(&999_999u32.to_le_bytes());
    assert!(engine_assets::VirtualMesh::decode(&wild).is_err());
}

#[test]
fn open_borders_keep_every_vertex_at_every_level() {
    let tile = crate::procedural::terrain(64, 16.0, 5);
    let bytes = build_virtual_mesh(&tile.source()).unwrap();
    let mesh = engine_assets::VirtualMesh::decode(&bytes).unwrap();
    // The coarsest cut still holds all 65 vertices of the western border.
    let coarse = mesh.cut(f32::MAX / 2.0);
    assert!(mesh.triangles_of(&coarse) < 64 * 64 * 2 / 4);
    let mut rows = std::collections::BTreeSet::new();
    for cluster in coarse {
        let (vertices, _) = mesh.cluster_geometry(cluster as usize);
        for vertex in vertices {
            if (vertex.position[0] + 8.0).abs() < 1.0e-3 {
                rows.insert(((vertex.position[2] + 8.0) / 0.25).round() as i32);
            }
        }
    }
    assert_eq!(rows.len(), 65, "{rows:?}");
}

#[test]
fn terrain_tiles_meet_their_neighbors() {
    let cells = 64;
    let tile = crate::procedural::terrain(cells, 16.0, 5);
    let row = cells as usize + 1;
    for index in 0..row {
        // East edge against west, north against south: same heights, same
        // normals.
        for (a, b) in [
            (index * row, index * row + cells as usize),
            (index, cells as usize * row + index),
        ] {
            assert_eq!(tile.positions[a][1], tile.positions[b][1]);
            assert_eq!(tile.normals[a], tile.normals[b]);
        }
    }
}
