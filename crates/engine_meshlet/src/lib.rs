#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bounds;
mod bvh;
mod dag;
mod encode;
mod format;
mod group;
mod manifest;
mod partition;
mod simplify;
mod verify;

pub use bounds::{
    aabb_of_points, bounding_sphere_points, compute_normal_cone, merge_spheres, morton_3d,
    pack_normal_cone,
};
pub use dag::build;
pub use encode::{
    compress_page_words, compute_base_quant_step, decode_cluster_words, decode_compressed_pages,
    decode_octahedral, decompress_page_words, dequantize_position, encode_cluster_words,
    encode_compressed_pages, encode_octahedral, page_bytes_as_words, words_to_le_bytes,
};
pub use format::{
    BVH_BRANCHING, BuildStats, BuiltMeshlet, BvhNode, CLUSTER_HEADER_WORDS, CLUSTER_TRI_WORDS,
    CLUSTER_VERTEX_WORDS, ClusterHeader, ClusterVertex, DecodedCluster, FLAG_IN_ROOT_PAGE,
    FLAG_PINNED_COARSE, FLAG_ROOT_CLUSTER, MANIFEST_MAGIC, MANIFEST_VERSION, MAX_CLUSTER_TRIS,
    MAX_CLUSTER_VERTS, MeshletManifest, NO_BVH_CHILD, PAGE_SIZE_BYTES, PAGE_WORDS, PageSpan,
    ROOT_PARENT_ERROR, SourceMesh, VerifyReport,
};
pub use verify::verify;

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use engine_math::{cos, max, min, sin};
    use glam::{Vec2, Vec3};

    use super::*;

    fn make_cloth_grid(nx: usize, ny: usize) -> SourceMesh {
        let mut positions = Vec::with_capacity((nx + 1) * (ny + 1));
        let mut normals = Vec::with_capacity((nx + 1) * (ny + 1));
        let mut uvs = Vec::with_capacity((nx + 1) * (ny + 1));
        for j in 0..=ny {
            let v = (j as f32) / (ny as f32);
            for i in 0..=nx {
                let u = (i as f32) / (nx as f32);
                let wave = sin(u * TAU * 2.0) * 0.12 + cos(v * TAU) * 0.05;
                positions.push(Vec3::new(u * 4.0, v * 6.0, wave));
                normals.push(Vec3::new(-cos(u * TAU * 2.0) * 0.2, 0.0, 1.0).normalize());
                uvs.push(Vec2::new(u * 2.0, v * 3.0));
            }
        }
        let mut indices = Vec::with_capacity(nx * ny * 6);
        let stride = (nx + 1) as u32;
        for j in 0..ny as u32 {
            for i in 0..nx as u32 {
                let a = j * stride + i;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                indices.extend_from_slice(&[a, b, d, a, d, c]);
            }
        }
        SourceMesh {
            positions,
            normals,
            uvs,
            colors: Vec::new(),
            ao: Vec::new(),
            indices,
            material_ids: Vec::new(),
        }
    }

    fn make_disconnected_leaves(count: usize) -> SourceMesh {
        let mut positions = Vec::with_capacity(count * 4);
        let mut normals = Vec::with_capacity(count * 4);
        let mut uvs = Vec::with_capacity(count * 4);
        let mut indices = Vec::with_capacity(count * 6);
        let mut material_ids = Vec::with_capacity(count * 2);
        let cols = 32_usize;
        for idx in 0..count {
            let cx = ((idx % cols) as f32) * 0.15;
            let cy = ((idx / cols) as f32) * 0.15;
            let cz = ((idx * 7) % 13) as f32 * 0.04;
            let s = 0.04 + ((idx % 5) as f32) * 0.01;
            let base = (idx * 4) as u32;
            positions.push(Vec3::new(cx, cy, cz));
            positions.push(Vec3::new(cx + s, cy, cz + 0.005));
            positions.push(Vec3::new(cx + s, cy + s, cz));
            positions.push(Vec3::new(cx, cy + s, cz - 0.005));
            for _ in 0..4 {
                normals.push(Vec3::Z);
            }
            uvs.extend_from_slice(&[
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(1.0, 1.0),
                Vec2::new(0.0, 1.0),
            ]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            let mat = (idx % 2) as u8;
            material_ids.push(mat);
            material_ids.push(mat);
        }
        SourceMesh {
            positions,
            normals,
            uvs,
            colors: Vec::new(),
            ao: Vec::new(),
            indices,
            material_ids,
        }
    }

    #[test]
    fn open_cloth_grid_reduces_cleanly_without_stalling() {
        let mesh = make_cloth_grid(64, 64);
        let built = build(&mesh).expect("cloth grid meshlet build must succeed");
        let report = verify(&built.manifest, &built.page_words)
            .expect("cloth grid meshlet verify must succeed");
        assert!(
            report.lod_levels <= 10,
            "open cloth grid must reduce cleanly without stalling (got {} LODs)",
            report.lod_levels
        );
        assert!(
            report.cullable_cone_clusters * 100 >= report.cluster_count * 85,
            "wavy cloth clusters should have >= 85% cullable cones"
        );
        assert!(
            report.max_quant_error < 1.0e-3,
            "sub-millimeter quantization error expected (got {})",
            report.max_quant_error
        );
        let raw = built.manifest.to_bytes();
        let roundtrip = MeshletManifest::from_bytes(&raw).expect("manifest roundtrip must parse");
        assert_eq!(
            roundtrip, built.manifest,
            "v3 manifest binary roundtrip must be bit-identical"
        );
    }

    #[test]
    fn disconnected_foliage_islands_prune_without_stalling() {
        let mesh = make_disconnected_leaves(1200);
        let built = build(&mesh).expect("foliage islands build must succeed");
        let report = verify(&built.manifest, &built.page_words)
            .expect("foliage islands verify must succeed");
        assert!(
            report.lod_levels <= 12,
            "disconnected leaves must reduce via island pruning (got {} LODs)",
            report.lod_levels
        );
        assert!(
            report.root_cluster_count <= 20,
            "disconnected leaves must collapse to compact roots (got {})",
            report.root_cluster_count
        );
    }

    #[test]
    fn shared_boundary_vertices_quantize_bit_identically() {
        let base_step = compute_base_quant_step(Vec3::new(-18.0, 0.0, -10.0), Vec3::new(18.0, 15.0, 10.0));
        let shared_p = Vec3::new(3.123_456, 2.654_321, -1.357_913);
        let v_shared = ClusterVertex {
            position: shared_p,
            ..ClusterVertex::default()
        };
        let verts_a = [
            v_shared,
            ClusterVertex {
                position: shared_p + Vec3::new(-0.5, 0.1, 0.0),
                ..ClusterVertex::default()
            },
            ClusterVertex {
                position: shared_p + Vec3::new(0.0, 0.4, -0.3),
                ..ClusterVertex::default()
            },
        ];
        let verts_b = [
            v_shared,
            ClusterVertex {
                position: shared_p + Vec3::new(0.7, -0.2, 0.4),
                ..ClusterVertex::default()
            },
            ClusterVertex {
                position: shared_p + Vec3::new(0.2, 0.5, 0.6),
                ..ClusterVertex::default()
            },
        ];
        let tris = [[0, 1, 2]];
        let mut buf_a = Vec::new();
        let mut buf_b = Vec::new();
        encode_cluster_words(&verts_a, &tris, base_step, &mut buf_a).expect("encode A");
        encode_cluster_words(&verts_b, &tris, base_step, &mut buf_b).expect("encode B");
        let meta = ClusterHeader::pack_meta(3, 1, 0, 0);
        let hdr = ClusterHeader {
            errors_and_cone: [0, ROOT_PARENT_ERROR.to_bits(), 0, meta],
            ..ClusterHeader::default()
        };
        let dec_a = decode_cluster_words(&hdr, &buf_a).expect("decode A");
        let dec_b = decode_cluster_words(&hdr, &buf_b).expect("decode B");
        assert_eq!(
            dec_a.vertices[0].position.to_array().map(f32::to_bits),
            dec_b.vertices[0].position.to_array().map(f32::to_bits),
            "adjacent clusters must reconstruct shared boundary vertices bit-identically"
        );
    }

    #[test]
    fn flat_floor_slab_preserves_90_degree_corners_at_root() {
        let nx = 32_usize;
        let nz = 32_usize;
        let mut positions = Vec::with_capacity((nx + 1) * (nz + 1));
        let mut normals = Vec::with_capacity((nx + 1) * (nz + 1));
        let mut uvs = Vec::with_capacity((nx + 1) * (nz + 1));
        for j in 0..=nz {
            let v = (j as f32) / (nz as f32);
            for i in 0..=nx {
                let u = (i as f32) / (nx as f32);
                positions.push(Vec3::new(u * 4.0, 0.0, v * 4.0));
                normals.push(Vec3::Y);
                uvs.push(Vec2::new(u, v));
            }
        }
        let mut indices = Vec::with_capacity(nx * nz * 6);
        let stride = (nx + 1) as u32;
        for j in 0..nz as u32 {
            for i in 0..nx as u32 {
                let a = j * stride + i;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                indices.extend_from_slice(&[a, d, b, a, c, d]);
            }
        }
        let mesh = SourceMesh {
            positions,
            normals,
            uvs,
            colors: Vec::new(),
            ao: Vec::new(),
            indices,
            material_ids: Vec::new(),
        };
        let built = build(&mesh).expect("floor slab build must succeed");
        verify(&built.manifest, &built.page_words).expect("floor slab verify must succeed");
        let mut root_pts = Vec::new();
        for h in &built.manifest.clusters {
            if !h.is_root() {
                continue;
            }
            let p_start = (h.page_id() as usize) * PAGE_WORDS;
            let page = &built.page_words[p_start..p_start + PAGE_WORDS];
            let dec = decode_cluster_words(h, page).expect("decode root cluster");
            for v in dec.vertices {
                root_pts.push(v.position);
            }
        }
        let corners = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 4.0),
            Vec3::new(0.0, 0.0, 4.0),
        ];
        for corner in corners {
            let best = root_pts
                .iter()
                .map(|p| (*p - corner).length())
                .fold(f32::INFINITY, engine_math::min);
            assert!(
                best < 0.02,
                "90-degree slab corner {:?} must not be chopped at root LOD (nearest={})",
                corner,
                best
            );
        }
    }

    #[test]
    fn large_disconnected_banners_are_not_pruned() {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();
        for banner in 0..4_usize {
            let ox = (banner as f32) * 3.0;
            let base = positions.len() as u32;
            let nx = 16_usize;
            let ny = 16_usize;
            for j in 0..=ny {
                let v = (j as f32) / (ny as f32);
                for i in 0..=nx {
                    let u = (i as f32) / (nx as f32);
                    positions.push(Vec3::new(ox + u * 2.0, v * 3.0, 0.0));
                    normals.push(Vec3::Z);
                    uvs.push(Vec2::new(u, v));
                }
            }
            let stride = (nx + 1) as u32;
            for j in 0..ny as u32 {
                for i in 0..nx as u32 {
                    let a = base + j * stride + i;
                    let b = a + 1;
                    let c = a + stride;
                    let d = c + 1;
                    indices.extend_from_slice(&[a, b, d, a, d, c]);
                }
            }
        }
        let mesh = SourceMesh {
            positions,
            normals,
            uvs,
            colors: Vec::new(),
            ao: Vec::new(),
            indices,
            material_ids: Vec::new(),
        };
        let built = build(&mesh).expect("banners build must succeed");
        verify(&built.manifest, &built.page_words).expect("banners verify must succeed");
        let mut root_xs = Vec::new();
        for h in &built.manifest.clusters {
            if !h.is_root() {
                continue;
            }
            let p_start = (h.page_id() as usize) * PAGE_WORDS;
            let page = &built.page_words[p_start..p_start + PAGE_WORDS];
            let dec = decode_cluster_words(h, page).expect("decode root cluster");
            for v in dec.vertices {
                root_xs.push(v.position.x);
            }
        }
        for banner in 0..4_usize {
            let lo = (banner as f32) * 3.0 - 0.1;
            let hi = (banner as f32) * 3.0 + 2.1;
            assert!(
                root_xs.iter().any(|&x| x >= lo && x <= hi),
                "large 2m x 3m banner {} must survive in root LOD",
                banner
            );
        }
    }

    fn make_curved_vine_tube(segs: usize, sides: usize) -> SourceMesh {
        let mut positions = Vec::with_capacity((segs + 1) * sides);
        let mut normals = Vec::with_capacity((segs + 1) * sides);
        let mut uvs = Vec::with_capacity((segs + 1) * sides);
        let radius = 0.015_f32;
        for s in 0..=segs {
            let t = (s as f32) / (segs as f32);
            let angle = t * TAU * 0.75;
            let center = Vec3::new(t * 4.0, sin(angle) * 0.45, cos(angle) * 0.35);
            let tangent = Vec3::new(
                4.0,
                cos(angle) * TAU * 0.75 * 0.45,
                -sin(angle) * TAU * 0.75 * 0.35,
            )
            .normalize();
            let up = if tangent.y.abs() < 0.9 {
                Vec3::Y
            } else {
                Vec3::Z
            };
            let right = tangent.cross(up).normalize();
            let binorm = right.cross(tangent).normalize();
            for k in 0..sides {
                let phi = (k as f32) / (sides as f32) * TAU;
                let dir = right * cos(phi) + binorm * sin(phi);
                positions.push(center + dir * radius);
                normals.push(dir);
                uvs.push(Vec2::new(t * 8.0, (k as f32) / (sides as f32)));
            }
        }
        let mut indices = Vec::with_capacity(segs * sides * 6);
        for s in 0..segs as u32 {
            for k in 0..sides as u32 {
                let k1 = (k + 1) % (sides as u32);
                let a = s * (sides as u32) + k;
                let b = s * (sides as u32) + k1;
                let c = (s + 1) * (sides as u32) + k;
                let d = (s + 1) * (sides as u32) + k1;
                indices.extend_from_slice(&[a, b, d, a, d, c]);
            }
        }
        SourceMesh {
            positions,
            normals,
            uvs,
            colors: Vec::new(),
            ao: Vec::new(),
            indices,
            material_ids: Vec::new(),
        }
    }

    #[test]
    fn thin_curved_vine_tube_does_not_inflate_edges() {
        let mesh = make_curved_vine_tube(160, 8);
        let built = build(&mesh).expect("vine tube build must succeed");
        verify(&built.manifest, &built.page_words).expect("vine tube verify must succeed");
        let mut lod_areas = std::collections::BTreeMap::<u32, f32>::new();
        let mut max_edge = 0.0_f32;
        for h in &built.manifest.clusters {
            let p_start = (h.page_id() as usize) * PAGE_WORDS;
            let page = &built.page_words[p_start..p_start + PAGE_WORDS];
            let dec = decode_cluster_words(h, page).expect("decode vine cluster");
            let mut area = 0.0_f32;
            for &[a, b, c] in &dec.triangles {
                let pa = dec.vertices[usize::from(a)].position;
                let pb = dec.vertices[usize::from(b)].position;
                let pc = dec.vertices[usize::from(c)].position;
                max_edge = max(max_edge, (pb - pa).length());
                max_edge = max(max_edge, (pc - pb).length());
                max_edge = max(max_edge, (pa - pc).length());
                area += 0.5 * (pb - pa).cross(pc - pa).length();
            }
            *lod_areas.entry(h.lod_level()).or_insert(0.0) += area;
        }
        assert!(
            max_edge <= 0.35,
            "thin vine tube must never inflate edges past 0.35m (got {max_edge})"
        );
        let leaf_area = lod_areas.get(&0).copied().unwrap_or(0.0);
        for (lod, area) in lod_areas {
            assert!(
                area <= leaf_area * 1.15 + 1.0e-3,
                "vine tube LOD {lod} surface area {area} exceeded 1.15x leaf area {leaf_area}"
            );
        }
    }

    #[test]
    fn alcove_curb_and_wall_seam_preserve_coverage() {
        let nx = 24_usize;
        let nz = 32_usize;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();
        let mut material_ids = Vec::new();
        for j in 0..=nz {
            let z = -3.0 + 6.0 * (j as f32) / (nz as f32);
            for i in 0..=nx {
                let u = (i as f32) / (nx as f32);
                if i < 4 {
                    let y = -0.30 + 0.32 * (i as f32) / 3.0;
                    positions.push(Vec3::new(0.0, y, z));
                    normals.push(Vec3::X);
                } else {
                    let x = 4.0 * ((i - 3) as f32) / ((nx - 3) as f32);
                    positions.push(Vec3::new(x, 0.02, z));
                    normals.push(Vec3::Y);
                }
                uvs.push(Vec2::new(u, (j as f32) / (nz as f32)));
            }
        }
        let stride = (nx + 1) as u32;
        for j in 0..nz as u32 {
            for i in 0..nx as u32 {
                let a = j * stride + i;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                indices.extend_from_slice(&[a, d, b, a, c, d]);
                material_ids.extend_from_slice(&[0, 0]);
            }
        }
        let wall_base = positions.len() as u32;
        for j in 0..=nz {
            let z = -3.0 + 6.0 * (j as f32) / (nz as f32);
            for k in 0..=12_usize {
                let y = -0.30 + 3.30 * (k as f32) / 12.0;
                positions.push(Vec3::new(-0.10, y, z));
                normals.push(Vec3::X);
                uvs.push(Vec2::new((k as f32) / 12.0, (j as f32) / (nz as f32)));
            }
        }
        for j in 0..nz as u32 {
            for k in 0..12_u32 {
                let a = wall_base + j * 13 + k;
                let b = a + 1;
                let c = a + 13;
                let d = c + 1;
                indices.extend_from_slice(&[a, b, d, a, d, c]);
                material_ids.extend_from_slice(&[1, 1]);
            }
        }
        let mesh = SourceMesh {
            positions,
            normals,
            uvs,
            colors: Vec::new(),
            ao: Vec::new(),
            indices,
            material_ids,
        };
        let built = build(&mesh).expect("alcove curb build must succeed");
        verify(&built.manifest, &built.page_words).expect("alcove curb verify must succeed");
        let mut min_curb_y = f32::INFINITY;
        let mut curb_pts = 0_usize;
        for h in &built.manifest.clusters {
            if !h.is_root() || h.material_id() != 0 {
                continue;
            }
            let p_start = (h.page_id() as usize) * PAGE_WORDS;
            let page = &built.page_words[p_start..p_start + PAGE_WORDS];
            let dec = decode_cluster_words(h, page).expect("decode curb root");
            for v in dec.vertices {
                if v.position.x < 0.05 && v.position.y < 0.0 {
                    curb_pts += 1;
                    if v.position.y < min_curb_y {
                        min_curb_y = v.position.y;
                    }
                }
            }
        }
        assert!(curb_pts >= 2, "root curb must retain vertical curb vertices");
        assert!(
            min_curb_y <= -0.28,
            "root curb must reach down to y=-0.30 without pull-up (got {min_curb_y})"
        );
    }

    #[test]
    fn balustrade_strip_preserves_area_at_root() {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();
        let mut material_ids = Vec::new();
        let nx = 64_usize;
        let nz = 8_usize;
        for j in 0..=nz {
            let z = (j as f32) / (nz as f32) * 0.4;
            for i in 0..=nx {
                let x = (i as f32) / (nx as f32) * 8.0;
                positions.push(Vec3::new(x, 6.0, z));
                normals.push(Vec3::Y);
                uvs.push(Vec2::new(x * 0.25, z));
            }
        }
        let stride = (nx + 1) as u32;
        for j in 0..nz as u32 {
            for i in 0..nx as u32 {
                let a = j * stride + i;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                indices.extend_from_slice(&[a, d, b, a, c, d]);
                material_ids.extend_from_slice(&[0, 0]);
            }
        }
        for post in 0..16_usize {
            let px = 0.25 + (post as f32) * 0.5;
            let base = positions.len() as u32;
            let corners = [
                Vec3::new(px - 0.13, 6.0, 0.07),
                Vec3::new(px + 0.13, 6.0, 0.07),
                Vec3::new(px + 0.13, 6.0, 0.33),
                Vec3::new(px - 0.13, 6.0, 0.33),
                Vec3::new(px - 0.13, 6.7, 0.07),
                Vec3::new(px + 0.13, 6.7, 0.07),
                Vec3::new(px + 0.13, 6.7, 0.33),
                Vec3::new(px - 0.13, 6.7, 0.33),
            ];
            for c in corners {
                positions.push(c);
                normals.push(Vec3::Y);
                uvs.push(Vec2::ZERO);
            }
            let faces = [
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
                [4, 5, 6],
                [4, 6, 7],
            ];
            for f in faces {
                indices.extend_from_slice(&[base + f[0], base + f[1], base + f[2]]);
                material_ids.push(0);
            }
        }
        let mesh = SourceMesh {
            positions,
            normals,
            uvs,
            colors: Vec::new(),
            ao: Vec::new(),
            indices,
            material_ids,
        };
        let built = build(&mesh).expect("balustrade build must succeed");
        verify(&built.manifest, &built.page_words).expect("balustrade verify must succeed");
        let mut leaf_area = 0.0_f32;
        let mut root_area = 0.0_f32;
        let mut root_xs = Vec::new();
        for h in &built.manifest.clusters {
            if h.lod_level() != 0 && !h.is_root() {
                continue;
            }
            let p_start = (h.page_id() as usize) * PAGE_WORDS;
            let page = &built.page_words[p_start..p_start + PAGE_WORDS];
            let dec = decode_cluster_words(h, page).expect("decode balustrade cluster");
            let mut area = 0.0_f32;
            for &[a, b, c] in &dec.triangles {
                let pa = dec.vertices[usize::from(a)].position;
                let pb = dec.vertices[usize::from(b)].position;
                let pc = dec.vertices[usize::from(c)].position;
                area += 0.5 * (pb - pa).cross(pc - pa).length();
            }
            if h.lod_level() == 0 {
                leaf_area += area;
            }
            if h.is_root() {
                root_area += area;
                for v in dec.vertices {
                    root_xs.push(v.position.x);
                }
            }
        }
        assert!(
            root_area >= leaf_area * 0.85,
            "balustrade root area {root_area} must retain >= 85% of leaf area {leaf_area}"
        );
        for span in 0..8_usize {
            let lo = (span as f32) - 0.1;
            let hi = (span as f32) + 1.1;
            assert!(
                root_xs.iter().any(|&x| x >= lo && x <= hi),
                "balustrade root must cover X-span [{lo}, {hi}]"
            );
        }
    }

    #[test]
    fn closed_manifold_alcove_corner_does_not_bulge() {
        let ny = 16_usize;
        let nx = 8_usize;
        let nz = 24_usize;
        let total_u = nz + nx;
        let mut positions = Vec::with_capacity((ny + 1) * (total_u + 1));
        let mut normals = Vec::with_capacity((ny + 1) * (total_u + 1));
        let mut uvs = Vec::with_capacity((ny + 1) * (total_u + 1));
        for j in 0..=ny {
            let y = 3.0 * (j as f32) / (ny as f32);
            for i in 0..=total_u {
                if i <= nz {
                    let z = -2.0 + 4.0 * (i as f32) / (nz as f32);
                    positions.push(Vec3::new(-1.0, y, z));
                    normals.push(Vec3::X);
                } else {
                    let x = -1.0 + 3.0 * ((i - nz) as f32) / (nx as f32);
                    positions.push(Vec3::new(x, y, 2.0));
                    normals.push(Vec3::NEG_Z);
                }
                uvs.push(Vec2::new((i as f32) / (total_u as f32), (j as f32) / (ny as f32)));
            }
        }
        let mut indices = Vec::with_capacity(ny * total_u * 6);
        let stride = (total_u + 1) as u32;
        for j in 0..ny as u32 {
            for i in 0..total_u as u32 {
                let a = j * stride + i;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                indices.extend_from_slice(&[a, b, d, a, d, c]);
            }
        }
        let mesh = SourceMesh {
            positions,
            normals,
            uvs,
            colors: Vec::new(),
            ao: Vec::new(),
            indices,
            material_ids: Vec::new(),
        };
        let built = build(&mesh).expect("alcove corner build must succeed");
        verify(&built.manifest, &built.page_words).expect("alcove corner verify must succeed");
        for h in &built.manifest.clusters {
            if !h.is_root() {
                continue;
            }
            let p_start = (h.page_id() as usize) * PAGE_WORDS;
            let page = &built.page_words[p_start..p_start + PAGE_WORDS];
            let dec = decode_cluster_words(h, page).expect("decode alcove root");
            for &[a, b, c] in &dec.triangles {
                let pa = dec.vertices[usize::from(a)].position;
                let pb = dec.vertices[usize::from(b)].position;
                let pc = dec.vertices[usize::from(c)].position;
                let min_x = min(pa.x, min(pb.x, pc.x));
                let max_x = max(pa.x, max(pb.x, pc.x));
                let min_z = min(pa.z, min(pb.z, pc.z));
                let max_z = max(pa.z, max(pb.z, pc.z));
                if min_z < 1.0 && max_z > -1.0 {
                    assert!(
                        !(min_x < -0.90 && max_x > -0.90),
                        "90-degree alcove corner must not bulge across center z=[-1,1] (x={min_x}..{max_x})"
                    );
                }
            }
        }
    }

    #[test]
    fn fallback_lod_jumps_are_bounded() {
        let mesh = make_cloth_grid(96, 64);
        let built = build(&mesh).expect("fallback bound build must succeed");
        verify(&built.manifest, &built.page_words).expect("fallback bound verify must succeed");
        for (ci, h) in built.manifest.clusters.iter().enumerate() {
            let fb_s = h.fallback_start() as usize;
            let fb_c = h.fallback_count() as usize;
            for p in fb_s..fb_s + fb_c {
                let parent = &built.manifest.clusters[p];
                let jump = parent.lod_level().saturating_sub(h.lod_level());
                assert!(
                    parent.lod_level() > h.lod_level() && jump <= 2,
                    "cluster {ci} (lod={}) fallback parent {p} (lod={}) jumped > 2 LODs",
                    h.lod_level(),
                    parent.lod_level()
                );
            }
        }
    }

    fn make_grid_meshlet(
        ox: f32,
        oy: f32,
        size: f32,
        n: usize,
        pos_base: u32,
    ) -> crate::partition::MeshletData {
        let mut vertices = Vec::with_capacity((n + 1) * (n + 1));
        let mut pos_ids = Vec::with_capacity((n + 1) * (n + 1));
        for j in 0..=n {
            let fy = (j as f32) / (n as f32);
            for i in 0..=n {
                let fx = (i as f32) / (n as f32);
                let id = pos_base + (j * (n + 1) + i) as u32;
                vertices.push(ClusterVertex {
                    position: Vec3::new(ox + fx * size, oy, fx * 0.01 + fy * size),
                    normal: Vec3::Y,
                    uv: Vec2::new(fx, fy),
                    ..ClusterVertex::default()
                });
                pos_ids.push(id);
            }
        }
        let mut triangles = Vec::with_capacity(n * n * 2);
        let stride = (n + 1) as u8;
        for j in 0..n as u8 {
            for i in 0..n as u8 {
                let a = j * stride + i;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                triangles.push([a, d, b]);
                triangles.push([a, c, d]);
            }
        }
        crate::partition::MeshletData {
            vertices,
            pos_ids,
            triangles,
        }
    }

    #[test]
    fn hard_locked_sub_patch_preserved_when_relax_locks_true() {
        use engine_math::collections::{FixedState, HashSet};
        let capital = make_grid_meshlet(0.0, 2.5, 1.2, 5, 0);
        let plinth_a = make_grid_meshlet(0.0, 0.0, 0.16, 4, 100);
        let plinth_b = make_grid_meshlet(0.4, 0.0, 0.16, 4, 200);
        let mut hard_locked = HashSet::with_hasher(FixedState::default());
        hard_locked.insert(100);
        hard_locked.insert(200);
        let seam_locked = HashSet::with_hasher(FixedState::default());
        let out = crate::simplify::simplify_group(
            &[&capital, &plinth_a, &plinth_b],
            &hard_locked,
            &seam_locked,
            true,
            12,
            12,
        );
        let plinth_a_tris = out
            .triangles
            .iter()
            .filter(|t| {
                t.iter()
                    .all(|&vi| out.vertices[vi as usize].position.y < 0.5 && out.vertices[vi as usize].position.x < 0.25)
            })
            .count();
        let plinth_b_tris = out
            .triangles
            .iter()
            .filter(|t| {
                t.iter()
                    .all(|&vi| out.vertices[vi as usize].position.y < 0.5 && out.vertices[vi as usize].position.x > 0.30)
            })
            .count();
        assert!(
            plinth_a_tris > 0 && plinth_b_tris > 0,
            "hard-locked sub-patches must survive relax_locks=true (a={plinth_a_tris}, b={plinth_b_tris})"
        );
    }

    #[test]
    fn twelve_cm_open_bordered_step_sill_preserved_at_root() {
        let nx = 64_usize;
        let nz = 12_usize;
        let mut positions = Vec::with_capacity((nx + 1) * (nz + 1));
        let mut normals = Vec::with_capacity((nx + 1) * (nz + 1));
        let mut uvs = Vec::with_capacity((nx + 1) * (nz + 1));
        for j in 0..=nz {
            let v = (j as f32) / (nz as f32);
            for i in 0..=nx {
                let u = (i as f32) / (nx as f32);
                let x = -3.0 + u * 4.0;
                let (y, z, n) = if j <= 2 {
                    let t = (j as f32) / 2.0;
                    (-0.08 + 0.12 * t, 5.94, Vec3::NEG_Z)
                } else {
                    let t = ((j - 2) as f32) / ((nz - 2) as f32);
                    let end_drop = if u > 0.85 { ((u - 0.85) / 0.15) * 0.11 } else { 0.0 };
                    (0.04 - 0.01 * t - end_drop * t, 5.94 + t * 0.22, Vec3::Y)
                };
                positions.push(Vec3::new(x, y, z));
                normals.push(n);
                uvs.push(Vec2::new(u, v));
            }
        }
        let mut indices = Vec::with_capacity(nx * nz * 6);
        let stride = (nx + 1) as u32;
        for j in 0..nz as u32 {
            for i in 0..nx as u32 {
                let a = j * stride + i;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                indices.extend_from_slice(&[a, d, b, a, c, d]);
            }
        }
        let mesh = SourceMesh {
            positions,
            normals,
            uvs,
            colors: Vec::new(),
            ao: Vec::new(),
            indices,
            material_ids: Vec::new(),
        };
        let built = build(&mesh).expect("step sill build must succeed");
        verify(&built.manifest, &built.page_words).expect("step sill verify must succeed");
        let mut ridge_max_y = f32::NEG_INFINITY;
        for h in &built.manifest.clusters {
            if !h.is_root() {
                continue;
            }
            let p_start = (h.page_id() as usize) * PAGE_WORDS;
            let page = &built.page_words[p_start..p_start + PAGE_WORDS];
            let dec = decode_cluster_words(h, page).expect("decode sill root");
            for v in dec.vertices {
                if v.position.x >= -2.2 && v.position.x <= -0.8 && v.position.z <= 5.98 {
                    ridge_max_y = max(ridge_max_y, v.position.y);
                }
            }
        }
        assert!(
            ridge_max_y >= 0.025,
            "12cm doorway sill top crease ridge must stay at y>=0.025 at root LOD (got {ridge_max_y})"
        );
    }

    #[test]
    fn small_radius_open_cylinder_shaft_preserved_at_root() {
        let cols = 4_usize;
        let segs_y = 14_usize;
        let sides = 20_usize;
        let radius = 0.17_f32;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();
        for c in 0..cols {
            let cx = (c as f32) * 6.0;
            let base = positions.len() as u32;
            for iy in 0..=segs_y {
                let ty = (iy as f32) / (segs_y as f32);
                let y = 6.2 + ty * 2.0;
                for k in 0..sides {
                    let phi = (k as f32) / (sides as f32) * TAU;
                    let dir = Vec3::new(cos(phi), 0.0, sin(phi));
                    positions.push(Vec3::new(cx, y, 0.0) + dir * radius);
                    normals.push(dir);
                    uvs.push(Vec2::new((k as f32) / (sides as f32), ty));
                }
            }
            for iy in 0..segs_y as u32 {
                for k in 0..sides as u32 {
                    let k1 = (k + 1) % (sides as u32);
                    let a = base + iy * (sides as u32) + k;
                    let b = base + iy * (sides as u32) + k1;
                    let c_idx = base + (iy + 1) * (sides as u32) + k;
                    let d = base + (iy + 1) * (sides as u32) + k1;
                    indices.extend_from_slice(&[a, b, d, a, d, c_idx]);
                }
            }
        }
        let mesh = SourceMesh {
            positions,
            normals,
            uvs,
            colors: Vec::new(),
            ao: Vec::new(),
            indices,
            material_ids: Vec::new(),
        };
        let built = build(&mesh).expect("open cylinder build must succeed");
        verify(&built.manifest, &built.page_words).expect("open cylinder verify must succeed");
        for col in 0..cols {
            let cx = (col as f32) * 6.0;
            let mut min_x = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut min_z = f32::INFINITY;
            let mut max_z = f32::NEG_INFINITY;
            for h in &built.manifest.clusters {
                if !h.is_root() {
                    continue;
                }
                let p_start = (h.page_id() as usize) * PAGE_WORDS;
                let page = &built.page_words[p_start..p_start + PAGE_WORDS];
                let dec = decode_cluster_words(h, page).expect("decode cylinder root");
                for v in dec.vertices {
                    if (v.position.x - cx).abs() <= 1.0 && v.position.y >= 6.8 && v.position.y <= 7.8 {
                        min_x = min(min_x, v.position.x);
                        max_x = max(max_x, v.position.x);
                        min_z = min(min_z, v.position.z);
                        max_z = max(max_z, v.position.z);
                    }
                }
            }
            let dx = max_x - min_x;
            let dz = max_z - min_z;
            assert!(
                dx >= 0.22 && dz >= 0.22,
                "0.17m-radius cylinder {col} mid-shaft must stay cylindrical at root LOD (dx={dx}, dz={dz})"
            );
        }
    }
}
