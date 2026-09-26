use std::sync::Arc;

use glam::Vec3;

use crate::*;

fn triangle() -> Vec<MeshVertex> {
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        .map(|position| MeshVertex {
            position,
            normal: [0.0, 0.0, 1.0],
            uv: [position[0], position[1]],
        })
        .to_vec()
}

#[test]
fn immutable_mesh_bounds_include_all_vertices_without_charging_metadata_as_payload() {
    let mut vertices = triangle();
    vertices.push(MeshVertex {
        position: [-3.0, 5.0, 7.0],
        normal: [0.0, 1.0, 0.0],
        uv: [0.0; 2],
    });
    let mesh = Mesh::new(vertices, vec![0, 1, 2]).unwrap();
    assert_eq!(mesh.bounds().min(), Vec3::new(-3.0, 0.0, 0.0));
    assert_eq!(mesh.bounds().max(), Vec3::new(1.0, 5.0, 7.0));
    assert_eq!(
        mesh.byte_len(),
        4 * size_of::<MeshVertex>() + 3 * size_of::<u32>()
    );
    assert_eq!(Mesh::cube().bounds().min(), Vec3::splat(-0.5));
    assert_eq!(Mesh::cube().bounds().max(), Vec3::splat(0.5));
    assert_eq!(Mesh::plane().bounds().min().y, 0.0);
    assert_eq!(Mesh::plane().bounds().max().y, 0.0);
}

#[test]
fn finite_extreme_mesh_bounds_never_require_overflowing_center_or_extent_math() {
    let mut vertices = triangle();
    vertices[0].position = [-f32::MAX; 3];
    vertices[1].position = [f32::MAX; 3];
    vertices[2].position = [f32::from_bits(1); 3];
    let mesh = Mesh::new(vertices, vec![0, 1, 2]).unwrap();
    assert_eq!(mesh.bounds().min(), Vec3::splat(-f32::MAX));
    assert_eq!(mesh.bounds().max(), Vec3::splat(f32::MAX));
    assert!(mesh.bounds().min().is_finite() && mesh.bounds().max().is_finite());
}

#[test]
fn mesh_validation_reports_the_affected_vertex_or_index() {
    assert_eq!(
        Mesh::new(vec![], vec![]).unwrap_err(),
        AssetError::EmptyMesh
    );
    assert_eq!(
        Mesh::new(triangle(), vec![0, 1]).unwrap_err(),
        AssetError::InvalidTriangleCount { indices: 2 }
    );
    assert_eq!(
        Mesh::new(triangle(), vec![]).unwrap_err().code(),
        "asset.invalid_triangle_count"
    );
    assert_eq!(
        Mesh::new(triangle(), vec![0, 1, 3]).unwrap_err(),
        AssetError::IndexOutOfBounds {
            index: 2,
            value: 3,
            vertices: 3
        }
    );
    let mut vertices = triangle();
    vertices[1].position[2] = f32::NAN;
    assert_eq!(
        Mesh::new(vertices, vec![0, 1, 2]).unwrap_err(),
        AssetError::InvalidVertex { vertex: 1 }
    );
    for normal in [[0.0; 3], [f32::INFINITY, 0.0, 1.0]] {
        let mut vertices = triangle();
        vertices[2].normal = normal;
        assert_eq!(
            Mesh::new(vertices, vec![0, 1, 2]).unwrap_err(),
            AssetError::InvalidNormal { vertex: 2 }
        );
    }
    for uv in [[f32::NAN, 0.0], [0.0, f32::INFINITY]] {
        let mut vertices = triangle();
        vertices[1].uv = uv;
        assert_eq!(
            Mesh::new(vertices, vec![0, 1, 2]).unwrap_err(),
            AssetError::InvalidUv { vertex: 1 }
        );
    }
    let mut vertices = triangle();
    vertices[0].uv = [-2.0, 3.0];
    assert!(Mesh::new(vertices, vec![0, 1, 2]).is_ok());
}

#[test]
fn mesh_normalization_handles_all_finite_nonzero_scales() {
    for magnitude in [2.0, f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1)] {
        let mut vertices = triangle();
        vertices[0].normal = [0.0, magnitude, 0.0];
        let mesh = Mesh::new(vertices, vec![0, 1, 2]).unwrap();
        assert_eq!(mesh.vertices()[0].normal, [0.0, 1.0, 0.0]);
    }
}

#[test]
fn builtin_geometry_has_consistent_winding_and_unit_bounds() {
    for mesh in [Mesh::cube(), Mesh::plane(), Mesh::octahedron()] {
        for vertex in mesh.vertices() {
            assert!(
                vertex
                    .position
                    .iter()
                    .all(|value| (-0.5..=0.5).contains(value))
            );
            assert!((Vec3::from_array(vertex.normal).length() - 1.0).abs() < 1e-6);
            assert!(vertex.uv.iter().all(|value| (0.0..=1.0).contains(value)));
        }
        for indices in mesh.indices().as_chunks::<3>().0 {
            let a = mesh.vertices()[indices[0] as usize];
            let b = mesh.vertices()[indices[1] as usize];
            let c = mesh.vertices()[indices[2] as usize];
            let normal = (Vec3::from_array(b.position) - Vec3::from_array(a.position))
                .cross(Vec3::from_array(c.position) - Vec3::from_array(a.position));
            assert!(normal.dot(Vec3::from_array(a.normal)) > 0.0);
            let uv_ab = glam::Vec2::from_array(b.uv) - glam::Vec2::from_array(a.uv);
            let uv_ac = glam::Vec2::from_array(c.uv) - glam::Vec2::from_array(a.uv);
            assert!(uv_ab.perp_dot(uv_ac).abs() > 0.0);
            if mesh.indices().len() != 6 {
                assert!(normal.dot(Vec3::from_array(a.position)) > 0.0);
            }
        }
    }
}

#[test]
fn material_accepts_only_finite_opaque_linear_color() {
    assert_eq!(
        Material::new([0.0, 0.25, 1.0, 1.0]).unwrap().color(),
        [0.0, 0.25, 1.0, 1.0]
    );
    for value in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
        assert_eq!(
            Material::new([0.0, value, 0.0, 1.0]).unwrap_err(),
            AssetError::InvalidColor { channel: 1 }
        );
    }
    assert_eq!(
        Material::new([1.0, 1.0, 1.0, 0.5]).unwrap_err(),
        AssetError::UnsupportedAlpha
    );
}

#[test]
fn handles_reject_removed_reused_and_foreign_slots() {
    let mut first = AssetLibrary::default();
    let mut second = AssetLibrary::default();
    let old_mesh = first.insert_mesh(Mesh::cube()).unwrap();
    let old_material = first
        .insert_material(Material::new([1.0; 4]).unwrap())
        .unwrap();
    let foreign_mesh = second.insert_mesh(Mesh::cube()).unwrap();
    let foreign_material = second
        .insert_material(Material::new([1.0; 4]).unwrap())
        .unwrap();
    assert!(first.mesh(foreign_mesh).is_none());
    assert!(first.material(foreign_material).is_none());
    assert!(!first.remove_mesh(foreign_mesh));
    assert!(!first.remove_material(foreign_material));
    assert!(first.remove_mesh(old_mesh));
    assert!(first.remove_material(old_material));
    let new_mesh = first.insert_mesh(Mesh::plane()).unwrap();
    let new_material = first
        .insert_material(Material::new([1.0; 4]).unwrap())
        .unwrap();
    assert_ne!(old_mesh, new_mesh);
    assert_ne!(old_material, new_material);
    assert!(first.mesh(old_mesh).is_none());
    assert!(first.material(old_material).is_none());
    assert!(!first.remove_mesh(old_mesh));
    assert!(!first.remove_material(old_material));
    assert_eq!(first.mesh_count(), 1);
    assert_eq!(first.material_count(), 1);
}

#[test]
fn capacity_and_payload_budgets_fail_without_partial_admission() {
    let mut library = AssetLibrary::new(AssetBudget {
        max_meshes: 1,
        max_materials: 1,
        max_textures: 0,
        max_cpu_bytes: Mesh::plane().byte_len() + size_of::<Material>(),
    });
    assert_eq!(
        library.insert_mesh(Mesh::cube()).unwrap_err().code(),
        "asset.cpu_budget"
    );
    assert_eq!(library.resident_bytes(), 0);
    let mesh = library.insert_mesh(Mesh::plane()).unwrap();
    let material = library
        .insert_material(Material::new([1.0; 4]).unwrap())
        .unwrap();
    assert_eq!(
        library.insert_mesh(Mesh::plane()).unwrap_err(),
        AssetError::MeshCapacity { limit: 1 }
    );
    assert_eq!(
        library
            .insert_material(Material::new([1.0; 4]).unwrap())
            .unwrap_err(),
        AssetError::MaterialCapacity { limit: 1 }
    );
    assert_eq!(
        library.resident_bytes(),
        Mesh::plane().byte_len() + size_of::<Material>()
    );
    library.remove_mesh(mesh);
    library.remove_material(material);
    assert_eq!(library.resident_bytes(), 0);
    assert_eq!(library.mesh_count(), 0);
    assert_eq!(library.material_count(), 0);
    assert!(library.insert_mesh(Mesh::plane()).is_ok());
}

#[test]
fn byte_budget_includes_materials_and_zero_limits_are_supported() {
    let mut library = AssetLibrary::new(AssetBudget {
        max_cpu_bytes: size_of::<Material>() - 1,
        ..AssetBudget::default()
    });
    assert_eq!(
        library
            .insert_material(Material::new([1.0; 4]).unwrap())
            .unwrap_err(),
        AssetError::CpuBudget {
            requested: size_of::<Material>(),
            available: size_of::<Material>() - 1
        }
    );
    assert_eq!(library.material_count(), 0);
    let mut disabled = AssetLibrary::new(AssetBudget {
        max_meshes: 0,
        max_materials: 0,
        max_textures: 0,
        max_cpu_bytes: 0,
    });
    assert_eq!(
        disabled.insert_mesh(Mesh::plane()).unwrap_err().code(),
        "asset.mesh_capacity"
    );
    assert_eq!(
        disabled
            .insert_material(Material::new([1.0; 4]).unwrap())
            .unwrap_err()
            .code(),
        "asset.material_capacity"
    );
    assert_eq!(
        disabled.insert_texture(one_pixel()).unwrap_err().code(),
        "asset.texture_capacity"
    );
}

#[test]
fn retained_source_survives_removal_and_library_disposal() {
    let mut library = AssetLibrary::default();
    let handle = library.insert_mesh(Mesh::cube()).unwrap();
    let source = Arc::clone(library.mesh(handle).unwrap());
    assert!(library.remove_mesh(handle));
    assert_eq!(library.resident_bytes(), 0);
    drop(library);
    assert_eq!(source.vertices().len(), 24);
    assert_eq!(source.indices().len(), 36);
}

fn one_pixel() -> Texture {
    Texture::new(1, 1, vec![10, 20, 30, 255]).unwrap()
}

#[test]
fn texture_rejects_invalid_dimensions_payload_and_alpha() {
    for (width, height) in [(0, 1), (1, 0), (2049, 1), (1, 2049), (u32::MAX, u32::MAX)] {
        assert_eq!(
            Texture::new(width, height, vec![]).unwrap_err(),
            AssetError::InvalidTextureDimensions {
                width,
                height,
                limit: MAX_TEXTURE_DIMENSION,
            }
        );
    }
    for actual in [0, 3, 5] {
        assert_eq!(
            Texture::new(1, 1, vec![255; actual]).unwrap_err(),
            AssetError::InvalidTextureLength {
                expected: 4,
                actual
            }
        );
    }
    assert_eq!(
        Texture::new(2, 1, vec![0, 0, 0, 255, 255, 255, 255, 254]).unwrap_err(),
        AssetError::UnsupportedTextureAlpha { pixel: 1 }
    );
    let texture = Texture::new(MAX_TEXTURE_DIMENSION, 1, vec![255; 2048 * 4]).unwrap();
    assert_eq!(texture.width(), MAX_TEXTURE_DIMENSION);
    assert_eq!(texture.height(), 1);
    assert_eq!(texture.byte_len(), 2048 * 4);
}

#[test]
fn texture_handles_are_weak_and_reject_removed_reused_and_foreign_slots() {
    let mut library = AssetLibrary::default();
    let mut other = AssetLibrary::default();
    let handle = library.insert_texture(one_pixel()).unwrap();
    let source = Arc::clone(library.texture(handle).unwrap());
    let foreign = other.insert_texture(one_pixel()).unwrap();
    assert!(library.texture(foreign).is_none());
    assert!(!library.remove_texture(foreign));
    assert!(library.remove_texture(handle));
    assert!(!library.remove_texture(handle));
    assert_eq!(library.texture_count(), 0);
    assert_eq!(library.texture_bytes(), 0);
    assert_eq!(library.resident_bytes(), 0);
    let replacement = library.insert_texture(one_pixel()).unwrap();
    assert_ne!(handle, replacement);
    assert!(library.texture(handle).is_none());
    assert!(library.texture(replacement).is_some());
    drop(library);
    assert_eq!(source.pixels(), [10, 20, 30, 255]);
}

#[test]
fn textured_material_admission_requires_live_texture_in_same_library() {
    let mut library = AssetLibrary::default();
    let mut other = AssetLibrary::default();
    let foreign = other.insert_texture(one_pixel()).unwrap();
    assert_eq!(
        library
            .insert_material(Material::textured([1.0; 4], foreign).unwrap())
            .unwrap_err(),
        AssetError::InvalidTextureHandle
    );
    assert_eq!(library.material_count(), 0);
    assert_eq!(library.resident_bytes(), 0);
    let texture = library.insert_texture(one_pixel()).unwrap();
    let material = Material::textured([0.5, 0.75, 1.0, 1.0], texture).unwrap();
    assert_eq!(material.texture(), Some(texture));
    assert_eq!(material.color(), [0.5, 0.75, 1.0, 1.0]);
    let material_handle = library.insert_material(material).unwrap();
    assert_eq!(Material::new([1.0; 4]).unwrap().texture(), None);
    assert!(library.remove_texture(texture));
    assert_eq!(
        library.material(material_handle).unwrap().texture(),
        Some(texture)
    );
    assert!(library.texture(texture).is_none());
    assert_eq!(
        library.insert_material(material).unwrap_err(),
        AssetError::InvalidTextureHandle
    );
    assert_eq!(library.material_count(), 1);
    assert_eq!(library.resident_bytes(), size_of::<Material>());
}

#[test]
fn texture_admission_obeys_count_and_shared_cpu_budget_without_partial_changes() {
    let mut library = AssetLibrary::new(AssetBudget {
        max_textures: 1,
        max_cpu_bytes: 4 + size_of::<Material>(),
        ..AssetBudget::default()
    });
    let texture = library.insert_texture(one_pixel()).unwrap();
    let material = library
        .insert_material(Material::textured([1.0; 4], texture).unwrap())
        .unwrap();
    assert_eq!(library.texture_count(), 1);
    assert_eq!(library.texture_bytes(), 4);
    assert_eq!(library.resident_bytes(), 4 + size_of::<Material>());
    assert_eq!(
        library.insert_texture(one_pixel()).unwrap_err(),
        AssetError::TextureCapacity { limit: 1 }
    );
    assert!(library.remove_texture(texture));
    let oversized = Texture::new(2, 1, vec![255; 8]).unwrap();
    assert_eq!(
        library.insert_texture(oversized).unwrap_err(),
        AssetError::CpuBudget {
            requested: 8,
            available: 4
        }
    );
    assert_eq!(library.texture_count(), 0);
    assert_eq!(library.texture_bytes(), 0);
    assert_eq!(library.resident_bytes(), size_of::<Material>());
    assert!(library.remove_material(material));
    assert!(
        library
            .insert_texture(Texture::new(2, 1, vec![255; 8]).unwrap())
            .is_ok()
    );
    assert_eq!(library.texture_bytes(), 8);
    assert_eq!(library.resident_bytes(), 8);
}
