use std::sync::Arc;

use glam::{Quat, Vec3};

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
fn materials_become_transparent_only_through_a_valid_opacity() {
    let opaque = Material::new([0.2, 0.4, 0.6, 1.0]).unwrap();
    assert!(!opaque.is_transparent());
    assert_eq!(opaque.opacity(), 1.0);
    let glass = opaque.with_opacity(0.35).unwrap();
    assert!(glass.is_transparent());
    assert_eq!(glass.color(), [0.2, 0.4, 0.6, 0.35]);
    assert!(!opaque.with_opacity(1.0).unwrap().is_transparent());
    for opacity in [0.0, -0.1, 1.01, f32::NAN, f32::INFINITY] {
        assert_eq!(
            opaque.with_opacity(opacity).unwrap_err(),
            AssetError::InvalidOpacity
        );
    }
    assert_eq!(AssetError::InvalidOpacity.code(), "asset.invalid_opacity");
}

#[test]
fn material_surfaces_default_to_matte_and_reject_values_outside_their_ranges() {
    let material = Material::new([1.0; 4]).unwrap();
    assert_eq!(material.surface(), Surface::MATTE);
    assert_eq!(Surface::default(), Surface::MATTE);
    let glass = Surface {
        roughness: 0.0,
        metalness: 1.0,
        emission: Surface::MAX_EMISSION,
        ripples: 1.0,
    };
    assert_eq!(
        material.with_surface(glass).unwrap().surface(),
        glass,
        "range ends are accepted"
    );
    for (field, surface) in [
        (
            "roughness",
            Surface {
                roughness: -0.1,
                ..glass
            },
        ),
        (
            "metalness",
            Surface {
                metalness: 1.5,
                ..glass
            },
        ),
        (
            "emission",
            Surface {
                emission: 16.5,
                ..glass
            },
        ),
        (
            "emission",
            Surface {
                emission: f32::INFINITY,
                ..glass
            },
        ),
        (
            "ripples",
            Surface {
                ripples: f32::NAN,
                ..glass
            },
        ),
    ] {
        assert_eq!(
            material.with_surface(surface).unwrap_err(),
            AssetError::InvalidSurface { field }
        );
    }
    assert_eq!(
        AssetError::InvalidSurface { field: "ripples" }.code(),
        "asset.invalid_surface"
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
    let over = MAX_TEXTURE_DIMENSION + 1;
    for (width, height) in [(0, 1), (1, 0), (over, 1), (1, over), (u32::MAX, u32::MAX)] {
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
    let widest = MAX_TEXTURE_DIMENSION as usize * 4;
    let texture = Texture::new(MAX_TEXTURE_DIMENSION, 1, vec![255; widest]).unwrap();
    assert_eq!(texture.width(), MAX_TEXTURE_DIMENSION);
    assert_eq!(texture.height(), 1);
    assert_eq!(texture.byte_len(), widest);
    // Emission masks accept any alpha, with the same payload checks.
    let masked = Texture::with_emission_mask(2, 1, vec![0, 0, 0, 0, 255, 255, 255, 128]).unwrap();
    assert_eq!(masked.pixels()[7], 128);
    assert!(masked.has_emission_mask());
    assert!(!texture.has_emission_mask());
    assert!(
        !Texture::with_emission_mask(1, 1, vec![9, 9, 9, 255])
            .unwrap()
            .has_emission_mask()
    );
    assert_eq!(
        Texture::with_emission_mask(0, 1, vec![]).unwrap_err(),
        AssetError::InvalidTextureDimensions {
            width: 0,
            height: 1,
            limit: MAX_TEXTURE_DIMENSION,
        }
    );
    assert_eq!(
        Texture::with_emission_mask(1, 1, vec![0; 3]).unwrap_err(),
        AssetError::InvalidTextureLength {
            expected: 4,
            actual: 3
        }
    );
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

#[test]
fn gltf_material_maps_validate_and_every_texture_must_be_live() {
    let mut library = AssetLibrary::default();
    let mut other = AssetLibrary::default();
    let texture = library.insert_texture(one_pixel()).unwrap();
    let foreign = other.insert_texture(one_pixel()).unwrap();
    let map = TextureRef::new(texture);
    // glTF's defaults: repeating trilinear sampling, no transform.
    assert_eq!(map.sampler, Sampler::REPEAT);
    assert_eq!(map.transform, UvTransform::IDENTITY);
    let plain = Material::new([1.0; 4]).unwrap();
    assert!(!plain.has_maps());
    let material = plain
        .with_base_color_texture(map.with_sampler(Sampler::CLAMP))
        .unwrap()
        .with_metallic_roughness_texture(map)
        .unwrap()
        .with_normal_texture(map, 0.5)
        .unwrap()
        .with_occlusion_texture(map, 0.25)
        .unwrap()
        .with_emissive([4.0, 2.0, 0.0])
        .unwrap()
        .with_emissive_texture(map)
        .unwrap();
    assert!(material.has_maps());
    assert_eq!(material.textures().count(), 5);
    assert_eq!(material.texture(), Some(texture));
    assert_eq!(
        (material.normal_scale(), material.occlusion_strength()),
        (0.5, 0.25)
    );
    library.insert_material(material).unwrap();
    // A foreign texture in any map is refused.
    let foreign_normal = plain
        .with_normal_texture(TextureRef::new(foreign), 1.0)
        .unwrap();
    assert_eq!(
        library.insert_material(foreign_normal).unwrap_err(),
        AssetError::InvalidTextureHandle
    );
    // Parameters stay within their ranges.
    let field = |error: AssetError| match error {
        AssetError::InvalidMaterialMap { field } => field,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        field(plain.with_normal_texture(map, f32::NAN).unwrap_err()),
        "normal_scale"
    );
    assert_eq!(
        field(plain.with_occlusion_texture(map, 1.5).unwrap_err()),
        "occlusion_strength"
    );
    assert_eq!(
        field(plain.with_emissive([-1.0, 0.0, 0.0]).unwrap_err()),
        "emissive"
    );
    let unplaced = map.with_transform(UvTransform {
        rotation: f32::INFINITY,
        ..UvTransform::IDENTITY
    });
    assert_eq!(
        field(plain.with_base_color_texture(unplaced).unwrap_err()),
        "transform"
    );
    assert_eq!(
        field(
            plain
                .with_alpha_mode(AlphaMode::Mask { cutoff: -0.5 })
                .unwrap_err()
        ),
        "alpha_cutoff"
    );
    assert_eq!(
        AssetError::InvalidMaterialMap { field: "emissive" }.code(),
        "asset.invalid_material_map"
    );
}

#[test]
fn alpha_modes_decide_transparency_and_which_materials_have_maps() {
    let plain = Material::new([0.2, 0.4, 0.6, 1.0]).unwrap();
    // An opacity below 1 blends, as before alpha modes.
    let glass = plain.with_opacity(0.4).unwrap();
    assert_eq!(glass.alpha_mode(), AlphaMode::Blend);
    assert!(glass.is_transparent() && !glass.has_maps());
    // A mask tests alpha instead: it is opaque, and binds its cutoff.
    let cutout = glass
        .with_alpha_mode(AlphaMode::Mask { cutoff: 0.5 })
        .unwrap();
    assert!(!cutout.is_transparent() && cutout.has_maps());
    assert_eq!(cutout.opacity(), 0.4);
    // Opaque materials ignore alpha.
    let opaque = glass.with_alpha_mode(AlphaMode::Opaque).unwrap();
    assert_eq!((opaque.opacity(), opaque.is_transparent()), (1.0, false));
    // Blending at full alpha reads texture alpha, so it binds.
    let blended = plain.with_alpha_mode(AlphaMode::Blend).unwrap();
    assert!(blended.is_transparent() && blended.has_maps());
    assert!(plain.with_double_sided(true).has_maps());
    assert!(plain.with_emissive([0.0, 0.1, 0.0]).unwrap().has_maps());
}

#[test]
fn materials_replace_in_place_keeping_their_handles() {
    let mut library = AssetLibrary::default();
    let mut other = AssetLibrary::default();
    let handle = library
        .insert_material(Material::new([1.0, 0.0, 0.0, 1.0]).unwrap())
        .unwrap();
    let before = library.revision();
    let bytes = library.resident_bytes();
    library
        .replace_material(handle, Material::new([0.0, 1.0, 0.0, 1.0]).unwrap())
        .unwrap();
    assert_eq!(
        library.material(handle).unwrap().color(),
        [0.0, 1.0, 0.0, 1.0]
    );
    assert_ne!(library.revision(), before);
    assert_eq!(library.resident_bytes(), bytes);
    // Its textures must be live here, and the handle live too.
    let foreign = other.insert_texture(one_pixel()).unwrap();
    assert_eq!(
        library
            .replace_material(handle, Material::textured([1.0; 4], foreign).unwrap())
            .unwrap_err(),
        AssetError::InvalidTextureHandle
    );
    assert!(library.remove_material(handle));
    assert_eq!(
        library
            .replace_material(handle, Material::new([1.0; 4]).unwrap())
            .unwrap_err(),
        AssetError::InvalidMaterialHandle
    );
}

#[test]
fn optional_vertex_colors_and_second_coordinates_validate() {
    let mesh = || Mesh::new(triangle(), vec![0, 1, 2]).unwrap();
    let colored = mesh()
        .with_colors(vec![[1.0, 0.5, 0.0, 1.0]; 3])
        .unwrap()
        .with_uv1(vec![[0.5, 0.5]; 3])
        .unwrap();
    assert_eq!(colored.colors().unwrap()[0], [1.0, 0.5, 0.0, 1.0]);
    assert_eq!(colored.uv1().unwrap().len(), 3);
    assert_eq!(colored.byte_len(), mesh().byte_len() + 3 * 16 + 3 * 8);
    assert!(mesh().colors().is_none() && mesh().uv1().is_none());
    assert_eq!(
        mesh().with_colors(vec![[1.0; 4]; 2]).unwrap_err(),
        AssetError::AttributeCount {
            attribute: "colors",
            expected: 3,
            actual: 2
        }
    );
    assert_eq!(
        mesh()
            .with_colors(vec![[1.0; 4], [2.0, 0.0, 0.0, 1.0], [1.0; 4]])
            .unwrap_err(),
        AssetError::InvalidAttribute {
            attribute: "colors",
            vertex: 1
        }
    );
    assert_eq!(
        mesh()
            .with_uv1(vec![[0.0; 2], [0.0; 2], [f32::NAN, 0.0]])
            .unwrap_err()
            .code(),
        "asset.invalid_attribute"
    );
    // Maps read the first or second coordinates only.
    let mut library = AssetLibrary::default();
    let texture = library.insert_texture(one_pixel()).unwrap();
    let plain = Material::new([1.0; 4]).unwrap();
    let second = TextureRef::new(texture).with_tex_coord(1);
    assert_eq!(
        plain
            .with_base_color_texture(second)
            .unwrap()
            .base_color_texture()
            .unwrap()
            .tex_coord,
        1
    );
    assert_eq!(
        plain
            .with_base_color_texture(second.with_tex_coord(2))
            .unwrap_err(),
        AssetError::InvalidMaterialMap { field: "tex_coord" }
    );
    // The set is part of the binding.
    assert_ne!(
        plain
            .with_base_color_texture(TextureRef::new(texture))
            .unwrap()
            .binding(),
        plain.with_base_color_texture(second).unwrap().binding()
    );
}

#[test]
fn animation_channels_sample_step_linear_and_cubic_splines_like_gltf() {
    let vectors =
        |values: &[f32]| Keyframes::Vectors(values.iter().map(|x| Vec3::splat(*x)).collect());
    let at = |channel: &AnimationChannel, time| match channel.sample(time) {
        AnimationSample::Translation(value) | AnimationSample::Scale(value) => value.x,
        AnimationSample::Rotation(_) => panic!("vector channel"),
    };
    let step = AnimationChannel::new(
        0,
        AnimatedProperty::Translation,
        KeyInterpolation::Step,
        vec![1.0, 2.0],
        vectors(&[3.0, 5.0]),
    )
    .unwrap();
    // Before the first key, between keys and after the last.
    assert_eq!(
        [at(&step, 0.0), at(&step, 1.5), at(&step, 9.0)],
        [3.0, 3.0, 5.0]
    );
    let linear = AnimationChannel::new(
        0,
        AnimatedProperty::Scale,
        KeyInterpolation::Linear,
        vec![0.0, 2.0],
        vectors(&[1.0, 3.0]),
    )
    .unwrap();
    assert_eq!([at(&linear, 0.5), at(&linear, 2.0)], [1.5, 3.0]);
    // Cubic splines: in-tangent, value, out-tangent per key; tangents are
    // per second and scale by the key span.
    let cubic = AnimationChannel::new(
        0,
        AnimatedProperty::Translation,
        KeyInterpolation::CubicSpline,
        vec![0.0, 1.0],
        vectors(&[0.0, 0.0, 2.0, 0.0, 1.0, 0.0]),
    )
    .unwrap();
    assert!((at(&cubic, 0.5) - 0.75).abs() < 1.0e-6);
    assert_eq!(at(&cubic, 1.0), 1.0);
    // Rotations take the shortest arc and stay unit length.
    let quarter = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let rotation = AnimationChannel::new(
        0,
        AnimatedProperty::Rotation,
        KeyInterpolation::Linear,
        vec![0.0, 1.0],
        Keyframes::Rotations(vec![Quat::IDENTITY, quarter * 2.0].into_boxed_slice()),
    )
    .unwrap();
    let AnimationSample::Rotation(half) = rotation.sample(0.5) else {
        panic!("rotation channel");
    };
    assert!(half.abs_diff_eq(Quat::from_rotation_y(std::f32::consts::FRAC_PI_4), 1.0e-5));
    // Clips span their latest key and bind as many targets as they name.
    let clip = AnimationClip::new(vec![
        step,
        AnimationChannel::new(
            3,
            AnimatedProperty::Rotation,
            KeyInterpolation::Step,
            vec![0.0, 4.5],
            Keyframes::Rotations(vec![Quat::IDENTITY; 2].into_boxed_slice()),
        )
        .unwrap(),
    ])
    .unwrap();
    assert_eq!((clip.duration(), clip.target_count()), (4.5, 4));
    let mut library = AssetLibrary::default();
    let handle = library.insert_clip(clip).unwrap();
    assert_eq!(library.clip(handle).unwrap().channels().len(), 2);
    assert!(library.resident_bytes() > 0);
    assert!(library.remove_clip(handle));
    assert_eq!(library.resident_bytes(), 0);
    // Invalid keys.
    let reject = |times: Vec<f32>, values| {
        AnimationChannel::new(
            0,
            AnimatedProperty::Scale,
            KeyInterpolation::Linear,
            times,
            values,
        )
        .unwrap_err()
        .code()
    };
    assert_eq!(
        reject(vec![1.0, 1.0], vectors(&[1.0, 2.0])),
        "asset.invalid_animation"
    );
    assert_eq!(
        reject(vec![0.0, 1.0], vectors(&[1.0])),
        "asset.invalid_animation"
    );
    assert_eq!(
        reject(
            vec![0.0],
            Keyframes::Rotations(vec![Quat::IDENTITY].into_boxed_slice())
        ),
        "asset.invalid_animation"
    );
    assert!(AnimationClip::new(Vec::new()).is_err());
}
