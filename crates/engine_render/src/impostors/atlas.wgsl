import package::engine::mesh::impostor_frames::ImpostorRecord;

@group(0) @binding(0) var<storage, read> impostor_records: array<ImpostorRecord>;
@group(0) @binding(1) var impostor_albedo: texture_2d_array<f32>;
@group(0) @binding(2) var impostor_emissive: texture_2d_array<f32>;
@group(0) @binding(3) var impostor_geometry: texture_2d_array<f32>;
@group(0) @binding(4) var impostor_surface: texture_2d_array<f32>;
@group(0) @binding(5) var impostor_atlas: texture_2d_array<f32>;
@group(0) @binding(6) var<storage, read> placed_impostor_records: array<ImpostorRecord>;
@group(0) @binding(7) var placed_impostor_albedo: texture_2d_array<f32>;
@group(0) @binding(8) var placed_impostor_emissive: texture_2d_array<f32>;
@group(0) @binding(9) var placed_impostor_geometry: texture_2d_array<f32>;
@group(0) @binding(10) var placed_impostor_surface: texture_2d_array<f32>;
@group(0) @binding(11) var placed_impostor_atlas: texture_2d_array<f32>;
@group(0) @binding(12) var impostor_sampler: sampler;
