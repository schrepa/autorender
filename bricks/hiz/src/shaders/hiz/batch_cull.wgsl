import package::generated::hiz::HizCullParams;
import package::generated::hiz::HizInfo;
import package::generated::hiz::HizSphere;
import package::hiz::cull::hiz_project_sphere;
import package::hiz::cull::hiz_test_rect;

@group(1) @binding(0) var<uniform> cull_params: HizCullParams;
@group(1) @binding(1) var<uniform> cull_hiz_info: HizInfo;
@group(1) @binding(2) var cull_hiz_tex: texture_2d<f32>;
@group(1) @binding(3) var<storage, read> cull_spheres: array<HizSphere>;
@group(1) @binding(4) var<storage, read_write> cull_visible: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn cs_hiz_cull_spheres(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= cull_params.count {
        return;
    }
    let sphere = cull_spheres[idx].center_radius;
    let center_vs = (cull_params.view * vec4<f32>(sphere.xyz, 1.0)).xyz;
    let radius = max(sphere.w, 0.0);
    let near_z = max(cull_params.near_z, 1.0e-6);
    if -center_vs.z + radius < near_z {
        cull_visible[idx] = 0u;
        return;
    }
    let rect = hiz_project_sphere(
        center_vs,
        radius,
        cull_params.proj[0][0],
        cull_params.proj[1][1],
        near_z,
    );
    let nearest_z_vs = min(center_vs.z + radius, -near_z);
    let p22 = cull_params.proj[2][2];
    let p32 = cull_params.proj[3][2];
    let nearest_ndc_z = clamp((p22 * nearest_z_vs + p32) / (-nearest_z_vs), 0.0, 1.0);
    let occluded = hiz_test_rect(
        cull_hiz_tex,
        rect.xy,
        rect.zw,
        nearest_ndc_z,
        cull_hiz_info.mip0_size_f,
        cull_hiz_info.mip_count,
    );
    cull_visible[idx] = select(1u, 0u, occluded);
}
