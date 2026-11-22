import package::generated::sanctuary::GroundVertex;
import package::nanite::frame::nanite_frame_uniforms;

@group(2) @binding(0) var<storage, read> ground_verts: array<GroundVertex>;

struct GroundOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) tag: u32,
}

struct GroundFrag {
    @location(0) vis_hw: u32,
    @location(1) depth_p1: f32,
}

@vertex
fn vs_ground(@builtin(vertex_index) vid: u32) -> GroundOut {
    let v = ground_verts[vid];
    var out: GroundOut;
    out.clip_pos = nanite_frame_uniforms.render_view_proj * vec4<f32>(v.pos_tag.xyz, 1.0);
    out.tag = bitcast<u32>(v.pos_tag.w);
    return out;
}

@fragment
fn fs_ground(in: GroundOut) -> GroundFrag {
    var out: GroundFrag;
    out.vis_hw = in.tag;
    out.depth_p1 = in.clip_pos.z;
    return out;
}
