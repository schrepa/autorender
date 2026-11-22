import package::generated::nanite::NaniteUniforms;
import package::generated::nanite::NaniteInstance;
import package::generated::nanite::GpuClusterHeader;
import package::generated::nanite::VisibleCluster;

@group(0) @binding(10) var<uniform> nanite_frame_uniforms: NaniteUniforms;
@group(0) @binding(11) var<storage, read> nanite_frame_instances: array<NaniteInstance>;
@group(0) @binding(12) var<storage, read> nanite_frame_clusters: array<GpuClusterHeader>;
@group(0) @binding(13) var<storage, read> nanite_frame_page_heap: array<u32>;
@group(0) @binding(14) var<storage, read> nanite_frame_vis_clusters: array<VisibleCluster>;
@group(0) @binding(15) var<storage, read> nanite_frame_sw_depth: array<u32>;
@group(0) @binding(16) var<storage, read> nanite_frame_sw_vis: array<u32>;
@group(0) @binding(17) var<storage, read> nanite_frame_shadow_depth: array<u32>;
