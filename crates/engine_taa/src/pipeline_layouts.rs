use crate::bindings::TAA_RESOLVE;

engine_render::pipeline_layouts! {
    TAA_RESOLVE_PIPELINES = [TAA_RESOLVE, (composite without [])];
}
