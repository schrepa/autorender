engine_render::shader_modules! {
    "post::exposure" in "auto_exposure" = "exposure.wgsl";
}

engine_render::entry_points! {
    exposure {
        compute: [meter_exposure @ "post::exposure", adapt_exposure @ "post::exposure"],
    }
}
