engine_render::shader_modules! {
    "particles::draw" in "particles" = "particles.wgsl";
}

engine_render::entry_points! {
    particles {
        vertex: [vs_particle @ "particles::draw"],
        fragment: [
            fs_particle @ "particles::draw",
            fs_particle_mist @ "particles::draw",
            fs_particle_add @ "particles::draw",
            fs_particle_add_post @ "particles::draw",
        ],
    }
}
