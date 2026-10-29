engine_render::shader_modules! {
    "water::refraction" in "water" = "refraction.wgsl";
    "water::surface" in "water" = "surface.wgsl";
    "water::footprint" in "water" = "footprint.wgsl";
    "water::flow" in "water" = "flow.wgsl";
    "water::setup" in "water" = "setup.wgsl";
    "water::falls" in "water" = "falls.wgsl";
    "water::foam" in "water" = "foam.wgsl";
    "water::particles" in "water" = "particles.wgsl";
    "water::bursts" in "water" = "bursts.wgsl";
    "water::sim_shared" in "water" = "sim_shared.wgsl";
    "water::sim" in "water" = "sim.wgsl";
    "water::debug" in "water" = "debug.wgsl";
}

engine_render::entry_points! {
    meshes {
        vertex: [
            vs_water @ "water::surface",
            vs_foam @ "water::particles",
            vs_fall @ "water::falls",
            vs_fall_temporal @ "water::falls",
        ],
        fragment: [
            fs_water @ "water::surface",
            fs_water_post @ "water::surface",
            fs_water_temporal @ "water::surface",
            fs_foam @ "water::particles",
            fs_fall @ "water::falls",
            fs_fall_post @ "water::falls",
            fs_fall_temporal @ "water::falls",
            fs_debug_water @ "water::debug",
            fs_debug_fall @ "water::debug",
        ],
        compute: [
            setup_blocks @ "water::setup",
            setup_ring @ "water::setup",
            setup_raise @ "water::setup",
            particles_begin @ "water::particles",
            particles_gather @ "water::particles",
            particles_currents @ "water::particles",
            particles_age @ "water::particles",
            particles_births @ "water::particles",
            particles_sum @ "water::particles",
            particles_scan @ "water::particles",
            particles_place @ "water::particles",
            particles_splat @ "water::particles",
            particles_look @ "water::particles",
            particles_down @ "water::particles",
        ],
    }
    water_sim {
        compute: [
            sim_exchange @ "water::sim",
            sim_waves @ "water::sim",
            sim_seconds @ "water::sim",
            sim_drain @ "water::sim",
            sim_update @ "water::sim",
            sim_source @ "water::sim",
            sim_touch @ "water::sim",
            sim_burst @ "water::sim",
            sim_push @ "water::sim",
            sim_ended @ "water::sim",
            sim_tally @ "water::sim",
            sim_activity @ "water::sim",
            sim_look @ "water::sim",
            sim_drawn @ "water::sim",
        ],
    }
}
