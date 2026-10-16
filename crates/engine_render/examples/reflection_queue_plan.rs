//! Host planning driver for reflection-classification-smoke, using production code.

#[path = "../src/reflection_queue_plan.rs"]
mod reflection_queue_plan;

use reflection_queue_plan::{Plan, grid};

fn main() {
    let limits = wgpu::Limits::default();
    for (count, expected) in [
        (0, [0, 1]),
        (63, [1, 1]),
        (64, [1, 1]),
        (65, [2, 1]),
        (64 * 64 - 1, [64, 1]),
        (64 * 64, [64, 1]),
        (64 * 64 + 1, [64, 2]),
    ] {
        assert_eq!(grid(count, 64), Some(expected), "grid for {count}");
    }
    assert_eq!(
        grid(u32::MAX, u32::MAX),
        None,
        "padded indexing must not overflow"
    );
    assert_eq!(grid(65, 1), None, "dispatch height is limited");
    assert_eq!(grid(0, 0), None, "zero dispatch limit is unsupported");
    let invalid: &[fn(&mut wgpu::Limits)] = &[
        |l| l.max_buffer_size = 47,
        |l| l.max_storage_buffer_binding_size = 47,
        |l| l.max_uniform_buffer_binding_size = 47,
        |l| l.max_bind_groups = 1,
        |l| l.max_bindings_per_bind_group = 14,
        |l| l.max_storage_buffers_per_shader_stage = 5,
        |l| l.max_uniform_buffers_per_shader_stage = 1,
        |l| l.max_sampled_textures_per_shader_stage = 5,
        |l| l.max_samplers_per_shader_stage = 0,
        |l| l.max_dynamic_uniform_buffers_per_pipeline_layout = 0,
        |l| l.max_compute_invocations_per_workgroup = 63,
        |l| l.max_compute_workgroup_size_x = 63,
        |l| l.max_compute_workgroup_size_y = 7,
        |l| l.max_compute_workgroup_size_z = 0,
        |l| l.max_compute_workgroup_storage_size = 1039,
        |l| l.max_compute_workgroups_per_dimension = 1,
        |l| l.max_texture_dimension_2d = 16,
        |l| l.min_uniform_buffer_offset_alignment = 0,
        |l| l.min_uniform_buffer_offset_alignment = 1 << 31,
    ];
    for (index, change) in invalid.iter().enumerate() {
        let mut low = limits.clone();
        change(&mut low);
        assert!(
            Plan::new(1026, 1026, &low).is_none(),
            "low limit {index} must fall back"
        );
    }
    let mut enormous = limits.clone();
    enormous.max_texture_dimension_2d = u32::MAX;
    assert!(
        Plan::new(u32::MAX, u32::MAX, &enormous).is_none(),
        "pixel count overflow"
    );
    assert!(Plan::new(0, 1, &limits).is_none(), "zero full width");
    let sizes = [
        [1, 1],
        [1, 17],
        [17, 1],
        [17, 13],
        [31, 33],
        [127, 131],
        [128, 128],
        [1008, 16],
        [1024, 16],
        [1040, 16],
        [1026, 1026],
    ];
    println!("{{\"rejections\":{},\"plans\":[", invalid.len() + 2);
    for (index, [width, height]) in sizes.into_iter().enumerate() {
        let Some(plan) = Plan::new(width, height, &limits) else {
            panic!("supported {width}x{height}")
        };
        assert_eq!(
            plan.pixels,
            (width / 2).max(1) * (height / 2).max(1),
            "floor-half capacity"
        );
        println!(
            "{}{{\"full\":[{width},{height}],\"size\":{:?},\"pixels\":{},\"row_log2\":{},\"stride\":{},\"bytes\":{:?},\"capacity\":{},\"levels\":[",
            if index == 0 { "" } else { "," },
            plan.size,
            plan.pixels,
            plan.row_log2,
            plan.stride,
            plan.bytes,
            plan.capacity_bytes()
        );
        for (index, level) in plan.levels.iter().enumerate() {
            println!(
                "{}{{\"offset\":{},\"length\":{},\"grid\":{:?}}}",
                if index == 0 { "" } else { "," },
                level.offset,
                level.length,
                level.grid
            );
        }
        println!("]}}");
    }
    println!("]}}");
}
