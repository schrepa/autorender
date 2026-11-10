import package::engine::drawn::{drawn_records, drawn_shown};
import package::engine::view::{camera, view_clip};
import package::generated::sprites::SpriteInstance;

@group(2) @binding(0) var<storage, read> instances: array<SpriteInstance>;

const SPRITE_FACING: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
const SPRITE_WHITE: u32 = 0xffffffffu;
const SPRITE_ROUGH: u32 = 0xffu;

struct SpriteCorner {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
}

struct ShownSpriteCorner {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}

fn sprite_corner(corner: u32, sprite: SpriteInstance) -> SpriteCorner {
    let unit = vec2<f32>(f32((50u >> corner) & 1u), f32((44u >> corner) & 1u));
    let world = vec3<f32>(sprite.place.xy + (unit - 0.5) * sprite.place.w, sprite.place.z);
    let uv = sprite.rect.xy + vec2<f32>(unit.x, 1.0 - unit.y) * sprite.rect.zw;
    let shown = camera.previous_view_projection * vec4<f32>(world, 1.0);
    var out: SpriteCorner;
    out.clip_position = view_clip(world);
    out.normal_u = vec4<f32>(SPRITE_FACING, uv.x);
    out.world_v = vec4<f32>(world, uv.y);
    out.material = vec4<u32>(SPRITE_WHITE, SPRITE_WHITE, SPRITE_ROUGH, 0u);
    out.previous = shown.xyw;
    out.color = sprite.tint * drawn_records[sprite.drawn.x].tint;
    out.uv1 = uv;
    return out;
}

@vertex fn vs_sprite(@builtin(vertex_index) corner: u32, @builtin(instance_index) index: u32)
        -> ShownSpriteCorner {
    let sprite = instances[index];
    var out: ShownSpriteCorner;
    if !drawn_shown(sprite.drawn.x) {
        out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let placed = sprite_corner(corner, sprite);
    out.clip_position = placed.clip_position;
    out.normal_u = placed.normal_u;
    out.world_v = placed.world_v;
    out.material = placed.material;
    out.previous = placed.previous;
    out.color = placed.color;
    out.uv1 = placed.uv1;
    out.object = vec4<u32>(0u, sprite.drawn.x, 0u, 0u);
    return out;
}

@vertex fn vs_sprite_view(@builtin(vertex_index) corner: u32,
        @builtin(instance_index) index: u32) -> SpriteCorner {
    return sprite_corner(corner, instances[index]);
}
