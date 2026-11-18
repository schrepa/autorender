import package::effects::items::effects_frame;
import package::effects::volume::effects_depth;

fn screen_over(under: vec4<f32>, top: vec4<f32>) -> vec4<f32> {
    return top + under * (1.0 - top.a);
}

@fragment fn fs_screen(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(effects_depth));
    let unit = position.xy / size * 2.0 - 1.0;
    let reach = length(unit);
    let cover = effects_frame.cover;
    let damage = effects_frame.damage;
    let flash = effects_frame.flash;
    var shown = vec4<f32>(cover.rgb, 1.0) * cover.a;
    let edge = smoothstep(effects_frame.clock.y, 1.0 + effects_frame.clock.y, reach) * damage.a;
    shown = screen_over(shown, vec4<f32>(damage.rgb, 1.0) * edge);
    shown = screen_over(shown, vec4<f32>(flash.rgb, 1.0) * flash.a);
    return shown;
}
