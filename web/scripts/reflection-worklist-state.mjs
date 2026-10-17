/** Diagnostic shader wrappers expose state without editing production arithmetic. */
export function reflectionStateSource(source, fragment, extracted = false) {
  const signature = fragment && !extracted
    ? "@fragment\nfn fs_reflections(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {"
    : "fn reflection_at(half_texel: vec2<i32>) -> vec4<f32> {";
  if (!source.includes(signature)) throw new Error("reflection diagnostic signature changed");
  let result = source.replace(signature, `fn diagnostic_at(half_texel: vec2<i32>) -> vec4<f32> {
    diagnostic_state = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    diagnostic_point = vec4<f32>(0.0);
    var processed = 0u;`);
  if (fragment) result = result.replace("let pixel = traced_pixel(vec2<i32>(position.xy));", "let pixel = traced_pixel(half_texel);");
  result = result.replace("        let size = exp2(f32(mip));", "        processed += 1u;\n        let size = exp2(f32(mip));");
  result = result.replace("    if mip >= 0 || t > 1.0 {", `    diagnostic_state = vec4<f32>(0.0, f32(processed), t, f32(mip));
    diagnostic_point = vec4<f32>(point, ray.z);
    if mip >= 0 || t > 1.0 {`);
  result = result.replace("    return vec4<f32>(color, reflectivity * fade);", "    diagnostic_state.x = 1.0;\n    return vec4<f32>(color, reflectivity * fade);");
  result = "var<private> diagnostic_state: vec4<f32>;\nvar<private> diagnostic_point: vec4<f32>;\n" + result;
  if (fragment) return result.replaceAll("reflection_at(", "diagnostic_at(") + `
    struct DiagnosticState { @location(0) state: vec4<f32>, @location(1) point: vec4<f32> }
    @fragment fn diagnostic_fragment(@builtin(position) position: vec4<f32>) -> DiagnosticState {
      let color = diagnostic_at(vec2<i32>(position.xy));
      return DiagnosticState(diagnostic_state, diagnostic_point);
    }`;
  return result.replace("reflection_at(vec2<i32>(texel))", "diagnostic_at(vec2<i32>(texel))")
    .replace("    raw[texel.y * params.size.x + texel.x] = diagnostic_at(vec2<i32>(texel));", `    raw[texel.y * params.size.x + texel.x] = diagnostic_at(vec2<i32>(texel));
    diagnostic_output[(texel.y * params.size.x + texel.x) * 2u] = diagnostic_state;
    diagnostic_output[(texel.y * params.size.x + texel.x) * 2u + 1u] = diagnostic_point;`)
    + "\n@group(1) @binding(4) var<storage, read_write> diagnostic_output: array<vec4<f32>>;\n";
}
