# Decision 018: Atmosphere and stylized lighting

Status: adopted for every renderer path. The mesh shader's fixed lighting
constants become scene data; captures, previews and CPU presentation shade
exactly like ordinary frames. Decision 019 replaces the sun highlight with
material surfaces and makes `specular` a physically based scale. Decision
022 lets scenes author the atmosphere as a preset with overrides and lets
agents inspect and temporarily replace it. Decision 027 keeps bright hues
through the tone curve, adds contrast, vibrance and split toning to the
grade, and retunes the presets.

## Context

Lighting was fixed in `meshes.wgsl`: a warm sun, a dim blue-gray ambient, a
flat near-black background and distance fog in that near-black color. Beyond
about 300 units fog covered about 80% of every surface, shaded sides were dim
and gray, and the output went straight to the screen without tone mapping, so
large scenes such as the Stress Lab city read as a gray haze. Games could not
change any of it.

## Decision

`Atmosphere`, an engine resource inserted by default, describes the lighting:
a sun direction and color, three sky colors (zenith, horizon, ground), ambient
scale, fog start, density, height falloff and maximum, rim and specular
strength, exposure and saturation. Colors are linear and may exceed 1.
Presets: `day`, `golden_hour` (the default), `dusk` and `night`. Extraction
copies the sanitized resource (`Atmosphere::sanitized`: unusable values fall
back to the default) and the camera's eye (`Camera::eye`) into every
`RenderSnapshot`; the renderer writes them to a 96-byte lighting uniform when
they change and a 224-byte camera uniform.

Every opaque fragment:

- takes ambient light from the sky: the zenith from above, an even mix of
  horizon and zenith from the side and the ground from below, so shade is
  cooler than sunlight;
- takes wrapped sun diffuse (softer on flat low-poly faces), a sun highlight
  and a rim light in the horizon color on walls;
- is covered by fog whose density decays exponentially with height,
  integrated along the view ray, in the color of the sky haze behind it;
- is tone mapped (Narkowicz's ACES fit) after exposure, then saturated.

A sky pass draws one triangle at the far plane after the meshes, depth-tested
without writing depth, so it shades only uncovered pixels: the gradient, a
halo and a sun disc, graded the same way. Temporally antialiased frames
(decision 017) draw it into their scene target, with motion from the camera's
turn alone. Transcendental functions stay few: the fog's haze color omits the
sun disc and uses products for its halo, and the rim light squares twice.

## Observable behavior

- Every game gets the golden-hour look unless it sets `Atmosphere`; a
  `Atmosphere::night()` or a custom resource restores a darker scene.
- The Stress Lab picks night over the tower field and golden hour over the
  city, and its Lighting control (and `?look=golden|dusk|night|day`) overrides
  that, kept across workload changes. Recordings and profiles report `look`.
- Exact captures, previews and CPU-presented frames shade identically to
  ordinary frames; the GPU-presentation and TAA smoke checks pass unchanged.

## Measurements

Firefox 156 on the Intel iGPU, 2262×987 canvas, the 90,000-object city at 10%
moving, four page loads per build, alternating: the opaque pass (which now
includes the sky) measured 4.5–5.7 ms against 4.2–5.0 ms before, within this
session's variation. Frame rates were 84–91 fps against 92–99 fps; page-to-page
variation of ±15% leaves open whether shading costs frame rate there. The
shader's transcendental math was trimmed after these runs.

## Consequences

The scene's palette now comes from three sky colors and a sun. There are no
shadows, emissive materials or bloom: lit windows are bright texture texels,
not light sources. Shadow maps would add a depth pass over all geometry, the
cost that already bounds large scenes; material emission and a bloom pass are
the natural next steps for night scenes.
