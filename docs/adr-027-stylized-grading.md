# Decision 027: Stylized grading

Status: adopted for every renderer path. `Atmosphere` gains contrast,
vibrance and split toning, the tone curve keeps the hue of bright colors,
and the four presets are retuned for a vivid, stylized palette.

## Context

With decisions 018 to 026 in place, large scenes still read as grayish and
bland. Capture statistics of the Stress Lab city from above at golden hour
(mean HSV saturation, the share of pixels with saturation above 0.5, and the
standard deviation of luminance) located the causes:

- Fog covered up to 75% of distant surfaces in the pale horizon color.
  Without fog, vivid pixels rose from 2% to 8% and contrast from 0.18 to
  0.24.
- The per-channel filmic curve pushes bright colors toward white, so sunlit
  paint and lit windows lost their hue.
- Ambient occlusion, bloom and sun shafts changed little.
- A richer sky and thinner fog alone raised vivid pixels to 37% but made the
  image one orange: saturation without separate hues for light and shade
  reads as a color cast, not as a palette.

## Decision

Every shaded color (meshes, sky, transparent surfaces, particles; every path
alike) is graded in the same function after exposure:

- **Tone curve.** Half Narkowicz's per-channel ACES fit, half the same curve
  applied to the brightest channel and scaling all three, which keeps a
  bright color's hue and saturation while highlights still roll off.
- **Contrast** around mid gray in a perceptual (square root) space: an
  S-curve that keeps black and white, its slope at mid gray the contrast,
  from 0 up to its steepest at 1.5. (A first version scaled linearly and
  clipped dark shade to black.)
- **Split toning.** `shadow_tint` and `highlight_tint` are color gains,
  scaled to unit luminance when uploaded so they shift hue without changing
  brightness (white or black changes nothing), blended by perceptual
  luminance from 0.2 to 0.8. Cool shadows against warm highlights separate
  light from shade.
- **Saturation**, then **vibrance**: extra saturation weighted by how muted a
  color is, so grays and pastels gain color while vivid colors do not clip.

`Atmosphere::contrast`, `vibrance`, `shadow_tint` and `highlight_tint` are
sanitized like the other fields (finite and nonnegative). The lighting
uniform grows from 112 to 160 bytes; it still uploads only when it changes.

The presets thin and delay the fog (it starts at 80 units by day and at
golden hour, and covers at most 40–45%, 70% at night), deepen the zenith,
and grade with contrast 1.12–1.2, vibrance 0.35–0.5, blue or violet shadows
and warm highlights. Golden hour, the default, pairs an amber horizon with a
deep blue zenith.

Scene version 6 authors the four fields in the environment's `atmosphere`
like the other overrides; earlier versions refuse them (`version_required`).
Agents inspect them in `Atmosphere` and `Environment` and try them with the
`environment` control.

## Observable behavior

- Every game and scene that uses a preset, including the default, looks
  different: more saturated and contrasted, with less haze. Scenes that name
  a preset take the new values; the migration note lists the earlier ones.
- Neutral values (contrast 1, vibrance 0, white tints, saturation 1) leave
  only the tone curve; they come close to, but do not reproduce, the earlier
  curve.
- Exact captures, previews and CPU-presented frames grade identically to
  ordinary frames.

## Measurements

Image statistics of the city overview in headless Chrome with temporal
antialiasing (1208×648 canvas region, same camera):

| Look | Saturation | Vivid pixels | Contrast | Luminance |
| --- | --- | --- | --- | --- |
| Golden hour before | 0.258 | 2.1% | 0.183 | 0.606 |
| Golden hour | 0.434 | 26.1% | 0.215 | 0.510 |
| Day | 0.378 | 27.7% | 0.206 | 0.557 |
| Dusk | 0.463 | 34.0% | 0.216 | 0.513 |
| Night | 0.636 | 77.0% | 0.206 | 0.365 |

Frame cost is not measured yet. Grading adds four square roots, a smoothstep
and a few multiplies per shaded fragment, including overdraw.

## Consequences

Authored colors shift: vibrance saturates muted materials and split toning
tints shade, so a scene that needs exact colors grades neutrally. The grade
is applied per fragment rather than as a post pass, so temporal
antialiasing, bloom and transparency blend graded colors, as before. See the
[migration](migrations/0.1-stylized-grading.md).
