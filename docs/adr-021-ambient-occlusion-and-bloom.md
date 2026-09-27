# Decision 021: Ambient occlusion and bloom

Status: adopted for every renderer path, on by default (`AmbientOcclusion`,
`Bloom`). Temporally antialiased frames apply both in their resolve; other
frames, including exact captures, in a composite pass. Decision 022 makes
both resources optional (missing means off), creates the post-processing
pipelines and targets only while an effect or temporal antialiasing runs,
and exposes the settings to scenes and agents.

## Context

Ambient light reached creases, corners and the ground under objects as
fully as open ground, so scenes lacked contact and scale. Bright light had
no glow: emissive windows (decision 019) and sun glints stopped at their
pixels.

## Decision

`AmbientOcclusion` (intensity in `[0, 1]` and a world radius) and `Bloom`
(intensity in `[0, 4]` and an exposed-brightness threshold), engine
resources inserted by default, are carried sanitized in snapshots. An
intensity of 0 turns an effect off.

While either runs, or temporal antialiasing does, the opaque pass draws into
an sRGB scene target whose alpha holds the ambient share of each pixel's
light (sky light and reflections over all light, less fog), and an 8-bit
glow target holding each pixel's exposed brightness above the threshold as
`x / (1 + x)`. With ambient occlusion the depth buffer is kept for it.

- Ambient occlusion runs at half resolution: scalable ambient obscurance
  over 8 depth samples on a spiral of the world radius (at most 96 pixels),
  turned per pixel by interleaved gradient noise, with normals from the
  nearer depth neighbors. The noise stays fixed from frame to frame: turned
  per frame, the temporal resolve's neighborhood clamp chased the changing
  darkness and still views flickered past the TAA smoke check's limit.
  Applying it, four bilinear taps blur the samples over three texels and
  darken only the pixel's ambient share.
- Bloom decodes the glowing light (color times its glow weight) into a
  half-resolution level, weighting taps by `1 / (1 + luminance)` so single
  pixels cannot flicker, filters it down five more levels and back up (a
  dual filter), and adds it to the canvas as a screen blend, which never
  passes white. Temporal frames add it after their history, so glow never
  accumulates in it.

## Observable behavior

- Captures show both effects with the same noise as GPU captures: the
  GPU-presentation smoke check compares them pixel for pixel.
- The Stress Lab's Effects control (and `?effects=`) turns either off.

## Measurements

None yet: to be measured with decisions 019 and 020, in particular the
composite pass frames without temporal antialiasing now add.

## Consequences

Frames with either effect write two render targets instead of the canvas
and add a full-screen composite unless they are temporally antialiased.
Occlusion comes from what the depth buffer shows: off-screen and hidden
geometry does not occlude, and it fades at silhouettes over a pixel or two.
Bloom follows the scene target's 8-bit color, so its hue saturates toward
white for the brightest lights.
