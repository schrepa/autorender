# MONOLITH — Scene Specification (CH 04)

**Sunlit Monumental Brutalist-Classical Sanctuary.** A processional avenue of 196 fluted marble columns runs north through twin coffered triumphal arches toward a 32 m winged marble Colossus Sentinel, which stands in a sunken travertine plaza ringed by stratified basalt canyon megaliths. Groves of sculpted opaque cypresses line the way. To the east, the *Arcade of Echoes* is a 151 m tunnel of 14 back-to-back coffered arches, built so that HI-Z FREEZE has an extreme occluder to show off.

Total virtualized triangles in the full layout: **107,244,276** from **457 instances** of **5 unique hero meshes**, which add up to **1,632,178 unique source triangles**.

---

## 1. Conventions

| Item | Value |
|---|---|
| Units | meters |
| Handedness | right-handed, **+Y up** |
| North | **−Z** (the avenue runs from z = +150 in the south to the Sentinel at z = −215) |
| Instance transform | `M = T(pos) · Ry(yaw) · Rz(roll) · S(scale)` (uniform scale) |
| Yaw | degrees about +Y, counter-clockwise seen from above (yaw +90 turns asset +Z toward world +X) |
| Asset pivot | every asset is grounded at `y = 0` and centered on `x = 0, z = 0` (bounding-box center; the Sentinel is pedestal-centered) |
| Asset facing | asset **+Z** is the hero front: Sentinel face/sword, arch inscription facade. The canyon stair flank faces asset **−X** |
| Winding | CCW outward, back-face culling safe (all closed or occluded-backed shells) |

### Asset vertex contract (all 5 `.glb`)

- `POSITION` VEC3 f32, `NORMAL` VEC3 f32 (unit length; auto-smoothed with crisp architectural creases split at about 40–50°), `TEXCOORD_0` VEC2 f32 (metric, about 1 unit per meter), `COLOR_0` VEC4 f32, indices `SCALAR UNSIGNED_INT` (u32), `mode = TRIANGLES`.
- `COLOR_0.rgb` holds the **final linear base color** with the per-material variation already baked in (marble veins, terracotta crevice dirt, basalt dust on ledges, bronze gold rims and verdigris recesses, foliage sun/shade tint). Use it directly as albedo. The glTF `baseColorFactor` is only the mean fallback.
- `COLOR_0.a` holds **baked crevice/ambient occlusion** in `[0.15, 1.0]` (multi-scale voxel AO times a curvature cavity term). Apply it to indirect/sky lighting and specular occlusion, not to direct sun (the sun is shadowed by the shadow map).
- One glTF primitive per material. `primitive.material` equals the global MONOLITH material ID (0–4). Every `.glb` carries the full 5-entry `materials` array, so the IDs match across files.

---

## 2. Hero assets

Source directory (not committed; binary): `C:\Users\a\.gemini\antigravity-cli\brain\3e6ba844-80c3-4b86-b31f-fd6dca13a7b0\scratch\monolith_assets\`
Reproducible generators live in the same directory (`mlib.py` sculpting/AO/glTF core, `column.py`, `arch.py`, `sentinel.py`, `canyon.py`, `cypress.py`, plus the `render.py` / `check.py` software preview validator).

| ID | File | Triangles | Per material | Bounds (min → max, m) | Description |
|---|---|---:|---|---|---|
| A | `colossus_sentinel.glb` | **349,116** | mat0 304,448 · mat2 44,668 | (−6.00, 0, −8.08) → (6.00, 20.27, 3.60) | Hooded, crowned winged guardian holding a planted greatsword on a stepped travertine pedestal inscribed `CVSTOS` / `AETERNVS`. Each wing has 12 primaries, 18 secondaries, 4 tertials, 28 greater coverts, 24 median coverts and 3 rows of scale coverts. Robe has 15 angular skirt pleats, a zigzag overfold, a studded belt and a diagonal sash. The bronze astrolabe halo has 4 concentric rings, 3 armillary rings, 72 ticks, 24 flame rays, 12 studs and a hub. The crown is bronze with 11 spikes. |
| B | `fluted_column.glb` | **260,616** | mat0 237,048 · mat2 23,568 | (−1.45, 0, −1.45) → (1.45, 16.00, 1.45) | 24-flute column with fillets, entasis, chipped arrises, a two-step chipped plinth and an Attic torus–scotia–torus base. The bronze astragal collar has a band, 2 tori and 44 beads and reels. The Corinthian capital has 3 tiers of lobed acanthus (24 leaves), 4 corner volutes, 8 helices, 4 abacus rosettes and a chipped two-tier abacus. |
| C | `vaulted_arch.glb` | **438,688** | mat0 415,504 · mat3 23,184 | (−10.05, 0, −5.55) → (10.05, 22.66, 5.55) | Triumphal arch, 8 m opening, springing at y = 10, crown of opening at y = 14, passage floor at y = 1.0. Rusticated running-bond ashlar piers, 4 scallop niches with urns, procession reliefs (jambs plus outer flanks), festoon reliefs above, a barrel vault with 7 × 6 stepped coffers and 42 rosettes, 17 voussoirs and a scroll keystone on each face, wreathed spandrel medallions, an entablature of 3 fasciae, triglyph frieze, dentils and corona, and an attic inscription `MONOLITHVS · AETERNVS SPQR · SENTINELLA DCCC`. The plinth is sandstone. |
| D | `canyon_monolith.glb` | **409,948** | mat1 409,948 | (−16.91, 0, −14.10) → (16.91, 43.00, 14.10) | Stratified volcanic basalt butte. 9 uneven strata (0.8–6.9 m) tilted about 3° and noise-warped, with recessed soft bands and 2 big undercut overhang shelves. Each stratum has its own hexagonal columnar outline with staggered joints and broken tops. Asymmetric plan: sheer +X cliff, stepped −X flank, 6 vertical clefts and 12 wind-carved alcoves. Two chiseled stair flights (26 + 13 treads) climb the **−X flank** to a carved portal. Organ-pipe crown spires, and a scree apron with about 700 fallen hex prisms and boulders. |
| E | `cypress_cluster.glb` | **173,810** | mat3 34,730 · mat4 139,080 | (−1.31, 0, −1.27) → (1.31, 10.94, 1.27) | Mediterranean cypress: twisted, buttressed trunk with 291 upward-sweeping branches. 2,151 opaque sculpted fan sprigs (5-diamond cords plus 6 side cords) over 388 dark clump cores. **Zero alpha cards.** |

**Cooking:** run `cargo run --release -p cook_meshlet -- cook <file>.glb <out_dir> --name <stem>` once per asset. This produces one `.mlm` manifest and one streamed `.mlp` file of 64 KB pages per unique mesh. Every placement below is an **instance transform** of these 5 cooked meshes.

---

## 3. Layout

### 3.1 Ground and terraces (simple engine geometry, not Nanite assets)

| Element | Extent | Height y | Material |
|---|---|---:|---|
| Desert floor | 1600 × 1600 centered (0, 0, −150) | 0.0 | 3 sandstone (albedo × 0.85) |
| Processional pavement | x ∈ [−25, 25], z ∈ [−150, 160] | 0.02 | 0 marble/travertine, 1.5 m pavers, 8 mm joints |
| Arcade floor strip | x ∈ [52, 68], z ∈ [−120, 40] | 0.02 | 3 sandstone pavers |
| Plaza rim (terrace 0) | square 120 × 120 centered (0, ·, −205) | 0.0 | 0 travertine |
| Terrace 1 | square 100 × 100 centered (0, ·, −205) | −1.2 | 0 travertine |
| Terrace 2 | square 80 × 80 centered (0, ·, −205) | −2.4 | 3 sandstone |
| Plaza floor | square 60 × 60 centered (0, ·, −205) | −3.6 | 0 travertine, 3 m radial paving rings |
| Grand stair into the plaza | x ∈ [−12, 12], z ∈ [−150, −145], 3 flights of 4 × 0.3 m risers | 0 → −3.6 | 0 travertine |

### 3.2 Processional colonnade (B · fluted_column) — 196 instances

Four rows: x ∈ {−20, −14, +14, +20}. For `i = 0 … 48`: `z_i = 141 − 6·i` (so z runs from 141 to −147).
`pos = (x, 0, z_i)`, `yaw = 90 · (i mod 4)` (keeps the square abacus aligned with the avenue), `roll = 0`, `scale = 1.0`.
The inner rows at ±14 are what produces the deep, rhythmic occlusion in VISTA. The outer rows at ±20 double the depth complexity.

### 3.3 Twin triumphal arches (C · vaulted_arch) — 2 instances

| Name | pos | yaw | scale |
|---|---|---:|---:|
| ARCH_SOUTH | (0, 0, +42) | 0 | 1.2 |
| ARCH_NORTH | (0, 0, −66) | 0 | 1.2 |

At scale 1.2 the arch is 24.1 m wide (x ±12.06), so it clears the inner column plinths at x ±12.55. Opening is 9.6 m wide, crown of opening at y 16.8, attic top at y 27.2.

### 3.4 Colossus Sentinel (A · colossus_sentinel) — 3 instances

| Name | pos | yaw | scale | Notes |
|---|---|---:|---:|---|
| SENTINEL_PRIME | (0, −3.6, −215) | 0 | 1.6 | 32.4 m tall on the plaza floor, facing south down the avenue. Head at world (0, 18.19, −214.65), halo center at (0, 18.88, −216.68) |
| SENTINEL_WEST | (−22, −3.6, −196) | +20 | 0.85 | attendant, turned toward the axis |
| SENTINEL_EAST | (+22, −3.6, −196) | −20 | 0.85 | attendant, turned toward the axis |

### 3.5 Plaza ring colonnade (B) — 16 instances

Center `C = (0, −1.2, −205)` on terrace 1, radius 40. For `k = 0 … 17`, skipping `k = 8, 9` (this leaves the south opening toward the avenue): `θ_k = 10° + 20°·k`, `pos = (40·sin θ_k, −1.2, −205 − 40·cos θ_k)`, `yaw = −θ_k`, `scale = 1.0`.

### 3.6 Fallen weathered ruins (B, toppled) — 10 instances

The column lies on its side: `roll = 90°`, so its axis runs along the rotated −X. The pivot is the base center. `y = ground + 0.95` keeps the plinth partly buried and the shaft resting on the ground.

| # | pos | yaw | roll | scale |
|---|---|---:|---:|---:|
| 1 | (−30, 0.95, 95) | 20 | 90 | 1.0 |
| 2 | (31, 0.95, 18) | 160 | 90 | 1.0 |
| 3 | (−34, 0.95, −40) | 75 | 90 | 1.0 |
| 4 | (36, 0.95, −102) | −30 | 90 | 1.0 |
| 5 | (−45, −0.25, −176) | 10 | 90 | 1.0 |
| 6 | (47, −0.25, −230) | 200 | 90 | 1.0 |
| 7 | (−18, −2.65, −186) | 40 | 90 | 1.0 |
| 8 | (20, −2.65, −238) | −120 | 90 | 1.0 |
| 9 | (62, 0.95, −128) | 95 | 90 | 1.0 |
| 10 | (−60, 0.95, 130) | −15 | 90 | 1.0 |

### 3.7 Canyon megaliths (D · canyon_monolith) — 16 instances

The yaws turn the stair flank (asset −X) toward the avenue or plaza on the hero rocks. West-side rocks use yaw ≈ 180 and east-side rocks yaw ≈ 0.

| # | Role | pos | yaw | scale |
|---|---|---|---:|---:|
| 1 | backdrop crown | (0, −4, −340) | 90 | 3.0 |
| 2 | backdrop west | (−38, −2, −290) | 15 | 2.2 |
| 3 | backdrop east | (30, −2, −300) | −160 | 2.6 |
| 4 | north-west shoulder | (−85, −2, −270) | 60 | 2.0 |
| 5 | north-east shoulder | (85, −2, −265) | −70 | 2.1 |
| 6 | plaza west wall | (−95, 0, −200) | 175 | 1.8 |
| 7 | plaza east wall | (95, 0, −190) | −5 | 1.9 |
| 8 | west buttress | (−110, 0, −130) | 200 | 1.6 |
| 9 | east buttress | (112, 0, −120) | 15 | 1.7 |
| 10 | avenue west flank | (−120, 0, −30) | 165 | 1.5 |
| 11 | avenue east flank | (125, 0, −10) | −20 | 1.4 |
| 12 | south-west sentinel rock | (−130, 0, 70) | 190 | 1.3 |
| 13 | south-east sentinel rock | (135, 0, 95) | 10 | 1.6 |
| 14 | horizon west | (−200, −6, −420) | 30 | 4.0 |
| 15 | horizon east | (180, −6, −440) | −20 | 3.6 |
| 16 | horizon mesa | (0, −8, −520) | 0 | 5.0 |

### 3.8 Cypress groves (E · cypress_cluster) — 200 instances

Use this hash: `h(n) = fract(sin(n · 12.9898) · 43758.5453)`.

- **Avenue groves (144):** rows at x ∈ {−32, −27, +27, +32}. For `j = 0 … 35`: `z_j = 138 − 8·j`. Row index `r = 0…3` and `n = r·36 + j`. Position `pos = (x + (h(n) − 0.5)·1.2, 0, z_j + (h(n + 7) − 0.5)·2.0)`, `yaw = 360·h(n + 3)`, `scale = 0.85 + 0.35·h(n + 5)`.
- **Plaza crescent (24):** for `k = 0 … 23`: `θ = 20° + (320°/23)·k`, `pos = (58·sin θ, 0, −205 − 58·cos θ)`, `yaw = 360·h(200 + k)`, `scale = 1.05 + 0.4·h(240 + k)`.
- **Arcade screen (32):** x ∈ {50, 70}. For `m = 0 … 15`: `z = 40 − 10·m`, `pos = (x, 0, z)`, `yaw = 360·h(300 + m)`, `scale = 0.95 + 0.3·h(330 + m)`.

### 3.9 Arcade of Echoes: Hi-Z occlusion corridor (C) — 14 instances

`pos = (60, 0, 30 − 10.8·m)` for `m = 0 … 13`, `yaw = 0`, `scale = 1.0`.
The arches sit back to back with touching plinths, which makes one continuous coffered barrel-vault tunnel. It is 151.2 m long (z from +35.4 to −115.8), 8 m wide and 14 m to the crown, with a floor at y = 1.0. From inside the tunnel only the coffers, procession-relief jambs and niches are visible. The other 443 instances (about 101M triangles) are behind solid piers, so this is the textbook HI-Z FREEZE demonstration: freeze at `CORRIDOR`, then fly out to `CANYON` and every occluded instance is missing.

### 3.10 Triangle budget

| Asset | Instances | Tris / instance | Virtual triangles |
|---|---:|---:|---:|
| fluted_column (colonnade 196 + ring 16 + fallen 10) | 222 | 260,616 | 57,856,752 |
| cypress_cluster | 200 | 173,810 | 34,762,000 |
| vaulted_arch (twin 2 + arcade 14) | 16 | 438,688 | 7,019,008 |
| canyon_monolith | 16 | 409,948 | 6,559,168 |
| colossus_sentinel | 3 | 349,116 | 1,047,348 |
| **Total** | **457** | — | **107,244,276** |

---

## 4. Camera presets

`fov` is the vertical field of view in degrees. Near plane is 0.1 m, using reverse-Z with infinite far.

| Preset | Position | Look-at target | fov | Intent |
|---|---|---|---:|---|
| `VISTA` | (0.0, 7.5, 160.0) | (0.0, 14.0, −215.0) | 42 | Hero colonnade. Looks north down 196 columns through ARCH_SOUTH and ARCH_NORTH, which frame the backlit SENTINEL_PRIME and its halo. The sun disk is upper left of the Sentinel and god rays stream through both arch openings. The line of sight passes the arches at y 9.6 and y 11.4, inside both openings. |
| `SENTINEL` | (6.5, 15.0, −186.0) | (0.0, 18.4, −214.0) | 30 | Micro-detail close-up from about 29 m. Shows the hood and serene face, bronze crown spikes, astrolabe ticks and flame rays, and individually sculpted feathers. Rim light comes from the low sun behind. |
| `CANYON` | (−150.0, 110.0, 170.0) | (0.0, 0.0, −120.0) | 50 | High monumental overview: the full avenue, both arches, the plaza terraces, the canyon megalith ring, horizon mesas and the arcade. Uses the most streamed pages. |
| `CORRIDOR` | (60.0, 4.2, 28.0) | (60.0, 6.5, −120.0) | 70 | Deep occlusion tunnel inside the Arcade of Echoes. 14 nested coffered vaults recede to a bright exit portal. Freeze Hi-Z here. |

Suggested idle motion: slow dolly. VISTA moves z 160 → 120 over 40 s. SENTINEL orbits ±12° yaw around (0, 18.4, −214.6) over 30 s.

---

## 5. Lighting: golden hour

| Parameter | Value |
|---|---|
| Direction **to** sun `L` | **(−0.420, 0.240, −0.875)** (normalized). Elevation 13.9°, azimuth 25.6° west of north |
| Light travel direction | (0.420, −0.240, 0.875) |
| Sun color (linear) | (1.00, 0.72, 0.45) |
| Sun illuminance | 6.5 (relative to sky luminance 1.0) |
| Sun angular radius | 0.30° (disk), with a 2.5° soft corona |
| Shadow | 4 cascades: splits 12 / 45 / 160 / 600 m, 2048² each, PCF 5×5, normal bias 0.02 m. Long column shadows rake toward the camera across the avenue |
| Sky zenith (linear) | (0.17, 0.30, 0.60) |
| Sky horizon toward the sun | (1.00, 0.62, 0.34) |
| Sky horizon away from the sun | (0.78, 0.60, 0.50) |
| Ground bounce | (0.42, 0.30, 0.20) × 0.35 |
| Ambient | hemisphere sky/ground × `COLOR_0.a` |
| Height fog | density 0.0032 m⁻¹, height falloff 0.045 m⁻¹, base y −4, color (0.95, 0.66, 0.42) |
| Aerial perspective | Rayleigh scale height 8 km, Mie g = 0.76, Mie tint (1.0, 0.78, 0.55); the far canyon walls fade into warm haze |
| Exposure | EV100 13.2, ACES-style filmic, white point 11.2 |
| Bloom | threshold 1.1, intensity 0.08, radius 6 px (bronze rims and sun corona) |

### God-ray alignment

The sun sits about 26° to the left of and 14° above the VISTA forward axis, so it is inside the frame. Use screen-space radial light shafts from the projected sun position: 64 samples, density 0.85, decay 0.965, weight 0.42, exposure 0.32. The occlusion mask is depth == sky. The shafts pour through both arch openings, between the inner colonnade shafts, and fan around the Sentinel's wings and halo. In `CORRIDOR`, the exit portal at the north end is the only bright opening, so it blooms into a shaft cone down the tunnel.

---

## 6. Materials (PBR metallic-roughness)

| ID | Name | Albedo sRGB (linear) | Roughness | Metallic | Notes |
|---|---|---|---:|---:|---|
| 0 | Sun-bleached travertine / Carrara marble | `#E5DAC7` (0.784, 0.701, 0.565) | 0.58 (0.45 on up-facing, sun-polished tops) | 0.0 | Specular F0 0.04. Wrap/subsurface 0.25 with tint (0.95, 0.78, 0.62) for warm glow at grazing sun. `COLOR_0` already has gray veins and terracotta-brown crevice dirt (0.30, 0.16, 0.08) |
| 1 | Eroded volcanic basalt | `#343236` (0.034, 0.032, 0.037) | 0.86 | 0.0 | Warm dust (0.30, 0.20, 0.12) settles on up-facing ledges. Alternating reddish and gray strata bands are in `COLOR_0` |
| 2 | Patinated aged bronze | `#7A5E3B` (0.195, 0.111, 0.044) | 0.36 exposed / 0.70 verdigris | 1.0 exposed / 0.2 verdigris | Drive the verdigris mask from `COLOR_0` greenness (`g > r`). Exposed rims are gilded (0.62, 0.42, 0.14) |
| 3 | Sun-baked sandstone / terracotta masonry | `#C89B6E` (0.578, 0.332, 0.156) | 0.80 | 0.0 | Layered banding and dark crevices are in `COLOR_0`. Also used as the cypress bark tint |
| 4 | Opaque sculpted Mediterranean foliage | `#2F4A2C` (0.028, 0.069, 0.025) | 0.62 | 0.0 | Fully opaque geometry with no alpha test, so it goes straight through the visibility buffer. Optional backlit wrap 0.35 tint (0.35, 0.55, 0.12) |

---

## 7. Validation record

All 5 `.glb` files pass the following checks:

- glTF 2.0 binary header, JSON and BIN chunk lengths, and accessor/bufferView bounds are consistent.
- Normals are unit length within 1e-3.
- No NaN or Inf values.
- Zero zero-area triangles.
- Indices are in range.
- AO lies in [0.15, 1].
- Grounded at y = 0.

A software turnaround renderer with a sun shadow map and magenta back-face detection was run on every asset: front orthographic, hero 3/4, back 3/4 and a detail close-up. Preview PNGs are saved next to the assets as `preview_<name>.png`.
