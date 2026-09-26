# Crystal Garden

An external-style engine client: this crate imports only the public `engine` façade.
It loads [`playground.scene.json`](../../web/public/scenes/playground.scene.json), attaches game components once from authored tags, and registers movement, collection, and animation systems with explicit ordering.

Open `/playground.html`. Click the scene, use WASD/arrows to move, and Space to jump.
Touch devices show equivalent movement buttons. Collect all six crystals. Reset player moves the avatar back to its authored spawn; Reload scene starts a fresh instance from the saved source.

`Playground::new` installs the plugin before startup. `reload` parses/validates and instantiates the replacement before disposing the old instance; failures preserve the previous world and score. `unload` disposes owned entities/assets. Tags are read once during setup, never looked up in per-entity frame loops. Exactly one `player` tag is required.

The game installs `CollisionPlugin` and uses engine shape casts for movement,
landing and obstacle contacts. A child of the floor mesh authors its collision
volume. Player/world/pickup layer bits are 2/1/4; the controller queries solids
and collection consumes sensor enters after `CollisionSystems::Detect` in
`FixedPostSimulation`. Gravity, jump speed, axis order and scoring remain game
policy. Query failures stop movement and increment inspected `collision_errors`.
This is a small controller example, with no slope stepping or penetration solver.

The browser binding exposes a bounded progress record on request with frame/revision, player position, score, and asset residency. It does not expose unrestricted world reflection or serialize per entity every frame. Common lifecycle/input/diagnostics live in `web/src/host.ts`; the game's host module only fetches scene text and displays score.

The version 4 / revision 5 scene includes two one-entry placements of a reusable
`garden-lantern` prefab. Their two shared meshes use one 64×64 sRGB atlas, adding
two draws to the four original mesh batches. A golden crystal is parented to the
player and follows its movement. The scene has 30 top-level records expanding
to 34 entities, 31 renderables, six referenced meshes and one 16 KiB texture.
The original procedural GLB and generator live in `assets/source`; import/cooking
is native and placement stays in scene JSON. Decorative props have no collision
tags. The example requires authored colliders for `player`, `solid`, and `pickup`
tags. Player and animated pickup motion requires root transforms; static solids
can use arbitrary valid hierarchy transforms or have no render mesh.
See [asset import](../../docs/asset-import.md) for reproducible import and replacement.

The player and animated pickups opt into the engine's `InterpolateTransform`
component. The accessory inherits its parent's presented motion while collision
and collection use authoritative transforms. Reset player calls the public
`reset_interpolation` primitive for that subtree, then resets the clock. Paused
agent stepping and default captures stay exact; explicit fractional previews
exercise the same engine presentation path used by ordinary browser frames.
See [render interpolation](../../docs/adr-008-render-interpolation.md).

Tests exercise actual movement/collection, jumping once during catch-up, landing on boxes, reload failure preservation, repeated replacement without asset growth, and explicit disposal. Browser checks cover the same scene in the WebGPU build.
