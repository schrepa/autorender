# Architecture decision 005: bounded textured imports

Status: adopted. Extends the asset/scene contracts in decision 002.

[Decision 006](adr-006-hierarchy-and-prefabs.md) later adds version 3 hierarchy,
prefab import and current schema generation; texture ownership remains unchanged.

The concrete requirement is imported textured scenery in Crystal Garden, authored and verified through agent commands. `engine_assets` owns validated immutable opaque RGBA8 textures, weak generational handles, mesh UVs, and material dependencies. `engine_scene` owns the portable texture representation and transactional admission. `engine_core` extracts referenced CPU sources; `engine_render` owns texture uploads, samplers, bind groups, and retirement. No GPU or JavaScript types enter ordinary gameplay.

The native-only `tools/assetc` crate converts a deliberately limited GLB/PNG profile through shared scene validation. It has no browser dependency and does not become a runtime dependency. `cargo xtask asset import` owns files and revision-checked persistence; the importer itself is a pure bytes-to-document function. Unsupported glTF features fail explicitly. Source SHA-256, importer version and fixed settings appear in the result for reproducible build records. There is no implicit cache or background importer.

Scene version 2 adds textures, material texture references, and vertex UVs. Legacy untextured version 1 remains readable. Version 1 cannot silently acquire textured semantics; clients must upgrade the version and advance the revision. The v1 schema stays frozen, and the v2 schema is generated from Rust. Development metadata is also generated from these types in host builds rather than embedding a potentially stale schema file.

Cooked output uses the current self-contained 1 MiB JSON scene representation with base64 decoded RGBA8 pixels. This keeps one validation/save/reload path and supports small assets without an additional transport protocol. PNG decode and transform baking stay offline; base64 and JSON decode occur only during load/validation. A binary streaming format needs a larger concrete asset workload and a separate versioned contract. There is no claim that JSON is the final format for large production scenes.

Materials are opaque, with a linear color factor and optional sRGB texture. Sampling is clamp-to-edge, linear, single-mip; the importer rejects incompatible samplers rather than altering them. The renderer groups instances by mesh and optional texture handle. Color variations still share a batch. A white fallback keeps untextured content on the same pipeline. This adds one texture lookup in the fragment shader and 8 UV bytes per vertex; frame instance records remain unchanged.

CPU ownership is one `Arc` per unique referenced texture in the reusable snapshot, not per entity. Missing texture dependencies skip the instance and increment the existing missing-asset count. Default CPU payload remains 32 MiB across assets. GPU textures have an independent 64-entry / 32 MiB budget, are uploaded once per residency, and retire when no longer referenced at the next valid render boundary. Residency queries describe cached resources, not GPU completion. Texture counters exclude the permanent 4-byte white fallback and driver metadata.

Scene preparation validates all texture bytes and dependencies before world mutation. Failure during asset admission rolls back staged meshes, textures and materials. Replacements stage the new scene before disposing the old, preserving the existing game on validation or admission errors. Import replacement uses explicit generated namespaces and the existing revision-checked writer; matching source identities keep authored entity placement and tags.

Agent observations expose dimensions, per-ID CPU readiness, browser GPU residency, texture memory and upload counters without serializing pixels. Native contracts and real browser tests verify malformed-source preservation, visible texture changes, steady-state uploads, repeated replacement and retirement. Performance claims still require measured workloads and devices.
