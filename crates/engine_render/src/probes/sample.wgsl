const PROBE_ENTRY_WORDS: u32 = 3u;
@group(0) @binding(13) var<storage, read> probe_words: array<vec4<u32>>;

override PROBES: bool = true;
