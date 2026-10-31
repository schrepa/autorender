const COVER_FAR_SIDE: f32 = 0.0;
const COVER_ROUGHNESS: f32 = 0.0;

struct CoverLaw {
    span: f32,
}

struct CoverGrowth {
    height: f32,
    density: f32,
    tint: vec3<f32>,
}

struct CoverClumped {
    height: f32,
    tint: vec3<f32>,
}

struct CoverSeen {
    tip: f32,
    occlusion: f32,
}

fn cover_growth(lush: f32) -> CoverGrowth {
    var growth: CoverGrowth;
    growth.height = 1.0;
    growth.tint = vec3<f32>(1.0);
    return growth;
}

fn cover_tint(lush: f32, vegetated: f32) -> vec3<f32> {
    return vec3<f32>(1.0);
}

fn cover_clump_spacing(layer: u32) -> f32 {
    return 1.0;
}

fn cover_clump_layer(at: vec2<f32>, layer: u32) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn cover_clumped(look: vec4<f32>) -> CoverClumped {
    var clumped: CoverClumped;
    clumped.height = 1.0;
    clumped.tint = vec3<f32>(1.0);
    return clumped;
}

fn cover_share(away: f32, law: CoverLaw) -> f32 {
    return 0.0;
}

fn cover_shown(away: f32, share: f32, law: CoverLaw) -> f32 {
    return 0.0;
}

fn cover_seen(depth: f32) -> CoverSeen {
    var seen: CoverSeen;
    seen.occlusion = 1.0;
    return seen;
}
