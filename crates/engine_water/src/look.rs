use engine_core::math::clamp;

use crate::Water;
use crate::gpu::{
    WaterFallLook, WaterFlowLook, WaterFoamLook, WaterOpticsLook, WaterReliefLook, WaterRippleLook,
    WaterShoreLook, WaterUniform,
};
use crate::water_detail::RippleMeasure;

pub(crate) fn looked(water: &Water, measured: &RippleMeasure, frame: WaterUniform) -> WaterUniform {
    WaterUniform {
        flow: flow(water),
        ripples: ripples(water, measured),
        relief: relief(water),
        foam: foam(water),
        shore: shore(water),
        optics: optics(water),
        falls: falls(water),
        ..frame
    }
}

fn flow(water: &Water) -> WaterFlowLook {
    let flow = &water.flow;
    let [soonest, latest] = flow.drift_seconds;
    let seconds = |wavelength: f32| clamp(flow.drift_ride * wavelength, soonest, latest);
    WaterFlowLook {
        moving: flow.moving,
        shown: flow.shown,
        frame: flow.frame,
        carry_slowest: flow.carry_slowest,
        carry_step: flow.carry_step,
        carry_log: flow.carry_step.log2(),
        rung_band: flow.rung_band,
        relief_steepest: flow.relief_steepest,
        ripple_seconds: seconds(water.ripples.fine_tile),
        boil_seconds: seconds(water.boils.across),
    }
}

fn ripples(water: &Water, measured: &RippleMeasure) -> WaterRippleLook {
    let ripples = &water.ripples;
    let reach = |deviation: f32| {
        if deviation > 0.0 {
            ripples.reach / deviation
        } else {
            0.0
        }
    };
    let [coarse, fine] = measured.deviation.map(reach);
    WaterRippleLook {
        kept: measured.kept,
        reach: [coarse, coarse, fine, fine],
        boil_churning: water.boils.churning,
        tile: [ripples.tile * ripples.stretch, ripples.tile],
        fine_tile: ripples.fine_tile,
        stretch: ripples.stretch,
        coarse: ripples.coarse,
        skew: ripples.skew,
        least: ripples.least,
        share: ripples.share,
        fine_share: ripples.fine_share,
        boil_steep: water.boils.steep,
    }
}

fn relief(water: &Water) -> WaterReliefLook {
    let bumps = &water.bumps;
    WaterReliefLook {
        bump_froude: bumps.froude,
        roughness: water.optics.roughness,
        bump_steep: bumps.steep,
        bump_depths: bumps.depths,
        bump_deep: bumps.deep,
        bump_shortest: bumps.shortest,
        bump_crests: bumps.crests,
        bump_crest_gain: 1.0 / (0.5 + 0.5 / (bumps.crests * bumps.crests)).sqrt(),
        bump_flicker: bumps.flicker,
        boil_across: water.boils.across,
    }
}

fn foam(water: &Water) -> WaterFoamLook {
    let foam = &water.foam;
    let lines = &water.drift_lines;
    WaterFoamLook {
        fades: foam.fades,
        covers: foam.covers,
        tearing: foam.tearing,
        froude: foam.froude,
        breaking: foam.breaking,
        drop: foam.drop,
        blocking: foam.blocking,
        crested: foam.crested,
        around: foam.around,
        rough: foam.rough,
        drift_amount: lines.amount,
        drift_gathers: lines.gathers,
        tile: foam.tile,
        steep: foam.steep,
        soft: foam.soft,
        halo: foam.halo,
        body: foam.body,
        filmy: foam.filmy,
        optical: foam.optical,
        hollow: foam.hollow,
        albedo: foam.albedo,
        sky: foam.sky,
        mirrors: foam.mirrors,
        open: foam.open,
    }
}

fn shore(water: &Water) -> WaterShoreLook {
    let shore = &water.shore;
    let lace = &water.lace;
    WaterShoreLook {
        racing: shore.racing,
        lace_depth: lace.depth,
        lace_speed: lace.speed,
        fade: shore.fade,
        edge: shore.edge,
        edge_pixels: shore.edge_pixels,
        wet: shore.wet,
        soaked: shore.soaked,
        lace_amount: lace.amount,
    }
}

fn optics(water: &Water) -> WaterOpticsLook {
    let optics = &water.optics;
    let caustics = &water.caustics;
    let forward = optics.forward;
    let back = 1.0 / (1.0 + forward * forward).sqrt() - 1.0 / (1.0 + forward);
    WaterOpticsLook {
        phase_forward: forward,
        phase_mean: (1.0 - forward * forward) / forward * back,
        bubbles: optics.bubbles,
        bubble_albedo: optics.bubble_albedo,
        refraction_reach: optics.refraction_reach,
        caustic_contrast: caustics.contrast,
        caustic_most: caustics.most,
        caustic_depth: caustics.depth,
    }
}

fn falls(water: &Water) -> WaterFallLook {
    let falls = &water.falls;
    let [curtains, ropes, strands] = falls.scale_shares;
    WaterFallLook {
        scale_shares: [curtains, ropes, strands, 0.0],
        wander_scale: falls.wander_scale,
        curtain_scale: falls.curtain_scale,
        rope_scale: falls.rope_scale,
        strand_scale: falls.strand_scale,
        plunge_drowns: falls.plunge_drowns,
        pad: [0.0; 2],
        wander_reach: falls.wander_reach,
        gathering: falls.gathering,
        glassy_gathering: falls.glassy_gathering,
        tear_thickness: falls.tear_thickness,
        rock_roughness: falls.rock_roughness,
        broken_stands: falls.broken_stands,
        tear_softness: falls.tear_softness,
        edge_thinning: falls.edge_thinning,
        relief_steepest: falls.relief_steepest,
        furrow_shade: falls.furrow_shade,
        matte_roughness: falls.matte_roughness,
        albedo: falls.albedo,
        thin_tint: falls.thin_tint,
        scattering: falls.scattering,
        forward: falls.forward,
        faintest: falls.faintest,
    }
}
