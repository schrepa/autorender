//! The spread of the wind's waves about its heading, as measured seas spread
//! (Donelan, Hamilton and Hui 1985; Banner 1990: [`spreading`]), narrow at
//! the sea's peak and wider above it, by which the sea's own waves head
//! ([`spread_turn`]).

use std::f64::consts::PI;

/// The spread of the wind's waves about its heading as measured seas spread
/// (Donelan, Hamilton and Hui 1985; Banner 1990, as Horvath 2015 gives
/// them), D(θ) = β / (2 tanh(βπ)) sech²(βθ) of the angle θ from the
/// heading, from −π to π: its β for waves whose angular frequency is
/// `ratio` times the peak's, r, a ratio under 0.56 spreading as 0.56 does.
/// β is 2.61 r^1.3 below 0.95, 2.28 r^−1.3 below 1.6, and
/// 10^(−0.4 + 0.8393 e^(−0.567 ln r²)) beyond, so the spread is narrowest at
/// the peak (β about 2.3), where almost none of the waves run against the
/// wind, and wider above it.
pub fn spreading(ratio: f64) -> f64 {
    let ratio = ratio.max(0.56);
    if ratio < 0.95 {
        2.61 * libm::pow(ratio, 1.3)
    } else if ratio < 1.6 {
        2.28 * libm::pow(ratio, -1.3)
    } else {
        libm::pow(
            10.0,
            -0.4 + 0.8393 * libm::exp(-0.567 * libm::log(ratio * ratio)),
        )
    }
}

/// The angle (radians) from the wind's heading below which the spread of
/// [`spreading`] puts the share (1 + `draw`) / 2 of the waves `ratio` times
/// the peak's angular frequency, `draw` running from −1 to 1: D's inverse
/// distribution, atanh(u tanh(βπ)) / β for u = `draw`, so draws spread
/// evenly over that range head waves as the sea spreads them, either way
/// from the heading as `draw`'s sign is.
pub fn spread_turn(ratio: f64, draw: f64) -> f64 {
    let beta = spreading(ratio);
    libm::atanh(draw * libm::tanh(beta * PI)) / beta
}
