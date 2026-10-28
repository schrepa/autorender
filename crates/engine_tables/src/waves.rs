use std::f64::consts::PI;

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

pub fn spread_turn(ratio: f64, draw: f64) -> f64 {
    let beta = spreading(ratio);
    libm::atanh(draw * libm::tanh(beta * PI)) / beta
}
