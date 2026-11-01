use engine_math::{clamp, cos, max, sin};

use crate::tables::bytes::{Reader, Writer, f16_bits, f16_value};

pub const TRANSMITTANCE_SIZE: (u32, u32) = (256, 64);
pub const MULTIPLE_SIDE: u32 = crate::gpu_layouts::SKY_MULTIPLE_SIZE as u32;
pub const MULTIPLE_SIZE: (u32, u32) = (MULTIPLE_SIDE, MULTIPLE_SIDE);

const TRANSMITTANCE_STEPS: u32 = 40;
const MULTIPLE_STEPS: u32 = 20;
const MULTIPLE_DIRECTIONS: u32 = 64;
const RADIUS_OFFSET: f32 = 0.01;
const PI: f32 = std::f32::consts::PI;

const TAG: &[u8; 4] = b"SKYT";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Air {
    pub rayleigh: [f32; 3],
    pub rayleigh_height: f32,
    pub mie: f32,
    pub mie_absorption: f32,
    pub mie_height: f32,
    pub ozone: [f32; 3],
    pub ground: [f32; 3],
    pub bottom: f32,
    pub top: f32,
}

impl Air {
    pub fn identity(&self) -> Vec<u8> {
        let mut writer = Writer::new(TAG, 17 * 4);
        for value in self.values() {
            writer.f32(value);
        }
        writer.finish()
    }

    fn values(&self) -> [f32; 17] {
        let [r, g, b] = self.rayleigh;
        let [or, og, ob] = self.ozone;
        let [gr, gg, gb] = self.ground;
        [
            r,
            g,
            b,
            self.rayleigh_height,
            self.mie,
            self.mie_absorption,
            self.mie_height,
            or,
            og,
            ob,
            gr,
            gg,
            gb,
            self.bottom,
            self.top,
            0.0,
            0.0,
        ]
    }

    fn from_identity(bytes: &[u8]) -> Result<Self, String> {
        let mut reader = Reader::new(bytes, TAG)?;
        let mut values = [0.0; 17];
        for value in &mut values {
            *value = reader.f32()?;
        }
        reader.finish()?;
        let [
            r,
            g,
            b,
            rayleigh_height,
            mie,
            mie_absorption,
            mie_height,
            or,
            og,
            ob,
            gr,
            gg,
            gb,
            bottom,
            top,
            ..,
        ] = values;
        Ok(Self {
            rayleigh: [r, g, b],
            rayleigh_height,
            mie,
            mie_absorption,
            mie_height,
            ozone: [or, og, ob],
            ground: [gr, gg, gb],
            bottom,
            top,
        })
    }

    fn medium(&self, height: f32) -> Medium {
        let above = max(height, 0.0);
        let rayleigh_density = libm::expf(-above / self.rayleigh_height);
        let mie_density = libm::expf(-above / self.mie_height);
        let ozone_density = max(0.0, 1.0 - (height - 25.0).abs() / 15.0);
        let scattering: [f32; 3] = std::array::from_fn(|channel| {
            self.rayleigh[channel] * rayleigh_density + self.mie * mie_density
        });
        Medium {
            scattering,
            extinction: std::array::from_fn(|channel| {
                scattering[channel]
                    + self.mie_absorption * mie_density
                    + self.ozone[channel] * ozone_density
            }),
        }
    }
}

struct Medium {
    scattering: [f32; 3],
    extinction: [f32; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct SkyTables {
    pub transmittance: Vec<u16>,
    pub multiple: Vec<u16>,
}

pub(crate) fn record(identity: &[u8]) -> Result<Vec<u8>, String> {
    let air = Air::from_identity(identity)?;
    let transmittance = Table::transmittance(&air);
    let multiple = multiple(&air, &transmittance);
    let mut writer = Writer::new(TAG, (transmittance.texels.len() + multiple.len()) * 2 + 16);
    for size in [TRANSMITTANCE_SIZE, MULTIPLE_SIZE] {
        writer.u32(size.0).u32(size.1);
    }
    for &bits in transmittance.texels.iter().chain(&multiple) {
        writer.u16(bits);
    }
    Ok(writer.finish())
}

pub fn decode(bytes: &[u8]) -> Result<SkyTables, String> {
    let mut reader = Reader::new(bytes, TAG)?;
    for size in [TRANSMITTANCE_SIZE, MULTIPLE_SIZE] {
        if (reader.u32()?, reader.u32()?) != size {
            return Err("the sky tables' record holds other sizes".into());
        }
    }
    let mut texels = |(width, height): (u32, u32)| -> Result<Vec<u16>, String> {
        let bytes = reader.bytes((width * height * 4 * 2) as usize)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect())
    };
    let transmittance = texels(TRANSMITTANCE_SIZE)?;
    let multiple = texels(MULTIPLE_SIZE)?;
    reader.finish()?;
    Ok(SkyTables {
        transmittance,
        multiple,
    })
}

type Vec3 = [f32; 3];

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn along(origin: Vec3, direction: Vec3, distance: f32) -> Vec3 {
    std::array::from_fn(|axis| origin[axis] + direction[axis] * distance)
}

fn sphere_hit(origin: Vec3, direction: Vec3, radius: f32) -> f32 {
    let b = dot(origin, direction);
    let c = dot(origin, origin) - radius * radius;
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return -1.0;
    }
    let root = discriminant.sqrt();
    let near = -b - root;
    let far = -b + root;
    if near >= 0.0 {
        near
    } else if far >= 0.0 {
        far
    } else {
        -1.0
    }
}

fn from_sub_uv(u: f32, size: f32) -> f32 {
    (u - 0.5 / size) * (size / (size - 1.0))
}

struct Table {
    texels: Vec<u16>,
    values: Vec<Vec3>,
}

impl Table {
    fn transmittance(air: &Air) -> Self {
        let (width, height) = TRANSMITTANCE_SIZE;
        let mut texels = Vec::with_capacity((width * height * 4) as usize);
        let mut values = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let uv = [
                    (x as f32 + 0.5) / width as f32,
                    (y as f32 + 0.5) / height as f32,
                ];
                let (r, mu) = transmittance_parameters(air, uv);
                let origin = [0.0, r, 0.0];
                let direction = [max(1.0 - mu * mu, 0.0).sqrt(), mu, 0.0];
                let reach = sphere_hit(origin, direction, air.top);
                let mut depth = [0.0_f32; 3];
                if reach > 0.0 {
                    let dt = reach / TRANSMITTANCE_STEPS as f32;
                    for step in 0..TRANSMITTANCE_STEPS {
                        let point = along(origin, direction, (step as f32 + 0.5) * dt);
                        let extinction = air.medium(length(point) - air.bottom).extinction;
                        for (depth, extinction) in depth.iter_mut().zip(extinction) {
                            *depth += extinction * dt;
                        }
                    }
                }
                let texel = depth.map(|depth| f16_bits(libm::expf(-depth)));
                texels.extend(texel);
                texels.push(f16_bits(1.0));
                values.push(texel.map(f16_value));
            }
        }
        Self { texels, values }
    }

    fn sample(&self, uv: [f32; 2]) -> Vec3 {
        let (width, height) = TRANSMITTANCE_SIZE;
        let x = uv[0] * width as f32 - 0.5;
        let y = uv[1] * height as f32 - 0.5;
        let (column, row) = (x.floor(), y.floor());
        let (fx, fy) = (x - column, y - row);
        let index = |value: f32, size: u32| clamp(value, 0.0, (size - 1) as f32) as usize;
        let (x0, x1) = (index(column, width), index(column + 1.0, width));
        let (y0, y1) = (index(row, height), index(row + 1.0, height));
        let at = |x: usize, y: usize| self.values[y * width as usize + x];
        let mix = |a: Vec3, b: Vec3, t: f32| -> Vec3 {
            std::array::from_fn(|channel| a[channel] + (b[channel] - a[channel]) * t)
        };
        mix(
            mix(at(x0, y0), at(x1, y0), fx),
            mix(at(x0, y1), at(x1, y1), fx),
            fy,
        )
    }

    fn through(&self, air: &Air, r: f32, mu: f32) -> Vec3 {
        let origin = [0.0, r, 0.0];
        let direction = [max(1.0 - mu * mu, 0.0).sqrt(), mu, 0.0];
        if sphere_hit(origin, direction, air.bottom) >= 0.0 {
            return [0.0; 3];
        }
        self.sample(transmittance_uv(air, r, mu))
    }
}

fn length(point: Vec3) -> f32 {
    dot(point, point).sqrt()
}

fn transmittance_uv(air: &Air, r: f32, mu: f32) -> [f32; 2] {
    let (bottom, top) = (air.bottom, air.top);
    let h = max(top * top - bottom * bottom, 0.0).sqrt();
    let rho = max(r * r - bottom * bottom, 0.0).sqrt();
    let discriminant = r * r * (mu * mu - 1.0) + top * top;
    let d = max(0.0, -r * mu + max(discriminant, 0.0).sqrt());
    let d_min = top - r;
    let d_max = rho + h;
    [(d - d_min) / max(d_max - d_min, 1e-6), rho / max(h, 1e-6)]
}

fn transmittance_parameters(air: &Air, uv: [f32; 2]) -> (f32, f32) {
    let (bottom, top) = (air.bottom, air.top);
    let h = max(top * top - bottom * bottom, 0.0).sqrt();
    let rho = h * uv[1];
    let r = (rho * rho + bottom * bottom).sqrt();
    let d_min = top - r;
    let d_max = rho + h;
    let d = d_min + uv[0] * (d_max - d_min);
    let mu = if d > 0.0 {
        clamp((h * h - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0)
    } else {
        1.0
    };
    (r, mu)
}

fn sphere_direction(index: u32, count: u32) -> Vec3 {
    let golden = 2.399_963_1;
    let y = 1.0 - (index as f32 + 0.5) / count as f32 * 2.0;
    let radius = max(1.0 - y * y, 0.0).sqrt();
    let angle = golden * index as f32;
    [cos(angle) * radius, y, sin(angle) * radius]
}

fn multiple(air: &Air, transmittance: &Table) -> Vec<u16> {
    let (width, height) = MULTIPLE_SIZE;
    let mut texels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let uv = [
                from_sub_uv((x as f32 + 0.5) / width as f32, width as f32),
                from_sub_uv((y as f32 + 0.5) / height as f32, height as f32),
            ];
            let sun_cos = clamp(uv[0] * 2.0 - 1.0, -1.0, 1.0);
            let sun = [0.0, sun_cos, max(1.0 - sun_cos * sun_cos, 0.0).sqrt()];
            let r = air.bottom
                + clamp(uv[1] + RADIUS_OFFSET, 0.0, 1.0) * (air.top - air.bottom - RADIUS_OFFSET);
            let origin = [0.0, r, 0.0];
            let mut light = [0.0_f32; 3];
            let mut again = [0.0_f32; 3];
            for index in 0..MULTIPLE_DIRECTIONS {
                let direction = sphere_direction(index, MULTIPLE_DIRECTIONS);
                let (scattered, scattered_again) =
                    scattered(air, transmittance, (origin, direction), sun);
                for channel in 0..3 {
                    light[channel] += scattered[channel];
                    again[channel] += scattered_again[channel];
                }
            }
            let count = MULTIPLE_DIRECTIONS as f32;
            for channel in 0..3 {
                let light = light[channel] / count;
                let again = again[channel] / count;
                let series = 1.0 / max(1.0 - again, 1e-3);
                texels.push(f16_bits(light * series));
            }
            texels.push(f16_bits(1.0));
        }
    }
    texels
}

fn scattered(
    air: &Air,
    transmittance: &Table,
    (origin, direction): (Vec3, Vec3),
    sun: Vec3,
) -> (Vec3, Vec3) {
    let mut light = [0.0_f32; 3];
    let mut again = [0.0_f32; 3];
    let top_hit = sphere_hit(origin, direction, air.top);
    if top_hit < 0.0 {
        return (light, again);
    }
    let ground_hit = sphere_hit(origin, direction, air.bottom);
    let span = if ground_hit >= 0.0 {
        ground_hit
    } else {
        top_hit
    };
    let uniform_phase = 1.0 / (4.0 * PI);
    let mut throughput = [1.0_f32; 3];
    let mut travelled = 0.0;
    for step in 0..MULTIPLE_STEPS {
        let t = span * (step as f32 + 0.3) / MULTIPLE_STEPS as f32;
        let dt = t - travelled;
        travelled = t;
        let point = along(origin, direction, t);
        let r = length(point);
        let up = point.map(|axis| axis / r);
        let medium = air.medium(r - air.bottom);
        let sun_cos = dot(sun, up);
        let toward_sun = transmittance.through(air, r, sun_cos);
        let shadowed = sphere_hit(point, sun, air.bottom) >= 0.0;
        for channel in 0..3 {
            let sample = libm::expf(-medium.extinction[channel] * dt);
            let scattering = medium.scattering[channel];
            let source =
                if shadowed { 0.0 } else { toward_sun[channel] } * scattering * uniform_phase;
            let extinction = max(medium.extinction[channel], 1e-6);
            light[channel] += throughput[channel] * (source - source * sample) / extinction;
            again[channel] += throughput[channel] * (scattering - scattering * sample) / extinction;
            throughput[channel] *= sample;
        }
    }
    if ground_hit >= 0.0 {
        let point = along(origin, direction, span);
        let up = point.map(|axis| axis / length(point));
        let sun_cos = dot(sun, up);
        if sun_cos > 0.0 {
            let through = transmittance.sample(transmittance_uv(air, air.bottom, sun_cos));
            for channel in 0..3 {
                light[channel] +=
                    throughput[channel] * through[channel] * sun_cos * air.ground[channel] / PI;
            }
        }
    }
    (light, again)
}
