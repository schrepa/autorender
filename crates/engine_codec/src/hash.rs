pub fn content_hash(bytes: &[u8]) -> u128 {
    const C1: u64 = 0x87c3_7b91_1142_53d5;
    const C2: u64 = 0x4cf5_ad43_2745_937f;
    let mix = |mut k: u64| {
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        k ^ (k >> 33)
    };
    let word = |bytes: &[u8]| {
        let mut word = [0_u8; 8];
        word[..bytes.len()].copy_from_slice(bytes);
        u64::from_le_bytes(word)
    };
    let (mut h1, mut h2) = (0_u64, 0_u64);
    let (blocks, tail) = bytes.as_chunks::<16>();
    for block in blocks {
        let (k1, k2) = (word(&block[..8]), word(&block[8..]));
        h1 ^= k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 = h1
            .rotate_left(27)
            .wrapping_add(h2)
            .wrapping_mul(5)
            .wrapping_add(0x52dc_e729);
        h2 ^= k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 = h2
            .rotate_left(31)
            .wrapping_add(h1)
            .wrapping_mul(5)
            .wrapping_add(0x3849_5ab5);
    }
    if tail.len() > 8 {
        h2 ^= word(&tail[8..])
            .wrapping_mul(C2)
            .rotate_left(33)
            .wrapping_mul(C1);
    }
    if !tail.is_empty() {
        h1 ^= word(&tail[..tail.len().min(8)])
            .wrapping_mul(C1)
            .rotate_left(31)
            .wrapping_mul(C2);
    }
    let len = bytes.len() as u64;
    h1 ^= len;
    h2 ^= len;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = mix(h1);
    h2 = mix(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    (u128::from(h2) << 64) | u128::from(h1)
}
