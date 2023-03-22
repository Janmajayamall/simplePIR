use rand::{CryptoRng, RngCore};

/// Taken from https://github.com/tlepoint/fhe.rs/blob/bbe69178b65e04e8b93605b664b895d83289c5a4/crates/fhe-util/src/lib.rs#L38
pub fn sample_vec_cbd<R: RngCore + CryptoRng>(
    vector_size: usize,
    variance: usize,
    rng: &mut R,
) -> Result<Vec<i64>, &'static str> {
    if !(1..=16).contains(&variance) {
        return Err("The variance should be between 1 and 16");
    }

    let mut out = Vec::with_capacity(vector_size);

    let number_bits = 4 * variance;
    let mask_add = ((u64::MAX >> (64 - number_bits)) >> (2 * variance)) as u128;
    let mask_sub = (mask_add << (2 * variance)) as u128;

    let mut current_pool = 0u128;
    let mut current_pool_nbits = 0;

    for _ in 0..vector_size {
        if current_pool_nbits < number_bits {
            current_pool |= (rng.next_u64() as u128) << current_pool_nbits;
            current_pool_nbits += 64;
        }
        debug_assert!(current_pool_nbits >= number_bits);
        out.push(
            ((current_pool & mask_add).count_ones() as i64)
                - ((current_pool & mask_sub).count_ones() as i64),
        );
        current_pool >>= number_bits;
        current_pool_nbits -= number_bits;
    }

    Ok(out)
}

pub fn reconstruct_val_from_basep(p: u64, values: &[u64]) -> u32 {
    let mut res = 0;
    let mut coeff = 1u64;
    values.iter().for_each(|v| {
        res += coeff * v;
        coeff *= p;
    });
    res as u32
}
