use crate::utils::sample_vec_cbd;
use rand::{distributions::Uniform, CryptoRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct Matrix<const R: usize, const C: usize>
where
    [(); R * C]:,
{
    data: [u32; R * C],
}

impl<const R: usize, const C: usize> Matrix<R, C>
where
    [(); R * C]:,
{
    pub fn zeros() -> Matrix<R, C> {
        Matrix { data: [0; R * C] }
    }

    pub fn from_data(data: [u32; R * C]) -> Matrix<R, C> {
        Matrix { data }
    }

    pub fn random<A: CryptoRng + RngCore>(rng: &mut A, logp: usize) -> Matrix<R, C> {
        let last = ((1u64 << logp) - 1) as u32;
        let distr = Uniform::new_inclusive(0, last);
        let mut out = Matrix::<R, C>::zeros();
        out.data.iter_mut().for_each(|v| *v = rng.sample(distr));
        out
    }

    pub fn random_from_seed(seed: <ChaCha8Rng as SeedableRng>::Seed, logp: usize) -> Matrix<R, C> {
        let mut prng = ChaCha8Rng::from_seed(seed);
        Matrix::random(&mut prng, logp)
    }

    pub fn gaussian_matrix<A: CryptoRng + RngCore>(variance: usize, rng: &mut A) -> Matrix<R, C> {
        let values = sample_vec_cbd(R * C, variance, rng)
            .expect("Sampling from gaussian distribution should work")
            .iter()
            .map(|v| *v as u32)
            .collect::<Vec<u32>>();
        let mut out = Matrix::zeros();
        out.data.copy_from_slice(values.as_slice());
        out
    }

    pub fn add(&self, rhs: &Matrix<R, C>) -> Matrix<R, C> {
        let mut out = Matrix::<R, C>::zeros();
        for i in 0..self.data.len() {
            out.data[i] = self.data[i].wrapping_add(rhs.data[i]);
        }
        out
    }

    pub fn sub(&self, rhs: &Matrix<R, C>) -> Matrix<R, C> {
        let mut out = Matrix::<R, C>::zeros();
        for i in 0..self.data.len() {
            out.data[i] = self.data[i].wrapping_sub(rhs.data[i]);
        }
        out
    }

    pub fn mul<const L: usize>(&self, rhs: &Matrix<C, L>) -> Matrix<R, L>
    where
        [(); C * L]:,
        [(); R * L]:,
    {
        let mut out = Matrix::<R, L>::zeros();
        for i in 0..R {
            for j in 0..C {
                for k in 0..L {
                    out.data[(i * L) + k] = out.data[(i * L) + k]
                        .wrapping_add(self.data[(i * C) + j].wrapping_mul(rhs.data[(j * L) + k]));
                }
            }
        }
        out
    }

    pub fn set_at(&mut self, row: usize, col: usize, value: u32) {
        assert!(row < R);
        assert!(col < C);

        self.data[row * C + col] = value;
    }

    pub fn get_at(&self, row: usize, col: usize) -> u32 {
        assert!(row < R);
        assert!(col < C);

        self.data[row * C + col]
    }

    pub fn get_row(&self, row: usize) -> &[u32] {
        assert!(row < R);

        self.data[(row * C)..((row * C) + C)].as_ref()
    }

    pub fn get_data(&self) -> &[u32] {
        self.data.as_ref()
    }

    pub fn get_data_mut(&mut self) -> &mut [u32] {
        self.data.as_mut_slice()
    }

    pub fn transpose(&self) -> Matrix<C, R>
    where
        [(); C * R]:,
    {
        let mut out = Matrix::zeros();
        for i in 0..R {
            for j in 0..C {
                out.data[j * R + i] = self.data[i * C + j];
            }
        }
        out
    }

    pub fn concat_row(&self, values: &[u32]) -> Matrix<{ R + 1 }, C>
    where
        [(); (R + 1) * C]:,
    {
        let mut out = Matrix::<{ R + 1 }, C>::zeros();
        out.data[..(R * C)].copy_from_slice(self.data.as_slice());
        out.data[(R * C)..].copy_from_slice(values);
        out
    }

    pub fn concat_matrix<const L: usize>(&self, other: &Matrix<L, C>) -> Matrix<{ R + L }, C>
    where
        [(); L * C]:,
        [(); (R + L) * C]:,
    {
        let mut out = Matrix::<{ R + L }, C>::zeros();
        out.data[..(R * C)].copy_from_slice(self.data.as_slice());
        out.data[(R * C)..].copy_from_slice(other.get_data());
        out
    }

    pub fn expand<const DELTA: usize>(&self, modp: u32) -> Matrix<{ R * DELTA }, C>
    where
        [(); R * DELTA * C]:,
    {
        let mut out = Matrix::<{ R * DELTA }, C>::zeros();
        for i in 0..R {
            for j in 0..C {
                let mut val = self.data[i * C + j];
                for k in 0..DELTA {
                    let v = val % modp;
                    out.data[(((i * k) + k) * C) + j] = v;
                }
            }
        }
        out
    }

    pub fn recomp<const DELTA: usize>(&self, logp: usize) -> Matrix<{ R / DELTA }, C>
    where
        [(); R / DELTA * C]:,
    {
        let mut out = Matrix::<{ R / DELTA }, C>::zeros();
        for i in 0..(R / DELTA) {
            for j in 0..C {
                let mut val = out.data.get_mut(i * C + j).unwrap();
                for k in 0..DELTA {
                    *val += self.data[(((i * k) + k) * C) + j] << (logp * k);
                }
            }
        }
        out
    }

    pub fn shape(&self) {
        println!("row: {}, col: {}", R, C);
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;

    #[test]
    fn new() {
        let mut rng = thread_rng();
        let a = Matrix::<10, 20>::random(&mut rng, 32);
        let a_expanded = a.expand::<4>(256);

        dbg!(a_expanded);
    }
}
