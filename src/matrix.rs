use crate::utils::sample_vec_cbd;
use rand::{distributions::Uniform, CryptoRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct Matrix<const R: usize, const C: usize, const L: usize> {
    data: [u32; L],
}

impl<const R: usize, const C: usize, const L: usize> Matrix<R, C, L> {
    const MATCHES: () = assert!(L == R * C);

    pub const fn zeros() -> Matrix<R, C, L> {
        let _ = Self::MATCHES;
        Matrix { data: [0; L] }
    }

    pub fn from_data(data: [u32; L]) -> Matrix<R, C, L> {
        let _ = Self::MATCHES;
        Matrix { data }
    }

    pub fn random<A: CryptoRng + RngCore>(rng: &mut A, logp: usize) -> Matrix<R, C, L> {
        let _ = Self::MATCHES;

        let last = ((1u64 << logp) - 1) as u32;
        let distr = Uniform::new_inclusive(0, last);
        let mut out = Matrix::<R, C, L>::zeros();
        out.data.iter_mut().for_each(|v| *v = rng.sample(distr));
        out
    }

    pub fn random_from_seed(
        seed: <ChaCha8Rng as SeedableRng>::Seed,
        logp: usize,
    ) -> Matrix<R, C, L> {
        let _ = Self::MATCHES;

        let mut prng = ChaCha8Rng::from_seed(seed);
        Matrix::random(&mut prng, logp)
    }

    pub fn gaussian_matrix<A: CryptoRng + RngCore>(
        variance: usize,
        rng: &mut A,
    ) -> Matrix<R, C, L> {
        let _ = Self::MATCHES;

        let values = sample_vec_cbd(R * C, variance, rng)
            .expect("Sampling from gaussian distribution should work")
            .iter()
            .map(|v| *v as u32)
            .collect::<Vec<u32>>();
        let mut out = Matrix::zeros();
        out.data.copy_from_slice(values.as_slice());
        out
    }

    pub fn add(&self, rhs: &Matrix<R, C, L>) -> Matrix<R, C, L> {
        let mut out = Matrix::<R, C, L>::zeros();
        for i in 0..self.data.len() {
            out.data[i] = self.data[i].wrapping_add(rhs.data[i]);
        }
        out
    }

    pub fn sub(&self, rhs: &Matrix<R, C, L>) -> Matrix<R, C, L> {
        let mut out = Matrix::<R, C, L>::zeros();
        for i in 0..self.data.len() {
            out.data[i] = self.data[i].wrapping_sub(rhs.data[i]);
        }
        out
    }

    pub fn mul<const C1: usize, const L1: usize, const LR: usize>(
        &self,
        rhs: &Matrix<C, C1, L1>,
    ) -> Matrix<R, C1, LR> {
        let mut out = Matrix::<R, C1, LR>::zeros();
        for i in 0..R {
            for j in 0..C {
                for k in 0..C1 {
                    out.data[(i * C1) + k] = out.data[(i * C1) + k]
                        .wrapping_add(self.data[(i * C) + j].wrapping_mul(rhs.data[(j * C1) + k]));
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

    pub fn transpose(&self) -> Matrix<C, R, L> {
        let mut out = Matrix::zeros();
        for i in 0..R {
            for j in 0..C {
                out.data[j * R + i] = self.data[i * C + j];
            }
        }
        out
    }

    pub fn concat_matrix<const R1: usize, const L1: usize, const RR: usize, const LR: usize>(
        &self,
        other: &Matrix<R1, C, L1>,
    ) -> Matrix<RR, C, LR> {
        const {
            assert!(RR == R1 + R);
        }
        let mut out = Matrix::<RR, C, LR>::zeros();
        out.data[..L].copy_from_slice(self.data.as_slice());
        out.data[L..].copy_from_slice(other.data.as_slice());
        out
    }

    pub fn expand<const R1: usize, const L1: usize>(&self, modp: u32) -> Matrix<R1, C, L1> {
        const {
            //TODO: enforce that `R1 / R` should `logQ` / `logP`
            assert!(R1 % R == 0);
        }
        let mut out = Matrix::<R1, C, L1>::zeros();
        let delta = R1 / R;

        for i in 0..R {
            for j in 0..C {
                let mut val = self.data[i * C + j];
                for k in 0..delta {
                    let v = val % modp;
                    out.data[(((i * delta) + k) * C) + j] = v;
                    val /= modp;
                }
            }
        }
        out
    }

    //TODO: enforce `R % R1 == 0`
    pub fn recomp<const R1: usize, const L1: usize>(&self, logp: usize) -> Matrix<R1, C, L1> {
        const {
            assert!(R % R1 == 0);
        }
        let mut out = Matrix::<R1, C, L1>::zeros();
        let delta = R / R1;
        for i in 0..R1 {
            for j in 0..C {
                let mut val = out.data.get_mut(i * C + j).unwrap();
                for k in 0..delta {
                    *val += self.data[(((i * delta) + k) * C) + j] << (logp * k);
                }
            }
        }
        out
    }

    pub fn print(&self) {
        println!();
        for i in 0..R {
            for j in 0..C {
                print!("{},", self.data[i * C + j]);
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_expand() {
        // let mut rng = thread_rng();
        let seed: [u8; 32] = [0; 32];
        let a1 = Matrix::<5, 5, 25>::random_from_seed(seed, 32);
        // let a2 = Matrix::<5, 5, 25>::random_from_seed(seed, 32);

        let c1 = a1.expand::<{ 5 * 4 }, { 5 * 5 * 4 }>(256);
        let c2 = c1.recomp::<5, 25>(8);
        // dbg!(a1.data);
        // dbg!(c1.data);
        assert_eq!(a1.data, c2.data);
    }

    #[test]
    fn test() {
        // let mut rng = thread_rng();
        let seed: [u8; 32] = [0; 32];
        let a1 = Matrix::<1024, 1, 1024>::random_from_seed(seed, 32);
        dbg!(a1.print());
    }
}
