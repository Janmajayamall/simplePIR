use std::usize;

use ndarray::{prelude::*, Array, Array5, ArrayD, Dimension, Ix, IxDyn};
use rand::{distributions::Uniform, thread_rng, CryptoRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

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

#[derive(Debug, Clone)]
struct Matrix {
    data: Array2<u32>,
}
impl Matrix {
    // sample zero
    pub fn zero(rows: usize, cols: usize) -> Matrix {
        let data = Array::zeros((rows, cols));
        Matrix { data }
    }

    // sample random
    pub fn random_2d_matrix<R: CryptoRng + RngCore>(
        q: u32,
        rng: &mut R,
        rows: usize,
        cols: usize,
    ) -> Matrix {
        let values = rng
            .sample_iter(Uniform::new(0, q))
            .take(rows * cols)
            .collect::<Vec<u32>>();
        let data =
            Array2::from_shape_vec((rows, cols), values).expect("values length should match shape");

        Matrix { data }
    }

    pub fn seeded_random_2d_matrix(
        q: u32,
        seed: <ChaCha8Rng as SeedableRng>::Seed,
        rows: usize,
        cols: usize,
    ) -> Matrix {
        let prng = ChaCha8Rng::from_seed(seed);
        let values = prng
            .sample_iter(Uniform::new(0, q))
            .take(rows * cols)
            .collect::<Vec<u32>>();
        let data =
            Array2::from_shape_vec((rows, cols), values).expect("values length should match shape");

        Matrix { data }
    }

    // sample gaussian error
    pub fn sample_gaussian<R: CryptoRng + RngCore>(
        q: u32,
        variance: usize,
        rng: &mut R,
        rows: usize,
        cols: usize,
    ) -> Matrix {
        let values = sample_vec_cbd(rows * cols, variance, rng)
            .expect("Gaussian sampling should work")
            .into_iter()
            .map(|v| {
                if v < 0 {
                    // ((q as i64) + v) as u32
                    1
                } else {
                    v as u32
                }
            })
            .collect();
        let data =
            Array2::from_shape_vec((rows, cols), values).expect("Values should match space vec");
        Matrix { data }
    }

    pub fn dot(&self, rhs: &Matrix) -> Matrix {
        let data = self.data.dot(&rhs.data);
        Matrix { data }
    }

    pub fn add(&self, rhs: &Matrix) -> Matrix {
        Matrix {
            data: &self.data + &rhs.data,
        }
    }

    pub fn sub(&self, rhs: &Matrix) -> Matrix {
        Matrix {
            data: &self.data - &rhs.data,
        }
    }

    pub fn set(&mut self, row: usize, col: usize, value: u32) {
        // self.data[row][col] = value;
        let val = self
            .data
            .get_mut((row, col))
            .expect("Row and Col value should be correct");
        *val = value;
    }

    pub fn get(&self, row: usize, col: usize) -> u32 {
        *self
            .data
            .get((row, col))
            .expect("Row and Col value should be correct")
    }
}

#[derive(Clone, Debug)]
struct Params {
    // ciphertext space
    q: u32,
    // plaintext space
    p: u32,
    // number of elements in Database
    db_dim: usize,
    db_dim_sqrt: usize,
    // n param of lwe
    n: usize,
}

struct Server {
    // Seed for A
    hint_s: <ChaCha8Rng as SeedableRng>::Seed,
    // D * A
    hint_c: Matrix,
    // D
    db: Matrix,
    // params
    params: Params,
}

impl Server {
    // setup server for a given db
    fn setup<R: RngCore + CryptoRng>(db: Matrix, params: &Params, rng: &mut R) -> Server {
        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        rng.fill(&mut seed);

        let a = Matrix::random_2d_matrix(params.q, rng, params.db_dim_sqrt, params.n);
        let hint_c = db.dot(&a);

        Server {
            hint_s: seed,
            hint_c,
            db,
            params: params.clone(),
        }
    }

    // answer: takes in db and query and returns ans
    fn answer(&self, query: &Matrix) -> Matrix {
        self.db.dot(query)
    }
}

struct Client {
    params: Params,
    hint_s: <ChaCha8Rng as SeedableRng>::Seed,
    a: Matrix,
    // D * A
    hint_c: Matrix,
}

impl Client {
    /// Takes in a server and returns a client.
    /// Note that client is stateful
    fn new(server: &Server) -> Client {
        let a = Matrix::seeded_random_2d_matrix(server.params.q, server.hint_s, server.params.n, 1);
        Client {
            params: server.params.clone(),
            hint_s: server.hint_s.clone(),
            a,
            hint_c: server.hint_c.clone(),
        }
    }

    /// Prepare a new query
    fn query<R: CryptoRng + RngCore>(&self, rng: &mut R, index: usize) {
        let row = index / self.params.db_dim_sqrt;
        let col = index % self.params.db_dim_sqrt;

        let mut u = Matrix::zero(self.params.db_dim_sqrt, 1);
        // TODO: scale 1 by delta
        u.set(row, 1, 1);

        let sk = Matrix::random_2d_matrix(self.params.q, rng, self.params.n, 1);
        let e = Matrix::sample_gaussian(self.params.q, 10, rng, self.params.db_dim_sqrt, 1);
        // A * sk + e
        let tmp = self.a.dot(&sk).add(&e);
    }
}

fn main() {
    println!("Hello, world!");
    let mut rng = thread_rng();
    let a = Matrix::random_2d_matrix(2147483647, &mut rng, 10, 10);
    let b = Matrix::random_2d_matrix(4, &mut rng, 10, 1);
    // let b = Matrix::sample_gaussian(2, 10, &mut rng, 10, 1);
    let c = a.data.dot(&b.data);
    dbg!(c);
}
