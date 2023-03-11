#![feature(generic_const_exprs)]
use std::usize;

use matrix::Matrix;
use rand::{distributions::Uniform, thread_rng, CryptoRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

mod matrix;
mod utils;

struct Server<const DB_DIM: usize, const N: usize, const LOGQ: usize>
where
    [(); DB_DIM * N]:,
    [(); DB_DIM * DB_DIM]:,
{
    // Seed for A
    hint_s: <ChaCha8Rng as SeedableRng>::Seed,
    // D * A
    hint_c: Matrix<DB_DIM, N>,
    // D
    db: Matrix<DB_DIM, DB_DIM>,
}

impl<const DB_DIM: usize, const N: usize, const LOGQ: usize> Server<DB_DIM, N, LOGQ>
where
    [(); DB_DIM * N]:,
    [(); DB_DIM * DB_DIM]:,
    [(); DB_DIM * 1]:,
{
    // setup server for a given db
    fn setup<A: RngCore + CryptoRng>(
        db: Matrix<DB_DIM, DB_DIM>,
        rng: &mut A,
    ) -> Server<DB_DIM, N, LOGQ> {
        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        rng.fill(&mut seed);

        let a = Matrix::<DB_DIM, N>::random_from_seed(seed, LOGQ);
        let hint_c = db.mul(&a);

        Server {
            hint_s: seed,
            hint_c,
            db,
        }
    }

    // answer: takes in db and query and returns ans
    fn answer(&self, query: &Matrix<DB_DIM, 1>) -> Matrix<DB_DIM, 1> {
        self.db.mul(query)
    }
}

struct QueryState<const N: usize, const DB_DIM: usize>
where
    [(); DB_DIM * 1]:,
    [(); N * 1]:,
{
    query: Matrix<DB_DIM, 1>,
    sk: Matrix<N, 1>,
    row: usize,
}

struct Client<const DB_DIM: usize, const N: usize, const LOGQ: usize, const P: usize>
where
    [(); DB_DIM * N]:,
{
    hint_s: <ChaCha8Rng as SeedableRng>::Seed,
    a: Matrix<DB_DIM, N>,
    // D * A
    hint_c: Matrix<DB_DIM, N>,
}

impl<const DB_DIM: usize, const N: usize, const LOGQ: usize, const P: usize>
    Client<DB_DIM, N, LOGQ, P>
where
    [(); DB_DIM * N]:,
    [(); DB_DIM * DB_DIM]:,
    [(); N * 1]:,
    [(); DB_DIM * 1]:,
{
    /// Takes in a server and returns a client.
    /// Note that client is stateful
    pub fn new(server: &Server<DB_DIM, N, LOGQ>) -> Client<DB_DIM, N, LOGQ, P> {
        let a = Matrix::random_from_seed(server.hint_s, LOGQ);
        Client {
            hint_s: server.hint_s,
            a,
            hint_c: server.hint_c.clone(),
        }
    }

    /// Prepare a new query
    pub fn query<R: CryptoRng + RngCore>(
        &self,
        rng: &mut R,
        index: usize,
    ) -> QueryState<N, DB_DIM> {
        let row = index / DB_DIM;
        let col = index % DB_DIM;

        let mut u = Matrix::<DB_DIM, 1>::zeros();
        u.set_at(col, 0, self.delta());

        let sk = Matrix::random(rng, LOGQ);
        let e = Matrix::<DB_DIM, 1>::gaussian_matrix(10, rng);

        // A * sk + e + (delta * u)
        let query = self.a.mul(&sk).add(&e).add(&u);

        QueryState { query, sk, row }
    }

    /// Run the recovery process
    /// Takes in query state, hint_c, ans
    fn recover(&self, query_state: &QueryState<N, DB_DIM>, ans: &Matrix<DB_DIM, 1>) -> u32 {
        let row = self.hint_c.get_row(query_state.row);
        let sk = query_state.sk.get_data();
        let mut inner_product = 0u32;
        for i in 0..N {
            inner_product = inner_product.wrapping_add(row[i].wrapping_mul(sk[i]));
        }
        let tmp = ans.get_at(query_state.row, 0);
        let tmp = tmp.wrapping_sub(inner_product);
        self.scale_down(tmp) % (P as u32)
    }

    pub fn delta(&self) -> u32 {
        (1 << LOGQ) / (P as u32)
    }

    pub fn scale_down(&self, value: u32) -> u32 {
        let delta = self.delta();
        (value + (delta / 2)) / delta
    }
}

fn main() {
    // log of ciphertext modulus
    const LOGQ: u32 = 1;
    const DELTA: u32 = (1 << LOGQ) / LOGP;

    // plaintext modulus
    const P: u32 = 890;
    const LOGP: u32 = 10;

    // Database params
    const DIM_DB: usize = 100;
    const DIM_DB_SQRT: usize = 10;

    // n param of lwe
    const N: usize = 10;
    /// variance
    const VARIANCE: usize = 10;
}
