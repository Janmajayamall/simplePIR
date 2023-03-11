#![feature(generic_const_exprs)]
use std::usize;

use matrix::Matrix;
use rand::{distributions::Uniform, thread_rng, CryptoRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

mod matrix;
mod utils;

#[derive(Clone, Debug)]
struct Params {
    /// log of ciphertext modulus
    logq: u32,
    /// plaintext modulus
    p: u32,
    /// number of elements in Database
    db_dim: usize,
    db_dim_sqrt: usize,
    /// n param of lwe
    n: usize,
    /// variance
    variance: usize,
}

impl Params {
    fn delta(&self) -> u32 {
        (1 << self.logq) / self.p
    }
}

struct Server {
    // Seed for A
    hint_s: <ChaCha8Rng as SeedableRng>::Seed,
    // D * A
    hint_c: Ma,
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

        let a = Matrix::random_2d_matrix(params.logq, rng, params.db_dim_sqrt, params.n);
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

struct QueryState {
    query: Matrix,
    sk: Matrix,
    row: usize,
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
        let a =
            Matrix::seeded_random_2d_matrix(server.params.logq, server.hint_s, server.params.n, 1);
        Client {
            params: server.params.clone(),
            hint_s: server.hint_s.clone(),
            a,
            hint_c: server.hint_c.clone(),
        }
    }

    /// Prepare a new query
    fn query<R: CryptoRng + RngCore>(&self, rng: &mut R, index: usize) -> QueryState {
        let row = index / self.params.db_dim_sqrt;
        let col = index % self.params.db_dim_sqrt;

        let mut u = Matrix::zero(self.params.db_dim_sqrt, 1);
        u.set(col, 1, self.params.delta());

        let sk = Matrix::random_2d_matrix(self.params.logq, rng, self.params.n, 1);
        let e = Matrix::sample_gaussian(10, rng, self.params.db_dim_sqrt, 1);
        // A * sk + e + (delta * u)
        let query = self.a.dot(&sk).add(&e).add(&u);

        QueryState { query, sk, row }
    }

    /// Run the recovery process
    /// Takes in query state, hint_c, ans
    fn recover(query_state: &QueryState, ans: &Matrix) {}
}

fn main() {}
