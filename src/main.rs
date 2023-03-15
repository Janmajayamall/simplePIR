#![feature(inline_const)]
use std::usize;

use matrix::Matrix;
use rand::{distributions::Uniform, thread_rng, CryptoRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

mod matrix;
mod server;
mod utils;

struct Server<
    const DB_R: usize,
    const DB_C: usize,
    const DB_RC: usize,
    const N: usize,
    const DB_RN: usize,
    const DB_CN: usize,
> {
    // Seed for A
    hint_s: <ChaCha8Rng as SeedableRng>::Seed,
    // D * A
    hint_c: Matrix<DB_R, N, DB_RN>,
    // D
    db: Matrix<DB_R, DB_C, DB_RC>,
}

impl<
        const DB_R: usize,
        const DB_C: usize,
        const DB_RC: usize,
        const N: usize,
        const DB_RN: usize,
        const DB_CN: usize,
    > Server<DB_R, DB_C, DB_RC, N, DB_RN, DB_CN>
{
    // setup server for a given db
    fn setup<A: RngCore + CryptoRng>(
        db: Matrix<DB_R, DB_C, DB_RC>,
        rng: &mut A,
    ) -> Server<DB_R, DB_C, DB_RC, N, DB_RN, DB_CN> {
        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        rng.fill(&mut seed);

        //TODO: LogQ
        let a = Matrix::<DB_C, N, DB_CN>::random_from_seed(seed, 32);
        let hint_c = db.mul(&a);

        Server {
            hint_s: seed,
            hint_c,
            db,
        }
    }

    // answer: takes in db and query and returns ans
    fn answer(&self, query: &Matrix<DB_C, 1, DB_C>) -> Matrix<DB_R, 1, DB_R> {
        self.db.mul(query)
    }
}

struct QueryState<const N: usize, const DB_C: usize> {
    query: Matrix<DB_C, 1, DB_C>,
    sk: Matrix<N, 1, N>,
    row: usize,
}

struct Client<
    const DB_R: usize,
    const DB_C: usize,
    const N: usize,
    const DB_RC: usize,
    const DB_RN: usize,
    const DB_CN: usize,
> {
    hint_s: <ChaCha8Rng as SeedableRng>::Seed,
    a: Matrix<DB_C, N, DB_CN>,
    /// D * A
    hint_c: Matrix<DB_R, N, DB_RN>,
}

impl<
        const DB_R: usize,
        const DB_C: usize,
        const N: usize,
        const DB_RC: usize,
        const DB_RN: usize,
        const DB_CN: usize,
    > Client<DB_R, DB_C, N, DB_RC, DB_RN, DB_CN>
{
    /// Takes in a server and returns a client.
    /// Note that client is stateful
    pub fn new(
        server: &Server<DB_R, DB_C, DB_RC, N, DB_RN, DB_CN>,
    ) -> Client<DB_R, DB_C, N, DB_RC, DB_RN, DB_CN> {
        //TODO: logq
        let a = Matrix::random_from_seed(server.hint_s, 32);
        Client {
            hint_s: server.hint_s,
            a,
            hint_c: server.hint_c.clone(),
        }
    }

    /// Prepare a new query
    pub fn query<R: CryptoRng + RngCore>(&self, rng: &mut R, index: usize) -> QueryState<N, DB_C> {
        let row = index / DB_C;
        let col = index % DB_C;

        let mut u = Matrix::<DB_C, 1, DB_C>::zeros();
        u.set_at(col, 0, self.delta());

        //TODO: logq
        let sk = Matrix::random(rng, 32);
        let e = Matrix::<DB_C, 1, DB_C>::gaussian_matrix(10, rng);

        // A * sk + e + (delta * u)
        let query = self.a.mul(&sk).add(&e).add(&u);

        QueryState { query, sk, row }
    }

    /// Run the recovery process
    /// Takes in query state, hint_c, ans
    fn recover(&self, query_state: &QueryState<N, DB_C>, ans: &Matrix<DB_R, 1, DB_R>) -> u32 {
        let row = self.hint_c.get_row(query_state.row);
        let sk = query_state.sk.get_data();
        let mut inner_product = 0u32;
        for i in 0..N {
            inner_product = inner_product.wrapping_add(row[i].wrapping_mul(sk[i]));
        }
        let tmp = ans.get_at(query_state.row, 0);
        let tmp = tmp.wrapping_sub(inner_product);

        //TODO: place P
        self.scale_down(tmp) % (256 as u32)
    }

    pub const fn delta(&self) -> u32 {
        //TODO: logq and p
        ((1u64 << 32) / (256 as u64)) as u32
    }

    pub const fn scale_down(&self, value: u32) -> u32 {
        let delta = self.delta();
        (value + (delta / 2)) / delta
    }
}

fn main() {
    // log of ciphertext modulus
    const LOGQ: usize = 32;

    // plaintext modulus
    const P: usize = 276;

    // Database params
    const ENTRIES: usize = 1 << 18;
    const DB_C: usize = 1 << 9;
    const DB_R: usize = 1 << 9;
    const DB_RC: usize = DB_C * DB_R;
    const DB_RN: usize = DB_R * N;
    const DB_CN: usize = DB_C * N;

    // n param of lwe
    const N: usize = 10;
    /// variance
    const VARIANCE: usize = 10;

    let mut rng = thread_rng();
    let mut entries = [0u32; ENTRIES];
    let distr = Uniform::new(0, 256u32);
    entries.iter_mut().for_each(|v| *v = rng.sample(distr));
    let db = Matrix::<DB_R, DB_C, DB_RC>::from_data(entries);

    let server = Server::<DB_R, DB_C, DB_RC, N, DB_RN, DB_CN>::setup(db, &mut rng);
    let client = Client::<DB_R, DB_C, N, DB_RC, DB_RN, DB_CN>::new(&server);

    for i in 0..1000 {
        let index = rng.sample(Uniform::new(0, ENTRIES));
        let query_state = client.query(&mut rng, index);
        let mut now = std::time::Instant::now();
        let ans = server.answer(&query_state.query);
        println!("Server response time: {:?}", now.elapsed());
        now = std::time::Instant::now();
        let res = client.recover(&query_state, &ans);
        println!("Client recover time: {:?}", now.elapsed());
        assert_eq!(res, entries[index]);
    }
}
