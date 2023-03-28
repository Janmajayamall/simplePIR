#![feature(inline_const)]
#![feature(inherent_associated_types)]
use std::usize;

use crate::database::Database;
use crate::double_pir::DoublePir;
use matrix::Matrix;
use rand::{distributions::Uniform, thread_rng, CryptoRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

mod database;
mod double_pir;
mod matrix;
mod matrix2;
mod utils;

struct Server<
    const DB_R: usize,
    const DB_C: usize,
    const DB_RC: usize,
    const N: usize,
    const DB_RN: usize,
    const DB_CN: usize,
    const LOGQ: usize,
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
        const LOGQ: usize,
    > Server<DB_R, DB_C, DB_RC, N, DB_RN, DB_CN, LOGQ>
{
    // setup server for a given db
    fn setup<A: RngCore + CryptoRng>(
        db: Matrix<DB_R, DB_C, DB_RC>,
        rng: &mut A,
    ) -> Server<DB_R, DB_C, DB_RC, N, DB_RN, DB_CN, LOGQ> {
        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        rng.fill(&mut seed);

        let a = Matrix::<DB_C, N, DB_CN>::random_from_seed(seed, LOGQ);
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
    const LOGQ: usize,
    const P: usize,
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
        const LOGQ: usize,
        const P: usize,
    > Client<DB_R, DB_C, N, DB_RC, DB_RN, DB_CN, LOGQ, P>
{
    /// Takes in a server and returns a client.
    /// Note that client is stateful
    pub fn new(
        server: &Server<DB_R, DB_C, DB_RC, N, DB_RN, DB_CN, LOGQ>,
    ) -> Client<DB_R, DB_C, N, DB_RC, DB_RN, DB_CN, LOGQ, P> {
        let a = Matrix::random_from_seed(server.hint_s, LOGQ);
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
        u.set_at(col, 0, Self::delta());

        let sk = Matrix::random(rng, LOGQ);
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

        self.scale_down(tmp) % (P as u32)
    }

    pub const fn delta() -> u32 {
        ((1u64 << LOGQ) / (P as u64)) as u32
    }

    pub const fn scale_down(&self, value: u32) -> u32 {
        let delta = Self::delta();
        (value + (delta / 2)) / delta
    }
}

fn simple_pir() {
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
    const N: usize = 1024;
    /// variance
    const VARIANCE: usize = 10;

    let mut rng = thread_rng();
    let mut entries = [0u32; ENTRIES];
    let distr = Uniform::new(0, 256u32);
    entries.iter_mut().for_each(|v| *v = rng.sample(distr));
    let db = Matrix::<DB_R, DB_C, DB_RC>::from_data(entries);

    let server = Server::<DB_R, DB_C, DB_RC, N, DB_RN, DB_CN, LOGQ>::setup(db, &mut rng);
    let client = Client::<DB_R, DB_C, N, DB_RC, DB_RN, DB_CN, LOGQ, P>::new(&server);

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

fn estimate_cost(n_entries: usize, row_length: usize, n: usize, logq: usize, qc: f64) {
    let params = DoublePir::pick_params(n_entries, row_length, n, logq);
    println!("Params: {:?}", params);

    let (_, ne, _) = Database::number_db_entries(n_entries, row_length, params.p);

    let d = ne as f64;
    println!("d: {d}");
    let k = (params.logq as f64 / (params.p as f64).log2()).ceil();
    let n = params.n as f64;
    let logq = logq as f64;
    let l = params.l as f64;
    let m = params.m as f64;

    // offline download size
    // d * k * n^2
    let offline_download_size = (d * k * n * n * logq) / 8.0 / 1024.0 / 1024.0;
    println!("Offline download size: {} Mb", offline_download_size);

    // online upload size
    // (m + l/d) * qc
    let online_upload_size = (m + l / d) * logq * qc / 8.0 / 1024.0;
    println!("Online upload size: {} Kb", online_upload_size);

    // online download size
    // dkn + qc(dkn + dk)
    let online_upload_size = ((d * k * n) + qc * (d * k * n + d * k)) * logq / 8.0 / 1024.0;
    println!("Online download size: {} Kb", online_upload_size);
}

fn main() {
    // return;
    let n_entries = 100000000;
    let row_length = 32;
    let n = 1 << 10;
    let logq = 32;

    estimate_cost(n_entries, row_length, n, logq, 2.0);
    return;

    let params = DoublePir::pick_params(n_entries, row_length, n, logq);
    println!("Params: {:?}", params);

    let db = Database::random(n_entries, row_length, &params);
    println!("Db Info: {:?}", db.db_info);

    let indices = vec![1, 0];
    let query_count = indices.len();
    let batch_size = db.data.rows / (query_count * db.db_info.ne) * db.data.cols;

    let mut rng = thread_rng();
    let shared_state = DoublePir::init_shared_state(&mut rng);

    let (a1, a2) = DoublePir::ret_shared_state(&shared_state, &params, &db);

    println!("Setup...");
    let pir = DoublePir::setup(db, &params, &shared_state);

    let clue_size = (pir.msg.size() as f64 * (params.logq as f64)) / 8.0 / 1024.0;
    println!("Offline download size: {:?} Kb", clue_size);

    // query
    println!("Query...");
    let mut client_states = vec![];
    let mut msgs = vec![];
    for i in 0..indices.len() {
        let index_to_query = indices[i] + batch_size * i;
        let (st, msg) = pir.query(index_to_query, &a1, &a2);
        client_states.push(st);
        msgs.push(msg);
    }

    let online_upload = msgs.iter().map(|m| m.size()).sum::<usize>();
    let online_upload = (online_upload as f64 * (params.logq as f64)) / 8.0 / 1024.0;
    println!("Online upload size: {:?} Kb", online_upload);

    println!("Answer...");
    let (h2_p2, ans_msgs) = pir.answer(msgs);

    let mut online_download = ans_msgs.iter().map(|m| m.size()).sum::<usize>();
    online_download += h2_p2.data.len();
    let online_download = (online_download as f64 * (params.logq as f64)) / 8.0 / 1024.0;
    println!("Online download size: {:?} Kb", online_download);

    let h2_p1 = &pir.msg.data[0];
    println!("Recovery...");
    pir.recover(h2_p1, &h2_p2, client_states, ans_msgs);
}
