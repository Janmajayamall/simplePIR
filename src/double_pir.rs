use std::ops::Add;

use itertools::Itertools;
use ndarray::Data;
use rand::{thread_rng, CryptoRng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    database::{self, Params},
    matrix2::Matrix,
    utils::reconstruct_val_from_basep,
};

use super::database::Database;

/// Ratio between first level db and second level db
const COMP_RATIO: usize = 64;

pub struct SharedState {
    seed: <ChaCha8Rng as SeedableRng>::Seed,
}

pub struct State {
    data: Vec<Matrix>,
}

impl State {
    pub fn size(&self) -> usize {
        let mut sum = 0;
        self.data.iter().for_each(|m| {
            sum += m.data.len();
        });
        sum
    }
}

pub struct Msg {
    pub data: Vec<Matrix>,
}

impl Msg {
    pub fn size(&self) -> usize {
        let mut sum = 0;
        self.data.iter().for_each(|m| {
            sum += m.data.len();
        });
        sum
    }
}

pub struct DoublePir {
    pub params: Params,
    pub db: Database,
    pub state: State,
    pub msg: Msg,
}
impl DoublePir {
    pub fn init_shared_state<R: CryptoRng + RngCore>(rng: &mut R) -> SharedState {
        let mut seed: <ChaCha8Rng as SeedableRng>::Seed = Default::default();
        rng.fill_bytes(&mut seed);
        SharedState { seed }
    }

    pub fn ret_shared_state(
        shared_state: &SharedState,
        params: &Params,
        db: &Database,
    ) -> (Matrix, Matrix) {
        let mut prng = ChaCha8Rng::from_seed(shared_state.seed);
        let a1 = Matrix::random(params.m, params.n, 1u64 << db.db_info.logq, &mut prng);
        let mut a2 = Matrix::random(
            params.l / db.db_info.x,
            params.n,
            1u64 << db.db_info.logq,
            &mut prng,
        );
        (a1, a2)
    }

    pub fn setup(mut db: Database, params: &Params, shared_state: &SharedState) -> DoublePir {
        let (a1, mut a2) = DoublePir::ret_shared_state(shared_state, params, &db);

        let mut h1 = db.data.mul(&a1);
        h1 = h1.transpose();
        h1 = h1.expand(params.delta_expansion(), params.p);
        h1 = h1.concat_cols(db.db_info.ne);
        let h2 = h1.mul(&a2);

        db.squish(10, 3);
        h1 = h1.squish(10, 3);

        // pad with zero rows to accommodate packed multiplication with delta 3
        if a2.rows % 3 != 0 {
            let zeros = Matrix::zeros(3 - (a2.rows % 3), a2.cols);
            a2.concat_matrix(&zeros);
        }
        a2 = a2.transpose();
        let state = State { data: vec![h1, a2] };
        let msg = Msg { data: vec![h2] };

        DoublePir {
            db,
            state,
            msg,
            params: params.clone(),
        }
    }

    pub fn pick_params(n_entries: usize, row_length: usize, n: usize, logq: usize) -> Params {
        let mut good_params = None;
        let mut found = false;

        // Iteratively find p and dimensions for tight fit
        let mut modp = 2u32;
        loop {
            let (l, m) =
                Database::approximate_database_dims(n_entries, row_length, modp, COMP_RATIO * n);

            let params = Params::pick_params(n, logq, l, m);
            if params.p < modp {
                if !found {
                    panic!("Error; Should not happen")
                }
                return good_params.unwrap();
            }

            modp += 1;
            good_params = Some(params);
            found = true;
        }
    }

    // returns client state and message
    pub fn query<R: CryptoRng + RngCore>(
        &self,
        index: usize,
        a1: &Matrix,
        a2: &Matrix,
        rng: &mut R,
    ) -> (State, Msg) {
        let row = index / self.params.m;
        let col = index % self.params.m;
        dbg!(row, col, index);

        let err1 = Matrix::gaussian(self.params.m, 1, 10, rng);
        // FIXME
        // let err1 = Matrix::zeros(self.params.m, 1);
        let sk1 = Matrix::random(self.params.n, 1, 1 << self.params.logq, rng);
        let mut query1 = a1.mul(&sk1).add(&err1);
        query1.data[col] += self.params.delta();

        if query1.rows % 3 != 0 {
            let zeros = Matrix::zeros(3 - (query1.rows % 3), 1);
            query1.concat_matrix(&zeros);
        }

        let err2 = Matrix::gaussian(self.params.l / self.db.db_info.x, 1, 10, rng);
        // FIXME
        // let err2 = Matrix::zeros(self.params.l / self.db.db_info.x, 1);
        let sk2 = Matrix::random(self.params.n, 1, 1 << self.params.logq, rng);
        let mut query2 = a2.mul(&sk2).add(&err2);
        query2.data[row] += self.params.delta();

        if query2.rows % 3 != 0 {
            let zeros = Matrix::zeros(3 - (query2.rows % 3), 1);
            query2.concat_matrix(&zeros);
        }

        let state = State {
            data: vec![sk1, sk2],
        };
        let msg = Msg {
            data: vec![query1, query2],
        };

        (state, msg)
    }

    pub fn answer(&self, msgs: Vec<Msg>) -> (Matrix, Vec<Msg>) {
        let batch_size = self.db.data.rows / msgs.len();
        let batch_count = msgs.len();

        let mut ans1 = Matrix::zeros(0, 1);
        let mut last = 0;
        // TODO: take care of last batch
        for i in 0..batch_count {
            let q1 = &msgs[i].data[0];

            let a1 = self.db.data.select_rows(last, batch_size);
            let a1 = a1.matrix_mul_vec_packed(q1, 10, 3);

            ans1.concat_matrix(&a1);
            last += batch_size;
        }
        let ans1 = ans1.transpose_and_expand_and_concat_cols_and_squish(
            self.db.db_info.p,
            self.params.delta_expansion(),
            self.db.db_info.ne,
            10,
            3,
        );

        let h1 = &self.state.data[0];
        let a2_transposed = &self.state.data[1];
        let h2 = ans1.matrix_mul_transposed_packed(a2_transposed, 10, 3);

        let mut ans_msgs: Vec<Msg> = vec![];
        for i in 0..batch_count {
            let q2 = &msgs[i].data[1];
            let h = h1.matrix_mul_vec_packed(q2, 10, 3);
            let ans2 = ans1.matrix_mul_vec_packed(q2, 10, 3);
            ans_msgs.push(Msg {
                data: vec![h, ans2],
            })
        }

        (h2, ans_msgs)
    }

    pub fn recover(
        &self,
        h2_p1: &Matrix,
        h2_p2: &Matrix,
        client_states: Vec<State>,
        msgs: Vec<Msg>,
    ) -> Vec<u32> {
        let batch_count = client_states.len();
        let mut res = vec![];
        for i in 0..batch_count {
            // let a2_sk2 = a2.mul(&client_states[i].data[1]);
            let tmp1 = h2_p1.mul(&client_states[i].data[1]);
            let tmp2 = h2_p2.mul(&client_states[i].data[1]);
            let mut h = msgs[i].data[0].sub(&tmp1);
            let mut ans = msgs[i].data[1].sub(&tmp2);
            h.scale_down(self.params.delta() as u64);
            ans.scale_down(self.params.delta() as u64);

            h = h.contract(self.params.delta_expansion(), self.params.p);
            ans = ans.contract(self.params.delta_expansion(), self.params.p);

            assert!(
                h.data.len() / self.params.n / self.db.db_info.ne
                    == ans.data.len() / self.db.db_info.ne
            );
            // dbg!(h.data.chunks(self.db.db_info.ne).len());
            let values = h
                .data
                .chunks_exact(self.params.n)
                .zip(ans.data.iter())
                .map(|(h_chunks, a)| {
                    assert_eq!(h_chunks.len(), client_states[i].data[0].data.len());
                    let mut inner_product: u32 = 0;
                    h_chunks
                        .iter()
                        .zip(client_states[i].data[0].data.iter())
                        .for_each(|(v, s)| {
                            inner_product = inner_product.wrapping_add(v.wrapping_mul(*s));
                        });
                    self.params.scale_down(a.wrapping_sub(inner_product)) as u64
                })
                .collect_vec();
            let v = reconstruct_val_from_basep(self.db.db_info.p as u64, &values);
            res.push(v);

            // dbg!(h.data.chunks(self.db.db_info.ne).len());
        }
        res
    }
}
