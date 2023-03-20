use ndarray::Data;
use rand::{CryptoRng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    database::{self, Params},
    matrix2::Matrix,
};

use super::database::Database;

/// Ratio between first level db and second level db
const COMP_RATIO: usize = 64;

struct SharedState {
    seed: <ChaCha8Rng as SeedableRng>::Seed,
}

struct State {
    data: Vec<Matrix>,
}

struct Msg {
    data: Vec<Matrix>,
}

pub struct DoublePir {
    db: Database,
    state: State,
    msg: Msg,
}
impl DoublePir {
    pub fn setup<R: CryptoRng + RngCore>(
        mut db: Database,
        params: &Params,
        rng: &mut R,
    ) -> DoublePir {
        let a1 = Matrix::random(params.m, params.n, db.db_info.logq, rng);
        let mut a2 = Matrix::random(params.l / db.db_info.x, params.n, db.db_info.logq, rng);

        let mut h1 = db.data.mul(&a1);
        h1 = h1.transpose();
        h1 = h1.expand(params.delta(), params.p);
        h1 = h1.concat_cols(db.db_info.ne);

        let h2 = h1.mul(&a2);

        db.squish(10, 3);
        h1 = h1.squish(10, 3);

        if a2.rows % 3 == 0 {
            let zeros = Matrix::zeros(3 - (a2.rows % 3), a2.cols);
            a2.concat_matrix(&zeros);
        }
        a2 = a2.transpose();

        let state = State { data: vec![h1, a2] };
        let msg = Msg { data: vec![h2] };
        DoublePir { db, state, msg }
    }

    pub fn pick_params(n_entries: usize, row_length: usize, n: usize, logq: usize) -> Params {
        let mut good_params = None;
        let mut found = false;

        // Iteratively find p and dimensions for tight fit
        let mut modp = 2u32;
        loop {
            let (l, m) =
                Database::approximate_database_dims(n_entries, row_length, modp, COMP_RATIO * n);

            let params = Params::pick_params(l, m);
            if params.p < modp {
                if !found {
                    panic!("Error; Should not happen")
                }
                return good_params.unwrap();
            }

            good_params = Some(params);
            found = true;
        }
    }
}
