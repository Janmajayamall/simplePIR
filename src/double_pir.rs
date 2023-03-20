use core::panicking::panic;

use ndarray::Data;

use crate::database::Params;

use super::database::Database;

/// Ratio between first level db and second level db
const COMP_RATIO: usize = 64;

pub struct DoublePir {}
impl DoublePir {
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
