use rand::thread_rng;

use crate::matrix2::Matrix;

/// log(n),log(m),log(q),sigma,log(p_simple),p_simple,p_double
const PARAMS: [(usize, usize, usize, f64, usize, u32, u32); 9] = [
    (10, 13, 32, 6.400000, 9, 991, 929),
    (10, 14, 32, 6.400000, 9, 833, 781),
    (10, 15, 32, 6.400000, 9, 701, 657),
    (10, 16, 32, 6.400000, 9, 589, 552),
    (10, 17, 32, 6.400000, 8, 495, 464),
    (10, 18, 32, 6.400000, 8, 416, 390),
    (10, 19, 32, 6.400000, 8, 350, 328),
    (10, 20, 32, 6.400000, 8, 294, 276),
    (10, 21, 32, 6.400000, 7, 247, 231),
];

#[derive(Debug)]
pub struct DatabaseInfo {
    /// Number of DB entries
    pub n_entries: usize,
    /// Number of bits per entry
    pub row_length: usize,

    /// Number of DB entries per Z_p element, if log(p) > row_length
    pub packing: usize,
    /// Number of Z_p elements per DB entry, if log(p) < row_length
    pub ne: usize,

    /// Governs communication cost. Must be in the range [1, ne] and divisor of ne.
    /// Represents no. of times the scheme is repeated.
    pub x: usize,
    /// Plaintext modulus
    pub p: u32,
    /// Log of ciphertext modulus
    pub logq: usize,

    // For in-memory DB compression
    pub basis: usize,
    pub squishing: usize,
}
impl DatabaseInfo {
    pub fn new(
        n_entries: usize,
        row_length: usize,
        packing: usize,
        ne: usize,
        x: usize,
        p: u32,
        logq: usize,
        basis: usize,
        squishing: usize,
    ) -> DatabaseInfo {
        DatabaseInfo {
            n_entries,
            row_length,
            packing,
            ne,
            x,
            p,
            logq,
            basis,
            squishing,
        }
    }
}

pub struct Database {
    pub db_info: DatabaseInfo,
    pub data: Matrix,
}

impl Database {
    pub fn random(n_entries: usize, row_length: usize, params: &Params) -> Database {
        let (db_elements, elements_per_entry, entries_per_element) =
            Database::number_db_entries(n_entries, row_length, params.p);
        let db_info = DatabaseInfo::new(
            n_entries,
            row_length,
            entries_per_element,
            elements_per_entry,
            elements_per_entry,
            params.p,
            params.logq,
            10,
            3,
        );

        dbg!(&db_info);
        assert!(params.l * params.m >= db_elements);
        assert!(params.l % db_info.ne == 0);

        let mut rng = thread_rng();
        let data = Matrix::random(params.l, params.m, db_info.p as u64, &mut rng);

        Database { db_info, data }
    }

    pub fn approximate_square_database_dims(
        n_entries: usize,
        row_length: usize,
        p: u32,
    ) -> (usize, usize) {
        let (db_entries, elements_per_entry, _) =
            Database::number_db_entries(n_entries, row_length, p);

        let mut l = (db_entries as f64).sqrt().floor() as usize;
        let r = l % elements_per_entry;
        if r != 0 {
            l += elements_per_entry - r;
        }

        let m = (db_entries as f64 / l as f64).ceil() as usize;

        (l, m)
    }

    pub fn approximate_database_dims(
        n_entries: usize,
        row_length: usize,
        p: u32,
        m_lower_bound: usize,
    ) -> (usize, usize) {
        let (l, m) = Database::approximate_square_database_dims(n_entries, row_length, p);
        if m >= m_lower_bound {
            return (l, m);
        }

        let m = m_lower_bound;
        let (db_entries, elements_per_entry, _) =
            Database::number_db_entries(n_entries, row_length, p);
        let mut l = ((db_entries as f64) / m as f64).ceil() as usize;
        let r = l % elements_per_entry;
        if r != 0 {
            l += elements_per_entry - r;
        }
        (l, m)
    }

    /// Returns number of Z_p elements needed for n_elements DB entries
    pub fn number_db_entries(n_entries: usize, row_length: usize, p: u32) -> (usize, usize, usize) {
        if (row_length as f64) <= (p as f64).log2() {
            let no_of_elems = ((p as f64).log2() / row_length as f64).floor();
            (
                (n_entries as f64 / no_of_elems) as usize,
                1,
                no_of_elems as usize,
            )
        } else {
            let no_of_p = (row_length as f64 / (p as f64).log2()).ceil() as usize;
            (no_of_p * n_entries, no_of_p, 0)
        }
    }

    pub fn squish(&mut self, basis: usize, delta: usize) {
        self.data = self.data.squish(basis, delta);
    }
}

#[derive(Debug, Clone)]
pub struct Params {
    /// LWE secret dimension
    pub n: usize,
    /// LWE standard deviation
    pub sigma: f64,
    /// DB height
    pub l: usize,
    /// DB width
    pub m: usize,
    /// Log of ciphertext modulus
    pub logq: usize,
    /// plaintext modulus
    pub p: u32,
}

impl Params {
    pub fn pick_params(n: usize, logq: usize, l: usize, m: usize) -> Params {
        assert!(n != 0);
        assert!(logq != 0);

        let num_samples = l.max(m);

        for params in PARAMS {
            // find params that exactly matches LWE security parameter `n`
            // AND
            // holds 128 bit security after revealing `num_samples` samples
            // AND
            // ciphertext modulus matches desired ciphertext modulus `1<<logq`
            if n == (1 << params.0) && num_samples <= (1 << params.1) && logq == params.2 {
                return Params {
                    n,
                    sigma: params.3,
                    l,
                    m,
                    logq,
                    ///TODO: remove double pir assumption
                    p: params.6,
                };
            }
        }
        panic!("No suitable params known")
    }

    pub fn delta_expansion(&self) -> usize {
        (self.logq as f64 / (self.p as f64).log2()).ceil() as usize
    }

    /// Scaling factor
    pub fn delta(&self) -> u32 {
        ((1u64 << self.logq) / (self.p as u64)) as u32
    }
}
