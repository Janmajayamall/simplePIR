use rand::thread_rng;

use crate::matrix2::Matrix;

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
    db_info: DatabaseInfo,
    data: Matrix,
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

        let mut rng = thread_rng();
        let data = Matrix::random(params.l, params.m, db_info.logq, &mut rng);

        assert!(params.l * params.m >= db_elements);
        assert!(params.l % db_info.ne == 0);

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
}

pub struct Params {
    /// LWE secret dimension
    n: u32,
    /// LWE standard deviation
    sigma: f64,
    /// DB height
    l: usize,
    /// DB width
    m: usize,
    /// Log of ciphertext modulus
    logq: usize,
    /// plaintext modulus
    pub p: u32,
}

impl Params {
    pub fn pick_params(l: usize, m: usize) -> Params {
        todo!()
    }
}
