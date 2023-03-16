use crate::matrix::Matrix;
use rand::{CryptoRng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub struct Server<
    const DB_R: usize,
    const DB_C: usize,
    const N: usize,
    const DELTA: usize,
    const NBD: usize,
    const DB_RC: usize,
    const NBD_N: usize,
    const NBD_DBR: usize,
    const DB_RN: usize,
    const DB_CN: usize,
    const LOGQ: usize,
    const P: u32,
> {
    hint_s: <ChaCha8Rng as SeedableRng>::Seed,
    hint_c: Matrix<NBD, N, NBD_N>,
    // D
    db: Matrix<DB_R, DB_C, DB_RC>,
    db_a1_transposed: Matrix<NBD, DB_R, NBD_DBR>,
    a2: Matrix<DB_R, N, DB_RN>,
}

impl<
        const DB_R: usize,
        const DB_C: usize,
        const N: usize,
        const DELTA: usize,
        const NBD: usize,
        const DB_RC: usize,
        const NBD_N: usize,
        const NBD_DBR: usize,
        const DB_RN: usize,
        const DB_CN: usize,
        const LOGQ: usize,
        const P: u32,
    > Server<DB_R, DB_C, N, DELTA, NBD, DB_RC, NBD_N, NBD_DBR, DB_RN, DB_CN, LOGQ, P>
{
    pub fn setup<A: RngCore + CryptoRng>(
        db: &Matrix<DB_R, DB_C, DB_RC>,
        rng: &mut A,
    ) -> Server<DB_R, DB_C, N, DELTA, NBD, DB_RC, NBD_N, NBD_DBR, DB_RN, DB_CN, LOGQ, P> {
        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        rng.fill_bytes(&mut seed);

        let mut prng = <ChaCha8Rng as SeedableRng>::from_seed(seed);

        let a1 = Matrix::<DB_C, N, DB_CN>::random(&mut prng, LOGQ);
        let a2 = Matrix::<DB_R, N, DB_RN>::random(&mut prng, LOGQ);

        let db_a1_transposed = db
            .mul::<N, DB_CN, DB_RN>(&a1)
            .transpose()
            .expand::<NBD, NBD_DBR>(P);
        let hint_c = db_a1_transposed.mul(&a2);

        Server {
            hint_s: seed,
            hint_c,
            db: db.clone(),
            db_a1_transposed,
            a2,
        }
    }

    pub fn ans<const DNPD: usize, const DR: usize, const DNPD_R: usize>(
        &self,
        c1: &Matrix<DB_C, 1, DB_C>,
        c2: &Matrix<DB_R, 1, DB_R>,
    ) -> (Matrix<DELTA, N, NBD>, Matrix<DNPD, 1, DNPD>) {
        let ans_1 = self
            .db
            .mul::<1, DB_C, DB_R>(c1)
            .transpose()
            .expand::<DELTA, DR>(P);
        let h = ans_1.mul(&self.a2);
        let ans_2 = self
            .db_a1_transposed
            .concat_matrix::<DELTA, DR, DNPD, DNPD_R>(&ans_1)
            .mul(c2);
        (h, ans_2)
    }
}

pub struct QueryState<const DB_C: usize, const DB_R: usize, const N: usize> {
    pub query: (Matrix<DB_C, 1, DB_C>, Matrix<DB_R, 1, DB_R>),
    s1: Matrix<N, 1, N>,
    s2: Matrix<N, 1, N>,
}

pub struct Client<
    const DB_R: usize,
    const DB_C: usize,
    const N: usize,
    const DELTA: usize,
    const NBD: usize,
    const DB_RC: usize,
    const NBD_N: usize,
    const NBD_DBR: usize,
    const DB_RN: usize,
    const DB_CN: usize,
    const LOGQ: usize,
    const P: u32,
> {
    hint_s: <ChaCha8Rng as SeedableRng>::Seed,
    a1: Matrix<DB_C, N, DB_CN>,
    a2: Matrix<DB_R, N, DB_RN>,
    hint_c: Matrix<NBD, N, NBD_N>,
}

impl<
        const DB_R: usize,
        const DB_C: usize,
        const N: usize,
        const DELTA: usize,
        const NBD: usize,
        const DB_RC: usize,
        const NBD_N: usize,
        const NBD_DBR: usize,
        const DB_RN: usize,
        const DB_CN: usize,
        const LOGQ: usize,
        const P: u32,
    > Client<DB_R, DB_C, N, DELTA, NBD, DB_RC, NBD_N, NBD_DBR, DB_RN, DB_CN, LOGQ, P>
{
    pub fn new(
        server: &Server<DB_R, DB_C, N, DELTA, NBD, DB_RC, NBD_N, NBD_DBR, DB_RN, DB_CN, LOGQ, P>,
    ) -> Client<DB_R, DB_C, N, DELTA, NBD, DB_RC, NBD_N, NBD_DBR, DB_RN, DB_CN, LOGQ, P> {
        let mut prng = <ChaCha8Rng as SeedableRng>::from_seed(server.hint_s);
        let a1 = Matrix::random(&mut prng, LOGQ);
        let a2 = Matrix::random(&mut prng, LOGQ);
        Client {
            hint_s: server.hint_s,
            a1,
            a2,
            hint_c: server.hint_c.clone(),
        }
    }

    pub fn query<A: CryptoRng + RngCore>(
        &self,
        rng: &mut A,
        index: usize,
    ) -> QueryState<DB_C, DB_R, N> {
        let row = index / DB_C;
        let col = index % DB_C;

        let s1 = Matrix::<N, 1, N>::random(rng, LOGQ);
        let s2 = Matrix::<N, 1, N>::random(rng, LOGQ);

        let e1 = Matrix::<DB_C, 1, DB_C>::gaussian_matrix(10, rng);
        let e2 = Matrix::<DB_R, 1, DB_R>::gaussian_matrix(10, rng);

        let mut u_col = Matrix::<DB_C, 1, DB_C>::zeros();
        u_col.set_at(col, 0, Self::delta());
        let mut u_row = Matrix::<DB_R, 1, DB_R>::zeros();
        u_row.set_at(row, 0, Self::delta());

        let c1 = self.a1.mul(&s1).add(&u_col).add(&e1);
        let c2 = self.a2.mul(&s2).add(&u_row).add(&e2);

        QueryState {
            query: (c1, c2),
            s1,
            s2,
        }
    }

    pub fn recover<const NP1: usize, const DNPD: usize, const DNPD_N: usize>(
        &self,
        query_state: &QueryState<DB_C, DB_R, N>,
        ans: (Matrix<DELTA, N, NBD>, Matrix<DNPD, 1, DNPD>),
    ) -> u32 {
        let hint_c = self
            .hint_c
            .concat_matrix::<DELTA, NBD, DNPD, DNPD_N>(&ans.0);
        let mut tmp = ans.1.sub(&hint_c.mul(&query_state.s2));

        // scale down
        tmp.get_data_mut()
            .iter_mut()
            .for_each(|(v)| *v = self.scale_down(*v));

        // recompose
        let tmp = tmp.recomp::<NP1, NP1>(Self::delta_df());
        let tmp_data = tmp.get_data();
        let s1_data = query_state.s1.get_data();
        let mut inner_product = 0u32;
        for i in 0..N {
            inner_product = inner_product.wrapping_add(tmp_data[i].wrapping_mul(s1_data[i]));
        }

        let d = tmp_data[N] - inner_product;

        self.scale_down(d)
    }

    pub const fn delta() -> u32 {
        ((1u64 << LOGQ) / P as u64) as u32
    }

    //TODO: this isn't correct
    pub const fn delta_df() -> usize {
        LOGQ - (P.next_power_of_two().leading_zeros() as usize) - 1
    }

    pub fn scale_down(&self, value: u32) -> u32 {
        let delta = Self::delta();
        (value + (delta / 2)) / delta
    }
}
