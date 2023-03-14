use crate::matrix::Matrix;
use rand::{CryptoRng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

struct Server<
    const DB_DIM: usize,
    const N: usize,
    const LOGQ: usize,
    const DELTA: usize,
    const P: usize,
> where
    [(); N * N]:,
    [(); N * DELTA * N]:,
    [(); DB_DIM * DB_DIM]:,
    [(); DB_DIM * N]:,
    [(); N * DB_DIM]:,
    [(); N * DELTA * DB_DIM]:,
{
    hint_s: <ChaCha8Rng as SeedableRng>::Seed,
    hint_c: Matrix<{ N * DELTA }, N>,
    // D
    db: Matrix<DB_DIM, DB_DIM>,
    db_a1_transposed: Matrix<{ N * DELTA }, DB_DIM>,
    a2: Matrix<DB_DIM, N>,
}

impl<
        const DB_DIM: usize,
        const N: usize,
        const LOGQ: usize,
        const DELTA: usize,
        const P: usize,
    > Server<DB_DIM, N, LOGQ, DELTA, P>
where
    [(); N * DELTA * DB_DIM]:,
    [(); N * DELTA * N]:,
    [(); DB_DIM * N]:,
    [(); N * DB_DIM]:,
    [(); DB_DIM * DB_DIM]:,
    [(); DB_DIM * 1]:,
    [(); 1 * DB_DIM]:,
    [(); 1 * DELTA * DB_DIM]:,
    [(); N * N]:,
    [(); 1 * DELTA * N]:,
    [(); { 1 * DELTA } * N]:,
    [(); { (N * DELTA) + (1 * DELTA) } * DB_DIM]:,
    [(); { (N * DELTA) + (1 * DELTA) } * 1]:,
{
    fn setup<A: RngCore + CryptoRng>(
        db: Matrix<DB_DIM, DB_DIM>,
        rng: &mut A,
    ) -> Server<DB_DIM, N, LOGQ, DELTA, P> {
        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        rng.fill_bytes(&mut seed);

        let mut prng = <ChaCha8Rng as SeedableRng>::from_seed(seed);

        let a1 = Matrix::<DB_DIM, N>::random(&mut prng, LOGQ);
        let a2 = Matrix::<DB_DIM, N>::random(&mut prng, LOGQ);

        let db_a1_transposed = db.mul(&a1).transpose().expand::<DELTA>(P as u32);
        let hint_c = db_a1_transposed.mul(&a2);

        Server {
            hint_s: seed,
            hint_c,
            db,
            db_a1_transposed,
            a2,
        }
    }

    pub fn ans(
        &self,
        c1: Matrix<DB_DIM, 1>,
        c2: Matrix<DB_DIM, 1>,
    ) -> (
        Matrix<{ 1 * DELTA }, N>,
        Matrix<{ (N * DELTA) + (1 * DELTA) }, 1>,
    ) {
        let ans_1 = self.db.mul(&c1).transpose().expand::<DELTA>(P as u32);
        let h = ans_1.mul(&self.a2);
        let ans_2 = self.db_a1_transposed.concat_matrix(&ans_1).mul(&c2);
        (h, ans_2)
    }
}

struct QueryState<const N: usize, const DB_DIM: usize>
where
    [(); DB_DIM * 1]:,
    [(); N * 1]:,
{
    query: (Matrix<DB_DIM, 1>, Matrix<DB_DIM, 1>),
    s1: Matrix<N, 1>,
    s2: Matrix<N, 1>,
}

struct Client<
    const DB_DIM: usize,
    const N: usize,
    const LOGQ: usize,
    const DELTA: usize,
    const P: usize,
> where
    [(); DB_DIM * N]:,
    [(); N * DELTA * N]:,
{
    hint_s: <ChaCha8Rng as SeedableRng>::Seed,
    a1: Matrix<DB_DIM, N>,
    a2: Matrix<DB_DIM, N>,
    hint_c: Matrix<{ N * DELTA }, N>,
}

impl<
        const DB_DIM: usize,
        const N: usize,
        const LOGQ: usize,
        const DELTA: usize,
        const P: usize,
    > Client<DB_DIM, N, LOGQ, DELTA, P>
where
    [(); DB_DIM * N]:,
    [(); N * DELTA * N]:,
    [(); N * N]:,
    [(); N * DB_DIM]:,
    [(); DB_DIM * DB_DIM]:,
    [(); DB_DIM * 1]:,
    [(); N * 1]:,
    [(); N * DELTA * DB_DIM]:,
    [(); 1 * DELTA * N]:,
    [(); ((N * DELTA) + (1 * DELTA)) * 1]:,
{
    pub fn new(server: &Server<DB_DIM, N, LOGQ, DELTA, P>) -> Client<DB_DIM, N, LOGQ, DELTA, P> {
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
    ) -> QueryState<N, DB_DIM> {
        let row = index / DB_DIM;
        let col = index % DB_DIM;

        let s1 = Matrix::<N, 1>::random(rng, LOGQ);
        let s2 = Matrix::<N, 1>::random(rng, LOGQ);

        let e1 = Matrix::<DB_DIM, 1>::gaussian_matrix(10, rng);
        let e2 = Matrix::<DB_DIM, 1>::gaussian_matrix(10, rng);

        let mut u_col = Matrix::<DB_DIM, 1>::zeros();
        u_col.set_at(col, 0, self.delta());
        let mut u_row = Matrix::<DB_DIM, 1>::zeros();
        u_row.set_at(row, 0, self.delta());

        let c1 = self.a1.mul(&s1).add(&u_col).add(&e1);
        let c2 = self.a2.mul(&s2).add(&u_row).add(&e2);

        QueryState {
            query: (c1, c2),
            s1,
            s2,
        }
    }

    pub fn recover(
        &self,
        query_state: &QueryState<N, DB_DIM>,
        ans: (
            Matrix<{ 1 * DELTA }, N>,
            Matrix<{ (N * DELTA) + (1 * DELTA) }, 1>,
        ),
    ) -> u32 {
        let hint_c = self.hint_c.concat_matrix(&ans.0);
        let tmp = ans.1.sub(&hint_c.mul(&query_state.s2));

        // scale down
        tmp.get_data_mut()
            .iter_mut()
            .for_each(|(v)| *v = self.scale_down(v));

        // recompose
        let tmp = tmp.recomp::<DELTA>(256).get_data();
        let s1_data = query_state.s1.get_data();
        let mut inner_product = 0u32;
        for i in 0..N {
            inner_product = inner_product.wrapping_add(tmp[i].wrapping_mul(s1_data[i]));
        }
        let d = tmp[N] - inner_product;

        self.scale_down(d)
    }

    pub const fn delta(&self) -> u32 {
        ((1 << LOGQ) as u64 / (P as u64)) as u32
    }

    pub fn scale_down(&self, value: u32) -> u32 {
        let delta = self.delta();
        (value + (delta / 2)) / delta
    }
}
