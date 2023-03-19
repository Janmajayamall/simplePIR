use itertools::izip;
use rand::{distributions::Uniform, CryptoRng, Rng, RngCore};

use crate::utils::sample_vec_cbd;

// Hard coded for compiler optimization
const COMPRESSION: usize = 3;
const BASIS: usize = 10;
const BASIS2: usize = BASIS * 2;
const MASK: u32 = (1 << BASIS) - 1;

/// 5. Get (row , col)
/// 6. Set (row , col)
/// 7. MatrixMulVec
/// 8. MatrixMulVecPacked
#[derive(PartialEq, Debug)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<u32>,
}

impl Matrix {
    fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0u32; rows * cols],
        }
    }

    fn random<R: RngCore + CryptoRng>(
        rows: usize,
        cols: usize,
        modp: usize,
        rng: &mut R,
    ) -> Matrix {
        let p = ((1u64 << modp) - 1) as u32;
        let mut out = Matrix::zeros(rows, cols);
        let distr = Uniform::new(0, p);
        out.data.iter_mut().for_each(|v| {
            *v = rng.sample(distr);
        });
        out
    }

    fn gaussian<R: RngCore + CryptoRng>(
        rows: usize,
        cols: usize,
        variance: usize,
        rng: &mut R,
    ) -> Matrix {
        let values = sample_vec_cbd(rows * cols, variance, rng)
            .expect("Sampling from gaussian distribution should work")
            .iter()
            .map(|v| *v as u32)
            .collect::<Vec<u32>>();
        let mut out = Matrix::zeros(rows, cols);
        out.data.copy_from_slice(values.as_slice());
        out
    }

    fn squish(&self, basis: usize, delta: usize) -> Matrix {
        // validate squish params
        assert!(delta * basis <= 32);
        assert!(delta == 3);
        assert!(basis == 10);

        dbg!((self.cols + delta - 1) / delta);
        let mut out = Matrix::zeros(self.rows, (self.cols + delta - 1) / delta);

        for i in 0..out.rows {
            for j in 0..out.cols {
                for k in 0..delta {
                    if delta * j + k < self.cols {
                        // dbg!((self.data[(i * self.cols) + (delta * j + k)] << (basis * k)));
                        out.data[i * out.cols + j] +=
                            (self.data[(i * self.cols) + (delta * j + k)] << (basis * k));
                    }
                }
            }
        }

        out
    }

    fn unquish(&self, basis: usize, delta: usize, cols: usize) -> Matrix {
        assert!(delta * basis <= 32);

        let mut out = Matrix::zeros(self.rows, cols);
        let mask = (1u32 << basis) - 1;

        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = self.data[i * self.cols + j];
                for k in 0..delta {
                    if delta * j + k < cols {
                        out.data[i * cols + delta * j + k] = (value >> (k * basis)) & mask;
                    }
                }
            }
        }

        out
    }

    fn concat_rows(&mut self, rows: &[u32]) {
        assert!(rows.len() % self.cols == 0);
        self.rows += rows.len() / self.cols;
        self.data.extend(rows.iter());
    }

    fn drop_rows(&mut self, rows: usize) {
        self.data.truncate((self.rows - rows) * self.cols);
        self.rows -= rows;
    }

    ///TODO: make inplace version
    pub fn add(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let mut out = Matrix::zeros(self.rows, self.cols);

        izip!(out.data.iter_mut(), self.data.iter(), rhs.data.iter()).for_each(|(o, l, r)| {
            *o = *l + *r;
        });
        out
    }

    ///TODO: make inplace version
    pub fn sub(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let mut out = Matrix::zeros(self.rows, self.cols);

        izip!(out.data.iter_mut(), self.data.iter(), rhs.data.iter()).for_each(|(o, l, r)| {
            *o = *l - *r;
        });
        out
    }

    pub fn matrix_mul_vec(&self, v: &Matrix) -> Matrix {
        assert_eq!(v.rows, self.cols);
        let mut out = Matrix::zeros(self.rows, 1);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[i] =
                    out.data[i].wrapping_add(self.data[i * self.cols + j].wrapping_mul(v.data[j]));
            }
        }
        out
    }

    //TODO: make this fast
    pub fn matrix_mul_vec_packed(&self, v: &Matrix, basis: usize, delta: usize) -> Matrix {
        assert_eq!(self.cols * delta, v.rows);
        assert_eq!(v.cols, 1);
        assert_eq!(basis, 10);
        assert_eq!(delta, 3);

        let mut out = Matrix::zeros(self.rows + 8, 1);

        let o: &mut [u32] = &mut out.data;
        let a: &[u32] = self.data.as_ref();
        let b: &[u32] = v.data.as_ref();

        let (mut db, mut db2, mut db3, mut db4, mut db5, mut db6, mut db7, mut db8);
        let (mut val, mut val2, mut val3, mut val4, mut val5, mut val6, mut val7, mut val8);
        let (mut tmp, mut tmp2, mut tmp3, mut tmp4, mut tmp5, mut tmp6, mut tmp7, mut tmp8);
        let mut index = 0;

        for i in (0..self.rows).step_by(8) {
            tmp = 0u32;
            tmp2 = 0u32;
            tmp3 = 0u32;
            tmp4 = 0u32;
            tmp5 = 0u32;
            tmp6 = 0u32;
            tmp7 = 0u32;
            tmp8 = 0u32;

            let mut index2 = 0;
            for j in (0..self.cols) {
                db = a[index];
                db2 = a[index + 1 * self.cols];
                db3 = a[index + 2 * self.cols];
                db4 = a[index + 3 * self.cols];
                db5 = a[index + 4 * self.cols];
                db6 = a[index + 5 * self.cols];
                db7 = a[index + 6 * self.cols];
                db8 = a[index + 7 * self.cols];

                val = db & MASK;
                val2 = db2 & MASK;
                val3 = db3 & MASK;
                val4 = db4 & MASK;
                val5 = db5 & MASK;
                val6 = db6 & MASK;
                val7 = db7 & MASK;
                val8 = db8 & MASK;
                tmp = tmp.wrapping_add(val.wrapping_mul(b[index2]));
                tmp2 = tmp2.wrapping_add(val2.wrapping_mul(b[index2]));
                tmp3 = tmp3.wrapping_add(val3.wrapping_mul(b[index2]));
                tmp4 = tmp4.wrapping_add(val4.wrapping_mul(b[index2]));
                tmp5 = tmp5.wrapping_add(val5.wrapping_mul(b[index2]));
                tmp6 = tmp6.wrapping_add(val6.wrapping_mul(b[index2]));
                tmp7 = tmp7.wrapping_add(val7.wrapping_mul(b[index2]));
                tmp8 = tmp8.wrapping_add(val8.wrapping_mul(b[index2]));
                index2 += 1;

                val = (db >> BASIS) & MASK;
                val2 = (db2 >> BASIS) & MASK;
                val3 = (db3 >> BASIS) & MASK;
                val4 = (db4 >> BASIS) & MASK;
                val5 = (db5 >> BASIS) & MASK;
                val6 = (db6 >> BASIS) & MASK;
                val7 = (db7 >> BASIS) & MASK;
                val8 = (db8 >> BASIS) & MASK;
                tmp = tmp.wrapping_add(val.wrapping_mul(b[index2]));
                tmp2 = tmp2.wrapping_add(val2.wrapping_mul(b[index2]));
                tmp3 = tmp3.wrapping_add(val3.wrapping_mul(b[index2]));
                tmp4 = tmp4.wrapping_add(val4.wrapping_mul(b[index2]));
                tmp5 = tmp5.wrapping_add(val5.wrapping_mul(b[index2]));
                tmp6 = tmp6.wrapping_add(val6.wrapping_mul(b[index2]));
                tmp7 = tmp7.wrapping_add(val7.wrapping_mul(b[index2]));
                tmp8 = tmp8.wrapping_add(val8.wrapping_mul(b[index2]));
                index2 += 1;

                val = (db >> BASIS2) & MASK;
                val2 = (db2 >> BASIS2) & MASK;
                val3 = (db3 >> BASIS2) & MASK;
                val4 = (db4 >> BASIS2) & MASK;
                val5 = (db5 >> BASIS2) & MASK;
                val6 = (db6 >> BASIS2) & MASK;
                val7 = (db7 >> BASIS2) & MASK;
                val8 = (db8 >> BASIS2) & MASK;
                tmp = tmp.wrapping_add(val.wrapping_mul(b[index2]));
                tmp2 = tmp2.wrapping_add(val2.wrapping_mul(b[index2]));
                tmp3 = tmp3.wrapping_add(val3.wrapping_mul(b[index2]));
                tmp4 = tmp4.wrapping_add(val4.wrapping_mul(b[index2]));
                tmp5 = tmp5.wrapping_add(val5.wrapping_mul(b[index2]));
                tmp6 = tmp6.wrapping_add(val6.wrapping_mul(b[index2]));
                tmp7 = tmp7.wrapping_add(val7.wrapping_mul(b[index2]));
                tmp8 = tmp8.wrapping_add(val8.wrapping_mul(b[index2]));
                index2 += 1;
                index += 1;
            }
            index += self.cols * 7;
            o[i] = tmp;
            o[i + 1] = tmp2;
            o[i + 2] = tmp3;
            o[i + 3] = tmp4;
            o[i + 4] = tmp5;
            o[i + 5] = tmp6;
            o[i + 6] = tmp7;
            o[i + 7] = tmp8;
        }

        out.drop_rows(8);
        out
    }

    fn print_dims(&self) {
        println!("{} x {}: {}", self.rows, self.cols, self.data.len());
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;

    #[test]
    fn squish_unsquish() {
        let mut rng = thread_rng();
        let a = Matrix::random(1024, 1024, 8, &mut rng);
        let a_squished = a.squish(10, 3);
        let a_back = a_squished.unquish(10, 3, 1024);
        assert_eq!(a, a_back);
    }

    #[test]
    fn test1() {
        let mut rng = thread_rng();
        let a = Matrix::random(1024, 1024, 8, &mut rng);
        let mut b = Matrix::random(1024, 1, 32, &mut rng);

        let c = a.matrix_mul_vec(&b);

        b.concat_rows(&[0, 0]);

        let a_squished = a.squish(10, 3);
        let c2 = a_squished.matrix_mul_vec_packed(&b, 10, 3);

        assert_eq!(c, c2);
    }
}
