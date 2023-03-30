use itertools::izip;
use rand::{distributions::Uniform, CryptoRng, Rng, RngCore};

use crate::{
    database::Params,
    utils::{reconstruct_val_from_basep, sample_vec_cbd},
};

// Hard coded for compiler optimization
const COMPRESSION: usize = 3;
const BASIS: usize = 10;
const BASIS2: usize = BASIS * 2;
const MASK: u32 = (1 << BASIS) - 1;

#[derive(PartialEq, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0u32; rows * cols],
        }
    }

    pub fn random<R: RngCore + CryptoRng>(rows: usize, cols: usize, p: u64, rng: &mut R) -> Matrix {
        let mut out = Matrix::zeros(rows, cols);
        let distr = Uniform::new_inclusive(0, (p - 1) as u32);
        out.data.copy_from_slice(
            rng.sample_iter(distr)
                .take(rows * cols)
                .collect::<Vec<u32>>()
                .as_slice(),
        );
        out
    }

    pub fn gaussian<R: RngCore + CryptoRng>(
        rows: usize,
        cols: usize,
        variance: usize,
        rng: &mut R,
    ) -> Matrix {
        let values = sample_vec_cbd(rows * cols, variance, rng)
            .expect("Sampling from gaussian distribution should work")
            .iter()
            // FIXME
            .map(|v| 0)
            .collect::<Vec<u32>>();
        let mut out = Matrix::zeros(rows, cols);
        out.data.copy_from_slice(values.as_slice());
        out
    }

    pub fn squish(&self, basis: usize, delta: usize) -> Matrix {
        // validate squish params
        assert!(delta * basis <= 32);
        assert!(delta == 3);
        assert!(basis == 10);

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

    pub fn unsquish(&self, basis: usize, delta: usize, cols: usize) -> Matrix {
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

    pub fn scale_down(&mut self, delta: u64) {
        self.data
            .iter_mut()
            .for_each(|v| *v = ((*v as u64 + delta / 2) / delta) as u32)
    }

    pub fn select_rows(&self, offset: usize, num_rows: usize) -> Matrix {
        assert!(offset <= self.rows);

        if offset == self.rows {
            return Matrix {
                rows: self.rows,
                cols: self.cols,
                data: self.data.clone(),
            };
        }

        if offset + num_rows <= self.rows {
            let mut out = Matrix::zeros(num_rows, self.cols);
            out.data
                .copy_from_slice(&self.data[offset * self.cols..(offset + num_rows) * self.cols]);
            return out;
        }

        let mut out = Matrix::zeros((self.rows - offset) * self.cols, self.cols);
        out.data
            .copy_from_slice(&self.data[offset * self.cols..(offset + num_rows) * self.cols]);
        out
    }

    pub fn concat_matrix(&mut self, other: &Matrix) {
        assert!(self.cols == other.cols);
        self.data.extend_from_slice(&other.data);
        self.rows += other.rows;
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

    pub fn set_at(&mut self, row: usize, col: usize, value: u32) {
        self.data[self.cols * row + col] = value;
    }

    ///TODO: make inplace version
    pub fn add(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let mut out = Matrix::zeros(self.rows, self.cols);

        izip!(out.data.iter_mut(), self.data.iter(), rhs.data.iter()).for_each(|(o, l, r)| {
            *o = l.wrapping_add(*r);
        });
        out
    }

    ///TODO: make inplace version
    pub fn sub(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let mut out = Matrix::zeros(self.rows, self.cols);

        izip!(out.data.iter_mut(), self.data.iter(), rhs.data.iter()).for_each(|(o, l, r)| {
            *o = l.wrapping_sub(*r);
        });
        out
    }

    pub fn mul(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.cols, rhs.rows);
        let mut out = Matrix::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                for k in 0..rhs.cols {
                    out.data[(i * rhs.cols) + k] = out.data[(i * rhs.cols) + k].wrapping_add(
                        self.data[(i * self.cols) + j].wrapping_mul(rhs.data[(j * rhs.cols) + k]),
                    );
                }
            }
        }
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
    pub fn _matrix_mul_vec_packed(
        o: &mut [u32],
        a: &[u32],
        b: &[u32],
        a_rows: usize,
        a_cols: usize,
        basis: usize,
        delta: usize,
    ) {
        // assert_eq!(self.cols * delta, v.rows);
        // assert_eq!(v.cols, 1);
        assert_eq!(basis, 10);
        assert_eq!(delta, 3);

        let (mut db, mut db2, mut db3, mut db4, mut db5, mut db6, mut db7, mut db8);
        let (mut val, mut val2, mut val3, mut val4, mut val5, mut val6, mut val7, mut val8);
        let (mut tmp, mut tmp2, mut tmp3, mut tmp4, mut tmp5, mut tmp6, mut tmp7, mut tmp8);
        let mut index = 0;

        for i in (0..a_rows).step_by(8) {
            tmp = 0u32;
            tmp2 = 0u32;
            tmp3 = 0u32;
            tmp4 = 0u32;
            tmp5 = 0u32;
            tmp6 = 0u32;
            tmp7 = 0u32;
            tmp8 = 0u32;

            let mut index2 = 0;
            for j in (0..a_cols) {
                db = a[index];
                db2 = a[index + a_cols];
                db3 = a[index + 2 * a_cols];
                db4 = a[index + 3 * a_cols];
                db5 = a[index + 4 * a_cols];
                db6 = a[index + 5 * a_cols];
                db7 = a[index + 6 * a_cols];
                db8 = a[index + 7 * a_cols];

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
            index += a_cols * 7;
            o[i] = tmp;
            o[i + 1] = tmp2;
            o[i + 2] = tmp3;
            o[i + 3] = tmp4;
            o[i + 4] = tmp5;
            o[i + 5] = tmp6;
            o[i + 6] = tmp7;
            o[i + 7] = tmp8;
        }
    }

    //TODO: make this fast
    pub fn matrix_mul_vec_packed(&self, v: &Matrix, basis: usize, delta: usize) -> Matrix {
        assert_eq!(self.cols * delta, v.rows);
        assert_eq!(v.cols, 1);
        assert_eq!(basis, 10);
        assert_eq!(delta, 3);

        let mut out = Matrix::zeros(self.rows + 8, 1);

        let a_rows_floor = (self.rows / 8) * 8;
        Matrix::_matrix_mul_vec_packed(
            &mut out.data,
            &self.data,
            &v.data,
            a_rows_floor,
            self.cols,
            basis,
            delta,
        );

        if a_rows_floor < self.rows {
            let mut eight_rows = vec![0u32; 8 * self.cols];
            eight_rows[..((self.rows - a_rows_floor) * self.cols)]
                .copy_from_slice(&self.data[a_rows_floor * self.cols..]);
            Matrix::_matrix_mul_vec_packed(
                &mut out.data[a_rows_floor..],
                &eight_rows,
                &v.data,
                8,
                self.cols,
                basis,
                delta,
            );
        }

        out.drop_rows(8);
        out
    }

    pub fn matrix_mul_transposed_packed(&self, rhs: &Matrix, basis: usize, delta: usize) -> Matrix {
        assert!(self.cols * delta == rhs.cols);
        assert_eq!(basis, 10);
        assert_eq!(delta, 3);

        let mut out = Matrix::zeros(self.rows, rhs.rows);

        let o: &mut [u32] = &mut out.data;
        let a: &[u32] = self.data.as_ref();
        let b: &[u32] = rhs.data.as_ref();

        let mut index1 = 0;
        if self.rows > self.cols {
            for i in 0..self.rows {
                for k in 0..self.cols {
                    let db = a[index1];
                    let val = db & MASK;
                    let val2 = (db >> BASIS) & MASK;
                    let val3 = (db >> BASIS2) & MASK;
                    for j in 0..rhs.rows {
                        o[i * rhs.rows + j] = o[i * rhs.rows + j]
                            .wrapping_add(val.wrapping_mul(b[k * COMPRESSION + rhs.cols * j]));
                        o[i * rhs.rows + j] = o[i * rhs.rows + j]
                            .wrapping_add(val2.wrapping_mul(b[k * COMPRESSION + rhs.cols * j + 1]));
                        o[i * rhs.rows + j] = o[i * rhs.rows + j]
                            .wrapping_add(val3.wrapping_mul(b[k * COMPRESSION + rhs.cols * j + 2]));
                    }
                    index1 += 1;
                }
            }
        } else {
            let (mut tmp, mut tmp2, mut tmp3, mut tmp4, mut tmp5, mut tmp6, mut tmp7, mut tmp8);
            for j in (0..rhs.rows).step_by(8) {
                let mut index1 = 0;
                for i in 0..self.rows {
                    let mut index2 = 0;
                    tmp = 0u32;
                    tmp2 = 0u32;
                    tmp3 = 0u32;
                    tmp4 = 0u32;
                    tmp5 = 0u32;
                    tmp6 = 0u32;
                    tmp7 = 0u32;
                    tmp8 = 0u32;
                    for k in 0..self.cols {
                        let db = a[index1];
                        for m in 0..COMPRESSION {
                            let val = (db >> (m * BASIS)) & MASK;
                            tmp = tmp.wrapping_add(val.wrapping_mul(b[index2 + j * rhs.cols]));
                            tmp2 =
                                tmp2.wrapping_add(val.wrapping_mul(b[index2 + (j + 1) * rhs.cols]));
                            tmp3 =
                                tmp3.wrapping_add(val.wrapping_mul(b[index2 + (j + 2) * rhs.cols]));
                            tmp4 =
                                tmp4.wrapping_add(val.wrapping_mul(b[index2 + (j + 3) * rhs.cols]));
                            tmp5 =
                                tmp5.wrapping_add(val.wrapping_mul(b[index2 + (j + 4) * rhs.cols]));
                            tmp6 =
                                tmp6.wrapping_add(val.wrapping_mul(b[index2 + (j + 5) * rhs.cols]));
                            tmp7 =
                                tmp7.wrapping_add(val.wrapping_mul(b[index2 + (j + 6) * rhs.cols]));
                            tmp8 =
                                tmp8.wrapping_add(val.wrapping_mul(b[index2 + (j + 7) * rhs.cols]));
                            index2 += 1;
                        }
                        index1 += 1;
                    }
                    o[rhs.rows * i + j] = tmp;
                    o[rhs.rows * i + j + 1] = tmp2;
                    o[rhs.rows * i + j + 2] = tmp3;
                    o[rhs.rows * i + j + 3] = tmp4;
                    o[rhs.rows * i + j + 4] = tmp5;
                    o[rhs.rows * i + j + 5] = tmp6;
                    o[rhs.rows * i + j + 6] = tmp7;
                    o[rhs.rows * i + j + 7] = tmp8;
                }
            }
        }

        out
    }

    pub fn transpose_and_expand_and_concat_cols_and_squish(
        &self,
        modp: u32,
        delta: usize,
        concat: usize,
        basis: usize,
        d: usize,
    ) -> Matrix {
        let mut out = Matrix::zeros(self.cols * concat * delta, (self.rows / concat + d - 1) / d);
        let o: &mut [u32] = &mut out.data;
        for j in 0..self.rows {
            for i in 0..self.cols {
                let mut val = self.data[i + self.cols * j];
                for k in 0..delta {
                    let v = val % modp;
                    let r = i * delta + k + self.cols * delta * (j % concat);
                    let c = j / concat;
                    o[r * out.cols + c / d] = v << (basis * (c % d));
                    val /= modp;
                }
            }
        }

        out
    }

    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[i + j * self.rows] = self.data[j + i * self.cols];
            }
        }

        out
    }

    pub fn expand(&self, delta: usize, p: u32) -> Matrix {
        let mut out = Matrix::zeros(self.rows * delta, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                let mut val = self.data[i * self.cols + j];
                for k in 0..delta {
                    out.data[(i * delta + k) * self.cols + j] = val % p;
                    val /= p;
                }
            }
        }

        out
    }

    pub fn contract(&self, delta: usize, p: u32) -> Matrix {
        let mut out = Matrix::zeros(self.rows / delta, self.cols);
        for i in 0..out.rows {
            for j in 0..out.cols {
                let mut parts = vec![];
                for k in 0..delta {
                    parts.push(self.data[(i * delta + k) * self.cols + j] as u64);
                }
                out.data[i * out.cols + j] += reconstruct_val_from_basep(p as u64, &parts);
            }
        }
        out
    }

    pub fn concat_cols(&self, concat: usize) -> Matrix {
        let mut out = Matrix::zeros(self.rows * concat, self.cols / concat);

        for i in 0..self.rows {
            for j in 0..self.cols {
                let r = i + self.rows * (j % concat);
                let c = j / concat;
                out.data[r * self.cols / concat + c] = self.data[i * self.cols + j];
            }
        }

        out
    }

    pub fn print_dims(&self) {
        println!("{} x {}: {}", self.rows, self.cols, self.data.len());
    }

    pub fn get(&self, row: usize, col: usize) -> u32 {
        self.data[row * self.cols + col]
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
        let a_back = a_squished.unsquish(10, 3, 1024);
        assert_eq!(a, a_back);
    }

    #[test]
    fn test_matrix_mul_vec_packed() {
        let mut rng = thread_rng();
        let a = Matrix::random(1, 65536, 8, &mut rng);
        let mut b = Matrix::random(65536, 1, 32, &mut rng);

        let c = a.matrix_mul_vec(&b);

        b.concat_rows(&[0, 0]);

        let a_squished = a.squish(10, 3);
        let c2 = a_squished.matrix_mul_vec_packed(&b, 10, 3);

        assert_eq!(c, c2);
    }

    #[test]
    fn test_matrix_mul_transposed() {
        let mut rng = thread_rng();
        let a = Matrix::random(16, 129, 8, &mut rng);

        let a_squished = a.squish(10, 3);
        let mut b = Matrix::random(1024, 129, 32, &mut rng);
        let c2 = a_squished.matrix_mul_transposed_packed(&b, 10, 3);
    }

    #[test]
    fn transpose_and_expand_and_concat_cols_and_squish() {
        let mut rng = thread_rng();
        let a = Matrix::random(1024, 1024, 8, &mut rng);
        let a2 = a.transpose_and_expand_and_concat_cols_and_squish(256, 4, 3, 10, 3);
        a2.print_dims();
    }

    // #[test]
    // fn trial() {
    //     let f = vec![0,1,2,3,4];
    //     f.iter()
    // }
}
