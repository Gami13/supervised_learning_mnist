use core::panic;
use rand::prelude::*;
use std::fs::File;
use std::io::prelude::*;

pub fn uniform_distribution(low: f32, high: f32) -> f32 {
    let difference = high - low;
    let scale: f32 = 10000.0;
    let scaled_difference = (difference * scale) as i32;
    let rand: i32 = rand::random();
    let res: f32 = low * (1.0 * (rand % scaled_difference) as f32 / scale);
    return res as f32;
}
pub fn check_dimensions(a: &Matrix, b: &Matrix) -> bool {
    return a.rows == b.rows && a.cols == b.cols;
}
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f32>>,
}
impl Matrix {
    pub fn create(r: usize, c: usize) -> Matrix {
        return Matrix {
            rows: r,
            cols: c,
            data: vec![vec![0.0; c]; r],
        };
    }

    pub fn fill(&mut self, value: f32) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] = value;
            }
        }
    }

    pub fn print(&self) {
        for i in 0..self.cols {
            for j in 0..self.rows {
                if self.data[j][i] == 0.0 {
                    print!(" ");
                } else {
                    print!("◼");
                }
            }
            println!();
        }
    }
    pub fn copy(&self) -> Matrix {
        let mut matrix = Matrix::create(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                matrix.data[i][j] = self.data[i][j];
            }
        }
        return matrix;
    }
    pub fn save(&self, filename: String) {
        let mut file = File::create(filename).unwrap();
        for i in 0..self.rows {
            for j in 0..self.cols {
                file.write_all(self.data[i][j].to_string().as_bytes())
                    .unwrap();
                file.write_all(" ".as_bytes()).unwrap();
            }
            file.write_all("\n".as_bytes()).unwrap();
        }
    }
    pub fn load(&mut self, filename: String) {
        let mut file = File::open(filename).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        let mut i = 0;
        for line in contents.lines() {
            let mut j = 0;

            for number in line.split_whitespace() {
                self.data[i][j] = number.parse::<f32>().unwrap();
                j += 1;
            }
            i += 1;
        }
    }
    pub fn randomize(&mut self, n: i32) {
        let min: f32 = -1.0 / f32::sqrt(n as f32);
        let max: f32 = 1.0 / f32::sqrt(n as f32);
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] = uniform_distribution(min, max);
            }
        }
    }
    pub fn argmax(&self) -> i32 {
        let mut max_score = 0.0;
        let mut max_index = 0;
        for i in 0..self.rows {
            if self.data[i][0] > max_score {
                {
                    max_score = self.data[i][0];
                    max_index = i;
                }
            }
        }
        return max_index as i32;
    }
    pub fn flatten(&self, axis: i16) -> Matrix {
        let mut matrix: Matrix = Matrix::create(0, 0);
        if axis == 0 {
            matrix = Matrix::create(self.rows * self.cols, 1);
        } else if axis == 1 {
            matrix = Matrix::create(1, self.rows * self.cols);
        } else {
            println!("axis must be 0 or 1");
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                if axis == 0 {
                    matrix.data[i * self.cols + j][0] = self.data[i][j];
                } else if axis == 1 {
                    matrix.data[0][i * self.cols + j] = self.data[i][j];
                }
            }
        }
        return matrix;
    }
    pub fn multiply(m1: &Matrix, m2: &Matrix) -> Matrix {
        if !check_dimensions(&m1, &m2) {
            println!("m1 and m2 must have the same dimensions");
            panic!("error")
        }
        let mut m = Matrix::create(m1.rows, m2.cols);
        for i in 0..m1.rows {
            for j in 0..m2.cols {
                m.data[i][j] = m1.data[i][j] * m2.data[i][j];
            }
        }
        return m;
    }
    pub fn add(m1: &Matrix, m2: &Matrix) -> Matrix {
        if !check_dimensions(&m1, &m2) {
            println!("m1 and m2 must have the same dimensions");
            panic!("error")
        }
        let mut m = Matrix::create(m1.rows, m2.cols);
        for i in 0..m.rows {
            for j in 0..m.cols {
                m.data[i][j] = m1.data[i][j] + m2.data[i][j];
            }
        }
        return m;
    }
    pub fn subtract(m1: &Matrix, m2: &Matrix) -> Matrix {
        if !check_dimensions(&m1, &m2) {
            println!("m1 and m2 must have the same dimensions");
            panic!("error")
        }
        let mut m = Matrix::create(m1.rows, m2.cols);
        for i in 0..m.rows {
            for j in 0..m.cols {
                m.data[i][j] = m1.data[i][j] - m2.data[i][j];
            }
        }
        return m;
    }

    pub fn apply(&self, f: fn(f32) -> f32) -> Matrix {
        let mut mat = self.copy();
        for i in 0..self.rows {
            for j in 0..self.cols {
                mat.data[i][j] = f(self.data[i][j]);
            }
        }
        return mat;
    }
    pub fn dot(m1: &Matrix, m2: &Matrix) -> Matrix {
        if m1.cols != m2.rows {
            panic!("m1.cols must be equal to m2.rows");
        }

        let mut m = Matrix::create(m1.rows, m2.cols);
        for i in 0..m1.rows {
            for j in 0..m2.cols {
                let mut sum: f32 = 0.0;
                for k in 0..m2.rows {
                    sum += m1.data[i][k] * m2.data[k][j];
                }
                m.data[i][j] = sum;
            }
        }
        return m;
    }
    pub fn scale(&self, n: f32) -> Matrix {
        let mut mat = self.copy();
        for i in 0..self.rows {
            for j in 0..self.cols {
                mat.data[i][j] *= n;
            }
        }
        return mat;
    }
    pub fn add_scalar(&self, n: f32) -> Matrix {
        let mut mat = self.copy();
        for i in 0..self.rows {
            for j in 0..self.cols {
                mat.data[i][j] += n;
            }
        }
        return mat;
    }
    pub fn transpose(&self) -> Matrix {
        let mut mat = Matrix::create(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                mat.data[j][i] = self.data[i][j];
            }
        }
        return mat;
    }
}
