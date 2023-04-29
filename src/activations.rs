use crate::matrices;
pub fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + (-1.0 * x).exp());
}

pub fn sigmoid_prime(m: &matrices::Matrix) -> matrices::Matrix {
    let mut ones = matrices::Matrix::create(m.rows, m.cols);
    ones.fill(1.0);
    let subtracted = matrices::Matrix::subtract(&ones, m);
    let multiplied = matrices::Matrix::multiply(m, &subtracted);
    return multiplied;
}
pub fn softmax(m: matrices::Matrix) -> matrices::Matrix {
    let mut total = 0.0;
    for i in 0..m.rows {
        for j in 0..m.cols {
            total += (m.data[i][j]).exp();
        }
    }
    let mut mat = matrices::Matrix::create(m.rows, m.cols);
    for i in 0..mat.rows {
        for j in 0..mat.cols {
            mat.data[i][j] = (m.data[i][j]).exp() / total;
        }
    }
    return mat;
}
