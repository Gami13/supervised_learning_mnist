use crate::matrices::Matrix;
use std::fs::File;
use std::io::prelude::*;

pub struct Img {
    pub data: Matrix,
    pub label: i32,
}
impl Img {
    pub fn csv_to_imgs(path: String, n: i32) -> Vec<Img> {
        let mut file = File::open(path).unwrap();
        let mut imgs = Vec::new();

        let mut line = String::new();
        file.read_to_string(&mut line).unwrap();

        for i in 0..n {
            let tokens = line.lines().nth((i + 1) as usize).unwrap().split(",");
            let temp = Img {
                data: Matrix::create(28, 28),
                label: 0,
            };
            imgs.push(temp);
            if i % 1000 == 0 {
                println!("images: {}/{}", i, n);
            }
            let mut j = 0;
            for token in tokens {
                if j == 0 {
                    imgs[i as usize].label = token.parse::<i32>().unwrap();
                } else {
                    imgs[i as usize].data.data[(j - 1) / 28][(j - 1) % 28] =
                        token.parse::<f32>().unwrap();
                }
                j += 1;
            }
        }

        return imgs;
    }
    pub fn print(&self) {
        self.data.print();
        println!("label: {}", self.label);
    }
}
