use crate::activations;
use crate::imgs;
use crate::matrices;
use std::fs::File;
use std::io::prelude::*;
pub struct NeuralNetwork {
    input: i32,
    hidden: i32,
    output: i32,
    learning_rate: f32,
    hidden_weights: matrices::Matrix,
    output_weights: matrices::Matrix,
}

impl NeuralNetwork {
    pub fn create(input: i32, hidden: i32, output: i32, learning_rate: f32) -> NeuralNetwork {
        let mut nn = NeuralNetwork {
            input: input,
            hidden: hidden,
            output: output,
            learning_rate: learning_rate,
            hidden_weights: matrices::Matrix::create(hidden as usize, input as usize),
            output_weights: matrices::Matrix::create(output as usize, hidden as usize),
        };
        nn.hidden_weights.randomize(hidden);
        nn.output_weights.randomize(output);
        return nn;
    }
    pub fn train(&mut self, input_data: matrices::Matrix, output_data: matrices::Matrix) {
        let hidden_inputs = matrices::Matrix::dot(&self.hidden_weights, &input_data);
        let hidden_outputs = hidden_inputs.apply(activations::sigmoid);
        let final_inputs = matrices::Matrix::dot(&self.output_weights, &hidden_outputs);
        let final_outputs = final_inputs.apply(activations::sigmoid);

        let output_errors = matrices::Matrix::subtract(&output_data, &final_outputs);
        let transposed_mat = self.output_weights.transpose();
        let hidden_errors = matrices::Matrix::dot(&transposed_mat, &output_errors);
        //  back propagation
        // self.output_weights = matrices::Matrix::add(
        //     &self.output_weights,
        //     &matrices::Matrix::dot(
        //         &activations::sigmoid_prime(&final_outputs),
        //         &(matrices::Matrix::multiply(
        //             &output_errors,
        //             &activations::sigmoid_prime(&final_outputs),
        //         )),
        //     )
        //     .scale(self.learning_rate),
        // );
        let sigmoid_primed_mat = activations::sigmoid_prime(&final_outputs);
        let multiplied_mat = matrices::Matrix::multiply(&output_errors, &sigmoid_primed_mat);
        let transposed_mat = hidden_outputs.transpose();

        let dot_mat = matrices::Matrix::dot(&multiplied_mat, &transposed_mat);
        let scaled_mat = dot_mat.scale(self.learning_rate);
        let added_mat = matrices::Matrix::add(&self.output_weights, &scaled_mat);

        self.output_weights = added_mat;

        let sigmoid_primed_mat = activations::sigmoid_prime(&hidden_outputs);
        let multiplied_mat = matrices::Matrix::multiply(&hidden_errors, &sigmoid_primed_mat);
        let transposed_mat = input_data.transpose();
        let dot_mat = matrices::Matrix::dot(&multiplied_mat, &transposed_mat);
        let scaled_mat = dot_mat.scale(self.learning_rate);
        let added_mat = matrices::Matrix::add(&self.hidden_weights, &scaled_mat);
        self.hidden_weights = added_mat;
    }
    pub fn predict(&self, input_data: matrices::Matrix) -> matrices::Matrix {
        let hidden_inputs = matrices::Matrix::dot(&self.hidden_weights, &input_data);
        let hidden_outputs = hidden_inputs.apply(activations::sigmoid);
        let final_inputs = matrices::Matrix::dot(&self.output_weights, &hidden_outputs);
        let final_outputs = final_inputs.apply(activations::sigmoid);
        let result = activations::softmax(final_outputs);
        return result;
    }
    pub fn save(&self, path: &str) {
        let mut file = File::create(path).unwrap();
        file.write_all((self.input.to_string() + "\n").to_string().as_bytes())
            .unwrap();
        file.write_all((self.hidden.to_string() + "\n").to_string().as_bytes())
            .unwrap();
        file.write_all((self.output.to_string() + "\n").to_string().as_bytes())
            .unwrap();
        file.write_all((self.learning_rate.to_string() + "\n").as_bytes())
            .unwrap();

        matrices::Matrix::save(&self.hidden_weights, "hidden".to_string());
        matrices::Matrix::save(&self.output_weights, "output".to_string());
    }
    pub fn load(path: &str) -> NeuralNetwork {
        let mut file = File::open(path).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        let mut lines = contents.lines();
        let input = lines.next().unwrap().parse::<i32>().unwrap();
        let hidden = lines.next().unwrap().parse::<i32>().unwrap();
        let output = lines.next().unwrap().parse::<i32>().unwrap();
        let learning_rate = lines.next().unwrap().parse::<f32>().unwrap();

        let mut nn = NeuralNetwork::create(input, hidden, output, learning_rate);
        nn.hidden_weights.load("hidden".to_string());
        nn.output_weights.load("output".to_string());
        return nn;
    }
    pub fn print(&self) {
        println!("input: {}", self.input);
        println!("hidden: {}", self.hidden);
        println!("output: {}", self.output);
        println!("learning rate: {}", self.learning_rate);
        println!("hidden weights: ");
        self.hidden_weights.print();
        println!("output weights: ");
        self.output_weights.print();
    }
    pub fn train_batch_imgs(&mut self, imgs: Vec<imgs::Img>, batch_size: i32) {
        for i in 0..batch_size {
            if i % 100 == 0 {
                println!("{}", i);
            }
            let cur_img = imgs.get(i as usize).unwrap();
            let img_data = cur_img.data.flatten(0);
            let mut output = matrices::Matrix::create(10, 1);
            output.data[cur_img.label as usize][0] = 1.0;
            self.train(img_data, output);
        }
    }
    pub fn predict_img(&self, img: &imgs::Img) -> matrices::Matrix {
        let img_data = img.data.flatten(0);
        let result = self.predict(img_data);
        return result;
    }
    pub fn predict_imgs(&self, imgs: Vec<imgs::Img>, n: i32) -> f32 {
        let mut n_correct = 0;
        for i in 0..n {
            if i % 100 == 0 {
                println!("{}", i);
            }
            let img = imgs.get(i as usize).unwrap();
            let prediction = self.predict_img(img);
            img.print();
            println!("prediction: {}", prediction.argmax());

            if (prediction.argmax() == imgs.get(i as usize).unwrap().label) {
                n_correct += 1;
            }
        }
        return n_correct as f32 / n as f32;
    }
}
