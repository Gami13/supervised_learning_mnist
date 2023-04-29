//SUPERVISIED LEARNING

mod activations;
mod imgs;
mod matrices;
mod neural_network;

fn main() {
    //TRAINING
    let number_imgs = 50000;
    println!("training");
    let imgs = imgs::Img::csv_to_imgs("./mnist_train.csv".to_string(), number_imgs);

    println!("images created");
    // let mut net = neural_network::NeuralNetwork::create(784, 300, 10, 0.0001);
    let mut net = neural_network::NeuralNetwork::load("testing_net");
    println!("net loaded");
    net.train_batch_imgs(imgs, number_imgs);
    println!("net trained");
    net.save("testing_net");

    // PREDICTING
    println!("predicting");
    let n = 500;
    let imgs = imgs::Img::csv_to_imgs("./mnist_train.csv".to_string(), n);
    let mut net = neural_network::NeuralNetwork::load("testing_net");

    let score = net.predict_imgs(imgs, n);
    println!("Score: {}", score);
}
