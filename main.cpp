#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <print>
#include <stdexcept>
#include <vector>

#include <mlfo>

namespace {

// mnist idx files, see http://yann.lecun.com/exdb/mnist/
struct Mnist {
    std::size_t count;
    std::size_t image_size;
    // pixels are kept as bytes and converted to floats one sample at a time
    std::vector<std::uint8_t> images;
    std::vector<std::uint8_t> labels;
};

std::uint32_t read_u32(std::ifstream& file) {
    // idx headers are big endian
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (
        (std::uint32_t{bytes[0]} << 24) |
        (std::uint32_t{bytes[1]} << 16) |
        (std::uint32_t{bytes[2]} << 8) |
        std::uint32_t{bytes[3]}
    );
}

std::ifstream open(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file " + path.string());
    }
    return file;
}

Mnist load_mnist(const std::filesystem::path& images_path, const std::filesystem::path& labels_path) {
    Mnist mnist;
    std::ifstream images = open(images_path);
    if (read_u32(images) != 2051) {
        throw std::runtime_error("Not an idx image file: " + images_path.string());
    }
    mnist.count = read_u32(images);
    const std::size_t rows = read_u32(images);
    const std::size_t cols = read_u32(images);
    mnist.image_size = rows * cols;
    mnist.images.resize(mnist.count * mnist.image_size);
    images.read(reinterpret_cast<char*>(mnist.images.data()), mnist.images.size());

    std::ifstream labels = open(labels_path);
    if (read_u32(labels) != 2049) {
        throw std::runtime_error("Not an idx label file: " + labels_path.string());
    }
    if (read_u32(labels) != mnist.count) {
        throw std::runtime_error("Image and label counts don't match");
    }
    mnist.labels.resize(mnist.count);
    labels.read(reinterpret_cast<char*>(mnist.labels.data()), mnist.labels.size());

    if (!images || !labels) {
        throw std::runtime_error("Unexpected end of file while reading mnist");
    }
    return mnist;
}

void fill_sample(const Mnist& mnist, std::size_t index, std::vector<float>& x, std::vector<float>& y) {
    // scales pixels to [0, 1] and one-hot encodes the label
    const std::uint8_t* pixels = mnist.images.data() + index * mnist.image_size;
    for (std::size_t i = 0; i < mnist.image_size; i++) {
        x[i] = static_cast<float>(pixels[i]) / 255.0f;
    }
    std::fill(y.begin(), y.end(), 0.0f);
    y[mnist.labels[index]] = 1.0f;
}

std::size_t argmax(const std::vector<float>& values) {
    return std::distance(values.begin(), std::max_element(values.begin(), values.end()));
}

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path data_dir = argc > 1 ? argv[1] : "data";
    constexpr std::size_t classes = 10;
    constexpr std::size_t epochs = 5;
    constexpr float learning_rate = 0.01f;
    // print the running loss every this many samples
    constexpr std::size_t log_every = 10000;

    Mnist train;
    Mnist test;
    try {
        train = load_mnist(data_dir / "train-images.idx3-ubyte", data_dir / "train-labels.idx1-ubyte");
        test = load_mnist(data_dir / "t10k-images.idx3-ubyte", data_dir / "t10k-labels.idx1-ubyte");
    } catch (const std::exception& e) {
        std::println(std::cerr, "{}", e.what());
        return 1;
    }
    std::println("loaded {} training and {} test images", train.count, test.count);

    // one generator shared by every initializer, so layers get different values
    mlfo::initializer::Generator generator(42);
    // he for layers followed by relu, xavier for the rest
    auto he = std::make_shared<mlfo::initializer::He<std::normal_distribution<float>>>(generator);
    auto xavier = std::make_shared<mlfo::initializer::Xavier<std::uniform_real_distribution<float>>>(generator);
    // similar to LeNet
    std::vector<std::unique_ptr<mlfo::layer::Layer<float>>> layers;
    layers.push_back(std::make_unique<mlfo::layer::Conv2DLayer<float>>(1, 28, 28, 5, 5, 6, 1, 1, 2, 2, he));
    layers.push_back(std::make_unique<mlfo::layer::activation::ReLUActivation<float>>());
    layers.push_back(std::make_unique<mlfo::layer::pool::MaxPool<float>>(28, 28, 2, 2));
    layers.push_back(std::make_unique<mlfo::layer::Conv2DLayer<float>>(6, 14, 14, 5, 5, 16, 1, 1, 0, 0, he));
    layers.push_back(std::make_unique<mlfo::layer::activation::ReLUActivation<float>>());
    layers.push_back(std::make_unique<mlfo::layer::pool::MaxPool<float>>(10, 10, 2, 2));
    layers.push_back(std::make_unique<mlfo::layer::FlattenLayer<float>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(16 * 5 * 5, 120, he));
    layers.push_back(std::make_unique<mlfo::layer::activation::ReLUActivation<float>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(120, 84, he));
    layers.push_back(std::make_unique<mlfo::layer::activation::ReLUActivation<float>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(84, 10, xavier));
    // no softmax or cross entropy yet, so the raw outputs are fit to one-hot labels with mse
    auto loss = std::make_unique<mlfo::layer::loss::MSELoss<float>>(classes);
    auto model = mlfo::model::Model<float>(std::move(layers), std::move(loss));
    model.build();
    auto optim = mlfo::optimizer::SGD<float>(model, learning_rate);
    // the first output is the prediction, the second is the loss, see Model::build()
    const auto& prediction = model.outputs()[0].get();
    const auto& loss_value = model.outputs()[1].get();

    // reused for every sample, so the loop doesn't allocate
    std::vector<float> x(train.image_size);
    std::vector<float> y(classes);
    std::vector<std::size_t> order(train.count);
    std::iota(order.begin(), order.end(), 0);

    for (std::size_t epoch = 1; epoch <= epochs; epoch++) {
        const auto start = std::chrono::steady_clock::now();
        std::shuffle(order.begin(), order.end(), generator);
        double running_loss = 0.0;
        std::size_t running_correct = 0;
        // batch size is 1, so the parameters are updated after every sample
        for (std::size_t step = 0; step < train.count; step++) {
            fill_sample(train, order[step], x, y);
            model.set_inputs({x});
            model.set_labels({y});
            model.forward();
            optim.zero_grad();
            model.backward();
            optim.step();
            running_loss += loss_value.values()[0];
            running_correct += argmax(prediction.values()) == train.labels[order[step]];
            if ((step + 1) % log_every == 0) {
                std::println(
                    "epoch {} [{}/{}] loss {:.5f} accuracy {:.2f}%",
                    epoch, step + 1, train.count,
                    running_loss / log_every, 100.0 * running_correct / log_every
                );
                running_loss = 0.0;
                running_correct = 0;
            }
        }

        // evaluate on the test set, forward only
        double test_loss = 0.0;
        std::size_t test_correct = 0;
        for (std::size_t i = 0; i < test.count; i++) {
            fill_sample(test, i, x, y);
            model.set_inputs({x});
            model.set_labels({y});
            model.forward();
            test_loss += loss_value.values()[0];
            test_correct += argmax(prediction.values()) == test.labels[i];
        }
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
        std::println(
            "epoch {} done in {:.1f}s, test loss {:.5f} test accuracy {:.2f}%",
            epoch, elapsed.count(), test_loss / test.count, 100.0 * test_correct / test.count
        );
    }
    return 0;
}
