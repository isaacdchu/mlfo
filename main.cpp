#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <print>
#include <span>
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

void fill_batch(
    const Mnist& mnist,
    std::span<const std::size_t> indices,
    std::size_t classes,
    std::vector<float>& x,
    std::vector<float>& y
) {
    // scales pixels to [0, 1] and one-hot encodes the labels, one sample after another
    // shrinking keeps the capacity, so a smaller last batch doesn't allocate
    x.resize(indices.size() * mnist.image_size);
    y.resize(indices.size() * classes);
    std::fill(y.begin(), y.end(), 0.0f);
    for (std::size_t b = 0; b < indices.size(); b++) {
        const std::uint8_t* pixels = mnist.images.data() + indices[b] * mnist.image_size;
        for (std::size_t i = 0; i < mnist.image_size; i++) {
            x[b * mnist.image_size + i] = static_cast<float>(pixels[i]) / 255.0f;
        }
        y[b * classes + mnist.labels[indices[b]]] = 1.0f;
    }
}

std::size_t count_correct(
    const Mnist& mnist,
    std::span<const std::size_t> indices,
    const std::vector<float>& logits
) {
    // the number of samples in the batch whose largest logit is their label
    const std::size_t classes = logits.size() / indices.size();
    std::size_t correct = 0;
    for (std::size_t b = 0; b < indices.size(); b++) {
        const auto begin = logits.begin() + b * classes;
        const auto best = std::distance(begin, std::max_element(begin, begin + classes));
        correct += static_cast<std::size_t>(best) == mnist.labels[indices[b]];
    }
    return correct;
}

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path data_dir = argc > 1 ? argv[1] : "data";
    constexpr std::size_t classes = 10;
    constexpr std::size_t epochs = 5;
    // the loss is averaged over the batch, so a larger batch can take a larger step
    constexpr std::size_t batch_size = 32;
    constexpr float learning_rate = 0.05f;
    // print the running loss every this many batches
    constexpr std::size_t log_every = 300;

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
    // similar to LeNet, each layer takes its input shape from the layer before it
    std::vector<std::unique_ptr<mlfo::layer::Layer<float>>> layers;
    // (1, 28, 28) -> (6, 28, 28)
    layers.push_back(std::make_unique<mlfo::layer::Conv2DLayer<float>>(6, 5, 5, 1, 1, 2, 2, he));
    layers.push_back(std::make_unique<mlfo::layer::activation::ReLUActivation<float>>());
    // -> (6, 14, 14)
    layers.push_back(std::make_unique<mlfo::layer::pool::MaxPool<float>>(2, 2));
    // -> (16, 10, 10)
    layers.push_back(std::make_unique<mlfo::layer::Conv2DLayer<float>>(16, 5, 5, 1, 1, 0, 0, he));
    layers.push_back(std::make_unique<mlfo::layer::activation::ReLUActivation<float>>());
    // -> (16, 5, 5)
    layers.push_back(std::make_unique<mlfo::layer::pool::MaxPool<float>>(2, 2));
    // -> (1, 400)
    layers.push_back(std::make_unique<mlfo::layer::FlattenLayer<float>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(120, he));
    layers.push_back(std::make_unique<mlfo::layer::activation::ReLUActivation<float>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(84, he));
    layers.push_back(std::make_unique<mlfo::layer::activation::ReLUActivation<float>>());
    layers.push_back(std::make_unique<mlfo::layer::DenseLayer<float>>(classes, xavier));
    // the last dense layer outputs logits, softmax is applied inside the loss
    auto loss = std::make_unique<mlfo::layer::loss::CrossEntropyLoss<float>>();
    // mnist images are single channel 28 x 28
    auto model = mlfo::model::Model<float>({1, 28, 28}, std::move(layers), std::move(loss), batch_size);
    model.build();
    auto optim = mlfo::optimizer::SGD<float>(model, learning_rate);
    // the first output is the prediction, the second is the loss, see Model::build()
    const auto& prediction = model.outputs()[0].get();
    const auto& loss_value = model.outputs()[1].get();

    // reused for every batch, so the loop doesn't allocate
    std::vector<float> x(batch_size * train.image_size);
    std::vector<float> y(batch_size * classes);
    std::vector<std::size_t> train_order(train.count);
    std::iota(train_order.begin(), train_order.end(), 0);
    std::vector<std::size_t> test_order(test.count);
    std::iota(test_order.begin(), test_order.end(), 0);

    for (std::size_t epoch = 1; epoch <= epochs; epoch++) {
        const auto start = std::chrono::steady_clock::now();
        std::shuffle(train_order.begin(), train_order.end(), generator);
        double running_loss = 0.0;
        std::size_t running_correct = 0;
        std::size_t running_samples = 0;
        // the last batch holds the remainder when the dataset doesn't divide evenly
        for (std::size_t first = 0, step = 1; first < train.count; first += batch_size, step++) {
            const auto indices = std::span<const std::size_t>(train_order).subspan(
                first, std::min(batch_size, train.count - first)
            );
            fill_batch(train, indices, classes, x, y);
            model.set_inputs({x});
            model.set_labels({y});
            model.forward();
            optim.zero_grad();
            model.backward();
            optim.step();
            // the loss output holds one loss per sample
            for (const float value : loss_value.values()) {
                running_loss += value;
            }
            running_correct += count_correct(train, indices, prediction.values());
            running_samples += indices.size();
            if (step % log_every == 0) {
                std::println(
                    "epoch {} [{}/{}] loss {:.5f} accuracy {:.2f}%",
                    epoch, first + indices.size(), train.count,
                    running_loss / running_samples, 100.0 * running_correct / running_samples
                );
                running_loss = 0.0;
                running_correct = 0;
                running_samples = 0;
            }
        }

        // evaluate on the test set, forward only
        double test_loss = 0.0;
        std::size_t test_correct = 0;
        for (std::size_t first = 0; first < test.count; first += batch_size) {
            const auto indices = std::span<const std::size_t>(test_order).subspan(
                first, std::min(batch_size, test.count - first)
            );
            fill_batch(test, indices, classes, x, y);
            model.set_inputs({x});
            model.set_labels({y});
            model.forward();
            for (const float value : loss_value.values()) {
                test_loss += value;
            }
            test_correct += count_correct(test, indices, prediction.values());
        }
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
        std::println(
            "epoch {} done in {:.1f}s, test loss {:.5f} test accuracy {:.2f}%",
            epoch, elapsed.count(), test_loss / test.count, 100.0 * test_correct / test.count
        );
    }
    return 0;
}
