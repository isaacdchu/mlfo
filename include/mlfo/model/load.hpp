#ifndef MLFO_MODEL_LOAD_HPP
#define MLFO_MODEL_LOAD_HPP

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <istream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../layer/activation/relu_activation.hpp"
#include "../layer/activation/sigmoid_activation.hpp"
#include "../layer/activation/tanh_activation.hpp"
#include "../layer/conv2d_layer.hpp"
#include "../layer/dense_layer.hpp"
#include "../layer/flatten_layer.hpp"
#include "../layer/layer.hpp"
#include "../layer/loss/cross_entropy_loss.hpp"
#include "../layer/loss/loss_layer.hpp"
#include "../layer/loss/mse_loss.hpp"
#include "../layer/pool/max_pool.hpp"
#include "../misc/concepts.hpp"
#include "../misc/serialization.hpp"
#include "model.hpp"

// reading models back is kept out of model.hpp, as it is the only place that has to know every
// concrete layer type. a new layer is added to the chain in deserialize() and included above

namespace mlfo::model {

// reads a model written by Model::serialize() and builds it for batch_size samples at a time
// the batch size is not serialized, as it is chosen by whoever uses the model
template <mlfo::misc::Number T>
Model<T> deserialize(std::istream& is, std::size_t batch_size = 1) {
    const std::string name = mlfo::misc::read_line(is);
    // the shape line is the only one whose length is not known up front
    const std::string shape_line = mlfo::misc::read_line(is);
    const std::size_t rank = std::ranges::count(shape_line, ',') + 1;
    std::istringstream shape_stream(shape_line);
    std::vector<std::size_t> input_shape =
        mlfo::misc::read_values<std::size_t>(shape_stream, rank);
    std::vector<std::unique_ptr<mlfo::layer::Layer<T>>> layers;
    std::unique_ptr<mlfo::layer::loss::LossLayer<T>> loss;
    std::string type;
    while (std::getline(is, type)) {
        if (!type.empty() && type.back() == '\r') {
            type.pop_back();
        }
        if (type.empty()) {
            // a trailing blank line, e.g. added by an editor
            continue;
        }
        if (loss) {
            throw std::runtime_error(
                "[mlfo::model::deserialize] Loss layer must be the last layer"
            );
        }
        if (type == "DenseLayer") {
            layers.push_back(mlfo::layer::DenseLayer<T>::deserialize(is));
        } else if (type == "Conv2DLayer") {
            layers.push_back(mlfo::layer::Conv2DLayer<T>::deserialize(is));
        } else if (type == "FlattenLayer") {
            layers.push_back(mlfo::layer::FlattenLayer<T>::deserialize(is));
        } else if (type == "MaxPool") {
            layers.push_back(mlfo::layer::pool::MaxPool<T>::deserialize(is));
        } else if (type == "ReLUActivation") {
            layers.push_back(mlfo::layer::activation::ReLUActivation<T>::deserialize(is));
        } else if (type == "SigmoidActivation") {
            layers.push_back(mlfo::layer::activation::SigmoidActivation<T>::deserialize(is));
        } else if (type == "TanhActivation") {
            layers.push_back(mlfo::layer::activation::TanhActivation<T>::deserialize(is));
        } else if (type == "MSELoss") {
            loss = mlfo::layer::loss::MSELoss<T>::deserialize(is);
        } else if (type == "CrossEntropyLoss") {
            loss = mlfo::layer::loss::CrossEntropyLoss<T>::deserialize(is);
        } else {
            throw std::runtime_error(
                "[mlfo::model::deserialize] Unknown layer type '" + type + "'"
            );
        }
    }
    Model<T> model(std::move(input_shape), std::move(layers), std::move(loss), batch_size);
    model.name(name);
    // the layers' initializers hold the read parameters, so building restores them
    model.build();
    return model;
}

template <mlfo::misc::Number T>
Model<T> load(const std::filesystem::path& path, std::size_t batch_size = 1) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("[mlfo::model::load] Could not open " + path.string());
    }
    return deserialize<T>(file, batch_size);
}

} // namespace mlfo::model

#endif // MLFO_MODEL_LOAD_HPP
