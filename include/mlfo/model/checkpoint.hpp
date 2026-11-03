#ifndef MLFO_MODEL_CHECKPOINT_HPP
#define MLFO_MODEL_CHECKPOINT_HPP

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <istream>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../misc/concepts.hpp"
#include "../misc/serialization.hpp"
#include "../optimizer/load.hpp"
#include "../optimizer/optimizer.hpp"
#include "load.hpp"
#include "model.hpp"

// checkpoints hold everything needed to pause and resume training: the model (architecture and
// parameter values), the parameters' gradients (so a checkpoint taken while accumulating
// gradients over several backward() calls resumes exactly), and the optimizer with its
// hyperparameters and state, e.g. Adam's moments and step count.
// the training loop's own state, e.g. the epoch or a data loader's shuffle generator, is not
// part of the checkpoint

namespace mlfo::model {

// a model and the optimizer that trains it, as returned by load_checkpoint() and
// clone_checkpoint(). the model is heap allocated so that moving a Checkpoint keeps the
// optimizer's reference to it valid
template <mlfo::misc::Number T>
struct Checkpoint {
    // declared before the optimizer, so the optimizer is destroyed first
    std::unique_ptr<Model<T>> model;
    std::unique_ptr<mlfo::optimizer::Optimizer<T>> optimizer;
};

namespace detail {

inline constexpr const char* checkpoint_header = "mlfo checkpoint 1";

template <mlfo::misc::Number T>
void check_optimizer(
    const Model<T>& model, const mlfo::optimizer::Optimizer<T>& optimizer, const char* where
) {
    if (&optimizer.model() != &model) {
        throw std::invalid_argument(
            std::string("[mlfo::model::") + where + "] Optimizer does not belong to the model"
        );
    }
}

} // namespace detail

// the header line, then the number of lines the model takes, then the model as written by
// Model::serialize(), then the number of parameters followed by one line of gradients per
// parameter, then the optimizer as written by Optimizer::serialize()
template <mlfo::misc::Number T>
void serialize_checkpoint(
    std::ostream& os, const Model<T>& model, const mlfo::optimizer::Optimizer<T>& optimizer
) {
    detail::check_optimizer(model, optimizer, "serialize_checkpoint");
    // the model's reader consumes the rest of its input, so the model is written with its
    // length up front and read back from a stream of just those lines
    std::ostringstream model_stream;
    model.serialize(model_stream);
    const std::string model_text = model_stream.str();
    std::size_t lines = 0;
    for (const char c : model_text) {
        lines += c == '\n';
    }
    os << detail::checkpoint_header << "\n";
    mlfo::misc::write_line(os, std::vector<std::size_t>{lines});
    os << model_text;
    const auto& parameters = model.parameters();
    mlfo::misc::write_line(os, std::vector<std::size_t>{parameters.size()});
    for (const auto& param_ref : parameters) {
        mlfo::misc::write_line(os, param_ref.get().gradients());
    }
    optimizer.serialize(os);
}

template <mlfo::misc::Number T>
void save_checkpoint(
    const std::filesystem::path& path,
    const Model<T>& model,
    const mlfo::optimizer::Optimizer<T>& optimizer
) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error(
            "[mlfo::model::save_checkpoint] Could not open " + path.string()
        );
    }
    serialize_checkpoint(file, model, optimizer);
}

// reads a checkpoint written by serialize_checkpoint(), with the model built for batch_size
// samples at a time. the batch size is not saved, as with deserialize()
template <mlfo::misc::Number T>
Checkpoint<T> deserialize_checkpoint(std::istream& is, std::size_t batch_size = 1) {
    if (mlfo::misc::read_line(is) != detail::checkpoint_header) {
        throw std::runtime_error(
            "[mlfo::model::deserialize_checkpoint] Not an mlfo checkpoint"
        );
    }
    const std::size_t lines = mlfo::misc::read_values<std::size_t>(is, 1)[0];
    std::string model_text;
    for (std::size_t i = 0; i < lines; i++) {
        model_text += mlfo::misc::read_line(is);
        model_text += '\n';
    }
    std::istringstream model_stream(model_text);
    Checkpoint<T> checkpoint;
    checkpoint.model = std::make_unique<Model<T>>(deserialize<T>(model_stream, batch_size));
    const auto& parameters = checkpoint.model->parameters();
    if (mlfo::misc::read_values<std::size_t>(is, 1)[0] != parameters.size()) {
        throw std::runtime_error(
            "[mlfo::model::deserialize_checkpoint] Gradients do not match the model's parameters"
        );
    }
    for (const auto& param_ref : parameters) {
        param_ref.get().set_gradients(
            mlfo::misc::read_values<T>(is, param_ref.get().gradients().size())
        );
    }
    checkpoint.optimizer = mlfo::optimizer::deserialize<T>(is, *checkpoint.model);
    return checkpoint;
}

template <mlfo::misc::Number T>
Checkpoint<T> load_checkpoint(const std::filesystem::path& path, std::size_t batch_size = 1) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error(
            "[mlfo::model::load_checkpoint] Could not open " + path.string()
        );
    }
    return deserialize_checkpoint<T>(file, batch_size);
}

// a deep copy of the model, its parameters' gradients and the optimizer's state, with the copied
// model's graph built for batch_size samples at a time. training the copy continues exactly as
// training the original would, without affecting it
template <mlfo::misc::Number T>
Checkpoint<T> clone_checkpoint(
    const Model<T>& model, const mlfo::optimizer::Optimizer<T>& optimizer, std::size_t batch_size
) {
    detail::check_optimizer(model, optimizer, "clone_checkpoint");
    Checkpoint<T> checkpoint;
    checkpoint.model = std::make_unique<Model<T>>(model.clone(batch_size));
    const auto& source = model.parameters();
    const auto& target = checkpoint.model->parameters();
    for (std::size_t p = 0; p < source.size(); p++) {
        target[p].get().set_gradients(source[p].get().gradients());
    }
    checkpoint.optimizer = optimizer.clone(*checkpoint.model);
    return checkpoint;
}

// as above, built for the same batch size as model
template <mlfo::misc::Number T>
Checkpoint<T> clone_checkpoint(
    const Model<T>& model, const mlfo::optimizer::Optimizer<T>& optimizer
) {
    return clone_checkpoint(model, optimizer, model.max_batch_size());
}

} // namespace mlfo::model

#endif // MLFO_MODEL_CHECKPOINT_HPP
