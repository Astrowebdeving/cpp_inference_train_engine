#include "mini_mla/checkpoint.hpp"

#include "mini_mla/io.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <set>
#include <unistd.h>

namespace mini_mla {
namespace {
void require(Status s) {
    if (!s)
        throw std::runtime_error(s.error().message);
}
class AtomicDirectory {
  public:
    explicit AtomicDirectory(std::filesystem::path path) : destination_(std::move(path)) {
        if (std::filesystem::exists(destination_))
            throw std::runtime_error("Destination already exists: " + destination_.string());
        auto parent = destination_.parent_path();
        if (parent.empty())
            parent = ".";
        std::filesystem::create_directories(parent);
        temporary_ = destination_.string() + ".tmp-" + std::to_string(::getpid()) + "-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        if (!std::filesystem::create_directory(temporary_))
            throw std::runtime_error("Cannot create checkpoint staging directory");
    }
    ~AtomicDirectory() {
        if (!temporary_.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(temporary_, ec);
        }
    }
    [[nodiscard]] const std::filesystem::path& path() const { return temporary_; }
    void commit() {
        require(sync_directory(temporary_));
        std::filesystem::rename(temporary_, destination_);
        temporary_.clear();
        auto parent = destination_.parent_path();
        require(sync_directory(parent.empty() ? "." : parent));
    }

  private:
    std::filesystem::path destination_, temporary_;
};
class TensorWriter {
  public:
    TensorWriter(std::filesystem::path directory, std::string filename)
        : path_(directory / filename), filename_(std::move(filename)),
          stream_(path_, std::ios::binary) {
        if (!stream_)
            throw std::runtime_error("Cannot open tensor output");
    }
    Json add(std::string_view name, ParameterRole role, const Matrix& m, std::string_view dtype) {
        if (!ops::finite(m.values))
            throw std::runtime_error("Cannot serialize nonfinite tensor: " + std::string(name));
        std::vector<std::uint16_t> half;
        auto bytes = std::as_bytes(std::span{m.values});
        if (dtype == "f16") {
            half.resize(m.size());
            for (std::size_t i = 0; i < m.size(); ++i) {
                if (std::abs(m.values[i]) > 65504)
                    throw std::runtime_error("Tensor exceeds FP16 range");
                half[i] = to_half(m.values[i]);
            }
            bytes = std::as_bytes(std::span{half});
        } else if (dtype != "f32")
            throw std::invalid_argument("Export dtype must be f32 or f16");
        stream_.write(reinterpret_cast<const char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
        if (!stream_)
            throw std::runtime_error("Tensor write failed");
        Json entry = Json::Object{
            {"name", std::string(name)},   {"role", std::string(role_name(role))},
            {"dtype", std::string(dtype)}, {"shape", Json::Array{m.rows, m.cols}},
            {"file", filename_},           {"offset", offset_},
            {"nbytes", bytes.size()},      {"crc32", static_cast<std::size_t>(crc32(bytes))}};
        offset_ += bytes.size();
        return entry;
    }
    std::string finish() {
        stream_.flush();
        if (!stream_)
            throw std::runtime_error("Tensor flush failed");
        stream_.close();
        if (stream_.fail())
            throw std::runtime_error("Tensor close failed");
        require(sync_file(path_));
        auto mapped = MappedFile::open(path_);
        if (!mapped)
            throw std::runtime_error(mapped.error().message);
        return hex(sha256(mapped.value().bytes()));
    }

  private:
    std::filesystem::path path_;
    std::string filename_;
    std::ofstream stream_;
    std::size_t offset_ = 0;
};
Status save(const std::filesystem::path& path, const Model& model, const Optimizer* optimizer,
            const Json& state, const std::filesystem::path& tokenizer_path,
            std::string_view dtype) {
    try {
        auto tokenizer = Tokenizer::load(tokenizer_path);
        if (!tokenizer)
            return Status::failure(tokenizer.error());
        if (tokenizer.value().vocab_size() != model.config().vocab_size)
            return Status::failure(ErrorCode::invalid_argument,
                                   "Model and tokenizer vocabulary sizes differ");
        AtomicDirectory dir(path);
        Json::Array tensors;
        TensorWriter weights(dir.path(), "weights.bin");
        for (const auto& p : model.parameters())
            tensors.push_back(weights.add(p.name, p.role, p.value, dtype));
        Json::Object hashes{{"weights.bin", weights.finish()}};
        if (optimizer) {
            if (optimizer->states().size() != model.parameters().size())
                throw std::invalid_argument("Cannot checkpoint uninitialized optimizer");
            TensorWriter moments(dir.path(), "optimizer.bin");
            for (std::size_t i = 0; i < model.parameters().size(); ++i) {
                const auto& p = model.parameters()[i];
                const auto& s = optimizer->states()[i];
                bool muon = optimizer->config().kind == "hybrid_muon_adamw" &&
                            p.role == ParameterRole::hidden_matrix;
                auto matches = [&](const Matrix& x) {
                    return x.rows == p.value.rows && x.cols == p.value.cols &&
                           x.size() == p.value.size();
                };
                if (!matches(s.first) || (muon ? s.second.size() != 0 : !matches(s.second)))
                    throw std::invalid_argument("Optimizer state shape mismatch");
                tensors.push_back(
                    moments.add("optimizer." + p.name + ".first", p.role, s.first, "f32"));
                if (s.second.size())
                    tensors.push_back(
                        moments.add("optimizer." + p.name + ".second", p.role, s.second, "f32"));
            }
            hashes["optimizer.bin"] = moments.finish();
            require(write_json(dir.path() / "trainer_state.json", state));
            auto trainer = read_text(dir.path() / "trainer_state.json");
            if (!trainer)
                return Status::failure(trainer.error());
            hashes["trainer_state.json"] = hash_text(trainer.value());
        }
        auto text = read_text(tokenizer_path);
        if (!text)
            return Status::failure(text.error());
        require(write_text(dir.path() / "tokenizer.json", text.value()));
        require(write_json(dir.path() / "config.json", model.config().json()));
        Json manifest =
            Json::Object{{"format", optimizer ? "mini-mla-checkpoint" : "mini-mla-model"},
                         {"version", 1},
                         {"model_config_sha256", model.config().hash()},
                         {"tokenizer_sha256", tokenizer.value().hash()},
                         {"parameter_count", model.config().parameter_count()},
                         {"dtype", std::string(dtype)},
                         {"chat_template", "mini-mla-chat-v1"},
                         {"step", static_cast<std::size_t>(optimizer ? optimizer->steps() : 0)},
                         {"tensors", std::move(tensors)},
                         {"sha256", std::move(hashes)}};
        if (optimizer)
            manifest["optimizer"] = optimizer->config().json();
        require(write_json(dir.path() / "manifest.json", manifest));
        auto m = read_text(dir.path() / "manifest.json");
        if (!m)
            return Status::failure(m.error());
        require(write_text(dir.path() / "COMPLETE", hash_text(m.value()) + "\n"));
        dir.commit();
        return ok();
    } catch (const std::exception& e) {
        return Status::failure(ErrorCode::io_error, e.what());
    }
}
} // namespace
Status save_checkpoint(const std::filesystem::path& p, const Model& m, const Optimizer& o,
                       const Json& s, const std::filesystem::path& t) {
    return save(p, m, &o, s, t, "f32");
}
Status export_model(const std::filesystem::path& p, const Model& m, const std::filesystem::path& t,
                    std::string_view dtype) {
    return save(p, m, nullptr, Json(), t, dtype);
}
Result<LoadedCheckpoint> load_checkpoint(const std::filesystem::path& path, bool training) {
    try {
        auto raw = read_text(path / "manifest.json");
        if (!raw)
            return Result<LoadedCheckpoint>::failure(raw.error());
        auto complete = read_text(path / "COMPLETE");
        if (!complete || complete.value() != hash_text(raw.value()) + "\n")
            return Result<LoadedCheckpoint>::failure(
                ErrorCode::io_error, "Checkpoint is incomplete or manifest checksum failed");
        auto parsed = Json::parse(raw.value());
        if (!parsed)
            return Result<LoadedCheckpoint>::failure(parsed.error());
        const auto& manifest = parsed.value();
        auto format = manifest.at("format").string();
        if ((format != "mini-mla-checkpoint" && format != "mini-mla-model") ||
            manifest.at("version").integer() != 1 || (training && format != "mini-mla-checkpoint"))
            return Result<LoadedCheckpoint>::failure(
                ErrorCode::unsupported,
                "Unsupported model format or inference export used for resume");
        auto cfg = ModelConfig::load(path / "config.json");
        if (!cfg)
            return Result<LoadedCheckpoint>::failure(cfg.error());
        if (cfg.value().hash() != manifest.at("model_config_sha256").string() ||
            cfg.value().parameter_count() != manifest.at("parameter_count").size_integer())
            return Result<LoadedCheckpoint>::failure(ErrorCode::invalid_argument,
                                                     "Checkpoint model configuration mismatch");
        auto tokenizer = Tokenizer::load(path / "tokenizer.json");
        if (!tokenizer)
            return Result<LoadedCheckpoint>::failure(tokenizer.error());
        if (tokenizer.value().hash() != manifest.at("tokenizer_sha256").string() ||
            tokenizer.value().vocab_size() != cfg.value().vocab_size)
            return Result<LoadedCheckpoint>::failure(ErrorCode::invalid_argument,
                                                     "Checkpoint tokenizer mismatch");
        std::map<std::string, MappedFile> files;
        for (const auto& [name, hash] : manifest.at("sha256").object()) {
            if (name != "weights.bin" && name != "optimizer.bin" && name != "trainer_state.json")
                throw std::invalid_argument("Invalid checkpoint filename");
            auto f = MappedFile::open(path / name);
            if (!f)
                return Result<LoadedCheckpoint>::failure(f.error());
            if (hex(sha256(f.value().bytes())) != hash.string())
                throw std::invalid_argument("Checkpoint file checksum failed: " + name);
            files.emplace(name, std::move(f).value());
        }
        auto made = Model::create(cfg.value(), 0, training, false);
        if (!made)
            return Result<LoadedCheckpoint>::failure(made.error());
        LoadedCheckpoint loaded;
        loaded.model = std::move(made).value();
        loaded.tokenizer = std::move(tokenizer).value();
        if (training) {
            auto config = OptimizerConfig::from_json(manifest.at("optimizer"));
            if (!config)
                return Result<LoadedCheckpoint>::failure(config.error());
            loaded.optimizer = std::make_unique<Optimizer>(config.value());
            auto init = loaded.optimizer->initialize(*loaded.model);
            if (!init)
                return Result<LoadedCheckpoint>::failure(init.error());
            loaded.optimizer->set_steps(manifest.at("step").size_integer());
            auto state = read_json(path / "trainer_state.json");
            if (!state)
                return Result<LoadedCheckpoint>::failure(state.error());
            loaded.trainer_state = std::move(state).value();
        }
        struct Destination {
            Matrix* matrix;
            ParameterRole role;
            std::string file;
        };
        std::map<std::string, Destination> expected;
        for (std::size_t i = 0; i < loaded.model->parameters().size(); ++i) {
            auto& p = loaded.model->parameters()[i];
            expected.emplace(p.name, Destination{&p.value, p.role, "weights.bin"});
            if (training) {
                auto& s = loaded.optimizer->states()[i];
                expected.emplace("optimizer." + p.name + ".first",
                                 Destination{&s.first, p.role, "optimizer.bin"});
                if (s.second.size())
                    expected.emplace("optimizer." + p.name + ".second",
                                     Destination{&s.second, p.role, "optimizer.bin"});
            }
        }
        std::set<std::string> seen;
        std::map<std::string, std::size_t> ends;
        for (const auto& entry : manifest.at("tensors").array()) {
            auto name = entry.at("name").string(), file = entry.at("file").string(),
                 dtype = entry.at("dtype").string();
            if (file != "weights.bin" && file != "optimizer.bin")
                throw std::invalid_argument("Invalid tensor filename");
            if (!seen.insert(name).second)
                throw std::invalid_argument("Duplicate named tensor");
            auto offset = entry.at("offset").size_integer(),
                 nbytes = entry.at("nbytes").size_integer();
            auto fit = files.find(file);
            if (fit == files.end() || offset != ends[file] || offset > fit->second.bytes().size() ||
                nbytes > fit->second.bytes().size() - offset)
                throw std::invalid_argument("Invalid tensor extent or missing tensor file");
            auto bytes = fit->second.bytes().subspan(offset, nbytes);
            ends[file] = offset + nbytes;
            if (crc32(bytes) != entry.at("crc32").size_integer())
                throw std::invalid_argument("Tensor checksum failed: " + name);
            auto it = expected.find(name);
            if (it == expected.end()) {
                if (!training && file == "optimizer.bin" && format == "mini-mla-checkpoint")
                    continue;
                throw std::invalid_argument("Unexpected tensor: " + name);
            }
            auto& destination = it->second;
            auto& m = *destination.matrix;
            const auto& shape = entry.at("shape").array();
            std::size_t item = dtype == "f32" ? 4 : dtype == "f16" ? 2 : 0;
            if (!item || (file == "optimizer.bin" && dtype != "f32") || shape.size() != 2 ||
                shape[0].integer() != m.rows || shape[1].integer() != m.cols ||
                nbytes != m.size() * item ||
                entry.at("role").string() != role_name(destination.role) ||
                destination.file != file)
                throw std::invalid_argument("Tensor shape, dtype, role, or file mismatch: " + name);
            if (dtype == "f32")
                std::memcpy(m.data(), bytes.data(), nbytes);
            else
                for (std::size_t k = 0; k < m.size(); ++k) {
                    std::uint16_t h;
                    std::memcpy(&h, bytes.data() + 2 * k, 2);
                    m.values[k] = from_half(h);
                }
            if (!ops::finite(m.values))
                throw std::invalid_argument("Nonfinite checkpoint tensor: " + name);
            expected.erase(it);
        }
        if (!expected.empty())
            throw std::invalid_argument("Checkpoint is missing tensor: " + expected.begin()->first);
        for (const auto& [file, end] : ends)
            if (end != files.at(file).bytes().size())
                throw std::invalid_argument("Unindexed trailing tensor data");
        return Result<LoadedCheckpoint>::success(std::move(loaded));
    } catch (const std::exception& e) {
        return Result<LoadedCheckpoint>::failure(ErrorCode::io_error, e.what());
    }
}
Status update_latest_link(const std::filesystem::path& link, const std::filesystem::path& target) {
    try {
        if (std::filesystem::exists(link) && !std::filesystem::is_symlink(link))
            return Status::failure(ErrorCode::invalid_argument,
                                   "Latest path must be absent or a symlink");
        auto temp = link.string() + ".tmp-" + std::to_string(::getpid());
        std::error_code ec;
        std::filesystem::remove(temp, ec);
        std::filesystem::create_symlink(std::filesystem::relative(target, link.parent_path()),
                                        temp);
        std::filesystem::rename(temp, link);
        return sync_directory(link.parent_path());
    } catch (const std::exception& e) {
        return Status::failure(ErrorCode::io_error, e.what());
    }
}
} // namespace mini_mla
