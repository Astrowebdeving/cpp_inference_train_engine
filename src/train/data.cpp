#include "mini_mla/data.hpp"

#include <bit>
#include <cstring>
#include <numeric>
#include <sstream>

namespace mini_mla {
namespace {
std::uint32_t u32(const std::byte* p) {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
std::uint64_t u64(const std::byte* p) {
    std::uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}
std::uint64_t decimal(const Json& j) {
    const auto& s = j.string();
    std::size_t n;
    auto v = std::stoull(s, &n);
    if (n != s.size() || s.empty() || s[0] == '-')
        throw std::invalid_argument("Invalid unsigned state integer");
    return v;
}
} // namespace
static_assert(std::endian::native == std::endian::little);
std::size_t Batch::supervised_tokens() const {
    return static_cast<std::size_t>(std::count(mask.begin(), mask.end(), 1.0F));
}
Result<std::vector<float>> Batch::coefficients(float scale) const {
    if (batch < 1 || sequence < 1 || input.size() != target.size() ||
        target.size() != mask.size() ||
        input.size() != static_cast<std::size_t>(batch) * static_cast<std::size_t>(sequence) ||
        !std::isfinite(scale))
        return Result<std::vector<float>>::failure(ErrorCode::invalid_argument,
                                                   "Invalid batch shape or gradient scale");
    double count = 0;
    for (float x : mask) {
        if (x != 0 && x != 1)
            return Result<std::vector<float>>::failure(ErrorCode::invalid_argument,
                                                       "Loss mask must contain zero or one");
        count += x;
    }
    if (count == 0)
        return Result<std::vector<float>>::failure(ErrorCode::invalid_argument,
                                                   "Batch has no supervised tokens");
    auto c = mask;
    float inv = static_cast<float>(static_cast<double>(scale) / count);
    for (auto& x : c)
        x *= inv;
    return Result<std::vector<float>>::success(std::move(c));
}
Result<TokenShard> TokenShard::open(const std::filesystem::path& p, int vocab,
                                    std::string_view tokenizer_hash) {
    auto f = MappedFile::open(p);
    if (!f)
        return Result<TokenShard>::failure(f.error());
    auto bytes = f.value().bytes();
    if (bytes.size() < 160 || std::memcmp(bytes.data(), "MLATOK01", 8) ||
        u32(bytes.data() + 8) != 1 || u32(bytes.data() + 12) != 160 ||
        u32(bytes.data() + 16) != 1 || u32(bytes.data() + 20) != static_cast<std::uint32_t>(vocab))
        return Result<TokenShard>::failure(ErrorCode::invalid_argument,
                                           "Invalid token shard header: " + p.string());
    auto count = u64(bytes.data() + 24);
    if (count > (bytes.size() - 160) / 2 || count * 2 != bytes.size() - 160)
        return Result<TokenShard>::failure(ErrorCode::invalid_argument,
                                           "Token shard size mismatch: " + p.string());
    std::span<const std::uint8_t> th(reinterpret_cast<const std::uint8_t*>(bytes.data() + 32), 32);
    if (hex(th) != tokenizer_hash)
        return Result<TokenShard>::failure(ErrorCode::invalid_argument,
                                           "Tokenizer hash mismatch: " + p.string());
    auto payload = bytes.subspan(160);
    auto computed = sha256(payload);
    if (std::memcmp(computed.data(), bytes.data() + 64, 32))
        return Result<TokenShard>::failure(ErrorCode::io_error,
                                           "Token shard checksum failed: " + p.string());
    std::span<const std::uint16_t> tokens(reinterpret_cast<const std::uint16_t*>(payload.data()),
                                          static_cast<std::size_t>(count));
    for (auto id : tokens)
        if (id >= vocab)
            return Result<TokenShard>::failure(ErrorCode::invalid_argument,
                                               "Out-of-vocabulary token in shard");
    TokenShard s;
    s.file_ = std::move(f).value();
    s.tokens_ = tokens;
    s.payload_hash_ = hex(computed);
    return Result<TokenShard>::success(std::move(s));
}
void DataLoader::begin_epoch() {
    position_ = 0;
    do {
        multiplier_ = rng_.bounded(count_) + 1;
    } while (std::gcd(multiplier_, count_) != 1);
    offset_ = rng_.bounded(count_);
}
Result<std::unique_ptr<DataLoader>> DataLoader::tokens(std::span<const std::filesystem::path> paths,
                                                       int sequence, int vocab,
                                                       std::string_view hash, std::uint64_t seed) {
    if (paths.empty() || sequence < 1)
        return Result<std::unique_ptr<DataLoader>>::failure(
            ErrorCode::invalid_argument, "No token files or invalid sequence length");
    auto d = std::unique_ptr<DataLoader>(new DataLoader(sequence, seed));
    std::string identity = "tokens-v1:" + std::to_string(sequence) + ":" + std::string(hash);
    for (const auto& p : paths) {
        auto s = TokenShard::open(p, vocab, hash);
        if (!s)
            return Result<std::unique_ptr<DataLoader>>::failure(s.error());
        auto spans = s.value().tokens().empty()
                         ? 0
                         : (s.value().tokens().size() - 1) / static_cast<std::size_t>(sequence);
        d->count_ += spans;
        d->ends_.push_back(static_cast<std::size_t>(d->count_));
        identity += ":" + s.value().payload_hash();
        d->shards_.push_back(std::move(s).value());
    }
    if (!d->count_ || d->count_ > UINT32_MAX)
        return Result<std::unique_ptr<DataLoader>>::failure(
            ErrorCode::invalid_argument,
            "Dataset needs at least one T+1 span and at most 2^32-1 spans");
    d->fingerprint_ = hash_text(identity);
    d->begin_epoch();
    return Result<std::unique_ptr<DataLoader>>::success(std::move(d));
}
Result<std::unique_ptr<DataLoader>> DataLoader::sft(std::span<const std::filesystem::path> paths,
                                                    int sequence, int vocab, std::string_view hash,
                                                    std::uint64_t seed) {
    if (paths.empty() || sequence < 1 || vocab < 264)
        return Result<std::unique_ptr<DataLoader>>::failure(ErrorCode::invalid_argument,
                                                            "No SFT files or invalid shape");
    auto d = std::unique_ptr<DataLoader>(new DataLoader(sequence, seed));
    d->pad_ = vocab - 8;
    std::string identity = "sft-v1:" + std::to_string(sequence) + ":" + std::string(hash);
    try {
        for (const auto& p : paths) {
            auto file = MappedFile::open(p);
            if (!file)
                return Result<std::unique_ptr<DataLoader>>::failure(file.error());
            auto bytes = file.value().bytes();
            std::string_view text =
                bytes.empty()
                    ? std::string_view{}
                    : std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            auto payload_hash = hex(sha256(bytes));
            auto manifest = read_json(p.string() + ".manifest.json");
            if (!manifest)
                return Result<std::unique_ptr<DataLoader>>::failure(manifest.error());
            const auto& info = manifest.value();
            if (info.at("format").string() != "mini-mla-sft-v1" ||
                info.at("tokenizer_sha256").string() != hash ||
                info.at("payload_sha256").string() != payload_hash)
                return Result<std::unique_ptr<DataLoader>>::failure(
                    ErrorCode::invalid_argument, "SFT manifest or checksum mismatch");
            identity += ":" + payload_hash;
            while (!text.empty()) {
                auto end = text.find('\n');
                auto line = text.substr(0, end);
                text.remove_prefix(end == std::string_view::npos ? text.size() : end + 1);
                if (line.empty())
                    continue;
                auto j = Json::parse(line);
                if (!j)
                    return Result<std::unique_ptr<DataLoader>>::failure(j.error());
                const auto& ids = j.value().at("tokens").array();
                const auto& mask = j.value().at("mask").array();
                if (ids.size() < 2 || ids.size() > static_cast<std::size_t>(sequence) + 1 ||
                    mask.size() != ids.size())
                    throw std::invalid_argument(
                        "SFT sample length exceeds training context or mask length differs");
                Sample s;
                bool has_loss = false;
                for (std::size_t i = 0; i < ids.size(); ++i) {
                    int id = ids[i].integer(), bit = mask[i].integer();
                    if (id < 0 || id >= vocab || (bit != 0 && bit != 1))
                        throw std::invalid_argument("Invalid SFT token or mask");
                    s.tokens.push_back(static_cast<std::uint16_t>(id));
                    s.mask.push_back(static_cast<float>(bit));
                    if (i && bit)
                        has_loss = true;
                }
                if (!has_loss)
                    throw std::invalid_argument("SFT sample has no assistant tokens");
                d->samples_.push_back(std::move(s));
            }
        }
        d->count_ = d->samples_.size();
        if (!d->count_ || d->count_ > UINT32_MAX)
            throw std::invalid_argument("Empty or oversized SFT dataset");
        d->fingerprint_ = hash_text(identity);
        d->begin_epoch();
        return Result<std::unique_ptr<DataLoader>>::success(std::move(d));
    } catch (const std::exception& e) {
        return Result<std::unique_ptr<DataLoader>>::failure(ErrorCode::invalid_argument, e.what());
    }
}
Result<Batch> DataLoader::next(int batch) {
    if (batch < 1 || batch > 4096)
        return Result<Batch>::failure(ErrorCode::invalid_argument, "Invalid microbatch size");
    Batch b;
    b.batch = batch;
    b.sequence = sequence_;
    auto n = static_cast<std::size_t>(batch) * static_cast<std::size_t>(sequence_);
    b.input.resize(n, static_cast<std::uint16_t>(pad_));
    b.target.resize(n, static_cast<std::uint16_t>(pad_));
    b.mask.resize(n, samples_.empty() ? 1.0F : 0.0F);
    for (int row = 0; row < batch; ++row) {
        if (position_ == count_) {
            ++epoch_;
            begin_epoch();
        }
        auto index = static_cast<std::size_t>((multiplier_ * position_ + offset_) % count_);
        ++position_;
        auto start = static_cast<std::size_t>(row) * static_cast<std::size_t>(sequence_);
        if (samples_.empty()) {
            auto shard = static_cast<std::size_t>(
                std::upper_bound(ends_.begin(), ends_.end(), index) - ends_.begin());
            auto before = shard ? ends_[shard - 1] : 0;
            auto span = shards_[shard].tokens().subspan((index - before) *
                                                            static_cast<std::size_t>(sequence_),
                                                        static_cast<std::size_t>(sequence_) + 1);
            std::copy_n(span.begin(), sequence_,
                        b.input.begin() + static_cast<std::ptrdiff_t>(start));
            std::copy_n(span.begin() + 1, sequence_,
                        b.target.begin() + static_cast<std::ptrdiff_t>(start));
        } else {
            const auto& s = samples_[index];
            for (std::size_t j = 0; j + 1 < s.tokens.size(); ++j) {
                b.input[start + j] = s.tokens[j];
                b.target[start + j] = s.tokens[j + 1];
                b.mask[start + j] = s.mask[j + 1];
            }
        }
    }
    return Result<Batch>::success(std::move(b));
}
Json DataLoader::state() const {
    return Json::Object{{"fingerprint", fingerprint_},
                        {"count", std::to_string(count_)},
                        {"position", std::to_string(position_)},
                        {"epoch", std::to_string(epoch_)},
                        {"multiplier", std::to_string(multiplier_)},
                        {"offset", std::to_string(offset_)},
                        {"rng", std::to_string(rng_.state())}};
}
Status DataLoader::restore(const Json& j) {
    try {
        if (j.at("fingerprint").string() != fingerprint_ || decimal(j.at("count")) != count_)
            throw std::invalid_argument("Dataset or sequence length changed since checkpoint");
        auto position = decimal(j.at("position")), epoch = decimal(j.at("epoch")),
             a = decimal(j.at("multiplier")), b = decimal(j.at("offset")),
             rng = decimal(j.at("rng"));
        if (position > count_ || a < 1 || a > count_ || b >= count_ || std::gcd(a, count_) != 1)
            throw std::invalid_argument("Invalid data-loader checkpoint state");
        position_ = position;
        epoch_ = epoch;
        multiplier_ = a;
        offset_ = b;
        rng_.restore(rng);
        return ok();
    } catch (const std::exception& e) {
        return Status::failure(ErrorCode::invalid_argument, e.what());
    }
}
PrefetchLoader::PrefetchLoader(std::unique_ptr<DataLoader> loader, int batch, std::size_t capacity)
    : loader_(std::move(loader)), batch_(batch), capacity_(std::max(std::size_t{1}, capacity)),
      consumed_(loader_->state()), worker_([this] { run(); }) {}
PrefetchLoader::~PrefetchLoader() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    readable_.notify_all();
    writable_.notify_all();
    if (worker_.joinable())
        worker_.join();
}
void PrefetchLoader::run() {
    for (;;) {
        {
            std::unique_lock lock(mutex_);
            writable_.wait(lock, [&] { return stopping_ || queue_.size() < capacity_; });
            if (stopping_)
                return;
        }
        auto prepare = [&]() -> Result<PreparedBatch> {
            try {
                auto batch = loader_->next(batch_);
                if (!batch)
                    return Result<PreparedBatch>::failure(batch.error());
                return Result<PreparedBatch>::success(
                    PreparedBatch{std::move(batch).value(), loader_->state()});
            } catch (const std::exception& e) {
                return Result<PreparedBatch>::failure(ErrorCode::io_error, e.what());
            }
        };
        auto prepared = prepare();
        {
            std::lock_guard lock(mutex_);
            if (stopping_)
                return;
            queue_.push_back(std::move(prepared));
        }
        readable_.notify_one();
    }
}
Result<Batch> PrefetchLoader::next() {
    std::unique_lock lock(mutex_);
    readable_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
    if (queue_.empty())
        return Result<Batch>::failure(ErrorCode::io_error, "Data prefetch stopped");
    auto prepared = std::move(queue_.front());
    queue_.pop_front();
    lock.unlock();
    writable_.notify_one();
    if (!prepared)
        return Result<Batch>::failure(prepared.error());
    consumed_ = std::move(prepared.value().state_after);
    return Result<Batch>::success(std::move(prepared.value().batch));
}
} // namespace mini_mla
