#include "mini_mla/server.hpp"

#include "mini_mla/workflow.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <poll.h>
#include <sstream>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace mini_mla {
namespace {
class Socket {
  public:
    explicit Socket(int fd = -1) : fd_(fd) {}
    ~Socket() {
        if (fd_ >= 0)
            ::close(fd_);
    }
    Socket(Socket&& s) noexcept : fd_(std::exchange(s.fd_, -1)) {}
    Socket& operator=(Socket&& s) noexcept {
        if (this != &s) {
            if (fd_ >= 0)
                ::close(fd_);
            fd_ = std::exchange(s.fd_, -1);
        }
        return *this;
    }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    [[nodiscard]] int get() const { return fd_; }

  private:
    int fd_;
};
bool send_all(int fd, std::string_view data) {
    while (!data.empty()) {
        auto n = ::send(fd, data.data(), data.size(), 0);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            return false;
        data.remove_prefix(static_cast<std::size_t>(n));
    }
    return true;
}
bool connected(int fd) {
    char byte;
    auto n = ::recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
    return n > 0 || (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR));
}
std::string reason(int status) {
    switch (status) {
    case 200:
        return "OK";
    case 400:
        return "Bad Request";
    case 404:
        return "Not Found";
    case 408:
        return "Request Timeout";
    case 413:
        return "Content Too Large";
    case 429:
        return "Too Many Requests";
    case 503:
        return "Service Unavailable";
    default:
        return "Internal Server Error";
    }
}
bool response(int fd, int status, std::string_view body,
              std::string_view type = "application/json") {
    return send_all(fd, "HTTP/1.1 " + std::to_string(status) + " " + reason(status) +
                            "\r\nContent-Type: " + std::string(type) + "\r\nContent-Length: " +
                            std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n") &&
           send_all(fd, body);
}
void error_response(int fd, int status, std::string_view message) {
    response(fd, status,
             Json(Json::Object{{"error", Json::Object{{"message", std::string(message)},
                                                      {"type", "request_error"}}}})
                 .dump());
}
struct Request {
    std::string method, path, body;
};
struct QueuedClient {
    Socket socket;
    std::chrono::steady_clock::time_point accepted = std::chrono::steady_clock::now();
};
Request read_request(int fd, std::size_t max_bytes) {
    std::string bytes;
    char buffer[4096];
    std::size_t end;
    while ((end = bytes.find("\r\n\r\n")) == std::string::npos) {
        if (bytes.size() >= 16384)
            throw std::invalid_argument("Request headers exceed 16 KiB");
        auto n = ::recv(fd, buffer, sizeof(buffer), 0);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            throw std::runtime_error("Client disconnected or timed out while sending headers");
        bytes.append(buffer, static_cast<std::size_t>(n));
    }
    if (end > 16384)
        throw std::invalid_argument("Request headers exceed 16 KiB");
    std::istringstream header(bytes.substr(0, end));
    Request r;
    std::string line, version, extra;
    std::getline(header, line);
    std::istringstream first(line);
    first >> r.method >> r.path >> version;
    if ((version != "HTTP/1.1" && version != "HTTP/1.0") || r.path.empty() || (first >> extra))
        throw std::invalid_argument("Invalid request line");
    std::size_t length = 0;
    bool saw_length = false;
    while (std::getline(header, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        auto colon = line.find(':');
        if (colon == std::string::npos)
            throw std::invalid_argument("Invalid HTTP header");
        auto key = line.substr(0, colon);
        for (auto& c : key)
            if (c >= 'A' && c <= 'Z')
                c = static_cast<char>(c + 'a' - 'A');
        auto value = std::string_view(line).substr(colon + 1);
        while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
            value.remove_prefix(1);
        if (key == "transfer-encoding")
            throw std::invalid_argument("Chunked request bodies are unsupported");
        if (key == "content-length") {
            if (saw_length)
                throw std::invalid_argument("Duplicate Content-Length");
            saw_length = true;
            auto parsed = std::from_chars(value.data(), value.data() + value.size(), length);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
                length > max_bytes)
                throw std::invalid_argument("Invalid or oversized request body");
        }
    }
    if (r.method == "POST" && !saw_length)
        throw std::invalid_argument("POST needs Content-Length");
    std::size_t start = end + 4;
    while (bytes.size() - start < length) {
        auto want = std::min(sizeof(buffer), length - (bytes.size() - start));
        auto n = ::recv(fd, buffer, want, 0);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            throw std::runtime_error("Client disconnected or timed out while sending body");
        bytes.append(buffer, static_cast<std::size_t>(n));
    }
    r.body = bytes.substr(start, length);
    if (!valid_utf8(r.body))
        throw std::invalid_argument("Request body must be valid UTF-8");
    return r;
}
class Server {
  public:
    Server(InferenceEngine& engine, ServerConfig config)
        : engine_(engine), config_(std::move(config)) {}
    ~Server() { stop(); }
    void run(const std::function<bool()>& external_stop) {
        if (config_.port < 0 || config_.port > 65535 || config_.max_queue < 1 ||
            config_.workers < 1 || config_.workers > 64 || config_.request_timeout_seconds < 1 ||
            config_.max_context < 1 || config_.max_context > engine_.config().max_sequence_length ||
            config_.max_new_tokens < 1 || config_.max_request_bytes < 1)
            throw std::invalid_argument("Invalid server limits");
        Socket listening(::socket(AF_INET, SOCK_STREAM, 0));
        if (listening.get() < 0)
            throw std::runtime_error(std::strerror(errno));
        int yes = 1;
        ::setsockopt(listening.get(), SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<std::uint16_t>(config_.port));
        if (::inet_pton(AF_INET, config_.host.c_str(), &address.sin_addr) != 1)
            throw std::invalid_argument("Host must be an IPv4 address");
        if (::bind(listening.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) ||
            ::listen(listening.get(), config_.max_queue))
            throw std::runtime_error(std::strerror(errno));
        socklen_t size = sizeof(address);
        if (::getsockname(listening.get(), reinterpret_cast<sockaddr*>(&address), &size))
            throw std::runtime_error(std::strerror(errno));
        for (int i = 0; i < config_.workers; ++i)
            sessions_.push_back(take(engine_.fork()));
        for (auto& session : sessions_)
            workers_.emplace_back([this, engine = session.get()] { worker(*engine); });
        std::printf("Listening on http://%s:%u\n", config_.host.c_str(), ntohs(address.sin_port));
        std::fflush(stdout);
        while (!external_stop || !external_stop()) {
            pollfd descriptor{listening.get(), POLLIN, 0};
            auto ready = ::poll(&descriptor, 1, 100);
            if (ready < 0 && errno == EINTR)
                continue;
            if (ready < 0)
                throw std::runtime_error(std::strerror(errno));
            if (!ready)
                continue;
            Socket client(::accept(listening.get(), nullptr, nullptr));
            if (client.get() < 0) {
                if (errno == EINTR)
                    continue;
                throw std::runtime_error(std::strerror(errno));
            }
            timeval timeout{std::min(config_.request_timeout_seconds, 10), 0};
            ::setsockopt(client.get(), SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            ::setsockopt(client.get(), SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            ::setsockopt(client.get(), SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
            {
                std::lock_guard lock(queue_mutex_);
                if (queue_.size() >= static_cast<std::size_t>(config_.max_queue)) {
                    ++rejected_;
                    error_response(client.get(), 429, "Request queue is full");
                    continue;
                }
                queue_.push_back(QueuedClient{std::move(client)});
            }
            available_.notify_one();
        }
        stop();
    }

  private:
    InferenceEngine& engine_;
    ServerConfig config_;
    std::atomic<bool> stopping_{false};
    std::mutex queue_mutex_;
    std::vector<std::unique_ptr<InferenceEngine>> sessions_;
    std::condition_variable available_;
    std::deque<QueuedClient> queue_;
    std::vector<std::thread> workers_;
    std::atomic<std::uint64_t> requests_{0}, completed_{0}, rejected_{0}, errors_{0}, generated_{0};
    std::atomic<double> queue_seconds_{0}, prefill_seconds_{0}, decode_seconds_{0},
        first_token_seconds_{0};
    void stop() {
        stopping_ = true;
        available_.notify_all();
        for (auto& t : workers_)
            if (t.joinable())
                t.join();
        workers_.clear();
    }
    void worker(InferenceEngine& session) {
        for (;;) {
            QueuedClient client;
            {
                std::unique_lock lock(queue_mutex_);
                available_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (stopping_)
                    return;
                client = std::move(queue_.front());
                queue_.pop_front();
            }
            handle(client.socket.get(), client.accepted, session);
        }
    }
    void handle(int fd, std::chrono::steady_clock::time_point accepted,
                InferenceEngine& session) noexcept {
        bool streaming_started = false;
        try {
            auto request = read_request(fd, config_.max_request_bytes);
            auto request_id = requests_.fetch_add(1) + 1;
            if (request.method == "GET" && request.path == "/healthz") {
                response(fd, 200, "{\"status\":\"ok\"}");
                return;
            }
            if (request.method == "GET" && request.path == "/v1/models") {
                response(
                    fd, 200,
                    "{\"object\":\"list\",\"data\":[{\"id\":\"mini_mla\",\"object\":\"model\"}]}");
                return;
            }
            if (request.method == "GET" && request.path == "/metrics") {
                std::string text =
                    "mini_mla_requests_total " + std::to_string(requests_.load()) +
                    "\nmini_mla_completed_total " + std::to_string(completed_.load()) +
                    "\nmini_mla_rejected_total " + std::to_string(rejected_.load()) +
                    "\nmini_mla_errors_total " + std::to_string(errors_.load()) +
                    "\nmini_mla_generated_tokens_total " + std::to_string(generated_.load()) + "\n";
                text += "mini_mla_queue_seconds_sum " + std::to_string(queue_seconds_.load()) +
                        "\nmini_mla_prefill_seconds_sum " +
                        std::to_string(prefill_seconds_.load()) + "\nmini_mla_decode_seconds_sum " +
                        std::to_string(decode_seconds_.load()) +
                        "\nmini_mla_first_token_seconds_sum " +
                        std::to_string(first_token_seconds_.load()) + "\n";
                {
                    std::lock_guard lock(queue_mutex_);
                    text += "mini_mla_queue_depth " + std::to_string(queue_.size()) + "\n";
                }
                response(fd, 200, text, "text/plain; version=0.0.4");
                return;
            }
            bool chat = request.path == "/v1/chat/completions";
            if (request.method != "POST" || (!chat && request.path != "/v1/completions")) {
                error_response(fd, 404, "Unknown endpoint");
                return;
            }
            auto j = take(Json::parse(request.body));
            std::vector<std::uint16_t> prompt;
            if (chat)
                prompt = take(engine_.tokenizer().chat_prompt(j.at("messages").array()));
            else {
                prompt = take(engine_.tokenizer().encode(j.at("prompt").string()));
                prompt.insert(prompt.begin(), engine_.tokenizer().special("<|bos|>"));
            }
            if (j.num_or("n", 1) != 1)
                throw std::invalid_argument("Only n=1 is supported");
            GenerationConfig c;
            c.max_new_tokens =
                j.contains("max_tokens")
                    ? j.at("max_tokens").integer()
                    : std::min(config_.max_new_tokens,
                               config_.max_context - static_cast<int>(prompt.size()));
            c.temperature = static_cast<float>(j.num_or("temperature", c.temperature));
            c.top_p = static_cast<float>(j.num_or("top_p", c.top_p));
            c.top_k = j.contains("top_k") ? j.at("top_k").integer() : 0;
            c.repetition_penalty = static_cast<float>(j.num_or("repetition_penalty", 1));
            if (j.contains("seed"))
                c.seed = j.at("seed").size_integer();
            if (j.contains("stop")) {
                if (j.at("stop").is_string())
                    c.stop.push_back(j.at("stop").string());
                else
                    for (const auto& x : j.at("stop").array())
                        c.stop.push_back(x.string());
            }
            check(c.validate());
            if (c.max_new_tokens > config_.max_new_tokens ||
                prompt.size() + static_cast<std::size_t>(c.max_new_tokens) >
                    static_cast<std::size_t>(config_.max_context))
                throw std::invalid_argument("Request exceeds context or token limit");
            bool stream = j.bool_or("stream", false);
            auto deadline = accepted + std::chrono::seconds(config_.request_timeout_seconds);
            if (std::chrono::steady_clock::now() >= deadline) {
                error_response(fd, 503, "Request queue deadline exceeded");
                return;
            }
            if (!connected(fd) || stopping_)
                return;
            double queue_delay =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - accepted).count();
            std::string id = "cmpl-" + std::to_string(request_id);
            if (stream) {
                streaming_started = send_all(
                    fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: "
                        "no-cache\r\nConnection: close\r\n\r\n");
                if (!streaming_started)
                    return;
            }
            auto chunk = [&](std::string_view text, const Json& finish) {
                Json choice = Json::Object{{"index", 0}, {"finish_reason", finish}};
                if (chat)
                    choice["delta"] = Json::Object{{"content", std::string(text)}};
                else
                    choice["text"] = std::string(text);
                return send_all(fd,
                                "data: " +
                                    Json(Json::Object{{"id", id},
                                                      {"model", "mini_mla"},
                                                      {"object", chat ? "chat.completion.chunk"
                                                                      : "text_completion"},
                                                      {"choices", Json::Array{std::move(choice)}}})
                                        .dump() +
                                    "\n\n");
            };
            auto result = take(session.generate_tokens(prompt, c, [&](std::string_view text) {
                return !stopping_ && std::chrono::steady_clock::now() < deadline && connected(fd) &&
                       (!stream || text.empty() || chunk(text, Json()));
            }));
            generated_ += result.tokens.size();
            if (result.finish_reason == "cancelled") {
                if (!stream && connected(fd))
                    error_response(fd, 408, "Generation timed out or was cancelled");
                return;
            }
            if (stream) {
                if (chunk({}, result.finish_reason))
                    send_all(fd, "data: [DONE]\n\n");
            } else {
                Json choice = Json::Object{{"index", 0}, {"finish_reason", result.finish_reason}};
                if (chat)
                    choice["message"] =
                        Json::Object{{"role", "assistant"}, {"content", result.text}};
                else
                    choice["text"] = result.text;
                response(
                    fd, 200,
                    Json(Json::Object{
                             {"id", id},
                             {"model", "mini_mla"},
                             {"object", chat ? "chat.completion" : "text_completion"},
                             {"choices", Json::Array{std::move(choice)}},
                             {"usage", Json::Object{{"prompt_tokens", result.prompt_tokens},
                                                    {"completion_tokens", result.tokens.size()},
                                                    {"total_tokens", result.prompt_tokens +
                                                                         result.tokens.size()}}}})
                        .dump());
            }
            ++completed_;
            queue_seconds_.fetch_add(queue_delay);
            prefill_seconds_.fetch_add(result.prefill_ms / 1000);
            decode_seconds_.fetch_add(result.decode_ms / 1000);
            first_token_seconds_.fetch_add(queue_delay + result.time_to_first_token_ms / 1000);
        } catch (const std::exception& e) {
            ++errors_;
            if (!streaming_started)
                error_response(fd, 400, e.what());
            else
                send_all(fd, "data: " + Json(Json::Object{{"error", e.what()}}).dump() + "\n\n");
        } catch (...) {
            ++errors_;
            if (!streaming_started)
                error_response(fd, 500, "Server failure");
        }
    }
};
} // namespace
Status serve(InferenceEngine& engine, const ServerConfig& config, std::function<bool()> stop) {
    try {
        Server server(engine, config);
        server.run(stop);
        return ok();
    } catch (const std::exception& e) {
        return Status::failure(ErrorCode::io_error, e.what());
    }
}
} // namespace mini_mla
