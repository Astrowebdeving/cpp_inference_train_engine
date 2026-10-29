#pragma once
#include "mini_mla/optimizer.hpp"
#include "mini_mla/tokenizer.hpp"

namespace mini_mla {
struct LoadedCheckpoint {
    std::unique_ptr<Model> model;
    std::unique_ptr<Optimizer> optimizer;
    Tokenizer tokenizer;
    Json trainer_state;
};
[[nodiscard]] Status save_checkpoint(const std::filesystem::path&, const Model&, const Optimizer&,
                                     const Json& trainer_state,
                                     const std::filesystem::path& tokenizer_path);
[[nodiscard]] Status export_model(const std::filesystem::path&, const Model&,
                                  const std::filesystem::path& tokenizer_path,
                                  std::string_view dtype = "f16");
[[nodiscard]] Result<LoadedCheckpoint> load_checkpoint(const std::filesystem::path&,
                                                       bool training = false);
[[nodiscard]] Status update_latest_link(const std::filesystem::path& link,
                                        const std::filesystem::path& target);
} // namespace mini_mla
