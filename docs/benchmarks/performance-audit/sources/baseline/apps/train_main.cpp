#include "app_utils.hpp"
#include "mini_mla/trainer.hpp"

#include <csignal>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void on_interrupt(int) {
    interrupted = 1;
}
} // namespace
int main(int argc, char** argv) {
    return mini_mla::app_main(argc, argv, [](const mini_mla::Args& a) {
        a.known({"config",
                 "train",
                 "validation",
                 "tokenizer",
                 "output",
                 "resume",
                 "initialize",
                 "reference",
                 "mode",
                 "backend",
                 "precision",
                 "attention",
                 "submission-window",
                 "sequence-length",
                 "microbatch",
                 "grad-accum",
                 "steps",
                 "warmup-steps",
                 "seed",
                 "learning-rate",
                 "min-lr-fraction",
                 "log-every",
                 "checkpoint-every",
                 "validate-every",
                 "validation-batches",
                 "optimizer",
                 "muon-momentum",
                 "muon-ns-steps",
                 "muon-nesterov",
                 "muon-layout",
                 "weight-decay",
                 "grad-clip",
                 "dpo-beta",
                 "group-size",
                 "completion-tokens",
                 "gradient-epochs",
                 "clip-epsilon",
                 "kl-coefficient",
                 "reward",
                 "help"});
        if (a.has("help")) {
            std::puts("mini_mla_train --config MODEL_OR_RUN.json --tokenizer tokenizer.json "
                      "--train FILES --output DIR\n  --backend cpu|mps --precision fp32|fp16 "
                      "--mode pretrain|sft|dpo|grpo\n  --steps 1000 --sequence-length 256 "
                      "--attention auto|reference|flash --submission-window 4\n  "
                      "--microbatch 2 --grad-accum 16\n  --learning-rate 0.0003 --warmup-steps 20 "
                      "--checkpoint-every 100\nmini_mla_train --resume DIR/checkpoint-latest");
            std::puts(
                "  --optimizer adamw|muon (Muon on hidden matrices, AdamW on embeddings/norms)\n"
                "  --muon-momentum 0.95 --muon-ns-steps 5 --muon-nesterov true\n"
                "  --muon-layout batched|individual (MPSGraph Metal execution)");
            return 0;
        }
        mini_mla::TrainOptions o;
        if (a.has("resume")) {
            auto state = mini_mla::take(
                mini_mla::read_json(std::filesystem::path(a.get("resume")) / "trainer_state.json"));
            o = mini_mla::take(mini_mla::TrainOptions::from_json(state.at("options")));
            o.resume = a.get("resume");
        } else if (a.has("config")) {
            auto config = mini_mla::take(mini_mla::read_json(a.get("config")));
            if (config.contains("model_config")) {
                o = mini_mla::take(mini_mla::TrainOptions::from_json(config));
                if (!config.contains("attention"))
                    o.attention = "auto";
                if (!config.contains("submission_window"))
                    o.submission_window = 4;
                if (!config.contains("muon_layout"))
                    o.muon_layout = "batched";
            } else
                o.model_config = a.get("config");
        }
        if (a.has("train"))
            o.train = a.paths("train");
        if (a.has("validation"))
            o.validation = a.paths("validation");
        o.tokenizer = a.get("tokenizer", o.tokenizer.string());
        o.output = a.get("output", o.output.string());
        o.initialize = a.get("initialize", o.initialize.string());
        o.reference = a.get("reference", o.reference.string());
        o.mode = a.get("mode", o.mode);
        o.backend = a.get("backend", o.backend);
        o.precision = a.get("precision", o.precision);
        o.attention = a.get("attention", o.attention);
        o.submission_window = a.integer("submission-window", o.submission_window);
        o.muon_layout = a.get("muon-layout", o.muon_layout);
        o.reward = a.get("reward", o.reward);
        o.sequence = a.integer("sequence-length", o.sequence);
        o.microbatch = a.integer("microbatch", o.microbatch);
        o.accumulation = a.integer("grad-accum", o.accumulation);
        o.steps = a.u64("steps", o.steps);
        o.warmup = a.u64("warmup-steps", o.warmup);
        o.seed = a.u64("seed", o.seed);
        o.learning_rate = a.real("learning-rate", o.learning_rate);
        o.minimum_lr_fraction = a.real("min-lr-fraction", o.minimum_lr_fraction);
        o.log_every = a.integer("log-every", o.log_every);
        o.checkpoint_every = a.integer("checkpoint-every", o.checkpoint_every);
        o.validate_every = a.integer("validate-every", o.validate_every);
        o.validation_batches = a.integer("validation-batches", o.validation_batches);
        o.optimizer = optimizer_options(a, o.optimizer);
        o.dpo_beta = a.real("dpo-beta", o.dpo_beta);
        o.group_size = a.integer("group-size", o.group_size);
        o.completion_tokens = a.integer("completion-tokens", o.completion_tokens);
        o.gradient_epochs = a.integer("gradient-epochs", o.gradient_epochs);
        o.clip_epsilon = a.real("clip-epsilon", o.clip_epsilon);
        o.kl_coefficient = a.real("kl-coefficient", o.kl_coefficient);
        std::signal(SIGINT, on_interrupt);
        std::signal(SIGTERM, on_interrupt);
        mini_mla::check(mini_mla::train_model(std::move(o), [] { return interrupted != 0; }));
        return 0;
    });
}
