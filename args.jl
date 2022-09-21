using ArgParse
using CUDA

function parse_commandline()

    s = ArgParseSettings()
    @add_arg_table s begin

        "--game_size"
        help = "game size"
        arg_type = Int
        default = 11

        "--game_display"
        help = "display game"
        arg_type = Bool
        default = true

        "--model_channels"
        help = "model channels"
        arg_type = Int
        default = 32

        "--model_cuda"
        help = "model cuda number"
        arg_type = Int
        default = -1

        "--model_loss_coef_entropy"
        help = "model loss entropy coefficient"
        arg_type = Float32
        default = Float32(0.6)

        "--model_loss_coef_theta"
        help = "model loss theta coefficient"
        arg_type = Float32
        default = Float32(1.0)

        "--mcts_n_multiplier"
        help = "mcts play count multiplier"
        arg_type = Int
        default = 2

        "--mcts_cpuct"
        help = "mcts cpuct"
        arg_type = Float32
        default = Float32(3.0)

        "--mcts_depth"
        help = "mcts depth"
        arg_type = Int
        default = 16

        "--mcts_gamma"
        help = "mcts gamma"
        arg_type = Float32
        default = Float32(0.99)

        "--mcts_temperature"
        help = "mcts temperature"
        arg_type = Float32
        default = Float32(1.0)

        "--mcts_noise_alpha"
        help = "mcts noise dirichlet alpha"
        arg_type = Float32
        default = Float32(0.1) # AlphaZero: 0.03

        "--mcts_noise_epsilon"
        help = "mcts noise epsilon"
        arg_type = Float32
        default = Float32(0.25) # AlphaZero: 0.25

        "--train_batch_size"
        help = "train batch size"
        arg_type = Int
        default = 1024 # 256, 1_024, 4_096, 16_384

        "--train_batch_num"
        help = "train batch number"
        arg_type = Int
        default = 8 # 1, 2, 4, 8, 16, 32, 64

        "--train_epochs"
        help = "train epochs"
        arg_type = Int
        default = 5

        "--train_trim_ratio"
        help = "train trim ratio"
        arg_type = Float32
        default = Float32(0.4)

        "--train_kl_target"
        help = "train kl target"
        arg_type = Float32
        default = Float32(0.005)

        "--learning_rate"
        help = "learning rate"
        arg_type = Float32
        default = Float32(0.001)

        "--learning_rate_range"
        help = "learning rate range"
        arg_type = Float32
        default = Float32(5.0)

        "--adamw_beta1"
        help = "adamw beta1"
        arg_type = Float32
        default = Float32(0.9)

        "--adamw_beta2"
        help = "adamw beta2"
        arg_type = Float32
        default = Float32(0.999)

        "--adamw_weight_decay"
        help = "adamw weight decay"
        arg_type = Float32
        default = Float32(0.0002)

        "--exp_min"
        help = "experience min"
        arg_type = Int
        default = 20_000

        "--exp_max"
        help = "experience max"
        arg_type = Int
        default = 100_000

        "--exp_trained_batch"
        help = "maximum trained batch count"
        arg_type = Int
        default = 10_000

        "--exp_preserve_ratio"
        help = "experience preserve ratio"
        arg_type = Float32
        default = Float32(0.5)

        "--exp_sample_sequential"
        help = "experience sample sequentially"
        arg_type = Bool
        default = true

        "--exp_server"
        help = "experience server"
        arg_type = String
        default = "localhost"

        "--exp_port"
        help = "experience port"
        arg_type = Int
        default = 5555

        "--infer_playouts"
        help = "infer playout count"
        arg_type = Int
        default = 20

        "--eval_playouts"
        help = "eval playout count"
        arg_type = Int
        default = 12

        "--player_1"
        help = "player 1"
        arg_type = String
        range_tester = (x -> x ∈ ["server", "curr", "best", "human"])
        default = "server"

        "--player_2"
        help = "player 2"
        arg_type = String
        range_tester = (x -> x ∈ ["server", "curr", "best", "human"])
        default = "best"

        "--population_size"
        help = "population size"
        arg_type = Int
        range_tester = (x -> x >= 1)
        default = 1

        "--population_allow_self_play"
        help = "population allow self play"
        arg_type = Bool
        default = true

    end

    return parse_args(s)

end

args = parse_commandline()

if args["model_cuda"] >= 0
    CUDA.allowscalar(false)
    CUDA.device!(args["model_cuda"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    @info args
    @info args["game_size"]
    @info args["train_batch_size"]
end
