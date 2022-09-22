include("./exp.jl")

using ProgressMeter: Progress, next!

using Unzip
# unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))

"""Opt path"""
opt_filename = (id::Int) -> begin
    opt_path = "./trained/$(t.size())x$(t.size())/opt_$(id).curr"
    ensure_filepath(opt_path)
    return opt_path
end

"""Exp path"""
exp_filename = (id::Int) -> begin
    exp_path = "./trained/$(t.size())x$(t.size())/exp_$(id).curr"
    ensure_filepath(exp_path)
    return exp_path
end

"""Save optimizer"""
function save_optimizer(path::String, opt::Flux.Optimise.AbstractOptimiser)
    open(path, "w") do io
        serialize(io, opt)
    end
end

"""Load optimizer"""
function load_optimizer(path::String)
    open(path, "r") do io
        return deserialize(io)
    end
end

mutable struct Train

    _experiences::Experiences

    # functions
    size::Function
    modelPath::Function
    expPath::Function
    optPath::Function
    train::Function

    # constructor
    function Train()

        t = new()

        """Size"""
        t.size = () -> args["game_size"]

        """Parameters"""
        BATCH_SIZE = args["train_batch_size"]
        BATCH_NUM = args["train_batch_num"]
        TRAIN_EPOCHS = args["train_epochs"]

        function init_training()

            # initialize model
            model_ = Model(args["game_size"], channels=args["model_channels"])
            params_size = sum([length(l) for l in Flux.params(model_._model)])
            @info "Initialize Model [$(model_.size())x$(model_.size()), c=$(model_.channels()), p=$(params_size)]"
            # load model if exists
            model_filename = model_filename(1)
            if isfile(model_filename)
                @info repeat("-", 50)
                @info "Loading Model from [$(model_filename)] ..."
                model_.load(model_filename)
                @info "Loaded Model [$(model_.size())x$(model_.size()), c=$(model_.channels()), p=$(params_size)]" model_._model
            end

            # initialize optimizer
            opt_ = AdamW(args["learning_rate"], (args["adamw_beta1"], args["adamw_beta2"]), args["adamw_weight_decay"])
            @info "Initialize Optimizer [η=$(round(opt_[1].eta, digits=4)), β=$(round.(opt_[1].beta, digits=4)), δ=$(opt_[2].wd)]"
            # load optimizer if exists
            opt_filename = opt_filename(1)
            if isfile(opt_filename)
                @info repeat("-", 50)
                @info "Loading Optimizer from [$(opt_filename)] ..."
                opt_ = load_optimizer(opt_filename)
                @info "Loaded Optimizer [η=$(round(opt_[1].eta, digits=4)), β=$(round.(opt_[1].beta, digits=4)), δ=$(opt_[2].wd)]"
            end
            # move to gpu
            if args["model_cuda"] >= 0
                opt_ = opt_ |> gpu
            end

            # initialize experiences
            t._experiences = Experiences(model_, opt_)
            id_ = t._experiences.elo().newPlayer()
            @info "Initialize Experiences [Player ID = $(id_)]"

            # load experiences if exists
            exp_filename = exp_filename(1)
            if isfile(exp_filename)
                @info repeat("-", 50)
                @info "Loading Experiences from [$(exp_filename)] ..."
                t._experiences.load(exp_filename)
                @info "Loaded Experiences [len=$(t._experiences.length(id)), total=$(t._experiences.totalCount(id)), trained=$(t._experiences.trainedBatch(id))]"
            end

            # start experiences server
            @info repeat("=", 50)
            t._experiences.startServer()

        end

        init_training()

        """Train epoch"""
        function train_epoch(epoch::Int)

            @info "Training epoch [$(epoch)] started [len=$(t._experiences.length()), tot=$(t._experiences.totalCount()), trn=$(t._experiences.trainedBatch())] ..."

            # re-sample each epoch
            (states, pis, vs) = t._experiences.sampleExperience(id, BATCH_SIZE * BATCH_NUM)

            data_loader = Flux.Data.DataLoader((states, pis, vs), batchsize=BATCH_SIZE, shuffle=true)

            progress_tracker = Progress(length(data_loader), 1, "Training epoch $(epoch): ")

            # keeps a list of kl divergence
            kl_list = Vector{Float32}()

            # keeps a list of loss
            loss_list = Vector{Float32}()
            loss_pi_list = Vector{Float32}()
            loss_v_list = Vector{Float32}()
            loss_entropy_list = Vector{Float32}()
            loss_reg_list = Vector{Float32}()

            for (state, pi, v) in data_loader

                if args["model_cuda"] >= 0
                    state = state |> gpu
                    pi = pi |> gpu
                    v = v |> gpu
                end

                # @info "sizes" size(state) size(pi) size(v)

                # calculate previous policy
                prev_pi, _ = t._experiences.model(id).forward(state)

                # convert from visit count to distribution
                pa = softmax(log.(pi .+ 1e-8), dims=[1, 2])

                params = t._experiences.model().params()
                loss_tuple, back = pullback(params) do
                    t._experiences.model(id).loss(state, pa, v)  # [IMPORTANT] use normalized distribution pa !!
                end

                grads = back((1.0, 0.0, 0.0))

                Flux.update!(t._experiences.opt(), params, grads)

                # get loss components
                new_loss, loss_pi, loss_v, loss_entropy, new_pi, new_v = loss_tuple

                if !args["exp_sample_sequential"]
                    t._experiences.addtrainedBatch(id, 1)
                end

                # keep track of loss
                push!(loss_list, new_loss |> cpu)
                push!(loss_pi_list, loss_pi |> cpu)
                push!(loss_v_list, loss_v |> cpu)
                push!(loss_entropy_list, loss_entropy |> cpu)
                push!(loss_reg_list, loss_reg |> cpu)

                next!(progress_tracker; showvalues=[
                    (:loss, round(loss_avg, digits=2)),
                    (:curr, round(loss_curr, digits=2)),
                    (:recon, round(loss_recon, digits=2)),
                    (:kl, round(loss_kl, digits=2)),
                    (:mu, round.(mu, digits=2)),
                    (:sigma, round.(sigma, digits=2)),
                ])

                # calculate KL divergence
                kl_sum = sum(prev_pi .* (log.(prev_pi) .- log.(new_pi)), dims=[1, 2])
                kl = reshape(kl_sum, BATCH_SIZE)
                kl_mean = mean(kl)

                # update global list
                kl_list = vcat(kl_list, kl |> cpu)

                # check if kl divergence is too high, break early if so
                if kl_mean > 4 * args["train_kl_target"]
                    @info "KL divergence [$(round(kl_mean, digits=4)) > 4 * $(args["train_kl_target"])] ... stopping early ."
                    break
                end

            end

            loss_epoch = round(mean(loss_list), digits=3)
            pi_epoch = round(mean(loss_pi_list), digits=2)
            v_epoch = round(mean(loss_v_list), digits=3)
            entropy_epoch = round(mean(loss_entropy_list), digits=2)
            reg_epoch = round(mean(loss_reg_list), digits=4)

            @info "Training epoch [$(epoch)] [$(loss_epoch),L = $(pi_epoch),π + $(v_epoch),ν - $(args["model_loss_coef_entropy"]) × $(entropy_epoch),H + $(args["model_loss_coef_theta"]) × $(reg_epoch),θ]"

            # check for kl divergence
            kl_epoch_mean = mean(kl_list)

            if kl_epoch_mean > args["train_kl_target"] * 2.0
                new_learning_rate = max(t._experiences.opt(id)[1].eta / 1.5, args["learning_rate"] / args["learning_rate_range"])
                @info "KL divergence [$(round(kl_epoch_mean, digits=4)) > $(args["train_kl_target"]) × 2.0] ... reducing learning rate to [$(round(new_learning_rate, digits=4))] ."
                # update learning rate
                t._experiences.opt()[1].eta = new_learning_rate
            elseif kl_epoch_mean < args["train_kl_target"] / 2.0
                new_learning_rate = min(t._experiences.opt(id)[1].eta * 1.5, args["learning_rate"] * args["learning_rate_range"])
                @info "KL divergence [$(round(kl_epoch_mean, digits=4)) < $(args["train_kl_target"]) ÷ 2.0] ... increasing learning rate to [$(round(new_learning_rate, digits=4))] ."
                # update learning rate
                t._experiences.opt()[1].eta = new_learning_rate
            else
                learning_rate = t._experiences.opt(id)[1].eta
                @info "KL divergence [$(round(kl_epoch_mean, digits=4)) within range] ... keeping learning rate as [$(round(learning_rate, digits=4))] ."
            end

        end

        """Save trained model, optimizer, and experiences"""
        function save_trained()

            # print separator
            @info repeat("-", 50)

            # save model
            model_ = t._experiences.model
            params_size = sum([length(l) for l in Flux.params(model_._model)])
            model_filename = model_filename(1)
            @info "Save model [$(model_filename)] : [$(model_.size())x$(model_.size()), c=$(model_.channels()), p=$(params_size)]"
            backup_file(model_filename)
            model_.save(model_filename)

            # save optimizer
            opt_ = t._experiences.opt()
            opt_filename = opt_filename(1)
            @info "Save optimizer [$(opt_filename)] : [η=$(round(opt_[1].eta, digits=4)), β=$(round.(opt_[1].beta, digits=4)), δ=$(opt_[2].wd)]"
            backup_file(opt_filename)
            save_optimizer(opt_filename, opt_)

            # save experiences
            exp_filename = exp_filename(1)
            @info "Save experiences [$(exp_filename)] : [len=$(t._experiences.length(id)), tot=$(t._experiences.totalCount(id)), trn=$(t._experiences.trainedBatch(id))]"
            backup_file(exp_filename)
            t._experiences.save(id, exp_filename)

            # print separator
            @info repeat("-", 50)

        end

        """Train"""
        t.train = () -> begin

            sleep_time = 15.0 # initialize to 15 seconds
            sleep_count = 0
            sleep_count_ema = 5.0 # initialize to target of 5.0

            while true

                try
                    # check trained batch count, exit if exceeded maximum
                    if t._experiences.trainedBatch(id) > args["exp_trained_batch"]
                        exit(0)
                    end

                    if (t._experiences.trainedBatch(id) * args["train_trim_ratio"] + BATCH_NUM * TRAIN_EPOCHS) * BATCH_SIZE < t._experiences.totalCount(id) * TRAIN_EPOCHS

                        sleep_count_ema = 0.9 * sleep_count_ema + 0.1 * sleep_count
                        sleep_time = max(1.0, min(120.0, (sleep_time * sleep_count_ema) / 5.0))

                        sleep_count = 0

                        # print separator
                        @info repeat("-", 50)

                        for epoch in 1:TRAIN_EPOCHS
                            try
                                train_epoch(epoch)
                            catch e
                                @error "Error" exception = (e, catch_backtrace())
                            finally
                                if args["exp_sample_sequential"]
                                    # always add BATCH_NUM to trainedBatch
                                    t._experiences.addtrainedBatch(id, BATCH_NUM)
                                end
                                # GC & reclaim CUDA memory
                                GC.gc(true)
                                if args["model_cuda"] >= 0
                                    CUDA.reclaim()
                                end
                            end
                        end

                        if args["exp_sample_sequential"]
                            # trim experiences by BATCH_SIZE * BATCH_NUM
                            t._experiences.trimExperience(id, trunc(Int, BATCH_SIZE * BATCH_NUM * args["train_trim_ratio"]))
                        end

                        # save trained model, optimizer, and experiences
                        save_trained()

                    else

                        @info "Training is waiting for more experiences [len=$(t._experiences.length()), tot=$(t._experiences.totalCount()), trn=$(t._experiences.trainedBatch())] ..."

                    end

                catch e

                    @error "Error" exception = (e, catch_backtrace())

                finally

                    # GC & reclaim CUDA memory
                    GC.gc(true)
                    if args["model_cuda"] >= 0
                        CUDA.reclaim()
                    end

                    sleep_count += 1

                    sleep(sleep_time)
                end

            end

        end

        return t

    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    @info repeat("=", 50)
    # train model[1] and model[2]
    train = Train()
    train.train()
end
