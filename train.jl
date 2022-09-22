include("./exp.jl")

using ProgressMeter: Progress, next!, finish!
using Zygote: pullback

using Unzip
# unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))

"""Opt path"""
opt_filename = (id::Int; size = args["game_size"]) -> begin
    return "./trained/$(size)x$(size)/opt_$(id).curr"
end

"""Exp path"""
exp_filename = (id::Int; size = args["game_size"]) -> begin
    return "./trained/$(size)x$(size)/exp_$(id).curr"
end

"""Save optimizer"""
function save_optimizer(path::String, opt::Flux.Optimise.AbstractOptimiser)
    backup_file(path)
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

    _exps::Experiences

    # functions
    size::Function
    modelPath::Function
    expPath::Function
    optPath::Function
    train::Function
    elo::Function

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
            model_path = model_filename(1)
            if isfile(model_path)
                @info repeat("-", 50)
                @info "Loading Model from [$(model_path)] ..."
                model_.load(model_path)
                @info "Loaded Model [$(model_.size())x$(model_.size()), c=$(model_.channels()), p=$(params_size)]" model_._model
            end

            # initialize optimizer
            opt_ = AdamW(args["learning_rate"], (args["adamw_beta1"], args["adamw_beta2"]), args["adamw_weight_decay"])
            @info "Initialize Optimizer [η=$(round(opt_[1].eta, digits=4)), β=$(round.(opt_[1].beta, digits=4)), δ=$(opt_[2].wd)]"
            # load optimizer if exists
            opt_path = opt_filename(1)
            if isfile(opt_path)
                @info repeat("-", 50)
                @info "Loading Optimizer from [$(opt_path)] ..."
                opt_ = load_optimizer(opt_path)
                @info "Loaded Optimizer [η=$(round(opt_[1].eta, digits=4)), β=$(round.(opt_[1].beta, digits=4)), δ=$(opt_[2].wd)]"
            end
            # move to gpu
            if args["model_cuda"] >= 0
                opt_ = opt_ |> gpu
            end

            # initialize elo
            elo_ = Elo()
            id_ = elo_.newPlayer()
            elo_.makeActive(id_)

            # initialize experiences
            t._exps = Experiences(elo_, model_, opt_)
            @info "Initialize Experiences [Player ID = $(id_)]"

            # load experiences if exists
            exp_path = exp_filename(1)
            if isfile(exp_path)
                @info repeat("-", 50)
                @info "Loading Experiences from [$(exp_path)] ..."
                t._exps.load(exp_path)
                @info "Loaded Experiences [len=$(t._exps.length()), total=$(t._exps.totalCount()), trained=$(t._exps.trainedBatch())]"
            end

            # start experiences server
            @info repeat("=", 50)
            t._exps.startServer()

        end

        init_training()

        """Train epoch"""
        function train_epoch(epoch::Int)

            @info "Training epoch [$(epoch)] started [len=$(t._exps.length()), tot=$(t._exps.totalCount()), trn=$(t._exps.trainedBatch())] ..."

            # re-sample each epoch
            (states, pis, vs) = t._exps.sampleExperience(BATCH_SIZE * BATCH_NUM)

            data_loader = Flux.Data.DataLoader((states, pis, vs), batchsize=BATCH_SIZE, shuffle=true)

            progress_tracker = Progress(length(data_loader), dt=0.2, desc="Training epoch $(epoch): ")

            # keeps a list of kl divergence
            kl_list = Vector{Float32}()

            # keeps a list of loss
            loss_list = Vector{Float32}()
            loss_pi_list = Vector{Float32}()
            loss_v_list = Vector{Float32}()
            loss_entropy_list = Vector{Float32}()

            msg = ""

            # params
            params = t._exps.model().params()

            for (state, pi, v) in data_loader

                if args["model_cuda"] >= 0
                    state = state |> gpu
                    pi = pi |> gpu
                    v = v |> gpu
                end

                # convert from visit count to distribution
                pa = softmax(log.(pi .+ 1e-8), dims=[1, 2])

                loss_tuple, back = pullback(params) do
                    t._exps.model().loss(state, pa, v) # [IMPORTANT] use normalized distribution pa !!
                end

                # calculate gradients
                grads = back((1.0f0, nothing, nothing, nothing, nothing, nothing))

                # get loss components
                batch_loss, loss_pi, loss_v, loss_entropy, prev_pi, _ = loss_tuple

                # train
                Flux.update!(t._exps.opt(), params, grads)

                if !args["exp_sample_sequential"]
                    t._exps.addtrainedBatch(1)
                end

                # keep track of loss
                push!(loss_list, batch_loss |> cpu)
                push!(loss_pi_list, loss_pi |> cpu)
                push!(loss_v_list, loss_v |> cpu)
                push!(loss_entropy_list, loss_entropy |> cpu)

                loss_avg = round(mean(loss_list), digits=3)
                loss_pi_avg = round(mean(loss_pi_list), digits=2)
                loss_v_avg = round(mean(loss_v_list), digits=3)
                loss_entropy_avg = round(mean(loss_entropy_list), digits=2)

                # calculate new policy after update
                new_pi, _ = t._exps.model().forward(state)

                # calculate KL divergence
                kl_sum = sum(new_pi .* (log.(new_pi) .- log.(prev_pi)), dims=[1, 2])
                kl_batch_mean = round(mean(kl_sum), digits=4)
                push!(kl_list, kl_batch_mean)

                msg = "[$(loss_avg),L = $(loss_pi_avg),π + $(loss_v_avg),ν - $(args["model_loss_coef_entropy"]) × $(loss_entropy_avg),H] [$(kl_batch_mean),KL]"

                next!(progress_tracker; showvalues=[
                    (:loss, msg),
                ])

                # check if kl divergence is too high, break early if so
                if kl_batch_mean > 4 * args["train_kl_target"]
                    @info "KL divergence [$(round(kl_batch_mean, digits=4)) > 4 * $(args["train_kl_target"])] ... stopping early ."
                    break
                end

            end

            # finish!(progress_tracker; showvalues=[
            #     (:loss, msg),
            # ])

            @info "Training epoch [$(epoch)] $(msg)"

            # check for kl divergence
            kl_epoch = round(mean(kl_list), digits=4)

            if kl_epoch > args["train_kl_target"] * 2.0
                new_lr = max(t._exps.opt()[1].eta / 1.5, args["learning_rate"] / args["learning_rate_range"])
                @info "KL divergence [$(kl_epoch) > $(args["train_kl_target"]) × 2.0] ... reducing learning rate to [$(round(new_lr, digits=4))] ."
                # update learning rate
                t._exps.opt()[1].eta = new_lr
            elseif kl_epoch < args["train_kl_target"] / 2.0
                new_lr = min(t._exps.opt()[1].eta * 1.5, args["learning_rate"] * args["learning_rate_range"])
                @info "KL divergence [$(kl_epoch) < $(args["train_kl_target"]) ÷ 2.0] ... increasing learning rate to [$(round(new_lr, digits=4))] ."
                # update learning rate
                t._exps.opt()[1].eta = new_lr
            else
                lr = t._exps.opt()[1].eta
                @info "KL divergence [$(kl_epoch) within range] ... keeping learning rate as [$(round(lr, digits=4))] ."
            end

        end

        """Save trained model, optimizer, and experiences"""
        function save_trained()

            function eval_player(player_id::Int)

                # if not enough active players, make this player active
                if t.elo().activeSize() < args["population_min"]
                    t.elo().makeActive(player_id)
                    return nothing
                else
                    t.elo().makeCandidate(player_id)
                end

                # get a list of active players and their ratings
                active_pool = t.elo().playerInfo()["active"]
                active_avg_rating = t.elo().activeAvgRating()

                # if too many active players, remove lowest elo player
                while t.elo().activeSize() > args["population_max"]
                    min_tuple = active_pool[argmin([p[2] for p in active_pool])]
                    t.elo().makeInactive(min_tuple[1])
                end

                # check if any active player with rating < 1800, or is 120 elo below avg
                for (id, rating) in active_pool
                    if rating < active_avg_rating - args["elo_below_avg_cutoff"]
                        # remove players with low elo
                        t.elo().makeInactive(id)
                        # add from candidate pool
                        if t.elo().candidateSize() > 0
                            candidate_id = t.elo().randCandidate()
                            t.elo().makeActive(candidate_id)
                            t.elo().clearCandidates() # clear all candidates after adding one to active
                        end
                    end
                end

                # if we have two or more candidates, and we have not reached the max number of players
                if t.elo().candidateSize() > t.elo().activeSize() && t.elo().activeSize() < args["population_max"]
                    t.elo().makeActive(t.elo().randCandidate()) # add a random candidate
                    t.elo().clearCandidates() # clear all candidates
                end

            end

            # print separator
            @info repeat("-", 50)

            # create a new player with average active player rating
            player_id = t.elo().newPlayer(init_rating=t.elo().activeAvgRating())

            # save model
            model_ = t._exps.model()
            params_size = sum([length(l) for l in Flux.params(model_._model)])
            model_filepath = model_filename(player_id)
            @info "Save model [$(player_id)] [$(model_filepath)] : [$(model_.size())x$(model_.size()), c=$(model_.channels()), p=$(params_size)]"
            model_.save(model_filepath)
            model_.save(model_filename(1)) # always save another copy for id = 1

            # save optimizer
            opt_ = t._exps.opt()
            opt_filepath = opt_filename(1)
            @info "Save optimizer [$(opt_filepath)] : [η=$(round(opt_[1].eta, digits=4)), β=$(round.(opt_[1].beta, digits=4)), δ=$(opt_[2].wd)]"
            save_optimizer(opt_filepath, opt_)

            # save experiences
            exp_filepath = exp_filename(1)
            @info "Save experiences [$(exp_filepath)] : [len=$(t._exps.length()), tot=$(t._exps.totalCount()), trn=$(t._exps.trainedBatch())]"
            t._exps.save(exp_filepath)

            # print separator
            @info repeat("-", 50)

            # evaluate player
            eval_player(player_id)

            # save elo ratings
            t.elo().savePlayers(elo_filename("player"))
            t.elo().saveRatings(elo_filename("rating"))
            t.elo().saveHistory(elo_filename("history"))
            t.elo().saveLog(elo_filename("log"))

        end

        """Train"""
        t.train = () -> begin

            sleep_time = 15.0 # initialize to 15 seconds
            sleep_count = 0
            sleep_count_ema = 5.0 # initialize to target of 5.0

            while true

                try
                    # check trained batch count, exit if exceeded maximum
                    if t._exps.trainedBatch() > args["exp_max_trained_batch"]
                        exit(0)
                    end

                    if (t._exps.trainedBatch() * args["train_trim_ratio"] + BATCH_NUM * TRAIN_EPOCHS) * BATCH_SIZE < t._exps.totalCount() * TRAIN_EPOCHS

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
                                    t._exps.addtrainedBatch(BATCH_NUM)
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
                            t._exps.trimExperience(trunc(Int, BATCH_SIZE * BATCH_NUM * args["train_trim_ratio"]))
                        end

                        # save trained model, optimizer, and experiences
                        save_trained()

                    else

                        @info "Training is waiting for more experiences [len=$(t._exps.length()), tot=$(t._exps.totalCount()), trn=$(t._exps.trainedBatch())] ..."

                    end

                catch e

                    @error "Error" exception = (e, catch_backtrace())

                finally

                    # GC & reclaim CUDA memory
                    GC.gc(true)
                    if args["model_cuda"] >= 0
                        CUDA.reclaim()
                    end

                    # sleep
                    sleep_count += 1
                    sleep(sleep_time)

                end

            end

        end

        """elo"""
        t.elo = () -> t._exps.elo()

        return t

    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    @info repeat("=", 50)
    # train model[1] and model[2]
    t = Train()
    t.train()
end
