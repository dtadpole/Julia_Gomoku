include("./exp.jl")

using Unzip
# unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))

function save_optimizer(path::String, opt::Flux.Optimise.AbstractOptimiser)
    open(path, "w") do io
        serialize(io, opt)
    end
end

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
    function Train(id_list::Vector{Int})

        t = new()

        """Size"""
        t.size = () -> args["game_size"]

        """Model path"""
        t.modelPath = (id::Int; tag = "curr") -> begin
            model_path = "./trained/$(id)/model_$(t.size())x$(t.size()).$(tag)"
            if !isdir(dirname(model_path))
                mkpath(dirname(model_path))
            end
            return model_path
        end

        """Opt path"""
        t.optPath = (id::Int) -> begin
            opt_path = "./trained/$(id)/opt_$(t.size())x$(t.size()).curr"
            if !isdir(dirname(opt_path))
                mkpath(dirname(opt_path))
            end
            return opt_path
        end

        """Exp path"""
        t.expPath = (id::Int) -> begin
            exp_path = "./trained/$(id)/exp_$(t.size())x$(t.size()).curr"
            if !isdir(dirname(exp_path))
                mkpath(dirname(exp_path))
            end
            return exp_path
        end

        # initialize models, optimizers
        model_list = Vector{Model}()
        opt_list = Vector{Flux.Optimise.AbstractOptimiser}()

        for id in id_list

            # initialize model
            model = Model(args["game_size"], channels=args["model_channels"])
            params_size = sum([length(l) for l in Flux.params(model._model)])
            @info "[$(id)] Initialize Model [$(model.size())x$(model.size()), c=$(model.channels()), p=$(params_size)]"
            # load model if exists
            model_filename = t.modelPath(id)
            if isfile(model_filename)
                @info repeat("-", 50)
                @info "[$(id)] Loading Model from [$(model_filename)] ..."
                model.load(model_filename)
                @info "[$(id)] Loaded Model [$(model.size())x$(model.size()), c=$(model.channels()), p=$(params_size)]" model._model
            end
            push!(model_list, model)

            # initialize optimizer
            opt = AdamW(args["learning_rate"], (args["adamw_beta1"], args["adamw_beta2"]), args["adamw_weight_decay"])
            # opt = Adam(args["learning_rate"])
            @info "[$(id)] Initialize Optimizer [η=$(round(opt[1].eta, digits=4)), β=$(round.(opt[1].beta, digits=4)), δ=$(opt[2].wd)]"
            # load optimizer if exists
            opt_filename = t.optPath(id)
            if isfile(opt_filename)
                @info repeat("-", 50)
                @info "[$(id)] Loading Optimizer from [$(opt_filename)] ..."
                opt = load_optimizer(opt_filename)
                @info "[$(id)] Loaded Optimizer [η=$(round(opt[1].eta, digits=4)), β=$(round.(opt[1].beta, digits=4)), δ=$(opt[2].wd)]"
            end
            # move to gpu
            if args["model_cuda"] >= 0
                opt = opt |> gpu
            end
            push!(opt_list, opt)

        end

        # initialize experiences
        t._experiences = Experiences(model_list, opt_list)

        for id in id_list
            # load experiences if exists
            exp_filename = t.expPath(id)
            if isfile(exp_filename)
                @info repeat("-", 50)
                @info "[$(id)] Loading Experiences from [$(exp_filename)] ..."
                t._experiences.load(id, exp_filename)
                @info "[$(id)] Loaded Experiences [len=$(t._experiences.length(id)), total=$(t._experiences.totalCount(id)), trained=$(t._experiences.trainedBatch(id))]"
            end
        end

        # start experiences server
        @info repeat("=", 50)
        t._experiences.startServer()

        """Parameters"""
        BATCH_SIZE = args["train_batch_size"]
        BATCH_NUM = args["train_batch_num"]
        TRAIN_EPOCHS = args["train_epochs"]

        """Train epoch"""
        function train_epoch(id::Int, epoch::Int)

            @info "[$(id)] Training epoch [$(epoch)] started [len=$(t._experiences.length(id)), tot=$(t._experiences.totalCount(id)), trn=$(t._experiences.trainedBatch(id))] ..."

            if args["exp_sample_sequential"]
                # re-sample each epoch
                (states, pis, vs) = t._experiences.sampleExperience(id, BATCH_SIZE * BATCH_NUM)
            else
                # re-sample each epoch
                (states, pis, vs) = t._experiences.sampleExperience(id, BATCH_SIZE * BATCH_NUM)
            end

            # keeps a list of kl divergence
            kl_list = Vector{Float32}()

            # keeps a list of loss
            loss_list = Vector{Float32}()
            loss_pi_list = Vector{Float32}()
            loss_v_list = Vector{Float32}()
            loss_entropy_list = Vector{Float32}()
            loss_reg_list = Vector{Float32}()

            for batch in 1:BATCH_NUM

                state = states[:, :, :, (batch-1)*BATCH_SIZE+1:batch*BATCH_SIZE]
                pi = pis[:, :, (batch-1)*BATCH_SIZE+1:batch*BATCH_SIZE]
                v = vs[:, (batch-1)*BATCH_SIZE+1:batch*BATCH_SIZE]

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

                lossF = () -> t._experiences.model(id).loss(state, pa, v)[1]  # [IMPORTANT] use normalized distribution pa !!
                params = t._experiences.model(id).params()
                grads = gradient(lossF, params)

                Flux.Optimise.update!(t._experiences.opt(id), params, grads)

                if !args["exp_sample_sequential"]
                    t._experiences.addtrainedBatch(id, 1)
                end

                # calculate new policy
                new_loss, new_pi, new_vi, loss_pi, loss_v, loss_entropy, loss_reg = t._experiences.model(id).loss(state, pa, v)

                # @info "new sizes" size(new_loss) size(new_pi) size(new_v)

                # keep track of loss
                push!(loss_list, new_loss |> cpu)
                push!(loss_pi_list, loss_pi |> cpu)
                push!(loss_v_list, loss_v |> cpu)
                push!(loss_entropy_list, loss_entropy |> cpu)
                push!(loss_reg_list, loss_reg |> cpu)

                # extract prev policy
                # prev_pi = prev_policy[:, :, (batch-1)*BATCH_SIZE+1:batch*BATCH_SIZE]

                # calculate KL divergence
                kl_sum = sum(prev_pi .* (log.(prev_pi) .- log.(new_pi)), dims=[1, 2])
                kl = reshape(kl_sum, BATCH_SIZE)
                kl_mean = mean(kl)

                # update global list
                kl_list = vcat(kl_list, kl |> cpu)

                # check if kl divergence is too high, break early if so
                if kl_mean > 4 * args["train_kl_target"]
                    @info "[$(id)] KL divergence [$(round(kl_mean, digits=4)) > 4 * $(args["train_kl_target"])] ... stopping early ."
                    break
                end

            end

            loss_epoch = round(mean(loss_list), digits=3)
            pi_epoch = round(mean(loss_pi_list), digits=2)
            v_epoch = round(mean(loss_v_list), digits=3)
            entropy_epoch = round(mean(loss_entropy_list), digits=2)
            reg_epoch = round(mean(loss_reg_list), digits=4)

            @info "[$(id)] Training epoch [$(epoch)] [$(loss_epoch),L = $(pi_epoch),π + $(v_epoch),ν - $(args["model_loss_coef_entropy"]) × $(entropy_epoch),H + $(args["model_loss_coef_theta"]) × $(reg_epoch),θ]"

            # check for kl divergence
            kl_epoch_mean = mean(kl_list)

            if kl_epoch_mean > args["train_kl_target"] * 2.0
                new_learning_rate = max(t._experiences.opt(id)[1].eta / 1.5, args["learning_rate"] / args["learning_rate_range"])
                @info "[$(id)] KL divergence [$(round(kl_epoch_mean, digits=4)) > $(args["train_kl_target"]) × 2.0] ... reducing learning rate to [$(round(new_learning_rate, digits=4))] ."
                # update learning rate
                t._experiences.opt(id)[1].eta = new_learning_rate
            elseif kl_epoch_mean < args["train_kl_target"] / 2.0
                new_learning_rate = min(t._experiences.opt(id)[1].eta * 1.5, args["learning_rate"] * args["learning_rate_range"])
                @info "[$(id)] KL divergence [$(round(kl_epoch_mean, digits=4)) < $(args["train_kl_target"]) ÷ 2.0] ... increasing learning rate to [$(round(new_learning_rate, digits=4))] ."
                # update learning rate
                t._experiences.opt(id)[1].eta = new_learning_rate
            else
                learning_rate = t._experiences.opt(id)[1].eta
                @info "[$(id)] KL divergence [$(round(kl_epoch_mean, digits=4)) within range] ... keeping learning rate as [$(round(learning_rate, digits=4))] ."
            end

        end

        """Save trained model, optimizer, and experiences"""
        function save_trained(id::Int)

            # print separator
            @info repeat("-", 50)

            # save model
            model_ = t._experiences.model(id)
            params_size = sum([length(l) for l in Flux.params(model_._model)])
            model_filename = t.modelPath(id)
            @info "[$(id)] Save model [$(model_filename)] : [$(model_.size())x$(model_.size()), c=$(model_.channels()), p=$(params_size)]"
            backup_file(model_filename)
            model_.save(model_filename)

            # save optimizer
            opt_ = t._experiences.opt(id)
            opt_filename = t.optPath(id)
            @info "[$(id)] Save optimizer [$(opt_filename)] : [η=$(round(opt_[1].eta, digits=4)), β=$(round.(opt_[1].beta, digits=4)), δ=$(opt_[2].wd)]"
            backup_file(opt_filename)
            save_optimizer(opt_filename, opt_)

            # save experiences
            exp_filename = t.expPath(id)
            @info "[$(id)] Save experiences [$(exp_filename)] : [len=$(t._experiences.length(id)), tot=$(t._experiences.totalCount(id)), trn=$(t._experiences.trainedBatch(id))]"
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

                for id in id_list

		    # check trained batch count, exit if exceeded maximum
		    if t._experiences.trainedBatch(id) > args["exp_trained_batch"]
		        exit(0)
		    end

                    # t._experiences.playGame() # do not play game in train server.  let the inference servers play games

                    if (t._experiences.trainedBatch(id) * args["train_trim_ratio"] + BATCH_NUM * TRAIN_EPOCHS) * BATCH_SIZE < t._experiences.totalCount(id) * TRAIN_EPOCHS

                        sleep_count_ema = 0.9 * sleep_count_ema + 0.1 * sleep_count
                        sleep_time = max(1.0, min(120.0, (sleep_time * sleep_count_ema) / 5.0))

                        sleep_count = 0

                        try
                            # print separator
                            @info repeat("-", 50)

                            for epoch in 1:TRAIN_EPOCHS

                                try
                                    train_epoch(id, epoch)
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

                        catch e

                            @error "Error" exception = (e, catch_backtrace())

                        finally

                            if args["exp_sample_sequential"]
                                # trip experiences by BATCH_SIZE * BATCH_NUM
                                t._experiences.trimExperience(id, trunc(Int, BATCH_SIZE * BATCH_NUM * args["train_trim_ratio"]))
                            end

                            # save trained model, optimizer, and experiences
                            save_trained(id)

                        end

                    else

                        @info "[$(id)] Training is waiting for more experiences [len=$(t._experiences.length(id)), tot=$(t._experiences.totalCount(id)), trn=$(t._experiences.trainedBatch(id))] ..."

                    end

                    # GC & reclaim CUDA memory
                    GC.gc(true)
                    if args["model_cuda"] >= 0
                        CUDA.reclaim()
                    end

                end

                sleep_count += 1

                sleep(sleep_time)

            end

        end

        return t

    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    @info repeat("=", 50)
    # train model[1] and model[2]
    train = Train([id for id in 1:args["population_size"]])
    train.train()
end
