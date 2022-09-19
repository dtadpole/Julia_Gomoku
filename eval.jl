include("./mcts.jl")


"""Model path"""
modelPath = (id::Int; size=args["game_size"], tag="curr") -> begin
    model_path = "./trained/$(id)/model_$(size)x$(size).$(tag)"
    if !isdir(dirname(model_path))
        mkpath(dirname(model_path))
    end
    return model_path
end

function model_step(game::Game, model::Model)

    TEMPERATURE = args["mcts_temperature"]

    SIZE = game.size()

    if game.is_over()
        @error "Game is over"
        return nothing
    end

    start_time = time()

    # create mcts node
    node = MctsNode(game, model; cpuct=args["mcts_cpuct"], gamma=args["mcts_gamma"]) # no noise during evaluation

    # mcts play
    prior_N = node._N
    while node._N < game.size()^2 * args["mcts_n_multiplier"]
        node.travelBranch(args["mcts_depth"])
    end
    post_N = node._N

    # policy distribution
    pi = node.rootN()
    if TEMPERATURE > 1e-4
        pa = softmax(log.(pi) / TEMPERATURE, dims=[1, 2])
        action = sample(1:model.size()^2, ProbabilityWeights(reshape(pa, model.size()^2)))
        action = CartesianIndex(mod(action - 1, SIZE) + 1, div(action - 1, SIZE) + 1)
    else
        action = argmax(pi)
    end

    # play action
    game.play_action(action)
    if args["game_display"]
        duration = round(time() - start_time, digits=1)
        @info "[$(-round(node.rootQ(), digits=2))] $(join(["($(game.str_action(a)), $(n))" for (a, n) in maxk(reshape(node.rootN(), SIZE^2), 5)], ", ")) [$(prior_N)->$(post_N), $(duration)s] "
        # display
        game.display()
    end

    # GC & reclaim CUDA memory
    GC.gc(true)
    if args["model_cuda"] >= 0
        CUDA.reclaim()
        # CUDA.memory_status()
        # println()
    end

    return nothing

end

function human_step(game::Game)

    if game.is_over()
        @error "Game is over"
        return nothing
    end

    # get action
    action = 0
    while action == 0
        print("What is your move : ")
        move = strip(uppercase(readline()))
        if move == ""
            continue
        end
        if move[1] in ('A':'Z')
            x = parse(Int, move[2:end])
            y = (move[1] - 'A') + 1
            action = CartesianIndex(x, y)
        else
            continue
        end
    end

    # play action
    game.play_action(action)
    if args["game_display"]
        # display
        game.display()
    end

    return nothing

end

function player_function(player::String; eval_id=1)

    if player == "curr"

        filename = modelPath(eval_id; tag="curr")
        model = Model(args["game_size"], channels=args["model_channels"])
        model.load(filename)
        params_size = sum([length(l) for l in Flux.params(model._model)])
        @info "[$(eval_id)] Loaded model [$(filename)] [$(model.size())x$(model.size()), c=$(model.channels()), p=$(params_size)]"
        testmode!(model)
        return (game) -> model_step(game, model), model

    elseif player == "best"

        filename = modelPath(eval_id; tag="best")
        model = Model(args["game_size"], channels=args["model_channels"])
        if isfile(filename)
            # load model if best file exist
            model.load(filename)
            params_size = sum([length(l) for l in Flux.params(model._model)])
            @info "[$(eval_id)] Loaded model [$(filename)] [$(model.size())x$(model.size()), c=$(model.channels()), p=$(params_size)]"
        end
        testmode!(model)
        return (game) -> model_step(game, model), model

    elseif player == "server"

        id, model = download_model(eval_id)
        testmode!(model)
        return (game) -> model_step(game, model), model

    elseif player == "human"

        return human_step, nothing

    else

        error("Unknown player")

    end
end

function eval_playout(player1_function::Function, player2_function::Function; turn=1.0)

    game = Game(args["game_size"]; turn=turn)
    if args["game_display"]
        game.display()
    end

    if turn < 0
        player2_function(game)
    end

    while !game.is_over()

        if game.turn() > 0
            player1_function(game)
        else
            player2_function(game)
        end

    end

    return game.score()

end

function eval_run(eval_id::Int)

    @info repeat("=", 50)
    @info "[$(eval_id)] Starting evaluation [$(args["player_1"])] vs [$(args["player_2"])] - [$(args["eval_playouts"]) playouts]"
    @info repeat("-", 50)

    player_1_wins = 0
    player_2_wins = 0
    draws = 0

    player1_function, model_1 = player_function(args["player_1"], eval_id=eval_id)
    player2_function, model_2 = player_function(args["player_2"], eval_id=eval_id)

    for i in 1:args["eval_playouts"]
        try
            @info repeat("-", 50)
            @info "[$(eval_id)] Evaluation Iteration [$(i)] : [$(args["player_1"]) -> X] vs [$(args["player_2"]) -> O]"
            @info repeat("-", 50)
            game_turn = rand() > 0.5 ? 1.0 : -1.0
            score = eval_playout(player1_function, player2_function; turn=game_turn)
            if score > 0
                player_1_wins += 1
            elseif score < 0
                player_2_wins += 1
            else
                draws += 1
            end
        catch e
            @error "[$(eval_id)] Error" exception = (e, catch_backtrace())
            exit(1)
        finally
            @info repeat("-", 50)
            @info "[$(eval_id)] Tally [$(i)] : [$(args["player_1"]) : $(player_1_wins)] vs [$(args["player_2"]) : $(player_2_wins)] , [$(draws) draws]"
            sleep(1)
        end
    end

    @info repeat("=", 50)
    @info "[$(eval_id)] [$(args["player_1"]) : $(player_1_wins)] vs [$(args["player_2"]) : $(player_2_wins)] , [$(draws) draws]"
    if (args["player_1"] == "curr" || args["player_1"] == "server") && args["player_2"] == "best" && (player_1_wins - player_2_wins) >= args["eval_playouts"] / 2
        @info repeat("-", 50)
        @info "[$(eval_id)] Replacing best model with current model ..."
        best_filename = modelPath(eval_id; tag="best")
        backup_file(best_filename)
        model_1.save(best_filename)
        @info "[$(eval_id)] Replaced best model with current model !"
    end
    @info repeat("=", 50)

end


if abspath(PROGRAM_FILE) == @__FILE__
    # eval_id ∈ [1, 2, 3, ...]
    for eval_id in 1:args["population_size"]
        eval_run(eval_id)
    end
end
