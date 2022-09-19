include("./mcts.jl")

"""infer playouts"""
function infer_playouts()

    global URL_BASE

    model_1 = nothing
    model_2 = nothing

    for i in 1:args["infer_playouts"]

        try

            id_1, id_2 = sample(1:args["population_size"], 2, replace=args["population_allow_self_play"])

            @info repeat("-", 50)
            @info "Inference Iteration [$(i)] ..."
            @info repeat("-", 50)

            # id_1, model_1 = download_model(1, prev_model=model_1) # reuse previous model_1 if exists
            # id_2, model_2 = download_model(2, prev_model=model_2) # reuse previous model_2 if exists
            sid_1, model_1 = download_model(id_1) # do NOT reuse previous model_1 if exists
            sid_2, model_2 = download_model(id_2) # do NOT reuse previous model_2 if exists

            @info repeat("-", 50)
            experiences, game_init_turn = mcts_play_game(model_1, model_2)

            id_first, id_second = game_init_turn > 0 ? (sid_1, sid_2) : (sid_2, sid_1)

            experiences_with_id = [(mod(step, 2) == 1 ? id_first : id_second, s, pi, v) for (step, (s, pi, v)) in enumerate(experiences)]

            io = IOBuffer()
            serialize(io, experiences_with_id)

            url = "$(URL_BASE)/experiences"
            @info "Posting experiences [$(url)] [len = $(length(experiences))]"
            r = HTTP.request(:POST, url, body=take!(io))
            @info "Posted experiences [$(url)] [status = $(r.status)]"

        catch e

            @error "Error" exception = (e, catch_backtrace())

            exit(1)

        finally

            # GC & reclaim CUDA memory
            GC.gc(true)
            if args["model_cuda"] >= 0
                CUDA.reclaim()
                @info repeat("-", 50)
                CUDA.memory_status()
                println()
            end

            sleep(1)

        end
    end

    @info repeat("-", 50)
    @info "Inference Completed [$(args["infer_playouts"])] Iterations ."
    @info repeat("-", 50)

end

if abspath(PROGRAM_FILE) == @__FILE__
    infer_playouts()
end
