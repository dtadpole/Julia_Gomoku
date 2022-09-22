include("./mcts.jl")

using JSON

"""infer playouts"""
function infer_playout(i::Int)

    global URL_BASE

    @info repeat("-", 50)
    @info "Inference Iteration [$(i)] ..."
    @info repeat("-", 50)

    url = "$(URL_BASE)/match"

    r = HTTP.request(:GET, "$(url)")
    if r.status != 200
        @warn "Requested match [$(url)] : [status = $(r.status)]"
        exit(1)
    end

    io = IOBuffer(r.body)
    id_1, id_2 = JSON.parse(io)
    @info "Requested match [$(url)] : [status = $(r.status)] [$(id_1) vs $(id_2)]"

    model_1 = download_model(id_1)
    model_2 = download_model(id_2)

    @info repeat("-", 50)

    experiences, game_init_turn, game_score = mcts_play_game(model_1, model_2)

    id_first, id_second = game_init_turn > 0 ? (id_1, id_2) : (id_2, id_1)

    normalized_score = game_init_turn > 0 ? game_score : -game_score

    io = IOBuffer()
    serialize(io, (id_first, id_second, normalized_score, experiences))

    url = "$(URL_BASE)/game"
    @info "Posting game [$(url)] [$(id_first) vs $(id_second)] [$(normalized_score)] [len = $(length(experiences))]"
    r = HTTP.request(:POST, url, body=take!(io))
    @info "Posted experiences [$(url)] [status = $(r.status)]"


end


if abspath(PROGRAM_FILE) == @__FILE__

    for i in 1:args["infer_playouts"]
        try

            infer_playout(i)

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
