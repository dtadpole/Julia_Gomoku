include("./mcts.jl")
include("./elo.jl")

using HTTP
using Random
using JSON

const _WEIGHTS = 1:floor(Int, args["exp_max"] * 2)

mutable struct Experiences

    # attributes
    _elo::Elo
    _model::Model
    _opt::Flux.Optimise.AbstractOptimiser
    _exp::Vector{Tuple{Matrix{Float32},Matrix{Float32},Float32,Int,Int}}
    _total_count::Int
    _trained_batches::Int

    # functions
    size::Function
    length::Function
    totalCount::Function
    trainedBatch::Function
    addtrainedBatch::Function
    addExperience::Function
    sampleExperience::Function
    trimExperience::Function
    startServer::Function
    save::Function
    load::Function
    model::Function
    opt::Function
    elo::Function

    # constructor
    function Experiences(elo::Elo, model::Model, opt::Flux.Optimise.AbstractOptimiser)

        global _WEIGHTS

        e = new(elo, model, opt)

        # initialize experiences, total count, and trained count
        e._exp = Vector{Tuple{Matrix{Float32},Matrix{Float32},Float32,Int,Int}}()
        e._total_count = 0
        e._trained_batches = 0

        """Size"""
        e.size = () -> e._model.size()

        """Length"""
        e.length = () -> length(e._exp)

        """Total count"""
        e.totalCount = () -> e._total_count

        """Trained count"""
        e.trainedBatch = () -> e._trained_batches

        """Add to trained count"""
        e.addtrainedBatch = (count) -> begin
            e._trained_batches += count
        end

        """Add experience"""
        e.addExperience = (s, pi, v, id_1, id_2) -> begin

            size_s = size(s)
            size_pi = size(pi)
            if size_s[1] != e.size() || size_s[2] != e.size() || size_pi[1] != e.size() || size_pi[2] != e.size()
                error("Invalid experience [s: $(size_s), pi: $(size_pi), v: $(v)]")
            end

            # flip and rotate per game symmetry
            s_flipped = flip(s)
            pi_flipped = flip(pi)
            push!(e._exp, (s, pi, v, id_1, id_2))
            push!(e._exp, (rotate90(s), rotate90(pi), v, id_1, id_2))
            push!(e._exp, (rotate180(s), rotate180(pi), v, id_1, id_2))
            push!(e._exp, (rotate270(s), rotate270(pi), v, id_1, id_2))
            push!(e._exp, (s_flipped, pi_flipped, v, id_1, id_2))
            push!(e._exp, (rotate90(s_flipped), rotate90(pi_flipped), v, id_1, id_2))
            push!(e._exp, (rotate180(s_flipped), rotate180(pi_flipped), v, id_1, id_2))
            push!(e._exp, (rotate270(s_flipped), rotate270(pi_flipped), v, id_1, id_2))

            # record total experience size
            e._total_count += 8

            if !args["exp_sample_sequential"]
                # target experience size
                target_exp_size = max(args["exp_min"], min(args["exp_max"],
                    floor(Int, (e._total_count - args["exp_min"]) * args["exp_preserve_ratio"]) + args["exp_min"]))

                # truncate experiences
                if length(e._exp) > target_exp_size
                    e._exp = e._exp[length(e._exp)-target_exp_size+1:end]
                end
            end
        end

        """Sample experience"""
        e.sampleExperience = (sample_size::Int) -> begin
            if args["exp_sample_sequential"]
                # sample sequentially
                s, pi, v, _, _ = unzip(e._exp[1:sample_size])
                # reshape to tensor
                s = reshape(cat(s...; dims=3), (e.size(), e.size(), 1, sample_size))
                pi = reshape(cat(pi...; dims=3), (e.size(), e.size(), sample_size))
                v = reshape(cat(v...; dims=2), (1, sample_size))
                return s, pi, v
            else
                # sample randomly
                s, pi, v, _, _ = unzip(sample(e._exp, ProbabilityWeights(_WEIGHTS[1:min(length(e._exp), args["exp_max"])]), sample_size, replace=false))
                # reshape to tensor
                s = reshape(cat(s...; dims=3), (e.size(), e.size(), 1, sample_size))
                pi = reshape(cat(pi...; dims=3), (e.size(), e.size(), sample_size))
                v = reshape(cat(v...; dims=2), (1, sample_size))
                return s, pi, v
            end
        end

        """Trim experiences"""
        e.trimExperience = (sample_size::Int) -> begin
            # remove sampled experiences
            @info "Trimming experience [$(length(e._exp)) -> $(length(e._exp)-sample_size)]"
            e._exp = e._exp[sample_size+1:end]
        end

        """Start server"""
        e.startServer = () -> begin

            router = HTTP.Router()

            # post a full game experiences
            HTTP.register!(router, "POST", "/game", (req) -> begin
                body = HTTP.payload(req)
                io = IOBuffer(body)
                id_1, id_2, score, game = deserialize(io)
                # update elo
                e._elo.addGame(id_1, id_2, score)
                # add experiences
                for (s, pi, v) in game
                    e.addExperience(s, pi, v, id_1, id_2)
                    id_1, id_2 = id_2, id_1 # switch position
                end
                return HTTP.Response(200, "OK")
            end)

            # game size
            HTTP.register!(router, "GET", "/game/size", (req) -> begin
                return HTTP.Response(200, Dict("Content-Type" => "application/json"), body=JSON.json(e.size()))
            end)

            # active and candidate players
            HTTP.register!(router, "GET", "/player/info", (req) -> begin
                return HTTP.Response(200, Dict("Content-Type" => "application/json"), body=JSON.json(e._elo.playerInfo()))
            end)

            # download model
            HTTP.register!(router, "GET", "/model/{id}", (req) -> begin
                id = parse(Int, HTTP.getparams(req)["id"])
                if id == 1 # return in memory model for id = 1
                    io = IOBuffer()
                    serialize(io, (e._model.size(), e._model.channels(), e._model._model |> cpu))
                    return HTTP.Response(200, Dict("Content-Type" => "application/octet-stream"), body=io.data)
                else
                    model_path = model_filename(id)
                    open(model_path, "r") do io
                        data = read(io)
                        return HTTP.Response(200, Dict("Content-Type" => "application/octet-stream"), body=data)
                    end
                end
            end)

            # get a match
            HTTP.register!(router, "GET", "/match", (req) -> begin
                if e._elo.activeSize() <= 0
                    return HTTP.Response(400, Dict("Content-Type" => "application/json"), body=JSON.json(()))
                elseif e._elo.activeSize() == 1
                    id_1, id_2 = e._elo.selfMatch(first(e._elo.activePlayers()))
                    return HTTP.Response(200, Dict("Content-Type" => "application/json"), body=JSON.json((id_1, id_2)))
                else
                    id_1, id_2 = e._elo.sampleMatch()
                    return HTTP.Response(200, Dict("Content-Type" => "application/json"), body=JSON.json((id_1, id_2)))
                end
            end)

            # start server async
            @async HTTP.serve(router, "0.0.0.0", args["exp_port"])
            @info "Experience server started on port [$(args["exp_port"])]"
            @info "Run [infer.sh] to generate experiences ..."

        end

        """Save"""
        e.save = (path) -> begin
            backup_file(path)
            open(path, "w") do io
                serialize(io, (e._total_count, e._trained_batches, e._exp))
            end
        end

        """Load"""
        e.load = (path) -> begin
            open(path, "r") do io
                total_count_, trained_batches_, exp_ = deserialize(io)
                e._total_count = total_count_
                e._trained_batches = trained_batches_
                e._exp = exp_
            end
        end

        """Return model"""
        e.model = () -> begin
            return e._model
        end

        """Return opt"""
        e.opt = () -> begin
            return e._opt
        end

        """Return elo"""
        e.elo = () -> begin
            return e._elo
        end

        return e

    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    # create models
    model_1 = Model(args["game_size"], channels=args["model_channels"])
    model_2 = Model(args["game_size"], channels=args["model_channels"])
    e = Experiences([model_1, model_2])
    e.startServer()
    e.playGame(1, 2)
end
