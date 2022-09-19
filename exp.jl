include("./mcts.jl")

using HTTP
using Random

const _WEIGHTS = 1:floor(Int, args["exp_max"] * 2)

function model_filename(id::Int)
    model_filename = "./model/$(id)/model_$(model.size())x$(model.size()).curr"
    return model_filename
end

mutable struct Experiences

    # attributes
    _model_list::Vector{Model}
    _opt_list::Vector{Flux.Optimise.Optimiser}
    _exp_list::Vector{Vector{Tuple{Matrix{Float32},Matrix{Float32},Float32}}}
    _total_count::Vector{Int}
    _trained_batches::Vector{Int}

    # functions
    size::Function
    length::Function
    totalCount::Function
    trainedBatch::Function
    addtrainedBatch::Function
    playGame::Function
    addExperience::Function
    sampleExperience::Function
    trimExperience::Function
    startServer::Function
    save::Function
    load::Function
    model::Function
    opt::Function

    # constructor
    function Experiences(models::Vector{Model}, opts::Vector{Flux.Optimise.Optimiser})

        global _WEIGHTS

        for model in models
            if model.size() != models[1].size()
                error("Model size does not match : [$(models[1].size()) vs $(model.size())]")
            end
        end

        e = new(models, opts)

        # initialize experiences, total count, and trained count
        e._exp_list = Vector{Vector{Tuple{Matrix{Float32},Matrix{Float32},Float32}}}()
        for i in 1:length(models)
            push!(e._exp_list, Vector{Tuple{Matrix{Float32},Matrix{Float32},Float32}}())
        end
        e._total_count = zeros(Int, length(models))
        e._trained_batches = zeros(Int, length(models))

        """Size"""
        e.size = () -> e._model_list[1].size()

        """Length"""
        e.length = (id) -> length(e._exp_list[id])

        """Total count"""
        e.totalCount = (id) -> e._total_count[id]

        """Trained count"""
        e.trainedBatch = (id) -> e._trained_batches[id]

        """Add to trained count"""
        e.addtrainedBatch = (id, count) -> begin
            e._trained_batches[id] += count
        end

        """Play game"""
        e.playGame = (id_1::Int, id_2::Int) -> begin

            experiences, game_init_turn = mcts_play_game(e._model_list[id_1], e._model_list[id_2])

            id_first, id_second = game_init_turn > 0 ? (id_1, id_2) : (id_2, id_1)

            experiences_with_id = [(mod(step, 2) == 1 ? id_first : id_second, s, pi, v) for (step, (s, pi, v)) in enumerate(experiences)]

            for (id, s, pi, v) in experiences_with_id
                e.addExperience(id, s, pi, v)
            end

        end

        """Add experience"""
        e.addExperience = (id, s, pi, v) -> begin

            size_s = size(s)
            size_pi = size(pi)
            if size_s[1] != e.size() || size_s[2] != e.size() || size_pi[1] != e.size() || size_pi[2] != e.size()
                error("Invalid experience [s: $(size_s), pi: $(size_pi), v: $(v)]")
            end

            # flip and rotate per game symmetry
            s_flipped = flip(s)
            pi_flipped = flip(pi)
            push!(e._exp_list[id], (s, pi, v))
            push!(e._exp_list[id], (rotate90(s), rotate90(pi), v))
            push!(e._exp_list[id], (rotate180(s), rotate180(pi), v))
            push!(e._exp_list[id], (rotate270(s), rotate270(pi), v))
            push!(e._exp_list[id], (s_flipped, pi_flipped, v))
            push!(e._exp_list[id], (rotate90(s_flipped), rotate90(pi_flipped), v))
            push!(e._exp_list[id], (rotate180(s_flipped), rotate180(pi_flipped), v))
            push!(e._exp_list[id], (rotate270(s_flipped), rotate270(pi_flipped), v))

            # record total experience size
            e._total_count[id] += 8

            if !args["exp_sample_sequential"]
                # target experience size
                target_exp_size = max(args["exp_min"], min(args["exp_max"],
                    floor(Int, (e._total_count[id] - args["exp_min"]) * args["exp_preserve_ratio"]) + args["exp_min"]))

                # truncate experiences
                if length(e._exp_list[id]) > target_exp_size
                    e._exp_list[id] = e._exp_list[id][length(e._exp_list[id])-target_exp_size+1:end]
                end
            end
        end

        """Sample experience"""
        e.sampleExperience = (id, sample_size::Int) -> begin
            if args["exp_sample_sequential"]
                # sample sequentially
                s, pi, v = unzip(shuffle(e._exp_list[id][1:sample_size]))
                # reshape to tensor
                s = reshape(cat(s...; dims=3), (e.size(), e.size(), 1, sample_size))
                pi = reshape(cat(pi...; dims=3), (e.size(), e.size(), sample_size))
                v = reshape(cat(v...; dims=2), (1, sample_size))
                return s, pi, v
            else
                # sample randomly
                s, pi, v = unzip(sample(e._exp_list[id], ProbabilityWeights(_WEIGHTS[1:min(length(e._exp_list[id]), args["exp_max"])]), sample_size, replace=false))
                # reshape to tensor
                s = reshape(cat(s...; dims=3), (e.size(), e.size(), 1, sample_size))
                pi = reshape(cat(pi...; dims=3), (e.size(), e.size(), sample_size))
                v = reshape(cat(v...; dims=2), (1, sample_size))
                return s, pi, v
            end
        end

        """Trim experiences"""
        e.trimExperience = (id, sample_size::Int) -> begin
            # remove sampled experiences
            @info "[$(id)] Trimming experience [$(length(e._exp_list[id])) -> $(length(e._exp_list[id])-sample_size)]"
            e._exp_list[id] = e._exp_list[id][sample_size+1:end]
        end

        """Start server"""
        e.startServer = () -> begin

            router = HTTP.Router()

            # post a single game experience
            HTTP.register!(router, "POST", "/experience", (req) -> begin
                body = HTTP.payload(req)
                io = IOBuffer(body)
                id, s, pi, v = deserialize(io)
                e.addExperience(id, s, pi, v)
                return HTTP.Response(200, "OK")
            end)

            # post a list of game experiences
            HTTP.register!(router, "POST", "/experiences", (req) -> begin
                body = HTTP.payload(req)
                io = IOBuffer(body)
                for (id, s, pi, v) in deserialize(io)
                    e.addExperience(id, s, pi, v)
                end
                return HTTP.Response(200, "OK")
            end)

            # download model
            HTTP.register!(router, "GET", "/model/{id}", (req) -> begin
                id = parse(Int, HTTP.getparams(req)["id"])
                io = IOBuffer()
                serialize(io, (id, e._model_list[id].size(), e._model_list[id].channels(), e._model_list[id]._model |> cpu))
                return HTTP.Response(200, Dict("Content-Type" => "application/octet-stream"), body=io.data)
            end)

            # game size
            HTTP.register!(router, "GET", "/game/size", (req) -> begin
                io = IOBuffer()
                serialize(io, e.size())
                return HTTP.Response(200, Dict("Content-Type" => "application/octet-stream"), body=io.data)
            end)

            # start server async
            @async HTTP.serve(router, "0.0.0.0", args["exp_port"])
            @info "Experience server started on port [$(args["exp_port"])]"
            @info "Run [infer.sh] to generate experiences ..."

        end

        """Save"""
        e.save = (id, path) -> begin
            io = open(path, "w")
            serialize(io, (e._total_count[id], e._trained_batches[id], e._exp_list[id]))
            close(io)
        end

        """Load"""
        e.load = (id, path) -> begin
            io = open(path, "r")
            total_count, trained_batches, exp_list = deserialize(io)
            e._total_count[id] = total_count
            e._trained_batches[id] = trained_batches
            e._exp_list[id] = exp_list
            close(io)
        end

        """Return model"""
        e.model = (id) -> begin
            return e._model_list[id]
        end

        """Return opt"""
        e.opt = (id) -> begin
            return e._opt_list[id]
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
