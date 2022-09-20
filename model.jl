include("./game.jl")

using Flux
using CUDA
using JLD
using Serialization
using HTTP

global URL_BASE = URL_BASE = "http://$(args["exp_server"]):$(args["exp_port"])"

# custom split layer
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

"""backup file"""
function backup_file(filename)
    filename_bak = "$(filename).bak"
    filename_bak_bak = "$(filename_bak).bak"
    # if isfile(filename_bak_bak)
    #     mv(filename_bak_bak, "$(filename_bak_bak).bak", force=true)
    # end
    if isfile(filename_bak)
        mv(filename_bak, filename_bak_bak, force=true)
    end
    if isfile(filename)
        mv(filename, filename_bak, force=true)
    end
    return nothing
end

"""download model"""
function download_model(id::Int; prev_model=nothing)

    global URL_BASE

    url = "$(URL_BASE)/model/$(id)"

    r = HTTP.request(:GET, "$(url)")
    @info "[$(id)] Requested model [$(url)] : [status = $(r.status)]"
    io = IOBuffer(r.body)
    _id, _size, _channels, _chain = deserialize(io)

    # move model to CPU if needed
    if args["model_cuda"] >= 0
        _chain = _chain |> gpu
    end

    # create new model
    model = Model(_size, channels=_channels)
    Flux.loadmodel!(model._model, _chain)
    # testmode!(model)

    params_size = sum([length(l) for l in Flux.params(model._model)])
    @info "[$(id)] Loaded model [$(url)] : [$(model.size())x$(model.size()), c=$(model.channels()), p=$(params_size)]" model._model

    return _id, model

end


# model
mutable struct Model

    # attributes
    _size::Int
    _channels::Int
    _model::Chain

    # functions
    size::Function
    channels::Function
    forward::Function
    loss::Function
    save::Function
    load::Function
    clone::Function
    params::Function

    # constructor
    function Model(size::Int; channels=32)

        m = new(size, channels)

        """Size"""
        m.size = () -> m._size

        """Channels"""
        m.channels = () -> m._channels

        """Model"""
        m._model = Chain(
            Conv((3, 3), 1 => channels, relu; pad=(1, 1)),
            Conv((3, 3), channels => channels * 2, relu; pad=(1, 1)),
            Conv((3, 3), channels * 2 => channels * 4, relu; pad=(1, 1)),
            Conv((3, 3), channels * 4 => channels * 4, relu; pad=(1, 1)),
            Split(
                Chain(
                    Conv((1, 1), channels * 4 => div(channels, 2), elu),
                    x -> reshape(x, (div(channels, 2) * size * size, :)),
                    Dropout(0.5),
                    Dense(div(channels, 2) * size * size => size * size, elu),
                    x -> reshape(x, (size, size, :))
                    # x -> log.(x + 1e-8), # add log layer
                    # softmax # softmax layer applies to only dims=1
                ),
                Chain(
                    Conv((1, 1), channels * 4 => div(channels, 2), elu),
                    x -> reshape(x, (div(channels, 2) * size * size, :)),
                    Dropout(0.5),
                    Dense(div(channels, 2) * size * size => channels * 4, elu),
                    Dropout(0.5),
                    Dense(channels * 4 => 1, tanh)
                )
            )
        )

        if args["model_cuda"] >= 0
            m._model = m._model |> gpu
        end


        """Forward pass"""
        m.forward = (x) -> begin
            p, v = m._model(x)
            # pi = softmax(p, dims=[1, 2]) .* reshape(abs.(abs.(x) .- 1.0), (m.size(), m.size(), :)) # mask out illegal moves
            pi = softmax(p, dims=[1, 2])
            return pi, v
        end

        """returns (loss, p_, v_)"""
        m.loss = (x, pi, v) -> begin
            # model forward
            p_, v_ = m.forward(x)
            entropy() = -sum(p_ .* log.(p_), dims=[1, 2])
            reg() = sum(x -> sum(x.^2), Flux.params(m._model))
            loss_pi = mean(-sum(pi .* log.(p_), dims=[1, 2]))
            loss_v = mean((v .- v_) .^ 2)
            loss_entropy = mean(entropy())
	    loss_reg = mean(reg())
            loss = loss_pi + loss_v - (args["model_loss_coef_entropy"] * loss_entropy) + args["model_loss_coef_theta"] * loss_reg
            return loss, p_, v_, loss_pi, loss_v, loss_entropy, loss_reg
        end

        """Save"""
        m.save = (path) -> begin
            io = open(path, "w")
            serialize(io, (m.size(), m.channels(), m._model))
            close(io)
        end

        """Load"""
        m.load = (path) -> begin
            io = open(path, "r")
            size, channels, model = deserialize(io)
            m._size = size
            m._channels = channels
            Flux.loadmodel!(m._model, model)
            close(io)
        end

        """Clone"""
        m.clone = () -> begin
            m2 = Model(m._size, channels=m._channels)
            m2._model = deepcopy(m._model)
            return m2
        end

        """Params"""
        m.params = () -> begin
            return Flux.params(m._model)
        end

        return m
    end
end

function _play_model(model::Model)
    # play game
    game = Game(model.size())
    while !game.is_over()
        if game.turn() > 0
            state = game.state()
        else
            state = -game.state()
        end
        if args["model_cuda"] >= 0
            state = state |> gpu
        end
        # get action
        pi, _ = model.forward(reshape(state, (model.size(), model.size(), 1, 1)))
        pi = pi .* game.available_actions()
        distribution = reshape(Array(pi), model.size()^2)
        action = sample(1:model.size()^2, ProbabilityWeights(distribution))
        action = CartesianIndex(mod(action - 1, model.size()) + 1, div(action - 1, model.size()) + 1)
        # play action
        game.play_action(action)
        if args["game_display"]
            game.display()
            sleep(0.2)
        end
    end
    game.display()
    @info "Game score : $(game.score())"
end

if abspath(PROGRAM_FILE) == @__FILE__
    m = Model(args["game_size"], channels=args["model_channels"])
    _play_model(m)
end
