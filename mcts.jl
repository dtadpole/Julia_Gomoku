include("./model.jl")

using Distributions

function maxk(a, k)
    b = partialsortperm(a, 1:k, rev=true)
    return collect(zip(b, a[b]))
end

function rotate90(a::Matrix)
    reverse(transpose(a), dims=2)
end

function rotate180(a::Matrix)
    reverse(reverse(a, dims=2), dims=1)
end

function rotate270(a::Matrix)
    reverse(transpose(a), dims=1)
end

function flip(a::Matrix; dims=2)
    reverse(a, dims=dims)
end

function random_transform(a::Matrix)
    r = rand(1:8)
    if mod(r, 4) == 1
        data = a
    elseif mod(r, 4) == 2
        data = rotate90(a)
    elseif mod(r, 4) == 3
        data = rotate180(a)
    else
        data = rotate270(a)
    end
    if r > 4
        data = flip(data)
    end
    return data, r
end

function revert_transform(a::Matrix, r::Int)
    if r > 4
        data = flip(a)
    else
        data = a
    end
    if mod(r, 4) == 1
        return data
    elseif mod(r, 4) == 2
        return rotate270(data)
    elseif mod(r, 4) == 3
        return rotate180(data)
    else
        return rotate90(data)
    end
end


mutable struct MctsNode

    # attributes
    _game::Game
    _model::Model
    _cpuct::Float32
    _gamma::Float32
    _parent::Union{MctsNode,Nothing}
    _children::Union{Matrix{Union{MctsNode,Nothing}},Nothing}
    _N::Int
    _P::Float32
    _Q::Float32
    # _U::Float32
    _Pi_V::Union{Tuple{Matrix{Float32},Float32},Nothing}
    _noise_epsilon::Float32
    _dirichlet::Union{Dirichlet,Nothing}

    # functions
    size::Function
    parent::Function
    clearParent::Function
    children::Function
    expandChildren::Function
    nodeValue::Function
    childrenValues::Function
    updateBranch::Function
    travelBranch::Function
    rootN::Function
    rootQ::Function

    # constructor
    function MctsNode(game::Game, model::Model; cpuct=3.0, gamma=0.98, parent=nothing, P=0.0, noise_epsilon=0.25, dirichlet=nothing)

        node = new(game, model, cpuct, gamma, parent, nothing, 0, P, 0.0, nothing, noise_epsilon, dirichlet)

        """Size"""
        node.size = () -> node._game.size()

        """Parent"""
        node.parent = () -> node._parent

        """Clear parent"""
        node.clearParent = () -> begin
            node._parent = nothing
        end

        """Children"""
        node.children = () -> node._children

        function _calc_Pi_V()
            SIZE = node._game.size()
            if node._Pi_V === nothing
                state = node._game.state()
                state, r = random_transform(state) # random transform
                if args["model_cuda"] >= 0
                    state = state |> gpu
                end
                pi, v = node._model.forward(reshape(state, (node.size(), node.size(), 1, 1)))
                pi = revert_transform(reshape(Array(pi), (node.size(), node.size())), r) # revert transform
                # apply dirichlet noise to prior probabilities
                # dirichlet = node._dirichlet === nothing ? fill(0.0f0, node.size(), node.size()) : reshape(rand(node._dirichlet), node.size(), node.size())
                # pi = (pi .* (1 - node._noise_epsilon) .+ dirichlet .* node._noise_epsilon) .* node._game.available_actions()
                pn = reshape(pi, SIZE*SIZE) .* (SIZE*SIZE*args["mcts_dirichlet_multiplier"])
                node._dirichlet = Dirichlet(pn)
                v = Array(v)[1]
                node._Pi_V = (pi, v)
            end
            pi_ = reshape(rand(node._dirichlet), (SIZE, SIZE)) .* node._game.available_actions()
	    v_ = node._Pi_V[2]
	    return pi_, v_
            # return node._Pi_V
        end

        """Expand children"""
        node.expandChildren = () -> begin
            if node._game.is_over()
                error("Cannot expand children - game is over")
            end
            pi, _ = _calc_Pi_V()
            node._children = Array{Union{MctsNode,Nothing},2}(nothing, node.size(), node.size())
            for i = 1:node.size(), j = 1:node.size()
                if pi[i, j] > 0.0
                    game = node._game.clone()
                    game.play_action(CartesianIndex(i, j))
                    node._children[i, j] = MctsNode(game, node._model; cpuct=node._cpuct, gamma=node._gamma, parent=node, P=pi[i, j], noise_epsilon=node._noise_epsilon, dirichlet=node._dirichlet)
                end
            end
        end

        """Node Value : Q(S,a) + cpuct * P(S,a) * sqrt(sum(N(S,x) for x in siblings)) / (1 + N(S,a))"""
        node.nodeValue = () -> node._Q + node._cpuct * node._P * sqrt(node._parent._N) / (1 + node._N)

        """Children Node Value : Q(S,a) + cpuct * P(S,a) * sqrt(sum(N(S,x) for x in siblings)) / (1 + N(S,a))"""
        node.childrenValues = () -> begin
            if node._children === nothing
                error("Cannot get children values - children not expanded")
            end
            values = fill(-Inf32, node.size(), node.size())
            # dirichlet = node._dirichlet === nothing ? fill(0.0f0, node.size(), node.size()) : reshape(rand(node._dirichlet), node.size(), node.size())
            for i in 1:node.size(), j in 1:node.size()
                if node._children[i, j] !== nothing
                    # values[i, j] = node._children[i, j]._Q + node._cpuct * (node._children[i, j]._P * (1 - node._noise_epsilon) + dirichlet[i, j] * node._noise_epsilon) * sqrt(node._N) / (1 + node._children[i, j]._N)
                    values[i, j] = node._children[i, j]._Q + node._cpuct * node._children[i, j]._P * sqrt(node._N) / (1 + node._children[i, j]._N)
                end
            end
            return values
        end

        """Update branch"""
        node.updateBranch = (v::Float32) -> begin
            node._Q = (node._Q * node._N + v) / (node._N + 1)
            node._N += 1
            if node._parent !== nothing
                node._parent.updateBranch(-v * node._gamma)
            end
        end

        """Travel branch"""
        node.travelBranch = (depth::Int = 1_000) -> begin
            if node._game.is_over()
                node.updateBranch(abs(node._game.score()))
            elseif depth <= 0
                _, v = _calc_Pi_V()
                node.updateBranch(v)
            else
                if node._children === nothing
                    node.expandChildren()
                end
                values = node.childrenValues()
                index = argmax(values)
                node._children[index].travelBranch(depth - 1)
            end
        end

        """Root N distribution"""
        node.rootN = () -> begin
            pi = zeros(Int, node.size(), node.size())
            for i in 1:node.size(), j in 1:node.size()
                if node._children[i, j] !== nothing
                    pi[i, j] = node._children[i, j]._N
                end
            end
            return pi
        end

        """Root Q"""
        node.rootQ = () -> begin
            return node._Q
        end

        return node

    end

end

function mcts_play_game(model_1::Model, model_2::Model)

    if model_1.size() != model_2.size()
        error("Models must have the same size")
    end

    SIZE = model_1.size()

    TEMPERATURE = args["mcts_temperature_mean"] + args["mcts_temperature_std"] * randn()

    dirichlet = Dirichlet(SIZE^2, args["mcts_noise_alpha"]) # create dirichlet distribution for noise

    experiences = Array{Tuple{Array{Float32,2},Array{Float32,2},Float32},1}()

    # initialize game with random turn
    game_init_turn = rand() > 0.5 ? 1.0 : -1.0
    game = Game(SIZE, turn=game_init_turn)

    # create mcts node
    node_1 = MctsNode(game, model_1; cpuct=args["mcts_cpuct"], gamma=args["mcts_gamma"], noise_epsilon=args["mcts_noise_epsilon"], dirichlet=dirichlet)
    node_2 = MctsNode(game, model_2; cpuct=args["mcts_cpuct"], gamma=args["mcts_gamma"], noise_epsilon=args["mcts_noise_epsilon"], dirichlet=dirichlet)

    # choose which node to travel
    if game.turn() > 0
        node = node_1
        node_opposite = node_2
        model = model_1
        model_opposite = model_2
    else
        node = node_2
        node_opposite = node_1
        model = model_2
        model_opposite = model_1
    end

    while !game.is_over()

        if args["game_display"]
            start_time = time()
        end

        # play
        prior_N = node._N
        while node._N < game.size()^2 * args["mcts_n_multiplier"]
            node.travelBranch(args["mcts_depth"])
            # GC & reclaim CUDA memory
            if mod(node._N, 250) == 0
                GC.gc(false)
                if args["model_cuda"] >= 0
                    CUDA.reclaim()
                end
            end
        end
        post_N = node._N

        # get distribution
        pi = reshape(node.rootN(), (SIZE, SIZE))
        v = node.rootQ()
        push!(experiences, (game.state(), pi, v))

        # play action
        if TEMPERATURE > 1e-4
            # pa = softmax(log.(pi .+ 1e-8) / TEMPERATURE, dims=[1, 2])
            pa = softmax(log.(pi) / TEMPERATURE, dims=[1, 2])
            action = sample(1:SIZE^2, ProbabilityWeights(reshape(pa, SIZE^2)))
            action = CartesianIndex(mod(action - 1, SIZE) + 1, div(action - 1, SIZE) + 1)
        else
            action = argmax(pi)
        end

        game = game.clone()
        game.play_action(action)
        if args["game_display"]
            duration = round(time() - start_time, digits=1)
            @info "[$(-round(node.rootQ(), digits=2)),ν] [$(join(["$(game.str_action(a)):$(n)" for (a, n) in maxk(reshape(node.rootN(), SIZE^2), 5)], ", "))] [$(prior_N)->$(post_N), $(round(TEMPERATURE, digits=2))τ, $(duration)s] "
            # display
            game.display()
        end

        # update mcts node
        node = node._children[action]
        node.clearParent()

        # update mcts node opposite
        if node_opposite._children !== nothing && node_opposite._children[action] !== nothing
	    # DO Reuse !! Dirichlet noises are dynamically added each step
            node_opposite = node_opposite._children[action]
            node_opposite.clearParent()
	    # Do NOT Reuse !!  always create new mcts tree, do not cache dirichlet noise
            # node_opposite = MctsNode(game, model_opposite; cpuct=args["mcts_cpuct"], gamma=args["mcts_gamma"], noise_epsilon=args["mcts_noise_epsilon"], dirichlet=dirichlet)
        else
            node_opposite = MctsNode(game, model_opposite; cpuct=args["mcts_cpuct"], gamma=args["mcts_gamma"], noise_epsilon=args["mcts_noise_epsilon"], dirichlet=dirichlet)
        end

        # swap node and node_opposite
        node, node_opposite = node_opposite, node
        model, model_opposite = model_opposite, model

        # GC & reclaim CUDA memory
        GC.gc(false)
        if args["model_cuda"] >= 0
            CUDA.reclaim()
        end

    end

    return experiences, game_init_turn, game.score()

end

if abspath(PROGRAM_FILE) == @__FILE__
    # create model
    model_1 = Model(args["game_size"], channels=args["model_channels"])
    model_2 = Model(args["game_size"], channels=args["model_channels"])
    mcts_play_game(model_1, model_2)
end
