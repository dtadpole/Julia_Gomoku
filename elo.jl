include("./args.jl")

import YAML

const _ELO_K_VALUE = 32.0f0
const _ELO_INIT_RATING = 2000

"""Elo path"""
elo_filename = (file_type::String; size = args["game_size"]) -> begin
    return "./trained/$(size)x$(size)/elo_$(file_type).yml"
end


mutable struct Elo

    # attributes
    _ratings::Vector{Int}
    _history::Vector{Tuple{Int,Int,Float32}}
    _activePlayers::Set{Int}
    _candidatePlayers::Set{Int}
    _log::Vector{String}

    # functions
    # players
    newPlayer::Function
    playerInfo::Function
    savePlayers::Function
    loadPlayers::Function
    # ratings
    rating::Function
    ratings::Function
    saveRatings::Function
    loadRatings::Function
    # history
    history::Function
    saveHistory::Function
    loadHistory::Function
    # active players
    activePlayers::Function
    activeSize::Function
    activeAvgRating::Function
    makeActive::Function
    makeInactive::Function
    # candidate players
    candidatePlayers::Function
    candidateSize::Function
    randCandidate::Function
    makeCandidate::Function
    clearCandidates::Function
    # match making
    sampleMatch::Function
    selfMatch::Function
    # game results
    addGame::Function
    # log
    saveLog::Function
    loadLog::Function


    # constructor
    function Elo()

        e = new()
        e._history = Vector{Tuple{Int,Int,Float32}}()
        e._ratings = Vector{Float32}()
        e._activePlayers = Set{Int}()
        e._candidatePlayers = Set{Int}()
        e._log = Vector{String}()

        """Add player.  Returns player id"""
        e.newPlayer = (; init_rating::Int=_ELO_INIT_RATING) -> begin
            push!(e._ratings, init_rating)
            return length(e._ratings)
        end

        """Player info"""
        e.playerInfo = () -> begin
            dict = Dict(
                "average" => e.activeAvgRating(),
                "active" => map(x -> [x, e._ratings[x]], collect(e._activePlayers)),
                "candidate" => map(x -> [x, e._ratings[x]], collect(e._candidatePlayers)),
            )
            return dict
        end

        """Save active and candidate players"""
        e.savePlayers = (path::String) -> begin
            backup_file(path)
            YAML.write_file(path, e.playerInfo())
        end

        """Load active and candidate players"""
        e.loadPlayers = (path::String) -> begin
            dict = YAML.load_file(path)
            e._activePlayers = Set{Int}(map(x -> x[1], (dict["active"])))
            e._candidatePlayers = Set{Int}(map(x -> x[1], (dict["candidate"])))
            return dict
        end

        """Rating"""
        e.rating = (id) -> e._ratings[id]

        """Ratings"""
        e.ratings = () -> e._ratings

        """Save ratings"""
        e.saveRatings = (path::String) -> begin
            backup_file(path)
            YAML.write_file(path, e._ratings)
        end

        """Load ratings"""
        e.loadRatings = (path::String) -> begin
            e._ratings = YAML.load_file(path)
        end

        """History"""
        e.history = () -> e._history

        """Save history"""
        e.saveHistory = (path::String) -> begin
            backup_file(path)
            YAML.write_file(path, map(x -> collect(x), e._history))
        end

        """Load history"""
        e.loadHistory = (path::String) -> begin
            e._history = map(x -> Tuple(x), YAML.load_file(path))
        end

        """Active players"""
        e.activePlayers = () -> e._activePlayers

        """Active size"""
        e.activeSize = () -> length(e._activePlayers)

        """Active average rating"""
        e.activeAvgRating = () -> e.activeSize() == 0 ? 2000 : round(Int, sum(map(x -> e._ratings[x], collect(e._activePlayers))) / e.activeSize())

        """Make player active"""
        e.makeActive = (id) -> begin
            # check if player is candidate
            if id ∈ e._candidatePlayers
                delete!(e._candidatePlayers, id) # delete from candidate list
                push!(e._log, "Player $(id) removed from candidate.  $(e.playerInfo()["candidate"])")
            end
            # check if player is active
            if id ∉ e._activePlayers
                push!(e._activePlayers, id)
                push!(e._log, "Player $(id) added to active.  $(e.playerInfo()["active"])")
            end
        end

        """Make player inactive"""
        e.makeInactive = (id) -> begin
            if id ∈ e._activePlayers
                delete!(e._activePlayers, id)
                push!(e._log, "Player $(id) removed from active.  $(e.playerInfo()["active"])")
            end
        end

        """Candidate players"""
        e.candidatePlayers = () -> e._candidatePlayers

        """Candidate size"""
        e.candidateSize = () -> length(e._candidatePlayers)

        """Random candidate"""
        e.randCandidate = () -> begin
            if length(e._candidatePlayers) == 0
                return nothing
            end
            return rand(collect(e._candidatePlayers))
        end

        """Make player candidate"""
        e.makeCandidate = (id) -> begin
            if id ∉ e._candidatePlayers
                push!(e._candidatePlayers, id)
                push!(e._log, "Player $(id) added to candidate.  $(e.playerInfo()["candidate"])")
            end
        end

        """Clear candidate players"""
        e.clearCandidates = () -> begin
            if length(e._candidatePlayers) > 0
                push!(e._log, "Candidate players cleared.  $(e.playerInfo()["candidate"]) => []")
                empty!(e._candidatePlayers)
            end
        end

        """Sample match"""
        e.sampleMatch = () -> begin
            if length(e._activePlayers) < 2
                error("Not enough active players [$(length(e._activePlayers))]")
            end
            # sample two active players
            active_players = collect(e._activePlayers)
            id_1 = active_players[rand(1:length(active_players))]
            id_2 = active_players[rand(1:length(active_players))]
            # make sure they are different
            while id_1 == id_2
                id_2 = active_players[rand(1:length(active_players))]
            end
            return id_1, id_2
        end

        """Self match"""
        e.selfMatch = (id) -> begin
            return id, id
        end

        """Add game.  Score ∈ [1.0, 0.0, -1.0]"""
        e.addGame = (id_1::Int, id_2::Int, score::Float32) -> begin
            # update ratings only if not self play
            if id_1 != id_2
                # calculate E
                e1 = 1.0f0 / (1.0f0 + 10.0f0^((e._ratings[id_2] - e._ratings[id_1]) / 400.0f0))
                e2 = 1.0f0 / (1.0f0 + 10.0f0^((e._ratings[id_1] - e._ratings[id_2]) / 400.0f0))
                # normalize score and calculate S
                normalized_score = (score + 1.0f0) / 2.0f0
                s1 = normalized_score
                s2 = 1.0f0 - normalized_score
                # calculate new ratings
                r1_new = e._ratings[id_1] + _ELO_K_VALUE * (s1 - e1)
                r2_new = e._ratings[id_2] + _ELO_K_VALUE * (s2 - e2)
                # update ratings
                e._ratings[id_1] = round(Int, r1_new)
                e._ratings[id_2] = round(Int, r2_new)
            end
            # update history
            push!(e._history, (id_1, id_2, score))
        end

        """Save log"""
        e.saveLog = (path::String) -> begin
            backup_file(path)
            YAML.write_file(path, e._log)
        end

        """Load log"""
        e.loadLog = (path::String) -> begin
            e._log = YAML.load_file(path)
        end

        return e
    end

end

if abspath(PROGRAM_FILE) == @__FILE__

    elo = Elo()
    SIZE = 40

    # add players
    for i = 1:SIZE
        elo.newPlayer(init_rating=i * 100)
    end
    # make players active
    for i = 1:2:SIZE
        elo.makeActive(i)
    end
    # make players candidate
    for i = 2:2:SIZE
        elo.makeCandidate(i)
    end

    # sample match
    id_s1, id_s2 = elo.sampleMatch()
    println("Sample match: $(id_s1) vs $(id_s2)")

    # self match
    id_f1, id_f2 = elo.selfMatch(rand(1:10))
    println("Self match: $(id_f1) vs $(id_f2)")

    # make active players inactive
    for i = 1:4:SIZE
        elo.makeInactive(i)
    end

    # make candidate players active
    for i = 2:4:SIZE
        elo.makeActive(i)
    end

    # add games
    for i = 1:SIZE*5
        id1, id2 = elo.sampleMatch()
        elo.addGame(id1, id2, 1.0f0)
        id3, id4 = elo.sampleMatch()
        elo.addGame(id3, id4, 0.0f0)
        id5, id6 = elo.sampleMatch()
        elo.addGame(id5, id6, -1.0f0)
    end

    # clear candidates
    elo.clearCandidates()

    # save ratings
    elo.saveRatings(elo_filename("rating"))
    # load ratings
    @info "Ratings" elo.loadRatings(elo_filename("rating"))
    # save history
    elo.saveHistory(elo_filename("history"))
    # load history
    @info "History" elo.loadHistory(elo_filename("history"))
    # save players
    elo.savePlayers(elo_filename("player"))
    # load players
    @info "Players" elo.loadPlayers(elo_filename("player"))
    # save log
    elo.saveLog(elo_filename("log"))
    # load log
    @info "Log" elo.loadLog(elo_filename("log"))

end
