include("./args.jl")

using Printf
using StatsBase

mutable struct Game

    # attributes
    _size::Int
    _turn::Float32
    _board::Array{Float32,2}
    _last_move::Tuple{Int,Int}
    _done::Bool

    # functions
    size::Function
    state::Function
    turn::Function
    available_actions::Function
    play::Function
    play_action::Function
    str_action::Function
    is_over::Function
    score::Function
    has_won::Function
    display::Function
    clone::Function

    # constructor
    function Game(size::Int; turn=1.0)
        g = new(size, sign(turn), zeros(Float32, size, size), (0, 0), false)

        """Size"""
        g.size = () -> g._size

        """State"""
        g.state = () -> g._board .* g.turn()

        """Turn"""
        g.turn = () -> sign(g._turn)

        """Available actions"""
        g.available_actions = () -> abs.(abs.(g.state()) .- 1.0)

        """Play a move at (x, y)"""
        g.play = (x::Int, y::Int) -> begin
            if g._done
                error("Game is over")
            end
            if g._board[x, y] != 0.0
                error("Invalid move [$(x), $(y)]")
            end
            g._board[x, y] = sign(g._turn)
            g._last_move = (x, y)
            g._turn = -g._turn
            if g.has_won()
                g._done = true
            end
        end

        """Play an action"""
        g.play_action = (action::Base.AbstractCartesianIndex) -> begin
            g.play(action[1], action[2])
        end

        """action string"""
        g.str_action = (action::Int) -> begin
            return "$(div(action-1,size)+'A')$(mod(action-1,size)+1)"
        end

        """Check if the game is over."""
        g.is_over = () -> begin
            if g._done
                return true
            end

            # check if the last move is a winning move
            if g.has_won()
                g._done = true
                return true
            end

            # check if there are any available moves
            if sum(g.available_actions()) == 0.0
                g._done = true
                return true
            end

            return false
        end

        """Return the score of the game."""
        g.score = () -> begin
            if g.has_won()
                return -sign(g._turn)
            else
                return Float32(0.0)
            end
        end

        """Check if there are n pieces in a row including (row, col) in the
        direction (dir_row, dir_col)."""
        _is_n_in_a_row = (g::Game, row::Int, col::Int, dir_row::Int, dir_col::Int; n=5) -> begin
            count = 0
            color = row <= 0 || col <= 0 ? sign(g._turn) : sign(g._board[row, col])
            for i in -n+1:n-1
                curr_row = row + i * dir_row
                if curr_row <= 0 || curr_row > g._size
                    continue
                end

                curr_col = col + i * dir_col
                if curr_col <= 0 || curr_col > g._size
                    continue
                end

                if g._board[curr_row, curr_col] == color
                    count += 1
                    if count >= n
                        # println("row [$(row)], col [$(col)], dir_row [$(dir_row)], dir_col [$(dir_col)]")
                        return true
                    end
                else
                    count = 0
                end
            end

            return false
        end

        """Check if the last move has won the game."""
        g.has_won = () -> begin
            (row, col) = g._last_move
            if row <= 0 || col <= 0
                return false
            end
            if _is_n_in_a_row(g, row, col, 1, 0)
                return true
            elseif _is_n_in_a_row(g, row, col, 0, 1)
                return true
            elseif _is_n_in_a_row(g, row, col, 1, 1)
                return true
            elseif _is_n_in_a_row(g, row, col, 1, -1)
                return true
            else
                return false
            end
        end

        """Display the game board."""
        g.display = () -> begin
            displayLine() = begin
                # print("    -")
                print("       ")
                for j in 1:g._size
                    # print(" $(args["game_spot"]) -")
                    print(" $(args["game_space"]) ")
                end
                println(" ")
            end
            displayLegend() = begin
                print("       ")
                for i in 1:g._size
                    # print("$(args["game_spot"]) $('A' + i - 1) ")
                    print(" $('Ａ' + i - 1) ")
                end
                println()
            end
            println()
            displayLegend()
            for i in 1:g._size
                displayLine()
                # @printf(" %-2d |", i)
                @printf("    %-2d ", i)
                for j in 1:g._size
                    if g._board[i, j] > 0
                        if (i, j) == g._last_move
                            # print("($(args["game_X"]))|")
                            print("($(args["game_X"]))")
                        else
                            # print(" $(args["game_X"]) |")
                            print(" $(args["game_X"]) ")
                        end
                    elseif g._board[i, j] < 0
                        if (i, j) == g._last_move
                            # print("($(args["game_O"]))|")
                            print("($(args["game_O"]))")
                        else
                            # print(" $(args["game_O"]) |")
                            print(" $(args["game_O"]) ")
                        end
                    else
                        # print(" $(args["game_spot"]) |")
                        print(" $(args["game_spot"]) ")
                    end
                end
                @printf(" %2d ", i)
                println()
            end
            displayLine()
            displayLegend()
            println()
        end

        """Clone the game."""
        g.clone = () -> begin
            g2 = Game(g._size, turn=g._turn)
            g2._board = copy(g._board)
            g2._last_move = g._last_move
            g2._done = g._done
            return g2
        end

        return g
    end
end


"""Play out a game and return the winner."""
function _random_playout(g::Game)
    SIZE = g.size()
    while !g.is_over()
        moves = reshape(g.available_actions(), SIZE^2)
        move = sample([(i, j) for i in 1:SIZE, j in 1:SIZE], ProbabilityWeights(moves))
        g.play(move[1], move[2])
        if args["game_display"]
            g.display()
            sleep(0.2)
        end
    end
    # game end
    g.display()
    score = g.score()
    println("score [$(score)]")
    return score
end

if abspath(PROGRAM_FILE) == @__FILE__
    g = Game(args["game_size"])
    _random_playout(g)
end
