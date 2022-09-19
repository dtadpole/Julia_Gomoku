# gomoku_julia

Julia implementation of Gomoku following AlphaZero MCTS

# How to play

Human against best trained

```
./eval.sh --game_size 11 --player_1 human [--player_2 best]
```

Best trained against best trained (self play)

```
./eval.sh --game_size 11 --player_1 best [--player_2 best]
```

# How to train

First, start training server

```
./train.sh --game_size 7 [--model_cuda 0]
```

Next, start inference game players.  Start as many as GPU memory allows

```
./infer.sh --exp_server localhost [--model_cuda 0]
```
