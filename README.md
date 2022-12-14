# Julia Gomoku

Julia implementation of Gomoku following AlphaZero MCTS

# How to play

Human vs best trained model (player 2 default is 'best')

```
./eval.sh --game_size 11 --player_1 human [--player_2 best]
```

Best trained model vs best trained model (machine self play)

```
./eval.sh --game_size 11 --player_1 best [--player_2 best]
```

# How to train

First, start the training server.

Optionally use 'model_cuda' parameters to specify GPU number.

```
./train.sh --game_size 7 [--model_cuda 0]
```

Next, start (multiple) inference game players.

Start as many inference players as resource allows (GPU memory, CPU cores, etc).

```
./infer.sh --exp_server localhost [--model_cuda 0]
```

While 7x7 game could be trained with CPU, bigger boards will benefit from GPUs.
