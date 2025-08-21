# Snake Game with Reinforcement Learning 🐍🤖  

## Overview
This project reimagines the classic **Snake Game** with **Reinforcement Learning (RL)**. The snake is trained to autonomously navigate the grid, collect food, and avoid collisions using a variety of RL algorithms.  

The implementation uses:  
- **Q-learning** (initial baseline)  
- **Deep Q-Networks (DQN)** for improved state generalisation  
- **Double DQN** to address Q-value overestimation  
- **Hamiltonian Cycle strategy** (with Prim’s algorithm and BFS fallback) to ensure safe traversal in complex board states  

## Key Features
- **Core Game**: Built in **Pygame** with custom snake, food, and scoreboard logic.  
- **Reinforcement Learning**:  
  - Q-learning (basic)  
  - DQN (neural network-based Q-learning)  
  - Double DQN (improves stability and performance)  
- **Fallback Strategies**:  
  - BFS-based safe path search for mid-game navigation  
  - Hamiltonian cycle generation (Prim’s algorithm) for long survival runs  
- **Saved Weights**: Automatically loads existing `.pth` or `.npy` weights for resuming training or running greedy play.  
- **Custom UI**: `GameOver` overlay with restart option.  
- **Experimentation**: Multiple play/test scripts (`Small Grid`, `Full Hamiltonian`, etc.) to evaluate different strategies.  

---

## Project Structure
```
**pygame_snake_game**/
├── **pygame_snake_game**
    ├── **AI_Snake**(Play)  
        ├── Full_Ham_implementation.py                  # Snake with Hamiltonian Cycle strategy
        ├── main.py                                     # Human Playable Game
        ├── Play_game(Small_grid).py                    # Play Snake on a 400x400 grid with RL + fallback
        ├── Play_game(Snake AI).py                      # Purely Snake AI with Double-Deep Q Networks
        └── Play_game_Advanced_Hamiltonian_cycle.py     # Play Snake on a 800x600 grid with RL + fallback
    ├── **game_attributes**
        ├── gameover.py                                       # GameOver screen logic
        └── snake.py, food.py, scoreboard.py, snake_game.py   # Core game logic
    ├── **Hamiltonian_Implementation**
        ├── ham_cycle.py
        ├── Hamiltonian_cycle.py
        ├── nan.py
        └── prims_algorithm.py
    ├── **RL_agents(Training)**
        ├── Rainbow-RL.py                     # Original RL code with just Q-Learning.
        ├── Rl_model.py                       # Pure Q-learning baseline
        ├── RL_Agent_with_DDQN_CNN.py         # Double-DQN agent implementation with a 7x7 or 5x5 Grid CNN
        ├── RL_model_optimizing_Q.py          # Optimisation for Q-learning
        └── SG_Double_DDQN_training.py        # Double DQN training on smaller grid
    └── **WEIGHTS**
    ├── old_q_table_files/            # Archived Q-tables
    ├── Current_q_TABLE/              # Current training tables/weights
    ├── RAINBOW_WEIGHTS/              # Training Weights for Rainbow Implementation
    └── weight_file_for_DQN/          # Saved DDQN model weights
├── **pygame_snake_game_gg_colab**        # This is a folder made specifically to train using google colab GPU for more complete algorithm-> work in progress
└── **Turtle_snake-game**                 # Turtle-game Version
```

---

## Training Process

### Q-Learning (Baseline)
- Ran for 16+ hours, struggled to generalise.  
- Agent often looped near food, exploiting reward shaping.  

### Deep Q-Network (DQN)
- Neural net approximates Q-values for large state space.  
- Much faster convergence compared to tabular Q-learning.  

### Double DQN
- Currently the main training algorithm.  
- Fixes overestimation of Q-values common in vanilla DQN.  
- Produces more stable and consistent snake behaviour.  

### Hamiltonian Cycle + Fallbacks
- Implemented **Prim’s algorithm** to generate Hamiltonian cycle.  
- Added **BFS safe path search** as a mid-game shortcut.  
- Uses **cycle rotation** to continue safe traversal when nearly full.  

### Current Focus
- Training and evaluating Double DQN agent.  
- Integrating Hamiltonian fallback for robust play.  
- Considering advanced algorithms (Rainbow DQN, etc.) for further performance boost.  

---

## How to Run

Clone the repo:
```bash
git clone https://github.com/kdavid001/snake_game.git
cd snake-game/pygame_snake_game
```
Install dependencies
```
pip install pygame torch numpy
```

Run specific Setup
```
python Play_game(Small_grid).py       # Double DQN on smaller grid
python Full_Ham_implementation.py     # With Hamiltonian fallback
python Rl_model.py                    # Tabular Q-learning baseline
```
## Training Results
| Model / Strategy                  | Highest score | Highest Mean Score | Notes |
|-----------------------------------|---------------|--------------------|-------|
| `Current DQN WEIGHTS/snake_dqn.pth` | 59            | N/A                | Baseline DQN training run |
| `Current DQN WEIGHTS/Weight_wn_Reward_sys.pth` | 67            | 1900               | Improved with reward shaping |
| `Current DQN WEIGHTS/Best_current_weight.pth.pth` | 52            | 1500               | Considered best performing DQN so far |
| `CNN_weights(5x5).pth`            | 32            | 1000               | CNN filter size 5x5 |
| `CNN_weights(7x7).pth`            | 16            | 600                | CNN filter size 7x7 |
| Hamiltonian Cycle                  | ∞             | ∞                  | Survives indefinitely once path is set |
| BFS – Breadth First Search         | 150           | N/A                | Reliable but not optimal |
| SG – 400×400 Grid                  | 54            | 1700               | Double DQN on small grid |
# Training Videos


# Future Works:  
- Further optimise Double DQN hyperparameters.
- Try Rainbow DQN and other advanced RL algorithms.
- Improve reward shaping to reduce looping behaviour.
- Enhance Pygame UI with visual overlays for agent decisions and cycle paths.  

## Contributions
Please Feel Free to contribute by Opening an issue.

 ## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

## Attribution

Credits for sources of inspiration:
- William Hamilton (for Hamiltonian cycles): [OpenStax link](https://openstax.org/books/contemporary-mathematics/pages/12-7-hamilton-cycles#:~:text=In%201857%2C%20a%20mathematician%20named,visited%20every%20vertex%20exactly%20once.)
- John Tapsell (Hamiltonian cycle method): [John Tapsell's blog](https://johnflux.com/page/2/)
- The YouTube video that inspired the Hamiltonian cycle idea: [YouTube link](https://www.youtube.com/watch?v=tjQIO1rqTBE)

This project was created by [David Ogunmola](https://github.com/kdavid001).  
If you use this project in any way (including derivatives or distributions), please include visible credit to the author in your documentation, app interface, or any public display of the software.

Thank you for respecting the work!
