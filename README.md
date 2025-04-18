# Snake Game with Reinforcement Learning

Welcome to my **Snake Game** project! This game is based on the classic Snake game, but with an additional layer of complexity where the snake is trained using **Reinforcement Learning (RL)** to navigate the grid and collect food autonomously. This project employs both traditional **Q-learning** and advanced **Deep Q-Networks (DQN)** to enable the agent to learn optimal behaviour.

## Overview

The **Snake Game** is a simple implementation of the Snake game using the **Pygame** library. The game involves a snake that moves around the grid, collecting food, and growing longer with each food item it eats. The objective is to avoid running into walls or the snake's own body. The game is controlled using a Reinforcement Learning agent that learns optimal policies to maximise its reward by collecting food without colliding.

### Key Features
- **Game implementation**: Originally, I built the game using the Python Turtle graphics, but I could not rely on it to implement the RL model due to it being slow and complex
- **Pygame implementation**: The core mechanics of the snake game are implemented using Pygame of which i had to learn in a short period, so it is not perfect 
- **Reinforcement Learning (RL)**: The snake learns to navigate the grid using both **Q-learning** and **Deep Q-Networks (DQN)**.
- **DQN Optimisation**: After extensive training using Q-learning (16+ hours with minimal progress), the agent has been switched to a DQN model for better performance.
- **Backup Path Strategy**: The snake is designed to follow a **saved path** when no optimal policy is available, ensuring safe movement in complex situations.
- The saved paths are inside some specifically named files, e.g "old_q_table_files", "weight_file_for_DQN"..............

## Project Structure
```
Turtle_snake_game
pygame_snake_game/
├── Readme.md                # Project overview and details
├── RL_Agent_with_DQN.py     # Implementation of RL Agent with DQN
├── RL_model_optimizing_Q.py # Optimization of Q-values in RL
├── Rl_model.py              # Original RL model with just Q-Learning for training the agent
├── csv_files                # Training data and CSV files for Q-table
├── old_q_table_files        # Older Q-tables (training data used in Q-learning)
├── weight_file_for_DQN     # Saved weights for the trained DQN model
├── example.py               # Example of using pygame
├── food.py                  # Game logic for food generation
├── highscore_for_snake_game.txt # Stores the highscore for the game
├── main.py                  # Human playable game file
├── scoreboard.py            # Logic to display and track the score
├── snake.py                 # Snake class and movement logic
├── snake_game.py            # Full game logic for running the Snake game
└── current_q_table.csv      # The current Q-table used in training (before DQN)
```

## Training Process

### Q-Learning (Initial Attempts)
Initially, I tried to train the snake using **Q-learning**, spending over **16 hours** optimising the Q-table. However, due to the complexity of the game environment, the agent struggled to generalise its learning. The snake often failed to find optimal paths and would get stuck moving around food. Even after fine-tuning the rewards, it still preferred staying near the food and accumulating rewards for being close for each step; it was like it was cheating to gain rewards.

### Transition to DQN
After recognising the limitations of traditional Q-learning, I decided to use **Deep Q-Networks (DQN)**. DQN leverages neural networks to approximate Q-values and handle large state spaces more effectively. Unfortunately, I am not so good with DQN so I had to outsource this part. While the DQN model is still a work in progress, it has shown better performance compared to Q-learning, and the agent is steadily learning to navigate the game environment.

### Key Challenges
- **State-space complexity**: The large number of possible states in the game made it difficult for Q-learning to find a solution.
- **Food and collision detection**: Despite optimised rewards, the snake often moved inefficiently or got stuck in loops.
- **Training time**: The switch to DQN has significantly reduced training time compared to Q-learning, although there are still challenges to overcome before the game is fully completed.

## Current Status

- The snake is currently being trained using **DQN** and is gradually improving.
- I am going to research the Hamiltonian cycle strategy to implement as a fallback to prevent the snake from getting stuck in complex situations it was invented by <a href="https://openstax.org/books/contemporary-mathematics/pages/12-7-hamilton-cycles#:~:text=In%201857%2C%20a%20mathematician%20named,visited%20every%20vertex%20exactly%20once.">William Hamilton</a>
	- I got this idea from <a href="https://www.youtube.com/watch?v=tjQIO1rqTBE">this youtube video</a> but the method was gotten from <a href="https://johnflux.com/page/2/">John Tapsell</a>
- The game mechanics are complete, but there are still optimisations to be made, especially around the DQN model and its convergence.
- Training is ongoing, and further improvements are expected as more data is collected.

## Training Videos

### Training with just Q-learning
https://github.com/user-attachments/assets/9a7807bd-ef47-43d8-91c9-7b2612a8846d

### Training with DQN


https://github.com/user-attachments/assets/c5fbc0c0-d70d-4c23-a66e-f8ea0ef7c8a0 



## How to Run the Game

To run the snake game with the RL agent:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/snake_game.git
   cd pygame_snake_game
   ```
2. Install dependencies (make sure you have Python 3 and Pygame installed):
   ```pip install pygame```
3. Run the game:
   ```python <file name>.py```  e.g Rl_Model
4. The snake will automatically start training. If a trained model file is not found, it will create one and start training from the beginning.
   
Future Work
	•	Policy Improvement: Continue optimising the DQN model to improve the agent’s decision-making process.
	•	Enhanced Agent Behaviour: Implement more sophisticated reward shaping to speed up convergence.
	•	Use of Hamiltonian Cycle: Incorporate Hamiltonian cycle logic for fail-safe movement.
	•	User Interface: Enhance the game interface for better user interaction and visualisation of training progress.

## Contributions
Feel free to contribute by:
	<li> Suggesting improvements to the RL model.</li>
	<li> Proposing optimisations for the game’s mechanics.</li>
	<li> Reporting any bugs or issues you encounter.</li>
 
 ## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.
