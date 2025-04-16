Turtle Snake Game with Reinforcement Learning (DQN)

Welcome to the Turtle Snake Game project! This game is based on the classic Snake game, but with an additional layer of complexity where the snake is trained using Reinforcement Learning (RL) to navigate the grid and collect food autonomously. This project employs both traditional Q-learning and advanced Deep Q-Networks (DQN) to enable the agent to learn optimal behavior.

Overview

The Turtle Snake Game is a simple implementation of the Snake game using the Pygame library. The game involves a snake that moves around the grid, collecting food, and growing longer with each food item it eats. The objective is to avoid running into walls or the snake’s own body. The game is controlled using a Reinforcement Learning agent that learns optimal policies to maximize its reward by collecting food without colliding.

Key Features
	•	Pygame implementation: The core mechanics of the snake game are implemented using Pygame.
	•	Reinforcement Learning (RL): The snake learns to navigate the grid using both Q-learning and Deep Q-Networks (DQN).
	•	DQN Optimization: After extensive training using Q-learning (16+ hours with minimal progress), the agent has been switched to a DQN model for better performance.
	•	Backup Path Strategy: The snake is designed to follow a Hamiltonian cycle path when no optimal policy is available, ensuring safe movement in complex situations.

Project Structure

Turtle_snake_game/
├── README.md                # Project overview and details
├── RL_Agent_with_DQN.py     # Implementation of RL Agent with DQN
├── RL_model_optimizing_Q.py # Optimization of Q-values in RL
├── Rl_model.py              # Core RL model for training the agent
├── __pycache__              # Compiled Python files
├── csv_files                # Training data and CSV files for Q-table
├── old_q_table_files        # Older Q-tables (training data used in Q-learning)
├── weight_file_for_DQN     # Saved weights for the trained DQN model
├── example.py               # Example of the game with basic functionality
├── food.py                  # Game logic for food generation
├── highscore_for_snake_game.txt # Stores the highscore for the game
├── main.py                  # Main entry point for the game (running the RL model)
├── scoreboard.py            # Logic to display and track the score
├── snake.py                 # Snake class and movement logic
├── snake_game.py            # Full game logic for running the Snake game
└── current_q_table.csv      # The current Q-table used in training (before DQN)

Training Process

Q-Learning (Initial Attempts)

Initially, I trained the snake using Q-learning, spending over 16 hours optimizing the Q-table. However, due to the complexity of the game environment, the agent struggled to generalize its learning. The snake often failed to find optimal paths and would get stuck moving around food, even after fine-tuning the rewards.

Transition to DQN

After recognizing the limitations of traditional Q-learning, I transitioned to using Deep Q-Networks (DQN). DQN leverages neural networks to approximate Q-values and handle large state spaces more effectively. While the DQN model is still a work in progress, it has shown better performance compared to Q-learning, and the agent is steadily learning to navigate the game environment.

Key Challenges
	•	State-space complexity: The large number of possible states in the game made it difficult for Q-learning to find a solution.
	•	Food and collision detection: Despite optimized rewards, the snake often moved inefficiently or got stuck in loops.
	•	Training time: The switch to DQN has significantly reduced training time compared to Q-learning, although there are still challenges to overcome before the game is fully completed.

Current Status
	•	The snake is currently being trained using DQN and is gradually improving.
	•	The Hamiltonian cycle strategy is being explored as a fallback to prevent the snake from getting stuck in complex situations.
	•	The game mechanics are complete, but there are still optimizations to be made, especially around the DQN model and its convergence.
	•	Training is ongoing, and further improvements are expected as more data is collected.

How to Run the Game

To run the snake game with the RL agent:
	1.	Clone the repository:

git clone https://github.com/yourusername/Turtle_snake_game.git
cd Turtle_snake_game


	2.	Install dependencies (make sure you have Python 3 and Pygame installed):

pip install pygame


	3.	Run the game:

python main.py


	4.	The snake will automatically start training if a trained model is not found, or you can continue training by re-running the game.

Future Work
	•	Policy Improvement: Continue optimizing the DQN model to improve the agent’s decision-making process.
	•	Enhanced Agent Behavior: Implement more sophisticated reward shaping to speed up convergence.
	•	Use of Hamiltonian Cycle: Incorporate Hamiltonian cycle logic for fail-safe movement.
	•	User Interface: Enhance the game interface for better user interaction and visualization of training progress.

Contributions

Feel free to contribute by:
	•	Suggesting improvements to the RL model.
	•	Proposing optimizations for the game’s mechanics.
	•	Reporting any bugs or issues you encounter.

⸻

I hope this README.md captures the essence of your work and provides a clear structure for your project. If you need further tweaks or additions, feel free to let me know!
