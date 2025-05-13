import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

CHOOSE = 0
ANSWER = 1
DD = 2

random.seed(42)

class Player:
    def __init__(self, pid, smartness=0.8, dd_smartness = 0.75):
        self.score = 0
        self.pid = pid
        self.smartness = smartness
        self.dd_smartness = dd_smartness

    def change_score(self, score_change):
        self.score += int(score_change)
        return self.score

    def get_pid(self):
        return self.pid
    
class Jeopardy:
    def __init__(self, board_size, num_players, alpha, gamma, epsilon, decay):
        # Set up board
        self.board_size = board_size
        self.board = np.ones((board_size, board_size))

        self.picks = np.zeros_like(self.board)

        self.num_players = num_players

        for row_num in range(board_size):
            self.board[row_num,:] *= (row_num + 1)

        # Set up players
        self.players = []
        for pid in range(num_players):
            self.players.append(Player(pid))

        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.decay = decay
        self.min_epsilon = 0.05

        self.q_table = defaultdict(lambda: 0)

        # daily double
        if random.random() > 0.9:
            self.dd = (board_size-2, random.randint(0, board_size-1))
        else:
            self.dd = (board_size-1, random.randint(0, board_size-1))

        self.action_type = CHOOSE

        self.dd_found = False

        self.training = True

    def get_state(self, pid, wrong_guesses=0):
        
        board_state = tuple(int(self.board[i,:].sum()/(i+1)) for i in range(self.board_size))

        pids = [id for id in range(self.num_players)]
        pids.remove(pid)
        
        opponent_state = tuple(sorted([(self.players[p].score > self.players[pid].score) for p in pids]))
        
        player_state = self.players[pid].score

        dd_state = self.dd_found

        state = (self.action_type, board_state, opponent_state, player_state, dd_state, wrong_guesses)
        # state = (self.action_type, board_state, opponent_state, dd_state, wrong_guesses)

        return state

    def get_action_train(self, state, pid):

        if self.training:

            if state[0] == CHOOSE:
                
                # choices = list(zip(*np.nonzero(self.board)))
                choices = [i for i in range(len(state[1])) if state[1][i] != 0.0]

                if(random.random() < self.epsilon):
                    # return random.choice(choices)
                    row = random.choice(choices)
                    cols = np.nonzero(self.board[row])[0]
                    indices = [(row, col) for col in cols]
                    return random.choice(indices)

                q_values = [self.q_table[(state, choice)] for choice in choices]

                if(len(q_values) == 0):
                    # return random.choice(choices)
                    row = random.choice(choices)
                    cols = np.nonzero(self.board[row])[0]
                    indices = [(row, col) for col in cols]
                    return random.choice(indices)

                max_q = max(q_values)

                best_actions = [a for a, q in zip(choices, q_values) if q == max_q]

                row = random.choice(best_actions)
                cols = np.nonzero(self.board[row])[0]
                indices = [(row, col) for col in cols]
                return random.choice(indices)

                # return random.choice(best_actions)
            
            elif state[0] == ANSWER:

                choices = [0, 1] # 0 is don't try to answer, 1 is try to answer

                if(random.random() < self.epsilon):
                    return random.choice(choices)
                
                q_values = [self.q_table[(state, choice)] for choice in choices]

                if(len(q_values) == 0):
                    return random.choice(choices)

                max_q = max(q_values)

                best_actions = [a for a, q in zip(choices, q_values) if q == max_q]

                return random.choice(best_actions)
            
            elif state[0] == DD:

                if self.players[pid].score <= 0:
                    return 0
                
                choices = range(self.players[pid].score)

                if(random.random() < self.epsilon):
                    return random.choice(choices)

                q_values = [self.q_table[(state, choice)] for choice in choices]

                if(len(q_values) == 0):
                    return random.choice(choices)

                max_q = max(q_values)

                best_actions = [a for a, q in zip(choices, q_values) if q == max_q]

                return random.choice(best_actions)
            
        else:

            if state[0] == CHOOSE:
                
                # choices = list(zip(*np.nonzero(self.board)))
                choices = [i for i in range(len(state[1])) if state[1][i] != 0.0]

                q_values = [self.q_table[(state, choice)] for choice in choices]

                if(len(q_values) == 0):
                    # return random.choice(choices)
                    row = random.choice(choices)
                    cols = np.nonzero(self.board[row])[0]
                    indices = [(row, col) for col in cols]
                    return random.choice(indices)

                max_q = max(q_values)

                best_actions = [a for a, q in zip(choices, q_values) if q == max_q]

                row = random.choice(best_actions)
                cols = np.nonzero(self.board[row])[0]
                indices = [(row, col) for col in cols]
                return random.choice(indices)

                # return random.choice(best_actions)
            
            elif state[0] == ANSWER:

                choices = [0, 1] # 0 is don't try to answer, 1 is try to answer
                
                q_values = [self.q_table[(state, choice)] for choice in choices]

                if(len(q_values) == 0):
                    return random.choice(choices)

                max_q = max(q_values)

                best_actions = [a for a, q in zip(choices, q_values) if q == max_q]

                return random.choice(best_actions)
            
            elif state[0] == DD:

                if self.players[pid].score <= 0:
                    return 0
                
                choices = range(self.players[pid].score)

                q_values = [self.q_table[(state, choice)] for choice in choices]

                if(len(q_values) == 0):
                    return random.choice(choices)

                max_q = max(q_values)

                best_actions = [a for a, q in zip(choices, q_values) if q == max_q]

                return random.choice(best_actions)
            
    def get_best_q_value(self, state, pid):

        if state[0] == CHOOSE:
            
            # choices = list(zip(*np.nonzero(self.board)))
            choices = [i for i in range(len(state[1])) if state[1][i] != 0.0]

            q_values = [self.q_table[(state, choice)] for choice in choices]

            if(len(q_values) == 0):
                return 0

            max_q = max(q_values)

            return max_q
        
        elif state[0] == ANSWER:

            choices = [0, 1] # 0 is don't try to answer, 1 is try to answer
            
            q_values = [self.q_table[(state, choice)] for choice in choices]

            if(len(q_values) == 0):
                return 0

            max_q = max(q_values)

            return max_q
        
        elif state[0] == DD:

            if self.players[pid].score <= 0:
                return self.q_table[(state,0)]
            
            choices = range(self.players[pid].score)

            q_values = [self.q_table[(state, choice)] for choice in choices]

            if(len(q_values) == 0):
                return 0

            max_q = max(q_values)

            return max_q
              
         
            
    def train_game(self):

        pid = random.randint(0, len(self.players) - 1)

        while self.board.any():
            
            # Choosing
            state = self.get_state(pid)

            choice = self.get_action_train(state, pid)

            potential_points = self.board[choice[0], choice[1]]
            self.board[choice[0], choice[1]] = 0
            

            reward = 0

            new_q_value = (1 - self.alpha) * self.q_table[(state, choice[0])]

            if self.dd == choice:
                # print("Daily Double Found!, Player ", pid, " has ", self.players[pid].score)
                self.dd_found = True

                self.action_type = DD
                next_state = self.get_state(pid)

                new_q_value += self.alpha * (reward + self.gamma * self.get_best_q_value(next_state, pid))

                self.q_table[(state, choice[0])] = new_q_value

                state = next_state

                wager = self.get_action_train(state, pid)
                # print("Wagering ", wager)
                if random.random() > self.players[pid].dd_smartness:
                    reward = -wager
                else:
                    reward = wager

                self.players[pid].change_score(reward)

                new_q_value = (1 - self.alpha) * self.q_table[(state, wager)]
                self.action_type = CHOOSE
                next_state = self.get_state(pid)
                new_q_value += self.alpha * (reward + self.gamma * self.get_best_q_value(next_state, pid))

                self.q_table[(state, wager)] = new_q_value

                state = next_state

                continue

            else:
                self.action_type = ANSWER
                next_state = self.get_state(pid)

                new_q_value += self.alpha * (reward + self.gamma * self.get_best_q_value(next_state, pid))

                self.q_table[(state, choice[0])] = new_q_value

                state = next_state
                
            players_not_guessed = self.players.copy()
            wrong_guesses = 0
            while(len(players_not_guessed) > 0):
                wants_to_guess = []
                not_want_to_guess = []
                for player in players_not_guessed:
                    state = self.get_state(player.get_pid(), wrong_guesses=wrong_guesses)
                    if self.get_action_train(state, player.get_pid()):
                        wants_to_guess.append(player)
                    else:
                        not_want_to_guess.append(player)

                for player in not_want_to_guess:

                        state = self.get_state(player.get_pid(), wrong_guesses=wrong_guesses)
                        new_q_value = (1 - self.alpha) * self.q_table[(state, 0)]

                        self.action_type = CHOOSE

                        next_state = self.get_state(player.get_pid())
                        new_q_value += self.alpha * (reward + self.gamma * self.get_best_q_value(next_state, player.get_pid()))

                        self.q_table[(state, 0)] = new_q_value

                        self.action_type = ANSWER

                if len(wants_to_guess) == 0:
                    self.action_type = CHOOSE
                    state = self.get_state(pid)

                    break

                else:
                    guesser = random.choice(wants_to_guess)
                    if random.random() > guesser.smartness:
                        reward = -potential_points

                        state = self.get_state(guesser.get_pid(), wrong_guesses=wrong_guesses)
                        new_q_value = (1 - self.alpha) * self.q_table[(state, 1)] # 1 is the action of answering

                        wrong_guesses+=1
                        guesser.change_score(reward)
                        players_not_guessed.remove(guesser)

                        if len(players_not_guessed) == 0:
                            self.action_type = CHOOSE

                        next_state = self.get_state(guesser.get_pid(), wrong_guesses=wrong_guesses)
                        new_q_value += self.alpha * (reward + self.gamma * self.get_best_q_value(next_state, guesser.get_pid()))

                        self.q_table[(state, 1)] = new_q_value

                        state = next_state
                        
                    else:
                        reward = potential_points

                        state = self.get_state(guesser.get_pid(), wrong_guesses=wrong_guesses)
                        new_q_value = (1 - self.alpha) * self.q_table[(state, 1)]

                        self.action_type = CHOOSE
                        guesser.change_score(reward)

                        next_state = self.get_state(guesser.get_pid())
                        new_q_value += self.alpha * (reward + self.gamma * self.get_best_q_value(next_state, guesser.get_pid()))

                        self.q_table[(state, 1)] = new_q_value

                        players_not_guessed.remove(guesser)

                        state = next_state

                        break

    def play_game(self):
        self.training = False
        picks_left = self.board_size ** 2

        pid = random.randint(0, len(self.players) - 1)

        while self.board.any():
            
            # Choosing
            state = self.get_state(pid)

            choice = self.get_action_train(state, pid)

            potential_points = self.board[choice[0], choice[1]]
            self.board[choice[0], choice[1]] = 0
            self.picks[choice[0], choice[1]] += picks_left
            picks_left -= 1
            

            reward = 0

            self.action_type = ANSWER

            state = self.get_state(pid)

            if self.dd == choice:
                # print("Daily Double Found!")
                self.dd_found = True

                self.action_type = DD
                next_state = self.get_state(pid)

                state = next_state

                wager = self.get_action_train(state, pid)
                if random.random() > self.players[pid].dd_smartness:
                    reward = -wager
                else:
                    reward = wager

                self.players[pid].change_score(reward)

                self.action_type = CHOOSE
                next_state = self.get_state(pid)


                state = next_state

                continue

            else:
                self.action_type = ANSWER
                next_state = self.get_state(pid)

                state = next_state
                
            players_not_guessed = self.players.copy()
            wrong_guesses = 0
            while(len(players_not_guessed) > 0):
                wants_to_guess = []
                not_want_to_guess = []
                for player in players_not_guessed:
                    state = self.get_state(player.get_pid(), wrong_guesses=wrong_guesses)
                    if self.get_action_train(state, player.get_pid()):
                        wants_to_guess.append(player)
                    else:
                        not_want_to_guess.append(player)

                for player in not_want_to_guess:

                        state = self.get_state(player.get_pid(), wrong_guesses=wrong_guesses)

                        self.action_type = CHOOSE

                        next_state = self.get_state(player.get_pid())

                        self.action_type = ANSWER

                if len(wants_to_guess) == 0:
                    self.action_type = CHOOSE
                    state = self.get_state(pid)

                    break

                else:
                    guesser = random.choice(wants_to_guess)
                    if random.random() > guesser.smartness:
                        reward = -potential_points

                        state = self.get_state(guesser.get_pid(), wrong_guesses=wrong_guesses)

                        wrong_guesses+=1
                        guesser.change_score(reward)
                        players_not_guessed.remove(guesser)

                        if len(players_not_guessed) == 0:
                            self.action_type = CHOOSE

                        next_state = self.get_state(guesser.get_pid(), wrong_guesses=wrong_guesses)


                        state = next_state
                        
                    else:
                        reward = potential_points

                        state = self.get_state(guesser.get_pid(), wrong_guesses=wrong_guesses)

                        self.action_type = CHOOSE
                        guesser.change_score(reward)

                        next_state = self.get_state(guesser.get_pid())

                        players_not_guessed.remove(guesser)

                        state = next_state

                        break
        self.training = True
            
    def reset_game(self):

        self.board_size = self.board_size
        self.board = np.ones((self.board_size, self.board_size))

        for row_num in range(self.board_size):
            self.board[row_num,:] *= (row_num + 1)

        # Set up players
        self.players = []
        for pid in range(self.num_players):
            self.players.append(Player(pid))

        # daily double
        if random.random() > 0.9:
            self.dd = (self.board_size-2, random.randint(0, self.board_size-1))
        else:
            self.dd = (self.board_size-1, random.randint(0, self.board_size-1))

        self.action_type = CHOOSE

        self.dd_found = False

        self.training = True

        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        

    def train(self, num_games=500):
        for game_num in range(num_games):
            print(f"Simulating game {game_num}...")
            print(f"EPSILON: {self.epsilon}")
            self.train_game()
            # print("Results: ")
            # for player_num in range(self.num_players):
                # print(f"Player {player_num}: {self.players[player_num].score}")
            # print(self.q_table.values())
            self.reset_game()


jeopardy = Jeopardy(board_size=5, num_players=3, alpha=0.3, gamma=0.9, epsilon=1.0, decay=(1 - 2e-4))
jeopardy.train(20000)
games = 1000
print(jeopardy.q_table)
for i in range(games):
    jeopardy.play_game()
    jeopardy.reset_game()
print(jeopardy.picks/games)

plt.figure(figsize=(6, 5))
heatmap = plt.imshow((jeopardy.picks/games), cmap='viridis', origin='upper')

# Add color bar legend
cbar = plt.colorbar(heatmap)
cbar.set_label('Average Picks Left')

# Add value labels inside the heatmap
for i in range((jeopardy.picks/games).shape[0]):
    for j in range((jeopardy.picks/games).shape[1]):
        text = f"{(jeopardy.picks/games)[i, j]:.1f}"
        plt.text(j, i, text, ha='center', va='center', color='white' if (jeopardy.picks/games)[i, j] < 13 else 'black')

plt.title(f'Average Picks Left per Tile (after {games} training games)')
plt.xlabel('Column')
plt.ylabel('Row')
plt.xticks(range((jeopardy.picks/games).shape[1]))
plt.yticks(range((jeopardy.picks/games).shape[0]))
plt.tight_layout()
plt.show()

            
                        
                    

    








    


