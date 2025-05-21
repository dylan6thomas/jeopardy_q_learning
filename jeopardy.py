import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

CHOOSE = 0
ANSWER = 1
DD = 2
GAMEOVER = 3

SMARTNESS = [0.9, 0.8, 0.7, 0.6, .43]
DD_SMARTNESS = 0.65 # 0.65
DAILY_DOUBLE_ODDS = [0.02, 0.06, 0.20, 0.32, 0.40] # Must sum to 1

# TODO: Add reward for winning the game


random.seed(42)

class Player:
    def __init__(self, pid, smartness=[0.8, 0.8, 0.8, 0.8, 0.8], dd_smartness = 0.75):
        self.score = 0
        self.pid = pid
        self.smartness = smartness
        self.dd_smartness = dd_smartness
        self.last_state_action = None

    def change_score(self, score_change):
        self.score += int(score_change)
        return self.score

    def get_pid(self):
        return self.pid
    
    def set_last_state_action(self, state_action):
        self.last_state_action = state_action
    
class Jeopardy:
    def __init__(self, board_size, num_players, alpha, gamma, epsilon, decay, smartness, dd_smartness, daily_double_odds=DAILY_DOUBLE_ODDS):
        # Set up board
        self.board_size = board_size
        self.board = np.ones((board_size, board_size + 1))
        # self.board = np.ones((board_size, board_size))

        self.picks = np.zeros_like(self.board)

        self.num_players = num_players

        for row_num in range(board_size):
            self.board[row_num,:] *= (row_num + 1)

        # Set up players
        self.players = []
        for pid in range(num_players):
            self.players.append(Player(pid, smartness=smartness, dd_smartness=dd_smartness))

        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.decay = decay
        self.min_epsilon = 0.05
        self.daily_double_odds = daily_double_odds
        self.smartness = smartness
        self.dd_smartness = dd_smartness
        self.average_scores = []

        self.q_table = defaultdict(lambda: 0)

        # daily double
        row = 0
        cdf = daily_double_odds[row]
        p = random.random()
        while cdf < 1:
            if p < cdf:
                break
            row += 1
            cdf += daily_double_odds[row]

        if random.random() > 0.9:
            self.dd = (row, random.randint(0, board_size))
            # self.dd = (board_size-2, random.randint(0, board_size - 1))
        else:
            self.dd = (row, random.randint(0, board_size))
            # self.dd = (board_size-1, random.randint(0, board_size - 1))

        self.action_type = CHOOSE

        self.dd_found = False

        self.training = True

        self.scores_over_time = [[] for _ in range(num_players)]
        self.epsilon_history = []

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
                
                choices = range(self.players[pid].score + 1)

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
                
                choices = range(self.players[pid].score + 1)

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
            
            choices = range(self.players[pid].score + 1)

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

            self.players[pid].set_last_state_action((state, choice[0]))
            

            reward = 0

            new_q_value = (1 - self.alpha) * self.q_table[(state, choice[0])]

            if self.dd == choice:
                print("Daily Double Found!, Player ", pid, " has ", self.players[pid].score)
                self.dd_found = True

                self.action_type = DD
                next_state = self.get_state(pid)

                new_q_value += self.alpha * (reward + self.gamma * self.get_best_q_value(next_state, pid))

                self.q_table[(state, choice[0])] = new_q_value

                state = next_state

                wager = self.get_action_train(state, pid)
                self.players[pid].set_last_state_action((state, wager))

                print("Wagering ", wager)
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
                        player.set_last_state_action((state, 0))

                        new_q_value = (1 - self.alpha) * self.q_table[(state, 0)]

                        self.action_type = CHOOSE

                        next_state = self.get_state(player.get_pid())
                        new_q_value += self.alpha * (0 + self.gamma * self.get_best_q_value(next_state, player.get_pid()))

                        self.q_table[(state, 0)] = new_q_value

                        self.action_type = ANSWER

                if len(wants_to_guess) == 0:
                    self.action_type = CHOOSE
                    state = self.get_state(pid)

                    break

                else:
                    guesser = random.choice(wants_to_guess)
                    if random.random() > guesser.smartness[choice[0]]:
                        reward = -potential_points

                        state = self.get_state(guesser.get_pid(), wrong_guesses=wrong_guesses)
                        guesser.set_last_state_action((state, 1))

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
                        guesser.set_last_state_action((state, 1))

                        new_q_value = (1 - self.alpha) * self.q_table[(state, 1)]

                        self.action_type = CHOOSE
                        guesser.change_score(reward)

                        next_state = self.get_state(guesser.get_pid())
                        new_q_value += self.alpha * (reward + self.gamma * self.get_best_q_value(next_state, guesser.get_pid()))

                        self.q_table[(state, 1)] = new_q_value

                        players_not_guessed.remove(guesser)

                        state = next_state

                        break
            # if not self.board.any(): # board is empty, game over

        scores = [p.score for p in self.players]
        max_score = max(scores)
        winners = [p for p in self.players if p.score == max_score]

        WIN_REWARD = 100
        LOSS_REWARD = -10

        # Give final rewards and update Q-table
        for p in self.players:
            reward = WIN_REWARD if p in winners else LOSS_REWARD

            state_action = p.last_state_action
            
            if state_action is not None:
                self.q_table[state_action] += self.alpha * (reward)



    def play_game(self, track=False):
        self.training = False
        picks_left = self.board_size * (self.board_size + 1)
        # picks_left = self.board_size ** 2

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
                    if random.random() > guesser.smartness[choice[0]]:
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

        if track:
            for i, p in enumerate(self.players):
                self.scores_over_time[i].append(p.score)
            
    def reset_game(self):

        # Set up board
        self.board_size = self.board_size
        self.board = np.ones((self.board_size, self.board_size + 1))
        # self.board = np.ones((self.board_size, self.board_size))

        self.num_players = self.num_players

        for row_num in range(self.board_size):
            self.board[row_num,:] *= (row_num + 1)

        # Set up players
        self.players = []
        for pid in range(self.num_players):
            self.players.append(Player(pid, smartness=self.smartness, dd_smartness=self.dd_smartness))

        # daily double
        row = 0
        cdf = self.daily_double_odds[row]
        p = random.random()
        while cdf < 1:
            if p < cdf:
                break
            row += 1
            cdf += self.daily_double_odds[row]

        if random.random() > 0.9:
            self.dd = (row, random.randint(0, self.board_size))
            # self.dd = (self.board_size-2, random.randint(0, self.board_size - 1))
        else:
            self.dd = (row, random.randint(0, self.board_size))
            # self.dd = (self.board_size-1, random.randint(0, self.board_size - 1))

        self.action_type = CHOOSE

        self.dd_found = False

        self.training = True

        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        self.epsilon_history.append(self.epsilon)
        

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


jeopardy = Jeopardy(board_size=5, num_players=3, alpha=0.3, gamma=0.9, epsilon=1.0, decay=(1 - 3e-4), smartness=SMARTNESS, dd_smartness=DD_SMARTNESS)
jeopardy.train(10000)
games = 1000
for i in range(games):
    if i % 100 == 0:
        jeopardy.play_game(track=True)
    else:
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


plt.figure()
plt.plot(jeopardy.epsilon_history)
plt.title("Epsilon Decay Over Training")
plt.xlabel("Training Game #")
plt.ylabel("Epsilon")
plt.grid(True)
plt.tight_layout()
plt.show()

for i, scores in enumerate(jeopardy.scores_over_time):
    plt.plot(scores, label=f"Player {i}")
plt.xlabel("Game #")
plt.ylabel("Final Score")
plt.title("Player Scores Over Time")
plt.legend()
plt.show()       
                        
                    

    








    


