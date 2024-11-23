from abc import ABC, abstractmethod
import copy
import numpy as np


def build_player(player_config, game):
    assert player_config["type"] in ["human", "random", "minimax"]

    if player_config["type"] == "human":
        return HumanPlayer(game)

    if player_config["type"] == "random":
        return RandomComputerPlayer(game)

    if player_config["type"] == "minimax":
        return MinimaxComputerPlayer(game, player_config)


class Player(ABC):
    def __init__(self, game):
        self.game = game
        self.score = 0

    @abstractmethod
    def get_move(self, event_position):
        pass


class HumanPlayer(Player):
    def get_move(self, event_position):
        return event_position


class RandomComputerPlayer(Player):
    def get_move(self, event_position):
        available_moves = self.game.available_moves()
        move_id = np.random.choice(len(available_moves))
        return available_moves[move_id]


class MinimaxComputerPlayer(Player):
    def __init__(self, game, config):
        super().__init__(game)
        self.player_x = None
        # TODO: lab3 - load pruning depth from config

    def get_move(self, event_position):
        # TODO: lab3 - implement algorithm
        self.player_x = self.game.player_x_turn

        score = -float('inf')
        available_moves = self.game.available_moves()
        sorted_moves = self.sort_by_heuristic(available_moves)
        for i in range(len(sorted_moves)):
            move = sorted_moves[i]
            self.game.move(move)
            new_score = self.MiniMax(copy.deepcopy(self.game), 1, False, -float('inf'), float('inf'))
            self.game.undo_move(move)
            if new_score > score:
                score = new_score
                move_id = i
        return sorted_moves[move_id]

    def MiniMax(self, game, depth, is_max, alpha, beta):
        winner = game.get_winner()
        if winner != "" or depth == 0:
            return self.evaluate_state(game)

        if is_max:
            score = -float('inf')
            available_moves = game.available_moves()
            sorted_moves = self.sort_by_heuristic(available_moves)
            for i in range(len(sorted_moves)):
                move = sorted_moves[i]
                game.move(move)
                new_score = self.MiniMax(copy.deepcopy(game), depth - 1, False, alpha, beta)
                game.undo_move(move)
                score = max(score, new_score)
                alpha = max(alpha, new_score)
                if beta <= alpha:
                    break
        else:
            score = float('inf')
            available_moves = game.available_moves()
            sorted_moves = self.sort_by_heuristic(available_moves)
            for i in range(len(sorted_moves)):
                move = sorted_moves[i]
                game.move(move)
                new_score = self.MiniMax(copy.deepcopy(game), depth - 1, True, alpha, beta)
                game.undo_move(move)
                score = min(score, new_score)
                beta = min(beta, new_score)
                if beta <= alpha:
                    break

        return score

    def evaluate_state(self, game):
        score_dict = {
            "x": 10 if self.player_x else -10,
            "o": 10 if not self.player_x else -10,
            "t": 0
        }

        winner = game.get_winner()
        if winner != "":
            return score_dict.get(winner, 0)

        state_eval = 0
        x = np.argwhere((game.board == "x"))
        o = np.argwhere((game.board == "o"))

        for move in x:
            if score_dict["x"] == 10:
                state_eval += self.get_heuristic(move)
            else:
                state_eval -= self.get_heuristic(move)
        for move in o:
            if score_dict["o"] == 10:
                state_eval += self.get_heuristic(move)
            else:
                state_eval -= self.get_heuristic(move)
        return state_eval


    def get_heuristic(self, move):
        grades = {
            4: [[1, 1]],
            3: [[0, 0], [0, 2], [2, 0], [2, 2]],
            2: [[0, 1], [1, 0], [1, 2], [2, 1]]
        }
        for grade, vectors in grades.items():
            vectors = np.array(vectors)
            if any(np.array_equal(move, vector) for vector in vectors):
                return grade

    def sort_by_heuristic(self, moves):
        sorted_moves_id = np.argsort([self.get_heuristic(x) for x in moves])
        sorted_moves = moves[sorted_moves_id]
        return sorted_moves[::-1]

