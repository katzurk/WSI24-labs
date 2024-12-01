from game import *
from player import *

class GameSimulation:
    def __init__(self, game, player_x, player_o):
        self.player_x = player_x
        self.player_o = player_o
        self.game = game
        self.results = {"x": 0, "o": 0, "t": 0}

    def start_game(self, iterations=1, display=False, change_started=True):
        for i in range(iterations):
            if display: self.display_board()
            while self.game.get_winner() == "":
                move = (
                self.player_x.get_move(None)
                if self.game.player_x_turn
                else self.player_o.get_move(None) )
                self.game.move(move)
                if display: self.display_board()
            self.results[self.game.get_winner()] += 1
            if change_started: self.game.play_again()
            else: self.game.play_again_same_started()
        self.display_result()

    def display_board(self):
        turn = "x" if self.game.player_x_turn else "o"
        winner = self.game.get_winner()
        if winner != "":
            if winner == "t":
                print("It's a tie")
            else:
                print(f"Player {self.game.get_winner()} won")
        else:
            print(f"Player {turn} turn")
        print(self.game.board)
        print()

    def display_result(self):
        print("Results:")
        x = self.results["x"]
        o = self.results["o"]
        t = self.results["t"]
        print(f"x - {x}")
        print(f"o - {o}")
        print(f"t - {t}")