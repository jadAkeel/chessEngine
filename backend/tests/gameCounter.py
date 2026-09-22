import chess.pgn

def count_games(pgn_path):
    count = 0
    with open(pgn_path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            count += 1
    return count

path = r"C:\Users\10User\Desktop\ai\chessEngine - Copy\backend\data\lichess_db_standard_rated_2013-01.pgn"
print("Games:", count_games(path))