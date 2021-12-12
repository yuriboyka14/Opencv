ai, human = 'x', 'o'


def moves_left(grid):
    for i in range(3):
        for j in range(3):
            if grid[i][j] == '':
                return True
    return False


def evaluate(grid):
    for row in range(3):                                                  # checking rows for win
        if grid[row][0] == grid[row][1] == grid[row][2]:
            if grid[row][0] == ai:
                return 1
        if grid[row][0] == grid[row][1] == grid[row][2]:
            if grid[row][0] == human:
                return -1

    for col in range(3):                                                  # checking colums for win
        if grid[0][col] == grid[1][col] == grid[2][col]:
            if grid[0][col] == ai:
                return 1
        if grid[0][col] == grid[1][col] == grid[2][col]:
            if grid[0][col] == human:
                return -1

    if grid[0][0] == grid[1][1] == grid[2][2]:                            # checking diagonals for win
        if grid[0][0] == ai:
            return 1
    if grid[0][0] == grid[1][1] == grid[2][2]:
        if grid[0][0] == human:
            return -1

    if grid[0][2] == grid[1][1] == grid[2][0]:
        if grid[0][2] == ai:
            return 1
    if grid[0][2] == grid[1][1] == grid[2][0]:
        if grid[0][2] == human:
            return -1

    return 0                                                               # not finished or tie


def minimax(grid, depth, isMax):
    score = evaluate(grid)

    if score == 0:
        if not moves_left(grid):
            return score                                        # tie
    elif score == 1 or score == -1:
        return score                                            # either 1 (ai win) or -1 (human win)

    if isMax:
        best_score = -1000000
        for row in range(3):
            for col in range(3):
                if grid[row][col] == '':
                    grid[row][col] = ai
                    score = minimax(grid, depth + 1, False)    # false is here since we check the possibilities of
                    grid[row][col] = ''                        # the next move (human move). We are on another depth
                    best_score = max(score, best_score)
        return best_score

    else:
        best_score = 1000000
        for row in range(3):
            for col in range(3):
                if grid[row][col] == '':
                    grid[row][col] = human
                    score = minimax(grid, depth + 1, True)     # analogically here with true - now is minimizing move
                    grid[row][col] = ''                        # and we want to check one after it
                    best_score = min(score, best_score)
        return best_score


def find_best_move(grid):
    best_move = None
    best_score = -1000000
    for row in range(3):
        for col in range(3):
            if grid[row][col] == '':
                grid[row][col] = ai
                score = minimax(grid, 0, False)
                grid[row][col] = ''                             # undoing the move
                if score > best_score:
                    best_score = score
                    best_move = [row, col]

    grid[best_move[0]][best_move[1]] = ai

    return best_move


def Game(grid):                      
    try:
        best_move = find_best_move(grid)              # Exception handler for situation in which we have the last move
        grid[best_move[0]][best_move[1]] = 'x'      # (does not know what to do because best_move is NONE)
    except Exception:                               # It is a tie automatically since we cannot win with the last move
        return grid, 0, True, None                  # (we cannot win at all in general...)

    score = evaluate(grid)
    isFinished = False
    winner = None

    if score == 1:
        winner = "AI"
        isFinished = True
    elif score == -1:
        winner = "Human"
        isFinished = True

    return grid, score, isFinished, winner



