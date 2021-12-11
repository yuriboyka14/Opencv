# Python3 program to find the next optimal move for a player
player, opponent = 'o', 'x'


# This function returns true if there are moves
# remaining on the board. It returns false if
# there are no moves left to play.
def isMovesLeft(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == '':
                return True
    return False


# This is the evaluation function as discussed
# in the previous article ( http://goo.gl/sJgv68 )
def evaluate(b):
    # Checking for Rows for X or O victory.
    for row in range(3):
        if b[row][0] == b[row][1] == b[row][2] == player:
            # print(f"Player wins")
            return 10
        elif b[row][0] == b[row][1] == b[row][2] == opponent:
            # print(f"Opponent wins")
            return -10

    # Checking for Columns for X or O victory.
    for col in range(3):
        if b[0][col] == b[1][col] == b[2][col] == player:
            # print(f"Player wins")
            return 10
        elif b[0][col] == b[1][col] == b[2][col] == opponent:
            # print(f"Opponent wins")
            return -10

    # Checking for Diagonals for X or O victory.
    if b[0][0] == b[1][1] == b[2][2] == player:
        # print(f"Player wins")
        return 10
    elif b[0][0] == b[1][1] == b[2][2] == opponent:
        # print(f"Opponent wins")
        return -10

    if b[0][2] == b[1][1] == b[2][0] == player:
        # print(f"Player wins")
        return 10
    elif b[0][2] == b[1][1] == b[2][0] == opponent:
        # print(f"Opponent wins")
        return -10

    # Else if none of them have won then return 0
    return 0


# This is the minimax function. It considers all
# the possible ways the game can go and returns
# the value of the board
def minimax(board, depth, isMax):
    score = evaluate(board)

    # If Maximizer has won the game return his/her
    # evaluated score
    if score == 10:
        return score

    # If Minimizer has won the game return his/her
    # evaluated score
    if score == -10:
        return score

    # If there are no more moves and no winner then
    # it is a tie
    if not isMovesLeft(board):
        return 0

    # If this maximizer's move
    if isMax:
        best = -1000

        # Traverse all cells
        for i in range(3):
            for j in range(3):

                # Check if cell is empty
                if board[i][j] == '':
                    # Make the move
                    board[i][j] = player

                    # Call minimax recursively and choose
                    # the maximum value
                    best = max(best, minimax(board,
                                             depth + 1,
                                             not isMax))

                    # Undo the move
                    board[i][j] = ''
        return best

    # If this minimizer's move
    else:
        best = 1000

        # Traverse all cells
        for i in range(3):
            for j in range(3):

                # Check if cell is empty
                if board[i][j] == '':
                    # Make the move
                    board[i][j] = opponent

                    # Call minimax recursively and choose
                    # the minimum value
                    best = min(best, minimax(board, depth + 1, not isMax))

                    # Undo the move
                    board[i][j] = ''
        return best


# This will return the best possible move for the player
def findBestMove(board):
    bestVal = -1000
    bestMove = (-1, -1)

    # Traverse all cells, evaluate minimax function for
    # all empty cells. And return the cell with optimal
    # value.
    for i in range(3):
        for j in range(3):

            # Check if cell is empty
            if board[i][j] == '':

                # Make the move
                board[i][j] = player

                # compute evaluation function for this
                # move.
                moveVal = minimax(board, 0, False)

                # Undo the move
                board[i][j] = ''

                # If the value of the current move is
                # more than the best value, then update
                # best/
                if moveVal > bestVal:
                    bestMove = (i, j)
                    bestVal = moveVal

    # print("The value of the best Move is :", bestVal)
    # print(f"Position for the best move - Row: {bestMove[0]}, Column: {bestMove[1]}")
    return bestMove


def Game(grid):                         # probably redundant function

    best_move = findBestMove(grid)
    # print(f"best move: {best_move}")

    # print(f"Position for the best move - Row: {best_move[0]}, Column: {best_move[1]}")

    grid[best_move[0]][best_move[1]] = 'x'

    return grid