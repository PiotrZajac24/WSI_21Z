import pygame
import sys
import numpy as np

import time
from source_files.connect_four import Connect4, parse_arguments

BLACK = (0,0,0)
RED = (216,44,0)
YELLOW = (239,236,0)
GRAY = (179,176,167)
GREEN = (0,253,186)


def swap_turn(turn):
    return "R" if turn == "Y" else "Y"


def main():
    args = parse_arguments()

    # prepare screen
    HEIGHT = 800
    WIDTH = 800
    SIDE = min(HEIGHT//args.height, WIDTH//args.width)
    SIDE = SIDE + SIDE % 2
    HEIGHT = SIDE*args.height
    WIDTH = SIDE*args.width

    pygame.init()
    pygame.display.set_caption("Connect four")
    CLOCK = pygame.time.Clock()
    CLOCK.tick(60)
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    SCREEN.fill(GRAY)

    for x in range(0, WIDTH, SIDE):
        for y in range(0, HEIGHT, SIDE):
            pygame.draw.circle(SCREEN, BLACK, (x+SIDE//2, y+SIDE//2), SIDE//2 - 2)

    board = np.full((args.width, args.height), " ")
    turn = "Y"

    game = Connect4(args.height, args.width)

    ai_turn = True
    max_move = True
    last_move = -1
    moves = []
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ai_turn:
                state = ''.join(''.join(list(row)) for row in board)
                
                _, state, last_move = game.move(state, args.depth, max_move, last_move)
                
                x = last_move//args.height
                y = last_move % args.height
                board[x, y] = turn
                
                pygame.draw.circle(SCREEN, YELLOW if turn == "Y" else
                    RED, (x*SIDE+SIDE//2, (args.height-1-y)*SIDE+SIDE//2), SIDE//2 - 2)
                game.print_board(state)
                pygame.display.update()
                result = game.check_win(state, last_move)
                if result:
                    moves.append(last_move)
                    pygame.draw.circle(SCREEN, YELLOW if turn == "Y" else
                        RED, (x*SIDE+SIDE//2, (args.height-1-y)*SIDE+SIDE//2), SIDE//2 - 2)
                    pygame.display.update()
                    if result != "D":
                            print("GAME OVER", result, "wins")
                    else:
                        print("DRAW")
                    pygame.display.update()
                    time.sleep(5)
                    pygame.quit()
                    sys.exit()
                max_move = not max_move
                turn = swap_turn(turn)
                ai_turn = not ai_turn
            elif not ai_turn and event.type == pygame.MOUSEBUTTONUP:
                state = ''.join(''.join(list(row)) for row in board)
                x, y = pygame.mouse.get_pos()
                x, y = x//SIDE, y//SIDE
                empty = np.argwhere(board[x] == ' ')
                if len(empty) > 0:
                    y = empty[0][0]
                    board[x, y] = turn
                    state = ''.join(''.join(list(row)) for row in board)
                    pygame.draw.circle(SCREEN, YELLOW if turn == "Y" else
                        RED, (x*SIDE+SIDE//2, (args.height-1-y)*SIDE+SIDE//2), SIDE//2 - 2)
                    game.print_board(state)
                    pygame.display.update()
                    
                    last_move = x*args.height + y
                    result = game.check_win(state, last_move)
                    if result:
                        moves.append(last_move)    
                        pygame.draw.circle(SCREEN, YELLOW if turn == "Y" else
                            RED, (x*SIDE+SIDE//2, (args.height-1-y)*SIDE+SIDE//2), SIDE//2 - 2)
                        pygame.display.update()
                        if result != "D":
                            print("GAME OVER", result, "wins")
                        else:
                            print("DRAW")
                        pygame.display.update()
                        time.sleep(5)
                        pygame.quit()
                        sys.exit()
                        return
                    turn = swap_turn(turn)
                    max_move = not max_move
                    last_move = args.height*x + y
                    ai_turn = not ai_turn
        moves.append(last_move)       
        pygame.display.update()

if __name__ == "__main__":
    main()
