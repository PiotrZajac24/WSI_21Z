import argparse
import pygame
import sys
import numpy as np
import time
from .q_learning import Board, QLearning

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (216,44,0)
YELLOW = (239,236,0)
GRAY = (179,176,167)
GREEN = (0,253,186)



def main():
    n = 20
    obstacle_ratio = 0.3
    m_e = 0.1
    mx_e = 1
    start = (0, 0)
    end = (n-1, n-1)
    HEIGHT = 800
    WIDTH = 800
    SIDE = min(HEIGHT//n, WIDTH//n)
    SIDE = SIDE + SIDE % 2

    HEIGHT = WIDTH = SIDE*n

    pygame.init()
    pygame.display.set_caption("Elon Piżmo's training ground")
    CLOCK = pygame.time.Clock()
    CLOCK.tick(60)
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    SCREEN.fill(GRAY)

    QAgent = QLearning(n, start, end, obstacle_ratio)
    board = QAgent.board
    
    for x in range(0, WIDTH, SIDE):
        for y in range(0, HEIGHT, SIDE):
            color = RED if board.board[x//SIDE, y//SIDE] < -1 else WHITE
            pygame.draw.rect(SCREEN, color, (x, y, SIDE, SIDE), 0)
            pygame.draw.rect(SCREEN, BLACK, (x, y, SIDE, SIDE), 2)
    for cord in [board.start, board.end]:
        pygame.draw.rect(SCREEN, GREEN, (cord[0]*SIDE, cord[1]*SIDE, SIDE, SIDE), 0)
        pygame.draw.rect(SCREEN, BLACK, (cord[0]*SIDE, cord[1]*SIDE, SIDE, SIDE), 2)

    pygame.display.update()
    print(QAgent.board)
    QAgent.learn(0.7, 0.5, m_e, mx_e, 4000, n**2)
    print("DONE")
    moves = QAgent.move()

    state = np.array(start)
    for move in moves[:-1]:
        state = state + move
        pygame.draw.rect(SCREEN, YELLOW, (state[0]*SIDE, state[1]*SIDE, SIDE, SIDE), 0)
        pygame.draw.rect(SCREEN, BLACK, (state[0]*SIDE, state[1]*SIDE, SIDE, SIDE), 2)
        pygame.display.update()
        time.sleep(0.05)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
if __name__ == "__main__":
    main()