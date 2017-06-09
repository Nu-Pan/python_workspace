""" Q学習お勉強用兼Pythonお勉強用プログラム """

import copy
import random

# 学習系定数
LEARN_RATE = 0.1
DISCOUNT_FACTOR = 0.7

# 迷路設定
MAZE_TABLE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]

def is_maze_wall(maze_table, state):
    "指定ステートが壁の中か？"
    position_u, position_v = state
    return maze_table[position_v][position_u] == 1

# 行動定数
ACTION_LEFT = (-1, 0)
ACTION_RIGHT = (+1, 0)
ACTION_UP = (0, -1)
ACTION_DOWN = (0, +1)
ACTION_MOVE_AMOUNTS = [
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    ACTION_DOWN
]

def transit_state(table, state, action):
    "state で action を取った時の次の state を得る"
    u, v = state
    du, dv = table[action]
    return u + du, v + dv

# スタート/ゴール定数
STATE_START = (1, 1)
STATE_GOAL = (6, 6)

# 報酬テーブル初期化
# ゴールイン : 1.0
# 壁に突入 : -1.0
# 通路 : 0.0
REWORD_DICT = {}
for v in range(1, len(MAZE_TABLE)-1):
    for u in range(1, len(MAZE_TABLE[v])-1):
        current_state = (u, v)
        REWORD_DICT[current_state] = {}
        for action in range(0, len(ACTION_MOVE_AMOUNTS)):
            next_state = transit_state(ACTION_MOVE_AMOUNTS, current_state, action)
            isInnerWallNext = is_maze_wall(MAZE_TABLE, next_state)
            isInnerWallCurrent = is_maze_wall(MAZE_TABLE, current_state)
            if (next_state == STATE_GOAL) and (not isInnerWallNext) and (not isInnerWallCurrent):
                REWORD_DICT[current_state][action] = +10.0
            elif not isInnerWallNext:
                REWORD_DICT[current_state][action] = -1.0

# Qテーブル初期化
Q_TABLE = {}
for key in REWORD_DICT:
    Q_TABLE[key] = copy.deepcopy(REWORD_DICT[key])

# とりあえず 1000 回くらいゴールに着くまで学習を繰り返す
for episode_index in range(0, 1000):
    # ゴールに着くまで学習をすすめる
    current_state = STATE_START
    while True:
        if current_state == STATE_GOAL:
            break
        current_action = int(random.choice(list(Q_TABLE[current_state].keys())))
        next_state = transit_state(ACTION_MOVE_AMOUNTS, current_state, current_action)
        # 本当は(0,0)みたいな外壁に移動しようとして Q_TABLE が存在しないことをチェックすべき
        # （外壁から更に外側に行こうとしたら out of index）
        # だが、データの作り方的に壁に突入はしないはずなのでスルー
        next_action = int(random.choice(list(Q_TABLE[next_state].keys())))
        Q_TABLE[current_state][current_action] = \
            (1 - DISCOUNT_FACTOR) * Q_TABLE[current_state][current_action] \
            + LEARN_RATE \
            * ( \
                REWORD_DICT[current_state][current_action] \
                + DISCOUNT_FACTOR * Q_TABLE[next_state][next_action] \
            )
        current_state = next_state
        #print(current_state)
    # Qテーブルの行動をリダクション（最大値を採用）
    print("setp =" + str(episode_index))
    for v in range(1, len(MAZE_TABLE)-1):
        temp_list = []
        for u in range(1, len(MAZE_TABLE[v])-1):
            if Q_TABLE[u, v] is None:
                temp_list.append(0.0)
            else:
                temp_value = max(Q_TABLE[u, v].values())
                temp_list.append('%6f' % temp_value)
        print(temp_list)
