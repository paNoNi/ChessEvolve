from typing import List, Dict

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt, colors

from algorithm import get_weighted_table, generate_population, step_evolve, selection, fitness_func


def draw_board(width: int = 20) -> np.ndarray:
    nrows, ncols = 9, 9
    image = np.ones(nrows * ncols * 3 * width * width) * 255

    # Reshape things into a 9x9 grid.
    image = image.reshape((nrows * width, ncols * width, 3))
    image[::width, :] = 0
    image[:, ::width] = 0
    return image.astype(np.int)


def draw_figures(board: np.ndarray, fig_positions: List[List[int]], width: int = 20) -> np.ndarray:
    board = Image.fromarray(board)
    draw = ImageDraw.Draw(board)
    for fig_pos in fig_positions:
        fig_pos_left_up_point = fig_pos[1] * width
        fig_pos_right_down_point = fig_pos[0] * width
        left_up_point = (fig_pos_left_up_point, fig_pos_right_down_point)
        right_down_point = (fig_pos_left_up_point + width, fig_pos_right_down_point + width)
        two_point_list = [left_up_point, right_down_point]
        draw.ellipse(two_point_list, fill=(0, 200, 112, 255))

    return np.array(board)


def write_weights(board: np.ndarray, weights: np.ndarray, width: int = 20):
    board = Image.fromarray(board.astype(np.uint8))
    draw = ImageDraw.Draw(board)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            fig_pos_left_point = i * width + width // 3
            fig_pos_up_point = j * width + width // 3
            draw.text((fig_pos_left_point, fig_pos_up_point), text=str(weights[j, i]))
    return np.array(board)


COLORS = np.array([[c * 25, c, 255 - c * 25] for c in range(10)])
COLORS = np.concatenate([COLORS, np.repeat([[255, 0, 0]], axis=0, repeats=15)])


def draw_chest(fig_positions: List[List[int]]):
    width = 20
    board = draw_board(width)
    weighted, _ = get_weighted_table(fig_positions)
    board = write_weights(board, weighted, width)
    color_map = np.repeat(weighted, repeats=20, axis=0)
    color_map = np.repeat(color_map, repeats=20, axis=1).astype(np.int)
    color_map = COLORS[color_map.reshape(-1)].reshape(board.shape)
    board[(board != 0)] = color_map[(board != 0)]
    board = draw_figures(board, fig_positions, width)
    board = write_weights(board, weighted.astype(np.int), width)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    col_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    row_labels = range(len(col_labels))
    ax.matshow(board, cmap='brg')
    ax.set_xticks(range(width // 2, len(col_labels) * width, width), col_labels)
    ax.set_yticks(range(width // 2, len(col_labels) * width, width), row_labels)
    return fig


def loop_table(col_right):
    last_fig = None

    if st.session_state['status'] == 0:
        return
    population = generate_population(st.session_state['num_population'], st.session_state['fig_count'])
    with col_right:
        st_plt = None
        text = st.text(f'Gen: 0, Fitness: Inf')

        for it in range(st.session_state['num_iters']):
            population = step_evolve(population)
            best_individ = selection(population, top_count=1)[0]
            fig = draw_chest(best_individ)
            score = fitness_func(best_individ)
            text.text(f'Gen: {it + 1}, Fitness: {score}')
            if st_plt is None:
                st_plt = st.pyplot(fig)

            else:
                st_plt.pyplot(fig, clear_figure=True)

            if last_fig is not None:
                plt.close(last_fig)

            if score == 0 or st.session_state['status'] == 0:
                return

            last_fig = fig


def state(new_status: int):
    st.session_state['status'] = abs(1 - new_status)
