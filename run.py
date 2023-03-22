import random
from typing import List

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


# Algorithm

def generate_figures(fig_count: int) -> List[List[int]]:
    positions = np.zeros(shape=(fig_count, 2))
    cur_index = 0
    while cur_index != fig_count:
        new_pos = [random.randint(0, st.session_state['fig_count'] - 1), random.randint(0, st.session_state['fig_count'] - 1)]
        if (positions == new_pos).all(axis=1).any():
            continue
        positions[cur_index] = new_pos
        cur_index += 1

    return list(positions.astype(np.uint8))


def generate_population(num_population: int, fig_count: int) -> List[List[List[int]]]:
    population = list()
    for i in range(num_population):
        population.append(generate_figures(fig_count))
    return population


def fitness_func(individual: List[List[int]]) -> float:
    instance, mask = get_weighted_table(individual)
    result = np.sum(instance[mask])
    return result


def get_weighted_table(individual: List[List[int]]):
    instance = np.zeros(shape=(st.session_state['fig_count'], st.session_state['fig_count']))
    mask = np.zeros(shape=(st.session_state['fig_count'], st.session_state['fig_count']), dtype=bool)
    for pos in individual:
        instance[pos[0]] += 1
        instance[:, pos[1]] += 1
        instance += np.eye(N=st.session_state['fig_count'], k=int(pos[1]) - int(pos[0]))
        instance += np.fliplr(np.eye(N=st.session_state['fig_count'], k=st.session_state['fig_count'] - 1 - pos[1] - pos[0]))
        instance[pos[0], pos[1]] -= 4
        mask[pos[0], pos[1]] = True

    return instance, mask


def selection(population: List[List[List[int]]], top_count: int) -> List[List[List[int]]]:
    scores = np.ones(shape=(len(population, ))) * np.inf
    for i, individual in enumerate(population):
        score = fitness_func(individual)
        scores[i] = score

    top_n_scores_indexes = np.argsort(scores)
    return list(np.array(population)[top_n_scores_indexes][:top_count])


def budding(individual: List[List[int]], n=5):
    for _ in range(n):
        instance, mask = get_weighted_table(individual)
        inst_copy = instance.copy()
        inst_copy[~mask] = np.inf
        fig_rows, fig_cols = np.where(inst_copy == np.max(inst_copy[mask]))
        rand_fig = random.randint(0, len(fig_rows) - 1)

        nofig_rows, nofig_cols = np.where(inst_copy == np.min(inst_copy[~mask]))
        norand_fig = random.randint(0, len(nofig_rows) - 1)

        individual = np.array(individual, dtype=int)
        fig_mask = np.where((individual == (fig_rows[rand_fig], fig_cols[rand_fig])).all(axis=1))[0]
        individual[fig_mask] = np.array([nofig_rows[norand_fig], nofig_cols[norand_fig]])

    return list(individual)


def mutate(individual: List[List[int]]):
    instance, mask = get_weighted_table(individual)
    fig_rows, fig_cols = np.where(mask)
    rand_fig = random.randint(0, len(fig_rows) - 1)
    nofig_rows, nofig_cols = np.where(~mask)
    norand_fig = random.randint(0, len(nofig_rows) - 1)
    individual = np.array(individual, dtype=int)
    fig_mask = np.where((individual == (fig_rows[rand_fig], fig_cols[rand_fig])).all(axis=1))
    individual[fig_mask] = np.array([nofig_rows[norand_fig], nofig_cols[norand_fig]])
    return list(individual)


def step_evolve(population: List[List[List[int]]]):
    best_individuals = selection(population, st.session_state['top_individuals'])
    new_individuals = list()
    budding_count = max(round(st.session_state['bud_decay'] * st.session_state['bud_count']), 1)
    st.session_state['bud_decay'] *= st.session_state['bud_decay']
    for individual in best_individuals:
        for _ in range(st.session_state['num_population'] // st.session_state['top_individuals'] - 1):
            new_individ = budding(individual.copy(), n=budding_count)
            if random.random() > st.session_state['mut_prob']:
                new_individ = mutate(new_individ)
            new_individuals.append(new_individ)

    best_individuals.extend(new_individuals)

    return best_individuals


# GUI

def draw_board(width: int = 20) -> np.ndarray:
    nrows, ncols = st.session_state['fig_count'], st.session_state['fig_count']
    image = np.ones(nrows * ncols * 3 * width * width) * 255

    # Reshape things into a 9x9 grid.
    image = image.reshape((nrows * width, ncols * width, 3))
    image[::width, :] = 0
    image[:, ::width] = 0
    return image.astype(int)


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
    color_map = np.repeat(color_map, repeats=20, axis=1).astype(int)
    color_map = COLORS[color_map.reshape(-1)].reshape(board.shape)
    board[(board != 0)] = color_map[(board != 0)]
    board = draw_figures(board, fig_positions, width)
    board = write_weights(board, weighted.astype(int), width)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    col_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    row_labels = range(len(col_labels))
    ax.matshow(board, cmap='brg')
    ax.set_xticks(range(width // 2, len(col_labels) * width, width), col_labels)
    ax.set_yticks(range(width // 2, len(col_labels) * width, width), row_labels)
    return fig


def loop_table(col_right):
    last_fig = None
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


if 'num_iters' not in st.session_state.keys():
    st.session_state['num_iters'] = 100
    st.session_state['num_population'] = 300
    st.session_state['fig_count'] = 5
    st.session_state['top_individuals'] = 30
    st.session_state['mut_prob'] = .1
    st.session_state['bud_count'] = 1
    st.session_state['bud_decay'] = .99

status = {
    0: 'run',
    1: 'stop'
}

st.title('Chess optimization')

col_left, col_right = st.columns(2)

if 'status' not in st.session_state.keys():
    st.session_state['status'] = 0

with col_left:
    num_iters = st.number_input('Количество итераций', min_value=1, step=1, value=st.session_state['num_iters'])
    num_population = st.number_input('Размер популяции', min_value=1, step=1, value=st.session_state['num_population'])
    top_individuals = st.number_input('Количество особей для отбора', min_value=1, max_value=num_population, step=1,
                                      value=st.session_state['top_individuals'])
    fig_count = st.slider('Количество фигур', min_value=4, max_value=200, step=1, value=st.session_state['fig_count'])
    mut_prob = st.slider('Вероятность мутации', min_value=.0, max_value=1., value=.1, step=.001)
    bud_count = st.slider('Количество перестановок', min_value=1, max_value=st.session_state['fig_count'], value=st.session_state['bud_count'], step=1)

    container = st.empty()
    is_clicked = container.button(status[st.session_state['status']])
    if is_clicked:
        container.empty()
        state(st.session_state['status'])
        container.button(status[st.session_state['status']])

st.session_state['num_iters'] = num_iters
st.session_state['num_population'] = num_population
st.session_state['fig_count'] = fig_count
st.session_state['top_individuals'] = top_individuals
st.session_state['mut_prob'] = mut_prob

if st.session_state['status'] != 0:
    loop_table(col_right)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
