import random
from typing import List, Tuple, Dict

import numpy as np
import streamlit as st

def generate_figures(fig_count: int) -> List[List[int]]:
    positions = np.zeros(shape=(fig_count, 2))
    cur_index = 0
    while cur_index != fig_count:
        new_pos = [random.randint(0, 8), random.randint(0, 8)]
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
    instance = np.zeros(shape=(9, 9))
    mask = np.zeros(shape=(9, 9), dtype=bool)
    for pos in individual:
        instance[pos[0]] += 1
        instance[:, pos[1]] += 1
        instance += np.eye(N=9, k=int(pos[1]) - int(pos[0]))
        instance += np.fliplr(np.eye(N=9, k=8 - pos[1] - pos[0]))
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


def budding(individual: List[List[int]]):
    instance, mask = get_weighted_table(individual)
    inst_copy = instance.copy()
    inst_copy[~mask] = 100
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
    for individual in best_individuals:
        for _ in range(st.session_state['num_population'] // st.session_state['top_individuals'] - 1):
            new_individ = budding(individual.copy())
            if random.random() > st.session_state['mut_prob']:
                new_individ = mutate(new_individ)
            new_individuals.append(new_individ)

    best_individuals.extend(new_individuals)

    return best_individuals
