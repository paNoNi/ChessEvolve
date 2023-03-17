from gui import loop_table, state

import streamlit as st

if 'num_iters' not in st.session_state.keys():
    st.session_state['num_iters'] = 100
    st.session_state['num_population'] = 300
    st.session_state['fig_count'] = 5
    st.session_state['top_individuals'] = 30
    st.session_state['mut_prob'] = .1


status = {
    0: 'run',
    1: 'stop'
}

st.title('Chest optimization')

col_left, col_right = st.columns(2)

if 'status' not in st.session_state.keys():
    st.session_state['status'] = 0

with col_left:
    num_iters = st.number_input('Количество итераций', min_value=1, step=1, value=st.session_state['num_iters'])
    num_population = st.number_input('Размер популяции', min_value=1, step=1, value=st.session_state['num_population'])
    top_individuals = st.number_input('Количество особей для отбора', min_value=1, max_value=num_population, step=1,
                                      value=st.session_state['top_individuals'])
    fig_count = st.slider('Количество фигур', min_value=1, max_value=9, step=1, value=st.session_state['fig_count'])
    mut_prob = st.slider('Вероятность мутации', min_value=.0, max_value=1., value=.1, step=.001)

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

loop_table(col_right)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
