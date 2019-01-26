import numpy as np
import copy
from util import print_iter


def q_from_v(env, V, s, gamma=1):
    q = np.zeros(env.nA)
    for action in range(env.nA):
        for prob, next_state, reward, done in env.P[s][action]:
            q[action] += prob * (reward + gamma * V[next_state])
    return q


def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for state in range(env.nS):
            Vs = 0
            for action, action_prop in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    Vs += action_prop * prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(Vs - V[state]))
            V[state] = Vs
        if delta < theta:
            break
    return V


def truncated_policy_evaluation(env, policy, V, max_it=1, gamma=1):
    counter = 0
    while counter < max_it:
        for state in range(env.nS):
            Vs = 0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    Vs += action_prob * prob * (reward + gamma * V[next_state])
            V[state] = Vs
        counter += 1
    return V


def policy_improvement(env, V, gamma=1):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for state in range(env.nS):
        q = q_from_v(env, V, state, gamma)
        best_actions = np.argwhere(q == max(q)).flatten()
        policy[state, best_actions[0]] = 1.0
    return policy


def policy_iteration(env, gamma=1, theta=1e-8):
    policy = np.ones([env.nS, env.nA]) / env.nA
    i = 1
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        if (new_policy == policy).all():
            break
        policy = copy.copy(new_policy)
        print_iter(i, V, env, policy)
        i += 1
    return policy, V


def truncated_policy_iteration(env, max_it=90000000000, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA]) / env.nA
    i = 1
    while True:
        policy = policy_improvement(env, V, gamma)
        old_V = copy.copy(V)
        V = truncated_policy_evaluation(env, policy, V, max_it, gamma)
        if max(abs(V - old_V)) < theta:
            break
        print_iter(i, V, env, policy)
        i += 1
    return policy, V


def value_iteration(env, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    i = 1
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            V[s] = max(q_from_v(env, V, s, gamma))
            delta = max(delta, abs(V[s] - v))
        if delta < theta:
            break
        print_iter(i, V, env)
        i += 1
    policy = policy_improvement(env, V, gamma)
    return policy, V
