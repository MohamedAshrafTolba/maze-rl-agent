import numpy as np
import copy


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


def policy_improvement(env, V, gamma=1):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for state in range(env.nS):
        q = q_from_v(env, V, state, gamma)
        best_actions = np.argwhere(q == max(q)).flatten()
        policy[state] = np.sum([np.eye(env.nA)[i] for i in best_actions], axis=0) / len(best_actions)
    return policy


def policy_iteration(env, gamma=1, theta=1e-8):
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        if (new_policy == policy).all():
            break
        policy = copy.copy(new_policy)
    return policy, V


def value_iteration(env, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            V[s] = max(q_from_v(env, V, s, gamma))
            delta = max(delta, abs(V[s] - v))
        if delta < theta:
            break
    policy = policy_improvement(env, V, gamma)
    return policy, V
