import itertools

import numpy as np
import tensorflow as tf


class PHPAgent(object):
    def __init__(self):
        self.o = None
        self.tau = None
        self.root = None
        self.stack = None
        self.php_index = None

    def get_stack(self):
        return [(self.php_index[h], tau) for (h, tau) in self.stack]

    def step(self, observation, reset=False):
        sess = tf.get_default_session()
        if reset:
            self.stack = [[self.root, 0]]
        else:
            while True:
                [term] = sess.run(self.stack[-1][0].psi, {self.o: [observation], self.tau: [self.stack[-1][1]]})
                if not term:
                    break
                self.stack.pop()
                if not self.stack:
                    return None
            self.stack[-1][1] += 1
        while not self.stack[-1][0].is_producer:
            [proc] = sess.run(self.stack[-1][0].eta, {self.o: [observation], self.tau: [self.stack[-1][1]]})
            self.stack.append([self.stack[-1][0].lookup[proc], 0])
        [action] = sess.run(self.stack[-1][0].pi, {self.o: [observation], self.tau: [self.stack[-1][1]]})
        return action


class PHP(object):
    def __init__(self, name, pi, psi=None, lookup=None, end_tau=None):
        self.name = name
        self.lookup = lookup
        if self.lookup is None:
            self.is_producer = True
            self.pi = pi
        else:
            self.is_producer = False
            self.eta = pi
        self.psi = psi
        self.end_tau = end_tau

    def __repr__(self):
        return self.name


class Trajectory(object):
    def __init__(self, branch_index):
        self.o = []
        self.m = []
        self.a = []
        self.length = None
        self.branch_index = branch_index

    def step(self, o, m, a):
        self.o.append(o)
        if m:
            # print(m)
            self.m.append(self.branch_index[tuple(m)])
            self.a.append(a)
        else:
            self.length = len(self.a)
            self.o = np.asarray(self.o)
            self.m = np.asarray(self.m)
            self.a = np.asarray(self.a)


def rollout_agent(env, agent, branch_index, *args, reset=True, max_length=None):
    traj = Trajectory(branch_index)
    if reset:
        o = env.reset(*args)
    else:
        o = env.observe()
    for step in itertools.count():
        a = agent.step(o, step == 0)
        traj.step(o, agent.get_stack(), a)
        if a is None:
            break
        o = env.step(a)
        if step == max_length:
            print('trajectory too long!')
            break
    return traj


def get_trajs(env, agent, branch_index, num_trajs, *args):
    return [rollout_agent(env, agent, branch_index, *args) for _ in range(num_trajs)]


class Hierarchy(object):
    def __init__(self, phps):
        self.phps = phps
        self.php_index = {p: i for i, p in enumerate(self.phps)}
        self.num_states = sum(1 for _ in self.gen_branches())
        self.num_producers = sum(p.is_producer for p in self.phps)
        self.num_internals = len(self.phps) - self.num_producers
        M = self.num_states
        H = self.num_internals
        K = self.num_producers
        T = tf.shape(self.phps[0].log_eta)[0]
        TAU = max(p.end_tau for p in self.phps)
        assert all(not p.is_producer for p in self.phps[:H])
        assert all(p.is_producer for p in self.phps[H:])
        lookups = [np.zeros([len(p.lookup), H + K], np.float32) for p in self.phps[:H]]
        for h, p in enumerate(self.phps[:H]):
            for sub_h, sub in enumerate(p.lookup):
                lookups[h][sub_h, self.php_index[sub]] = 1
        self.log_eta = tf.stack([tf.tensordot(p.log_eta, lookups[h], 1) for h, p in enumerate(self.phps[:H])], 1)  # [T, H, TAU, H+K]
        self.log_pi = tf.stack([p.log_pi for p in self.phps[H:]], 1)  # [T, K, TAU]
        self.log_psi = tf.stack([p.log_psi for p in self.phps], 1)  # [T, H+K, TAU, 2]
        log_init = [tf.constant(-np.inf)] * M
        log_tran = [[tf.fill([T - 1], -np.inf)] * M] * M
        log_term = [tf.constant(-np.inf)] * M
        log_emit = [tf.fill([T], -np.inf)] * M
        self.w_eta_selector = np.zeros([M, H, TAU, H + K], np.float32)
        self.w_pi_selector = np.zeros([M, K, TAU], np.float32)
        self.w_psi_selector = np.zeros([M, M, H + K, TAU, 2], np.float32)
        self.w_psi_term_selector = np.zeros([M, H + K, TAU, 2], np.float32)
        for m, branch in enumerate(self.gen_branches()):
            if all(f[1] == 0 for f in branch):
                log_init[m] = tf.add_n([self.log_eta[0, branch[l - 1][0], 0, branch[l][0]] for l in range(1, len(branch))])
            log_term[m] = tf.add_n([self.log_psi[-1, f[0], f[1], 1] for f in branch])
            log_emit[m] = self.log_pi[:, branch[-1][0] - H, branch[-1][1]]
            for l in range(1, len(branch)):
                if branch[l][1] == 0:
                    self.w_eta_selector[m, branch[l - 1][0], branch[l - 1][1], branch[l][0]] = 1
            self.w_pi_selector[m, branch[-1][0] - H, branch[-1][1]] = 1
            for f in branch:
                self.w_psi_term_selector[m, f[0], f[1], 1] = 1
            for next_m, next_branch in enumerate(self.gen_branches()):
                for l in range(len(branch)):
                    if branch[l] != next_branch[l]:
                        break
                else:
                    continue
                if branch[l][0] != next_branch[l][0]:
                    continue
                if branch[l][1] + 1 != next_branch[l][1]:
                    continue
                if not all(f[1] == 0 for f in next_branch[l + 1:]):
                    continue
                log_tran[m][next_m] = tf.add_n(
                    [self.log_psi[:-1, f[0], f[1], 1] for f in branch[l + 1:]]
                    + [self.log_psi[:-1, branch[l][0], branch[l][1], 0]]
                    + [self.log_eta[:-1, next_branch[l_ - 1][0], next_branch[l_ - 1][1], next_branch[l_][0]]
                       for l_ in range(l + 1, len(next_branch))])
                self.w_psi_selector[m, next_m, branch[l][0], branch[l][1], 0] = 1
                for f in branch[l + 1:]:
                    self.w_psi_selector[m, next_m, f[0], f[1], 1] = 1
        self.log_init = tf.stack(log_init)
        self.log_tran = tf.stack([tf.stack(l, 1) for l in log_tran], 1)
        self.log_term = tf.stack(log_term)
        self.log_emit = tf.stack(log_emit, 1)
        self.all = {
            'log_eta': self.log_eta,
            'log_init': self.log_init,
            'log_tran': self.log_tran,
            'log_term': self.log_term,
            'log_emit': self.log_emit
        }

    def gen_branches(self, p=None, branch=list()):
        if p is None:
            p = self.phps[0]
        h = self.php_index[p]
        for tau in range(p.end_tau):
            if p.is_producer:
                yield branch + [[h, tau]]
            else:
                for sub_p in p.lookup:
                    for br in self.gen_branches(sub_p, branch + [[h, tau]]):
                        yield br

    def get_w_eta_selector(self):
        return self.w_eta_selector

    def get_w_pi_selector(self):
        return self.w_pi_selector

    def get_w_psi_selector(self):
        return self.w_psi_selector

    def get_w_psi_term_selector(self):
        return self.w_psi_term_selector
