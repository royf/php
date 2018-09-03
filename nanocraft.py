import os
import pickle

import numpy as np
import tensorflow as tf

import hhmm
import php
import tftools


class NanoCraft(object):
    def __init__(self):
        self.num_actions = 5
        self.state = None
        self.maxw = None
        self.maxh = None

    def reset(self, maxw=6, maxh=6):
        self.maxw = maxw
        self.maxh = maxh
        self.state = {
            'x': 0,
            'y': 0,
            'board': np.zeros((self.maxw, self.maxh), np.int8)
        }
        bx, bex = sorted(np.random.choice(self.maxw - 1, 2, False))
        self.state['bx'] = bx + 1
        self.state['bw'] = bex - bx + 1
        by, bey = sorted(np.random.choice(self.maxh - 1, 2, False))
        self.state['by'] = by + 1
        self.state['bh'] = bey - by + 1
        for x in range(self.state['bx'], self.state['bx'] + self.state['bw']):
            self.state['board'][x, self.state['by']] = np.random.randint(2)
            self.state['board'][x, self.state['by'] + self.state['bh'] - 1] = np.random.randint(2)
        for y in range(self.state['by'] + 1, self.state['by'] + self.state['bh'] - 1):
            self.state['board'][self.state['bx'], y] = np.random.randint(2)
            self.state['board'][self.state['bx'] + self.state['bw'] - 1, y] = np.random.randint(2)
        return self.observe()

    def observe(self):
        return [self.state['board'][self.state['x'], self.state['y']],
                self.state['bx'],
                self.state['by'],
                self.state['bw'],
                self.state['bh']]

    def step(self, action):
        dx, dy, db = [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0]
        ][action]
        if db:
            self.state['board'][self.state['x'], self.state['y']] = 1
        else:
            x = self.state['x'] + dx
            y = self.state['y'] + dy
            if 0 <= x < self.maxw and 0 <= y < self.maxh:
                self.state['x'] = x
                self.state['y'] = y
        return self.observe()


class Expert(php.PHPAgent):
    def __init__(self):
        super().__init__()
        self.o = tf.placeholder(tf.int32, [None, 5])
        self.tau = tf.placeholder(tf.int32, [None])
        move_r_pi = tf.constant([1])
        move_r_psi = tf.constant([1])
        move_r = php.PHP('move_r', move_r_pi, move_r_psi, end_tau=1)
        move_d_pi = tf.constant([2])
        move_d_psi = tf.constant([1])
        move_d = php.PHP('move_d', move_d_pi, move_d_psi, end_tau=1)
        build_r_pi = tf.matmul(tf.one_hot(self.tau, 2, dtype=tf.int32), [[1], [0]])[0]
        build_r_psi = self.o[:, 0]
        build_r = php.PHP('build_r', build_r_pi, build_r_psi, end_tau=2)
        build_d_pi = tf.matmul(tf.one_hot(self.tau, 2, dtype=tf.int32), [[2], [0]])[0]
        build_d_psi = self.o[:, 0]
        build_d = php.PHP('build_d', build_d_pi, build_d_psi, end_tau=2)
        build_l_pi = tf.matmul(tf.one_hot(self.tau, 2, dtype=tf.int32), [[3], [0]])[0]
        build_l_psi = self.o[:, 0]
        build_l = php.PHP('build_l', build_l_pi, build_l_psi, end_tau=2)
        build_u_pi = tf.matmul(tf.one_hot(self.tau, 2, dtype=tf.int32), [[4], [0]])[0]
        build_u_psi = self.o[:, 0]
        build_u = php.PHP('build_u', build_u_pi, build_u_psi, end_tau=2)
        move_many_r_eta = tf.constant([0])
        move_many_r_psi = tf.equal(self.tau, self.o[:, 1] - 1)
        move_many_r_lookup = [move_r]
        move_many_r = php.PHP('move_many_r', move_many_r_eta, move_many_r_psi, move_many_r_lookup, end_tau=4)
        move_many_d_eta = tf.constant([0])
        move_many_d_psi = tf.equal(self.tau, self.o[:, 2] - 1)
        move_many_d_lookup = [move_d]
        move_many_d = php.PHP('move_many_d', move_many_d_eta, move_many_d_psi, move_many_d_lookup, end_tau=4)
        build_many_r_eta = tf.constant([0])
        build_many_r_psi = tf.equal(self.tau, self.o[:, 3] - 2)
        build_many_r_lookup = [build_r]
        build_many_r = php.PHP('build_many_r', build_many_r_eta, build_many_r_psi, build_many_r_lookup, end_tau=4)
        build_many_d_eta = tf.constant([0])
        build_many_d_psi = tf.equal(self.tau, self.o[:, 4] - 2)
        build_many_d_lookup = [build_d]
        build_many_d = php.PHP('build_many_d', build_many_d_eta, build_many_d_psi, build_many_d_lookup, end_tau=4)
        build_many_l_eta = tf.constant([0])
        build_many_l_psi = tf.equal(self.tau, self.o[:, 3] - 2)
        build_many_l_lookup = [build_l]
        build_many_l = php.PHP('build_many_l', build_many_l_eta, build_many_l_psi, build_many_l_lookup, end_tau=4)
        build_many_u_eta = tf.constant([0])
        build_many_u_psi = tf.equal(self.tau, self.o[:, 4] - 2)
        build_many_u_lookup = [build_u]
        build_many_u = php.PHP('build_many_u', build_many_u_eta, build_many_u_psi, build_many_u_lookup, end_tau=4)
        nano_eta = tf.matmul(tf.one_hot(self.tau, 6, dtype=tf.int32), [[0], [1], [2], [3], [4], [5]])[0]
        nano_psi = tf.equal(self.tau, 5)
        nano_lookup = [move_many_r, move_many_d, build_many_r, build_many_d, build_many_l, build_many_u]
        self.root = php.PHP('nano', nano_eta, nano_psi, nano_lookup, end_tau=6)
        self.phps = [self.root]
        for p in self.phps:
            if not p.is_producer:
                self.phps.extend(sorted(set(p.lookup) - set(self.phps), key=lambda x: x.name))
        self.php_index = {p: i for i, p in enumerate(self.phps)}


def build_training_agent(name, A, phps):
    with tf.variable_scope(name) as scope:
        O = 5
        o = tf.placeholder(tf.float32, [None, O])
        a = tf.placeholder(tf.int32, [None])
        a_one_hot = tf.one_hot(a, A)  # [T, A]
        T = tf.shape(a)[0]
        TAU = max(p.end_tau for p in phps)
        o_tau = tf.concat([
            tf.tile(o[:, None, :], [1, TAU, 1]),
            tf.tile(tf.cast(tf.range(TAU), tf.float32)[None, :, None], [T + 1, 1, 1])], 2)  # [T + 1, TAU, O + 1]
        hidden_size = 256

        for p in phps:
            if p.is_producer:
                del p.pi
                p.log_pi = tf.reduce_sum(tf.reshape(
                    tf.nn.log_softmax(tftools.mlp(tf.reshape(o_tau[:-1], [-1, O + 1]), hidden_size, A)),
                    [T, TAU, A]) * a_one_hot[:, None, :], 2)  # [T, TAU]
            else:
                del p.eta
                Kp = len(p.lookup)
                p.log_eta = tf.reshape(tf.nn.log_softmax(tftools.mlp(tf.reshape(o_tau[:-1], [-1, O + 1]), hidden_size, Kp)), [T, TAU, Kp])  # [T, TAU, Kp]
            del p.psi
            p.log_psi = tf.reshape(tf.nn.log_softmax(tftools.mlp(tf.reshape(o_tau[1:], [-1, O + 1]), hidden_size, 2)), [T, TAU, 2])  # [T, TAU, 2]
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        hierarchy = php.Hierarchy(phps)
        return hhmm.PHPTrainer(theta, o, a, hierarchy)


def train(path, sup, unsup, valid=100, *args):
    os.makedirs(path)
    sess = tf.InteractiveSession()
    nano = NanoCraft()
    expert = Expert()
    agent = build_training_agent('train', nano.num_actions, Expert().phps)
    trajs = php.get_trajs(nano, expert, {tuple(tuple(f) for f in b): i for i, b in enumerate(agent.hierarchy.gen_branches())}, sup + unsup + valid, *args)
    for traj in trajs[sup:sup + unsup]:
        del traj.m
    trajs_fn = '{}/trajs.pkl'.format(path)
    pickle.dump(trajs, open(trajs_fn, 'wb'))
    sess.run(tf.global_variables_initializer())
    agent.train(path, trajs[:-valid], trajs[-valid:])


def run_all():
    with open('results.txt', 'a') as f:
        for sup in [4]:  # [4, 8, 16, 32, 64, 128, 256]:
            for total in [4]:  # [4, 8, 16, 32, 64, 128, 256]:
                if sup > total:
                    continue
                unsup = total - sup
                path = 'results/nc_{}_{}'.format(sup, unsup)
                with tf.variable_scope('train_{}_{}'.format(sup, unsup)):
                    train(path, sup, unsup)
                with tf.variable_scope('test_{}_{}'.format(sup, unsup)):
                    res = 'Supervised: {}, Unsupervised: {}, all corr: {}, out corr: {}'.format(sup, unsup, *evaluate(path))
                    print(res)
                    f.write('{}\n'.format(res))
                    f.flush()


if __name__ == '__main__':
    run_all()
