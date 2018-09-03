import pickle
import time

import numpy as np
import tensorflow as tf
from scipy.misc import logsumexp


class PHPTrainer(object):
    def __init__(self, theta, o, a, hierarchy, opt=tf.train.AdamOptimizer(1e-4)):
        """

        :param o: [T, O]
        :param a: [T, A]
        hierarchy.log_eta:  [T + 1, O]         -> [T, H, TAU, H+K] ; log_eta[t, h, tau, h'] = log eta_h^tau(h' | o_t)
        hierarchy.log_pi:   [T + 1, O], [T, A] -> [T, K, TAU]      ; log_pi[t, h, tau]      = log pi_h^tau(a_t | o_t)
        hierarchy.log_psi:  [T + 1, O]         -> [T, H+K, TAU, 2] ; log_psi[t, h, tau, 1]  = log psi_h^tau(o_{t+1})
        hierarchy.log_init: [T + 1, O]         -> [M]              ; log_init[m]            = log Pr(m_0 = m | o_0)
        hierarchy.log_tran: [T + 1, O]         -> [T-1, M, M]      ; log_tran[t, m, m']     = log Pr(m_{t+1} = m' | m_t = m, o_{t+1})
        hierarchy.log_term: [T + 1, O]         -> [M]              ; log_term[m]            = log Pr(m_T = [] | m_{T-1} = m, o_T)
        hierarchy.log_emit: [T + 1, O], [T, A] -> [T, M]           ; log_emit[t, m]         = log Pr(a_t | m_t = m, o_t)
        """
        self.theta = theta
        self.o = o
        self.a = a
        self.hierarchy = hierarchy
        M = self.hierarchy.num_states
        self.vij = tf.placeholder(tf.float32, [None, M, M])  # [T-1, M, M]
        self.vi = tf.placeholder(tf.float32, [None, M])  # [T, M]
        self.entropy_weight = tf.placeholder(tf.float32, [])
        self.gradients = list(zip(*opt.compute_gradients(self.make_loss(), self.theta)))[0]
        self.opt = opt.apply_gradients(zip(self.gradients, self.theta))

    def make_loss(self):
        w_eta = tf.tensordot(self.vi, self.hierarchy.get_w_eta_selector(), 1)  # [T, H, TAU, H+K]
        w_pi = tf.tensordot(self.vi, self.hierarchy.get_w_pi_selector(), 1)  # [T, K, TAU]
        w_psi = tf.concat([
            tf.tensordot(self.vij, self.hierarchy.get_w_psi_selector(), [[1, 2], [0, 1]]),
            tf.tensordot(self.vi[-1:, :], self.hierarchy.get_w_psi_term_selector(), 1)], 0)  # [T, H+K, TAU, 2]
        log_prob = (tf.reduce_sum(w_eta * self.hierarchy.log_eta)
                    + tf.reduce_sum(w_pi * self.hierarchy.log_pi)
                    + tf.reduce_sum(w_psi * self.hierarchy.log_psi))
        entropy = -(tf.reduce_sum(tf.exp(self.hierarchy.log_eta) * self.hierarchy.log_eta)
                    + tf.reduce_sum(tf.exp(self.hierarchy.log_pi) * self.hierarchy.log_pi)
                    + tf.reduce_sum(tf.exp(self.hierarchy.log_psi) * self.hierarchy.log_psi))
        return -log_prob - self.entropy_weight * entropy

    @staticmethod
    def forward(results):
        log_init = results['log_init']
        log_tran = results['log_tran']
        log_emit = results['log_emit']
        T, M = log_emit.shape
        log_phi = np.full([T, M], -np.inf, np.float32)
        log_phi[0, :] = log_init[:]
        for t in range(T - 1):
            log_phi[t + 1, :] = logsumexp(log_phi[t, :, None] + log_emit[t, :, None] + log_tran[t, :, :], 0)
        results['log_phi'] = log_phi

    @staticmethod
    def backward(results):
        log_tran = results['log_tran']
        log_term = results['log_term']
        log_emit = results['log_emit']
        T, M = log_emit.shape
        log_omega = np.full([T, M], -np.inf, np.float32)
        log_omega[T - 1, :] = log_emit[T - 1, :] + log_term[:]
        for t in range(T - 2, -1, -1):
            log_omega[t, :] = log_emit[t, :] + logsumexp(log_tran[t, :, :] + log_omega[t + 1, None, :], 1)
        results['log_omega'] = log_omega

    def opt_step(self, traj, entropy_weight):
        sess = tf.get_default_session()
        results = sess.run(self.hierarchy.all, feed_dict={self.o: traj.o, self.a: traj.a})
        log_tran = results['log_tran']
        log_emit = results['log_emit']
        T, M = log_emit.shape
        if hasattr(traj, 'm'):
            vij = np.zeros([T - 1, M, M])
            vi = np.zeros([T, M])
            for t in range(T - 1):
                vij[t, traj.m[t], traj.m[t + 1]] = 1.
            vi[:-1, :] = vij.sum(2)
            vi[-1, traj.m[-1]] = 1.
        else:
            self.forward(results)
            self.backward(results)
            log_phi = results['log_phi']
            log_omega = results['log_omega']
            log_likelihood = logsumexp(log_phi[0, :] + log_omega[0, :])
            vij = np.exp(log_phi[:-1, :, None] + log_emit[:-1, :, None] + log_tran[:, :, :] + log_omega[1:, None, :] - log_likelihood)
            vi = np.zeros([T, M])
            vi[:-1, :] = vij.sum(2)
            vi[-1, :] = np.exp(log_phi[-1, :] + log_omega[-1, :] - log_likelihood)
        return sess.run(self.gradients, feed_dict={self.o: traj.o, self.a: traj.a, self.vij: vij, self.vi: vi, self.entropy_weight: entropy_weight})

    def score(self, traj, correctness=True):
        sess = tf.get_default_session()
        results = sess.run(self.hierarchy.all, feed_dict={self.o: traj.o, self.a: traj.a})
        self.backward(results)
        log_init = results['log_init']
        log_tran = results['log_tran']
        log_term = results['log_term']
        log_emit = results['log_emit']
        log_omega = results['log_omega']
        log_likelihood = logsumexp(log_init[:] + log_omega[0, :])
        if not correctness:
            return log_likelihood
        T = traj.length
        m_llik = log_init[traj.m[0]] + sum(log_tran[t, traj.m[t], traj.m[t + 1]] for t in range(T - 1)) + log_term[traj.m[-1]]
        a_llik = sum(log_emit[t, traj.m[t]] for t in range(T))
        m_corr = (log_init.argmax() == traj.m[0]) + (log_term.argmax() == traj.m[-1]) + sum(
            (log_tran[t].argmax() == [traj.m[t], traj.m[t + 1]]).all()
            for t in range(T - 1))
        a_corr = sum(log_emit[t].argmax() == traj.m[t] for t in range(T))
        return log_likelihood, m_llik, a_llik, m_corr, a_corr

    def train(self, out_path, train_trajs, valid_trajs, num_steps=1000):
        sess = tf.get_default_session()
        start_time = time.time()
        for step in range(num_steps + 1):
            entropy_weight = 1e-2 / (step + 1.)
            if step % 10 == 0 or step == num_steps:
                its_per_sec = (step + 1) / (time.time() - start_time)
                train_log_likelihood = 0.
                total_length = 0
                for traj in train_trajs:
                    train_log_likelihood += self.score(traj, False)
                    total_length += traj.length
                train_log_likelihood /= total_length
                total_log_likelihood = 0.
                total_m_llik = 0
                total_a_llik = 0
                total_m_corr = 0
                total_a_corr = 0
                total_length = 0
                for traj in valid_trajs:
                    log_likelihood, m_llik, a_llik, m_corr, a_corr = self.score(traj)
                    total_log_likelihood += log_likelihood
                    total_m_llik += m_llik
                    total_a_llik += a_llik
                    total_m_corr += m_corr
                    total_a_corr += a_corr
                    total_length += traj.length
                mean_log_likelihood = total_log_likelihood / total_length
                mean_m_llik = total_m_llik / (total_length + len(valid_trajs))
                mean_a_llik = total_a_llik / total_length
                mean_m_corr = total_m_corr / (total_length + len(valid_trajs))
                mean_a_corr = total_a_corr / total_length
                print('step #{} train ll: {}, test ll: {}, m llik: {}, a llik: {}, m corr: {}, a corr: {}, ent w: {} ({} it/s)'
                      .format(step, train_log_likelihood, mean_log_likelihood, mean_m_llik, mean_a_llik, mean_m_corr, mean_a_corr, entropy_weight, its_per_sec))
            if step % 100 == 0:
                pickle.dump(sess.run(self.theta), open('{}/snapshot{:06d}.{}.pkl'.format(out_path, step, round(mean_log_likelihood, 2)), 'wb'))
            if step < num_steps:
                grad = None
                for traj in train_trajs:
                    grad = self.acc_grad(grad, self.opt_step(traj, entropy_weight))
                sess.run(self.opt, feed_dict={self.gradients: grad})
        pickle.dump(sess.run(self.theta), open('{}/params.pkl'.format(out_path), 'wb'))
        return self.theta

    @staticmethod
    def acc_grad(acc, grad):
        if acc is None:
            return grad
        assert (len(acc) == len(grad))
        return [acc[i] + grad[i] for i in range(len(acc))]
