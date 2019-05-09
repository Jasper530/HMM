import numpy as np

def generate_markov(r, c):
    X = np.random.random((r,c))
    X = X / np.sum(X, axis=1, keepdims=True)
    return X

class HMM():

    def __init__(self, n_state_space, n_obs_space):
        self.N = n_state_space
        self.M = n_obs_space
        self.n_steps = None

        self.pi = np.ones(n_state_space)/n_state_space
        self.A = generate_markov(n_state_space, n_state_space)
        self.B = generate_markov(n_state_space, n_obs_space)

    def _print_info(self):
        print('transition prob:')
        print(self.A)
        print('emission prob:')
        print(self.B)

    def _forward(self, X):
        alpha = np.zeros((self.N, self.n_steps))
        # t = 1
        alpha[:,0] = self.pi * self.B[:,X[0]]
        # t = 2 to t = n
        for t in range(1, self.n_steps):
            alpha[:,t] = self.A.T @ alpha[:,t-1] * self.B[:,X[t]]
            
        return alpha

    def _forward_test(self, X):
        alpha = np.zeros((self.N, self.n_steps))
        # t = 1
        for i in range(self.N):
            alpha[i,0] = self.pi[i] * self.B[i,X[0]]
        # t = 2 to t = n
        for t in range(self.n_steps-1):
            for j in range(self.N):
                tmp = 0
                for i in range(self.N):
                    tmp += self.A[i,j]*alpha[i,t]
                alpha[j,t+1] = tmp * self.B[j,X[t+1]]
        return alpha

    def _backward(self, X):
        beta = np.zeros((self.N, self.n_steps))
        # t = n
        beta[:,-1] = 1
        # t = n-1 to t = 1
        for t in range(self.n_steps-2, -1, -1):
            beta[:,t] = self.A @ (self.B[:,X[t+1]]*beta[:,t+1])
        return beta

    def _backward_test(self, X):
        beta = np.zeros((self.N, self.n_steps))
        # t = n
        beta[:,-1] = 1
        # t = n-1 to t = 1
        for t in range(self.n_steps-2, -1, -1):
            for i in range(self.N):
                tmp = 0
                for j in range(self.N):
                    tmp += self.A[i,j]*self.B[j,X[t+1]]*beta[j,t+1]
                beta[i,t] = tmp
        return beta

    def fit(self, X, max_iters=500):
        self.n_steps = len(X)

        for iter in range(max_iters):
            # calculate alpha & beta
            alpha = self._forward(X)
            beta = self._backward(X)

            # test for alpha & beta
            """ alpha_test = self._forward_test(X)
            print(alpha_test - alpha)
            beta_test = self._backward_test(X)
            print(beta_test - beta)
            break """

            # calculate u & w
            u = alpha * beta
            u /= np.sum(u, axis=0)
            w = np.zeros((self.N, self.N, self.n_steps-1))
            for t in range(1, self.n_steps):
                w[:,:,t-1] = self.A * np.outer(alpha[:,t-1], beta[:,t]) * self.B[:,X[t]]
            w /= np.sum(w, axis=(0,1))

            # test for w
            """ w_test = np.zeros((self.N, self.N, self.n_steps-1))
            for t in range(1, self.n_steps):
                for i in range(self.N):
                    for j in range(self.N):
                        w_test[i,j,t-1] = self.A[i,j]*self.B[j,X[t]]*alpha[i,t-1]*beta[j,t]
            print(np.linalg.norm(w-w_test))
            break """
    
            # e-step
            C = np.sum(w, axis=2)
            D = np.zeros((self.N, self.M))
            idx = {x: X == x for x in range(self.M)}
            for x in range(self.M):
                D[:,x] = np.sum(u[:,idx[x]], axis=1)

            # m-step
            C /= np.sum(C, axis=1, keepdims=True)
            D /= np.sum(D, axis=1, keepdims=True)

            if np.allclose(C, self.A, atol=1e-2) and np.allclose(D, self.B, atol=1e-2):
                print('training finished')
                self._print_info()
                print(iter+1)
                break
            
            self.A = C
            self.B = D
        else:
            print('not converged')
            self._print_info()
            print(iter+1)

    def predict(self, X):

        delta = np.zeros((self.N, self.n_steps))
        gamma = np.zeros((self.N, self.n_steps))

        delta[:,0] = self.pi * self.B[:,X[0]]
        for t in range(1, self.n_steps):
            tmp = self.A.T * delta[:,t-1]
            delta[:,t] = np.max(tmp, axis=1) * self.B[:,X[t]]
            gamma[:,t] = np.argmax(tmp, axis=1)
        
        #print(delta)
        #print(gamma)

        # backward tracking
        Z = np.zeros(self.n_steps, dtype=np.int16)
        Z[-1] = np.argmax(delta[:,-1])
        for t in range(self.n_steps-2, -1, -1):
            Z[t] = gamma[Z[t+1],t+1]
        
        return Z

if __name__ == "__main__":


    X = np.array([1,0,1,1,0,1])
    hmm = HMM(2,2)
    hmm.fit(X)

   
    