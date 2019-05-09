import numpy as np
from hmmlearn import hmm
from hmm_naive import HMM, generate_markov

def get_sample(n_steps=100):

    model = hmm.MultinomialHMM(3)
    model.startprob_ = np.array([1, 1, 1]) / 3
    """ model.transmat_ = np.array([[0.35, 0.2, 0.45],
                                [0.3, 0.5, 0.2],
                                [0.3, 0.3, 0.4]]) """
    model.transmat_ = generate_markov(3, 3)
    print('ground true')
    print(model.transmat_)
    model.n_features = 2
    model.emissionprob_ = np.array([[0.5, 0.5], [0.2, 0.8], [0.6, 0.4]])
    X, _ = model.sample(n_steps)
    return X

def get_hmmlearn_ref(X):

    model = hmm.MultinomialHMM(3, random_state=42, n_iter=10, tol=1e-2)
    model.n_features = 2
    model.fit(X)
    print(model.predict(X))

    return model.transmat_, model.emissionprob_


if __name__ == "__main__":
    n = 50
    X = get_sample(n)

    A, B = get_hmmlearn_ref(X)
    print('hmmlearn:')
    print(A)
    print('-'*50)
    print(B)

    X = X.reshape(n,)
    model = HMM(3,2)
    model.fit(X)
    Z = model.predict(X)
    print(Z)


