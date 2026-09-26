import numpy as np

class NaiveBayes:
    def __init__(self,epsilon=1e-6):
        self.classes=None
        self.means=None
        self.variances=None
        self.priors=None
        self.epsilon=epsilon
        self.nfeatures=None
        self.nclass=None

    def fit(self,X,y):
        self.classes=np.unique(y)

        self.nclasses=len(self.classes)

        self.nfeatures=X.shape[1]

        self.means=np.zeros((self.nclasses, self.nfeatures))

        self.priors=np.zeros(self.nclasses)

        self.variances=np.zeros((self.nclasses, self.nfeatures))

        for index, k in enumerate(self.classes):
            Xk=X[y == k]

            Nk=len(Xk)

            self.priors[index]=Nk/len(y)

            muk=np.mean(Xk, axis=0)

            self.means[index]=muk

            self.variances[index]=np.var(Xk, axis=0)+self.epsilon
        return self

    def log_gaussian(self,X, mean, var):
        log_gauss=-0.5*np.sum(np.log(2*np.pi*var))-0.5*np.sum(((X-mean)**2)/var)
        return log_gauss

    def log_likelihood(self, X):
        scores=np.zeros((X.shape[0], self.nclasses))
        for i, x in enumerate(X):
            for index, k in enumerate(self.classes):

                mean=self.means[index]

                var=self.variances[index]

                prior=self.priors[index]

                scores[i,index]=self.log_gaussian(x, mean, var)+np.log(prior)
        return scores

    def predict(self, X):
        y_pred = np.empty(X.shape[0],dtype=self.classes.dtype)
        scores=self.log_likelihood(X)
        for i,x in enumerate(X):
            y_pred[i]=self.classes[np.argmax(scores[i])]
        return y_pred
    
    def predict_proba(self,X):
        scores=self.log_likelihood(X)
        numerator=np.exp(scores)
        denominator=np.sum(numerator, axis=1, keepdims=True)
        probabilities=numerator/denominator
        return probabilities






