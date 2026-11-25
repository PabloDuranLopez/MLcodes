class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters=n_clusters
        self.max_iter=max_iter
        self.random_state=random_state

    def fit(self, X):
        
        X= np.asarray(X, dtype=float)
        N,D = X.shape

        rng=np.random.default_rng(self.random_state)

        centros=rng.choice(N, size=self.n_clusters, replace=False)

        self.centroides=X[centros].copy()
        
        self.historial=[]
        
        self.etiqueta=np.full(N, -1)
        
        for i in range(self.max_iter):

            centroides_viejos=self.centroides.copy()
        
            distances=((X[:,None,:]-self.centroides[None,:,:])**2).sum(axis=2)
        
            self.etiqueta=np.argmin(distances, axis=1)

            for k in range(self.n_clusters):

                xk=X[self.etiqueta==k]

                if len(xk)>0:
                    self.centroides[k]=xk.mean(axis=0)

                else:
                    j = rng.integers(N)
                    self.centroides[k] = X[j]
            self.n_iter = i + 1
            distances = ((X[:,None,:] -self.centroides[None,:,:])**2).sum(axis=2)
            self.etiqueta = np.argmin(distances, axis=1)
            errores = np.sum((X-self.centroides[self.etiqueta])**2)
            self.historial.append(errores)
            if np.allclose(self.centroides,centroides_viejos):
                            break
        print(f"Numero de iteraciones:{self.n_iter}")

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = ((X[:, None, :] - self.centroides[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(distances, axis=1)
