from keras.models import Model

import sys

import numpy as np

from sklearn.cluster import KMeans
from time import time

from ..layers import ClusteringLayer


class DCEC(object):
    def __init__(
            self,
            input_shape,
            n_clusters=10,
            alpha=1.0,
            save_dir='/',
            CAE=None,
            name=''
    ):

        super(DCEC, self).__init__()

        self.name = name
        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []
        self.y_pred_last = []
        self.save_dir = save_dir

        self.cae = CAE

        hidden = self.cae.get_layer(name='embedding').output

        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

        clustering_layer = ClusteringLayer(
            self.n_clusters,
            name='clustering'
        )(hidden)

        self.model = Model(
            inputs=self.cae.input,
            outputs=[
                clustering_layer,
                self.cae.output
            ]
        )

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    def fit(
            self,
            x,
            batch_size=256,
            maxiter=2e4,
            tol=1e-3,
            update_interval=140
    ):
        save_interval = x.shape[0] / batch_size * 5

        t1 = time()
        print('Initializing cluster centers with k-means.')

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)

        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)

        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        t2 = time()
        loss = [0, 0, 0]
        index = 0

        print('Compiling Model')
        self.model.compile(loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam')

        for _ in range(int(maxiter / update_interval)):
            q, _ = self.model.predict(x, verbose=0)
            p = self.target_distribution(q)

        history = {
            'loss': [],
            'delta': []
        }

        print('Starting Training')
        for ite in range(int(maxiter)):
            print(loss)
            if ite % update_interval == 0:
                y_pred = self.model.predict(x, verbose=0)

                q, _ = y_pred

                p = self.target_distribution(q)

                self.y_pred = q.argmax(1)

                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32)
                delta_label /= self.y_pred.shape[0]

                history['delta'].append(delta_label)

                sys.stdout.write('\riteration: {:<5} delta: {:.5f}                 '.format(ite, delta_label))

                y_pred_last = np.copy(self.y_pred)

                if ite > 0 and delta_label < tol:
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(
                    x=x[index * batch_size::],
                    y=[
                        p[index * batch_size::],
                        x[index * batch_size::]
                    ]
                )

                index = 0
            else:
                loss = self.model.train_on_batch(
                    x=x[index * batch_size:(index + 1) * batch_size],
                    y=[
                        p[index * batch_size:(index + 1) * batch_size],
                        x[index * batch_size:(index + 1) * batch_size]
                    ]
                )

                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                self.model.save(self.save_dir + '/dcec_model_100.h5')

            history['loss'].append(loss)
            ite += 1

        # save the trained model
        print('saving model to:', self.save_dir + f'/dcec_model_{self.name}.h5')
        self.model.save(self.save_dir + f'/dcec_model_{self.name}.h5')
        t3 = time()
        print('Time:', t3 - t1)
        return history
