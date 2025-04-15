from .model import build_models
from .utils import *
import time
import numpy as np
from tensorflow.keras.utils import to_categorical

def train(X,
          A,
          y,
          yt=None,
          idx=None,
          emb_dim=64,
          use_bias=False,
          enable_dann=True,
          n_iterations=1000,
          loss_weight = [0.1,1],
          alpha_lr=[0.01,0.01],
          initial_train=True,
          early_stopping = 10, 
          initial_train_epochs=10):

    inp_dim = X.shape[1]
    ncls_source = y.shape[1]
    X_preinit = np.zeros_like(X)

    model, source_classification_model, domain_classification_model, embeddings_model = \
        build_models(inp_dim, emb_dim, ncls_source, use_bias=use_bias, alpha_lr=alpha_lr,
                     loss_weight = loss_weight)

    y_adversarial_1 = to_categorical(np.array(([0] * len(idx[0]) + [1] * len(idx[1]) +
                                               [0] * len(idx[2]) + [0] * len(idx[3]))))
    y_adversarial_2 = to_categorical(np.array(([1] * len(idx[0]) + [0] * len(idx[1]) +
                                               [1] * len(idx[2]) + [1] * len(idx[3]))))
    sample_weights_class = np.array(([1] * len(idx[0]) + [0] * len(idx[1]) +
                                     [1] * len(idx[2]) + [1] * len(idx[3])))
    sample_weights_adversarial = np.array(([1] * len(idx[0]) + [1] * len(idx[1]) +
                                           [1] * len(idx[2]) + [1] * len(idx[3])))

    if initial_train:

        source_classification_model.fit([X, A], [y],
                                        sample_weight=[sample_weights_class],
                                        batch_size=A.shape[0],
                                        epochs=initial_train_epochs,
                                        shuffle=False,
                                        verbose=0)
        encoder_weights = []
        for layer in model.layers:
            if (layer.name.startswith("en")):
                encoder_weights.append(layer.get_weights())

        domain_classification_model.fit([X, A], y_adversarial_2,
                                        batch_size=A.shape[0],
                                        epochs=1,
                                        shuffle=False,
                                        verbose=0)

        k = 0
        for layer in model.layers:
            if (layer.name.startswith("en")):
                layer.set_weights(encoder_weights[k])
                k += 1
        print("initial_train_done")

    for i in range(n_iterations):
        t = time.time()


        adv_weights = []
        for layer in model.layers:
            if (layer.name.startswith("do")):
                adv_weights.append(layer.get_weights())

        if (enable_dann):

            model.fit([X, A], [y, y_adversarial_1],
                      sample_weight=[sample_weights_class, sample_weights_adversarial],
                      batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

            encoder_weights = []
            for layer in model.layers:
                if (layer.name.startswith("en")):
                    encoder_weights.append(layer.get_weights())

            k = 0
            for layer in model.layers:
                if (layer.name.startswith("do")):
                    layer.set_weights(adv_weights[k])
                    k += 1

            domain_classification_model.fit([X, A], [y_adversarial_2],
                                            batch_size=A.shape[0], epochs=1,
                                            shuffle=False, verbose=0)

            k = 0
            for layer in model.layers:
                if (layer.name.startswith("en")):
                    layer.set_weights(encoder_weights[k])
                    k += 1

        else:
            source_classification_model.fit([X, A], [y],
                                            sample_weight=[sample_weights_class],
                                            batch_size=A.shape[0], epochs=1, shuffle=False,
                                            verbose=0)
        loss_val = []
        preds = source_classification_model.predict([X, A], batch_size=A.shape[0], verbose=0)
        loss_, acc = evaluate_preds(preds, yt, [idx[0], idx[2], idx[3]])
        train_loss = loss_[0]
        val_loss = loss_[1]
        test_loss = loss_[2]
        loss_val.append(val_loss)
        emb = embeddings_model.predict([X, A], batch_size=A.shape[0], verbose=0)

        if (i % 200 == 0):
            print(
                "Epoch: {:04d}".format(i),
                "train_loss= {:.4f}".format(train_loss),
                "train_acc= {:.4f}".format(acc[0]),
                "val_loss= {:.4f}".format(val_loss),
                "val_acc= {:.4f}".format(acc[1]),
                "test_loss= {:.4f}".format(test_loss),
                "test_acc= {:.4f}".format(acc[2]),
                "time= {:.4f}".format(time.time() - t)
            )

        # early stop condition 
        if i > early_stopping and loss_val > np.mean(loss_val[-(early_stopping + 1):-1]):
            print('Epoch {}: early stopping'.format(i))
            break
    return preds, emb









