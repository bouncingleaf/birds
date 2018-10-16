import os
import nn_30birds

for i in [0.01, 0.005, 0.001, 0.0001]:
    nn_30birds.main(epochs=6, display_every=10, learning_rate=i, batch_size=100)