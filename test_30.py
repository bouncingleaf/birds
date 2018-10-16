import nn_30birds

for i in [0.1, 0.01, 0.005, 0.001]:
    nn_30birds.main(epochs=1, learning_rate=i)