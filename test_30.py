import sys
import nn_30birds

count = 0

hypatia=sys.argv[0]

for i in [0.01, 0.005, 0.001, 0.0001]:
    count = count + 1
    file_id = "L" + str(count)
    nn_30birds.main(epochs=6, display_every=20, learning_rate=i, batch_size=100, validation_mode=True, model_file_id=file_id, hypatia=hypatia)