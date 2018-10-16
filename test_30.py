import os
import nn_30birds

if os.path.exists('/Users/leaf/CS767/'):
    image_dir = '/Users/leaf/CS767/30birds128/'
    output_dir = '/Users/leaf/CS767/birds/output/'
    model_file = '/Users/leaf/CS767/birds/models/nn_30birds.ckpt'
else:
    image_dir = 'C:/datasets/Combined/processed/30birds128/'
    output_dir = 'C:/Users/Leaf/Google Drive/School/BU-MET-CS-767/Project/birds/output'
    model_file = 'C:/Users/Leaf/Google Drive/School/BU-MET-CS-767/Project/birds/models/nn_30birds.ckpt'

for i in [0.01, 0.005, 0.001, 0.0001]:
    nn_30birds.main(epochs=6, learning_rate=i, image_dir=image_dir, output_dir=output_dir, model_file=model_file)