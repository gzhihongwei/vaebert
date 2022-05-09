# vaebert

# Training
1. Download ShapeNet as ShapeNetCore.v2.zip [here](https://shapenet.org/ "ShapeNetCore.v2.zip") and mid2desc.pkl from the 3D Generative Design shape repository (https://github.com/starstorms9/shape) [here](https://github.com/starstorms9/shape/blob/master/data/mid2desc.pkl?raw=true).
2. Place these files in the "shapenet" folder in the project's root folder.
3. Run "preprocess.py" to preprocess the files.
4. Run "vaebert/vae/train.py" to train the vae. By default this will run for 20 epochs and save their checkpoints to the "vaebert/vae/checkpoints/" folder.
5. Run "vaebert/vae/encode.py -checkpoint vaebert/vae/checkpoints/epoch20.pt" to encode latents vectors for the saved voxelization dataset from the 20th epoch.
6. Run "vaebert/text/bert/train.py" to train the BERT model. By default this will train for 20 epochs and save their checkpoints to the "vaebert/text/bert/checkpoints/" folder.
7. Run "vaebert/text/gru/train.py" to train the GRU model. By default this will train for 1000 epochs and save their checkpoints to the "vaebert/text/gru/checkpoints/" folder.

# Testing
1. Run  "test.py -checkpoints bert_checkpoint gru_checkpoint vae_checkpoint", where bert_checkpoint, gru_checkpoint, and vae_checkpoint are the locations of the checkpoints for the BERT model, GRU model, and VAE model, to get the average chamfer distance for the BERT and GRU model and visualize a test set example from each shape category for both the BERT model and the GRU model.
2. Run "visualize_caption.py -checkpoints bert_checkpoint gru_checkpoint vae_checkpoint", where bert_checkpoint, gru_checkpoint, and vae_checkpoint are the locations of the checkpoints for the BERT model, GRU model, and VAE model, to get an interactive terminal where custom captions can be inputted into both models to get visualizations.