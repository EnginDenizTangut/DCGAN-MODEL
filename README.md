# Pokémon GAN - Image Generation

This project uses a Generative Adversarial Network (GAN) to generate Pokémon images based on real sprites from the Pokémon API. The goal of this project is to train a GAN model to generate realistic Pokémon images.

## Objective

The primary objective is to train a GAN model that can generate Pokémon images. The model is trained using images of Pokémon obtained from the [Pokémon API](https://pokeapi.co/). The generator network learns to generate realistic Pokémon images, and the discriminator network learns to distinguish between real and generated images.

### Key Features:
- **Generator**: Generates Pokémon images from random noise.
- **Discriminator**: Classifies images as real or fake based on the generator's output.
- **Image Dataset**: Pokémon images are fetched from the Pokémon API and saved locally.
- **Training**: The GAN is trained over multiple epochs to refine the generator's ability to produce realistic images.

## How It Works

1. **Image Collection**: The Pokémon sprites are fetched from the Pokémon API using their IDs (from 1 to 649). These images are stored in the `./data/pokemon/poke` directory.
   
2. **GAN Architecture**: The GAN consists of two main components:
   - **Generator**: Takes random noise as input and outputs a Pokémon image.
   - **Discriminator**: Takes an image as input (real or fake) and outputs a value between 0 and 1, where 1 indicates a real image and 0 indicates a fake image.

3. **Training**: The model is trained over several epochs. In each epoch, the generator creates fake Pokémon images, and the discriminator classifies them. The discriminator is trained to improve its ability to classify real vs fake, while the generator is trained to improve its ability to generate realistic images.

4. **Output**: After each epoch, the model saves generated images in the `outputs3` folder for visual inspection.

## Requirements

To run this project, you will need the following libraries:
- Python 3.x
- PyTorch
- torchvision
- PIL
- tqdm
- requests

You can install the necessary libraries using `pip`:

```bash
pip install torch torchvision tqdm Pillow requests
```
## Usage
1.**Fetch Pokémon Sprites**: The first step is to fetch Pokémon sprite images using the Pokémon API. This is done automatically when you run the script. The images will be saved in the

2.**Train The DCGAN Model**:
```bash
python train.py
```
- This will start the training process. The script will:
- Download Pokémon images.
- Train the GAN model for 1000 epochs (you can modify this number).

Save generated images after each epoch in the outputs3 folder.

3.**Evaluate the model**
Once the model has trained for the specified number of epochs, you can view the generated images in the outputs3 folder. These images show the output of the generator at different stages of training.
## Hyperparameters:
The model is trained with the following hyperparameters:
- Batch size: 128
- Image size: 64x64
- Z dimension (latent space): 100
- Learning rate: 0.0002
- Epochs: 1000 (you can modify this based on your needs)

## Training Details
- During each training step:
- The discriminator (D) is trained to correctly classify real and fake images.
- The generator (G) is trained to generate fake images that the discriminator classifies as real.
- The loss for both the generator and the discriminator is calculated using the binary cross-entropy loss function (BCELoss).

## Folder Structure
- ./data/pokemon/poke: Directory where Pokémon images are saved.
- outputs3: Directory where generated images are saved after each epoch.
