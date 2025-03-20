# AI Song Lyrics Generator

This project generates song lyrics using an LSTM (Long Short-Term Memory) neural network. It is trained on a dataset of lyrics from various artists, and users can select an artist or use the entire dataset to generate new lyrics. The model allows for customization of the generation process through parameters like temperature and sequence length.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Acknowledgements](#acknowledgements)

## Overview
This project uses TensorFlow and Keras to build and train an LSTM model for text generation. The model is trained on a dataset of song lyrics, and users can generate new lyrics by selecting an artist or using the entire dataset. The generated text can be controlled using a "temperature" parameter, which adjusts the randomness of the output.

## Features
- **Artist Selection**: Choose from a list of artists or use the entire dataset.
- **Customizable Generation**: Control the randomness of the generated text using temperature.
- **LSTM Model**: Uses a deep learning model to learn patterns in the lyrics and generate new text.
- **User-Friendly Interface**: Simple command-line interface for selecting options and generating lyrics.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ra-Verse/AI-Song-lyrics-Generation/tree/main
   cd lyrics-generation-lstm
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have TensorFlow installed:
   ```bash
   pip install tensorflow
   ```

## Usage
1. Prepare the dataset by placing your lyrics file in the project directory.
2. Run the script to train the model and generate lyrics:
   ```bash
   python train.py
   ```
3. Use the generated model to produce new lyrics:
   ```bash
   python generate.py
   ```
4. Adjust parameters like temperature to modify randomness:
   ```bash
   python generate.py --temperature 0.8
   ```

## Dataset
The dataset consists of song lyrics from various artists. You can modify the dataset by adding or removing text files containing lyrics.

## Acknowledgements
This project was originally a **guided project for poetry generation**, but I **modified it extensively** to improve its **functionality, performance, and usability**.

## License
This project is open-source and available under the MIT License.

