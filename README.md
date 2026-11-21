\# Chinese Handwritten Character Recognition



DIT5411 Machine Learning Assignment  

Student: Liu Sung Pan

Student ID: 250406298



\## Introduction



This project uses TensorFlow to recognise 13,065 Chinese handwritten

characters. I used Google Colab with a T4 GPU for development and training.



\## Dataset



I first downloaded the common-character dataset, but it contained only

4,803 classes. I then used the full dataset, which contains 684,677 images

across 13,065 classes.



Dataset sources:

\- https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset

\- https://github.com/chenkenanalytic/handwritting\_data\_all



For each class, I selected the first 40 available images by sample number

for the training pool and reserved the remaining images for testing.

43 classes had fewer than 40 images.



I also reserved the last four images from each class in the training

pool for validation.



| Split | Images |

|---|---:|

| Training, excluding validation | 469,867 |

| Validation | 52,260 |

| Test | 162,550 |



\## Model and Training



Images are converted to grayscale, resized to 50 × 50, and normalised

to values between 0 and 1.



The CNN has three convolutional blocks with 32, 64, and 128 filters.

Each block uses batch normalisation, ReLU, and max pooling.

These are followed by a 256-unit dense layer, dropout, and a softmax

output layer with 13,065 units.



I used Adam with a learning rate of 0.001 and a batch size of 128.



For augmentation, I used random rotation, translation, and zoom.

Repeated sampling provides 200 inputs per class per full training pass.

Augmentation is applied during training, rather than saved as separate

image files.



\## Experiments and Results



The first training run had very low validation accuracy.

When I checked the training batches, the first 3,840 inputs contained

only 118 different labels. The shuffle buffer was not mixing the

classes well after repeated sampling.



I changed the pipeline to shuffle all training originals before

repeating them, followed by another shuffle. The same check then

contained 2,401 different labels.



A separately initialised model was trained with the corrected pipeline

for a 1,000-batch trial and one additional full training pass.



| Training pipeline | Validation accuracy | Validation top-5 accuracy | Validation loss |

|---|---:|---:|---:|

| Original | 0.0115% | 0.0670% | 12.9848 |

| Corrected | 25.2622% | 53.0501% | 3.5282 |



These results are from the validation set, not the test set.



\## Running the Notebook



Open `DIT5411\_Liu\_Sung\_Pan.ipynb` in Google Colab.



The notebook includes dataset download, TFRecord conversion, training,

and diagnostic experiments. Check the file paths and select the required

cells before running it; not all experiment cells need to be rerun.



The saved outputs show the previous experiments. They do not restore

the dataset files or model weights after a Colab runtime reset.



\## References



The dataset was developed by AI . FREE Team based on

STUST EECS Chinese MNIST and is distributed under CC BY-NC-SA 4.0.



\- https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset

\- https://github.com/chenkenanalytic/handwritting\_data\_all

\- https://www.tensorflow.org/

