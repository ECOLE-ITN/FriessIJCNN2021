# Structured Data Formats and Feature Learning with Procedural Optimization Data


Contains code to harness procedural optimization data by converting it into a structured format and perform experiments for feature learning and analysis.

## Introduction

The code we provide with our repository is based upon our paper [Artificial Neural Networks as Feature Extractors in Continuous Evolutionary Optimization](https://ieeexplore.ieee.org/document/9533915/) (Friess, Tiňo, Xu, Menzel, Sendhoff & Yao, 2021) and can be used to replicate the experiments for feature learning, extraction and analysis on the given synthetic benchmark function set. The code is implemented in Python 3.0 and the required libraries are elaborated in the following up technical requirements part.

 We provide experimental data from in the intermediate steps within our paper in the folders  `model_data`,  `ea_data` and `histogram_data` respectively. The folder `gcn` contains the necessary custom operations for graph convolution and pooling. Where the former is our own custom Keras implementation and the latter is based upon code accompanying the original paper from Defferrard et al. (2016). Different implementations of evolutionary algorithms from the DEAP library can be used for experimentation are provided in `ea_generate` folder.  The scripts `training_\*.py` can be used to generate partition models, `histogram_calculation_\*.py` to obtain structured data formats and `Notebook-\*nb` to experiment with network architectures and analyze their feature extraction capabilities. All them are contained within the main folder.

In the following, we will give a more in-depth description on the technical requirements and how to use our scripts according to the given step-by-step experiments within our paper.

## Technical Requirements

To obtain structured data formats using search space partition methods, we use a k-means implementation from the Scikit-Learn software package. For the construction of Delaunay graphs, the SciPy software package can be used. Further, optimized implementations of the self-organized map and growing neural gas are contained within the NeuPy library. For implementing the neural network architectures we use Keras with a TensorFlow backend.  

All required libraries can be installed by executing `pip install -r requirements.txt` from the main directory via the command line. 


| Library       | Description |
| ------------- |:-------------|
| TensorFlow  | Library for constructing neural network architectures. |
| Keras  | Python interface for the usage of Tensorflow.   |
| DEAP        | Provides different implementations of EAs. |
| SciPy    | Efficient calculations of distances and of Voronoi graphs.  |
| Scikit-Learn       | Implements the k-Means clustering.  |
| NeuPy       | Implements SOM and GNG. |

The following sections elaborate on how to replicate the steps and experiments presented within our paper. 

## 1. Setting up a Search Space Partition Method

The scripts `training_som.py`, `training_kmeans.py` and `training_gng.py` implement the differently elaborated search space partitions. Upon being called from the command line or via an IDE, the script clustering_training_data_generator.py is imported and the generateTrainingData method is called to generate a training data set which exhaustively fills the search space and can subsequently be used for the setup of the aforementioned search space partition methods. Trained partition models are subsequently stored in the model_data folder.

## 2. Data Generation from Synthetic Experiments

Calling from the ea_generate folder e.g. the (𝜇+𝜆)-ES via `m-plus-l_evolution-stategy.py` using the command line or within an IDE, starts experiments with preset experimental parameters contained in the file. Note that we rescale any generated solutions to a search space size of [-30, 30]d such that to ensure a uniform format for the subsequent application of a search space partition methods. Any subsequently generated files are stored within the ea_data folder for further processing.

## 3. Converting Unstructured Raw Data into Structured Data 

To convert the unstructured draw data into a structured data format the scripts `histogram_calculation_som.py` for the self-organized map, int, `histogram_calculation_kmeans.py`  for the k-means algorithm as well as as well as  `histogram_calculation_gng.py` for the growing neural gas can be used. The obtained structured data format from running the scripts is stored in the folder `histogram_data`. Subsequently, it can be used for the training of neural network architectures.

## 4. Replication of Experiments
The Jupyter Notebooks `Notebook-SOM-MLP.ipynb`, `Notebook-SOM-CNN.ipynb` and `Notebook-SOM-GNN.ipynb` contain the necessary code to replicate the previously elaborated experiments on in regards feature extraction and analysis. 

## How to Cite

### Paper Reference
* Friess, S., Tiňo, P., Xu, Z., Menzel, S., Sendhoff, B. and Yao, X., 2021, July. Artificial Neural Networks as Feature Extractors in Continuous Evolutionary Optimization. In 2021 International Joint Conference on Neural Networks (IJCNN) (pp. 1-9). IEEE.

### BibTeX Reference
```
@inproceedings{friess2021artificial,
  title={Artificial Neural Networks as Feature Extractors in Continuous Evolutionary Optimization},
  author={Friess, Stephen and Ti{\v{n}}o, Peter and Xu, Zhao and Menzel, Stefan and Sendhoff, Bernhard and Yao, Xin},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--9},
  year={2021},
  organization={IEEE}
}
```

## Acknowledgements

This research has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement number 766186 (ECOLE). It was also supported by the Program for Guangdong Introducing Innovative and Enterpreneurial Teams (Grant No. 2017ZT07X386), Shenzhen Science and Technology Program (Grant No. KQTD2016112514355531), and the Program for University Key Laboratory of Guangdong Province (Grant No. 2017KSYS008).
