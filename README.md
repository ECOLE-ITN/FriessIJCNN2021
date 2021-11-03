# Structured Data Formats and Feature Learning with Procedural Optimization Data


Contains code to harness procedural optimization data by converting it into a structured format and perform experiments for feature learning and analysis.

## Introduction


The code we provide with our report is based upon our work [Artificial Neural Networks as Feature Extractors in Continuous Evolutionary Optimization](https://ieeexplore.ieee.org/document/9533915/) (Friess, Tiňo, Xu, Menzel, Sendhoff & Yao) and as elaborated in Section 2.1.2 can be used to replicate the results for feature learning, extraction and analysis on the given synthetic benchmark function set. The tool is implemented with Python 3.0 and the required packages are elaborated in the aforementioned technical requirements part of Section 2.1.1 . 

We provide in summary within our code mainly two different functionalities:

* The first one being methods to generate training data for a given search space volume and subsequently setup the different search space partition methods.
* And the second one being methods to replicate the feature learning, extraction and analysis studies on the synthetic benchmark function set as contained within our paper.

The software tool is organized as follows: 

1.	Selection / Setup of a Search Partition Method 
2.	Data Generation from Synthetic Experiments
3.	Conversion of Unstructured Raw Data into a Structured Format
4.	Loading and Preparation of the Structured Data for Training
5.	Training of a Neural Network Architecture
6.	Subsequent Feature Extraction and Analysis 

 We provide experimental data from step 1), 2) and 3) in the folders  `model_data`,  `ea_data` and `histogram_data` respectively. So in principle, depending of one’s preferences, one can skip any step between 1) – 3) without requiring the execution of the previous ones. The folder gcn contains the necessary custom operations for graph convolution and pooling. Where the former is a custom Keras implementation and the latter is based upon MIT licensed code accompanying the original paper from Defferrard et al. [24]. Different implementations of evolutionary algorithms from the DEAP library licensed under LGPL-3.0 that can be used for experimentation are provided in the ea_generate folder.  Otherwise, the scripts `training_\*.py` can be used to generate partition models, `histogram_calculation_\*.py` to obtain structured data formats and `Notebook-\*
nb` to experiment with network architectures and analyze their feature extraction capabilities are contained within the main folder.

In the following, we will give a more in-depth description on how to use the scripts according to the previously elaborated step-by-step description.

## Technical Requirements

To obtain structured data formats (c.f. Figure 2) using the previously search space partition methods, a variety of existing techniques can be used. As the k-means algorithm is usually fairly well known (e.g. [18]), we used a standard library implementation based upon the Scikit-Learn software package [19]. Constructing Delaunay graphs of the retrieved partitions is less trivial. However, the SciPy software package can be used for this purpose [20].  At last, optimized implementations of the self-organized map and growing neural gas are contained within the NeuPy library [21]. Implementations of the bespoken library methods are well supplied within the code accompanying our report. 
`pip install -r requirements.txt`

| Library       | Description |
| ------------- |:-------------|
| Tensorflow  | Library for constructing neural network architectures. |
| Keras  | Python interface for the usage of Tensorflow.   |
| DEAP        | Provides different implementations of EAs. |
| SciPy    | Efficient calculations of distances and of Voronoi graphs.  |
| Scikit-Learn       | Implements the k-Means clustering.  |
| NeuPy       | Implements SOM and GNG. |

## 1. Setting up a Search Space Partition Method

The scripts `training_som.py`, `training_kmeans.py` and `training_gng.py` implement the differently elaborated search space partitions. Upon being called from the command line or via an IDE, the script clustering_training_data_generator.py is imported and the generateTrainingData method is called to generate a training data set which exhaustively fills the search space and can subsequently be used for the setup of the aforementioned search space partition methods. Trained partition models are subsequently stored in the model_data folder.

## 2. Data Generation from Synthetic Experiments

Calling from the ea_generate folder e.g. the (𝜇+𝜆)-ES via `m-plus-l_evolution-stategy.py` using the command line or within an IDE, starts experiments with preset experimental parameters contained in the file. Note that we rescale any generated solutions to a search space size of [-30, 30]d such that to ensure a uniform format for the subsequent application of a search space partition methods. Any subsequently generated files are stored within the ea_data folder for further processing.

## 3. Converting Unstructured Raw Data into Structured Data 

To convert the unstructured draw data into a structured data format the scripts histogram_calculation_som.py for the self-organized map, int, histogram_calculation_kmeans.py for the k-means algorithm as well as as well as  histogram_calculation_gng.py for the growing neural gas can be used. The obtained structured data format from running the scripts is stored in the folder histogram_data. Subsequently, it can be used for the training of neural network architectures.

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


