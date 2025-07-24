Graph-NIDS: Detecting Network Intrusions via HGT-based Edge Classification on Network Traffic Graphs
<div align="center">

</div>

This repository contains the official implementation for Graph-NIDS, a novel Network Intrusion Detection System (NIDS). This project leverages the power of Graph Neural Networks (GNNs), specifically the Heterogeneous Graph Transformer (HGT), to perform edge classification on network traffic graphs, effectively identifying malicious activities.

📜 Overview
The core objective of this research is to re-frame the network intrusion detection problem from a graph-based perspective. By modeling network traffic data from the UNSW-NB15 dataset as a large, heterogeneous graph, we can effectively capture the complex relationships between IP addresses, ports, and protocols. An HGT model is then trained to perform edge classification, discerning whether a given network flow (an edge between two IP address nodes) is benign ('Normal') or an 'Intrusion'.

⚙️ Workflow
The end-to-end workflow of the system, from data preprocessing to model evaluation, is illustrated below:

✨ Key Features
Heterogeneous Graph Construction: Builds a rich graph structure with multiple node types (ip, port, proto) and edge types to represent diverse network interactions.

Heterogeneous Graph Transformer (HGT): Employs the state-of-the-art HGT architecture, which is specifically designed to handle the complexity and heterogeneity of the constructed graphs.

Edge Classification Task: Frames intrusion detection as an edge classification problem, classifying network flows between IP nodes as 'Normal' or 'Intrusion'.

Comprehensive Data Processing: Includes robust scripts for sampling, stratification, and preprocessing of the large-scale UNSW-NB15 dataset.

🔬 Methodology
The methodology is broken down into four primary stages: data preparation, graph construction, model training, and evaluation.

Preprocessing: Raw network flow records from the UNSW-NB15 dataset undergo a rigorous preprocessing pipeline. This includes encoding categorical features (proto, service, state), handling special values, and normalizing numerical features using StandardScaler fitted exclusively on the training data to prevent data leakage.

Graph Construction: We construct heterogeneous graphs for the training, validation, and test sets.

Node Types: ip, port, proto

Edge Types: Edges represent relationships such as ('ip', 'flows_to', 'ip'), ('ip', 'uses_port', 'port'), and ('port', 'uses_proto', 'proto'), along with their inverses to facilitate bidirectional message passing.

The preprocessed flow features are assigned as attributes to the ('ip', 'flows_to', 'ip') edges, which are the primary targets for classification.

A visualization of a subgraph is shown below:

HGT Model Training: An HGT model is trained for 10 epochs. The model learns to classify the ('ip', 'flows_to', 'ip') edges by aggregating and propagating information across the different node and edge types in the graph. Performance is monitored on the validation set at the end of each epoch.

Evaluation: The trained model's performance is assessed on the unseen test graph. A statistically optimal probability threshold (0.4724) was determined on the validation set to maximize the F1-score for the minority (Intrusion) class before final evaluation.

The entire implementation is detailed in the HGT_Training_Code.ipynb notebook.

📊 Results
The model demonstrates strong performance in identifying intrusive network traffic, achieving high recall and a robust F1-score for the intrusion class.

# Final Training Performance (Epoch 10)

| **Metric**           | **Training Set** | **Validation Set** |
|----------------------|------------------|---------------------|
| **Loss**             | 0.6016           | 0.4200              |
| **Accuracy**         | 0.7437           | 0.9880              |
| **Intrusion F1-Score** | 0.1361           | 0.8308              |
| **Intrusion Recall** | 0.6361           | 0.9299              |


<br>

# Final Test Metrics (with Optimal Threshold: 0.4724)

| **Metric**                   | **Score** |
|-----------------------------|-----------|
| **Test Accuracy**           | 0.9883    |
| **Test F1 (Intrusion)**     | 0.8405    |
| **Test Recall (Intrusion)** | 0.9740    |
| **Test Precision (Intrusion)** | 0.7392 |


Performance Visualization
The plot below illustrates the training and validation metrics over 10 epochs.

🗂️ Repository Structure
Graph_NIDS/
├── Datasets.zip              # Zip archive: train_set.csv, val_set.csv, test_set.csv
├── HGT_Training_Code.ipynb   # Main notebook for graph building, HGT training, and evaluation
├── Graph_Dataset_Code.ipynb  # Notebook detailing the dataset sampling and splitting process
├── NUSW-NB15_features.csv    # Description of features in the UNSW-NB15 dataset
└── UNSW-NB15_LIST_EVENTS.csv # List of attack categories in the dataset

🚀 Getting Started
Follow these instructions to set up the project and reproduce the results.

1. Clone the Repository
git clone https://github.com/bharateesha2004/Graph_NIDS.git
cd Graph_NIDS

2. Set Up a Virtual Environment (Recommended)
python -m venv graph_env
# On Linux/macOS
source graph_env/bin/activate
# On Windows
graph_env\Scripts\activate

3. Install Dependencies
Install PyTorch first, ensuring it matches your system's CUDA version if applicable. Then, install the remaining packages.

# Refer to the official PyTorch website for installation commands: https://pytorch.org/
# Example for CPU-only:
# pip install torch torchvision torchaudio

# Install other required libraries
pip install pandas numpy scikit-learn torch-geometric tqdm matplotlib networkx

Note: torch-geometric installation is sensitive to the torch version. Please consult the PyG installation guide.

4. Prepare Data
Unzip the Datasets.zip archive to access the training, validation, and test CSV files.

unzip Datasets.zip

5. Run the Project
(Optional) Open Graph_Dataset_Code.ipynb to review the data sampling and splitting logic.

Open HGT_Training_Code.ipynb.

Important: Update the file paths in the initial cells (e.g., DATA_PATH, TRAIN_FILE) to point to the location of the unzipped datasets.

Execute the cells sequentially to load data, build the graph, train the HGT model, and evaluate its performance. Training is significantly faster on a CUDA-enabled GPU.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

🙏 Acknowledgements
This project utilizes the UNSW-NB15 dataset. We extend our gratitude to the researchers at the University of New South Wales (UNSW) for creating and providing this valuable resource.

Dataset Homepage: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Citation: If you use this dataset in your work, please refer to the homepage for the recommended papers to cite.

📝 License
This project is licensed under the MIT License. See the LICENSE file for more details.
