# 🧬 Genomic Subtype Explorer

An unsupervised machine learning dashboard built with **Streamlit** to identify hidden cancer subtypes from genomic data.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://genomic-subtype-discovery-h3br3jd8sxb5w72gavcbq7.streamlit.app/)

## 📖 Project Overview
This tool performs **Gene Expression Clustering** to group patients based on molecular similarities. It uses **Principal Component Analysis (PCA)** to reduce high-dimensional genetic data into interpretable visualizations and **Hierarchical Clustering** to find natural subgroups (subtypes).

### Key Features
* **📉 Dimensionality Reduction:** Compresses thousands of genes into 2 Principal Components (PC1 & PC2) using PCA.
* **🌳 Hierarchical Clustering:** Uses Agglomerative Clustering to group patients without needing pre-labeled data.
* **✅ Cluster Validation:** Automatically calculates the **Silhouette Score** to rate the quality of the clusters (Strong vs. Weak).
* **📊 Visualization Suite:**
    * **Interactive Scatter Plot:** Explore patient clusters in 2D space.
    * **Dendrogram:** A tree diagram showing the hierarchical relationships between samples.

## 💻 Tech Stack
* **Python 3.x**
* **Streamlit** (User Interface)
* **Scikit-Learn** (PCA & Agglomerative Clustering)
* **SciPy** (Dendrogram & Linkage Calculation)
* **Pandas & NumPy** (Data Processing)
* **Matplotlib** (Static Plotting)

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SaviruDesilva/genomic-subtype-discovery.git](https://github.com/SaviruDesilva/genomic-subtype-discovery.git)
    cd genomic-subtype-discovery
    ```
    *(Note: If your repository name is different, update the link above)*

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run genomic.py
    ```

## 📂 Project Structure
```text
├── data.csv             # Example genomic dataset
├── genomic.py           # Main application code
├── requirements.txt     # List of Python libraries
└── README.md            # Project documentation
