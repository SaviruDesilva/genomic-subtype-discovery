import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

# PAGE CONFIGURATION 
st.set_page_config(page_title="Genomic Subtype Explorer", page_icon="🧬", layout="wide")

st.title("🧬 Genomic Data Clustering Engine")
st.markdown("""
**Objective:** Identify hidden cancer subtypes using Unsupervised Learning (PCA + Hierarchical Clustering).
""")

# SIDEBAR: CONTROLS 
st.sidebar.header("⚙️ Model Parameters")
uploaded_file = st.sidebar.file_uploader("Upload Cancer Data (CSV)", type=["csv"])
n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, 4)

# MAIN LOGIC
if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    
    #  PREPROCESSING 
    
    df.columns = df.columns.str.strip()
    
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    st.write(f"**Data Dimensions:** {df.shape[0]} patients, {df.shape[1]} genes")
    
    # Show raw data preview
    with st.expander("👀 View Raw Data"):
        st.dataframe(df.head())

    # 2. Scaling
    st.subheader("1️⃣ Dimensionality Reduction (PCA)")
    ss = StandardScaler()
    # Ensure we only select numeric columns to avoid errors
    numeric_df = df.select_dtypes(include=[np.number])
    scale = ss.fit_transform(numeric_df)

    # 3. Apply PCA
    
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(scale)
    
    # Calculate Variance metrics
    variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)[-1]

    # Display Variance Metrics
    col1, col2 = st.columns(2)
    col1.metric("Explained Variance (PC1)", f"{variance_ratio[0]*100:.2f}%")
    col2.metric("Total Variance Retained (PC1+PC2)", f"{cumulative_variance*100:.2f}%")

    # 4. Clustering (Hierarchical)
    st.subheader("2️⃣ Hierarchical Clustering & Validation")
    
    model = AgglomerativeClustering(n_clusters=n_clusters)
    pred = model.fit_predict(x_pca)
    
    # VALIDATION PART 
    sil_score = silhouette_score(x_pca, pred)
    
    # Display Validation Score
    col_v1, col_v2, col_v3 = st.columns(3)
    col_v1.metric("Silhouette Score", f"{sil_score:.3f}")
    
    with col_v2:
        if sil_score > 0.5:
            st.success("✅ Strong Clusters")
        elif sil_score > 0.25:
            st.warning("⚠️ Moderate Structure")
        else:
            st.error("❌ Weak/Overlapping")
    
    col_v3.metric("Clusters Found", n_clusters)

    # 5. Visualization
    st.subheader("3️⃣ Visual Results")
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame(x_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = pred.astype(str) # Convert to string for categorical coloring

    # TAB 1: Scatter Plot
    tab1, tab2 = st.tabs(["🔴 Cluster Scatter Plot", "🌳 Dendrogram"])
    
    with tab1:
        st.markdown("Patients grouped by genetic similarity.")
        st.scatter_chart(pca_df, x='PC1', y='PC2', color='Cluster', size=20)

    # TAB 2: Dendrogram
    with tab2:
        st.markdown("Tree diagram showing how patients were merged.")
        with st.spinner("Calculating Dendrogram..."):
            fig_dendro, ax_dendro = plt.subplots(figsize=(10, 5))
            z = linkage(x_pca, method='ward')
            dendrogram(z, ax=ax_dendro)
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Patients')
            plt.ylabel('Euclidean Distance')
            st.pyplot(fig_dendro)

else:
    st.info("👈 Please upload your `data.csv` file to start.")
    st.markdown("""
    **How to use:**
    1. Click 'Browse files' in the sidebar.
    2. Upload your `cancer/data.csv`.
    3. Adjust the Number of Clusters slider to see how the score changes.
    """)
