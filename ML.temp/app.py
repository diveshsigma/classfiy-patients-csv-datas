from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

def perform_clustering(data, method="kmeans"):
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # Clustering
    if method == "kmeans":
        model = KMeans(n_clusters=3, random_state=42)
    else:  # Gaussian Mixture for EM
        model = GaussianMixture(n_components=3, random_state=42)

    model.fit(X_scaled)
    labels = model.predict(X_scaled)

    # Calculate metrics
    silhouette = silhouette_score(X_scaled, labels)
    db_index = davies_bouldin_score(X_scaled, labels)

    return labels, silhouette, db_index

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(file)

    # Drop non-numerical columns if any (adjust this to your dataset)
    data = data.select_dtypes(include=['float64', 'int64'])

    # Perform clustering using K-Means (can extend to EM)
    method = request.form.get('method', 'kmeans')
    labels, silhouette, db_index = perform_clustering(data, method=method)

    # Generate a plot for visualization
    fig, ax = plt.subplots()
    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'Clusters using {method.upper()}')

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Render the results
    return render_template(
        'index.html', 
        clusters=labels.tolist(),
        method=method,
        silhouette_score=silhouette,
        db_index=db_index,
        num_clusters=3,
        plot_url='/plot'
    )

@app.route('/plot')
def plot():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
