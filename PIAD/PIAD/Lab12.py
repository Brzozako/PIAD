import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.stats import mode
from sklearn.metrics import jaccard_score
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import cv2

iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Klasteryzacja za pomocą różnych metod
agglomerative_clustering_nn = AgglomerativeClustering(linkage='single')
Y_pred_nn = agglomerative_clustering_nn.fit_predict(X)

agglomerative_clustering_avg = AgglomerativeClustering(linkage='average')
Y_pred_avg = agglomerative_clustering_avg.fit_predict(X)

agglomerative_clustering_complete = AgglomerativeClustering(linkage='complete')
Y_pred_complete = agglomerative_clustering_complete.fit_predict(X)

agglomerative_clustering_ward = AgglomerativeClustering(linkage='ward')
Y_pred_ward = agglomerative_clustering_ward.fit_predict(X)

# Funkcja do wizualizacji wyników
def plot_clusters(X, Y_pred, title):
    plt.scatter(X[:, 0], X[:, 1], c=Y_pred)
    plt.title(title)
    plt.show()

plot_clusters(X, Y_pred_nn, 'Metoda najbliższego sąsiedztwa')
plot_clusters(X, Y_pred_avg, 'Metoda średnich połączeń')
plot_clusters(X, Y_pred_complete, 'Metoda najdalszych połączeń')
plot_clusters(X, Y_pred_ward, 'Metoda Warda')

def find_perm(clusters, Y_real, Y_pred):
    perm = []
    for i in range(clusters):
        idx = (Y_pred == i)
        print(f'Cluster {i}: idx={idx}')
        if np.sum(idx) == 0:
            continue  # Pomijamy puste klastry
        try:
            new_label = mode(Y_real[idx]).mode[0]
            print(f'Cluster {i}: new_label={new_label}')
            perm.append(new_label)
        except IndexError:
            print(f'Cluster {i}: IndexError')
            perm.append(-1)  # Wartość domyślna, jeśli wystąpi problem
    print(f'Final perm={perm}')
    return [perm[label] if label < len(perm) else -1 for label in Y_pred]



# Dopasowanie wyników klasteryzacji do rzeczywistych klas
Y_pred_mapped = find_perm(3, Y, Y_pred_ward)

jaccard = jaccard_score(Y, Y_pred_mapped, average='macro')
print(f'Indeks Jaccarda: {jaccard}')

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Funkcja do wizualizacji z powłoką wypukłą
def plot_with_hull(X, Y, title):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    for label in np.unique(Y):
        hull = ConvexHull(X[Y == label])
        for simplex in hull.simplices:
            plt.plot(X[Y == label][simplex, 0], X[Y == label][simplex, 1], 'k-')
    plt.title(title)
    plt.show()

# Wizualizacja danych z rzeczywistymi klasami
plot_with_hull(X_reduced, Y, 'Rzeczywiste klasy')

# Wizualizacja danych z klastrami uzyskanymi z klasteryzacji
plot_with_hull(X_reduced, Y_pred_ward, 'Wyniki klasteryzacji (Ward)')

pca_3d = PCA(n_components=3)
X_reduced_3d = pca_3d.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_reduced_3d[:, 0], X_reduced_3d[:, 1], X_reduced_3d[:, 2], c=Y)
plt.title('Rzeczywiste klasy w 3D')
plt.show()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_reduced_3d[:, 0], X_reduced_3d[:, 1], X_reduced_3d[:, 2], c=Y_pred_ward)
plt.title('Wyniki klasteryzacji (Ward) w 3D')
plt.show()

linked = linkage(X, 'ward')
dendrogram(linked)
plt.title('Dendrogram')
plt.show()

# K-means
kmeans = KMeans(n_clusters=3)
Y_pred_kmeans = kmeans.fit_predict(X)

# GMM
gmm = GaussianMixture(n_components=3)
Y_pred_gmm = gmm.fit_predict(X)

plot_with_hull(X_reduced, Y_pred_kmeans, 'Wyniki klasteryzacji (k-means)')
plot_with_hull(X_reduced, Y_pred_gmm, 'Wyniki klasteryzacji (GMM)')

# Wczytaj obraz (należy podać ścieżkę do pliku obrazu)
img = cv2.imread('obraz.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Przekształcenie obrazu do macierzy pikseli
pixels = img.reshape(-1, 3)

# Funkcja do kwantyzacji obrazu
def quantize_image(method, clusters, pixels, img_shape):
    method.fit(pixels)
    centers = method.cluster_centers_
    labels = method.predict(pixels)
    quantized_img = centers[labels].reshape(img_shape).astype(int)
    return quantized_img, labels

# Przykład dla k-means
kmeans = KMeans(n_clusters=5)
quantized_img_kmeans, labels = quantize_image(kmeans, 5, pixels, img.shape)

# Pokaż zkwantyzowany obraz
plt.imshow(quantized_img_kmeans)
plt.title('Obraz po kwantyzacji (k-means, 5 klastrów)')
plt.show()

# MSE
mse = np.mean((img - quantized_img_kmeans) ** 2)
print(f'Błąd średniokwadratowy: {mse}')

# Wizualizacja obrazu przed i po kwantyzacji
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Oryginalny obraz')
plt.subplot(1, 2, 2)
plt.imshow(quantized_img_kmeans)
plt.title('Obraz po kwantyzacji (k-means)')
plt.show()

# Permutacja pikseli
permuted_indices = np.random.permutation(len(pixels))
permuted_pixels = pixels[permuted_indices]

# Kwantyzacja permutowanych pikseli
quantized_img_permuted, permuted_labels = quantize_image(kmeans, 5, permuted_pixels, img.shape)

# Odwrócenie permutacji
inverse_permutation = np.argsort(permuted_indices)
unpermuted_img = quantized_img_permuted.reshape(-1, 3)[inverse_permutation].reshape(img.shape)

# Pokaż obraz po odwróceniu permutacji
plt.imshow(unpermuted_img)
plt.title('Obraz po odwróceniu permutacji')
plt.show()