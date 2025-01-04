from metaflow import FlowSpec, step, Parameter, resources, conda_base, profile
from sklearn.cluster import KMeans
from analyze_kmeans import top_data_points

@conda_base(python="3.12.7", libraries={"scikit-learn": "1.5.2", "pandas" : "2.2.2", "nltk":"3.9.1"})
class KMeansFlow(FlowSpec):
    num_docs = Parameter('num-docs', help='Number of documents', default=1000)

    @resources(memory=200)
    @step
    def start(self):
        import preprocessing
        docs = preprocessing.load_chat(self.num_docs)
        self.mtx, self.cols = preprocessing.scale_data(docs)
        self.kmeans_params = [4, 5, 6]
        self.next(self.train_kmeans, foreach='kmeans_params')

    @resources(cpu=1, memory=200)
    @step
    def train_kmeans(self):
        self.k = self.input
        with profile('kmeans'):
            model = KMeans(n_clusters=self.k, random_state=42, n_init=10)
            mtx_dense = self.mtx.toarray()
            model.fit(mtx_dense)
        self.clusters = model.predict(mtx_dense)
        self.next(self.analyze)

    @step
    def analyze(self):
        self.top = top_data_points(self.k, self.clusters, self.mtx, self.cols)
        self.next(self.join)

    @step
    def join(self, inputs):
        self.top = {inp.k: inp.top for inp in inputs}
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    KMeansFlow()