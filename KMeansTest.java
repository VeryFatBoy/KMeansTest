package org.apache.ignite.examples.ml.math;

import org.apache.ignite.ml.clustering.KMeansLocalClusterer;
import org.apache.ignite.ml.math.EuclideanDistance;
import org.apache.ignite.ml.math.impls.matrix.DenseLocalOnHeapMatrix;
import org.apache.ignite.ml.clustering.KMeansModel;
import org.apache.ignite.ml.math.impls.vector.DenseLocalOnHeapVector;

public class KMeansTest {
    public static void main(String[] args) {
        KMeansLocalClusterer clusterer = new KMeansLocalClusterer(new EuclideanDistance(), 1, 1L);
        double[] v1 = new double[] {1959, 325100};
        double[] v2 = new double[] {1960, 373200};

        DenseLocalOnHeapMatrix points = new DenseLocalOnHeapMatrix(new double[][] {v1, v2});
        KMeansModel mdl = clusterer.cluster(points, 1);
        // mdl.centers();
        // Integer clusterInx = mdl.predict(new DenseLocalOnHeapVector(new double[] {20.0, 30.0}));
    }
}
