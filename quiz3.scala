import org.apache.spark.mllib.clustering.{KMeans}
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.distributed.RowMatrix

object quiz3_solution{

  def main(args: Array[String]): Unit = {

    val sparkSession = {SparkSession
      .builder()
      .master("local")
      .appName("Quiz3")
      .getOrCreate()}

    val sc = sparkSession.sparkContext
    val data = sc.textFile("semeion.data")

/*  Task 1: We get rid of the last 10 columns taking only the first 256  */

    val parsedData = data.map(s => Vectors.dense(s.split(' ').take(256).map(_.toDouble))).cache()

/*  Task 2: We cluster the data into 10 means using KMeans method  */
 
    val numClusters = 10
    val numIterations = 250
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

/*  We calculate the metric WSSSE and print it */
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

/*  Task 3: We find the number of components which retain at least 80% of variance  */
 
    val mat: RowMatrix = new RowMatrix(parsedData)
    val (princ_comp, variance) = mat.computePrincipalComponentsAndExplainedVariance(256)
    val variance_retained = variance.toArray
    val relative_percentage = 0.80
    var P = 1

    while(variance_retained.take(P).sum < relative_percentage)
      {
          P=P+1
      }

    println("We retain at least 80% of the variance using "+P+" Principal Components")

/* We compute the SVD decomposition in order to find the top 5 eigenvalues */
    val SingularValueDecomposition(u, s, v) = mat.computeSVD(P)

/* We take only the top 5 eigenvalues  */
    val eigenval_top5 = s.toArray.take(5)

    println("These are the top 5 eigenvalues:")
    eigenval_top5.foreach(println)

/*  Task 4: We use the PCA features determined in Task 3, and project the
    rows to the linear space spanned by the P principal components */
    
    val new_princ_comp = mat.computePrincipalComponents(P)
    val projected: RowMatrix = mat.multiply(new_princ_comp)
    val projected_rows = projected.rows

    val newclusters = KMeans.train(projected_rows, numClusters, numIterations)
    val WSSSE_with_P = newclusters.computeCost(projected_rows)

    println("Within Set Sum of Squared Errors = "+WSSSE_with_P+" using "+P+" Principal Components")

  }

}
