import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}

object Assignment3_2 {

  def main(args: Array[String]): Unit = {

  val sparkSession = SparkSession.builder.master("local[8]").appName("Assignment3_2").getOrCreate()
  val sc = sparkSession.sparkContext
  val sqlContext= new org.apache.spark.sql.SQLContext(sc)
  import sqlContext.implicits._

  val data_string = sc.textFile("files/data.csv")
  val header_r = data_string.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
  val col_r = header_r.map(line => line.split(",").drop(1).map(_.toDouble))
  val rdd_d1 = col_r.map(s => Vectors.dense(s.take(178).map(_.toDouble))).cache()

  // Cluster the data into two classes using KMeans
  val numClusters = 5
  val numIterations = 100
  val model = KMeans.train(rdd_d1, numClusters, numIterations)

  // print the clusters
  println("Cluster Centers: ")
  model.clusterCenters.foreach(println)

  // calculate the size
  val size = model.predict(rdd_d1).map(s=>(s,1)).reduceByKey((a,b)=>a+b)
  val size_part = size.take(5)

  // sort the size
  val sortedBysize = size_part.sortBy{case(cluster,size) => size}
  val max = sortedBysize(4)
  val max_id = max._1
  val max_size = max._2
  val min = sortedBysize(0)
  val min_id = min._1
  val min_size = min._2

  println("The largest cluster is:")
  println(model.clusterCenters(max_id))
  println("the size of the largest cluster is " + max_size)
  println("The smallest cluster is:")
  println(model.clusterCenters(min_id))
  println("the size of the smallest cluster is " + min_size)

  // calculate the distance
  def euclidean(x: Vector, y: Vector) = {
    math.sqrt(x.toArray.zip(y.toArray).
      map(p => p._1 - p._2).map(d => d*d).sum)
  }
  println("the distance between the largest cluster and the smallest cluster is " + euclidean(model.clusterCenters(max_id),model.clusterCenters(min_id)))

  sparkSession.stop()
  }
}
