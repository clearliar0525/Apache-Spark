import org.apache.spark.sql.SparkSession 
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

object Assignment3_3 {
  def main(args: Array[String]): Unit = {
  
  val sparkSession = SparkSession.builder.master("local[8]").appName("Assignment3_3").getOrCreate()
  val sc = sparkSession.sparkContext
  val sqlContext= new org.apache.spark.sql.SQLContext(sc)
  import sqlContext.implicits._
  
  val data_train = sc.textFile("files/r1.train")
  val ratings_train = data_train.map(_.split("::") match { case Array(user, item, rate, timestamp) =>Rating(user.toInt, item.toInt, rate.toDouble)
  })
  val rank = 10
  val numIterations = 10
  val model = ALS.train(ratings_train, rank, numIterations, 0.01)
  
  val data_test = sc.textFile("files/r1.test")
  val ratings_test = data_test.map(_.split("::") match { case Array(user, item, rate, timestamp) =>Rating(user.toInt, item.toInt, rate.toDouble)
  })
  val users_feature = ratings_test.map { case Rating(user, product, rate) =>(user, product)}
  val prediction =model.predict(users_feature).map { case Rating(user, product, rate) =>((user, product), rate)}
  val combination = ratings_test.map { case Rating(user, product, rate) =>((user, product), rate)}.join(prediction)
  val MSE = combination.map { case ((user, product), (r1, r2)) => 
  val error = (r1 - r2) 
  error * error }.mean()
  println("Mean Squared Error = " + MSE)
  
  val productsArray=model.productFeatures.map{case (a,b)=>Vectors.dense(b)}
  val usersArray=model.userFeatures.map{case (a,b)=>Vectors.dense(b)}
  val productsmat: RowMatrix = new RowMatrix(productsArray)
  val usersmat: RowMatrix = new RowMatrix(usersArray)
  val productspc: Matrix = productsmat.computePrincipalComponents(2)
  val userspc: Matrix = usersmat.computePrincipalComponents(2)
  val Pprojected: RowMatrix = productsmat.multiply(productspc)
  val Uprojected: RowMatrix = usersmat.multiply(userspc)
  
  val Pprojectednew = Pprojected.rows.map( x => x.toArray.mkString(","))
  val Uprojectednew = Uprojected.rows.map( x => x.toArray.mkString(","))
  Pprojectednew.saveAsTextFile("files/projected1")
  Uprojectednew.saveAsTextFile("files/projected2")
  
  sparkSession.stop()
  }
}
