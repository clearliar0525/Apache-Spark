import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}

object Assignment3_1 {

  def main(args: Array[String]): Unit = {

  val sparkSession = SparkSession.builder.master("local[8]").appName("Assignment3_1").getOrCreate()
  val sc = sparkSession.sparkContext
  val sqlContext= new org.apache.spark.sql.SQLContext(sc)
  import sqlContext.implicits._

  val data_string = sc.textFile("files/NIPS_1987-2015.csv",100)
  val header_r = data_string.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
  
  val col_r = header_r.map(line => line.split(",").drop(1).map(_.toDouble))
  val rdd = col_r.take(11462)
  //def transpose( m: Array[Array[Double]] ): Array[Array[Double]] = {( for { c <- m(0).indices } yield { m.map( _(c)) } ).toArray }
  //val t_array = transpose(rdd)
  val rdd1 = rdd.flatMap( x => x)
  val t_array = new DenseMatrix(5811,11462,rdd1)
  def matrixToRDD(m: Matrix): RDD[Vector] = {
    val columns = m.toArray.grouped(m.numRows)
    val rows = columns.toSeq.transpose // Skip this if you want a column-major RDD.
    val vectors = rows.map(row => new DenseVector(row.toArray))
    sc.parallelize(vectors)
  }
  val rows = matrixToRDD(t_array)
  //val byColumnAndRow = col_r.zipWithIndex.flatMap {
    //case (row, rowIndex) => row.zipWithIndex.map {
      //case (number, columnIndex) => columnIndex -> (rowIndex, number)
    //}
  //}
  // Build up the transposed matrix. Group and sort by column index first.
  //val byColumn = byColumnAndRow.groupByKey.sortByKey().values
  // Then sort by row index.
  //val transposed = byColumn.map {
    //indexedRow => indexedRow.toSeq.sortBy(_._1).map(_._2)
  //}
  val mat = new RowMatrix(rows)
  // Compute the top 2 principal components.
  // Principal components are stored in a local dense matrix.
  val pc: Matrix = mat.computePrincipalComponents(2)
  // Project the rows to the linear space spanned by the top 4 principal components.
  val projected: RowMatrix = mat.multiply(pc)
  val projected_rdd = projected.rows.map( x => x.toArray.mkString(","))
  projected_rdd.saveAsTextFile("files/projected_real")

  val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(2, computeU = true)
  val s: Vector = svd.s
  val eigenvalues = s.toArray.map(x=>x*x)
  println(eigenvalues(0),eigenvalues(1))
  
  println(pc.toString(10,Int.MaxValue))

  println(mat.computePrincipalComponentsAndExplainedVariance(2))
  


  sparkSession.stop()
  }
}

  