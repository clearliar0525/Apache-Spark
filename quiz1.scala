import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.util.MLUtils

// imports for the text document pipeline
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row

// define main method (scala entry point)
object quiz {
  def main(args: Array[String]): Unit = {

// Create Spark session
  val sparkSession = SparkSession.builder.master("local[1]").appName("Spark dataframes and datasets").getOrCreate()

//Read a plain text file
import sparkSession.implicits._

// class converts from dataframe to dataset output
  val dataset = sparkSession.read.text("files/AnimalFarmChap1.txt").as[String]
  val words = dataset.flatMap(value => value.split("\\W")).filter(x=> x != "")
  val wordsleast6=words.filter(x=> x.length>=6)


// Configure an ML pipeline, which consists of two stages: tokenizer, and stopWordsRemover.
//val tokenizer = new RegexTokenizer().setInputCol("value").setOutputCol("words").setPattern("\\W")
  val tokenizer = new RegexTokenizer().setInputCol("value").setOutputCol("words")
  val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
  val pipeline = new Pipeline().setStages(Array(tokenizer,remover))

// fit the pipline and transform  
  val model1 = pipeline.fit(words)
  val result = model1.transform(words)

// Show Counts (15 most frequent)
  val showcounts=result.withColumn("Top 15 Words", explode(col("filtered")))
  showcounts.groupBy("Top 15 Words").agg(count("*") as "Top Frequency").orderBy(desc("Top Frequency")).show(15, false)

// Show Counts (15 least frequent with at least 6 characters )
  val model2=pipeline.fit(wordsleast6)
  val result2=model2.transform(wordsleast6)
  val showcountleast6=result2.groupBy($"filtered").count().orderBy($"count").show(15)
  

  sparkSession.stop()

 }
}







