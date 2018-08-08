import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.GeneralizedLinearRegression

import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.VectorAssembler

object Assignment2_2 {
  
  def main(args: Array[String]): Unit = {

  val sparkSession = SparkSession.builder.master("local[5]").appName("Assignment2_2").getOrCreate()
  val sc=sparkSession.sparkContext
	val sqlContext= new org.apache.spark.sql.SQLContext(sc)
	import sqlContext.implicits._

  val data = sparkSession.read.format("csv").option("header", "true").option("delimiter", ",").option("inferSchema", "true").load("files/train_set.csv").na.drop()
  val featureCol = data.columns
  var indexers: Array[StringIndexer] = Array()
  for (colName <- featureCol) {
      val index = new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(colName + "_newindex")
      indexers = indexers :+ index
    }
  val pipeline1 = new Pipeline().setStages(indexers)
  val newdata = pipeline1.fit(data).transform(data)
  val featuresArray = Array("Row_ID_newindex","Household_ID_newindex","Vehicle_newindex","Calendar_Year_newindex","Model_Year_newindex","Blind_Make_newindex","Blind_Model_newindex","Blind_Submodel_newindex","Cat1_newindex","Cat2_newindex","Cat3_newindex","Cat4_newindex","Cat5_newindex","Cat6_newindex","Cat7_newindex","Cat8_newindex","Cat9_newindex","Cat10_newindex","Cat11_newindex","Cat12_newindex","OrdCat_newindex","Var1_newindex","Var2_newindex","Var3_newindex","Var4_newindex","Var5_newindex","Var6_newindex","Var7_newindex","Var8_newindex","NVCat_newindex","NVVar1_newindex","NVVar2_newindex","NVVar3_newindex","NVVar4_newindex","Claim_Amount_newindex")
  val labelsArray = "Claim_Amount_newindex"
  val index = new StringIndexer().setInputCol(labelsArray).setOutputCol("label")
  val feature = new VectorAssembler().setInputCols(featuresArray).setOutputCol("features")
  val pipeline2 = new Pipeline().setStages(Array(index,feature))
  val model = pipeline2.fit(newdata)
  val newDF = model.transform(newdata)
  val availableDF = newDF.select("label","features")
  val Array(trainingData, testData) = availableDF.randomSplit(Array(0.7, 0.3), seed = 100)
  val glr_gau = new GeneralizedLinearRegression().setFamily("gaussian").setLink("identity").setMaxIter(10).setRegParam(0.3)
  val model_gau = glr_gau.fit(trainingData)
  val predictions_gau = model_gau.transform(testData)
	val evaluator_gau = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
  val rmse_gau = evaluator_gau.evaluate(predictions_gau)
	println("The GeneralizedLinearRegression rmse is :" + rmse_gau)
	val summary_gau = model_gau.summary
	println(s"Coefficient Standard Errors: ${summary_gau.coefficientStandardErrors.mkString(",")}")
  println(s"T Values: ${summary_gau.tValues.mkString(",")}")
  println(s"P Values: ${summary_gau.pValues.mkString(",")}")
  println(s"Dispersion: ${summary_gau.dispersion}")
  println(s"Null Deviance: ${summary_gau.nullDeviance}")
	println(s"Residual Degree Of Freedom Null: ${summary_gau.residualDegreeOfFreedomNull}")
  println(s"Deviance: ${summary_gau.deviance}")
  println(s"Residual Degree Of Freedom: ${summary_gau.residualDegreeOfFreedom}")
  println(s"AIC: ${summary_gau.aic}")
  summary_gau.residuals().show()
  
  
  val glr_poi = new GeneralizedLinearRegression(). setFamily("poisson").setLink("identity").setMaxIter(10).setRegParam(0.3)
	val model_poi = glr_poi.fit(trainingData)
	val predictions_poi = model_poi.transform(testData)
	
	val evaluator_poi = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
	val rmse_poi = evaluator_poi.evaluate(predictions_poi)
	println("The GeneralizedLinearRegression rmse is :" + rmse_poi)
	val summary_poi = model_poi.summary
	println(s"Coefficient Standard Errors: ${summary_poi.coefficientStandardErrors.mkString(",")}")
  println(s"T Values: ${summary_poi.tValues.mkString(",")}")
  println(s"P Values: ${summary_poi.pValues.mkString(",")}")
  println(s"Dispersion: ${summary_poi.dispersion}")
  println(s"Null Deviance: ${summary_poi.nullDeviance}")
	println(s"Residual Degree Of Freedom Null: ${summary_poi.residualDegreeOfFreedomNull}")
  println(s"Deviance: ${summary_poi.deviance}")
  println(s"Residual Degree Of Freedom: ${summary_poi.residualDegreeOfFreedom}")
  println(s"AIC: ${summary_poi.aic}")
  summary_poi.residuals().show()
      
  sparkSession.stop()
  }
}