import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.classification.LogisticRegression

object Assignment2_1 {

  def main(args: Array[String]): Unit = {   
  
  val sparkSession = SparkSession.builder.master("local[5]").appName("Assignment2_1").getOrCreate()
  val sc = sparkSession.sparkContext
  val sqlContext= new org.apache.spark.sql.SQLContext(sc)
  import sqlContext.implicits._

  val data_string = sparkSession.sparkContext.textFile("files/HIGGS_10%.csv")
  val data = data_string.map(line => line.split(',').map(_.toDouble)).map(t => (t(0), Vectors.dense(t.take(29).drop(1)))).toDF("label","features")
  val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
  // features with > 4 distinct values are treated as continuous.
  val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
  // Split the data into training and test sets (30% held out for testing)
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 100)
  var beginTrain1 = System.currentTimeMillis()
  // Train a DecisionTree model.
  val dtc = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
  // Convert indexed labels back to original labels.
  val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
  // Chain indexers and tree in a Pipeline.
  val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dtc, labelConverter))
  // We use a ParamGridBuilder to construct a grid of parameters to search over.
  val paramGrid = new ParamGridBuilder().addGrid(dtc.maxDepth, Array(3, 4, 5)).addGrid(dtc.maxBins, Array(10, 20, 30)).build()
  val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator). setEstimatorParamMaps(paramGrid).setNumFolds(3)
  val cvModel = cv.fit(trainingData)
  // Make predictions.
  val predictions = cvModel.transform(testData)
  // Select example rows to display.
  predictions.select("predictedLabel", "label", "features").show(5)
  // Select (prediction, true label) and compute test error.
  val evaluator_dtc = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
  val accuracy = evaluator_dtc.evaluate(predictions)
  println("Test Error = " + (1.0 - accuracy))
  var endTrain1 = System.currentTimeMillis()
  var trainTime1 = (endTrain1 - beginTrain1) / 1000.0
  println("Training time for DecisionTreeClassifier model: " + trainTime1 + " seconds")
  println(cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics).maxBy(_._2))

  var beginTrain2 = System.currentTimeMillis()
  // Train a DecisionTree Regression  model.
  val dtr = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
  // Chain indexer and tree in a Pipeline.
  val pipeline_dtr = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dtr, labelConverter))
  // We use a ParamGridBuilder to construct a grid of parameters to search over.
  val paramGrid_dtr = new ParamGridBuilder().addGrid(dtr.maxDepth, Array(3, 4, 5)).addGrid(dtr.maxBins, Array(10, 20, 30)).build()
  val evaluator_dtr = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
  val cv_dtr = new CrossValidator().setEstimator(pipeline_dtr).setEvaluator(evaluator_dtr). setEstimatorParamMaps(paramGrid_dtr).setNumFolds(3)
  val cvModel_dtr = cv_dtr.fit(trainingData)
  // Make predictions.
  val predictions_dtr = cvModel_dtr.transform(testData)
  // Select example rows to display.
  predictions_dtr.select("prediction", "label", "features").show(5)
  // Select (prediction, true label) and compute test error.
  val rmse_dtr = evaluator_dtr.evaluate(predictions)
  println("Root Mean Squared Error (RMSE) on test data = " + rmse_dtr)
  var endTrain2 = System.currentTimeMillis()
  var trainTime2 = (endTrain2 - beginTrain2) / 1000.0
  println("Training time for DecisionTreeRegressor model: " + trainTime2 + " seconds")
  println(cvModel_dtr.getEstimatorParamMaps.zip(cvModel_dtr.avgMetrics).maxBy(_._2))
  val tree_dtr = cvModel_dtr.bestModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[DecisionTreeClassificationModel]
  tree_dtr.getImpurity
  tree_dtr.getMaxBins
  tree_dtr.getMaxDepth
  println("Learned classification tree model:\n" + tree_dtr.toDebugString)

  var beginTrain3 = System.currentTimeMillis()
  val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
  // Chain indexers and tree in a Pipeline.
  val pipeline_lr = new Pipeline().setStages(Array(labelIndexer, featureIndexer, lr, labelConverter))
  // We use a ParamGridBuilder to construct a grid of parameters to search over.
  val paramGrid_lr = new ParamGridBuilder().addGrid(lr.regParam, Array(0.3, 0.6, 0.9)).build()
  val evaluator_lr = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
  val cv_lr = new CrossValidator().setEstimator(pipeline_lr).setEvaluator(evaluator_lr). setEstimatorParamMaps(paramGrid_lr).setNumFolds(3)
  val cvModel_lr = cv_lr.fit(trainingData)
  // Make predictions.
  val predictions_lr = cvModel_lr.transform(testData)
  val accuracy_lr = evaluator_lr.evaluate(predictions)
  println("rmse = " + (accuracy_lr))
  var endTrain3 = System.currentTimeMillis()
  var trainTime3 = (endTrain3 - beginTrain3) / 1000.0
  println("Training time for LogisticRegression model: " + trainTime3 + " seconds")
  println(cvModel_lr.getEstimatorParamMaps.zip(cvModel_lr.avgMetrics).maxBy(_._2))



  sparkSession.stop()
  }
}


