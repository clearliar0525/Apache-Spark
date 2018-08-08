import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor

val sparkSession = SparkSession.builder.master("local[1]").appName("Decision Tree Regression").getOrCreate()
val data_string = sparkSession.sparkContext.textFile("files/HIGGSsy.csv")
val data = data_string.map(line => line.split(',').map(_.toDouble)).map(t => (t(0), Vectors.dense(t.take(29).drop(1)))).toDF("label","features")
// Automatically identify categorical features, and index them.
// Here, we treat features with > 4 distinct values as continuous.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
// Train a DecisionTree model.
val dtr = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
// Chain indexer and tree in a Pipeline.
val pipeline = new Pipeline().setStages(Array(featureIndexer, dtr))
// We use a ParamGridBuilder to construct a grid of parameters to search over.
val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.3, 0.6, 0.9)).build()
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator). setEstimatorParamMaps(paramGrid).setNumFolds(2)
val cvModel = cv.fit(trainingData)
// Make predictions.
val predictions = model.transform(testData)
// Select example rows to display.
predictions.select("prediction", "label", "features").show(5)
// Select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)
