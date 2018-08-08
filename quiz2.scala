// The usual imports
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator

// "Open the bridge" 
val sparkSession = SparkSession.builder.
    master("local[1]").
    appName("Decision Tree Regression").
    getOrCreate()


// Import the data as text and remove the header// Impo 
val data_string = sparkSession.sparkContext.textFile("files/winequality-white.csv")
val header = data_string.first
val only_rows = data_string.filter(line => line != header)

// Convert to double
val data_double = only_rows.map(line => line.split(';').map(_.toDouble))

// Organise data in features and labels
val data_rdd = data_double.map(t => (t(11), Vectors.dense(t.take(11))))

// Convert to a dataframe
val data = sparkSession.createDataFrame(data_rdd).toDF("label", "features")


// Split the data into training and test sets (30% held out for testing. Use your registration number as the seed// Spli 
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 100)


// Index the feature vector// Inde 
val dt = new DecisionTreeRegressor().
    setLabelCol("label").
    setFeaturesCol("features")

// Create the pipeline, only for the decision tree
val pipeline = new Pipeline().
    setStages(Array(dt))

// We use a ParamGridBuilder to construct a grid of parameters to search over.
// With three values for dt.maxBin and three values for dt.maxDepth,
// this grid will have 3 x 3 = 9 parameter settings for CrossValidator to choose from.
val paramGrid = new ParamGridBuilder().
   addGrid(dt.maxBins, Array(5, 10, 20)).
   addGrid(dt.maxDepth, Array(1, 3, 5)).
   build()

// Setup the evaluator
val metric = "rmse"
val evaluator = new RegressionEvaluator().
    setLabelCol("label").
    setPredictionCol("prediction").
    setMetricName(metric)

// We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
// This will allow us to jointly choose parameters for all Pipeline stages.
// A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
// Note that the evaluator here is a RegressionEvaluator and its metric
// is the rmse.
val cv = new CrossValidator().
  setEstimator(pipeline).
  setEvaluator(evaluator).
  setEstimatorParamMaps(paramGrid).
  setNumFolds(5)

// Run cross-validation, and choose the best set of parameters.
val cvModel = cv.fit(trainingData)

// Compute the root mean square error
val pred = cvModel.transform(testData)
val rmse = evaluator.evaluate(pred)
println("The RMSE on the test data is " + rmse)