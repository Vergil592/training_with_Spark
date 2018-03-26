import org.apache.spark.mllib
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._

import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.configuration.Algo


val rawData =
sc.textFile(s"${sys.env.get("DATADIR").getOrElse("data")}/higgs100k.csv")
println(s"Number of rows: ${rawData.count}")

raw.take(2)


val data = rawData.map(line => line.split(',').map(_.toDouble))

val response: RDD[Int] = data.map(row => row(0).toInt)
val features: RDD[Vector] = data.map(line => Vectors.dense(line.slice(1,
line.size)))


val featuresMatrix = new RowMatrix(features)
val featuresSummary = featuresMatrix.computeColumnSummaryStatistics()

import org.apache.spark.utils.Tabulizer._
println(s"Higgs Features Mean Values = ${table(featuresSummary.mean, 8)}")


val nonZeros = featuresSummary.numNonzeros
println(s"Non-zero values count per column: ${table(nonZeros, cols = 8,
format = "%.0f")}")

val numRows = featuresMatrix.numRows
val numCols = featuresMatrix.numCols
val colsWithZeros = nonZeros.toArray.zipWithIndex.filter { case (rows, idx) => rows != numRows }


val sparsity = nonZeros.toArray.sum / (numRows * numCols)
println(f"Data sparsity: ${sparsity}%.2f")


val responseValues = response.distinct.collect
println(s"Response values: ${responseValues.mkString(", ")}")

val responseDistribution = response.map(v => (v,1)).countByKey
println(s"Response distribution:\n${table(responseDistribution)}")

import org.apache.spark.h2o._
val h2oContext = H2OContext.getOrCreate(sc)


val higgs = response.zip(features).map {case (response, features) => LabeledPoint(response, features) }

higgs.setName("higgs").cache()

## Create Train & Test Splits
val trainTestSplits = higgs.randomSplit(Array(0.8, 0.2))
val (trainingData, testData) = (trainTestSplits(0), trainTestSplits(1))


## First model : Decision Tree
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 6
val maxBins = 12

val dtreeModel = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

# return (label, prediction) for each obs using dtreeModel.predict
val treeLabelAndPreds = testData.map { point =>
val prediction = dtreeModel.predict(point.features)
(point.label.toInt, prediction.toInt)
}

# model accuracy
val treeTestAcc = treeLabelAndPreds.filter(r => r._1 == r._2).count.toDouble / testData.count()
println(f"Tree Model: Test Error = ${treeTestErr}%.3f")


val cm = treeLabelAndPreds.combineByKey(
createCombiner = (label: Int) => if (label == 0) (1,0) else (0,1),
mergeValue = (v:(Int,Int), label:Int) => if (label == 0) (v._1 +1, v._2)
else (v._1, v._2 + 1),
mergeCombiners = (v1:(Int,Int), v2:(Int,Int)) => (v1._1 + v2._1, v1._2 +
v2._2)).collect

val (tn, tp, fn, fp) = (cm(0)._2._1, cm(1)._2._2, cm(1)._2._1, cm(0)._2._2)

# To print as an array
# in order to show the confusion matrix
println(f"""Confusion Matrix
| ${0}%5d ${1}%5d ${"Err"}%10s
|0 ${tn}%5d ${fp}%5d ${tn+fp}%5d ${fp.toDouble/(tn+fp)}%5.4f
|1 ${fn}%5d ${tp}%5d ${fn+tp}%5d ${fn.toDouble/(fn+tp)}%5.4f
| ${tn+fn}%5d ${fp+tp}%5d ${tn+fp+fn+tp}%5d
${(fp+fn).toDouble/(tn+fp+fn+tp)}%5.4f
|""".stripMargin)


## Model 2 : Random Forest

val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 10
val featureSubsetStrategy = "auto"
val impurity = "gini"
val maxDepth = 5
val maxBins = 12
val seed = 42

val rfModel = RandomForest.trainClassifier(trainingData, numClasses,
categoricalFeaturesInfo,
numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed)

def computeError(model: Predictor, data: RDD[LabeledPoint]): Double = {
  val labelAndPreds = data.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
  }
  labelAndPreds.filter(r => r._1 != r._2).count.toDouble/data.count
}

val rfTestErr = computeError(rfModel, testData)
println(f"RF Model: Test Accuracy = ${rfTestErr}%.3f")


val randFLabelAndPreds = testData.map { point =>
val prediction = rfModel.predict(point.features)
(point.label.toInt, prediction.toInt)
}

# model accuracy
val randFTestAcc = randFLabelAndPreds.filter(r => r._1 == r._2).count.toDouble / testData.count()
println(f"Tree Model: Test Accuracy = ${randFTestAcc }%.3f")

# Confusion Matrix
val cmRandF = randFLabelAndPreds.combineByKey(
createCombiner = (label: Int) => if (label == 0) (1,0) else (0,1),
mergeValue = (v:(Int,Int), label:Int) => if (label == 0) (v._1 +1, v._2)
else (v._1, v._2 + 1),
mergeCombiners = (v1:(Int,Int), v2:(Int,Int)) => (v1._1 + v2._1, v1._2 +
v2._2)).collect



# GradientBoostedTrees Model

val gbmStrategy = BoostingStrategy.defaultParams(Algo.Classification)
gbmStrategy.setNumIterations(10)
gbmStrategy.setLearningRate(0.1)
gbmStrategy.treeStrategy.setNumClasses(2)
gbmStrategy.treeStrategy.setMaxDepth(10)
gbmStrategy.treeStrategy.setCategoricalFeaturesInfo(java.util.Collections.emptyMap[Integer, Integer])


val gbmModel = GradientBoostedTrees.train(trainingData, gbmStrategy)

# Accuracy
val gbmLabelAndPreds = testData.map { point =>
val prediction = gbmModel.predict(point.features)
(point.label.toInt, prediction.toInt)
}

val gbmTestAccuracy = gbmLabelAndPreds.filter(r => r._1 == r._2).count.toDouble / testData.count()
println(f"Tree Model: Test Error = ${gbmTestAcc }%.3f")


# Grid Search

val rfGrid =
for (
val gridNumTrees2 <- Array(15, 20);
gridImpurity2 <- Array("entropy", "gini");
gridDepth2 <- Array(20, 30);
gridBins2 <- Array(20, 50))
yield {
  val gridModel = RandomForest.trainClassifier(trainingData, 2, Map[Int,
  Int](), gridNumTrees2, "auto", gridImpurity2, gridDepth2, gridBins2)
  val labelAndPredstmp = testData.map{ point =>
  val prediction = gridModel.predict(point.features)
  (point.label, prediction)
  }
  labelAndPredstmp.filter(r => r._1 != r._2).count.toDouble/data.count
  ((gridNumTrees2, gridImpurity2, gridDepth2, gridBins2), gridErr)
}
