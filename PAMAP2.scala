import org.apache.spark.mllib
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._

import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.rdd.RDD

val path = "C:/spark/data_SparkR/PAMAP2_Dataset/Optional"
val dataFiles = sc.wholeTextFiles(path)

val colNames = Array("timestamp", "activityId", "hr") ++ Array(
    "hand", "chest", "ankle").flatMap(sensor =>
    Array(
        "temp",
        "accel1X", "accel1y", "accel1Z",
        "accel2X", "accel2y", "accel2Z",
        "gyroX", "gyroY", "gyroZ",
        "magnetX", "magnetY", "magnetZ",
        "orientX", "orientY", "orientZ").
        map(name => s"${sensor}_${name}")
    )


val ignoredColumns =Array(0,
    3 + 13, 3 + 14, 3 + 15, 3 + 16,
    20 + 13, 20 + 14, 20 + 15, 20 + 16,
    37 + 13, 37 + 14, 37 + 15, 37 + 16)


// on cree un compteur avec zipWithIndex pour retirer les colonnes
// qui sont dans ignoredColumns
val rawData = dataFiles.flatMap { case (path, content) =>
    content.split("\n")
    }.map { row =>
        row.split(" ").map(_.trim).map(_.toDouble)
        zipWithIndex.collect {
        case (cell, idx) if !ignoredColumns.contains(idx) => cell
        }
    }
rawData.cache()


val rawData = dataFiles.flatMap { case (path, content) => content.split("\n") }.
    map{row => row.split(" ").map(_.trim)
            .map(v => if(v.toUpperCase == "NAN") Double.NaN else v.toDouble)
            .zipWithIndex.collect{case (cell, idx) if !ignoredColumns.contains(idx) => cell}
        }


// Nom des colonnes
val columnNames = colNames.
zipWithIndex.
filter { case (_, idx) => !ignoredColumns.contains(idx) }.
map { case (name, _) => name }

// correspondance entre activité et classe
val activities = Map(
1 -> "lying", 2 -> "sitting", 3 -> "standing", 4 -> "walking",
5 -> "running", 6 -> "cycling", 7 -> "Nordic walking",
9 -> "watching TV", 10 -> "computer work", 11 -> "car driving",
12 -> "ascending stairs", 13 -> "descending stairs",
16 -> "vacuum cleaning", 17 -> "ironing",
18 -> "folding laundry", 19 -> "house cleaning",
20 -> "playing soccer", 24 -> "rope jumping", 0 -> "other")

// labels
val dataActivityId = rawData.map(l => l(0).toInt)

// Activity count
val activityIdCount = dataActivityId.map(n => (n,1)).reduceByKey(_+_)

// Activities Array
val activityCounts = activityIdCounts.sortBy { 
        case (activityId, count) => count
    }.map { case (activityId, count) =>(activities(activityId), count)
}

val nanCountPerRow = rawData.map { row =>
    row.foldLeft(0) { case (acc, v) =>
    acc + (if (v.isNaN) 1 else 0)
    }
}

val nrows = rawData.count
// Proportion of Nan
val nanCount = nanCountPerRow.sum
val nanRatio = nanCount * 100 / (rawData.take(1)(0).length * rawData.count)

val nanRowDistribution = nanCountPerRow.map(count => (count,1)).reduceByKey(_+_).sortBy(-_._1)

val nanRowTreshhold = 25
val rowWithManyNan = nanCountPerRow.zipWithIndex.zip(rawData).filter(_._1._1 > nanRowTreshhold).sortBy(_._1._1)

val nanCountPerColumn = rawData.map { row =>
row.map(v => if (v.isNaN) 1 else 0)
}.reduce((v1, v2) => v1.indices.map(i => v1(i) + v2(i)).toArray)

val heartRateColumn = rawData.map(row => row(1)).filter(_.isNaN).map(_.toInt)

// what to do with missing values ? -> replace them by mean

val imputedValues = columnNames.map {
    _ match {
    case "hr" => 60.0 // value to replace if there is a nan value for column "hr"
    case _ => 0.0
    }
}

// Function too impute nan
def imputeNaN(data: RDD[Array[Double]], values: Array[Double]): RDD[Array[Double]] = {
    data.map { row =>
    row.indices.map { i =>
    if (row(i).isNaN) values(i)
    else row(i)
    }.toArray
    }
}

// Function to filter rows with too many nan
def filterBadRows(rdd: RDD[Array[Double]], nanCountPerRow: RDD[Int], nanThreshold: Int): RDD[Array[Double]] = {
    rdd.zip(nanCountPerRow).filter { case (row, nanCount) =>
    nanCount < nanThreshold
    }.map { case (row, _) =>
    row
    }
}

val activityId2Idx = activityIdCounts.map(_._1).collect.zipWithIndex.toMap

val processedRawData = imputeNaN(filterBadRows(rawData, nanCountPerRow, nanThreshold = 26),imputedValues)

// Training
//RF
import org.apache.spark.mllib
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark.mllib.tree.configuration._
import org.apache.spark.mllib.tree.impurity._

val data = processedRawData.map { r =>
val activityId = r(0)
val activityIdx = activityId2Idx(activityId.toInt)
val features = r.drop(1)
LabeledPoint(activityIdx, Vectors.dense(features))
}

val split = data.randomSplit(Array(0.75,0.25))
val (trainingData, testData) = (split(0), split(1))

// hyperparameters
val rfStrategy = new Strategy(algo = Algo.Classification, impurity = Entropy, 
maxDepth = 6, maxBins = 20, numClasses = activityId2Idx.size,
categoricalFeaturesInfo = Map[Int, Int](), subsamplingRate = 0.65)

// Random Forest Model
val rfModel = RandomForest.trainClassifier(
    input = trainingData, strategy = rfStrategy, numTrees = 50,
    featureSubsetStrategy = "auto", seed = 42
)

// Testing
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree.model._

def getMetrics(model: RandomForestModel, data: RDD[LabeledPoint]):
MulticlassMetrics = {
val predictionsAndLabels = data.map(example =>
(model.predict(example.features), example.label)
)
new MulticlassMetrics(predictionsAndLabels)
}

val rfModelMetrics = getMetrics(rfModel, testData)
