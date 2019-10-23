// Databricks notebook source
// MAGIC %md # Apache Spark - Getting Started

// COMMAND ----------

println("Hello world")

// COMMAND ----------

display(dbutils.fs.ls("/databricks-datasets/samples/docs/"))

// COMMAND ----------

val textFile = spark.read.text("/databricks-datasets/samples/docs/README.md")

// COMMAND ----------

textFile.count

// COMMAND ----------

def ls(dir: String) = display(dbutils.fs.ls(dir))

// COMMAND ----------

ls("/databricks-datasets")

// COMMAND ----------

ls("/databricks-datasets/samples/docs/")

// COMMAND ----------

def read(file: String) = spark.read.textFile(file)

// COMMAND ----------

read("/databricks-datasets/samples/docs/README.md")

// COMMAND ----------

val textFile = read("/databricks-datasets/samples/docs/README.md")

// COMMAND ----------

textFile.count


// COMMAND ----------

textFile.first

// COMMAND ----------

val linesWithSpark = textFile.filter(_.contains("Spark"))

// COMMAND ----------

spark.read.text _

// COMMAND ----------

spark.read.textFile _

// COMMAND ----------

linesWithSpark.count

// COMMAND ----------

linesWithSpark.collect().take(5).foreach(println)

// COMMAND ----------

// MAGIC %md # DataFrames

// COMMAND ----------

val data = spark.read.option("header", "true").option("inferSchema", "true").csv("/databricks-datasets/samples/population-vs-price/data_geo.csv")

// COMMAND ----------

data.cache

// COMMAND ----------

val cleanData = data.na.drop


// COMMAND ----------

cleanData.take(10)

// COMMAND ----------

display(cleanData)

// COMMAND ----------

cleanData.createOrReplaceTempView("data_geo")

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select `State Code`, `2015 median sales price` from data_geo

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select City, `2014 Population estimate` from data_geo where `State Code` = 'WA';

// COMMAND ----------

// MAGIC %md # DataSets

// COMMAND ----------

val range100 = spark.range(100)

// COMMAND ----------

range100.collect()

// COMMAND ----------

val df = spark.read.json("/databricks-datasets/samples/people/people.json")

// COMMAND ----------

// MAGIC %md ### Data type coercion - with the Spark API

// COMMAND ----------

case class WrongPerson (name: String, age: Boolean)

val wrongDs = spark.read.json("/databricks-datasets/samples/people/people.json").as[WrongPerson]

// COMMAND ----------

case class Person (name: String, age: Long)


val ds = spark.read.json("/databricks-datasets/samples/people/people.json").as[Person]

// COMMAND ----------

// MAGIC %md ### Data type coercion - with the Frameless API (TODO)

// COMMAND ----------

import frameless.TypedDataset
import org.apache.spark.sql.Dataset

import spark.implicits._
import frameless.syntax._

case class WrongPerson (name: String, age: Boolean)


val wrongDs = spark.read.json("/databricks-datasets/samples/people/people.json")


// ???

// COMMAND ----------

// MAGIC %md # Machine Learning

// COMMAND ----------

val data = spark.read.option("header","true").option("inferSchema","true").csv("/databricks-datasets/samples/population-vs-price/data_geo.csv")

data.cache

// COMMAND ----------

display(data)

// COMMAND ----------

val cdata = data.na.drop

val columns = cdata.columns

// COMMAND ----------

import org.apache.spark.sql.functions.col

val exprs = columns.map(c => col(c).alias(c.replace(' ', '_')))

// COMMAND ----------

val vdata = cdata.select(exprs: _*).selectExpr("2014_Population_estimate as population", "2015_median_sales_price as label")

display(vdata)

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler


val assembler     = new VectorAssembler().setInputCols(Array("population")).setOutputCol("features")
val stages        = Array(assembler)
val pipeline      = new Pipeline().setStages(stages)
val pipelineModel = pipeline.fit(vdata)
val dataset       = pipelineModel.transform(vdata)

// COMMAND ----------

display(dataset.select("features", "label"))

// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression

val lr = new LinearRegression()

val modelA = lr.fit(dataset, lr.regParam -> 0.0)
val modelB = lr.fit(dataset, lr.regParam -> 100.0)

// COMMAND ----------

val predictionsA = modelA.transform(dataset)
display(predictionsA)

// COMMAND ----------

val predictionsB = modelB.transform(dataset)
display(predictionsB)

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

val evaluator = new RegressionEvaluator().setMetricName("rmse")
val RMSEa     = evaluator.evaluate(predictionsA)

println(s"ModelA: Root Mean Squared Error = $RMSEa")

val RMSEb     = evaluator.evaluate(predictionsB)
println(s"ModelB: Root Mean Squared Error = $RMSEb")

// COMMAND ----------

// MAGIC %md # Structured Streaming

// COMMAND ----------

