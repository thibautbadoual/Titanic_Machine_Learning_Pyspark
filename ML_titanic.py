# -*- coding: utf-8 -*-

import pyspark.sql.functions as F
import pyspark.sql.types as sparksqltypes
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
import pyspark.ml.tuning as tune

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification  import LogisticRegression
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
# from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import NaiveBayes

"""
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import AFTSurvivalRegression
from pyspark.ml.regression import IsotonicRegression
"""

sc =sc = SparkContext.getOrCreate()
spark  =SparkSession.builder.getOrCreate()
sqlContext = SQLContext(sc)

schema = sparksqltypes.StructType([
sparksqltypes.StructField("Survived", sparksqltypes.DoubleType(), True),
sparksqltypes.StructField("female", sparksqltypes.DoubleType(), True), 
sparksqltypes.StructField("male", sparksqltypes.DoubleType(), True),
sparksqltypes.StructField("Q", sparksqltypes.DoubleType(), True),
sparksqltypes.StructField("C", sparksqltypes.DoubleType(), True),
sparksqltypes.StructField("S", sparksqltypes.DoubleType(), True), 
sparksqltypes.StructField("low", sparksqltypes.DoubleType(), True),
sparksqltypes.StructField("mid", sparksqltypes.DoubleType(), True), 
sparksqltypes.StructField("Very_low", sparksqltypes.DoubleType(), True),
sparksqltypes.StructField("very_high", sparksqltypes.DoubleType(), True),
sparksqltypes.StructField("high", sparksqltypes.DoubleType(), True),
sparksqltypes.StructField("Pclass", sparksqltypes.DoubleType(), True),
sparksqltypes.StructField("Age", sparksqltypes.DoubleType(), True)])

titanic = spark.read.csv('file:///C:/Users/Thibaut/Documents/ML/titanic_pyspark/titanic_clean.csv',schema, header=True)

def prediction(titanic):

    performance = pd.DataFrame({'Name': ['Logistic Regression', "Logistic Regression - Cross Validation", "Random forest", "Random forest - Cross Validation", "Gradient-Boosted Tree Classifier", "Gradient-Boosted Tree Classifier - Cross Validation", "Decision Tree Classifier", "Decision Tree Classifier - Cross Validation", "Multilayer perceptron classifier", "Multilayer perceptron classifier - Cross Validation", "Naive Bayes"], 
                                'Test_SET (Area Under ROC)': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
                                'Accuracy': [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                'Best_Param': ["", "", "", "", "", "", "", "", "", "", ""]})

    # ================================DATA PREPROCESSING==================================================

    features = ['female', 'male', 'Q', 'C', 'S', 'low', 'mid', 'Very_low', 'very_high', 'high', 'Pclass', 'Age']
    titanic = titanic.select(F.col("Survived").alias("label"), *features)

    # Standardize features
    vectorAssembler = VectorAssembler(inputCols=features, outputCol="unscaled_features")
    standardScaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
    stages = [vectorAssembler, standardScaler]
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(titanic)
    titanic = model.transform(titanic)

    # Randomly split data into training and test sets. Set seed for reproducibility
    (X_train, X_test) = titanic.randomSplit([0.7, 0.3], seed=1)

    # ================================MACHINE LEARNING ALGORITHMS=========================================

    results = []
    names = []

    # Logistic Regression
    name = "Logistic Regression"
    lr = LogisticRegression(labelCol="label")
    predictions_lr, model_lr, performance = run_ML(lr, X_train, X_test, performance, name)
    performance = binaryClassificationEvaluator(predictions_lr,  name, performance)
    results.append(pre_plot(predictions_lr.select("probability").toPandas()['probability']))
    names.append(name)
    # With Cross Validation
    name = "Logistic Regression - Cross Validation"
    predictions_lr_cv, model_lr_cv, performance = run_ML_regression_crossValidation(lr, X_train, X_test, performance, name)
    performance = binaryClassificationEvaluator(predictions_lr_cv, name, performance)
    results.append(pre_plot(predictions_lr_cv.select("probability").toPandas()['probability']))
    names.append(name)
    # ROC_Curve(model_lr)
    
    # Random forest
    name = "Random forest"
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    predictions_rf, model_rf, performance = run_ML(rf, X_train, X_test, performance, name)
    performance= binaryClassificationEvaluator(predictions_rf, name, performance)
    performance = multiClassClassificationEvaluator(predictions_rf, name, performance)
    results.append(pre_plot(predictions_rf.select("probability").toPandas()['probability']))
    names.append(name)
    # With Cross Validation
    name = "Random forest - Cross Validation"
    predictions_rf_cv, model_rf_cv, performance = run_ML_random_crossValidation(rf, X_train, X_test, performance, name)
    performance = binaryClassificationEvaluator(predictions_rf_cv, name, performance)
    performance = multiClassClassificationEvaluator(predictions_rf_cv, name, performance)
    results.append(pre_plot(predictions_rf_cv.select("probability").toPandas()['probability']))
    names.append(name)

    # Gradient-Boosted Tree Classifier
    name = "Gradient-Boosted Tree Classifier"
    gbt = GBTClassifier(labelCol="label", featuresCol="features")
    predictions_gbt, model_gbt, performance = run_ML(gbt, X_train, X_test, performance, name)
    performance = binaryClassificationEvaluator(predictions_gbt, name, performance)
    performance = multiClassClassificationEvaluator(predictions_gbt, name, performance)
    results.append(pre_plot(predictions_gbt.select("probability").toPandas()['probability']))
    names.append(name)
    # With Cross Validation
    name = "Gradient-Boosted Tree Classifier - Cross Validation"
    predictions_gbt_cv, model_gbt_cv, performance = run_ML_gbt_crossValidation(gbt, X_train, X_test, performance, name)
    performance = binaryClassificationEvaluator(predictions_gbt_cv, name, performance)
    performance = multiClassClassificationEvaluator(predictions_gbt_cv, name, performance)
    results.append(pre_plot(predictions_gbt_cv.select("probability").toPandas()['probability']))
    names.append(name)

    # DecisionTree model
    name = "Decision Tree Classifier"
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
    predictions_dt, model_dt, performance = run_ML(dt, X_train, X_test, performance, name)
    performance = multiClassClassificationEvaluator(predictions_dt, name, performance)
    results.append(pre_plot(predictions_dt.select("probability").toPandas()['probability']))
    names.append(name)
    # With Cross Validation
    name = "Decision Tree Classifier - Cross Validation"
    predictions_dt_cv, model_dt_cv, performance = run_ML_dt_crossValidation(gbt, X_train, X_test, performance, name)
    performance = multiClassClassificationEvaluator(predictions_dt_cv, name, performance)
    results.append(pre_plot(predictions_dt_cv.select("probability").toPandas()['probability']))
    names.append(name)

    # Multilayer perceptron classifier
    name = "Multilayer perceptron classifier"
    layers = [len(features), 5, 4, 3]
    mpc = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", maxIter=100, layers=layers, blockSize=128)
    predictions_mpc, model_mpc, performance = run_ML(mpc, X_train, X_test, performance, name)
    performance = multiClassClassificationEvaluator(predictions_mpc,  name, performance)
    results.append(pre_plot(predictions_mpc.select("probability").toPandas()['probability']))
    names.append(name)
    # With Cross Validation
    name = "Multilayer perceptron classifier - Cross Validation"
    predictions_mpc_cv, model_mpc_cv, performance = run_ML_mpc_crossValidation(mpc, X_train, X_test, performance, name)
    performance = multiClassClassificationEvaluator(predictions_mpc_cv, name, performance)
    results.append(pre_plot(predictions_mpc_cv.select("probability").toPandas()['probability']))
    names.append(name)

    # Linear Support Vector Machine
    # lsvc = LinearSVC(maxIter=10, regParam=0.1)
    # run_ML(lsvc, X_train, X_test)
    # predictions_lsvc, model_lsvc  = run_ML(lsvc, X_train, X_test)
    # multiClassClassificationEvaluator(predictions_lsvc,  "Linear Support Vector Machine")
    # results.append(pre_plot(predictions_lsvc.select("probability").toPandas()['probability']))
    # names.append("Linear Support Vector Machine")

    # Naive Bayes
    name = "Naive Bayes"
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial", labelCol="label", featuresCol="features")
    predictions_nb, model_nb, performance = run_ML(nb, X_train, X_test, performance, name)
    performance = multiClassClassificationEvaluator(predictions_nb,  name, performance)
    results.append(pre_plot(predictions_nb.select("probability").toPandas()['probability']))
    names.append(name)
    
    """ Regression obviously doesn't work here, it's a classification problem
    # Linear regression
    # linr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    # run_ML(linr, X_train, X_test)
    # predictions_linr, model_linr  = run_ML(linr, X_train, X_test)
    # multiClassClassificationEvaluator(predictions_linr,  "Linear regression")
    # results.append(pre_plot(predictions_linr.select("probability").toPandas()['probability']))
    
    # Generalized linear regression
    # glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)
    # run_ML(glr, X_train, X_test)
    # predictions_glr, model_glr  = run_ML(glr, X_train, X_test)
    # multiClassClassificationEvaluator(predictions_glr,  "Generalized linear regression")
    # results.append(pre_plot(predictions_glr.select("probability").toPandas()['probability']))
    
    # Decision tree regression
    dtr = DecisionTreeRegressor(featuresCol="features")
    run_ML(dtr, X_train, X_test)
    predictions_dtr, model_dtr  = run_ML(dtr, X_train, X_test)
    regressionEvaluator(predictions_dtr,  "Decision tree regression")
    predictions_dtr.show()
    # results.append(pre_plot(predictions_dtr.select("probability").toPandas()['probability']))
    
    # Random forest regression
    rfr = RandomForestRegressor(featuresCol="features")
    run_ML(rfr, X_train, X_test)
    predictions_rfr, model_rfr  = run_ML(rfr, X_train, X_test)
    regressionEvaluator(predictions_rfr,  "Random forest regression")
    predictions_rfr.show()
    # results.append(pre_plot(predictions_rfr.select("probability").toPandas()['probability']))
    
    # Gradient-boosted tree regression
    gbtr = GBTRegressor(featuresCol="features", maxIter=10)
    run_ML(gbtr, X_train, X_test)
    predictions_gbtr, model_gbt  = run_ML(gbtr, X_train, X_test)
    regressionEvaluator(predictions_gbtr,  "Gradient-boosted tree regression")
    predictions_gbtr.show()
    # results.append(pre_plot(predictions_gbtr.select("probability").toPandas()['probability']))
    
    # Survival regression
    # quantileProbabilities = [0.3, 0.6]
    # aft = AFTSurvivalRegression(quantileProbabilities=quantileProbabilities, quantilesCol="quantiles")
    # run_ML(aft, "Survival regression", X_train, X_test)
    # predictions_aft, model_aft  = run_ML(aft, X_train, X_test)
    # multiClassClassificationEvaluator(predictions_aft,  "Survival regression")
    # results.append(pre_plot(predictions_aft.select("probability").toPandas()['probability']))
    
    # Isotonic regression
    # it = IsotonicRegression()
    # run_ML(it, "Isotonic regression", X_train, X_test)
    # predictions_it, model_it  = run_ML(it, X_train, X_test)
    # multiClassClassificationEvaluator(predictions_it,  "Isotonic regression")
    # results.append(pre_plot(predictions_it.select("probability").toPandas()['probability']))
    """
    
    #================================BOXPLOT ALGORITHM COMPARISON========================================
    
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    final_result = spark.createDataFrame(performance)

    return final_result

# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def run_ML(model, train, test, df, name):
    model = model.fit(train)
    predictions = model.transform(test)
    result = df.copy()
    for index, rows in result.iterrows():
        if rows['Name'] == name:
            result.at[index, 'Best_Param'] = "standard"
    return predictions, model, result




# ----------------------------------------------------------------------------------------------------------
# Cross Validation
# ----------------------------------------------------------------------------------------------------------


def run_ML_regression_crossValidation(model, train, test, df, name):
    # We’ll be using cross validation to choose the hyperparameters
    # by creating a grid of the possible pairs of values for the three hyperparameters,
    # elasticNetParam, regParam and maxIter
    # and using the cross validation error to compare all the different models so you can choose the best one

    # We will create a 5-fold CrossValidator

    # The first thing we need when doing cross validation for model selection is a way to compare different models
    evaluator = BinaryClassificationEvaluator()

    # Next, we need to create a grid of values to search over when looking for the optimal hyperparameters

    # Create the parameter grid
    grid = tune.ParamGridBuilder()
    # Add the hyperparameter
    grid = grid.addGrid(model.regParam, np.arange(0, .1, .01))
    grid = grid.addGrid(model.elasticNetParam, [0, 1])
    grid = grid.addGrid(model.maxIter, [1, 5, 10])
    # Build the grid
    grid = grid.build()

    # Create the CrossValidator
    cv = tune.CrossValidator(estimator=model,
               estimatorParamMaps=grid,
               evaluator=evaluator,
               numFolds=5,
               collectSubModels=True
               )

    # Fit cross validation models
    models = cv.fit(train)
    # Extract the best model
    bestModel = models.bestModel
    
    # obtain the best params
    result = df.copy()
    for index, rows in result.iterrows():
        if rows['Name'] == name:
            result.at[index, 'Best_Param'] = "regParam: " + str(bestModel._java_obj.getRegParam()) + " - MaxIter: " + str(bestModel._java_obj.getMaxIter()) + " - elasticNetParam: " + str(bestModel._java_obj.getElasticNetParam())
    
    finalPredictions = bestModel.transform(train)
    return finalPredictions, bestModel, result


def run_ML_random_crossValidation(model, train, test, df, name):
    # Same idea than run_ML_regression_crossValidation()
    evaluator = BinaryClassificationEvaluator()
    grid = tune.ParamGridBuilder()
    grid = grid.addGrid(model.maxDepth, [2, 4, 6])
    grid = grid.addGrid(model.maxBins, [20, 60])
    grid = grid.addGrid(model.numTrees, [5, 20])
    grid = grid.build()
    cv = tune.CrossValidator(estimator=model,
               estimatorParamMaps=grid,
               evaluator=evaluator,
               numFolds=5
               )
    models = cv.fit(train)
    bestModel = models.bestModel
    
    # obtain the best params
    result = df.copy()
    for index, rows in result.iterrows():
        if rows['Name'] == name:
            result.at[index, 'Best_Param'] = "maxDepth: " + str(bestModel._java_obj.getMaxDepth()) + " - maxBins: " + str(bestModel._java_obj.getMaxBins()) + " - numTrees: " + str(bestModel._java_obj.getNumTrees())
            
    finalPredictions = bestModel.transform(train)
    return finalPredictions, bestModel, result


def run_ML_gbt_crossValidation(model, train, test, df, name):
    # Same idea than run_ML_regression_crossValidation()
    evaluator = BinaryClassificationEvaluator()
    grid = tune.ParamGridBuilder()
    grid = grid.addGrid(model.maxDepth, [1, 10, 20, 30])
    grid = grid.addGrid(model.maxIter, [7, 10, 14, 18])
    # grid = grid.addGrid(model.minInstancesPerNode, np.linspace(1, 32, 32, endpoint=True))
    # grid = grid.addGrid(model.subsamplingRate, np.linspace(1, 10, 10, endpoint=True))
    # grid = grid.addGrid(model.maxBins, np.linspace(20, 44, 32, endpoint=True))
    # grid = grid.addGrid(model.minInfoGain, [0,1,2])

    # grid = grid.addGrid(model.min_samples_leaf, [40,50,60])
    grid = grid.build()
    cv = tune.CrossValidator(estimator=model,
               estimatorParamMaps=grid,
               evaluator=evaluator,
               numFolds=5
               )
    models = cv.fit(train)
    bestModel = models.bestModel
    
    # obtain the best params
    result = df.copy()
    for index, rows in result.iterrows():
        if rows['Name'] == name:
            result.at[index, 'Best_Param'] = "maxDepth: " + str(bestModel._java_obj.getMaxDepth()) + " - maxIter: " + str(bestModel._java_obj.getMaxIter())
            
    finalPredictions = bestModel.transform(train)
    return finalPredictions, bestModel, result

def run_ML_dt_crossValidation(model, train, test, df, name):
    # Same idea than run_ML_regression_crossValidation()
    evaluator = MulticlassClassificationEvaluator()
    grid = tune.ParamGridBuilder()
    grid = grid.addGrid(model.maxDepth, [4, 8])
    grid = grid.addGrid(model.maxBins, [2, 4, 6])
    grid = grid.build()
    cv = tune.CrossValidator(estimator=model,
               estimatorParamMaps=grid,
               evaluator=evaluator,
               numFolds=5
               )
    models = cv.fit(train)
    bestModel = models.bestModel
    
    # obtain the best params
    result = df.copy()
    for index, rows in result.iterrows():
        if rows['Name'] == name:
            result.at[index, 'Best_Param'] = "maxDepth: " + str(bestModel._java_obj.getMaxDepth()) + " - maxBins: " + str(bestModel._java_obj.getMaxBins())
            
    finalPredictions = bestModel.transform(train)
    return finalPredictions, bestModel, result

def run_ML_mpc_crossValidation(model, train, test, df, name):
    # Same idea than run_ML_regression_crossValidation()
    evaluator = MulticlassClassificationEvaluator()
    grid = tune.ParamGridBuilder()
    grid = grid.build()
    cv = tune.CrossValidator(estimator=model,
               estimatorParamMaps=grid,
               evaluator=evaluator,
               numFolds=5
               )
    models = cv.fit(train)
    bestModel = models.bestModel
    
    # obtain the best params
    result = df.copy()
    for index, rows in result.iterrows():
        if rows['Name'] == name:
            result.at[index, 'Best_Param'] = "unspecified"
            
    finalPredictions = bestModel.transform(train)
    return finalPredictions, bestModel, result

# ----------------------------------------------------------------------------------------------------------
# ROC Curve
# ----------------------------------------------------------------------------------------------------------


def ROC_Curve(model):
    # plotting the ROC Curve
    trainingSummary = model.summary
    roc = trainingSummary.roc.toPandas()
    plt.plot(roc['FPR'], roc['TPR'])
    plt.ylabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    print('Training set ROC: ' + str(trainingSummary.areaUnderROC))
    return

# ----------------------------------------------------------------------------------------------------------
# Evaluate our models
# ----------------------------------------------------------------------------------------------------------


def binaryClassificationEvaluator(predictions, name, df):
    evaluator = BinaryClassificationEvaluator()
    for index, rows in df.iterrows():
        if rows['Name'] == name:
            df.at[index, 'Test_SET (Area Under ROC)'] = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
    return df


def multiClassClassificationEvaluator(predictions, name, df):
    result = df.copy()
    evaluator = MulticlassClassificationEvaluator()
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    for index, rows in result.iterrows():
        if rows['Name'] == name:
            result.at[index, 'Accuracy'] = accuracy
    return result

# ----------------------------------------------------------------------------------------------------------
# boxplot algorithm comparison
# ----------------------------------------------------------------------------------------------------------


def pre_plot(data):
    result = [0]*len(data)
    for i in range(0, len(data)):
        result[i] = max(data[i][0], data[i][1])
    return result

# ----------------------------------------------------------------------------------------------------------

df = prediction(titanic)
df.show()