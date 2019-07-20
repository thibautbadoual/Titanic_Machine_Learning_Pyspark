# -*- coding: utf-8 -*-
# @author: Thibaut Badoual

import pyspark.sql.functions as F
import pyspark.sql.types as sparksqltypes
from pyspark.sql.types import StringType
from sklearn.ensemble import RandomForestRegressor
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler as StandardScaler_1
from sklearn.preprocessing import StandardScaler as StandardScaler_2
from pyspark.ml import Pipeline
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pandas as pd

sc =sc = SparkContext.getOrCreate()
spark  =SparkSession.builder.getOrCreate()
sqlContext = SQLContext(sc)

schema = sparksqltypes.StructType([ 
sparksqltypes.StructField("PassengerId", sparksqltypes.DoubleType(), True), 
sparksqltypes.StructField("Survived", sparksqltypes.DoubleType(), True),
sparksqltypes.StructField("Pclass", sparksqltypes.DoubleType(), True),
sparksqltypes.StructField("Name", sparksqltypes.StringType(), True),
sparksqltypes.StructField("Sex", sparksqltypes.StringType(), True), 
sparksqltypes.StructField("Age", sparksqltypes.DoubleType(), True),
sparksqltypes.StructField("SibSp", sparksqltypes.DoubleType(), True), 
sparksqltypes.StructField("Parch", sparksqltypes.DoubleType(), True),
sparksqltypes.StructField("Ticket", sparksqltypes.StringType(), True),
sparksqltypes.StructField("Fare", sparksqltypes.DoubleType(), True),
sparksqltypes.StructField("Cabin", sparksqltypes.StringType(), True),
sparksqltypes.StructField("Embarked", sparksqltypes.StringType(), True)])

titanic = spark.read.csv('file:///C:/Users/Thibaut/Documents/ML/titanic_pyspark/titanic.csv',schema, header=True)

# ----------------------------------------------------------------------------------------------------------

def my_compute_function(titanic):
    
    # first step = feature engineering
    titanic = feature_engineering(titanic)

    # Create dummies for 'Sex', 'family', 'Embarked', 'fare_group' features
    titanic = dummies(titanic)

    # Drop useless columns
    titanic = titanic.drop('Sex', 'family', 'Embarked', 'fare_group', 'PassengerId', 'None')
    titanic = titanic.select('Survived', 'female', 'male', 'Q', 'C', 'S', 'low', 'mid', 'Very_low', 'very_high', 'high', 'Pclass', 'Age')
    
    # Complete 'Age' column
    # By using Random forest regressor to predict the missing age values
    titanic = titanic.groupby('Pclass').apply(age)

    # Standardize features
    # /!\ Do this later, at the same time as ML 
    # 2 methods with different outputs!
    
    # Method 1: output = vector
    # titanic = feature_scaling_1(titanic)
    # Method 2: output = dataframe
    # titanic = titanic.groupby('Survived').apply(feature_scaling_2)
   
    return titanic

# ----------------------------------------------------------------------------------------------------------

def dummies(titanic):
    # Similar to "titanic = pd.get_dummies(titanic, columns=['Sex', 'family', 'Embarked', 'fare_group'], drop_first=False)"
    
    # Sex
    categ = titanic.select('Sex').distinct().rdd.flatMap(lambda x: x).collect()
    sex = [F.when(F.col('Sex') == cat, 1).otherwise(0).alias(str(cat)) for cat in categ]
    # family
    categ = titanic.select('family').distinct().rdd.flatMap(lambda x: x).collect()
    family = [F.when(F.col('family') == cat, 1).otherwise(0).alias(str(cat)) for cat in categ]
    # Embarked
    categ = titanic.select('Embarked').distinct().rdd.flatMap(lambda x: x).collect()
    Embarked = [F.when(F.col('Embarked') == cat, 1).otherwise(0).alias(str(cat)) for cat in categ]
    # fare
    categ = titanic.select('fare_group').distinct().rdd.flatMap(lambda x: x).collect()
    fare = [F.when(F.col('fare_group') == cat, 1).otherwise(0).alias(str(cat)) for cat in categ]
    
    titanic = titanic.select(sex+family+Embarked+fare+titanic.columns)
    return titanic

# ----------------------------------------------------------------------------------------------------------

def feature_engineering(titanic):
    titanic = titanic.withColumn("family_size", (F.col("SibSp") + F.col("Parch") + 1).cast(sparksqltypes.DoubleType()))
    titanic.select( F.col('family_size')).na.drop()
    
    family_udf = F.udf(family_group,  sparksqltypes.DoubleType())
    titanic = titanic.withColumn('family', family_udf(F.col('family_size')))
    
    titanic.select( F.col('Fare')).na.drop()
    fare_udf = F.udf(fare_group, StringType())
    titanic = titanic.withColumn('fare_group', fare_udf(F.col('Fare')))
    
    titanic = titanic.drop("Name", "ticket", "SibSp", "Parch", "family_size", "Fare", "Cabin")
    return titanic
    
# ----------------------------------------------------------------------------------------------------------

def family_group(size):
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a

# ----------------------------------------------------------------------------------------------------------

def fare_group(fare):
    a = ''
    if float(fare) <= 4:
        a = 'Very_low'
    elif float(fare) <= 10:
        a = 'low'
    elif float(fare) <= 20:
        a = 'mid'
    elif float(fare) <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a

# ----------------------------------------------------------------------------------------------------------

def cabin_estimator(i):
    a = 0
    if i < 16:
        a = "G"
    elif i >= 16 and i < 27:
        a = "F"
    elif i >= 27 and i < 38:
        a = "T"
    elif i >= 38 and i < 47:
        a = "A"
    elif i >= 47 and i < 53:
        a = "E"
    elif i >= 53 and i < 54:
        a = "D"
    elif i >= 54 and i < 116:
        a = 'C'
    else:
        a = "B"
    return a

# ----------------------------------------------------------------------------------------------------------

def feature_scaling_1(titanic):
    features = ['female', 'male', 'Q', 'C', 'S', 'low', 'mid', 'Very_low', 'very_high', 'high', 'Passenger', 'Pclass', 'Age']

    titanic = titanic.select(F.col("Survived").alias("label"), *features)
    vectorAssembler = VectorAssembler(inputCols=features, outputCol="unscaled_features")
    standardScaler = StandardScaler_1(inputCol="unscaled_features", outputCol="features")
    lr = LinearRegression(maxIter=10, regParam=.01)

    stages = [vectorAssembler, standardScaler, lr]
    pipeline = Pipeline(stages=stages)
    
    model = pipeline.fit(titanic)
    titanic = model.transform(titanic)
    return titanic

# ----------------------------------------------------------------------------------------------------------
    
schema_feature = sparksqltypes.StructType([
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
 
@pandas_udf(schema_feature, functionType=PandasUDFType.GROUPED_MAP)
def feature_scaling_2(data):
    result = pd.DataFrame(data)
    result = result.astype(dtype = {'Survived':float, 'female': float, "male":float, "Q": float, "C":float, 'S':float , 'low':float, 'mid': float, "Very_low":float, 
                                    "very_high": float, "high":float, 'Pclass':float , 'Age':float })
    result_final = result.drop(['Survived'], axis = 1)
    names = result_final.columns
    sc = StandardScaler_2()
    scalerModel = sc.fit(result_final)
    result_final = scalerModel.transform(result_final)
    result_final = pd.DataFrame(result_final, columns=names)
    result_final['Survived'] = result['Survived']
    return result_final

# ----------------------------------------------------------------------------------------------------------
    
@pandas_udf(schema_feature, functionType=PandasUDFType.GROUPED_MAP)
def age(data):
    result = pd.DataFrame(data)
    result = result.astype(dtype = {'Survived':float, 'female': float, "male":float, "Q": float, "C":float, 'S':float , 'low':float, 'mid': float, "Very_low":float, 
                                    "very_high": float, "high":float, 'Pclass':float , 'Age':float })
    return completing_age(result)


def completing_age(df):
    ## gettting all the features except survived
    age_df = df.drop(['Survived', 'Pclass'], axis=1)
    
    temp_train = age_df.loc[age_df.Age.notnull()] ## df with age values
    temp_test = age_df.loc[age_df.Age.isnull()] ## df without age values
    
    y = temp_train.Age.values ## setting target variables(age) in y 
    x = temp_train.loc[:, :'high'].values
    
    rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)
    rfr.fit(x, y)
    
    predicted_age = rfr.predict(temp_test.loc[:, :'high'])
    
    df.loc[df.Age.isnull(), "Age"] = predicted_age
    return df

# ----------------------------------------------------------------------------------------------------------

titanic = my_compute_function(titanic)
titanic.show()
titanic.repartition(1).write.format('com.databricks.spark.csv').save("file:///C:/Users/Thibaut/Documents/ML/titanic_pyspark/titanic_clean.csv")