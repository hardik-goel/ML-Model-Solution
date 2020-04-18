import seaborn as sns
from pyspark.sql import SparkSession
import datetime
import sys
from pyspark.sql.functions import col
import csv, re, traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as f
import pyspark
from pyspark.ml.regression import LinearRegression
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import when,exp
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import *
from pyspark.sql.functions import *
import math
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

def our_main(file_path_train='D:/projects/bicycle/bike-sharing-demand/train.csv',
             file_path_test='D:/projects/bicycle/bike-sharing-demand/test.csv', model_expect = "LR", format_="csv", delim_="NA", provider="jdbc:teradata",
server_name="://TDTEST02",
DBCName="/DBNAME",
Database="/myDataBase",
Logmech="/LDAP",
driver="com.teradata.jdbc.TeraDriver",
Uid="myUsername",
Pwd="myPassword",maxIter=60, regParam=0.3, elasticNetParam=0.8,spark=""):
    #defining functions
    evaluator = lambda x: RegressionEvaluator(predictionCol="prediction",labelCol="count",metricName="r2").evaluate(lrpredictions)
        
    def spark_read(path, format__, delim__):
        if format__ in ["csv","tsv","psv", "txt"]:
            return spark.read.load(path, format=format__, sep=delim__, inferSchema="true", header="true")
        elif format__ == "avro":
            return spark.read.format("avro").load(path)
        elif format__ == "json":
            return spark.read.json(path)
        elif format__ == "parquet":
            return spark.read.parquet(path)
        elif format__ == "orc":
            return spark.read.orc(path)
        elif format__ == "sequence":
            return sc.sequenceFile(path).toDF()
        elif format__.lower() == "teradata": 
            return spark.read \
                        .format("td") \
                        .option("provider",provider) \
                        .option("server_name",server_name) \
                        .option("DBCName",DBCName) \
                        .option("Database",Database) \
                        .option("Logmech",Logmech) \
                        .option("driver",driver) \
                        .option("Uid",Uid) \
                        .option("Pwd",Pwd) \
                        .load()   
    
    def generate_rownumber(df_):
        df_ = df_.withColumn("new_column",lit("ABC"))
        w = Window().partitionBy('new_column').orderBy(lit('A'))
        df_ = df_.withColumn("row_num", row_number().over(w)).drop("new_column")
        return df_
    intersection = lambda l1, l2: [value for value in l1 if value in l2] 
    def convert_nonInt_string(spark_df_):
            strFeature = [ elem[0] for elem in spark_df_.dtypes if elem[1] not in ["bigint","double", "int"]]
            rest = list(set(spark_df_.columns) - set(strFeature))
            strFeature = [ "cast({} as string)".format(elem) for elem in strFeature ]
            return spark_df_.selectExpr(rest+strFeature)
    
    def remove_nonintCols(spark_df_):
        strFeature = [ elem[0] for elem in spark_df_.dtypes if elem[1] not in ["bigint","double", "int"]]
        indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(spark_df_) for column in strFeature ]
        pipeline = Pipeline(stages=indexers)
        df_r_ = pipeline.fit(spark_df_).transform(spark_df_)
        dropFeature = strFeature
        featuresToKeep = list(set(df_r_.columns) - set(dropFeature))
        df_r_ = df_r_.select(featuresToKeep)
        return df_r_
    # Starting the main program
   #reading dataframes train and test
    spark_df = spark_read(file_path_train, format_, delim_)
    test2 = spark_read(file_path_test, format_, delim_)
    targetCol = spark_df.columns[-1]
    training_raw_features = spark_df.columns[:-1]
    commonCols = intersection(training_raw_features, test2.columns)
    print( "Common Columns: ", commonCols)
    spark_df=spark_df.select(commonCols+[targetCol])
    test2 = test2.select(commonCols)
    
    #Computation started
    spark_df = convert_nonInt_string(spark_df)
    df_r_index = remove_nonintCols(spark_df)
    a = df_r_index.columns
    a.remove(targetCol)
    assembler = VectorAssembler(inputCols=a, outputCol='inputs')
    spark_df1 = assembler.transform(df_r_index)
    print ("Assembled columns {} to vector column features".format("|".join(a)))
    output=spark_df1.select('inputs',targetCol)
    splits=output.randomSplit([0.7, 0.3], seed=1000)
    train_df = splits[0]
    test_df = splits[1]
    if model_expect=="LR":
        reg_model = LinearRegression(featuresCol = 'inputs', labelCol=targetCol, maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam)
    elif model_expect=="DT":
        reg_model = DecisionTreeRegressor(featuresCol ='inputs', labelCol = 'countLog')
    elif model_expect=="GBT":
        reg_model = GBTRegressor(featuresCol = 'inputs', labelCol = 'countLog', maxIter=60)
    else:
        print ("model not found")
        sys.exit(0)
        mymodel = reg_model.fit(train_df)
        trainingSummary = mymodel.summary
        print ("numIterations: %d" % trainingSummary.totalIterations)
        print ("RMSE on training train data: %f" % trainingSummary.rootMeanSquaredError)
        print ("R Squared (R2) on training train data = %f" % trainingSummary.r2)
        mypredictions = mymodel.transform(test_df.select('inputs',targetCol))
        print ("R Squared (R2) on training test data = %g" % evaluator(mypredictions))
        test_result = mymodel.evaluate(test_df)
        print ("Root Mean Squared Error (RMSE) on training test data = %g" % test_result.rootMeanSquaredError)
        print ("numIterations: %d" % trainingSummary.totalIterations)
        print ("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
        test2 = convert_nonInt_string(test2)
        test1_index = remove_nonintCols(test2)
        feature_test = test1_index.columns
        print ("Assembled columns {} to vector column features".format("|".join(feature_test)))
        assembler_new = VectorAssembler(inputCols=feature_test, outputCol='inputs')
        test_df1 = assembler_new.transform(test1_index)
        mypredictions_new = mymodel.transform(test_df1)
        
        #generating row num column:
        mypredictions_new = generate_rownumber(mypredictions_new).selectExpr("row_num","ceil(prediction) as prediction")
        test2 = generate_rownumber(test2)
        mypredictions_new.registerTempTable("mypredictions_new")
        test2.registerTempTable("test2")
#         print spark.sql("select a.*,prediction from test2 a inner join lrpredictions_new b on a.row_num=b.row_num").show()
        test2 = test2.alias('a').join(mypredictions_new.alias('b'), test2.row_num == mypredictions_new.row_num, "inner").selectExpr("a.*","prediction").drop("row_num")
#         print(test2.show())
        working_dir = re.search(".*\/",file_path_test).group(0)[1:]
        working_dir = "./"
        print ("working directory is : {}".format(working_dir))
        if model_expect=="LR":
            test2.write.format("csv").save(working_dir+'lr_model.csv')
            print("before model save")
            lrmodel.mode("overwrite").save(working_dir+"lr_model")
        elif model_expect=="DT":
            test2.write.format("csv").save(working_dir+'dt_model.csv')
            print("before model save")
            lrmodel.mode("overwrite").save(working_dir+"dt_model")
        elif model_expect=="GBT":
            test2.write.format("csv").save(working_dir+'gbt_model.csv')
            print("before model save")
            lrmodel.mode("overwrite").save(working_dir+"gbt_model")
        else:
            pass
        spark.stop()
        return my_joined,mymodel
        

try:
    if len(sys.argv) < 3: 
        print("Usage: python internal_function_bicycle_v1.py D:/projects/bicycle/bike-sharing-demand/train.csv D:/projects/bicycle/bike-sharing-demand/test.csv LR csv ,")
        sys.exit(-1)
    else:
        print ("printing sparksession else block")
        spark = SparkSession.builder.appName('Model Training').enableHiveSupport().getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        spark.conf.set("spark.sql.execution.arrow.enabled", "true")
        spark.conf.set("spark.sql.execution.arrow.fallback.enabled", "true")
        our_main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],spark=spark)
    
except Exception as e:
    f=open("./error.txt", 'w+')
    f.write("Error in file"+ str(e))
    f.close()
    print("stopping spark object")
    spark.stop()
    print (traceback.print_exc())