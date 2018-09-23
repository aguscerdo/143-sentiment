# Bunch of imports (may need more)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as func
# Initialize two logistic regression models.
# Replace labelCol with the column containing the label, and featuresCol with the column containing the features.

def regression(df, column, name):
    try:
        model = CrossValidatorModel.load("data/{}.model".format(name))
    except:

        LR = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
        if name[3] == 'P':
            LR.setThreshold(0.2)
        else:
            LR.setThreshold(0.25)

        eval = BinaryClassificationEvaluator()
        paramGrid = ParamGridBuilder().addGrid(LR.regParam, [1.0]).build()
        crossval = CrossValidator(
            estimator=LR,
            evaluator=eval,
            estimatorParamMaps=paramGrid,
            numFolds=5)
        # train, test = df.select('features', func.col(column).alias("label")).randomSplit([0.5, 0.5])
        print("Training '{}' classifier... Please wait".format(name))

        model = crossval.fit(df.select("*", func.col(column).alias("label")))
        model.save("data/{}.model".format(name))
    # df_test = model.transform(df)
    # df_test.filter(df_test.prediction == 1).show()
    return model


def predict_df(df, model, name):
    cols = [col for col in df.columns]
    cols.append(func.col("prediction").alias("{}_pred".format(name)))

    return model.transform(df).select(cols)
