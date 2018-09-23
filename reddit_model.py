from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

# IMPORT OTHER MODULES HERE
from pyspark.sql.functions import UserDefinedFunction  ## need these for import
from pyspark.sql.types import ArrayType
from pyspark.sql.types import StringType

from pyspark.sql.functions import udf
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel

import cleantext as cln
import spark_regression
import analysis


# --- Functions --- #
def sanitizeX(s):
    w = cln.sanitize(s)  # sanitize takes a string (commment) and returns a list
    return w[1].split(' ') + w[2].split(' ') + w[3].split(' ')


def read_comments_minimal(sc):
    try:
        df = sc.read.parquet("data/comments-minimal.parquet")
    except:
        print("\tReading fresh")
        df0 = sc.read.json("data/comments-minimal.json")
        df0.createOrReplaceTempView("df")
        Q = "SELECT id, body, author, author_flair_text, created_utc, score, SUBSTRING(link_id,4,10) as link_id, SUBSTRING(parent_id,4,10) as parent_id " \
            " FROM df"
        Q += " WHERE body NOT LIKE '%/s%' AND body NOT LIKE '%&gt%'"
        df = sc.sql(Q)
        df.write.parquet("data/comments-minimal.parquet")
    return df


def read_submission(sc):
    try:
        df = sc.read.parquet("data/submissions.parquet")
    except:
        print("\tReading fresh")
        df = sc.read.json("data/submissions.json")
        df.write.parquet("data/submissions.parquet")
    return df


def read_csv(name, sc):
    try:
        df = sc.read.parquet("{}.parquet".format(name))
    except:
        df = sc.read.format("csv").option("header", "true").load(name)
        df.write.parquet("{}.parquet".format(name))
    return df


def retrieve_labeled_comments(df_c, df_l, sc):
    try:
        df = sc.read.parquet("data/comments_labeled.parquet")
    except:
        df = df_c.join(df_l, df_c.id == df_l.Input_id).select(df_c.id, df_c.body, df_l.labeldem, df_l.labelgop, df_l.labeldjt)#.collect()
        df.write.parquet("data/comments_labeled.parquet")
    return df


def clean_df(df_cl, udf, sc, name="data/clean.parquet"):
    try:
        df = sc.read.parquet(name)
    except:
        df = df_cl.select('id', 'labeldem', 'labelgop', 'labeldjt', udf('body').alias('text'))
        df.write.parquet(name)
    return df



def createCV(df, sc, col_name='text'):
    cv = CountVectorizer(inputCol=col_name, outputCol="features", minDF=5)
    try:
        df_r = sc.read.parquet("data/vectorized.parquet")
        cv = CountVectorizer.load("data/vectorized_cv.parquet")
        model = CountVectorizerModel.load("data/vectorized_model.parquet")
    except:
        model = cv.fit(df)
        df_r = model.transform(df)
        df_r.write.parquet("data/vectorized.parquet")
        cv.save("data/vectorized_cv.parquet")
        model.save("data/vectorized_model.parquet")
    return (df_r, model, cv)


def binary_label(df, sc, name):
    try:
        df_r = sc.read.parquet("data/{}.parquet".format(name))
    except:
        df.createOrReplaceTempView("df")
        Q = "SELECT id, features, IF(labeldem=1, 1, 0) as demP, IF(labeldem=-1, 1, 0) as demN, "
        Q += "IF(labelgop=1, 1, 0) as gopP, IF(labelgop=-1, 1, 0) as gopN, "
        Q += "IF(labeldjt=1, 1, 0) as djtP, IF(labeldjt=-1, 1, 0) as djtN "
        Q += "FROM df"
        # Q += " WHERE body NOT LIKE '%/s%' AND body NOT LIKE '&gt%'"

        df_r = sc.sql(Q)
        df_r.write.parquet("data/{}.parquet".format(name))
    return df_r


def run_full(df, sc, udf, cv,  models, end_mode=False):
    try:
        df_r = sc.read.parquet("data/full_comments.parquet")
    except:
        if end_mode:
            raise Exception("End Mode reload")
        df_r = df.select("*",udf('body').alias('text'))
        df_r = cv.transform(df_r)
        df_r.printSchema()

        for key in models:
            print("Evaluating model %s" % key)
            df_r = spark_regression.predict_df(df_r, models[key], key)
            df_r.printSchema()
            # df_r = df_r.join(df_t, "id")
        df_r = df_r.drop("body").drop("features").drop("text")
        df_r.printSchema()
        df_r.write.parquet("data/full_comments.parquet")
    return df_r


def map_wrap_to_pandas(df, sc, tag, diff_tag=""):
    # Saves the data to CSV to evaluate on R
    ttag = tag + "_pred"
    diff_ttag = diff_tag + "_pred"

    states = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado",
  "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois",
  "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
  "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
  "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
  "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
  "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
  "Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]

    df.createOrReplaceTempView("df")
    if not diff_tag:
        Q = "SELECT author_flair_text as state, AVG({}) as sentiment FROM df WHERE author_flair_text is not NULL GROUP BY author_flair_text".format(ttag)
        T = "plots/map_data_{}.csv".format(tag)
    else:
        Q = "SELECT author_flair_text as state, ABS(AVG({})-AVG({})) as sentiment FROM df WHERE author_flair_text is not NULL GROUP BY author_flair_text".format(ttag, diff_ttag)
        T = "plots/map_data_diff_{}-{}.csv".format(tag, diff_tag)
    df_pd = sc.sql(Q).toPandas()
    df_pd = df_pd[df_pd['state'].isin(states)]
    df_pd.to_csv(T)

    # NOTE: Here we would call the plotter, but couldn't make it work. Instead we just saved the csvs
    # analysis.map_plot(df_pd, sc, tag)


# ---------------- MAIN ----------------
def main(context):
    """Main function takes a Spark SQL context."""

    # --- User defined functions ---
    try:
        print("Attempting to load full dataset...")
        tags = ["demP", "demN", "gopP", "gopN", "djtP", "djtN"]
        df_full = run_full(0,context,0,0,0, True)

        df_sub = read_submission(context)
    except:
        sanitize = udf(sanitizeX, ArrayType(StringType()))

        # --- Read Files --- #
        print("Loading Files")
        df_comm = read_comments_minimal(context)
        df_sub = read_submission(context)
        df_lab = read_csv("data/labeled_data.csv", context)

        # # # --- Retrieving labeled comments --- #
        print("Retrieving labeled comments")
        df_c_lab = retrieve_labeled_comments(df_comm, df_lab, context)

        # # # --- Clean --- #
        print("Sanitizing labeled")
        df_clean = clean_df(df_c_lab, sanitize, context)

        # # --- Vectorize --- #
        print("Vectorizing labeled")
        df_vector, CVmodel, count_v = createCV(df_clean, context)

        # # --- Binary Labeling --- #
        print("Setting binary labels")
        df_labeled_training = binary_label(df_vector, context, name="training_labeled")

        # # --- Regression Training --- #
        print("Training Regression model")
        tags = ["demP", "demN", "gopP", "gopN", "djtP", "djtN"]
        models = {}
        for t in tags:
            models[t] = spark_regression.regression(df_labeled_training, t, t)

        # --- Run on full file --- #
        print("Running on full set")
        df_full = run_full(df_comm, context, sanitize, CVmodel, models)

    # Top stories and map data over time
    for tag in tags:
        analysis.top_stories(df_full, df_sub, context, tag)
        analysis.top_stories(df_full, df_sub, context, tag, 10)
        map_wrap_to_pandas(df_full, context, tag)

    # Scatter, sentiment, map data
    for t in [["demP", "demN"], ["gopP", "gopN"], ["djtP", "djtN"]]:
        map_wrap_to_pandas(df_full, context, t[0], t[1])
        analysis.sentiment_over_time(df_full, context, t[0], t[1])
        analysis.scatter(df_full, df_sub, context, t[0], t[1], 1)
        analysis.scatter(df_full, df_sub, context, t[0], t[1], 100)

    # Total Republican Scatter
    analysis.total_scatter(df_full, df_sub, context)




# ---------------- END ----------------
if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    sc.setLogLevel("ERROR")

    main(sqlContext)
