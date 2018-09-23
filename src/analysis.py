from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np

from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon

def sentiment_over_time(df, sc, tag_p, tag_n):
	ttag = tag_p + "_pred"
	ttag2 = tag_n + "_pred"
	df.createOrReplaceTempView("df")
	Q = "SELECT DATE(FROM_UNIXTIME(created_utc)) AS date, AVG({}) AS sentiment_p, AVG({}) as sentiment_n FROM df GROUP BY date SORT BY date ASC".format(ttag, ttag2)

	ts = sc.sql(Q).toPandas().sort_values("date")	# TODO test
	plt.figure(figsize=(30,30))
	# ts = ts[ts['created_utc'] != '2018-12-31']
	ts.date = pd.to_datetime(ts['date'], format='%Y-%m-%d')
	ts.set_index(['date'],inplace=True)

	ax = ts.plot(title="Sentiment on r/politics over time for {}".format(tag_p[0:3]),
		color=['green', 'red'],
	   ylim=(0, 1.05))
	# fig = ax.get_figure()
	plt.tight_layout()
	plt.savefig("plots/sentiment-time-{}.png".format(tag_p[:3]))
	plt.close()


# TODO I tried, but I couldn't install the map toolkit. SO we used R for that.
def map_plot(state_data, sc, tag, diff_tag=""):
	m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
		projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
	shp_info = m.readshapefile('st99_d00','states',drawbounds=True)  # No extension specified in path here.
	pos_data = dict(zip(state_data.state, state_data.sentiment))

	# choose a color for each state based on sentiment.
	pos_colors = {}
	statenames = []
	pos_cmap = plt.cm.Greens # use 'hot' colormap

	vmin = 0; vmax = 1 # set range.
	for shapedict in m.states_info:
		statename = shapedict['NAME']
		# skip DC and Puerto Rico.
		if statename not in ['District of Columbia', 'Puerto Rico']:
			pos = pos_data[statename]
			pos_colors[statename] = pos_cmap(1. - np.sqrt(( pos - vmin )/( vmax - vmin)))[:3]
		statenames.append(statename)
	# cycle through state names, color each one.

	# POSITIVE MAP
	ax = plt.gca() # get current axes instance
	for nshape, seg in enumerate(m.states):
		# skip Puerto Rico and DC
		if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
			color = rgb2hex(pos_colors[statenames[nshape]])
			poly = Polygon(seg, facecolor=color, edgecolor=color)
			ax.add_patch(poly)

	if not diff_tag:
		plt.title('Sentiment across the United States for {}'.format(tag))
		plt.savefig("plots/map-{}.png".format(tag))
	else:
		plt.title('Sentiment difference across the United States for {} and {}'.format(tag, diff_tag))
		plt.savefig("plots/map-diff-{}-{}.png".format(tag, diff_tag))
	plt.close()


def top_stories(df_com, df_sub, sc, tag, min_score=1):
	ttag = tag + "_pred"
	df_com.createOrReplaceTempView("df_com")
	df_sub.createOrReplaceTempView("df_sub")
	# df_sub.printSchema()
	SQ = "SELECT link_id, AVG({}) as average FROM df_com GROUP BY link_id HAVING COUNT(link_id) > {} ORDER BY average DESC LIMIT 10".format(ttag, min_score)
	# TODO check name
	Q = "SELECT L.title, R.average FROM df_sub as L INNER JOIN ({}) as R ON L.id = R.link_id".format(SQ)

	df = sc.sql(Q)
	print("--- {} ---".format(tag))
	df.show()
	df.toPandas().to_csv("plots/top_{}-{}.csv".format(min_score, tag))


def scatter(df_com, df_sub, sc, tag_pos, tag_neg, group_num=1):
	ttag_pos = tag_pos + "_pred"
	ttag_neg = tag_neg + "_pred"
	df_com.createOrReplaceTempView("df_com")
	df_sub.createOrReplaceTempView("df_sub")

	# ---- Q1 Comment Score ---- #
	Q1 = "SELECT score/{} as score, AVG({}) as avg_pos, AVG({}) as avg_neg FROM df_com GROUP BY score/{}".format(group_num, ttag_pos, ttag_neg, group_num) # HAVING COUNT(score) > 10 (?)

	df1 = sc.sql(Q1).toPandas()
	df1['score'] *= group_num
	plt.scatter(df1['score'], df1['avg_pos'], c='r', marker='o', label="Positive")
	plt.scatter(df1['score'], df1['avg_neg'], c='b', marker='s', label="Negative")
	plt.title("Positive and Negative percentage by Comment Score for {}-{}".format(tag_pos, tag_neg))
	plt.legend(loc='lower right')
	# plt.xscale('log')
	# plt.tight_layout()
	plt.savefig("plots/scatter_com_{}_{}-{}.png".format(group_num, tag_pos, tag_neg))
	plt.close()

	# --- Q2 Sub score --- #
	SQ2 = "SELECT link_id, AVG({}) as avg_pos, AVG({}) as avg_neg FROM df_com GROUP BY link_id".format(ttag_pos,
																									   ttag_neg)
	Q2 = "SELECT L.score as score, R.avg_pos as avg_pos, R.avg_neg as avg_neg FROM df_sub as L INNER JOIN ({}) as R ON L.id = R.link_id".format(
		SQ2)
	df2 = sc.sql(Q2).toPandas()

	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.scatter(df2['score'], df2['avg_pos'], s=10, c='r', marker='o', label="Positive")
	ax1.scatter(df2['score'], df2['avg_neg'], s=10, c='b', marker='s', label="Negative")
	plt.title("Positive and Negative percentage by Submission Score for {}-{}".format(tag_pos, tag_neg))
	plt.legend(loc='lower right')

	plt.xlabel("Score")
	plt.ylabel("Percent")
	# ax1.('log')
	# plt.tight_layout()
	plt.savefig("plots/scatter_sub_{}-{}.png".format(tag_pos, tag_neg))
	plt.close()


def total_scatter(df_com, df_sub, sc):
	df_com.createOrReplaceTempView("df_com")
	df_sub.createOrReplaceTempView("df_sub")

	SQ = "SELECT link_id, AVG(djtP_pred+gopP_pred) as avg_pos, AVG(djtN_pred+gopN_pred) as avg_neg FROM df_com GROUP BY link_id"
	Q = "SELECT L.score as score, R.avg_pos as avg_pos, R.avg_neg as avg_neg FROM df_sub as L INNER JOIN ({}) as R ON L.id = R.link_id".format(SQ)

	# --- Q2 Sub score --- #
	df = sc.sql(Q).toPandas()
	plt.scatter(df['score'], df['avg_pos'], s=10, c='r', marker='o', label="Positive")
	plt.scatter(df['score'], df['avg_neg'], s=10, c='b', marker='s', label="Negative")
	plt.title("Overall Positive and Negative sentiment to Republicans by Submission Score")
	plt.legend(loc='lower right');
	plt.xlabel("Score")
	plt.ylabel("Percentage")
	plt.tight_layout()
	plt.savefig("plots/overall_scatter.png")
	plt.close()




