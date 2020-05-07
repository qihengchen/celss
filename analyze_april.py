import pandas as pd 
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import ceil, floor
from sys import argv


def group(df, dvmin, dvmax, step):
	""" dataframe, int, int, int -> list
	Helper function. Calculate the probability of choosing left for each bucket, and return them in a list.
	"""
	r = step/2
	res = []

	for ticker in range(dvmin, dvmax, step):
		#select values by left-right  difference in sum in range (x-r, x+r). x is the middle value of a bucket. 
		subgroup = df.loc[(df['diff']>ticker-r) & (df['diff']<ticker+r)
			& (df['choice'] != 0.5)]
		#count frequency of choosing left
		num = subgroup['choice'].sum()
		#total number of datapoints in the bucket
		denom = subgroup.shape[0]
		#calculate and append the prob. append 0 if empty bucket
		res.append(num/denom) if denom else res.append(0)
	return res


def plot_figure3(df, left_colNames, right_colNames, dvmin=-100, dvmax=101, step=20, show=False, write_to=None):
	"""
	dataframe, list, list, int, int, int, bool, str -> np.array, np.array
	"""
	#select left values and right values
	left = df[left_colNames]
	right = df[right_colNames]
	# 'diff' column: sum(left) - sum(right)
	lr_diff = pd.DataFrame({'diff': left.sum(axis=1)-right.sum(axis=1)})
	# 'side_chosen': 0 left, 1 right 
	# 'choice': 1 chose left, 0 chose right. Flipped 'side_chosen' column, b/c we will count the frequency of choosing left
	lr_diff['choice'] = np.logical_xor(df['side_chosen'],1).astype(int)
	# sort diff from lowest to highest, and group into buckets
	lr_diff = lr_diff.sort_values(by=['diff'], ascending=True)
	#array of prob(choosing left) for each bucket
	grouped = group(lr_diff, dvmin, dvmax, step)

	#plot data
	y = np.array(grouped)
	x = np.array([x for x in range(dvmin, dvmax, step)])
	if show:
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=x, y=y, name="linear", line_shape='linear'))
		fig.show()
	if write_to is not None:
		fig.write_image(write_to)
	return x, y

def plot_figure3_subplots(df, left_colNames, right_colNames, dvmin=-100, dvmax=101, step=20, show=False, write_to=None):
	fig = make_subplots(rows=2, cols=3, subplot_titles=('sim-numbers', 'alt-numbers', 'seq-numbers', 'sim-bars', 'alt-bars', 'seq-bars'))
	x = np.array([x for x in range(dvmin, dvmax, step)])
	
	#call plot_figure3() for each block in the order of 'sim-numbers', 'alt-numbers', 'seq-numbers', 'sim-bars', 'alt-bars', 'seq-bars'
	for i in df['block_progressive_ID'].unique():
		block_df = df[df['block_progressive_ID']==i]
		x, y = plot_figure3(block_df, left_colNames, right_colNames, -100, 101, 20, show=False)
		row = int((i-1)/3+1)
		col = int((i-1)%3+1)
		fig.add_trace(go.Scatter(x=x, y=y, line_shape='linear', name=str(i), line=dict(color='blue')), row=row, col=col)

	if show:
		fig.show()
	if write_to is not None:
		fig.write_image(write_to)


# FIGURE FOUR A
def plot_figure4(df, left_colNames, right_colNames, dvmin=-60, dvmax=61, step=10, show=False, write_to=None):
	"""
	dataframe, list, list, int, int, int, bool, str -> list, list
	"""
	left = df[left_colNames]
	right = df[right_colNames]
	left.rename(columns=dict(zip(left_colNames, range(1,7))), inplace=True)
	right.rename(columns=dict(zip(right_colNames, range(1,7))), inplace=True)

	# item-wise matrix operation, left minus right. positive if left value is greater
	diff = left.combine(right, lambda l, r: l-r) 
	# count how many times left >= right
	diff['count'] = diff.gt(0).sum(axis=1)
	# sum(left) - sum(right)
	diff['diff'] = left.sum(axis=1)-right.sum(axis=1)
	# 1 chose left, 0 chose right. Flipped 'side_chosen' column
	diff['choice'] = np.logical_xor(df['side_chosen'],1).astype(int)
	diff = diff.sort_values(by=['diff'], ascending=True)

	# Fig 4A Blue line
	# calculate prob(choose left) for each bucket when left >= right MORE than three times 
	# left win times: 6/0, 5/1, 4/2. tie 3/3 ignored
	wgt3 = diff.loc[diff['count'] > 3]
	wgt3_res = group(wgt3, dvmin, dvmax, step)

	# Fig 4A Red line
	# calculate prob(choose left) for each bucket when left >= right FEWER than three times 
	# left win times: 0/6, 1/5, 2/4. tie 3/3 ignored
	wlt3 = diff.loc[diff['count'] < 3]
	wlt3_res = group(wlt3, dvmin, dvmax, step)

	if show:
		x = np.array([x for x in range(dvmin, dvmax, step)])
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=x, y=wgt3_res, mode='lines', name='W>3', line=dict(color='blue')))
		fig.add_trace(go.Scatter(x=x, y=wlt3_res, mode='lines', name='W<3', line=dict(color='red')))
		fig.show()
	if write_to is not None:
		fig.write_image(write_to)
	return wgt3_res, wlt3_res


def plot_figure4_subplots(df, left_colNames, right_colNames, dvmin=-60, dvmax=61, step=10, show=False, write_to=None):
	fig = make_subplots(rows=2, cols=3, subplot_titles=('sim-numbers', 'alt-numbers', 'seq-numbers', 'sim-bars', 'alt-bars', 'seq-bars'))
	x = np.array([x for x in range(dvmin, dvmax, step)])
	
	# #call plot_figure4() for each block in the order of 'sim-numbers', 'alt-numbers', 'seq-numbers', 'sim-bars', 'alt-bars', 'seq-bars'
	for i in df['block_progressive_ID'].unique():
		block_df = df[df['block_progressive_ID']==i]
		wgt3_res, wlt3_res = plot_figure4(block_df, left_colNames, right_colNames, -60, 61, 10, show=False)
		row = int((i-1)/3+1)
		col = int((i-1)%3+1)
		fig.add_trace(go.Scatter(x=x, y=wgt3_res, mode='lines', name='W>3', line=dict(color='blue')), row=row, col=col)
		fig.add_trace(go.Scatter(x=x, y=wlt3_res, mode='lines', name='W<3', line=dict(color='red')), row=row, col=col)

	if show:
		fig.show()
	if write_to is not None:
		fig.write_image(write_to)


if __name__ == '__main__':
	#plot_dir = argv[1]
	#data = argv[2]
	plot_dir = "/Users/qiheng/Desktop/code/celss/plots/"
	data = '/Users/qiheng/Desktop/code/celss/Averaging_folder/all_files_averaging/averaging_online_files/1_transform/stacked_file.csv'

	df = pd.read_csv(data, header=0)	
	df.stack()
	df = df.set_index(['participant_ID', 'trial_ID']) #panel data. The indices are participant_ID and trial_ID.

	#list of column names for left and right values
	left_colNames = ['value_left_'+str(i) for i in range(1,7)]
	right_colNames = ['value_right_'+str(i) for i in range(1,7)]

	#toggle the comments to switch b/t analyzing the six trial blocks in aggregate or individually
	#range from -100 to 101 (not including 101), bucket range 20. So we have 10 buckets. 
	#plot_figure3(df, left_colNames, right_colNames, -100, 101, 20, show=True, write_to=plot_dir+"fig3.png")
	plot_figure3_subplots(df, left_colNames, right_colNames, -100, 101, 20, show=True, write_to=plot_dir+"fig3_subplots.png")
	#range from -60 to 61 (not including 61), bucket range 10. So we have 12 buckets. 
	#plot_figure_four(df, left_colNames, right_colNames, -60, 61, 10, show=True, write_to=plot_dir+"fig4.png")
	plot_figure4_subplots(df, left_colNames, right_colNames, -60, 61, 10, show=True, write_to=plot_dir+"fig4_subplots.png")


