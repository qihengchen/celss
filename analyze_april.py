import pandas as pd 
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import ceil, floor


def group(df, dvmin, dvmax, step):
	r = step/2
	res = []

	for ticker in range(dvmin, dvmax, step):
		subgroup = df.loc[(df['diff']>ticker-r) & (df['diff']<ticker+r)
			& (df['choice'] != 0.5)]
		num = subgroup['choice'].sum()
		denom = subgroup.shape[0]
		"""
		if denom:
			print(int(ticker-r), int(ticker+r), int(num), denom, num/denom)
		else:
			print('denom 0 \n')
		"""
		res.append(num/denom) if denom else res.append(0)
	return res

# stacked_df, left values, right values, 
def plot_figure3(df, left_colNames, right_colNames, dvmin=-100, dvmax=101, step=20):
	"""
	dataframe, list, list, int, int, int -> None
	"""
	left = df[left_colNames]
	right = df[right_colNames]
	lr_diff = pd.DataFrame({'diff': left.sum(axis=1)-right.sum(axis=1)})
	lr_diff['choice'] = np.logical_xor(df['side_chosen'],1).astype(int)
	lr_diff = lr_diff.sort_values(by=['diff'], ascending=True)

	grouped = group(lr_diff, -100, 101, 20)

	y = np.array(grouped)
	x = np.array([x for x in range(-100, 101, 20)])

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=x, y=y, name="linear", line_shape='linear'))
	fig.show()


# FIGURE FOUR A
def plot_figure4(df, left_colNames, right_colNames, dvmin=-60, dvmax=61, step=10, show=True):
	"""
	dataframe, list, list -> None
	"""
	left = df[left_colNames]
	right = df[right_colNames]
	left.rename(columns=dict(zip(left_colNames, range(1,7))), inplace=True)
	right.rename(columns=dict(zip(right_colNames, range(1,7))), inplace=True)

	diff = left.combine(right, lambda l, r: l-r) # item-wise left minus right 
	diff['count'] = diff.gt(0).sum(axis=1)

	diff['diff'] = left.sum(axis=1)-right.sum(axis=1)
	diff['choice'] = np.logical_xor(df['side_chosen'],1).astype(int)

	diff = diff.sort_values(by=['diff'], ascending=True)

	print('Fig 4A Blue line')
	wgt3 = diff.loc[diff['count'] > 3]
	wgt3_res = group(wgt3, dvmin, dvmax, step)

	print('\nFig 4A Red line')
	wlt3 = diff.loc[diff['count'] < 3]
	wlt3_res = group(wlt3, dvmin, dvmax, step)
	#print(wlt3)

	if show:
		x = np.array([x for x in range(dvmin, dvmax, step)])
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=x, y=wgt3_res, mode='lines', name='W>3', line=dict(color='blue')))
		fig.add_trace(go.Scatter(x=x, y=wlt3_res, mode='lines', name='W<3', line=dict(color='red')))
		fig.show()

	return wgt3_res, wlt3_res


def plot_figure4_subplots(df, left_colNames, right_colNames, dvmin=-60, dvmax=61, step=10):
	fig = make_subplots(rows=2, cols=3, subplot_titles=('sim-numbers', 'alt-numbers', 'seq-numbers', 'sim-bars', 'alt-bars', 'seq-bars'))
	x = np.array([x for x in range(dvmin, dvmax, step)])
	"""
	for i in df['block_progressive_ID'].unique():
		block_df = df[df['block_progressive_ID']==i]
		print(block_df.shape)
		wgt3_res, wlt3_res = plot_figure4(block_df, left_colNames, right_colNames, -60, 61, 10, show=False)
		row = int(i/3.0)
		col = (i-1)%3+1
		fig.add_trace(go.Scatter(x=x, y=wgt3_res, mode='lines', name='W>3', line=dict(color='blue')), row=1, col=2)
		fig.add_trace(go.Scatter(x=x, y=wlt3_res, mode='lines', name='W<3', line=dict(color='red')), row=2, col=3)
	"""
	block_df = df[df['block_type']==1]
	wgt3_res, wlt3_res = plot_figure4(block_df, left_colNames, right_colNames, -60, 61, 10, show=False)
	fig.add_trace(go.Scatter(x=x, y=wgt3_res, mode='lines', name='W>3', line=dict(color='blue')), row=1, col=1)
	fig.add_trace(go.Scatter(x=x, y=wlt3_res, mode='lines', name='W<3', line=dict(color='red')), row=1, col=1)

	block_df = df[df['block_type']==2]
	wgt3_res, wlt3_res = plot_figure4(block_df, left_colNames, right_colNames, -60, 61, 10, show=False)
	fig.add_trace(go.Scatter(x=x, y=wgt3_res, mode='lines', name='W>3', line=dict(color='blue')), row=1, col=2)
	fig.add_trace(go.Scatter(x=x, y=wlt3_res, mode='lines', name='W<3', line=dict(color='red')), row=1, col=2)

	block_df = df[df['block_type']==3]
	wgt3_res, wlt3_res = plot_figure4(block_df, left_colNames, right_colNames, -60, 61, 10, show=False)
	fig.add_trace(go.Scatter(x=x, y=wgt3_res, mode='lines', name='W>3', line=dict(color='blue')), row=1, col=3)
	fig.add_trace(go.Scatter(x=x, y=wlt3_res, mode='lines', name='W<3', line=dict(color='red')), row=1, col=3)

	block_df = df[df['block_type']==4]
	wgt3_res, wlt3_res = plot_figure4(block_df, left_colNames, right_colNames, -60, 61, 10, show=False)
	fig.add_trace(go.Scatter(x=x, y=wgt3_res, mode='lines', name='W>3', line=dict(color='blue')), row=2, col=1)
	fig.add_trace(go.Scatter(x=x, y=wlt3_res, mode='lines', name='W<3', line=dict(color='red')), row=2, col=1)

	block_df = df[df['block_type']==5]
	wgt3_res, wlt3_res = plot_figure4(block_df, left_colNames, right_colNames, -60, 61, 10, show=False)
	fig.add_trace(go.Scatter(x=x, y=wgt3_res, mode='lines', name='W>3', line=dict(color='blue')), row=2, col=2)
	fig.add_trace(go.Scatter(x=x, y=wlt3_res, mode='lines', name='W<3', line=dict(color='red')), row=2, col=2)

	block_df = df[df['block_type']==6]
	wgt3_res, wlt3_res = plot_figure4(block_df, left_colNames, right_colNames, -60, 61, 10, show=False)
	fig.add_trace(go.Scatter(x=x, y=wgt3_res, mode='lines', name='W>3', line=dict(color='blue')), row=2, col=3)
	fig.add_trace(go.Scatter(x=x, y=wlt3_res, mode='lines', name='W<3', line=dict(color='red')), row=2, col=3)

	fig.update_layout(height=800, width=1200, title_text="Figure Four, Side By Side Subplots")
	fig.show()


if __name__ == '__main__':
	#df = pd.read_csv('/Users/qiheng/Desktop/code/celss/Averaging_folder/video_averaging/new_video_aver.csv', header=0)
	df = pd.read_csv('/Users/qiheng/Desktop/code/celss/Averaging_folder/all_files_averaging/averaging_online_files/1_transform/stacked_file.csv', header=0)
	
	df.stack()
	df = df.set_index(['participant_ID', 'trial_ID'])

	left_colNames = ['setup_a'+str(i) for i in range(1,7)]
	right_colNames = ['setup_b'+str(i) for i in range(1,7)]

	#plot_figure3(df, left_colNames, right_colNames, -100, 101, 20)

	#plot_figure_four(df, left_colNames, right_colNames, -60, 61, 10)
	plot_figure4_subplots(df, left_colNames, right_colNames, -60, 61, 10)






