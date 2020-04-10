import pandas as pd 
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go


def group(df, dvmin, dvmax, step):
	r = step/2
	res = []

	for ticker in range(dvmin, dvmax, step):
		subgroup = df.loc[(df['diff']>ticker-r) & (df['diff']<ticker+r)
			& (df['choice'] != 0.5)]
		num = subgroup['choice'].sum()
		denom = subgroup.shape[0]
		if denom:
			print(int(ticker-r), int(ticker+r), int(num), denom, num/denom)
		else:
			print('denom 0 \n')
		res.append(num/denom) if denom else res.append(0)
	return res

# stacked_df, left values, right values, 
def plot_figure_three(df, left_colNames, right_colNames, dvmin=-100, dvmax=101, step=20):
	"""
	dataframe, list, list, int, int, int -> None
	"""
	left = df[left_colNames]
	right = df[right_colNames]
	lr_diff = pd.DataFrame({'diff': left.sum(axis=1)-right.sum(axis=1)})
	lr_diff['choice'] = df[['side_chosen']]
	lr_diff = lr_diff.sort_values(by=['diff'], ascending=True)

	grouped = group(lr_diff, -100, 101, 20)

	y = np.array(grouped)
	x = np.array([x for x in range(-100, 101, 20)])

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=x, y=y, name="linear", line_shape='linear'))
	fig.show()


# FIGURE FOUR A
def plot_figure_four(df, left_colNames, right_colNames, dvmin=-60, dvmax=61, step=10):
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
	diff['choice'] = df[['side_chosen']]

	diff = diff.sort_values(by=['diff'], ascending=True)

	print('Fig 4A Blue line')
	wgt3 = diff.loc[diff['count'] > 3]
	wgt3_res = group(wgt3, dvmin, dvmax, step)
	#wgt3['choice'] = wgt3['choice'].cumsum(axis=0)
	#wgt3['prob'] = wgt3['choice']/(wgt3.shape[0])
	#wgt3 = wgt3.loc[(wgt3['diff']>=-60) & (wgt3['diff']<=60)]
	print('\nFig 4A Red line')
	wlt3 = diff.loc[diff['count'] < 3]
	wlt3_res = group(wlt3, dvmin, dvmax, step)
	print(wlt3)

	x = np.array([x for x in range(dvmin, dvmax, step)])
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=x, y=wgt3_res, mode='lines', name='W>3', line=dict(color='blue')))
	fig.add_trace(go.Scatter(x=x, y=wlt3_res, mode='lines', name='W<3', line=dict(color='red')))
	fig.show()


if __name__ == '__main__':
	#df = pd.read_csv('/Users/qiheng/Desktop/code/celss/Averaging_folder/video_averaging/new_video_aver.csv', header=0)
	df = pd.read_csv('/Users/qiheng/Desktop/code/celss/Averaging_folder/all_files_averaging/averaging_online_files/1_transform/stacked_file.csv', header=0)
	
	df.stack()
	df = df.set_index(['participant_ID', 'trial_ID'])

	print(df.head())
	print(df[['side_chosen']])

	left_colNames = ['setup_a'+str(i) for i in range(1,7)]
	right_colNames = ['setup_b'+str(i) for i in range(1,7)]
	
	plot_figure_three(df, left_colNames, right_colNames, -100, 101, 20)

	plot_figure_four(df, left_colNames, right_colNames, -60, 61, 10)







