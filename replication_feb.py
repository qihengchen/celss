import pandas as pd 
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('./PriceObfuscation/table_task1.csv', header=None)
df.stack()
index = pd.MultiIndex.from_product(list(map(range, (51, 300))), names=['person', 'trial'])
df = df.set_index(index)

left = df.iloc[:, 3:9]
right = df.iloc[:, 9:15]

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


# FIGURE THREE
#print(left.loc[(1, slice(5,10)), :]) 
lr_diff = pd.DataFrame({
	'diff': left.sum(axis=1)-right.sum(axis=1), 
	'choice': df.iloc[:, 21]})
lr_diff = lr_diff.sort_values(by=['diff'], ascending=True)



res = group(lr_diff, -100, 101, 20)
y = np.array(res)
x = np.array([x for x in range(-100, 101, 20)])

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, name="linear", line_shape='linear'))
fig.show()


# FIGURE FOUR A
left.rename(columns=dict(zip(range(3,9), range(1,7))), inplace=True)
right.rename(columns=dict(zip(range(9,16), range(1,7))), inplace=True)
diff = left.combine(right, lambda l, r: l-r) # item-wise left minus right 
diff['count'] = diff.gt(0).sum(axis=1)

diff['diff'] = left.sum(axis=1)-right.sum(axis=1)
diff['choice'] = df.iloc[:, 21]
print(diff.head(20))

diff = diff.sort_values(by=['diff'], ascending=True)

print('Fig 4A Blue line')
wgt3 = diff.loc[diff['count'] > 3]
wgt3_res = group(wgt3, -60, 61, 10)
#wgt3['choice'] = wgt3['choice'].cumsum(axis=0)
#wgt3['prob'] = wgt3['choice']/(wgt3.shape[0])
#wgt3 = wgt3.loc[(wgt3['diff']>=-60) & (wgt3['diff']<=60)]
print('\nFig 4A Red line')
wlt3 = diff.loc[diff['count'] < 3]
wlt3_res = group(wlt3, -60, 61, 10)
print(wlt3)

x = np.array([x for x in range(-60, 61, 10)])
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=wgt3_res, mode='lines', name='W>3', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=x, y=wlt3_res, mode='lines', name='W<3', line=dict(color='red')))
fig.show()


# FIGURE FOUR B
eq3 = diff.loc[diff['count'] == 3]
eq3['abs_sum'] = 0
for i in range(1,7):
	eq3['abs_sum'] += eq3[i].abs()

wlt90 = eq3.loc[eq3['abs_sum'] <= 90]
wlt90_res = group(wlt90, -80, 81, 20)


wgt90 = eq3.loc[eq3['abs_sum'] > 90]
wgt90_res = group(wgt90, -80, 81, 20)

"""
x = np.array([x for x in range(-80, 81, 20)])
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=wlt90_res, mode='lines', name='abs_diff<=90', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=x, y=wgt90_res, mode='lines', name='abs_diff>90', line=dict(color='red')))
fig.show()
"""
# FIGURE 






