import pandas as pd 
import numpy as np 
from os import listdir
from os.path import isfile, join
import sys


def step1(df):
	block_type = {'sim-numbers':1, 'alt-numbers':2, 'seq-numbers':3, 'sim-bars':4, 'alt-bars':5, 'seq-bars':6}
	treatment = {'up':1, 'down':2}
	start_position = {'a_on_left':1, 'a_on_right':2, 'b_on_left':3, 'b_on_right':4}
	side_chosen = {'left':0, 'right':1}
	choice_correct = {False:0, True:1}
	rules = {'block_type':block_type, 'treatment':treatment, 'start_position':start_position, 
		'side_chosen':side_chosen, 'choice_correct':choice_correct, 'choice_correct.1':choice_correct}
	for col, rule in rules.items():
		for k, v in rule.items():
			df.loc[df[col]==k, col] = v
	return df 

def step2(df, ID):
	df['participant_ID'] = ID

	row_num = df.shape[0]
	if row_num == 180 or row_num == 300: # 180 or 300
		df['block_progressive_ID'] = [i for i in range(1,7) for _ in range(int(row_num/6))]
		df['trial_ID'] = [x for x in range(1, row_num+1)]
		df['trial_ID_block'] = [x for x in range(1, int(row_num/6 +1))] * 6 
	else:
		df['block_progressive_ID'] = 0
		df['trial_ID'] = 0
		df['trial_ID_block'] = 0 

	df['block_order'] = df['block_type'] % 3 # based on block_type: 1/4->1 sim; 2/5->2 alt; 3/6->3 seq
	df['block_stimuli'] = 1
	df.loc[df['block_type'] >= 4, 'block_stimuli'] = 2 # OR df.loc[df['???'].str.contains('bar', 'block_stimuli'] = 2
	
	df['left-0-right'] = df['start_position'].copy() # based on 'start_position', 1/3->1 left; 2/4->2 right; 0->0
	df.loc[df['left-0-right'] == 3, 'left-0-right'] = 1
	df.loc[df['left-0-right'] == 4, 'left-0-right'] = 2

	# start_position - 'a_on_right':2, 'b_on_left':3
	a_columns = ['setup_a'+str(i) for i in range(1,7)]
	b_columns = ['setup_b'+str(i) for i in range(1,7)]
	should_swap = df['start_position'].isin([2, 3])
	df.loc[should_swap, a_columns+b_columns] = df.loc[should_swap, b_columns+a_columns].values
	return df


if __name__ == '__main__':
	input_path = "/Users/qiheng/Desktop/code/celss/Averaging_folder/all_files_averaging/averaging_online_files/1_transform_copy"
	output_path = "/Users/qiheng/Desktop/code/celss/Averaging_folder/all_files_averaging/averaging_online_files/1_transform"
	
	#input_path = "/Users/qiheng/Desktop/code/celss/Averaging_folder/video_averaging"
	#output_path = "/Users/qiheng/Desktop/code/celss/Averaging_folder/video_averaging"
	
	files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and not f.startswith('.')]
	stacked_df = None
	ID = 1


	for f in files:
		df = pd.read_csv(join(input_path, f), skip_blank_lines=True, header=0)
		df = df.iloc[:df.shape[0]-7]
		print(f)
		df = step1(df)
		df = step2(df, ID)
		#df.to_csv(join(output_path, 'new_video_aver.csv'), index=True)
		df.to_csv(join(output_path, f), index=True)
		if stacked_df is None:
			stacked_df = df
		else:
			stacked_df = stacked_df.append(df)
		ID += 1

	stacked_df.to_csv(join(output_path, 'stacked_file.csv'), index=True)

