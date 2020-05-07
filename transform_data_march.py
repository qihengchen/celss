import pandas as pd 
import numpy as np 
from os import listdir
from os.path import isfile, join
from sys import argv


def step1(df):
	#change texts/booleans to numbers
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
	if row_num == 180 or row_num == 300: # if 180 or 300 rows
		# 111..., 222..., 333..., 444..., 555..., 666...
		df['block_progressive_ID'] = [i for i in range(1,7) for _ in range(int(row_num/6))]
		# 1,2,3,... until 180/300
		df['trial_ID'] = [x for x in range(1, row_num+1)]
		# 1,2,3,... until 30/50, repeat six times
		df['trial_ID_in_block'] = [x for x in range(1, int(row_num/6 +1))] * 6 
	else:
		print("NOT 30 OR 50 ROWS " + str(row_num))
		df['block_progressive_ID'] = 0
		df['trial_ID'] = 0
		df['trial_ID_in_block'] = 0 

	# sim 1, alt 2, seq 3.  Converted from block_type: 1,4->1; 2,5->2; 3,6->3
	df['block_order'] = df['block_type'] % 3 + 1 
	# numbers 1, bars 2
	df['block_stimuli'] = 1
	df.loc[df['block_type'] >= 4, 'block_stimuli'] = 2
	# left 1, right 2, 0 is 0. Converted 'start_position' on 0/1/2/3/4 scale to 'left-0-right' on 0/1/2 scale. 1,3->1 left; 2,4->2 right; 0->0
	df['left-0-right'] = df['start_position'].copy() 
	df.loc[df['left-0-right'] == 3, 'left-0-right'] = 1
	df.loc[df['left-0-right'] == 4, 'left-0-right'] = 2

	# Corrected the a/b left/right messup. start_position - 'a_on_right':2, 'b_on_left':3
	a_columns = ['setup_a'+str(i) for i in range(1,7)]
	b_columns = ['setup_b'+str(i) for i in range(1,7)]
	should_swap = df['start_position'].isin([2, 3])
	df.loc[should_swap, a_columns+b_columns] = df.loc[should_swap, b_columns+a_columns].values
	df.drop(columns=['choice_correct.1'], inplace=True)
	
	#rename columns
	for i in range(1, 7):
		df['value_left_'+str(i)] = df['setup_a'+str(i)]
		df['value_right_'+str(i)] = df['setup_b'+str(i)]
	return df


if __name__ == '__main__':
	# if you prefer typing paths in command line: 
	# input_path = argv[1]
	# output_path = argv[2]
	input_path = "/Users/qiheng/Desktop/code/celss/Averaging_folder/all_files_averaging/averaging_online_files/1_transform_original_data"
	output_path = "/Users/qiheng/Desktop/code/celss/Averaging_folder/all_files_averaging/averaging_online_files/1_transform"

	cols = ['participant_ID', 'treatment', 'block_progressive_ID', 'trial_ID_in_block', 'left-0-right', 'block_type', 'block_order', 'block_stimuli', 'trial_ID', 'start_position', 'value_left_1', 'value_left_2', 'value_left_3', 'value_left_4', 'value_left_5', 'value_left_6',
		'value_right_1', 'value_right_2', 'value_right_3', 'value_right_4', 'value_right_5', 'value_right_6', 'total_left', 'total_right', 'waiting_elapsed', 'side_chosen', 'choice_correct', 
		'bonus_block_type', 'bonus_trial_number_chosen', 'bonus_trial_correct', 'bonus_total']

	# filename-ID dataframe
	ref_df = pd.DataFrame(columns=['file_names', 'participant_ID', 'num_rows', 'payment', 'avg_resp_time'])
	# stacked dataframe
	stacked_df = None 
	
	files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and not f.startswith('.')]
	ID = 1

	for f in files:
		df = pd.read_csv(join(input_path, f), skip_blank_lines=True, header=0)
		row_num = df.shape[0] #get the number of rows
		payment = df.iloc[row_num-1:]['bonus_total']
		resp_time = df[['waiting_elapsed']].mean(axis=0)['waiting_elapsed']
		ref_df = ref_df.append(pd.DataFrame({'file_names':f, 'participant_ID':ID, 'num_rows':row_num-7, 'payment':payment, 'avg_resp_time':resp_time}))

		df = df.iloc[:row_num-7] #remove trailing rows that only contain "bonus..." columns. only experiment data left. 
		df = step1(df) 
		df = step2(df, ID) 
		df = df[cols] #rearrange columns in the order of cols variable above
		df.to_csv(join(output_path, f), index=True)
		if stacked_df is None: #add each csv to stacked dataframe
			stacked_df = df
		else:
			stacked_df = stacked_df.append(df)
		ID += 1

	stacked_df = stacked_df[cols] #rearrange column order
	stacked_df.drop(columns=['bonus_block_type', 'bonus_trial_number_chosen', 'bonus_trial_correct', 'bonus_total'], inplace=True)
	stacked_df.to_csv(join(output_path, 'stacked_file.csv'), index=False)
	ref_df.to_csv(join(output_path, 'reference.csv'), index=False)
	

