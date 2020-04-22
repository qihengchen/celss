%%


%% initialize

close all
clc
clear

m = csvread('stacked_file.csv', 1, 0);
m_length = length(m);
m = m([1:m_length],[1:26]);

%% fill the table

max_i_participant = max(m(:,2));
participant_table = zeros(max_i_participant,8);

for i_participant=1:max_i_participant
    
    participant_table(i_participant,1) = i_participant;
    
    for i_treatment=1:6
    
        selected_trials=m(m(:,2)==i_participant,:);
        selected_trials=selected_trials(selected_trials(:,5)==i_treatment,:);
        avg_score = mean(selected_trials(:,26));
        participant_table(i_participant,i_treatment+1)=avg_score;
    
    end

    participant_table(i_participant,8) = mean(participant_table(i_participant,2:7));

end

% remove noisy data

participant_table = participant_table(participant_table(:,8)>0.55,:);

disp(participant_table)
disp(mean(participant_table))