%% This file is to save all attacks in current folder to one cell array
% The training datasets are save in below structure:
%    Current parent folder:  attack_dataset                            (1)
%    Sub folder:             n_attack1, n_attack2, ..., n_atatck20    (20)   (for different number of attacks)
%        (Notice: sub folder would be void)
%    Sub Sub folder:         1, 2, 3, ... , 20                        (20)   (for different attack support)
%        (Notice: sub folder would not have 20 complete folders) 
%    file 1:          'I_attack.csv'               
%    file 2:          'attacks.csv'
%    file 3:          'y1_effect.csv'
%    file 4:          'y2_detect.csv'
%
% Output: "All_attacks.mat (MyData)" (cell array)
%          structure: 1. MyData{1,i}: one attack dataset (corresponding to an attack support and an attack percentage)
%                     2. MyData{1,i}{1,1}: attacks   (attack dataset)
%                        MyData{1,i}{1,2}: I_attack  (attack support)
%                        MyData{1,i}{1,3}: y1_effect  (effectiveness calculated by M1)
%                        MyData{1,i}{1,4}: y2_detect  (T-horizon cumulative detection power calculated by M2)
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.

%% Search all folders and load datasets
mainpath = 'attack_dataset';                      % parent folder
ContentInFold = dir(mainpath);                    % search all sub folders
SubFold = ContentInFold([ContentInFold.isdir]);   % search all sub sub folders

MyData =[];                                       % cache for saving all data (cell array)
for i = 3:length(SubFold)  % start at 3th folder to skip "." and ".." folders (this is a matlab problem)
    % enter the sub folder layer
    file_subFold = fullfile(mainpath,SubFold(i).name);
    ContentInSubFold = dir(file_subFold);
    SubSubFold =  ContentInSubFold([ContentInSubFold.isdir]);
    for j = 3:length(SubSubFold)
        % enter the sub sub folder layer
        filetoread1 = fullfile(mainpath,SubFold(i).name,SubSubFold(j).name,'I_attack.csv'); % construct a full path: mainpath/SubFold/SubSubFold/I_attack.mat
        filetoread2 = fullfile(mainpath,SubFold(i).name,SubSubFold(j).name,'attacks.csv');  % construct a full path: mainpath/SubFold/SubSubFold/attacks.mat
        filetoread3 = fullfile(mainpath,SubFold(i).name,SubSubFold(j).name,'y1_effect.csv');
        filetoread4 = fullfile(mainpath,SubFold(i).name,SubSubFold(j).name,'y2_detect.csv');
        % then load the dataset
        I_attack = readmatrix(filetoread1);
        attacks = readmatrix(filetoread2);
        y1_effect = readmatrix(filetoread3);
        y2_detect = readmatrix(filetoread4);
        
        % check if the dataset is correct
        if size(attacks,1) ~= size(y2_detect,2) || size(attacks,1) ~= size(y1_effect,2) || size(attacks,2) ~= size(I_attack,1)
            disp('wrong dataset found')
            disp(I_attack);
            keyboard;
            % please check the corresponding dataset
            % you can use I_attack to locate the wrong dataset
            % delete it and regenerate one by using '\1_Attack_generation\Attack_Generator_single.py'
            % change 'n_attack' to the one you need in line 190
            % then manually add the dataset to folder 'attack_dataset'
        end
        
        attack_data = {attacks,I_attack,y1_effect,y2_detect};
        MyData{end+1} = attack_data;
    end
end

% save All_attacks_final_sim.mat MyData
save('All_attacks.mat', 'MyData', '-v7.3')