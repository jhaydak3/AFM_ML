% List of .mat to process
matFiles = ["C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\one_CNN_one_biLSTM.mat"
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\two_conv_LSTM_sequence_pooling_relu_no_augmentation.mat"
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\two_conv_LSTM_sequence_pooling_relu.mat"
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\two_conv_LSTM_sequence_pooling.mat"
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\stacked_biLSTM_ReLu_no_augmentation.mat"
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\stacked_biLSTM_ReLu.mat"
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\Sotres2022_ResNet50-1D_no_augmentation.mat"
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\Sotres2022_ResNet50-1D_expanded_features_no_augmentation.mat"
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\Sotres2022_ResNet50-1D_expanded_features.mat"
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\Sotres2022_ResNet50-1D.mat"
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\Sotres2022_original_no_augmentation.mat"
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\Sotres2022_original.mat"
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\one_conv_LSTM_sequence_relu_no_augmentation.mat"
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\one_conv_LSTM_sequence_relu.mat"
"C:\Users\MrBes\Documents\MATLAB\AFM_ML\AFM_ML_v5\evaluation\regression\CNNs\one_CNN_one_biLSTM_no_augmentation.mat"];

% Initialize results table
resultsTable = table();

% Loop through each folder
for i = 1:length(matFiles)

        
    thisFile = matFiles(i);
    % Extract the last folder name
    [~, fileName] = fileparts(thisFile);
    
    % Load the MAT file
    data = load(thisFile);
    
    % Define variables to check
    varsToCheck = {'meanMAE', 'meanTrainMAE', 'meanMAENm', 'meanTrainMAENm', 'meanMSE', 'meanMSENm', 'mod500nmMAE', ...
                   'mod500nmMSE', 'mod500nmMAPE', 'hertzMAE', 'hertzMSE', 'hertzMAPE'};
               
    % Initialize a row with NA values
    row = cell(1, length(varsToCheck));
    
    % Check for each variable in the file
    for j = 1:length(varsToCheck)
        if isfield(data, varsToCheck{j})
            row{j} = data.(varsToCheck{j});
        else
            row{j} = NaN; % Add NA if variable does not exist
        end
    end
    
    % Add to results table
    resultsTable = [resultsTable; table({fileName}, row{:}, 'VariableNames', ['File', varsToCheck])];
end

% Display the results table
disp(resultsTable);

% Save the table to a file
writetable(resultsTable, 'results_summary.csv');

