function predictedCP = predict_NN_GUI(xData, yData, modelFilePath)
% predict_NN_GUI
% Predicts the contact point (CP) using a trained neural network.
%
% Inputs:
%   xData - Raw extension data (nm)
%   yData - Raw deflection data (nm)
%   modelFilePath - Path to the trained neural network .mat file
%
% Output:
%   predictedCP - Predicted contact point (normalized [0,1])

    % Load the trained neural network model
    if ~isfile(modelFilePath)
        error('Model file "%s" not found.', modelFilePath);
    end
    loadedModel = load(modelFilePath);
    
    % Extract the network object
    if isfield(loadedModel, 'trainedNet')
        net = loadedModel.trainedNet;
    else
        error('Loaded .mat file does not contain a valid trained network.');
    end

    % Define the number of interpolation points (must match the model training)
    n_points = 2000;

    % Preprocess the input curve
    X = preprocess_single_curve_GUI(xData, yData, n_points);

    % Format the input data to match the network's expected format
    X_for_prediction = permute(X, [1, 3, 2]); % [C x B x T]
    X_for_prediction = dlarray(X_for_prediction, 'CBT');

    % Predict the contact point
    predictedCP = predict(net, X_for_prediction);
    predictedCP = extractdata(predictedCP)'; % Convert to numeric output

    predictedCP = min(xData) + predictedCP*(max(xData) - min(xData));
    predictedCP = double(predictedCP);
end
