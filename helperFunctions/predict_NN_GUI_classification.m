function yeetThisCurve = predict_NN_GUI_classification(xData, yData, modelFilePath)
% predict_NN_GUI_classification
% Predicts the whether a given force curve is good or bad.
%
% Inputs:
%   xData - Raw extension data (nm)
%   yData - Raw deflection data (nm)
%   modelFilePath - Path to the trained neural network .mat file
%
% Output:
%   true/false (true = is bad, false = is good) (I'm known for my clarity)

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

    % Predict whether to yeet this curve
    yeetThisCurve = predict(net, X_for_prediction);
    yeetThisCurve  = extractdata(yeetThisCurve)'; 
    yeetThisCurve = yeetThisCurve(1);

end
