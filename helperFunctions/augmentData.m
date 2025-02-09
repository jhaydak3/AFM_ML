
function [X_augmented, Y_augmented] = augmentData(X, Y, pad)
%augmentData
% Augments the dataset by left-shifting the X data while ensuring that
% the contact part of the indentation is not wrapped.
%
% Inputs:
%   X   - [numFeatures x sequenceLength x numSamples]
%   Y   - [numSamples x 1] (normalized values from 0 to 1)
%   pad - Pad value to ensure CP does not shift to index less than 1 + pad
%
% Outputs:
%   X_augmented - Augmented X data including original and shifted samples
%   Y_augmented - Corresponding Y data

% Set default pad value if not provided
if nargin < 3
    pad = 100; % Adjust based on your data characteristics
end

% Initialize augmented data with original data
X_augmented = X;
Y_augmented = Y;

numSamples = size(X, 3);
sequenceLength = size(X, 2); % n_points

fprintf('Starting data augmentation with pad value %d...\n', pad);

for i = 1:numSamples
    % Calculate the contact point index for the current sample
    % Y(i) is a normalized value from 0 to 1
    contactPointIndex = round(Y(i) * sequenceLength);

    % Ensure the contactPointIndex is within valid bounds
    contactPointIndex = max(min(contactPointIndex, sequenceLength), 1);

    % Calculate maximum allowable left shift
    % After shifting, CP index should not be less than 1 + pad
    maxAllowableShift = contactPointIndex - (1 + pad);

    % If maxAllowableShift is less than or equal to zero, no shifting is possible
    if maxAllowableShift <= 0
        continue; % Skip shifting for this sample
    end

    % Randomly select a shift value within the allowable range
    % Shift values are negative (left shifts)
    shift = -randi(maxAllowableShift); % Random integer between -maxAllowableShift and -1

    % Shift the data to the left
    X_shifted = circshift(X(:, :, i), [0, shift]);

    % Calculate the new contact point index after shifting
    newContactPointIndex = contactPointIndex + shift;

    % Ensure the new contact point index is within bounds
    if newContactPointIndex < (1 + pad)
        % This shift moves the CP beyond the allowed minimum index; skip it
        continue;
    end

    % Calculate the new normalized Y value
    Y_shifted = newContactPointIndex / sequenceLength;

    % Append shifted data and corresponding Y to augmented dataset
    X_augmented = cat(3, X_augmented, X_shifted);
    Y_augmented = [Y_augmented; Y_shifted];
end

fprintf('Data augmentation completed. Total samples after augmentation: %d\n', size(X_augmented, 3));
end
