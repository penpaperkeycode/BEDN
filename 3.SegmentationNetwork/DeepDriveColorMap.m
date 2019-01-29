function [cmap,classes] = DeepDriveColorMap()
% Define the colormap used by CamVid dataset.

cmap = [
    128 128 128   %Void
    192 128 128    % Flat
    128 0 0       % construction
    60 40 222     % object
    128 128 0     % nature
    128 128 255   % sky
    64 0 128      % human
    0 128 192     % vehicle
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;

classes=[
    "void"
    "flat"
    "construction"
    "object"
    "nature"
    "sky"
    "human"
    "vehicle"
    ];

end