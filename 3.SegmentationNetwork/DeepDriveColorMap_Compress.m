function [cmap,classes] = DeepDriveColorMap_Compress()
% Define the colormap used by CamVid dataset.

cmap = [
    128 128 128   %Void
    192 128 128   % sky
    128 64 128    % Flat
    0 0 128     % vehicle
    0 0 255       % human
    128 128 64     % object
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;


classes=[
    "void"
    "sky"
    "flat"
    "vehicle"
    "human"
    "object"
    ];

end