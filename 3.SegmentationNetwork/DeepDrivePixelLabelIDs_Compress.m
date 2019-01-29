function labelIDs = DeepDrivePixelLabelIDs_Compress()
% Return the label IDs corresponding to each class.
%
% The CamVid dataset has 32 classes. Group them into 11 classes following
% the original SegNet training methodology [1].
%
% The 11 classes are:
%   "Sky" "Building", "Pole", "Road", "Pavement", "Tree", "SignSymbol",
%   "Fence", "Car", "Pedestrian",  and "Bicyclist".
%
% CamVid pixel label IDs are provided as RGB color values. Group them into
% 11 classes and return them as a cell array of M-by-3 matrices. The
% original CamVid class names are listed alongside each RGB value. Note
% that the Other/Void class are excluded below.
labelIDs = { ...
    
% "void"
[
0,0,0; ... %unlabeld, ego vehicle, rectification border, out of roi, static
111,74,0; ... %dynamic
]

%sky
[
70,130,180;
]

% "flat"
[
128, 64,128; ... % road
244, 35,232; ... % sidewalk
250,170,160; ... % parking
230,150,140; ... % rail track
81,0,81; %ground
]

% "vehicle"
[
0,0,142;
0,0,70;
0,60,100;
0,0,90;
0,0,110;
0,80,100;
0,0,230;
119,11,32;
]

% "human"
[
220, 20, 60;
255,0,0;
]

% "object"
[
107,142, 35;
152,251,152;
153,153,153;
250,170, 30;
220,220,  0;
70, 70, 70; ... %
102,102,156; ... % 
190,153,153;
180,165,180;
150,100,100;
150,120, 90;
]




};
end
