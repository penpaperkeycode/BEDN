


scale= SimNet.Layers(24, 1).Scale  ./(SimNet.Layers(24, 1).TrainedVariance+SimNet.Layers(24, 1).Epsilon)  ;
% offset= SimNet.Layers(24, 1).Offset  - SimNet.Layers(24, 1).TrainedMean  ./(SimNet.Layers(24, 1).TrainedVariance+SimNet.Layers(24, 1).Epsilon);
offset = -(SimNet.Layers(24, 1).TrainedMean - ((SimNet.Layers(24, 1).TrainedVariance+SimNet.Layers(24, 1).Epsilon)./SimNet.Layers(24, 1).Scale).*SimNet.Layers(24, 1).Offset );

tmp1 = sfi(scale,24,8);
Xvalue=double(str2num(tmp1.Value));
scale_24bit=reshape(Xvalue,[1,1,11]);


tmp1 = sfi(offset,24,8);
Xvalue=double(str2num(tmp1.Value));
offset_24bit=reshape(Xvalue,[1,1,11]);

scale_24bit = (reshape(permute(scale_24bit,[3 2 1]),1,[]))';
dlmwrite([address,'\scale_24bit.txt'],scale_24bit,'newline','pc','precision',24)

offset_24bit = (reshape(permute(offset_24bit,[3 2 1]),1,[]))';
dlmwrite([address,'\offset_24bit.txt'],offset_24bit,'newline','pc','precision',24)