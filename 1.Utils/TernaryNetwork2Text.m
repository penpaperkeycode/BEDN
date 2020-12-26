function TernaryNetwork2Text(SimNet,Address)
%This function act as a text generator for Binarized Neural Network Which
%is designed by Jeonghoon Kim. date: 2018.08.20
%Feel free to modify and redistribute for non-commercial purpose with first
%authors name and this notations.

%  input  =  DAGnetwork or SeriesNetwork mat file. (Image Input Only)
%  Output =  Parameters in .txt, Architecture in pdf format.


%======================:Archi_info Rules:========================%
% Layer Bit Definition
bit_highBitConv = 1;
bit_binConv = 2;
bit_TbinConv = 3;
bit_MaxPool = 4;
bit_MaxUnpool= -4;
bit_AvePool = 8;
bit_Affine  = 16;
bit_OutputScale =32;
bit_InputScale=0;

LayerInputSize=SimNet.Layers(1, 1).InputSize(1);

%[ Layer Bit, Input Size, Numb Channels, Numb Filters, Stride, Padding or Cropping;
%  Layer Bit, Input Size, Numb Channels, Numb Filters, Stride, Padding or Cropping];

%======================= :Save The Parameters: ======================%

convWindex=0;
convBindex=0;
thParamindex=0;
affineWindex=0;
Archi_info=[];
readMe={'bit_highBitConv = 1, bit_binConv = 2, bit_TbinConv = 3, bit_MaxPool = 4, bit_MaxUnpool= -4, bit_AvePool = 8, bit_Affine  = 16, bit_OutputScale =32, bit_InputScale=0;';
    'Format: [Layer Bit  | Layer Input Size(Height)  |  Channels  |  Filters  |  Stride[vertical,horizonal] |  Padding Size[t,b,l,r]'};

mkdir(Address)
mkdir([Address,'\info'])

for i=1:size(SimNet.Layers,1)
    tmp_string=SimNet.Layers(i,1).Name;
    disp(tmp_string)
    if ismethod(SimNet.Layers(i,1),'Convolution2DLayer')
        if ~isempty((strfind(SimNet.Layers(i,1).Name,'Affine')))
            %Affine Layer
            txtName=[Address,'\fc',num2str(affineWindex),'W.txt'];
            
            weight1d = int32(reshape(permute(SimNet.Layers(i,1).Weights,[2 1 3 4]),1,[]))';
            
            %             weight1d(weight1d==-1)=0;
            %             weight1d=logical(weight1d);
            dlmwrite(txtName,weight1d,'newline','pc')
            
            txtName=[Address,'\b',num2str(convBindex),'_BNFb.txt'];
            if ~isempty(SimNet.Layers(i,1).Bias)
                bias1d = int16(reshape(permute(SimNet.Layers(i,1).Bias,[3 2 1]),1,[]))';
                dlmwrite(txtName,bias1d,'newline','pc','precision',16)
            end
            Archi_info=[Archi_info;
                bit_Affine, LayerInputSize, SimNet.Layers(i, 1).NumChannels, SimNet.Layers(i,1).NumFilters, SimNet.Layers(i,1).Stride, SimNet.Layers(i,1).PaddingSize
                ];
            
            LayerInputSize=1+ceil(LayerInputSize-SimNet.Layers(i,1).FilterSize(1)+2*SimNet.Layers(i,1).PaddingSize(1)/SimNet.Layers(i,1).Stride(1));
            affineWindex=affineWindex+1;
            convBindex=convBindex+1;
            
            
        else %Conv Layer
            txtName=[Address,'\conv',num2str(convWindex),'W.txt'];
            
            weight1d = int32(reshape(permute(SimNet.Layers(i,1).Weights,[2 1 3 4]),1,[]))';
            %             weight1d(weight1d==-1)=0;
            %             weight1d=logical(weight1d);
            dlmwrite(txtName,weight1d,'newline','pc')
            
            txtName=[Address,'\b',num2str(convBindex),'_BNFb.txt'];
            
            
            
            if convWindex==0
                Archi_info=[Archi_info;
                    bit_highBitConv, LayerInputSize,SimNet.Layers(i, 1).NumChannels,SimNet.Layers(i,1).NumFilters, SimNet.Layers(i,1).Stride, SimNet.Layers(i,1).PaddingSize
                    ];
                
                %                 if ~isempty(SimNet.Layers(i,1).Bias)
                %                     bias1dtmp = (reshape(permute(SimNet.Layers(i,1).Bias,[3 2 1]),1,[]))';
                %                     bias1d_obj = sfi(bias1dtmp,24,8);
                %
                %
                %                     % str2num(bias1d_obj.Value);
                %                     bias1d_bin=bin(bias1d_obj);
                %                     dlmwrite(txtName,bias1d_bin,'delimiter','','precision',24)
                %                 end
            else
                Archi_info=[Archi_info;
                    bit_binConv, LayerInputSize,SimNet.Layers(i, 1).NumChannels,SimNet.Layers(i,1).NumFilters, SimNet.Layers(i,1).Stride, SimNet.Layers(i,1).PaddingSize
                    ];
                
                %                 if ~isempty(SimNet.Layers(i,1).Bias)
                %                     bias1d = int16(reshape(permute(SimNet.Layers(i,1).Bias,[3 2 1]),1,[]))';
                %                     dlmwrite(txtName,bias1d,'newline','pc','precision',16)
                %                 end
            end
            
            LayerInputSize=1+ceil((LayerInputSize-SimNet.Layers(i,1).FilterSize(1)+2*SimNet.Layers(i,1).PaddingSize(1))/SimNet.Layers(i,1).Stride(1));
            convWindex=convWindex+1;
            convBindex=convBindex+1;
        end
        
    elseif ismethod(SimNet.Layers(i,1),'TransposedConvolution2DLayer')
        txtName=[Address,'\conv',num2str(convWindex),'W.txt'];
        
        TransposedWeight=permute(SimNet.Layers(i,1).Weights,[1,2,4,3]);
        TransposedWeight=flip(TransposedWeight);
        TransposedWeight=flip(TransposedWeight,2);
        weight1d = int32(reshape(permute(TransposedWeight,[2 1 3 4]),1,[]))';
        weight1d(weight1d==-1)=0;
        weight1d=logical(weight1d);
        dlmwrite(txtName,weight1d,'newline','pc')
        
        txtName=[Address,'\b',num2str(convBindex),'_BNFb.txt'];
        
        
        Archi_info=[Archi_info;
            bit_TbinConv, LayerInputSize,SimNet.Layers(i, 1).NumChannels,SimNet.Layers(i,1).NumFilters, SimNet.Layers(i,1).Stride, SimNet.Layers(i,1).CroppingSize
            ];
        
        bias1d = int16(reshape(permute(SimNet.Layers(i,1).Bias,[3 2 1]),1,[]))';
        dlmwrite(txtName,bias1d,'newline','pc','precision',16)
        
        %                 LayerInputSize=1+ceil((LayerInputSize-Net.Layers(i,1).FilterSize(1)+Net.Layers(i,1).CroppingSize(1)+Net.Layers(i,1).CroppingSize(2))*Net.Layers(i,1).Stride(1)); %It is wrong, Need checking here
        LayerInputSize=ceil((LayerInputSize-1) *SimNet.Layers(i,1).Stride(1) + SimNet.Layers(i,1).FilterSize(1)-SimNet.Layers(i,1).CroppingSize(1)-SimNet.Layers(i,1).CroppingSize(2)); %It is wrong, Need checking here
        convWindex=convWindex+1;
        convBindex=convBindex+1;
        
    elseif ismethod(SimNet.Layers(i,1),'MaxPooling2DLayer')
        
        MaxNumFilters=Archi_info(end,4);
        
        Archi_info=[Archi_info;
            bit_MaxPool, LayerInputSize,MaxNumFilters,MaxNumFilters, SimNet.Layers(i,1).Stride, [0,0,0,0]
            ];
        
        LayerInputSize=1+ceil(       (LayerInputSize-SimNet.Layers(i, 1).PoolSize(1))/SimNet.Layers(i, 1).Stride(1)    );
        
        
    elseif ismethod(SimNet.Layers(i,1),'MaxUnpooling2DLayer')
        
        MaxNumFilters=Archi_info(end,4);
        
        Archi_info=[Archi_info;
            bit_MaxUnpool, LayerInputSize,MaxNumFilters,MaxNumFilters, SimNet.Layers(i,1).Stride, [0,0,0,0]
            ];
        
        LayerInputSize=LayerInputSize * 2;
        
    elseif ismethod(SimNet.Layers(i,1),'AveragePooling2DLayer')
        AveNumFilters=Archi_info(end,4);
        
        Archi_info=[Archi_info;
            bit_AvePool, LayerInputSize,AveNumFilters,AveNumFilters, SimNet.Layers(i,1).Stride, [0,0,0,0]
            ];
        
        LayerInputSize=1;
        
    elseif ismethod(SimNet.Layers(i,1),'OutputScaleLayer')
        
        Archi_info=[Archi_info;
            bit_OutputScale, LayerInputSize, size(SimNet.Layers(i, 1).Scale,3) ,0, [0,0], [0,0,0,0]
            ];
        
        txtName=[Address,'\OutputScale.txt'];
        
        scale1dtmp = (reshape(permute(SimNet.Layers(i,1).Scale,[3 2 1]),1,[]))';
        %         scale1d_obj = sfi(scale1dtmp,24,8);
        scale1d_obj = sfi(scale1dtmp,8);
        %         scale1d_bin=bin(scale1d_obj);
        dlmwrite(txtName,scale1d_obj,'delimiter','','precision',8)
        
    elseif ismethod(SimNet.Layers(i,1),'InputScaleLayer')
        
        Archi_info=[Archi_info;
            bit_InputScale, LayerInputSize,0,0, [0,0], [0,0,0,0]
            ];
        
    elseif ismethod(SimNet.Layers(i,1),'TernaryActivation')
        
        if thParamindex==0
            txtName=[Address,'\b',num2str(thParamindex),'_BNFbP.txt'];
            thP=SimNet.Layers(i,1).thP;
            bias1dtmp_P = (reshape(permute(thP,[3 2 1]),1,[]))';
            bias1d_obj_P = sfi(bias1dtmp_P,24,8);
            bias1d_bin_P=bin(bias1d_obj_P);
            dlmwrite(txtName,bias1d_bin_P,'delimiter','','precision',24)

            txtName=[Address,'\b',num2str(thParamindex),'_BNFbN.txt'];
            thN=SimNet.Layers(i,1).thN;
            bias1dtmp_N = (reshape(permute(thN,[3 2 1]),1,[]))';
            bias1d_obj_N = sfi(bias1dtmp_N,24,8);
            bias1d_bin_N=bin(bias1d_obj_N);
            dlmwrite(txtName,bias1d_bin_N,'delimiter','','precision',24)
        else
            txtName=[Address,'\b',num2str(thParamindex),'_BNFbP.txt'];
            thP=SimNet.Layers(i,1).thP;
            bias1d = int16(reshape(permute(thP,[3 2 1]),1,[]))';
            dlmwrite(txtName,bias1d,'newline','pc','precision',16)
            
            txtName=[Address,'\b',num2str(thParamindex),'_BNFbN.txt'];
            thN=SimNet.Layers(i,1).thN;
            bias1d = int16(reshape(permute(thN,[3 2 1]),1,[]))';
            dlmwrite(txtName,bias1d,'newline','pc','precision',16)
            
        end
        thParamindex=thParamindex+1;
        
        
    end
    disp(' ')
end


txtName_Architecture=[Address,'\info','\Architecture.txt'];
dlmwrite(txtName_Architecture,Archi_info,'newline','pc')
dlmwrite([Address,'\info','\readMe.txt'],char(readMe),'delimiter','')

%=================== :Save The Architecture Design: ==================%
% if ~isempty(Net.Connections)
%     %if Connection information does not exist, the error message will apear
%     txtName='NetworkConnections.txt';
%     writetable(Net.Connections, txtName)
% end

% figure('Units','normalized','Position',[0 0.1 0.3 0.8]);
% plot(layerGraph(Net.Layers))
% grid on
% title('Extracted Network Architecture')
% ax = gca;
% outerpos = ax.OuterPosition;
% ti = ax.TightInset;
% left = outerpos(1) + ti(1);
% bottom = outerpos(2) + ti(2);
% ax_width = outerpos(3) - ti(1) - ti(3);
% ax_height = outerpos(4) - ti(2) - ti(4);
% ax.Position = [left bottom ax_width ax_height];
% % axis off
% set(gca,'XTickLabel',[],'YTickLabel',[]) %Axis number off
% fig = gcf;
% fig.PaperPositionMode = 'auto';
% fig_pos = fig.PaperPosition;
% fig.PaperSize = [fig_pos(3) fig_pos(4)];
% print(fig,[Address,'\info','\ExtractedNetworkArchitecture'],'-dpdf') %as PDF
% close all

end

