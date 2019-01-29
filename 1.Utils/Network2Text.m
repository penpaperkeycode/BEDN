function Network2Text(Net,Address)
%This function act as a text generator for Binarized Neural Network Which
%is designed by Jeonghoon Kim. date: 2018.08.20
%Feel free to modify and redistribute for non-commercial purpose with first
%authors name and this notations.

%  input  =  DAGnetwork or SeriesNetwork mat file. (Image Input Only)
%  Output =  Parameters in .txt, Architecture in pdf format.


%======================:Archi_info Rules:========================%
% Layer Bit Definition
bit_intConv = 1;
bit_binConv = 2;
bit_MaxPool = 4;
bit_MaxUnpool= -4;
bit_AvePool = 8;
bit_Affine  = 16;

LayerInputSize=Net.Layers(1, 1).InputSize(1);

%[ Layer Bit, Input Size, Numb Channels, Numb Filters
%  Layer Bit, Input Size, Numb Channels, Numb Filters ];

%======================= :Save The Parameters: ======================%

convWindex=0;
convBindex=0;
affineWindex=0;
Archi_info=[];


mkdir(Address)
mkdir([Address,'\info'])

for i=1:size(Net.Layers,1)
    tmp_string=Net.Layers(i,1).Name;
    disp(tmp_string)
    if ismethod(Net.Layers(i,1),'Convolution2DLayer')
        if ~isempty((strfind(Net.Layers(i,1).Name,'Affine')))
            %Affine Layer
            txtName=[Address,'\fc',num2str(affineWindex),'W.txt'];
            
            
            weight1d = int32(reshape(permute(Net.Layers(i,1).Weights,[3 2 1 4]),1,[]))';
            
            weight1d(weight1d==-1)=0;
            weight1d=logical(weight1d);
            dlmwrite(txtName,weight1d,'newline','pc')
            
            txtName=[Address,'\b',num2str(convBindex),'_BNFb.txt'];
            
            bias1d = int32(reshape(permute(Net.Layers(i,1).Bias,[3 2 1]),1,[]))';
            dlmwrite(txtName,bias1d,'newline','pc','precision',15)
            
            %             txtName=[Address,'\info','\ConvLayer(Affine)',num2str(affineWindex),'_Info.txt'];
            %             tmpInfo.Name=Net.Layers(i,1).Name;
            %             tmpInfo.FilterSize=Net.Layers(i,1).FilterSize;
            %             tmpInfo.NumChannels=Net.Layers(i,1).NumChannels;
            %             tmpInfo.NumFilters=Net.Layers(i,1).NumFilters;
            %             tmpInfo.Stride=Net.Layers(i,1).Stride;
            %             tmpInfo.PaddingMode=Net.Layers(i,1).PaddingMode;
            %             tmpInfo.PaddingSize=Net.Layers(i,1).PaddingSize;
            %             writetable(struct2table(tmpInfo), txtName)
            
            
            
            Archi_info=[Archi_info;
                bit_Affine, LayerInputSize,Net.Layers(i, 1).NumChannels,Net.Layers(i,1).NumFilters
                ];
            
            
            
            LayerInputSize=1+floor(LayerInputSize-Net.Layers(i,1).FilterSize(1)+2*Net.Layers(i,1).PaddingSize(1)/Net.Layers(i,1).Stride(1));
            affineWindex=affineWindex+1;
            convBindex=convBindex+1;
            
            
        else %Conv Layer
            txtName=[Address,'\conv',num2str(convWindex),'W.txt'];
            
            %weight1d = int32(reshape(permute(Net.Layers(i,1).Weights,[3 2 1 4]),1,[]))';
            weight1d = int32(reshape(permute(Net.Layers(i,1).Weights,[2 1 3 4]),1,[]))';
            weight1d(weight1d==-1)=0;
            weight1d=logical(weight1d);
            dlmwrite(txtName,weight1d,'newline','pc')
            
            txtName=[Address,'\b',num2str(convBindex),'_BNFb.txt'];
            
            bias1d = int32(reshape(permute(Net.Layers(i,1).Bias,[3 2 1]),1,[]))';
            dlmwrite(txtName,bias1d,'newline','pc','precision',15)
            
            %             txtName=[Address,'\info','\ConvLayer',num2str(convWindex),'_Info.txt'];
            %             tmpInfo.Name=Net.Layers(i,1).Name;
            %             tmpInfo.FilterSize=Net.Layers(i,1).FilterSize;
            %             tmpInfo.NumChannels=Net.Layers(i,1).NumChannels;
            %             tmpInfo.NumFilters=Net.Layers(i,1).NumFilters;
            %             tmpInfo.Stride=Net.Layers(i,1).Stride;
            %             tmpInfo.PaddingMode=Net.Layers(i,1).PaddingMode;
            %             tmpInfo.PaddingSize=Net.Layers(i,1).PaddingSize;
            %             writetable(struct2table(tmpInfo), txtName)
            
            if convWindex==0
                Archi_info=[Archi_info;
                    bit_intConv, LayerInputSize,Net.Layers(i, 1).NumChannels,Net.Layers(i,1).NumFilters
                    ];
            else
                Archi_info=[Archi_info;
                    bit_binConv, LayerInputSize,Net.Layers(i, 1).NumChannels,Net.Layers(i,1).NumFilters
                    ];
            end
            
            %             if Net.Layers(i,1).Stride(1)~=1 || Net.Layers(i,1).PaddingSize~=1
            %
            %             end
            LayerInputSize=1+floor((LayerInputSize-Net.Layers(i,1).FilterSize(1)+2*Net.Layers(i,1).PaddingSize(1))/Net.Layers(i,1).Stride(1));
            convWindex=convWindex+1;
            convBindex=convBindex+1;
        end
        
    elseif ismethod(Net.Layers(i,1),'MaxPooling2DLayer')
        
        MaxNumFilters=Archi_info(end,4);
        
        Archi_info=[Archi_info;
            bit_MaxPool, LayerInputSize,MaxNumFilters,MaxNumFilters
            ];
        
        LayerInputSize=1+floor(       (LayerInputSize-Net.Layers(i, 1).PoolSize(1))/Net.Layers(i, 1).Stride(1)    );
        
        
    elseif ismethod(Net.Layers(i,1),'MaxUnpooling2DLayer')
        
        MaxNumFilters=Archi_info(end,4);
        
        Archi_info=[Archi_info;
            bit_MaxUnpool, LayerInputSize,MaxNumFilters,MaxNumFilters
            ];
        
        LayerInputSize=LayerInputSize * 2;
        
    elseif ismethod(Net.Layers(i,1),'AveragePooling2DLayer')
        AveNumFilters=Archi_info(end,4);
        
        Archi_info=[Archi_info;
            bit_AvePool, LayerInputSize,AveNumFilters,AveNumFilters
            ];
        
        LayerInputSize=1;
    end
    disp(' ')
end


txtName_Architecture=[Address,'\info','\Architecture.txt'];
dlmwrite(txtName_Architecture,Archi_info,'newline','pc')


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

