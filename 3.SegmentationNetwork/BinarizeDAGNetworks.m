function netout=BinarizeDAGNetworks(netin)

if ~isa(netin,'DAGNetwork')
    error('Networks must be DAGNetwork')
end

nettmp=AutoConstructHWLayer(netin);
Connectionstmp=netin.Connections;
ConnectionsArray=table2array(Connectionstmp);
ConnectionsArray1=ConnectionsArray(:,1);
ConnectionsArray2=ConnectionsArray(:,2);

ConnectionsArray1index=[];
ConnectionsArray2index=[];
for i=1:size(Connectionstmp,1)
    if strfind(ConnectionsArray1{i},'BatchNorm')
        ConnectionsArray1index=[ConnectionsArray1index,i];
    elseif strfind(ConnectionsArray2{i},'BatchNorm')
        ConnectionsArray2index=[ConnectionsArray2index,i];
    end
end
ConnectionsArray1(ConnectionsArray1index)=[];
ConnectionsArray2(ConnectionsArray2index)=[];
ConnectionsArray=[ConnectionsArray1,ConnectionsArray2];

ConnectionsTable = cell2table(ConnectionsArray,'VariableNames',{'Source','Destination'});

lgraph = createLgraphUsingConnections(nettmp,ConnectionsTable);

netout=DAGNetwork.loadobj(lgraph);

end