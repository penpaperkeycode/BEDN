function WeightHistogram(Net)
a=1;
for i=1:size(Net.Layers,1)
    tmp_string=Net.Layers(i,1).Name;
    disp(tmp_string)
    if ismethod(Net.Layers(i,1),'Convolution2DLayer')
        if a ~=7
            figure(1)
            rng default; % For reproducibility
            histogram(reshape(permute(Net.Layers(i,1).Weights,[3 2 1 4]),1,[]),100,'Normalization','probability','DisplayStyle','stairs','DisplayName',[num2str(a),'th Conv layer'])
            set(gca,'YScale','log')
            ylim([0 1])
            grid on
            hold on
            a=a+1;
        end
    end
end
legend('show')
title('Conv layers Weight Probability Density')


