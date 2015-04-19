function train(dataset, options)
    if nargin < 2; options = []; end
    
    addpath data
    [xTr, yTr] = prep_data(dataset);
    rmpath data
    
    classifier = trainClassifier(xTr, yTr, options);
    
    if exist(strcat(dataset, '_classifier.mat'), 'file') ~= 0
        [acc, time] = classify(dataset);
        fprintf('Prior error rate: %0.2f%% in %.0f milliseconds\n', acc, time);
    end
    save(strcat(dataset, '_classifier'), 'classifier');
    [acc, time] = classify(dataset);
    fprintf('Error rate: %0.2f%% in %.0f milliseconds\n', acc, time);
    
end

