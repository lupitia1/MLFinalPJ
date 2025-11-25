#-------------------------------------------------------------------------------------
# crossvalidation(N::Int64, k::Int64)
#-------------------------------------------------------------------------------------
function crossvalidation(N::Int64, k::Int64)
    #TODO
    folds = collect(1:k) # Vector with the k folds
    limit = Int64(ceil(N/k))
    indices = repeat(folds, limit)
    nIndices =  indices[1:N] # Select the first N indexes
    shuffle!(nIndices) # Shuffle indexes
    return nIndices
end

#-------------------------------------------------------------------------------------
# crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
#-------------------------------------------------------------------------------------
function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    #TODO
    # 1. Call the `oneHotEncoding` function, passing `targets` as the input. This will convert the labels into a binary matrix (multilabel format).
    oneHotTargets = oneHotEncoding(targets, unique(targets))

    # 2. Call the **previous version** of the `crossvalidation` function, passing the encoded matrix and the value `k`.
    return crossvalidation(oneHotTargets, k)
end

#-------------------------------------------------------------------------------------
# crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
#-------------------------------------------------------------------------------------
function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    #TODO
    targets_size = size(targets,1)

    # 1. Create a vector of indices with as many elements as there are rows in `targets`.
    indices = Vector{Int64}(zeros(targets_size))
    
    # 2. Call the previously developed `crossvalidation` function, passing the **number of positive instances** and the value of `k`.
    cv_pos = crossvalidation(sum(targets), k)
    
    # 3. Assign the result of the previous step to the positions of the index vector that correspond to **positive instances**.
    indices[targets] = cv_pos
    
    # 4. Repeat a similar operation for **negative instances**.
    cv_neg = crossvalidation((targets_size-sum(targets)), k) 
    indices[.!targets] = cv_neg 

    # 5. Return the index vector.
    return indices
end

#-------------------------------------------------------------------------------------
# crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
#-------------------------------------------------------------------------------------
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #TODO
    nInstances = size(targets,1)
    nClasses = size(targets,2)
    
    # 1. Create an index vector with as many elements as rows in `targets`.
    indices = Vector{Int64}(zeros(nInstances))

    # 2. Use a loop that **iterates over the classes** (i.e., the columns of `targets`).  
    for i in 1:nClasses
       # i. Count the number of instances that belong to that class using `sum(targets[:, i])`.  
       count = sum(targets[:, i])
       # ii. Call the previously defined `crossvalidation` function with the number of instances and `k`. 
       cv = crossvalidation(count, k) 
       # iii. Assign the result to the corresponding positions in the index vector where `targets[:, i]` is `true`.
       indices[targets[:, i]] = cv
    end

    # 3. Return the index vector.
    return indices
end


#-------------------------------------------------------------------------------------
# oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
#-------------------------------------------------------------------------------------
function oneHotEncoding(feature::AbstractArray{<:Any,1},      
        classes::AbstractArray{<:Any,1})

    """
    Parameters
    ----------
    feature :: AbstractVector
        The input vector of categorical values to be encoded.
    classes :: AbstractVector
        The list/array of unique classes used as encoding reference.
    """
    """
    AbstractArray{<:Any,1} :
    AbstractArray → not restricted to just Vector, could be any 1-dimensional array type.
    1 → one-dimensional array (i.e. a vector).
    <:Any → element type can be any subtype of Any (which basically means no restriction at all).
    So effectively, AbstractArray{<:Any,1} is just a very general way of saying:
    “Accept any 1-D array, regardless of element type.”
    """
    # Defensive: ensure feature is a vector
    # Check that all feature values exist in the set of classes
    @assert(all([in(value, classes) for value in feature]))
    
    # Number of classes
    numClasses = length(classes)
    
    # Defensive: require at least two classes
    @assert(numClasses > 1)
    
    if (numClasses == 2)
        # Special case: binary classification, use a single column
        oneHot = reshape(feature .== classes[1], :, 1)
    else
        # General case: more than two classes
        oneHot = BitArray{2}(undef, length(feature), numClasses)
        for numClass = 1:numClasses
            # Mark 1 where feature matches the class
            oneHot[:, numClass] .= (feature .== classes[numClass])
        end
    end


    """
    Returns
    -------
    AbstractArray
        A one-hot encoded array:
        - Shape (n, 1) if there are 2 classes (binary case).
        - Shape (n, numClasses) if there are more than 2 classes.
    """
    return oneHot
end

#-------------------------------------------------------------------------------------
# holdOut(N::Int, P::Real)
#-------------------------------------------------------------------------------------
function holdOut(N::Int, P::Real)
    #TODO
    @assert(0.0 < P < 1.0 , "P must be in (0,1)")
    #idx = shuffle(1:N)
    idx = randperm(N)
    #println("idx indices: ", idx)
    nTrain = Int(floor((1-P)*N))
    trainIdx = idx[1:nTrain]
    testIdx = idx[nTrain+1:end]
    return (trainIdx, testIdx)
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)     
    @assert(0.0 < Pval < 1.0 , "Pval must be in (0,1)")
    @assert(0.0 < Ptest < 1.0 , "Ptest must be in (0,1)")
    @assert(Pval+Ptest < 1.0 , "Pval + Ptest must be in lower than 1")

    train_val_idx, test_idx = holdOut(N, Ptest)

    # Compute the absolute number of validation samples as Pval * N (rounded down)
    nVal = Int(floor(Pval*N))          
    # Compute the total number of samples available for training + validation (everything except the test set)
    nTrainVal = Int(floor((1-Ptest)*N)) 
    # Recalculate the validation proportion relative to the train+validation pool
    Pval2 = nVal/nTrainVal             

    local_train_idx, local_val_idx = holdOut(length(train_val_idx), Pval2)

    # Map from local indices back to the global indices
    train_idx = train_val_idx[local_train_idx]
    val_idx   = train_val_idx[local_val_idx]

    return (train_idx, val_idx, test_idx)
end;

#-------------------------------------------------------------------------------------
# buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
#                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
#-------------------------------------------------------------------------------------
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;   


#-------------------------------------------------------------------------------------
# classifyOutputs(outputs::AbstractArray{<:Real,2}; 
#                        threshold::Real=0.5) 
#-------------------------------------------------------------------------------------
function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
                        threshold::Real=0.5) 
   numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        # Look for the maximum value using the findmax funtion
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Set up then boolean matrix to everything false while max values aretrue.
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Defensive check if all patterns are in a single class
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;

#-------------------------------------------------------------------------------------
# accuracy(outputs::Vector{Int64}, targets::Vector{Int64}) 
#    mean(outputs.==targets);
#-------------------------------------------------------------------------------------
function accuracy(outputs::Vector{Int64}, targets::Vector{Int64}) 
    mean(outputs.==targets);
end;


#-------------------------------------------------------------------------------------
# trainClassANN
#-------------------------------------------------------------------------------------
function trainClassANN(topology::AbstractArray{<:Int,1},  
            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
            # --- Requirement: optional validation dataset with default empty arrays ---
            validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            # --- Requirement: optional test dataset with default empty arrays ---
            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
            # --- Requirement: maxEpochsVal parameter (early stopping patience), default 20 ---
            maxEpochsVal::Int=20, showText::Bool=false) 

    # --- Unpacking datasets ---
    (inputs, targets) = trainingDataset
    (val_inputs, val_targets) = validationDataset
    (test_inputs, test_targets) = testDataset

    # --- Ensures dataset dimensions match ---
    @assert size(inputs,1) == size(targets,1)
    @assert size(val_inputs,1) == size(val_targets,1)
    @assert size(test_inputs,1) == size(test_targets,1)

    # --- Requirement: build ANN with given topology ---
    ann = buildClassANN(size(inputs,2), topology, size(targets,2))

    # --- Define loss function (binary or multi-class) ---
    # discriminates based on the number of output neurons
    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y)

    # --- Requirement: loss histories for training/validation/test ---
    trainingLosses = Float32[]
    validationLosses = Float32[]
    testLosses = Float32[]

    # --- Initial losses (cycle 0, before training) ---
    numEpoch = 0
    trainingLoss = loss(ann, inputs', targets')
    push!(trainingLosses, trainingLoss)
    
    # init message buffer
    log_message = []
    log_message = "Epoch $numEpoch - loss: $(round(trainingLoss, digits=4))"


    # --- if validation set is provided ---                          
    if size(val_inputs,1) > 0
        validationLoss = loss(ann, val_inputs', val_targets')
        push!(validationLosses, validationLoss)

        # update message buffer
        log_message *= " - val_loss: $(round(validationLoss, digits=4))"
    end

     # --- if test set is provided ---  
    if size(test_inputs,1) > 0
        testLoss = loss(ann, test_inputs', test_targets')
        push!(testLosses, testLoss)
        # update message buffer
        log_message *= " - test_loss: $(round(testLoss, digits=4))"
        if showText
            # do nothing
            #println("Epoch ", numEpoch, ": test loss: ", testLoss)
        end
    end

    if showText
        # print message buffer
        println(join(log_message))
    end
    # --- Optimizer setup ---
    opt_state = Flux.setup(Adam(learningRate), ann)

    # --- Requirement: variables for early stopping ---
    epochsWithoutImprovement = 0
    bestValLoss = Inf
    bestAnn = deepcopy(ann)  # Requirement: store best ANN (deepcopy to avoid overwriting)
    bestAnnEpoch = 0

    while (numEpoch < maxEpochs) && (trainingLoss > minLoss) && (epochsWithoutImprovement < maxEpochsVal)
        Flux.train!(loss, ann, [(inputs', targets')], opt_state)
        numEpoch += 1
        log_message = []
        # --- Compute training loss and store it ---
        trainingLoss = loss(ann, inputs', targets')
        push!(trainingLosses, trainingLoss)
        
        # update message buffer
        log_message = "Epoch $numEpoch - loss: $(round(trainingLoss, digits=4))"

        outputs=ann(inputs')
        outputs=classifyOutputs(outputs')
        predicted_classes = Flux.onecold(outputs')        # vector of predicted labels
        true_classes = Flux.onecold(targets')      # vector of true labels

        accuracy_train=accuracy(predicted_classes, true_classes)

        log_message *= " - acc: $(round(accuracy_train, digits=4))"

        # --- Requirement: if validation set provided, track its loss for early stopping ---
        if size(val_inputs,1) > 0
            validationLoss = loss(ann, val_inputs', val_targets')
            push!(validationLosses, validationLoss)

            # update message buffer
            log_message *= " - val_loss: $(round(validationLoss, digits=4))"

            outputs=ann(val_inputs')
            outputs=classifyOutputs(outputs')
            predicted_classes = Flux.onecold(outputs')        # vector of predicted labels
            true_classes = Flux.onecold(val_targets')      # vector of true labels

            accuracy_val=accuracy(predicted_classes, true_classes)

            log_message *= " - val_acc: $(round(accuracy_val, digits=4))"

            if validationLoss < bestValLoss
                bestValLoss = validationLoss
                epochsWithoutImprovement = 0
                bestAnn = deepcopy(ann)   # Requirement: update best ANN when improvement found
                bestAnnEpoch = numEpoch
            else
                epochsWithoutImprovement += 1
            end
        end

        # --- Requirement: also track test loss if provided ---
        if size(test_inputs,1) > 0
            testLoss = loss(ann, test_inputs', test_targets')
            push!(testLosses, testLoss)
            

            # update message buffer
            log_message *= " - test_loss: $(round(testLoss, digits=4))"

            outputs=ann(test_inputs')
            outputs=classifyOutputs(outputs')
            predicted_classes = Flux.onecold(outputs')        # vector of predicted labels
            true_classes = Flux.onecold(test_targets')      # vector of true labels

            accuracy_test=accuracy(predicted_classes, true_classes)
            test_error = 1-accuracy_test

            log_message *= " - test_acc: $(round(accuracy_test, digits=4))"
            log_message *= " - test_error: $(round(test_error, digits=4))"

        end
        
        if showText
            # update message buffer
            log_message *= " - epochsWithoutImprovement $(epochsWithoutImprovement)"
            println(join(log_message))
        end

        #print("trainingLoss > minLoss : $(trainingLoss > minLoss) \n")
        #print("epochsWithoutImprovement < maxEpochsVal : $(epochsWithoutImprovement < maxEpochsVal) \n")
    end  # closes while
    
    # --- Early stopping notice ---
    if (epochsWithoutImprovement >= maxEpochsVal) && showText
        println("⏹ Early stopping triggered after $numEpoch epochs (no improvement for $maxEpochsVal epochs).")
    end

    # --- Requirement: return the right ANN ---
    # If validation set was provided → return best ANN found
    # Otherwise → return last trained ANN
    finalAnn = size(val_inputs,1) > 0 ? bestAnn : ann

    bestEpoch = size(val_inputs,1) > 0 ? bestAnnEpoch : maxEpochs
    if showText
        println("The ANN corespond to the epoch $bestEpoch")
    end
    return finalAnn, trainingLosses, validationLosses, testLosses
end

#-------------------------------------------------------------------------------------
# trainClassANN
#-------------------------------------------------------------------------------------
function trainClassANN(topology::AbstractArray{<:Int,1},  
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
        maxEpochsVal::Int=20, showText::Bool=false)

    (inputs, targets) = trainingDataset
    (val_inputs, val_targets) = validationDataset
    (test_inputs, test_targets) = testDataset
    
    # This function assumes that each sumple is in a row
    # we are going to check the numeber of samples to have same inputs and targets
    @assert(size(inputs,1)==size(targets,1));
    @assert (size(val_inputs,1) == size(val_targets,1));
    @assert (size(test_inputs,1) == size(test_targets,1));

    trainClassANN(topology, 
        (inputs, reshape(targets, length(targets), 1)),
        (val_inputs, reshape(val_targets, length(val_targets), 1)), 
        (test_inputs, reshape(test_targets, length(test_targets), 1)),
        transferFunctions, 
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
        maxEpochsVal, showText);
end;


#-------------------------------------------------------------------------------------
# confusionMatrix(outrue_posuts::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
#    @assert length(outrue_posuts) == length(targets) 
#-------------------------------------------------------------------------------------
function confusionMatrix(outrue_posuts::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outrue_posuts) == length(targets) "Outrue_posuts and targets must have the same length"
    
    # True positives, true negatives, false positives, false negatives
    true_pos = sum(outrue_posuts .& targets)
    true_neg = sum(.!outrue_posuts .& .!targets)
    false_pos = sum(outrue_posuts .& .!targets)
    false_neg = sum(.!outrue_posuts .& targets)

    # 1.Confusion matrix 
    cm = [true_pos false_pos; false_neg true_neg]

    # 2. Accuracy 
    accuracy = (true_pos + true_neg) / length(targets)

    # 3. Error rate
    error_rate = 1 - accuracy

    # 4. Recall
    sensitivity = if true_pos + false_neg == 0
        # All targets are negative
        1.0
    else
        true_pos / (true_pos + false_neg)
    end

    # 5. Specificity
    specificity = if true_neg + false_pos == 0
        # All targets are positive
        1.0
    else
        true_neg / (true_neg + false_pos)
    end

    # 6. Precision
    pos_pred_val = if true_pos + false_pos == 0
        if true_pos + false_neg == 0  # all patterns negative
            1.0
        else
            0.0
        end
    else
        true_pos / (true_pos + false_pos)
    end

    # 7. Negative predictive value
    neg_pred_val = if true_neg + false_neg == 0
        if true_neg + false_pos == 0  # all patterns positive
            1.0
        else
            0.0
        end
    else
        true_neg / (true_neg + false_neg)
    end

    # 8. F-score
    fscore = if sensitivity + pos_pred_val == 0
        0.0
    else
        2 * (pos_pred_val * sensitivity) / (pos_pred_val + sensitivity)
    end
    
    # returns results as a dictionary
    return (
        accuracy,
        error_rate,
        sensitivity,
        specificity,
        pos_pred_val,
        neg_pred_val,
        fscore,
        cm
    )
end

#-------------------------------------------------------------------------------------
# confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
#-------------------------------------------------------------------------------------

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Convert real-valued outputs to boolean predictions using the threshold
    bool_outputs = outputs .>= threshold
    
    # Call the boolean version
    return confusionMatrix(bool_outputs, targets)
end

#-------------------------------------------------------------------------------------
# confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
#-------------------------------------------------------------------------------------
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # --- 1. Sanity checks ---
    n_classes_output = size(outputs, 2)
    n_classes_target = size(targets, 2)

    @assert n_classes_output == n_classes_target "Outputs and targets must have the same number of columns (classes)."

    # The first thing this function should do is to check that the number of columns of both matrices is equal and is different 
    # from 2. In case they have only one column,these columns are taken as vectors and the confusionMatrix function 
    # developed in the previous assignment is called.

    n_classes_output = size(outputs, 2)
    n_classes_target = size(targets, 2)
    @assert n_classes_output == n_classes_target "Outputs and targets must have the same number of columns (classes)."
    @assert n_classes_target >= 1 "Outputs and targets must have at least one column (class)."

    # --- 2. If single-column (binary case), call previous 1D confusionMatrix ---
    if n_classes_output == 1
        # Take columns as vectors
        y_pred = vec(outputs)
        y_true = vec(targets)
        return confusionMatrix(y_pred, y_true)  # call the binary version
    end
    
    # --- 3. Multi-class/multi-label
    n_classes = size(targets, 2)
    # Initialize vectors to store metrics per class
    sensitivities = zeros(Float64, n_classes)
    specificities = zeros(Float64, n_classes)
    positive_predicted_values = zeros(Float64, n_classes)
    negative_predicted_values = zeros(Float64, n_classes)
    f1_scores = zeros(Float64, n_classes)

    for class in 1:n_classes
        y_c_pred = outputs[:, class]
        y_c_true = targets[:, class]

        #Counts how many elements are true in both y_c_pred and y_c_true
        true_pos = sum(y_c_pred .& y_c_true) # "Apply the operation element by element across the two arrays."
                                             # & is the element-wise AND operator in Julia.

        # Counts how many samples are not predicted as class c and are not actually of class c
        true_neg = sum(.!y_c_pred .& .!y_c_true)

        # Counts how many samples are predicted as class c but are not actually of class c
        false_pos = sum(y_c_pred .& .!y_c_true)

        # Counts how many samples are not predicted as class c but are actually of class c
        false_neg = sum(.!y_c_pred .& y_c_true)
        
        sensitivities[class] = if true_pos + false_neg == 0
            # All targets are negative
            1.0
        else
            true_pos / (true_pos + false_neg)
        end

        specificities[class] = if true_neg + false_pos == 0
            # All targets are positive
            1.0
        else
            true_neg / (true_neg + false_pos)
        end
        
        #Precision = True Positives / (True Positives + False Positives
        positive_predicted_values[class] = if true_pos + false_pos == 0
            if true_pos + false_neg == 0
                # All targets are negative
                1.0
            else
                # No positive predictions, but there are positive targets
                0.0
            end
        else
            true_pos / (true_pos + false_pos)
        end


        #Negative Predictive Value = True Negatives / (True Negatives + False Negatives)
        negative_predicted_values[class] = if true_neg + false_neg == 0
            if true_neg + false_pos == 0
                # All targets are positive
                1.0
            else
                # No negative predictions, but there are negative targets
                0.0
            end
        else
            true_neg / (true_neg + false_neg)
        end

        #F1 Score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
        f1_scores[class] = if sensitivities[class] + positive_predicted_values[class] == 0
            0.0
        else
            2 * (positive_predicted_values[class] * sensitivities[class]) / (positive_predicted_values[class] + sensitivities[class])
        end
    end 

    
    # Overall accuracy and error rate
    #accuracy = mean(outputs .== targets)
    # In this case (multi-class), we should count whether the whole row was predicted correctly, not each value.
    accuracy = mean([argmax(outputs[i, :]) == argmax(targets[i, :]) for i in 1:size(targets, 1)])
    error_rate = 1 - accuracy

    # Aggregate metrics
   # **Weighted**. In this stratey, the metrics corresponding to each class are averaged, weighting them with 
    # the number of patterns that belong (desired output) to each class. It is therefore suitable when classes are unbalanced.

    if weighted

        # When using the **weighted** strategy, it is necessary to compute how many instances belong 
        # to each class in order to calculate the weighted average.  
        class_counts = vec(sum(targets, dims=1))  # flatten row matrix into vector
        total_instances = sum(class_counts)
        weights = class_counts / total_instances
        sensitivity = sum(weights .* sensitivities)
        specificity = sum(weights .* specificities)
        positive_predicted_values = sum(weights .* positive_predicted_values)
        negative_predicted_values = sum(weights .* negative_predicted_values)
        fscore = sum(weights .* f1_scores)

    # - **Macro**. In this strategy, those metrics such as the PPV or the F-score are calculated as the arithmetic mean of the metrics of each class.
    #  As it is an arithmetic average, it does not consider the possible imbalance between classes.
    else
        sensitivity = mean(sensitivities)
        specificity = mean(specificities)
        positive_predicted_values = mean(positive_predicted_values)
        negative_predicted_values = mean(negative_predicted_values)
        fscore = mean(f1_scores)    
    end

    cm = [sum(targets[:, i] .& outputs[:, j]) for i in 1:n_classes, j in 1:n_classes]


    return (
        accuracy,
        error_rate,
        sensitivity,
        specificity,
        positive_predicted_values,
        negative_predicted_values,
        fscore,
        cm
    )
    
end

#-------------------------------------------------------------------------------------
# confusionMatrix(outputs::AbstractArray{<:Real,2},targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
#-------------------------------------------------------------------------------------
function confusionMatrix(outputs::AbstractArray{<:Real,2},targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    #TODO

    # Convert continuous outputs into Boolean predictions
    classified_outputs = classifyOutputs(outputs, threshold=threshold)

    # Call the Boolean version of confusionMatrix
    return confusionMatrix(classified_outputs, targets; weighted=weighted)
end

#-------------------------------------------------------------------------------------
# confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
#-------------------------------------------------------------------------------------
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    # --- 1. Sanity checks ---
    @assert length(outputs) == length(targets) "Outputs and targets must have the same length."
    @assert (all([in(output, classes) for output in unique(outputs)])) "All output labels must exist in classes."
    @assert (all([in(target, classes) for target in unique(targets)])) "All target labels must exist in classes."

    # --- 2. One-hot encode both using the same classes vector ---
    outputs_onehot = oneHotEncoding(outputs, classes)
    targets_onehot = oneHotEncoding(targets, classes)

    # --- 3. Call the previous confusionMatrix version (the 2D boolean one) ---
    return confusionMatrix(outputs_onehot, targets_onehot; weighted=weighted)

end

#-------------------------------------------------------------------------------------
# confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
#-------------------------------------------------------------------------------------
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #TODO
    # --- 1. Construct the classes vector from both targets and outputs ---
    classes = unique(vcat(targets, outputs))

    # --- 2. Call the previous version (that expects classes explicitly) ---
    return confusionMatrix(outputs, targets, classes; weighted=weighted)

    
end
confusionMatrix

#-------------------------------------------------------------------------------------
# ANNCrossValidation
#-------------------------------------------------------------------------------------
function ANNCrossValidation(topology::AbstractArray{<:Int,1},
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        crossValidationIndices::Array{Int64,1};
        numExecutions::Int=50,
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
        validationRatio::Real=0, maxEpochsVal::Int=20)
    #TODO

    
    (inputs, targets) = dataset

    # 1. **Extract class labels**:
    classes = unique(targets)
    nClasses = size(classes,1)

    # 2. **One-hot encode** the categorical target labels using the `oneHotEncoding` function and the computed `classes`.
    oneHotTargets = oneHotEncoding(targets, classes)

    # 3. **Determine the number of folds** using:
    numFolds = maximum(crossValidationIndices)
    
    # 4. Create one vector per metric (`accuracy`, `error rate`, `sensitivity`, `specificity`, `PPV`, `NPV`, `F1`) to store fold results.
    accuracy = Float32[]
    error_rate = Float32[]
    sensitivity = Float32[]
    specificity = Float32[]
    ppv = Float32[]
    npv = Float32[]
    f1 = Float32[]

    # 5. Initialize a **confusion matrix accumulator** (matrix of real numbers with all entries set to 0) for the **global test confusion matrix**.
    cm_acc = zeros(Float32, nClasses, nClasses)

    # **For each fold**:
    for i in 1:numFolds
        # Extract `train` and `test` subsets for **inputs** and **outputs**, based on the cross-validation indices and current fold number.
        testInputs = inputs[crossValidationIndices .== i, :]
        testTargets = oneHotTargets[crossValidationIndices .== i, :]

        trainValInputs = inputs[crossValidationIndices .!= i, :]
        trainValTargets = oneHotTargets[crossValidationIndices .!= i, :]

        
        # Compute normalization parameters from TRAINING set only
        normParams = calculateMinMaxNormalizationParameters(trainValInputs)

        # Normalize training set IN PLACE
        normalizeMinMax!(trainValInputs, normParams)

        # Normalize test set (returns a new array)
        normalizeMinMax!(testInputs, normParams)

        # Since ANNs are **non-deterministic**, results from a single training per fold may not be representative.  
        # For this reason, train the ANN **multiple times per fold** (as specified in `numExecutions`).
        # **Inside each fold**:
        # 1. Initialize vectors to store the metric results for each execution.  
        accuracy_fold = Float32[]
        error_rate_fold = Float32[]
        sensitivity_fold = Float32[]
        specificity_fold = Float32[]
        ppv_fold = Float32[]
        npv_fold = Float32[]
        f1_fold = Float32[]
        # 2. Create a 3D array of size `(numClasses, numClasses, numExecutions)` to store test confusion matrices from each execution.
        confusionMatrices = zeros(Float32, nClasses, nClasses, numExecutions)
        # 3. For each execution:
        for j in 1:numExecutions
            # If `validationRatio > 0`, split the training set into training and validation sets using `holdOut`.

            if validationRatio > 0
                # Recalculate the validation proportion relative to the train+validation pool
                N = size(inputs,1)
                nTrainVal = size(trainValInputs,1)
                nVal = Int(floor(validationRatio*N)) 
                realValidationRatio = nVal/nTrainVal

                (train_idx, val_idx) = holdOut(nTrainVal, realValidationRatio)

                trainInputs = trainValInputs[train_idx, :]
                valInputs = trainValInputs[val_idx, :]
                trainTargets = trainValTargets[train_idx, :]
                valTargets = trainValTargets[val_idx, :]
                
                # Train the network using `trainClassANN`.
                finalANN, trainLoss, valLoss, testLoss = trainClassANN(
                    topology,
                    (trainInputs, trainTargets),
                    validationDataset = (valInputs, valTargets),
                    testDataset = (testInputs, testTargets)
                )
            end
            # Train the network using `trainClassANN`.
            finalANN, trainLoss, valLoss, testLoss = trainClassANN(
                topology,
                (trainValInputs, trainValTargets),
                validationDataset = (Array{Float32}(undef, 0, 0), falses(0, 0)),
                testDataset = (testInputs, testTargets)
            )

            # Evaluate it on the test set using `confusionMatrix`.
            testOutputs = finalANN(testInputs')
            testPredictions = classifyOutputs(testOutputs')
            testPredictionsClasses = Flux.onecold(testPredictions')
            testTargetClasses = Flux.onecold(testTargets')

            # Store the returned metrics and confusion matrix.
            accuracy_exec, error_rate_exec, sensitivity_exec, specificity_exec, ppv_exec, npv_exec, f1_exec, cm_exec = confusionMatrix(testPredictionsClasses, testTargetClasses)

            push!(accuracy_fold, accuracy_exec)
            push!(error_rate_fold, error_rate_exec)
            push!(sensitivity_fold, sensitivity_exec)
            push!(specificity_fold, specificity_exec)
            push!(ppv_fold, ppv_exec)
            push!(npv_fold, npv_exec)
            push!(f1_fold, f1_exec)
            confusionMatrices[:,:,j] = cm_exec
        end

        # 4. After all executions for this fold:
        # Compute the **average** of each metric vector and store it in the global metric vectors.
        push!(accuracy, mean(accuracy_fold))
        push!(error_rate, mean(error_rate_fold))
        push!(sensitivity, mean(sensitivity_fold))
        push!(specificity, mean(specificity_fold))
        push!(ppv, mean(ppv_fold))
        push!(npv, mean(npv_fold))
        push!(f1, mean(f1_fold))
        # Compute the **mean confusion matrix** using:
        cm_mean_fold = mean(confusionMatrices, dims=3)
        # This returns a 3D array with one slice; you must extract the 2D matrix from it.
        cm_mean_fold = dropdims(cm_mean_fold; dims=3)

        # Add the resulting matrix to the global confusion matrix.
        cm_acc += cm_mean_fold
    end
    return (
        (mean(accuracy), std(accuracy)),
        (mean(error_rate), std(error_rate)),
        (mean(sensitivity), std(sensitivity)),
        (mean(specificity), std(specificity)),
        (mean(ppv), std(ppv)),
        (mean(npv), std(npv)),
        (mean(f1), std(f1)),
        cm_acc / numFolds
    ) 
end

function ANNCrossValidationPCA(topology::AbstractArray{<:Int,1},
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        crossValidationIndices::Array{Int64,1};
        numExecutions::Int=50,
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
        validationRatio::Real=0, maxEpochsVal::Int=20)
    #TODO

    
    (inputs, targets) = dataset

    # 1. **Extract class labels**:
    classes = unique(targets)
    nClasses = size(classes,1)

    # 2. **One-hot encode** the categorical target labels using the `oneHotEncoding` function and the computed `classes`.
    oneHotTargets = oneHotEncoding(targets, classes)

    # 3. **Determine the number of folds** using:
    numFolds = maximum(crossValidationIndices)
    
    # 4. Create one vector per metric (`accuracy`, `error rate`, `sensitivity`, `specificity`, `PPV`, `NPV`, `F1`) to store fold results.
    accuracy = Float32[]
    error_rate = Float32[]
    sensitivity = Float32[]
    specificity = Float32[]
    ppv = Float32[]
    npv = Float32[]
    f1 = Float32[]

    # 5. Initialize a **confusion matrix accumulator** (matrix of real numbers with all entries set to 0) for the **global test confusion matrix**.
    cm_acc = zeros(Float32, nClasses, nClasses)

    # **For each fold**:
    for i in 1:numFolds
        # Extract `train` and `test` subsets for **inputs** and **outputs**, based on the cross-validation indices and current fold number.
        testInputs = inputs[crossValidationIndices .== i, :]
        testTargets = oneHotTargets[crossValidationIndices .== i, :]

        trainValInputs = inputs[crossValidationIndices .!= i, :]
        trainValTargets = oneHotTargets[crossValidationIndices .!= i, :]

        
        # Compute normalization parameters from TRAINING set only
        normParams = calculateMinMaxNormalizationParameters(trainValInputs)

        # Normalize training set IN PLACE
        normalizeMinMax!(trainValInputs, normParams)

        # Normalize test set (returns a new array)
        normalizeMinMax!(testInputs, normParams)

        pca_model = PCA_model(variance_ratio=0.95)
        # Train the PCA model
        pca_mach = machine(pca_model, MLJ.table(trainValInputs))
        MLJ.fit!(pca_mach, verbosity=0)
        # Transform the data
        trainValInputs = MLJBase.matrix(MLJBase.transform(pca_mach, MLJ.table(trainValInputs)))
        testInputs  = MLJBase.matrix(MLJBase.transform(pca_mach, MLJ.table(testInputs)))

        # Since ANNs are **non-deterministic**, results from a single training per fold may not be representative.  
        # For this reason, train the ANN **multiple times per fold** (as specified in `numExecutions`).
        # **Inside each fold**:
        # 1. Initialize vectors to store the metric results for each execution.  
        accuracy_fold = Float32[]
        error_rate_fold = Float32[]
        sensitivity_fold = Float32[]
        specificity_fold = Float32[]
        ppv_fold = Float32[]
        npv_fold = Float32[]
        f1_fold = Float32[]
        # 2. Create a 3D array of size `(numClasses, numClasses, numExecutions)` to store test confusion matrices from each execution.
        confusionMatrices = zeros(Float32, nClasses, nClasses, numExecutions)
        # 3. For each execution:
        for j in 1:numExecutions
            # If `validationRatio > 0`, split the training set into training and validation sets using `holdOut`.

            if validationRatio > 0
                # Recalculate the validation proportion relative to the train+validation pool
                N = size(inputs,1)
                nTrainVal = size(trainValInputs,1)
                nVal = Int(floor(validationRatio*N)) 
                realValidationRatio = nVal/nTrainVal

                (train_idx, val_idx) = holdOut(nTrainVal, realValidationRatio)

                trainInputs = trainValInputs[train_idx, :]
                valInputs = trainValInputs[val_idx, :]
                trainTargets = trainValTargets[train_idx, :]
                valTargets = trainValTargets[val_idx, :]
                
                # Train the network using `trainClassANN`.
                finalANN, trainLoss, valLoss, testLoss = trainClassANN(
                    topology,
                    (trainInputs, trainTargets),
                    validationDataset = (valInputs, valTargets),
                    testDataset = (testInputs, testTargets)
                )
            end
            # Train the network using `trainClassANN`.
            finalANN, trainLoss, valLoss, testLoss = trainClassANN(
                topology,
                (trainValInputs, trainValTargets),
                validationDataset = (Array{Float32}(undef, 0, 0), falses(0, 0)),
                testDataset = (testInputs, testTargets)
            )

            # Evaluate it on the test set using `confusionMatrix`.
            testOutputs = finalANN(testInputs')
            testPredictions = classifyOutputs(testOutputs')
            testPredictionsClasses = Flux.onecold(testPredictions')
            testTargetClasses = Flux.onecold(testTargets')

            # Store the returned metrics and confusion matrix.
            accuracy_exec, error_rate_exec, sensitivity_exec, specificity_exec, ppv_exec, npv_exec, f1_exec, cm_exec = confusionMatrix(testPredictionsClasses, testTargetClasses)

            push!(accuracy_fold, accuracy_exec)
            push!(error_rate_fold, error_rate_exec)
            push!(sensitivity_fold, sensitivity_exec)
            push!(specificity_fold, specificity_exec)
            push!(ppv_fold, ppv_exec)
            push!(npv_fold, npv_exec)
            push!(f1_fold, f1_exec)
            confusionMatrices[:,:,j] = cm_exec
        end

        # 4. After all executions for this fold:
        # Compute the **average** of each metric vector and store it in the global metric vectors.
        push!(accuracy, mean(accuracy_fold))
        push!(error_rate, mean(error_rate_fold))
        push!(sensitivity, mean(sensitivity_fold))
        push!(specificity, mean(specificity_fold))
        push!(ppv, mean(ppv_fold))
        push!(npv, mean(npv_fold))
        push!(f1, mean(f1_fold))
        # Compute the **mean confusion matrix** using:
        cm_mean_fold = mean(confusionMatrices, dims=3)
        # This returns a 3D array with one slice; you must extract the 2D matrix from it.
        cm_mean_fold = dropdims(cm_mean_fold; dims=3)

        # Add the resulting matrix to the global confusion matrix.
        cm_acc += cm_mean_fold
    end
    return (
        (mean(accuracy), std(accuracy)),
        (mean(error_rate), std(error_rate)),
        (mean(sensitivity), std(sensitivity)),
        (mean(specificity), std(specificity)),
        (mean(ppv), std(ppv)),
        (mean(npv), std(npv)),
        (mean(f1), std(f1)),
        cm_acc / numFolds
    ) 
end

#--------------------------------------------------------------------------------
## Normalization functions
#--------------------------------------------------------------------------------
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end;

#
function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    clamp!(dataset, 0.0, 1.0)
    # eliminate any atribute that do not add information
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;


function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset , calculateMinMaxNormalizationParameters(dataset));
end;

function normalizeMinMax( dataset::AbstractArray{<:Real,2},      
                normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    normalizeMinMax!(copy(dataset), normalizationParameters);
end;


#--------------------------------------------------------------------------------
## One-hot encoding function
#--------------------------------------------------------------------------------
function oneHotEncoding(feature::AbstractArray{<:Any,1},      
        classes::AbstractArray{<:Any,1})

    """
    Parameters
    ----------
    feature :: AbstractVector
        The input vector of categorical values to be encoded.
    classes :: AbstractVector
        The list/array of unique classes used as encoding reference.
    """
    """
    AbstractArray{<:Any,1} :
    AbstractArray → not restricted to just Vector, could be any 1-dimensional array type.
    1 → one-dimensional array (i.e. a vector).
    <:Any → element type can be any subtype of Any (which basically means no restriction at all).
    So effectively, AbstractArray{<:Any,1} is just a very general way of saying:
    👉 “Accept any 1-D array, regardless of element type.”
    """
    # Defensive: ensure feature is a vector
    # Check that all feature values exist in the set of classes
    @assert(all([in(value, classes) for value in feature]))
    
    # Number of classes
    numClasses = length(classes)
    
    # Defensive: require at least two classes
    @assert(numClasses > 1)
    
    if (numClasses == 2)
        # Special case: binary classification, use a single column
        oneHot = reshape(feature .== classes[1], :, 1)
    else
        # General case: more than two classes
        oneHot = BitArray{2}(undef, length(feature), numClasses)
        for numClass = 1:numClasses
            # Mark 1 where feature matches the class
            oneHot[:, numClass] .= (feature .== classes[numClass])
        end
    end


    """
    Returns
    -------
    AbstractArray
        A one-hot encoded array:
        - Shape (n, 1) if there are 2 classes (binary case).
        - Shape (n, numClasses) if there are more than 2 classes.
    """
    return oneHot
end

# utils.jl

"""
    oneHotEncoding!(feature::AbstractVector)

One-hot encode a categorical feature vector.

# Arguments
- `feature`: A 1-D array of categorical values.

# Returns
- `AbstractArray{Bool,2}`: One-hot encoded array:
    - Shape `(n, 1)` for binary features.
    - Shape `(n, num_classes)` for multi-class features.
"""
function oneHotEncoding!(feature::AbstractVector)
    # Get the unique classes
    classes = unique(feature)
    numClasses = length(classes)

    @assert numClasses > 1 "Need at least 2 classes to one-hot encode"

    if numClasses == 2
        # Binary case: single column
        oneHot = reshape(feature .== classes[1], :, 1)
    else
        # Multi-class case: create 2D Bool array
        oneHot = falses(length(feature), numClasses)
        for i in 1:numClasses
            oneHot[:, i] .= (feature .== classes[i])
        end
    end

    return oneHot
end

# ===============================================================
# modelCrossValidation
# ---------------------------------------------------------------
# This function performs k-fold cross-validation for different
# types of models (ANN, SVM, Decision Tree, kNN).  
# It trains the model on (k-1) folds and evaluates it on the
# remaining fold, repeating the process for all folds.  
# For each fold, it computes performance metrics and accumulates  
# the confusion matrix.  
# The function returns the mean and standard deviation of all  
# metrics across folds, along with the averaged confusion matrix.
# ===============================================================
function modelCrossValidation(
        modelType::Symbol, modelHyperparameters::Dict,
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        crossValidationIndices::Array{Int64,1})
    #TODO

   # The function will begin by checking whether the model to be trained is a neural network, by examining the `modelType` parameter. 
   if modelType == :ANN 
      # If this is the case, it will call the `ANNCrossValidation` function, passing the hyperparameters provided in `modelHyperparameters`.
      # Keep in mind that many of the hyperparameters for neural networks may not be defined in the dictionary. 
      topology = get(modelHyperparameters, "topology", [5,4,3])
      numExecutions = get(modelHyperparameters, "numExecutions", 5)
      transferFunctions = get(modelHyperparameters, "transferFunctions", fill(σ, length(topology)))
      maxEpochs = get(modelHyperparameters, "maxEpochs", 50)
      minLoss = get(modelHyperparameters, "minLoss", 0.0)
      learningRate = get(modelHyperparameters, "learningRate", 0.01)
      validationRatio = get(modelHyperparameters, "validationRatio", 0)
      maxEpochsVal = get(modelHyperparameters, "maxEpochsVal", 10)
      return ANNCrossValidation(topology, dataset, crossValidationIndices; numExecutions=numExecutions, transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)
   end   
   # If a different type of model is to be trained, the logic continues similarly to the previous exercise:
   (inputs, targets) = dataset
   classes = unique(targets)
   nClasses = size(classes,1)

   # - Create seven vectors to store the results of the metrics for each fold.
   accuracy = Float32[]
   error_rate = Float32[]
   sensitivity = Float32[]
   specificity = Float32[]
   ppv = Float32[]
   npv = Float32[]
   f1 = Float32[]
   # - Create a 2D array to accumulate the confusion matrix, initialized with zeros.
   cm_acc = zeros(Float32, nClasses, nClasses)

   # A key modification when using models from the MLJ library is to **convert the target labels to strings** before training any model.  
   # This helps prevent errors caused by internal type mismatches in some model implementations.
   targets = string.(targets);

   # Additionally, it will be necessary to compute the vector of unique classes, just like in the previous exercise.  
   # classes = unique(targets);  # already done

   # Once these initial steps are completed, the cross-validation loop can begin.
   numFolds = maximum(crossValidationIndices)

   # In each iteration, the following steps are performed:
   for i in 1:numFolds
      # 1. Extract the training and test input matrices and the corresponding target vectors.  
         # These should be of type `AbstractArray{<:Any,1}` for the targets.
      testInputs = inputs[crossValidationIndices .== i, :]
      testTargets = targets[crossValidationIndices .== i]

      trainInputs = inputs[crossValidationIndices .!= i, :]
      trainTargets = targets[crossValidationIndices .!= i]

      # Compute normalization parameters from TRAINING set only
      normParams = calculateMinMaxNormalizationParameters(trainInputs)

      # Normalize training set IN PLACE
      normalizeMinMax!(trainInputs, normParams)

      # Normalize test set (returns a new array)
      normalizeMinMax!(testInputs, normParams)

      # 2. Create the model with the specified hyperparameters.

      # 3. For MLJ models (SVM, Decision Tree, kNN):
      # - Instantiate the model using the appropriate constructor: `SVMClassifier`, `DTClassifier`, or `kNNClassifier`, depending on `modelType`.
      if modelType == :SVC
         kernel = get(modelHyperparameters, "kernel", "linear")
         C = get(modelHyperparameters, "C", 1.0)
         gamma = get(modelHyperparameters, "gamma", 2.0)
         degree = get(modelHyperparameters, "degree", 3)
         coef0 = get(modelHyperparameters, "coef0", 1.0)
         
         if kernel == "linear"
            model = SVMClassifier(kernel=LIBSVM.Kernel.Linear, cost=Float64(C))

         elseif kernel == "rbf"
            model = SVMClassifier(kernel=LIBSVM.Kernel.RadialBasis, cost=Float64(C), gamma=Float64(gamma), degree=Int32(degree))

         elseif kernel == "sigmoid"
            model = SVMClassifier(kernel=LIBSVM.Kernel.Sigmoid, cost=Float64(C), gamma=Float64(gamma), coef0=Float64(coef0))

         elseif kernel == "poly"
            model = SVMClassifier(kernel=LIBSVM.Kernel.Polynomial, cost=Float64(C), gamma=Float64(gamma), degree=Int32(degree), coef0=Float64(coef0))

         else
            error("Unsupported SVM kernel: $kernel. Please use one of [\"linear\", \"rbf\", \"sigmoid\", \"poly\"].")
         end
         
      elseif modelType == :DecisionTreeClassifier
         max_depth = get(modelHyperparameters, "max_depth", 4)
         rng = get(modelHyperparameters, "rng", Random.MersenneTwister(1))
         model = DTClassifier(max_depth=max_depth, rng=rng)
      
      elseif modelType == :KNeighborsClassifier
         K = get(modelHyperparameters, "K", 3)
         model = kNNClassifier(K=K)
      
      else
         error("Unsupported model type: $modelType. Please use one of [:ANN, :SVC, :DecisionTreeClassifier, :KNeighborsClassifier].")
      end

      # - Wrap the model in a `machine` with the training data.
      mach = machine(model, MLJ.table(trainInputs), categorical(trainTargets))

      # - Train the model using `fit!`.
      MLJ.fit!(mach, verbosity=0)
         
      # 4. Perform predictions on the test data using `predict`.
      testOutputs = MLJ.predict(mach, MLJ.table(testInputs));
      # - For Decision Trees and kNN, use `mode` to convert the probabilistic predictions into categorical labels:
      if modelType == :DecisionTreeClassifier || modelType == :KNeighborsClassifier
         testOutputs = mode.(testOutputs)
      end
      # - For SVMs, the output of `predict` can be compared directly with the ground truth, since it returns a `CategoricalArray`.

      # Once the predicted labels for the test set are available, the evaluation metrics and the confusion matrix should be computed using the `confusionMatrix` function.
      accuracy_fold, error_rate_fold, sensitivity_fold, specificity_fold, ppv_fold, npv_fold, f1_fold, cm_fold = confusionMatrix(testOutputs, testTargets)

      # - The metrics returned should be stored in their respective positions within the metric vectors.
      push!(accuracy, accuracy_fold)
      push!(error_rate, error_rate_fold)
      push!(sensitivity, sensitivity_fold)
      push!(specificity, specificity_fold)
      push!(ppv, ppv_fold)
      push!(npv, npv_fold)
      push!(f1, f1_fold)
      
      # - The confusion matrix obtained for each fold should be **added** to a global confusion matrix for the test set.
      cm_acc += cm_fold
   end
   
   return (
      (mean(accuracy), std(accuracy)),
      (mean(error_rate), std(error_rate)),
      (mean(sensitivity), std(sensitivity)),
      (mean(specificity), std(specificity)),
      (mean(ppv), std(ppv)),
      (mean(npv), std(npv)),
      (mean(f1), std(f1)),
      cm_acc / numFolds
   )
end

function modelCrossValidationPCA(
        modelType::Symbol, modelHyperparameters::Dict,
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        crossValidationIndices::Array{Int64,1})
    #TODO

   # The function will begin by checking whether the model to be trained is a neural network, by examining the `modelType` parameter. 
   if modelType == :ANN 
      # If this is the case, it will call the `ANNCrossValidation` function, passing the hyperparameters provided in `modelHyperparameters`.
      # Keep in mind that many of the hyperparameters for neural networks may not be defined in the dictionary. 
      topology = get(modelHyperparameters, "topology", [5,4,3])
      numExecutions = get(modelHyperparameters, "numExecutions", 5)
      transferFunctions = get(modelHyperparameters, "transferFunctions", fill(σ, length(topology)))
      maxEpochs = get(modelHyperparameters, "maxEpochs", 50)
      minLoss = get(modelHyperparameters, "minLoss", 0.0)
      learningRate = get(modelHyperparameters, "learningRate", 0.01)
      validationRatio = get(modelHyperparameters, "validationRatio", 0)
      maxEpochsVal = get(modelHyperparameters, "maxEpochsVal", 10)
      return ANNCrossValidation(topology, dataset, crossValidationIndices; numExecutions=numExecutions, transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)
   end   
   # If a different type of model is to be trained, the logic continues similarly to the previous exercise:
   (inputs, targets) = dataset
   classes = unique(targets)
   nClasses = size(classes,1)

   # - Create seven vectors to store the results of the metrics for each fold.
   accuracy = Float32[]
   error_rate = Float32[]
   sensitivity = Float32[]
   specificity = Float32[]
   ppv = Float32[]
   npv = Float32[]
   f1 = Float32[]
   # - Create a 2D array to accumulate the confusion matrix, initialized with zeros.
   cm_acc = zeros(Float32, nClasses, nClasses)

   # A key modification when using models from the MLJ library is to **convert the target labels to strings** before training any model.  
   # This helps prevent errors caused by internal type mismatches in some model implementations.
   targets = string.(targets);

   # Additionally, it will be necessary to compute the vector of unique classes, just like in the previous exercise.  
   # classes = unique(targets);  # already done

   # Once these initial steps are completed, the cross-validation loop can begin.
   numFolds = maximum(crossValidationIndices)

   # In each iteration, the following steps are performed:
   for i in 1:numFolds
      # 1. Extract the training and test input matrices and the corresponding target vectors.  
         # These should be of type `AbstractArray{<:Any,1}` for the targets.
      testInputs = inputs[crossValidationIndices .== i, :]
      testTargets = targets[crossValidationIndices .== i]

      trainInputs = inputs[crossValidationIndices .!= i, :]
      trainTargets = targets[crossValidationIndices .!= i]

      # Compute normalization parameters from TRAINING set only
      normParams = calculateMinMaxNormalizationParameters(trainInputs)

      # Normalize training set IN PLACE
      normalizeMinMax!(trainInputs, normParams)

      # Normalize test set (returns a new array)
      normalizeMinMax!(testInputs, normParams)

      
      pca_model = PCA_model(variance_ratio=0.95)
      # Train the PCA model
      pca_mach = machine(pca_model, MLJ.table(trainInputs))
      MLJ.fit!(pca_mach, verbosity=0)
      # Transform the data
      trainInputs = MLJBase.matrix(MLJBase.transform(pca_mach, MLJ.table(trainInputs)))
      testInputs  = MLJBase.matrix(MLJBase.transform(pca_mach, MLJ.table(testInputs)))


      # 2. Create the model with the specified hyperparameters.

      # 3. For MLJ models (SVM, Decision Tree, kNN):
      # - Instantiate the model using the appropriate constructor: `SVMClassifier`, `DTClassifier`, or `kNNClassifier`, depending on `modelType`.
      if modelType == :SVC
         kernel = get(modelHyperparameters, "kernel", "linear")
         C = get(modelHyperparameters, "C", 1.0)
         gamma = get(modelHyperparameters, "gamma", 2.0)
         degree = get(modelHyperparameters, "degree", 3)
         coef0 = get(modelHyperparameters, "coef0", 1.0)
         
         if kernel == "linear"
            model = SVMClassifier(kernel=LIBSVM.Kernel.Linear, cost=Float64(C))

         elseif kernel == "rbf"
            model = SVMClassifier(kernel=LIBSVM.Kernel.RadialBasis, cost=Float64(C), gamma=Float64(gamma), degree=Int32(degree))

         elseif kernel == "sigmoid"
            model = SVMClassifier(kernel=LIBSVM.Kernel.Sigmoid, cost=Float64(C), gamma=Float64(gamma), coef0=Float64(coef0))

         elseif kernel == "poly"
            model = SVMClassifier(kernel=LIBSVM.Kernel.Polynomial, cost=Float64(C), gamma=Float64(gamma), degree=Int32(degree), coef0=Float64(coef0))

         else
            error("Unsupported SVM kernel: $kernel. Please use one of [\"linear\", \"rbf\", \"sigmoid\", \"poly\"].")
         end
         
      elseif modelType == :DecisionTreeClassifier
         max_depth = get(modelHyperparameters, "max_depth", 4)
         rng = get(modelHyperparameters, "rng", Random.MersenneTwister(1))
         model = DTClassifier(max_depth=max_depth, rng=rng)
      
      elseif modelType == :KNeighborsClassifier
         K = get(modelHyperparameters, "K", 3)
         model = kNNClassifier(K=K)
      
      else
         error("Unsupported model type: $modelType. Please use one of [:ANN, :SVC, :DecisionTreeClassifier, :KNeighborsClassifier].")
      end

      # - Wrap the model in a `machine` with the training data.
      mach = machine(model, MLJ.table(trainInputs), categorical(trainTargets))

      # - Train the model using `fit!`.
      MLJ.fit!(mach, verbosity=0)
         
      # 4. Perform predictions on the test data using `predict`.
      testOutputs = MLJ.predict(mach, MLJ.table(testInputs));
      # - For Decision Trees and kNN, use `mode` to convert the probabilistic predictions into categorical labels:
      if modelType == :DecisionTreeClassifier || modelType == :KNeighborsClassifier
         testOutputs = mode.(testOutputs)
      end
      # - For SVMs, the output of `predict` can be compared directly with the ground truth, since it returns a `CategoricalArray`.

      # Once the predicted labels for the test set are available, the evaluation metrics and the confusion matrix should be computed using the `confusionMatrix` function.
      accuracy_fold, error_rate_fold, sensitivity_fold, specificity_fold, ppv_fold, npv_fold, f1_fold, cm_fold = confusionMatrix(testOutputs, testTargets)

      # - The metrics returned should be stored in their respective positions within the metric vectors.
      push!(accuracy, accuracy_fold)
      push!(error_rate, error_rate_fold)
      push!(sensitivity, sensitivity_fold)
      push!(specificity, specificity_fold)
      push!(ppv, ppv_fold)
      push!(npv, npv_fold)
      push!(f1, f1_fold)
      
      # - The confusion matrix obtained for each fold should be **added** to a global confusion matrix for the test set.
      cm_acc += cm_fold
   end
   
   return (
      (mean(accuracy), std(accuracy)),
      (mean(error_rate), std(error_rate)),
      (mean(sensitivity), std(sensitivity)),
      (mean(specificity), std(specificity)),
      (mean(ppv), std(ppv)),
      (mean(npv), std(npv)),
      (mean(f1), std(f1)),
      cm_acc / numFolds
   )
end