function output = execute_mlp_4layers(input,w1,w2,w3,w4,b1,b2,b3,b4)
%% function output = execute_mlp_3layers(input,w1,w2,w3,b1,b2,b3)
% 3-layer MLP feedforward 
% 1) hidden layer 1: w1 [input_size,hidenlayer1_size],      b1 [hidenlayer1_size,1], relu
% 2) hidden layer 2: w2 [hidenlayer1_size,hidenlayer2_size],b2 [hidenlayer2_size,1], relu
% 3) output layer 3: w3 [hidenlayer2_size,Output_size],     b3[hidenlayer3_size,1] , sigmoid
% The MLP is trained by "MLP_python.py"
%
% Yu Zheng, RASLab, FAMU-FSU College of Engineering, Tallahassee, 2021, Aug.

    output_layer1 = activ_layer1(w1.'*input + kron(ones(1,size(input,2)), b1));
    output_layer2 = activ_layer2(w2.'*output_layer1 + kron(ones(1,size(output_layer1,2)), b2));
    output_layer3 = activ_layer2(w3.'*output_layer2 + kron(ones(1,size(output_layer2,2)),b3));
    output        = activ_layer3(w4.'*output_layer3 + kron(ones(1,size(output_layer3,2)),b4));
end

function y = activ_layer1(x)
%% relu
% alpha = 0.01;       % For LeakyRelu

y = x.*(x>=0)+0*(x<0);

end

function y = activ_layer2(x)
%% relu
% alpha = 0.01;       % For LeakyRelu

y = x.*(x>=0)+0*(x<0);

end


function y = activ_layer3(x)
%% sigmoid

y = 1./(1+exp(-x));

end