% function that computes the neural network as if we are operating at the degenerate system 
function res = degenerate_neural_subsystem(x1, w1, alphas, b1, gamma)
    input_each_node = w1'*x1+b1';
    non_linear =  0.5*(input_each_node + sqrt(input_each_node.^2 + 4*gamma));
    res = non_linear'*alphas;
end