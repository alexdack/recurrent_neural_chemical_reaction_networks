% function that maps the neural network weights to the entire ODE system 
function dpdt = neural_crn_2dvis(t, y, betas, gamma, time_scale, w1, alphas, b1)
    betas = double(betas);
    gamma = double(gamma);
    alphas = double(alphas);
    w1 = double(w1);
    b1 = double(b1);

    hidden_nodes = length(b1);
    gammas = gamma*ones(hidden_nodes, 1);
    y_feedback = y(1:2);
    y_hidden = y(3:hidden_nodes+2);
    feedback_nodes = betas + y_feedback.*(alphas'*y_hidden);
    compute_nodes = (gammas + (w1'*y_feedback + b1').*y_hidden - y_hidden.^2)./time_scale;
    dpdt = [feedback_nodes; compute_nodes];
end