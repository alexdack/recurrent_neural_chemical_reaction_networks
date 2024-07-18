function dpdt = neural_crn_1dvis(t, y, gamma, betas, time_scale, w1, alphas, b1)
    gamma = double(gamma);
    betas = double(betas);
    time_scale = double(time_scale);
    w1 = double(w1);
    alphas = double(alphas);
    b1 = double(b1);
   
   
    % function that maps the neural network weights to the entire ODE
    % system for a one dimensional visible state
    hidden_nodes = length(b1);
    gammas = gamma*ones(hidden_nodes, 1);
    y_feedback = y(1);
    y_hidden = y(2:hidden_nodes+1);
    feedback_nodes = betas + y_feedback*(alphas'*y_hidden);
    compute_nodes = (gammas + (w1'*y_feedback + b1').*y_hidden - y_hidden.^2)./time_scale;
    dpdt = [feedback_nodes; compute_nodes];
end