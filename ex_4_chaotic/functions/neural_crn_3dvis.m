function dpdt = neural_crn_3dvis(t, y, gamma, betas, time_scale, w1, alphas, b1)
    gamma = double(gamma);
    betas = double(betas);
    time_scale = double(time_scale);   
    w1 = double(w1);   
    alphas = double(alphas);   
    b1 = double(b1);   
    hidden_nodes = length(b1);
    gammas = gamma*ones(hidden_nodes, 1);
    y_feedback = y(1:3);
    y_hidden = y(4:hidden_nodes+3);
    feedback_nodes = betas + y_feedback.*(alphas'*y_hidden);
    compute_nodes = (gammas + (w1'*y_feedback + b1').*y_hidden - y_hidden.^2)./time_scale;
    dpdt = [feedback_nodes; compute_nodes];
end