% function that maps the neural network weights to the entire ODE system 
function dpdt = degenerate_neural_crn_2dvis(x1, x2, betas, gamma, w1, alphas, b1)
    betas = double(betas);
    gamma = double(gamma);
    w1 = double(w1);
    alphas = double(alphas);
    b1 = double(b1);
    
    x = [x1; x2];
    A = w1'*x + b1';
    ys = (A + sqrt(A.^2 + 4*gamma))./(2);
    sums_ys = alphas'*ys;
    dpdt = betas + x.*sums_ys;

end