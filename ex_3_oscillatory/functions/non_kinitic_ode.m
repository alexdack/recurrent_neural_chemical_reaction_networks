function dpdt = non_kinitic_ode(t,p)
    x = p(1);
    y = p(2);
    dx = 6+4*besselj(0,1.5*x) -y;
    dy = x-4;
    dpdt = [dx; dy];
end