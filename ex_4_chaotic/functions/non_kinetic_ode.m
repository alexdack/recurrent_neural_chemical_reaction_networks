function dpdt = non_kinetic_ode(t, p, r, k, a1, a2, b1, b2, d1, d2)
    f1 = a1*(p(1)./(1+b1*p(1)));
    f2 = a2*p(2)./(1+b2*p(2));

    dx =  r*p(1)*(1-k*p(1)) -f1.*p(2);
    dy =  -d1*p(2)+f1.*p(2)-f2.*p(3);
    dz = -d2*p(3)+ f2.*p(3);

    dpdt = [dx;dy;dz];
end