function dpdt = non_kinetic_ode_1(t, p)
%non_kinetic_ode defines the ODE for the non kinetic ODE that is being
%approximated 
    x = p(1);
    dpdt = sin(x);
end