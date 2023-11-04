import numpy as np

def xi(a,b,d,r):
    return (b-a/2+(d*r)/(2*(r-d)))/(2*b-a/2-d+(d*r)/(2*(r-d)))

def R(a,b,d,r):
    xi_val = xi(a,b,d,r)
    return xi_val**2 - (r*(1-xi_val)**2)/(r-d)

def x_func(a,b,d,r):
    xi_val = xi(a,b,d,r)
    r_val = R(a,b,d,r)
    return [(xi_val+np.sqrt(r_val))/2,(1-xi_val)/2,(1-xi_val)/2,(xi_val-np.sqrt(r_val))/2]