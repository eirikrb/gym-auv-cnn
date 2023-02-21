import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, sqrt

m = 23.8
x_g = 0.046
I_z = 1.760
X_udot = -2.0
Y_vdot = -10.0
Y_rdot = 0.0
N_rdot = -1.0
N_vdot = 0.0
X_u = -2.0
Y_v = -7.0
Y_r = -0.1
N_v = -0.1
N_r = -0.5
X_uu = -1.32742
Y_vv = -80
Y_rr = 0.3
N_vv = -1.5
N_rr = -9.1
Y_uvb = -0.5*1000*np.pi*1.24*(0.15/2)**2
Y_uvf = -1000*3*0.0064
Y_urf = -0.4*Y_uvf
N_uvb = (-0.65*1.08 + 0.4)*Y_uvb
N_uvf = -0.4*Y_uvf
N_urf = -0.4*Y_urf
Y_uudr = 19.2
N_uudr = -0.4*Y_uudr

MAX_SPEED = 2

M =  np.array([[m - X_udot, 0, 0],
    [0, m - Y_vdot, m*x_g - Y_rdot],
    [0, m*x_g - N_vdot, I_z - N_rdot]]
)   
M_inv = np.linalg.inv(M)

D =  np.array([
    [2.0, 0, 0],
    [0, 7.0, -2.5425],
    [0, -2.5425, 1.422]
])

def C(nu):
    u = nu[0]
    v = nu[1]
    r = nu[2]
    C = np.array([
        [0, 0, -33.8*v + 11.748*r],
        [0, 0, 25.8*u],
        [33.8*v - 11.748*r, -25.8*u, 0]
    ])  
    return C

B = np.array([
        [1, 0],
        [0, -1.7244],
        [0, 1],
    ])

def N(nu):
    u = nu[0]
    v = nu[1]
    r = nu[2]
    N = np.array([
        [-X_u, 0, 0],
        [0, -Y_v, m*u - Y_r],
        [0, -N_v, m*x_g*u-N_r]
    ])  
    return N


def export_ship_model(model_type = 'simplified') -> AcadosModel:

    model_name = 'ship_ode'
    
    #Variables
    x = SX.sym('x')
    y = SX.sym('y')
    phi = SX.sym('phi')

    eta = vertcat(x,y,phi)

    x_dot = SX.sym('x_dot')
    y_dot = SX.sym('y_dot')
    phi_dot = SX.sym('phi_dot')

    eta_dot = vertcat(x_dot,y_dot,phi_dot)

    u = SX.sym('u')
    v = SX.sym('v')
    r = SX.sym('r')

    nu = vertcat(u,v,r)

    u_dot = SX.sym('u_dot')
    v_dot = SX.sym('v_dot')
    r_dot = SX.sym('r_dot')

    nu_dot = vertcat(u_dot,v_dot,r_dot)

    state = vertcat(eta,nu)
    state_dot = vertcat(eta_dot,nu_dot)

    F_u = SX.sym('F_u')
    F_r = SX.sym('F_r')
    F = vertcat(F_u,F_r)
    x_obs = SX.sym('x_obs')
    y_obs = SX.sym('y_obs')
    r_obs = SX.sym('r_obs')
    state_obs = vertcat(x_obs,y_obs,r_obs)
    origin_term_set = SX.sym('origin_term_set')
    
    #Dynamics
    # nu_expl = M_inv@(-C@nu - D@nu + B@F)

    cos_phi = cos(phi)
    sin_phi = sin(phi)

    eta_expl = vertcat(
        cos_phi*u -sin_phi*v,
        sin_phi*u + cos_phi*v,
        r
    )
    
    eta_impl = eta_dot - eta_expl
    if model_type == 'realistic':

        nu_impl = vertcat(
            M[0,0]*u_dot + (-33.8*v + 11.748*r)*r + 2*u - F_u,
            M[1,1]*v_dot + M[1,2]*r_dot + 25.8*u*r + 7*v - 2.5425*r - 1.7244*F_r,
            M[2,1]*v_dot + M[2,2]*r_dot + (33.8*v - 11.748*r)*u - 25.8*u*v - 2.5425*v + 1.422*r - F_r
        )
        nu_expl = vertcat(
            M_inv[0,0]*(-2.0*u - (-33.8*v + 11.748*r)*r + F_u),
            M_inv[1,1]*(-7.0*v + 2.5425*r - 25.8*u*r -1.7244*F_r) + M_inv[1,2]*(2.5425*v - 1.422*r - (33.8*v - 11.748*r)*u +25.8*u*v + F_r),
            M_inv[2,1]*(-7.0*v + 2.5425*r - 25.8*u*r -1.7244*F_r) + M_inv[2,2]*(2.5425*v - 1.422*r - (33.8*v - 11.748*r)*u +25.8*u*v + F_r)
        )

    elif model_type == 'simplified':

        nu_impl = vertcat(
            (m-X_udot)*u_dot + (-33.8*v + 11.748*r)*r -X_u*u - F_u,
            (m - Y_vdot)*v_dot + (m*x_g - Y_rdot)*r_dot -Y_v*v + (m*u - Y_r)*r,
            (m*x_g - N_vdot)*v_dot + (I_z - N_rdot)*r_dot -N_v*v + (m*x_g*u-N_r)*r - F_r
        )
        nu_expl = vertcat(
            M_inv[0,0]*(X_u*u + F_u),
            M_inv[1,1]*(Y_v*v - (m*u - Y_r)*r) + M_inv[1,2]*(N_v*v - (m*x_g*u-N_r)*r + F_r),
            M_inv[2,1]*(Y_v*v - (m*u - Y_r)*r) + M_inv[2,2]*(N_v*v - (m*x_g*u-N_r)*r + F_r)
        )
    
    # (-C_nu*nu - D_nu*nu + tau) = (
    #     -2.0*u - (-33.8*v + 11.748*r)*r + F_u,
    #     -7.0*v + 2.5425*r - 25.8*u*r -1.7244*F_r,
    #     2.5425*v - 1.422*r - (33.8*v - 11.748*r)*u +25.8*u*v + F_r
    # )
    # (-N_nu*nu + tau) = (
    #     X_u*u + F_u,
    #     Y_v*v - (m*u - Y_r)*r,
    #     N_v*v - (m*x_g*u-N_r)*r + F_r
    # )

    f_impl = vertcat(eta_impl,nu_impl)
    f_expl = vertcat(eta_expl,nu_expl)

    con_h_expr = sqrt((x - x_obs)**2 + (y - y_obs)**2)
    con_h_expr_e = con_h_expr

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = state
    model.xdot = state_dot
    model.u = F
    model.p = vertcat(state_obs)
    #model.con_h_expr = con_h_expr
    #model.con_h_expr_e = con_h_expr_e
    model.name = model_name

    return model