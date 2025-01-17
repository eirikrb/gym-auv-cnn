import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, sqrt, fmod, fabs

m = 23.8
x_g = 0.046
I_z = 1.760
X_udot = -2.0
Y_vdot = -10.0
Y_rdot = 0.0
N_rdot = -1.0
N_vdot = 0.0
X_u = -0.7225
Y_v = -0.8612
Y_r = 0.1079
N_v = 0.1052
N_r = -0.5
X_uu = -1.3274
Y_vv = -36.2823
Y_rr = -0.02
Y_rv = -0.01
Y_vr = -0.01
N_vv = 5.0437
N_rr = 0.005
N_rv = -0.001
N_vr = -0.001
Y_uvb = -0.5*1000*np.pi*1.24*(0.15/2)**2
Y_uvf = -1000*3*0.0064
Y_urf = -0.4*Y_uvf
N_uvb = (-0.65*1.08 + 0.4)*Y_uvb
N_uvf = -0.4*Y_uvf
N_urf = -0.4*Y_urf
Y_uudr = 19.2
N_uudr = -0.4*Y_uudr
X_uuu = -5.8664
m_11 = m - X_udot
m_22 = m-Y_v
m_23 = m*x_g - Y_rdot
m_32 = m*x_g - N_vdot
m_33 = I_z - N_rdot

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


def export_ship_PSF_model(model_type = 'simplified', 
                          max_detected_rays = 10, 
                          n_obstacles = 1, 
                          n_moving_obstacles = 0, 
                          safety_filter_mode=None,
                          use_disturbance_estimator = False) -> AcadosModel:
    """Export AcadosModel for use in predictive safety filter"""

    model_name = 'ship_PSF'
    
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
    if safety_filter_mode == "lidar_and_moving_obstacles":
        obs_x = SX.sym('obs_x',max_detected_rays)
        obs_y = SX.sym('obs_y',max_detected_rays)
        obs_r = SX.sym('obs_r',max_detected_rays)

        obs_list = []
        for i in range(n_moving_obstacles):
            obs_list.append(SX.sym('x_obs_' + str(i)))
            obs_list.append(SX.sym('y_obs_' + str(i)))
            obs_list.append(SX.sym('r_obs_' + str(i)))
        
        state_obs = vertcat(obs_x,obs_y,obs_r,*obs_list)

    elif safety_filter_mode == "lidar":
        obs_x = SX.sym('obs_x',max_detected_rays)
        obs_y = SX.sym('obs_y',max_detected_rays)
        obs_r = SX.sym('obs_r',max_detected_rays)
        
        state_obs = vertcat(obs_x,obs_y,obs_r)

    else: 
        obs_list = []
        for i in range(n_obstacles):
            obs_list.append(SX.sym('x_obs_' + str(i)))
            obs_list.append(SX.sym('y_obs_' + str(i)))
            obs_list.append(SX.sym('r_obs_' + str(i)))
        state_obs = vertcat(*obs_list)
    
    #Dynamics
    # nu_expl = M_inv@(-C@nu - D@nu + B@F)
    def princip(angle):
        return (fmod((angle + np.pi),(2*np.pi))) - np.pi

    cos_phi = cos(phi)
    sin_phi = sin(phi)

    eta_expl = vertcat(
        cos_phi*u -sin_phi*v,
        sin_phi*u + cos_phi*v,
        r
    )
    
    eta_impl = eta_dot - eta_expl

    if use_disturbance_estimator:
        # If disturbance estimator is used

        F_u_dist = SX.sym('F_u_dist') # Surge force disturbance estimate
        F_v_dist = SX.sym('F_v_dist') # Sway force disturbance estimate
        T_r_dist = SX.sym('T_r_dist') # Yaw moment disturbance estimate

        if model_type == 'realistic':
            # Use realistic model

            nu_impl = vertcat(
                (m-X_udot)*u_dot + (-m_11*v - m_23*r)*r + (-X_u - X_uu*fabs(u) - X_uuu*u**2)*u - F_u - F_u_dist,
                (m - Y_vdot)*v_dot + (m*x_g - Y_rdot)*r_dot + m_11*u*r + (-Y_v - Y_vv*fabs(v) - Y_rv*fabs(r))*v + (-Y_r - Y_vr*fabs(v) - Y_rr*fabs(r))*r - F_v_dist,
                (m*x_g - N_vdot)*v_dot + (I_z - N_rdot)*r_dot - (-m_11*v - m_23*r)*u - m_11*u*v + (-N_v - N_vv*fabs(v) - N_rv*fabs(r))*v + (-N_r - N_vr*fabs(v) - N_rr*fabs(r))*r - F_r - T_r_dist
            )
        
            nu_expl = vertcat(
                M_inv[0,0]*(-2.0*u - (-33.8*v + 11.748*r)*r + F_u),
                M_inv[1,1]*(-7.0*v + 2.5425*r - 25.8*u*r -1.7244*F_r) + M_inv[1,2]*(2.5425*v - 1.422*r - (33.8*v - 11.748*r)*u +25.8*u*v + F_r),
                M_inv[2,1]*(-7.0*v + 2.5425*r - 25.8*u*r -1.7244*F_r) + M_inv[2,2]*(2.5425*v - 1.422*r - (33.8*v - 11.748*r)*u +25.8*u*v + F_r)
            )

        elif model_type == 'simplified':
            # Use simplified model

            nu_impl = vertcat(
                (m-X_udot)*u_dot + (-33.8*v + 11.748*r)*r + 2.0*u - F_u - F_u_dist,
                (m - Y_vdot)*v_dot + (m*x_g - Y_rdot)*r_dot + 7.0*v + (m*u + 0.1)*r - F_v_dist,
                (m*x_g - N_vdot)*v_dot + (I_z - N_rdot)*r_dot + 0.1*v + (m*x_g*u + 0.5)*r - F_r - T_r_dist
            )
            nu_expl = vertcat(
                M_inv[0,0]*(X_u*u + F_u),
                M_inv[1,1]*(Y_v*v - (m*u - Y_r)*r) + M_inv[1,2]*(N_v*v - (m*x_g*u-N_r)*r + F_r),
                M_inv[2,1]*(Y_v*v - (m*u - Y_r)*r) + M_inv[2,2]*(N_v*v - (m*x_g*u-N_r)*r + F_r)
            )
    
    else:
        # No disturbance estimator used

        if model_type == 'realistic':
            # Use realistic model

            nu_impl = vertcat(
                (m-X_udot)*u_dot + (-m_11*v - m_23*r)*r + (-X_u - X_uu*fabs(u) - X_uuu*u**2)*u - F_u,
                (m - Y_vdot)*v_dot + (m*x_g - Y_rdot)*r_dot + m_11*u*r + (-Y_v - Y_vv*fabs(v) - Y_rv*fabs(r))*v + (-Y_r - Y_vr*fabs(v) - Y_rr*fabs(r))*r,
                (m*x_g - N_vdot)*v_dot + (I_z - N_rdot)*r_dot - (-m_11*v - m_23*r)*u - m_11*u*v + (-N_v - N_vv*fabs(v) - N_rv*fabs(r))*v + (-N_r - N_vr*fabs(v) - N_rr*fabs(r))*r - F_r
            )
        
            nu_expl = vertcat(
                M_inv[0,0]*(-2.0*u - (-33.8*v + 11.748*r)*r + F_u),
                M_inv[1,1]*(-7.0*v + 2.5425*r - 25.8*u*r -1.7244*F_r) + M_inv[1,2]*(2.5425*v - 1.422*r - (33.8*v - 11.748*r)*u +25.8*u*v + F_r),
                M_inv[2,1]*(-7.0*v + 2.5425*r - 25.8*u*r -1.7244*F_r) + M_inv[2,2]*(2.5425*v - 1.422*r - (33.8*v - 11.748*r)*u +25.8*u*v + F_r)
            )

        elif model_type == 'simplified':
            # Use simplified model

            nu_impl = vertcat(
                (m-X_udot)*u_dot + (-33.8*v + 11.748*r)*r + 2.0*u - F_u,
                (m - Y_vdot)*v_dot + (m*x_g - Y_rdot)*r_dot + 7.0*v + (m*u + 0.1)*r,
                (m*x_g - N_vdot)*v_dot + (I_z - N_rdot)*r_dot + 0.1*v + (m*x_g*u + 0.5)*r - F_r
            )
            nu_expl = vertcat(
                M_inv[0,0]*(X_u*u + F_u),
                M_inv[1,1]*(Y_v*v - (m*u - Y_r)*r) + M_inv[1,2]*(N_v*v - (m*x_g*u-N_r)*r + F_r),
                M_inv[2,1]*(Y_v*v - (m*u - Y_r)*r) + M_inv[2,2]*(N_v*v - (m*x_g*u-N_r)*r + F_r)
            )
    

    eta_expl[2] = princip(eta_expl[2])
    f_impl = vertcat(eta_impl,nu_impl)
    f_expl = vertcat(eta_expl,nu_expl)

    obstacle_constraint_list = []


    if safety_filter_mode == "lidar_and_moving_obstacles":
        for i in range(max_detected_rays):
            obstacle_constraint_list.append(sqrt((x - obs_x[i])**2 + (y - obs_y[i])**2) - obs_r[i])
        for i in range(n_moving_obstacles):
            obstacle_constraint_list.append(sqrt((x - obs_list[3*i])**2 + (y - obs_list[3*i+1])**2) - obs_list[3*i+2] - 7.0)

    elif safety_filter_mode == "lidar":
        for i in range(max_detected_rays):
            obstacle_constraint_list.append(sqrt((x - obs_x[i])**2 + (y - obs_y[i])**2) - obs_r[i])

    else:
        for i in range(n_obstacles):
            obstacle_constraint_list.append(sqrt((x - obs_list[3*i])**2 + (y - obs_list[3*i+1])**2) - obs_list[3*i+2] - 7.0)
    con_h_expr = vertcat(*obstacle_constraint_list)
    con_h_expr_e = con_h_expr
    

    P_nu_terminal = np.eye(3)
    diag_P_nu = np.array([2.5e-1,2.9e-1,2.45e1])
    np.fill_diagonal(P_nu_terminal,diag_P_nu)

    u_eq = 0.55
    nu_eq = vertcat(u-u_eq,
                    v,
                    r)

    terminal_set_expr_nu = nu_eq.T@P_nu_terminal@nu_eq

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = state
    model.xdot = state_dot
    model.u = F
    
    if use_disturbance_estimator:
        model.p = vertcat(state_obs, F_u_dist, F_v_dist, T_r_dist)
    else:
        model.p = state_obs
    model.con_h_expr = con_h_expr
    model.con_h_expr_e = vertcat(con_h_expr_e, terminal_set_expr_nu)
    model.name = model_name

    return model

