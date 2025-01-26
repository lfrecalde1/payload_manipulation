import casadi as ca
import numpy as np
from acados_template import AcadosModel
from payload_manipulation import payload_model
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from casadi import Function
from casadi import jacobian


def payload_model():
    #model
    model_name = "payload_model"    

    #payload params
    m = 0.2
    g = 9.81
    I_load = np.array([[0.0062, 0,  0], [0, 0.0062, 0], [0, 0, 0.0124]])
    M_load = m * np.eye((3))                    #mass distribution

    ##frames
    #inertial 
    e1 = ca.vertcat(1.0,0,0)
    e2 = ca.vertcat(0,1,0)
    e3 = ca.vertcat(0,0,1)

    #payload frame 
    L1 = ca.vertcat(1.0,0,0)
    L2 = ca.vertcat(0,1,0)
    L3 = ca.vertcat(0,0,1)

    #define input variables
    f1 = ca.MX.sym("f1")
    f2 = ca.MX.sym("f2")
    f3 = ca.MX.sym("f3")
    F = ca.vertcat(f1, f2, f3)              #force in Inertial frame

    m1 = ca.MX.sym("m1")
    m2 = ca.MX.sym("m2")
    m3 = ca.MX.sym("m3")
    M = ca.vertcat(m1, m2, m3)              #moments in Payload frame

    v1 = ca.MX.sym('v1')
    v2 = ca.MX.sym('v2')
    v3 = ca.MX.sym('v3')
    V = ca.vertcat(v1, v2, v3)              #null space vector 

    W = ca.vertcat(f1, f2, f3, m1, m2, m3, v1, v2, v3)
    
    #state variables - pos, linear vel, quaternions and angular velocities
    #position 
    x_p = ca.MX.sym('x_p')
    y_p = ca.MX.sym('y_p')
    z_p = ca.MX.sym('z_p')
    x_1 = ca.vertcat(x_p, y_p, z_p)
    
    #linear vel
    vx = ca.MX.sym("vx")
    vy = ca.MX.sym("vy")
    vz = ca.MX.sym("vz")   
    vel = ca.vertcat(vx,vy,vz)

    #angles quaternion 
    qw = ca.MX.sym('qw')
    qx = ca.MX.sym('qx')
    qy = ca.MX.sym('qy')
    qz = ca.MX.sym('qz')        
    quat = ca.vertcat(qw,qx, qy,qz)

    #angular velocity
    p = ca.MX.sym('p')
    q = ca.MX.sym('q',)
    r = ca.MX.sym('r')
    omega = ca.vertcat(p,q,r) 

    x = ca.vertcat(x_1, vel, quat, omega)

    ##derivatives
    #position in Inertial frame
    xp_dt = ca.MX.sym('xp_dt')
    yp_dt = ca.MX.sym('yp_dt')
    zp_dt = ca.MX.sym('zp_dt')
    x1_dt = ca.vertcat(xp_dt, yp_dt, zp_dt)
    
    #linear velocity in Inertial frame 
    vx_dt = ca.MX.sym('vx_dt')
    vy_dt = ca.MX.sym('vy_dt')
    vz_dt = ca.MX.sym('vz_dt')
    vel_dt = ca.vertcat(vx_dt, vy_dt, vz_dt)

    ##angles in Inertial frame
    qw_dt = ca.MX.sym('qw_dt')
    qx_dt = ca.MX.sym('qx_dt')
    qy_dt = ca.MX.sym('qy_dt')
    qz_dt = ca.MX.sym('qz_dt')
    quat_dt = ca.vertcat(qw_dt, qx_dt, qy_dt, qz_dt)

    #angular velolcity in Payload frame
    p_dt = ca.MX.sym('p_dt')
    q_dt = ca.MX.sym('q_dt')
    r_dt = ca.MX.sym('r_dt')
    omega_dt = ca.vertcat(p_dt,q_dt,r_dt)
    x_dot = ca.vertcat(x1_dt, vel_dt, quat_dt, omega_dt)

    ##refence input
    
    
    #system dynamics 
    #angular velocity
    K_quat = 2                                                          #this enforces the magnitude 1 constraint for the quaternion
    quaterror = 1 - (qw**2 + qx**2 + qy**2 + qz**2)                     #norm_2(quat) 
    a_matrix = ca.vertcat(  ca.horzcat(0,- p,- q,- r),
               ca.horzcat(p,0,r,-q),
               ca.horzcat(q,-r,0,p),
               ca.horzcat(r,q,-p,0))
    quat_dt = 1/2 *ca.mtimes(a_matrix, quat) + K_quat * ca.mtimes(quaterror, quat)

    #calculate rotation matrix
    #Rwb = CaQuatToRot(quat)
    #Inerttia and other forces     
    cc_forces = ca.cross(omega, ca.mtimes(I_load, omega))               #colaris and centripetel forces 
    
    f_expl = ca.vertcat(vel,
                        ca.mtimes(np.linalg.inv(M_load) ,(F - m * g * e3)),
                        quat_dt,
                        ca.mtimes(np.linalg.inv(I_load), (M - cc_forces))
                        )

    model = AcadosModel()                
    model.name = model_name
    f_impl = x_dot - f_expl 
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_dot
    model.u = W 
    #model.y = cc_forces
    nx = model.x.rows()
    nu = model.u.rows()
    reference_param = ca.MX.sym('references', (nx + nu), 1)
    model.p = reference_param    

    # Distance to obstacles
    # Constraints
    P_inv = np.array([
    [ 3.33333333e-01, -9.36648663e-05, -6.84744577e-18, -3.73019688e-17,  3.06188141e-18,  9.36648663e-01],
    [-3.10600309e-18,  3.33387392e-01, -2.12869954e-17, -1.08759727e-16,  2.94103127e-16, -5.40590108e-01],
    [-1.21674238e-02, -2.10674157e-02,  3.33441488e-01, -1.87265918e+00,  1.08154878e+00,  9.89274683e-18],
    [ 3.33333333e-01,  6.18935821e-18,  6.54044202e-19,  4.81656346e-17,  3.63634243e-18, -2.07941867e-16],
    [ 7.04096181e-23,  3.33225215e-01, -2.97245187e-18, -8.28336471e-18, -1.43087148e-20,  1.08118022e+00],
    [ 2.43348475e-02,  3.13213049e-17,  3.33117024e-01,  1.64061492e-15, -2.16309756e+00, -4.93551011e-17],
    [ 3.33333333e-01,  9.36648663e-05,  8.15547386e-18, -1.09608797e-17,  3.63634243e-18, -9.36648663e-01],
    [-2.37265528e-22,  3.33387392e-01,  1.00165344e-17,  2.18796632e-17,  7.63713816e-21, -5.40590108e-01],
    [-1.21674238e-02,  2.10674157e-02,  3.33441488e-01,  1.87265918e+00,  1.08154878e+00, -4.51720853e-18]])
    Null_space = np.array([
    [-2.87918784e-01,  2.14897788e-01, -5.36193238e-01],
    [-4.69514968e-01, -5.53097654e-01, -2.38747589e-01],
    [ 5.55111512e-17,  2.77555756e-17, -1.52655666e-16],
    [-3.01587147e-01, -7.01082736e-02,  7.55510880e-01],
    [ 5.06754053e-01, -2.07736340e-01,  1.83010565e-01],
    [ 0.00000000e+00,  0.00000000e+00, -2.77555756e-17],
    [ 5.89505930e-01, -1.44789515e-01, -2.19317642e-01],
    [-3.72390849e-02,  7.60833993e-01,  5.57370245e-02],
    [ 2.77555756e-17, -2.77555756e-17,  1.11022302e-16]])

    F = W[0:3]
    M = W[3:6]
    V = W[6:9]


    #Null space exploitation
    N_mat = Null_space
    R_mat = quatTorot_c(quat)    
    Fl = ca.mtimes(R_mat.T , F)

    
    #calulate wrench in payload frame
    rho = np.array([
    [-0.1540, -0.2670,  0.01125],
    [ 0.3083,  0.0000,  0.01125],
    [-0.1540,  0.2670,  0.01125]])
    Wl = ca.vertcat(Fl, M)
    mu_list = ca.mtimes(P_inv, Wl) - ca.mtimes(N_mat, V)
    print(mu_list.shape)
    rho_list = rho

    cabel_length = 0.5
    quad_pose_payload = cal_quad_pose(mu_list, rho_list, cabel_length)
    quad_pose_inertial = x_1 + R_mat@quad_pose_payload
    quad_pose_function =  Function('quadrotors', [x, W], [quad_pose_inertial])
    return model, quad_pose_function

def cal_quad_pose(mu_list, rho_list, cabel_length):
    mu_list = ca.reshape(mu_list, 3,3)
    number_robot = rho_list.shape[0]
    quad_position = ca.MX.zeros(3, number_robot)
    for i in range(0, number_robot):
        None
        mu = mu_list[:, i]
        rho = rho_list[i, :]
        lk = cabel_length
        zeta = 1 * mu / (ca.norm_2(mu)+ ca.np.finfo(np.float64).eps)
        quad_position[:, i] = rho  + lk * zeta
    return quad_position



def controller_setup(x0, N_horizon, t_horizon, ts):
    #read yaml files
    # pay_load_param_path = sys.argv[1]
    # nmpc_control_path = sys.argv[2]    
    #MPC Key parameters
    N = N_horizon
    Tf = t_horizon 

    #define optimization problem
    ocp = AcadosOcp()
    model, quad_position_f = payload_model()
     # Constructing the optimal control problem
    ocp.model = model

    # Dimension of the problem
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    # Set the dimension of the problem
    ocp.p = model.p
    ocp.dims.N = N_horizon

    Q_mat =  ca.vertcat(10,10,10, 10, 10, 10, 5, 5, 5, 5,  1e-4, 1e-4, 1)
    R_mat =  ca.vertcat(10, 10, 10, 10, 10, 10, 1e-2, 1e-2, 1e-2)
    Q_emat =  ca.vertcat(50,50,50, 10, 10, 10, 5, 5, 5, 5,  1e-4, 1e-4, 1)
    Q_emat = 1*Q_emat
    Q_mat = 1*Q_mat
    x_array = model.x
    u_aaray = model.u
    ref_array = model.p
    #print(ref_array)

    #calculate square cost    
    ocp.cost.cost_type = 'EXTERNAL'  
    pos_err = cal_square_cost(x_array[0:3], ref_array[0:3], Q_mat[0:3])
    vel_err = cal_square_cost(x_array[3:6], ref_array[3:6], Q_mat[3:6])
    quat_error = calc_quat_cost(x_array[6:10], ref_array[6:10], Q_mat[6:9])
    input_err = cal_square_cost(u_aaray, ref_array[13:22], R_mat)
    ocp.model.cost_expr_ext_cost = pos_err + vel_err + quat_error + input_err 
    ocp.model.cost_expr_ext_cost_0 = pos_err + vel_err + quat_error + input_err 
    

    #terminal cost
    ocp.cost.cost_type_e = 'EXTERNAL'
    pos_err = cal_square_cost(x_array[0:3], ref_array[0:3], Q_emat[0:3])
    vel_err = cal_square_cost(x_array[3:6], ref_array[3:6], Q_emat[3:6])
    quat_error = calc_quat_cost(x_array[6:10], ref_array[6:10], Q_emat[6:9])        
    ocp.model.cost_expr_ext_cost_e = pos_err + vel_err + quat_error
    
    ##constraints
    #input constraints
    # mg = 0.250*9.81 # 0.25 is the wrong mass value check the params file
    aux_init = np.array([0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0,
                                     1.0, 0.0, 0.0, 0.0,         # Angular velocity body frame
                                     0.0, 0.0, 0.0,         # Linear velocity body frame
                                     0.0, 0.0, 0.2*9.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                                     ])        
    print(aux_init.shape)
    ocp.parameter_values = aux_init

    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.lbu = np.array([-5,-5,-5, -1,-1,-1 , -0,-0,-0]) #np.array([-10,-10,-10, -10,-10,-10])
    ocp.constraints.ubu = np.array([5,5,5,1,1,1,0,0,0]) #np.array([20,20,20,10,10,10])
    ocp.constraints.idxbu = np.array([0,1,2,3,4,5,6,7,8]) #np.array([0,1,2,3,4,5])
    
    #initial state contraints
    ocp.constraints.x0 = x0

    # Constraints for payload and quadrotors
    quadrotor_pos = quad_position_f(x_array, u_aaray)
    ## Nonlinear constraints
    obstacle = ca.veccat(1.35, -1.34, 5.39)
    obstacles_quadrotor_diff = obstacle - quadrotor_pos
    quadrotor_12 = quadrotor_pos[:, 0] - quadrotor_pos[:, 1] 
    quadrotor_23 = quadrotor_pos[:, 1] - quadrotor_pos[:, 2] 
    quadrotor_31 = quadrotor_pos[:, 2] - quadrotor_pos[:, 0] 
    diff = obstacle - x_array[0:3]
    radius = 0.8
    radius_small = 0.1
    distance_payload = diff.T@diff - radius**2
    distance_quad_1 = obstacles_quadrotor_diff[:, 0].T@obstacles_quadrotor_diff[:, 0] - radius**2
    distance_quad_2 = obstacles_quadrotor_diff[:, 1].T@obstacles_quadrotor_diff[:, 1] - radius**2
    distance_quad_3 = obstacles_quadrotor_diff[:, 2].T@obstacles_quadrotor_diff[:, 2] - radius**2
    diatance_quad_12 = quadrotor_12.T@quadrotor_12 - radius_small**2
    diatance_quad_23 = quadrotor_23.T@quadrotor_23 - radius_small**2
    diatance_quad_31 = quadrotor_31.T@quadrotor_31 - radius_small**2

    constraints = ca.vertcat(distance_payload, distance_quad_1, distance_quad_2, distance_quad_3, diatance_quad_12, diatance_quad_23, diatance_quad_31)

    ocp.model.con_h_expr = constraints
    nsbx = 0
    nh = constraints.shape[0]
    nsh = nh
    ns = nsh + nsbx
#
    cost_values_steps = np.zeros((ns, ))
    cost_values_final = np.zeros((ns, ))


    cost_values_steps[0] = 1500
    cost_values_steps[1] = 1500
    cost_values_steps[2] = 1500
    cost_values_steps[3] = 1500
    cost_values_steps[4] = 1500
    cost_values_steps[5] = 1500
    cost_values_steps[6] = 1500

    cost_values_final[0] = 1500
    cost_values_final[1] = 1500
    cost_values_final[2] = 1500
    cost_values_final[3] = 1500
    cost_values_final[4] = 1500
    cost_values_final[5] = 1500
    cost_values_final[6] = 1500

    ocp.cost.zl = cost_values_steps
    ocp.cost.Zl = cost_values_final

    ocp.cost.zu = cost_values_steps
    ocp.cost.Zu = cost_values_final

    ##### Lower and upper limits
    ocp.constraints.lh = np.array([0, 0, 0, 0, 0, 0, 0])
    ocp.constraints.uh = np.array([30, 30, 30, 30, 0.5, 0.5, 0.5])
    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    ocp.constraints.idxsh = np.array(range(nsh))

    # Set options
    # Solver Options
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'  # Efficient QP solver
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # Fast real-time SQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # Gauss-Newton approximation
    ocp.solver_options.integrator_type = 'IRK'  # Implicit Runge-Kutta (IRK)

    ## Regularization (stabilizes optimization)
    ocp.solver_options.regularize_method = 'NO_REGULARIZE'  

    ## Levenberg-Marquardt regularization (optional)
    ocp.solver_options.levenberg_marquardt = 10.0

    ## NLP Solver Settings
    ocp.solver_options.nlp_solver_max_iter = 200  # Maximum iterations
    ocp.solver_options.nlp_solver_tol_stat = 1e-2  # Tolerance for stationarity
    ocp.solver_options.print_level = 0  # Suppress print output

    ocp.solver_options.tf = Tf

    ## QP Solver Options
    ##ocp.solver_options.qp_solver_warm_start = 1  # Enable QP hot-start
    ocp.solver_options.qp_solver_cond_N = N  # Number of QP stages for partial condensing
    ocp.solver_options.qp_solver_ric_alg = 1  # Use Riccati-based algorithm

    ## Compilation flags for external functions (optional for performance)
    ocp.solver_options.ext_fun_compile_flags = '-Ofast -march=native'
    ocp.solver_options.hpipm_mode = 'SPEED'  # Prioritize speed in QP solver

    # Parallelization
    ocp.solver_options.cg_use_openmp = True  # Enable OpenMP parallelization
    ocp.solver_options.cg_hardcode_constraints = False  # Allow runtime constraint changes
    ocp.solver_options.cg_use_variable_weighting_matrix = True  # Support time-varying costs

    # IRK Settings (Implicit Runge-Kutta)
    #ocp.solver_options.sim_method_num_stages = 4  # IRK-GL4: 4 stages for accuracy
    #ocp.solver_options.sim_method_num_steps = 2  # Number of integration steps
    #ocp.solver_options.sim_method_newton_iter = 3  # Newton iterations for convergence

    # Use Single Precision (optional)
    ocp.solver_options.use_single_precision = True  
    
    
    return ocp, quad_position_f


def cal_square_cost(ref_vec, state_vec, weights):
    #all inpu arrays are np array (n)
    # print(weights)
    # print((ref_vec - state_vec)**2)
    print((ref_vec - state_vec)**2)
    print(weights)
    cost = ca.dot((ref_vec - state_vec)**2, weights)
    #ipdb.set_trace()    
    return cost


def calc_quat_cost(q2, q1, weights ):
    #print(q2.size())
    #calculate quaternion difference 
    q_aux = np.array([
        q2[0] * q1[0] + q2[1] * q1[1] + q2[2] * q1[2] + q2[3] * q1[3],  # w
        q2[0] * q1[1] - q2[1] * q1[0] - q2[2] * q1[3] + q2[3] * q1[2],  # x
        q2[0] * q1[2] + q2[1] * q1[3] - q2[2] * q1[0] - q2[3] * q1[1],  # y
        q2[0] * q1[3] - q2[1] * q1[2] + q2[2] * q1[1] - q2[3] * q1[0],  # z
    ])
    #yaw rotation extraction
    q_att_denom = ca.sqrt(q_aux[0] * q_aux[0] + q_aux[3] * q_aux[3] + 1e-3)
    q_att = ca.vertcat(
    q_aux[0] * q_aux[1] - q_aux[2] * q_aux[3],
    q_aux[0] * q_aux[2] + q_aux[1] * q_aux[3],
    q_aux[3]) / q_att_denom
    result = ca.transpose(q_att) @ ca.diag(weights) @ q_att
    #ipdb.set_trace()
    return result
def quatTorot_c(quat):
    # Function to transform a quaternion to a rotational matrix
    # INPUT
    # quat                                                       - unit quaternion
    # OUTPUT                                     
    # R                                                          - rotational matrix

    # Normalized quaternion
    q = quat
    q = q/(q.T@q)

    # Create empty variable
    #q_hat = ca.MX.zeros(3, 3)
    #q_hat[0, 1] = -q[3]
    #q_hat[0, 2] = q[2]
    #q_hat[1, 2] = -q[1]
    #q_hat[1, 0] = q[3]
    #q_hat[2, 0] = -q[2]
    #q_hat[2, 1] = q[1]

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    Q = ca.vertcat(
        ca.horzcat(q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)),
        ca.horzcat(2*(q1*q2+q0*q3), q0**2+q2**2-q1**2-q3**2, 2*(q2*q3-q0*q1)),
        ca.horzcat(2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0**2+q3**2-q1**2-q2**2))

    # Compute Rotational Matrix
    #R = ca.MX.eye(3) + 2 * (q_hat@q_hat) + 2 * q[0] * q_hat
    R = Q
    return R 