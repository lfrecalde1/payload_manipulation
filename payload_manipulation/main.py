#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

import threading


from acados_template import AcadosOcpSolver, AcadosSimSolver


from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import time
import os
from payload_manipulation import payload_model, trajectory, controller_setup

class DQnmpcNode(Node):
    def __init__(self):
        super().__init__('PAYLOAD_MANIPULATION')
        # Lets define internal variables
        self.g = 9.80665
        self.mQ = (1.272)

        # Inertia Matrix
        self.Jxx = 0.00304475
        self.Jyy = 0.00454981
        self.Jzz = 0.00281995
        self.J = np.array([[self.Jxx, 0.0, 0.0], [0.0, self.Jyy, 0.0], [0.0, 0.0, self.Jzz]])
        self.L = [self.mQ, self.Jxx, self.Jyy, self.Jzz, self.g]

        # Desired sample time  self.ts, time where we want to init over trajectory t_initial, time for the trajectory t_trajectory
        # Time to go to the init state t_final
        # Initial time to established a stable connection self.initial
        self.ts = 0.05
        self.final = 30
        self.t =np.arange(0, self.final + self.ts, self.ts, dtype=np.double)
        self.x_d, self.x_d_dot, self.x_d_dot_dot, self.x_d_dot_dot_dot, self.x_d_dot_dot_dot_dot, _, _, _ = trajectory(self.t, 2, 1)
        # Initial States dual set zeros
        # Position of the system
        pos_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Linear velocity of the sytem respect to the inertial frame
        vel_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Angular velocity respect to the Body frame
        omega_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Initial Orientation expressed as quaternionn
        quat_0 = np.array([1.0, 0.0, 0.0, 0.0])

        # Auxiliary vector [x, v, q, w], which is used to update the odometry and the states of the system
        self.x_0 = np.hstack((pos_0, vel_0, quat_0, omega_0))


        # Define odometry publisher for the drone
        self.odom_msg = Odometry()
        self.publisher_odom_ = self.create_publisher(Odometry, "odom", 10)

        self.odom_quad_1_msg = Odometry()
        self.publisher_odom_quad_1_ = self.create_publisher(Odometry, "odom_quad_1", 10)

        self.odom_quad_2_msg = Odometry()
        self.publisher_odom_quad_2_ = self.create_publisher(Odometry, "odom_quad_2", 10)

        self.odom_quad_3_msg = Odometry()
        self.publisher_odom_quad_3_ = self.create_publisher(Odometry, "odom_quad_3", 10)

        # Define odometry publisher for the desired path
        self.ref_msg = Odometry()
        self.publisher_ref_ = self.create_publisher(Odometry, "desired_frame", 10)

        # Definition of the publihser for the desired parth
        self.marker_msg = Marker()
        self.points = None
        self.publisher_ref_trajectory_ = self.create_publisher(Marker, 'desired_path', 10)

        self.marker_msg_real = Marker()
        self.points_real = None
        self.publisher_real_trajectory_ = self.create_publisher(Marker, 'real_path', 10)

        # Definition of the prediction time in secs
        self.t_N = 0.5

        # Definition of the horizon
        self.N = np.arange(0, self.t_N + self.ts, self.ts)
        self.N_prediction = self.N.shape[0]
        print(self.N_prediction)


        # New Vector to save the states of the drone as a baseline formulation
        self.x = np.zeros((13, self.t.shape[0] + 1 - self.N_prediction), dtype=np.double)
        self.x[:, 0] = self.x_0

        # Create a thread to run the simulation and viewer
        self.simulation_thread = threading.Thread(target=self.run)
        # Start thread for the simulation
        self.simulation_thread.start()

    def send_odometry(self, h, q):
        # Function that send odometry

        self.odom_msg.header.frame_id = "world"
        self.odom_msg.header.stamp = self.get_clock().now().to_msg()

        self.odom_msg.pose.pose.position.x = h[0]
        self.odom_msg.pose.pose.position.y = h[1]
        self.odom_msg.pose.pose.position.z = h[2]

        self.odom_msg.pose.pose.orientation.x = q[1]
        self.odom_msg.pose.pose.orientation.y = q[2]
        self.odom_msg.pose.pose.orientation.z = q[3]
        self.odom_msg.pose.pose.orientation.w = q[0]

        # Send Messag
        self.publisher_odom_.publish(self.odom_msg)
        return None 

    def send_odometry_quad(self, h):
        # Function that send odometry



        self.odom_quad_1_msg.header.frame_id = "world"
        self.odom_quad_2_msg.header.frame_id = "world"
        self.odom_quad_3_msg.header.frame_id = "world"
        self.odom_quad_1_msg.header.stamp = self.get_clock().now().to_msg()
        self.odom_quad_2_msg.header.stamp = self.get_clock().now().to_msg()
        self.odom_quad_2_msg.header.stamp = self.get_clock().now().to_msg()

        self.odom_quad_1_msg.pose.pose.position.x = float(h[0, 0])
        self.odom_quad_1_msg.pose.pose.position.y = float(h[1, 0])
        self.odom_quad_1_msg.pose.pose.position.z = float(h[2, 0])

        self.odom_quad_2_msg.pose.pose.position.x = float(h[0, 1])
        self.odom_quad_2_msg.pose.pose.position.y = float(h[1, 1])
        self.odom_quad_2_msg.pose.pose.position.z = float(h[2, 1])

        self.odom_quad_3_msg.pose.pose.position.x = float(h[0, 2])
        self.odom_quad_3_msg.pose.pose.position.y = float(h[1, 2])
        self.odom_quad_3_msg.pose.pose.position.z = float(h[2, 2])

        # Send Messag
        self.publisher_odom_quad_1_.publish(self.odom_quad_1_msg)
        self.publisher_odom_quad_2_.publish(self.odom_quad_2_msg)
        self.publisher_odom_quad_3_.publish(self.odom_quad_3_msg)
        return None 
  
    def send_ref(self, h, q):
        self.ref_msg.header.frame_id = "world"
        self.ref_msg.header.stamp = self.get_clock().now().to_msg()

        self.ref_msg.pose.pose.position.x = h[0]
        self.ref_msg.pose.pose.position.y = h[1]
        self.ref_msg.pose.pose.position.z = h[2]

        self.ref_msg.pose.pose.orientation.x = q[1]
        self.ref_msg.pose.pose.orientation.y = q[2]
        self.ref_msg.pose.pose.orientation.z = q[3]
        self.ref_msg.pose.pose.orientation.w = q[0]

        # Send Message
        self.publisher_ref_.publish(self.ref_msg)
        return None 

    def init_marker(self, x):
        self.marker_msg.header.frame_id = "world"
        self.marker_msg.header.stamp = self.get_clock().now().to_msg()
        self.marker_msg.ns = "trajectory"
        self.marker_msg.id = 0
        self.marker_msg.type = Marker.LINE_STRIP
        self.marker_msg.action = Marker.ADD
        self.marker_msg.pose.orientation.w = 1.0
        self.marker_msg.scale.x = 0.01  # Line width
        self.marker_msg.color.a = 1.0  # Alpha
        self.marker_msg.color.r = 0.0  # Red
        self.marker_msg.color.g = 1.0  # Green
        self.marker_msg.color.b = 0.0  # Blue
        point = Point()
        point.x = x[0]
        point.y = x[1]
        point.z = x[2]
        self.points = [point]
        self.marker_msg.points = self.points
        return None
        
    def init_real_marker(self, x):
        self.marker_msg_real.header.frame_id = "world"
        self.marker_msg_real.header.stamp = self.get_clock().now().to_msg()
        self.marker_msg_real.ns = "trajectory"
        self.marker_msg_real.id = 0
        self.marker_msg_real.type = Marker.LINE_STRIP
        self.marker_msg_real.action = Marker.ADD
        self.marker_msg_real.pose.orientation.w = 1.0
        self.marker_msg_real.scale.x = 0.01  # Line width
        self.marker_msg_real.color.a = 1.0  # Alpha
        self.marker_msg_real.color.r = 0.0  # Red
        self.marker_msg_real.color.g = 0.0  # Green
        self.marker_msg_real.color.b = 1.0  # Blue
        point = Point()
        point.x = x[0]
        point.y = x[1]
        point.z = x[2]
        self.points_real = [point]
        self.marker_msg_real.points = self.points_real
        return None

    def send_marker(self, x):
        self.marker_msg.header.stamp = self.get_clock().now().to_msg()
        self.marker_msg.type = Marker.LINE_STRIP
        self.marker_msg.action = Marker.ADD
        point = Point()
        point.x = x[0]
        point.y = x[1]
        point.z = x[2]
        self.points.append(point)
        self.marker_msg.points = self.points
        self.publisher_ref_trajectory_.publish(self.marker_msg)
        return None

    def send_real_marker(self, x):
        self.marker_msg_real.header.stamp = self.get_clock().now().to_msg()
        self.marker_msg_real.type = Marker.LINE_STRIP
        self.marker_msg_real.action = Marker.ADD
        point = Point()
        point.x = x[0]
        point.y = x[1]
        point.z = x[2]
        self.points_real.append(point)
        self.marker_msg_real.points = self.points_real
        self.publisher_real_trajectory_.publish(self.marker_msg_real)
        return None

    def run(self):

        # Generalized control actions
        u_d = np.zeros((9, self.t.shape[0]), dtype=np.double)
        u_d[2, :] = 0.2*9.81

        u = np.zeros((9, self.t.shape[0]-self.N_prediction), dtype=np.double)
        u[2, :] = 0.2*9.81

        ocp, quad_positions_f = controller_setup(self.x[:, 0], self.N_prediction, self.t_N, self.ts)

        # No Cython
        json_name = "acados_ocp_" + ocp.model.name + ".json"
        json_name_sim = "acados_sim_" + ocp.model.name + ".json"
        
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file=json_name, build= False, generate= False)
        acados_integrator = AcadosSimSolver(ocp, json_file=json_name_sim, build= False, generate= False)

        # Reset Solver
        acados_ocp_solver.reset()

        # Initial States Acados
        for stage in range(self.N_prediction + 1):
            acados_ocp_solver.set(stage, "x", self.x[:, 0])
        for stage in range(self.N_prediction):
            acados_ocp_solver.set(stage, "u", u_d[:, 0])


        # Init Markers
        self.init_marker(self.x_d[:, 0])
        self.init_real_marker(self.x[0:3, 0])

        # Simulation loop
        for k in range(0, self.t.shape[0] - self.N_prediction):
            # Get model
            tic = time.time()
             # Send Desired States
            self.send_marker(self.x_d[:, k])
            self.send_real_marker(self.x[0:3, k])

            self.send_ref(self.x_d[:, k], np.array([1.0, 0.0, 0.0, 0.0]))
            self.send_odometry(self.x[0:3, k], self.x[6:10, k])
            
            ## Optimal control setting parameters
            acados_ocp_solver.set(0, "lbx", self.x[:, k])
            acados_ocp_solver.set(0, "ubx", self.x[:, k])

            # Desired Trajectory of the system
            for j in range(self.N_prediction):
                xref = self.x_d[:,k+j]
                xref_dot = self.x_d_dot[:,k+j]
                qref = np.array([1.0, 0.0, 0.0, 0.0])
                wref = np.array([0.0, 0.0, 0.0])
                uref = u_d[:,k+j]
                aux_ref = np.hstack((xref, xref_dot, qref, wref, uref))
                acados_ocp_solver.set(j, "p", aux_ref)

            # Desired Trayectory at the last Horizon
            xref = self.x_d[:,k+self.N_prediction]
            xref_dot = self.x_d_dot[:,k+self.N_prediction]
            qref = np.array([1.0, 0.0, 0.0, 0.0])
            wref = np.array([0.0, 0.0, 0.0])
            uref_N = u_d[:,k+self.N_prediction]

            aux_ref_N = np.hstack((xref, xref_dot, qref, wref, uref_N))
            acados_ocp_solver.set(self.N_prediction, "p", aux_ref_N)

            # Check Solution since there can be possible errors 
            #acados_ocp_solver.options_set("rti_phase", 2)
            acados_ocp_solver.solve()
            u[:, k] = acados_ocp_solver.get(0, "u")
            quadrotor_position = quad_positions_f(self.x[:, k], u[:, k])
            self.send_odometry_quad(quadrotor_position)
            # System evolution
            # Update Data of the system
            acados_integrator.set("x", self.x[:, k])
            acados_integrator.set("u", u[:, k])

            status_integral = acados_integrator.solve()
            xcurrent = acados_integrator.get("x")

            # Update Data of the system
            self.x[:, k+1] = xcurrent

            # Section to guarantee same sample times
            while (time.time() - tic <= self.ts):
                pass
            toc = time.time() - tic

            self.get_logger().info("PAYLOAD CONTROL")
        None
def main(args=None):
    rclpy.init(args=args)
    planning_node = DQnmpcNode()
    try:
        rclpy.spin(planning_node)  # Will run until manually interrupted
    except KeyboardInterrupt:
        planning_node.get_logger().info('Simulation stopped manually.')
        planning_node.destroy_node()
        rclpy.shutdown()
    finally:
        planning_node.destroy_node()
        rclpy.shutdown()
    return None

if __name__ == '__main__':
    main()