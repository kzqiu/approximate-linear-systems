import numpy as np
import random
from GRiD import RBDReference
from GRiD import URDFParser

class TrajoptPlant:
	def __init__(self, integrator_type: int = 0, options = {}, need_path: bool = False):
		self.validate_integrator_type(integrator_type)
		self.integrator_type = integrator_type
		self.set_default_options(options, need_path)
		self.options = options

	def validate_integrator_type(self, integrator_type: int):
		if not (integrator_type in [0, 1, 2, 3, 4, -1]):
			print("Invalid integrator options are [0 : euler, 1 : semi-implicit euler, 2 : midpoint, 3 : rk3, 4 : rk4, -1 : hard-coded as dynamics")
			exit()

	def set_default_options(self, options: dict, need_path: bool = False):
		options.setdefault('path_to_urdf', None)
		options.setdefault('gravity', -9.81)
		if need_path and (not options.get('path_to_urdf')):
			print("You must include the 'path_to_urdf' in the options.")
			exit()

	##############################
	# Child class must implement #
	##############################

	def forward_dynamics(self):
		raise NotImplementedError

	def forward_dynamics_gradient(self):
		raise NotImplementedError

	def get_num_pos(self):
		raise NotImplementedError

	def get_num_vel(self):
		raise NotImplementedError

	def get_num_cntrl(self):
		raise NotImplementedError

	##############################
	# Child class must implement #
	##############################

	#  [ v ;
	#   qdd ]
	def qdd_to_xdot(self, xk: np.ndarray, qdd: np.ndarray):
		nq = self.get_num_pos()
		nv = self.get_num_vel()
		nu = self.get_num_cntrl()
		return np.vstack((xk[nq:], qdd)).flatten()

	# [ 0       ; eye     ; 0
	#   dqdd/dq ; dqdd/dv ; dqdd/du ]
	def dqdd_to_dxdot(self, dqdd: np.ndarray):
		nq = self.get_num_pos()
		nv = self.get_num_vel()
		m = self.get_num_cntrl()
		top = np.hstack((np.zeros((nq,nq)), np.eye(nv), np.zeros((nq,m))))
		return np.vstack((top, dqdd))

	def integrator(self, xk: np.ndarray, uk: np.ndarray, dt: float, return_gradient: bool = False):
		n = len(xk)

		if self.integrator_type == -1: # hard coded into model
			if not return_gradient:
				return self.integrator(xk,uk)
			else:
				return self.integrator_gradient(xk,uk)

		if self.integrator_type == 0: # euler
			#  xkp1 = xk + dt * [vk,qddk]
			# dxkp1 = [Ix | 0u ] + dt*[ 0q, Iv, 0u; dqdd]
			qdd = self.forward_dynamics(xk,uk)
			xdot = self.qdd_to_xdot(xk, qdd)
			xkp1 = xk + dt*xdot
			if not return_gradient:
				return xkp1 #np.reshape(xkp1, (xkp1.shape[0],1))[:,0]
			else:
				dqdd = self.forward_dynamics_gradient(xk,uk)
				dxdot = self.dqdd_to_dxdot(dqdd)
				A = np.eye(n) + dt*dxdot[:,0:n]
				B = dt*dxdot[:,n:]
				return A, B
		
		elif self.integrator_type == 1: # semi-implicit euler
			#  vkp1 = vk + dt*qddk
			#  qkp1 = qk  + dt*vkp1
			#  xkp1 = [qkp1; vkp1]
			# dxkp1 = [Ix | 0u ] + dt*[[0q, Iv, 0u] + dt*dqdd; dqdd]
			nq = self.get_num_pos()
			nv = self.get_num_vel()
			nu = self.get_num_cntrl()
			qdd = self.forward_dynamics(xk,uk)
			vkp1 = xk[nq:]  + dt*qdd
			qkp1 = xk[0:nq] + dt*vkp1
			if not return_gradient:
				return np.hstack((qkp1,vkp1)).transpose()
			else:
				dqdd = self.forward_dynamics_gradient(xk,uk)
				zIz = np.hstack((np.zeros((nq,nq)),np.eye(nq),np.zeros((nq,nu))))
				Iz = np.hstack((np.eye(nq+nv),np.zeros((nq+nv,nu))))
				AB = Iz + dt*np.vstack((zIz + dt*dqdd, dqdd))
				return AB[:,0:nq+nv], AB[:,nq+nv:]
		
		elif self.integrator_type == 2: # midpoint
			xdot1 = self.qdd_to_xdot(xk, self.forward_dynamics(xk,uk))
			midpoint = xk + 0.5*dt*xdot1
			xdot2 = self.qdd_to_xdot(xk, self.forward_dynamics(midpoint,uk))
			xkp1 = xk + dt*xdot2
			if not return_gradient:
				return xkp1
			else:
				dxdot1 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(xk,uk))
				A1 = np.eye(n) + 0.5*dt*dxdot1[:,0:n]
				B1 = 0.5*dt*dxdot1[:,n:]
				
				dxdot2 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(midpoint,uk))
				A2 = np.eye(n) + 0.5*dt*dxdot2[:,0:n]
				B2 = 0.5*dt*dxdot2[:,n:]

				A = np.matmul(A2,A1)
				B = np.matmul(A2,B1) + B2
				return A, B

		elif self.integrator_type == 3: # rk3
			xdot1 = self.qdd_to_xdot(xk, self.forward_dynamics(xk,uk))
			point1 = xk + 0.5*dt*xdot1
			xdot2 = self.qdd_to_xdot(xk, self.forward_dynamics(point1,uk))
			point2 = xk + 0.75*dt*xdot2
			xdot3 = self.qdd_to_xdot(xk, self.forward_dynamics(point2,uk))
			xkp1 = xk + (dt/9)*(2*xdot1 + 3*xdot2 + 4*xdot3)
			if not return_gradient:
				return xkp1
			else:
				dxdot1 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(xk,uk))
				A1 = np.eye(n) + 2/9*dt*dxdot1[:,0:n]
				B1 = 2/9*dt*dxdot1[:,n:]

				dxdot2 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(point1,uk))
				A2 = np.eye(n) + 1/3*dt*dxdot2[:,0:n]
				B2 = 1/3*dt*dxdot1[:,n:]                
				
				dxdot3 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(point2,uk))
				A3 = np.eye(n) + 4/9*dt*dxdot3[:,0:n]
				B3 = 4/9*dt*dxdot1[:,n:]                
				
				A = np.matmul(A3,np.matmul(A2,A1))
				B = np.matmul(A3,np.matmul(A2,B1)) + np.matmul(A3,B2) + B3
				return A,B
		
		elif self.integrator_type == 4: # rk4
			xdot1 = self.qdd_to_xdot(xk, self.forward_dynamics(xk,uk))
			point1 = xk + 0.5*dt*xdot1
			xdot2 = self.qdd_to_xdot(xk, self.forward_dynamics(point1,uk))
			point2 = xk + 0.5*dt*xdot2
			xdot3 = self.qdd_to_xdot(xk, self.forward_dynamics(point2,uk))
			point3 = xk + dt*xdot3
			xdot4 = self.qdd_to_xdot(xk, self.forward_dynamics(point3,uk))
			xkp1 = xk + (dt/6)*(xdot1 + 2*xdot2 + 2*xdot3 + xdot4)
			if not return_gradient:
				return xkp1
			else:
				dxdot1 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(xk,uk))
				A1 = np.eye(n) + 1/6*dt*dxdot1[:,0:n]
				B1 = 1/6*dt*dxdot1[:,n:]

				dxdot2 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(point1,uk))
				A2 = np.eye(n) + 1/3*dt*dxdot2[:,0:n]
				B2 = 1/3*dt*dxdot1[:,n:]                
				
				dxdot3 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(point2,uk))
				A3 = np.eye(n) + 1/3*dt*dxdot3[:,0:n]
				B3 = 1/3*dt*dxdot1[:,n:]

				dxdot4 = self.dqdd_to_dxdot(self.forward_dynamics_gradient(point3,uk))
				A4 = np.eye(n) + 1/6*dt*dxdot4[:,0:n]
				B4 = 1/6*dt*dxdot1[:,n:]
				
				A = np.matmul(A4,np.matmul(A3,np.matmul(A2,A1)))
				B = np.matmul(A4,np.matmul(A3,np.matmul(A2,B1))) + np.matmul(A4,np.matmul(A3,B2)) + np.matmul(A4,B3) + B4
				
				return A,B

class DoubleIntegratorPlant(TrajoptPlant):
	def __init__(self, integrator_type: int = 0, options = {}):
		super().__init__(integrator_type, options)

	def forward_dynamics(self, x, u):
		return u

	def forward_dynamics_gradient(self, x, u):
		return np.array([0, 0, 1])

	def get_num_pos(self):
		return 1

	def get_num_vel(self):
		return 1

	def get_num_cntrl(self):
		return 1

class PendulumPlant(TrajoptPlant):
	def __init__(self, integrator_type = 0, options = {}):
		super().__init__(integrator_type, options)

	def forward_dynamics(self, x: np.ndarray, u: np.ndarray):
		# m * l^2 * theta_dd   +   b * theta_d   +   m * g * l * sin(theta) = u
		# assume 0 damping and m = l = 1
		# theta_dd = u - g * sin(theta)
		return u - 9.81 * np.sin(x[0])

	def forward_dynamics_gradient(self, x: np.ndarray, u: np.ndarray):
		return np.array([- 9.81 * np.cos(x[0]), 0, 1])

	def get_num_pos(self):
		return 1

	def get_num_vel(self):
		return 1

	def get_num_cntrl(self):
		return 1

# http://www.matthewpeterkelly.com/tutorials/cartPole/index.html
class CartPolePlant(TrajoptPlant):
	def __init__(self, integrator_type = 0, options = {}):
		super().__init__(integrator_type, options)

	def forward_dynamics(self, x: np.ndarray, u: np.ndarray):
		gravity = self.options['gravity']
		# assuming m_cart = m_pole = l_pole = 1
		q = x[0] # position of cart on track
		theta = x[1] # angle of pole
		q_d = x[2]
		theta_d = x[3]
		st = np.sin(theta)
		ct = np.cos(theta)

		LHS = np.array([[ct,1],[2, ct]])
		rhs = np.array([[gravity*st],[u[0] + theta_d*theta_d*st]])

		LHS_inv = np.linalg.inv(LHS)

		xdd = np.matmul(LHS_inv,rhs)

		return xdd.flatten()

	def forward_dynamics_gradient(self, x: np.ndarray, u: np.ndarray):
		gravity = self.options['gravity']
		# assuming m_cart = m_pole = l_pole = 1
		q = x[0] # position of cart on track
		theta = x[1] # angle of pole
		q_d = x[2]
		theta_d = x[3]
		st = np.sin(theta)
		ct = np.cos(theta)

		LHS = np.array([[ct,1],[2, ct]])
		rhs = np.array([[gravity*st],[u[0] + theta_d*theta_d*st]])
		
		LHS_inv = np.linalg.inv(LHS)

		LHS_dtheta = np.array([[-st,0],[0, -st]])
		rhs_dtheta = np.array([[gravity*ct],[theta_d*theta_d*ct]])
		rhs_dtheta_d = np.array([[0],[2*theta_d*st]])
		rhs_du = np.array([[0],[1]])

		LHS_inv_dtheta = -np.matmul(LHS_inv,np.matmul(LHS_dtheta,LHS_inv))

		result = np.zeros((2,5))
		result[:,1:2] = np.matmul(LHS_inv_dtheta,rhs) + np.matmul(LHS_inv,rhs_dtheta)
		result[:,3:4] = np.matmul(LHS_inv,rhs_dtheta_d)
		result[:,4:5] = np.matmul(LHS_inv,rhs_du)

		return result

	def get_num_pos(self):
		return 2

	def get_num_vel(self):
		return 2

	def get_num_cntrl(self):
		return 1

class URDFPlant(TrajoptPlant):
	def __init__(self, integrator_type = 0, options = {}):
		super().__init__(integrator_type, options, True)
		parser = URDFParser()
		self.robot = parser.parse(options['path_to_urdf'])
		self.rbdReference = RBDReference(self.robot)

	def forward_dynamics(self, x: np.ndarray, u: np.ndarray):
		nq = self.get_num_pos()
		q = x[0:nq]
		qd = x[nq:]
		(c, _, _, _) = self.rbdReference.rnea(q, qd, None, self.options['gravity'])
		Minv = self.rbdReference.minv(q)
		qdd = np.matmul(Minv,(u-c))
		return qdd

	def forward_dynamics_gradient(self, x: np.ndarray, u: np.ndarray):
		nq = self.get_num_pos()
		q = x[0:nq]
		qd = x[nq:]
		(c, _, _, _) = self.rbdReference.rnea(q, qd, None, self.options['gravity'])
		Minv = self.rbdReference.minv(q)
		qdd = np.matmul(Minv,(u-c))
		dc_du = self.rbdReference.rnea_grad(q, qd, qdd, self.options['gravity'])
		df_du = np.matmul(-Minv,dc_du)
		return np.hstack((df_du,Minv))

	def get_num_pos(self):
		return self.robot.get_num_pos()

	def get_num_vel(self):
		return self.robot.get_num_vel()

	def get_num_cntrl(self):
		return self.robot.get_num_cntrl()


class QuadrotorPlant(TrajoptPlant):
    def _init_(self, integrator_type = 0, options = {}):
        super()._init_(integrator_type, options)

        # base parameters
        self.g = 9.81 # m / s^2
        self.m = 0.468 # kg
        self.l = 0.225 # m, distance from center of mass to rotor
        self.k = 2.98e-6 # lift constant
        self.b = 1.14e-7 # drag constant

        # moments of inertia
        self.Ixx = 4.856e-3 # kg * m^2
        self.Iyy = 4.856e-3 # kg * m^2
        self.Izz = 8.801e-3 # kg * m^2
        self.Im = 3.3357e-5 # kg * m^2
        # drag coefficients
        self.Ax = 0.25 # kg / s
        self.Ay = 0.25 # kg / s
        self.Az = 0.25 # kg / s

    def forward_dynamics(self, x: np.ndarray, u: np.ndarray):
        # Assuming x = [px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r]
        phi, theta, psi = x[3:6]
        p, q, r = x[9:12]

        # Control inputs u = [w1, w2, w3, w4]
        fi = self.k * u**2
        T = np.sum(fi)
        # tau_phi = self.l * self.k * (u[2]*2 - u[0]*2)
        # tau_theta = self.l * self.k * (u[3]*2 - u[1]*2)
        # tau_psi = self.b * (u[1]*2 - u[0]*2 + u[3]*2 - u[2]*2)
        tau_phi = self.l * self.k * (-u[1] ** 2 + u[3] ** 2)
        tau_theta = self.l * self.k * (-u[0] ** 2 + u[2] ** 2)
        tau_psi = self.b * np.sum(u ** 2)

        # Assemble torques into vector
        tau = np.array([tau_phi, tau_theta, tau_psi])

        C_phi = np.cos(phi)
        S_phi = np.sin(phi)
        T_phi = np.tan(phi)
        C_theta = np.cos(theta)
        S_theta = np.sin(theta)
        T_theta = np.tan(theta)
        C_psi = np.cos(psi)
        S_psi = np.sin(psi)
        T_psi = np.tan(psi)

        # Compute rotation matrix R from body to inertial frame
        R = np.array([
            [C_psi * C_theta, C_psi * S_theta * S_phi - S_psi * C_phi, C_psi * S_theta * C_phi + S_psi * S_phi],
            [S_psi * C_theta, S_psi * S_theta * S_phi - C_psi * C_phi, S_psi * S_theta * C_phi - C_psi * S_theta],
            [-S_theta, C_theta * S_phi, C_theta * C_phi]
        ])

        # Translational dynamics
        TB = np.array([0, 0, T])  # Thrust in body frame
        translational_vec = np.array([
            [C_psi*S_theta*C_phi+S_psi*S_phi],
            [S_psi*S_theta*C_phi-C_psi*S_phi],
            [C_theta*C_phi]
        ])
        tranlation_acc = (
            -self.g * np.array([[0],[0],[1]]) 
            + (T / self.m) @ translational_vec 
            # - (1 / self.m ) * np.diag([self.Ax, self.Ay, self.Az]) @ [x0dot, x1dot, x2dot]
        )

        # Rotational dynamics
        W_n_inv = np.array([
            [1, S_phi*T_theta, C_phi*T_theta],
            [0, C_phi, -S_phi],
            [0, S_phi/C_theta, C_phi/C_theta]
        ])
        n_dot = W_n_inv @ np.array([p,q,r])
        phi_dot = n_dot[0]
        theta_dot = n_dot[1]
        psi_dot = n_dot[2]
        w_gamma = u[0]-u[1]+u[2]-u[3]
        Ir = np.sqrt(self.Ixx ** 2 + self.Iyy ** 2 + self.Izz ** 2) # didn't find what is Ir assumed it norm of total moment
        # Tb = np.array[[
        #     [self.l*self.k*(-u[1]*2+u[3]*2)]
        #     [self.l*self.k*(-u[1]*2+u[2]*2)]
        #     [self.b*((u[0])*2+u[1]*2+u[2]*2+u[3]*2)]
        # ]]
        v_dot = np.array([
            [(self.Iyy-self.Izz)*q*r/self.Ixx]
            [(self.Izz-self.Ixx)*p*r/self.Iyy]
            [(self.Ixx-self.Iyy)*p*q/self.Izz]
        ]) - w_gamma*Ir * np.array([
            [q/self.Ixx]
            [-p/self.Iyy]
            [0]
        ]) + np.array([
            [tau_phi/self.Ixx]
            [tau_theta/self.Iyy]
            [tau_psi/self.Izz]
        ])

        rotational_acc_matrix = np.array([
            [0, phi_dot*C_phi*T_theta+theta_dot*S_phi/(C_theta**2), -phi_dot*S_phi*C_theta+theta_dot*C_phi/(C_theta**2)],
            [0,-phi_dot*S_phi,-phi_dot*C_phi],
            [0, phi_dot*C_phi/C_theta+T_theta*phi_dot*S_phi/(C_theta), -phi_dot*S_phi/C_theta+theta_dot*C_phi*T_theta/(C_theta)]
        ])
        v = np.array([
            [p]
            [q]
            [r]
        ])

        # Compute rotational accelerations in inertial frame
        rotational_acc =  rotational_acc_matrix @ v + W_n_inv @ v_dot
        acceleration = np.concatenate((tranlation_acc, rotational_acc))
        return acceleration

    def forward_dynamics_gradient(self, x: np.ndarray, u: np.ndarray):
        phi, theta, psi = x[3:6]
        p, q, r = x[9:12]

        fi = self.k * u**2
        T = np.sum(fi)

        C_phi = np.cos(phi)
        S_phi = np.sin(phi)
        T_phi = np.tan(phi)
        C_theta = np.cos(theta)
        S_theta = np.sin(theta)
        T_theta = np.tan(theta)
        C_psi = np.cos(psi)
        S_psi = np.sin(psi)
        T_psi = np.tan(psi)

        # computing translational rotatation component gradients

        ang_trans_grad = T / self.m * np.array([
            [-C_psi * S_theta * S_phi + S_psi * C_phi, C_psi * C_theta * C_phi, - S_psi * S_theta * C_phi + C_psi * S_phi],
            [-S_psi * S_theta * S_phi - C_psi * C_phi, S_psi * C_theta * C_phi, C_psi * S_theta * C_phi + S_psi * S_phi],
            [-S_phi * C_theta, -S_theta * C_phi, 0]
        ])

        translational_vec = np.array([
            [C_psi*S_theta*C_phi+S_psi*S_phi],
            [S_psi*S_theta*C_phi-C_psi*S_phi],
            [C_theta*C_phi]
        ])

        torque_grad = 2 * self.k * u

        trans_control_grad = 1 / self.m * translational_vec @ torque_grad

        trans_grad = np.hstack((np.zeros(3, 3), ang_trans_grad, np.zeros(3, 6), trans_control_grad))

        # computing rotational component gradients, using central finite difference approximation
        # perturb by some h then grad = f(x + h) - f(x - h) / 2h

        h = 0.01
        
        angular_grad = np.hstack([self.central_diff_approx(x, u, i, h) for i in range(3, 6)])
        angular_vel_control_grad = np.hstack([self.central_diff_approx(x, u, i, h) for i in range(9, 16)])

        rot_grad = np.hstack((np.zeros(3, 3), angular_grad, np.zeros(3, 3), angular_vel_control_grad))

        return np.hstack((trans_grad, rot_grad))

    def central_diff_approx(self, x: np.ndarray, u: np.ndarray, i: int, h: float):
        if i < x.shape[0]:
            xa = x
            xb = x
            xa[i] += h
            xb[i] -= h

            return (self.forward_dynamics(xa, u) - self.forward_dynamics(xb, u)) / (2 * h)
            
        i -= x.shape[0]
        ua = u
        ub = u
        ua[i] += h
        ub[i] -= h

        return (self.forward_dynamics(x, ua) - self.forward_dynamics(x, ub)) / (2 * h)


    def get_num_pos(self):
        return 6

    def get_num_vel(self):
        return 6

    def get_num_cntrl(self):
        return 4
