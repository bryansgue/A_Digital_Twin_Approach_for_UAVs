# **Bridging Simulation and Reality: A Digital Twin Approach for UAVs**

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <img src="H_Comparacion_Kinetic.gif" alt="GIF 1" width="200">
  <img src="H_Comparacion_Com_din.gif" alt="GIF 2" width="200">
  <img src="H_Comparacion_MPC.gif" alt="GIF 3" width="200">
</div>

## **Description**
This project implements a digital twin framework for UAVs, integrating simulation and real-world experimentation. The focus is on validating and optimizing UAV controllers, particularly a Model Predictive Controller (MPC), developed using ACADOS and CASADI. The framework is built with Python and leverages ROS for communication and sensor data integration.  

The repository includes:
- Simulation models.
- UAV controller implementations (kinematic, dynamic compensation, and MPC).
- Scripts for trajectory-following experiments.
- Tools for comparing simulated and experimental results.

---

## **Dependencies**
To run this project, ensure you have the following dependencies installed:

### **Core Dependencies**
1. **Python 3.x**
2. **ACADOS**: Required for MPC formulation and optimization.
   - Installation guide: [ACADOS Installation](https://github.com/acados/acados)
3. **CASADI**: Used for symbolic computation and defining optimization problems.
   - Install via pip: `pip install casadi`

### **Additional Python Libraries**
The project uses standard Python libraries for robotics, data handling, and visualization:
- `rospy`: ROS Python library for node management and communication.
- `geometry_msgs`: For handling geometry-related ROS messages.
- `nav_msgs`: For working with odometry data.
- `sensor_msgs`: For processing sensor data.
- `std_msgs`: For working with standard ROS message types.
- `matplotlib`: For plotting results (`pip install matplotlib`).
- `numpy`: For numerical operations (`pip install numpy`).
- `scipy`: For advanced mathematical computations (`pip install scipy`).

---

## **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository-name.git
   cd your-repository-name
   ```

2. Install the required dependencies:
   ```bash
   pip install casadi matplotlib numpy scipy
   ```

3. Follow the installation steps for **ACADOS** from [ACADOS GitHub Repository](https://github.com/acados/acados). Make sure the `acados_template` package is correctly configured in your Python environment.

4. Set up your ROS environment:
   - Install ROS (e.g., ROS Noetic for Ubuntu).
   - Source your ROS workspace:
     ```bash
     source /opt/ros/noetic/setup.bash
     ```

5. Ensure the UAV simulation tools (e.g., DJI Assistance 2, Webots) are properly installed and configured if required for the simulation portion of the project.

---

## **Acknowledgements**
This project was developed as part of research on UAV control systems and digital twin frameworks. It leverages open-source tools such as ROS, ACADOS, and CASADI to bridge simulation and reality.

