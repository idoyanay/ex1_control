# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.linalg import solve_discrete_are


    



# def simulate_lqr_response(A, B, Q, R, x0, x_ref, T):
#     """Simulates LQR control for a given initial condition."""
#     P = solve_discrete_are(A, B, Q, R)
#     K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

#     x = np.zeros((T + 1, len(x0)))
#     u = np.zeros(T)
#     x[0] = x0

#     for t in range(T):
#         xtilde = x[t] - x_ref
#         u[t] = -K @ xtilde
#         x[t + 1] = A @ x[t] + (B * u[t]).flatten()

#     return x, u

# # --- System definition ---
# A = np.array([[1, 1], [0, 1]])
# B = np.array([[0], [1]])
# Q = np.eye(2)
# R = np.eye(1)
# x_ref = np.array([3, 0])
# T = 25  # time horizon

# x0s = {
#     "[0, 2]": np.array([0, 2]),

# }



# # # --- Q/R settings ---
# # cases = {
# #     "Q=[100, 1], R=1": (np.diag([100, 1]), np.eye(1)),
# #      "Q=I, R=100": (np.diag([1, 1]), np.eye(1)*100),
# #     "Q=I, R=0.01": (np.eye(2), np.eye(1) * 0.01)
    
    
# # }


# # # --- Plot each Q/R in a separate figure ---
# # for case_label, (Q, R) in cases.items():
# #     fig, axs = plt.subplots(3, 1, figsize=(8, 10))
# #     fig.suptitle(f"LQR Response for {case_label}", fontsize=14)

# #     for i, (x0_label, x0) in enumerate(x0s.items()):
# #         x, u = simulate_lqr_response(A, B, Q, R, x0, x_ref, T)

# #         time = np.arange(T + 1)
# #         axs[i].plot(time, x[:, 0], label="Position", color='blue')
# #         axs[i].plot(time, x[:, 1], label="Velocity", color='orange')
# #         axs[i].plot(np.arange(T), u, label="Control input", color='green')
# #         axs[i].grid(True)
# #         axs[i].set_title(f"Initial state: x₀ = {x0_label}")
# #         axs[i].set_xlabel("Time step")
# #         axs[i].set_ylabel("Value")
# #         axs[i].legend()

# #     plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
# #     plt.show()



# # --- Setup subplot grid: 2 rows (state, control) × 3 columns (conditions) ---
# fig, axs = plt.subplots(2, 3, figsize=(15, 8))
# time = np.arange(T + 1)

# for i, (label, x0) in enumerate(x0s.items()):
#     x, u = simulate_lqr_response(A, B, Q, R, x0, x_ref, T)

#     # Top row: position & velocity in one plot
#     axs[0, i].plot(time, x[:, 0], label="Position", color='blue')
#     axs[0, i].plot(time, x[:, 1], label="Velocity", color='orange')
#     axs[0, i].set_title(label)
#     axs[0, i].grid(True)
#     axs[0, i].legend()
#     axs[0, i].set_ylabel("States")

#     # Bottom row: control input
#     axs[1, i].plot(np.arange(T), u, color='green', label="Control input")

#     axs[1, i].grid(True)
#     axs[1, i].set_xlabel("Time step")
#     axs[1, i].set_ylabel("u")
#     axs[1, i].legend()

# plt.tight_layout()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

def simulate_lqr(A, B, Q, R, x0, x_ref, T):
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    x = np.zeros((T + 1, len(x0)))
    u = np.zeros(T)
    x[0] = x0

    for t in range(T):
        xtilde = x[t] - x_ref
        u[t] = -K @ xtilde
        x[t + 1] = A @ x[t] + (B * u[t]).flatten()

    return x, u

# --- System parameters ---
A = np.array([[1, 1], [0, 1]])
B = np.array([[0], [1]])
Q = np.eye(2)
R = np.eye(1)
T = 30

# --- Cases to simulate ---
cases = {
    "Case 1: x0=[0,2], x_ref=[3,0]": {
        "x0": np.array([0, 2]),
        "x_ref": np.array([3, 0])
    },
    "Case 2: x0=[0,3], x_ref=[3,1]": {
        "x0": np.array([0, 3]),
        "x_ref": np.array([3, 1])
    }
}

# --- Plotting ---
for title, data in cases.items():
    x0 = data["x0"]
    x_ref = data["x_ref"]
    x, u = simulate_lqr(A, B, Q, R, x0, x_ref, T)

    xtilde = x - x_ref  # shift all x to be relative to x_ref
    time = np.arange(T + 1)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    fig.suptitle(title + " (in deviation from x_ref)")

        # Plot deviation from reference (states)
    axs[0].plot(time, xtilde[:, 0], label="Position Error", color="blue")
    axs[0].plot(time, xtilde[:, 1], label="Velocity Error", color="orange")
    axs[0].set_ylabel("Deviation from x_ref")
    axs[0].grid(True)
    axs[0].legend()

    # Control input (step to match discrete action timing)
    axs[1].step(np.arange(T), u, where='post', label="Control Input", color="green")
    axs[1].set_xlabel("Time step")
    axs[1].set_ylabel("u")
    axs[1].grid(True)
    axs[1].legend()


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
