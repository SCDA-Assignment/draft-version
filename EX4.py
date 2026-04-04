import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =========================================================
# DEVICE / SEED
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# =========================================================
# 1. FFN: used directly for control a(t,x)
# =========================================================
class FFN(nn.Module):
    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity, batch_norm=False):
        super().__init__()

        layers = [nn.BatchNorm1d(sizes[0])] if batch_norm else []
        for j in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[j], sizes[j + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(sizes[j + 1], affine=True))
            if j < len(sizes) - 2:
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =========================================================
# 2. Net: used for value v(t,x)
# input dim = 3 = (t, x1, x2)
# output dim = 1
# =========================================================
class Net(nn.Module):
    def __init__(self, n_layer, n_hidden, dim):
        super().__init__()
        self.input_layer = nn.Linear(dim, n_hidden)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layer)])
        self.output_layer = nn.Linear(n_hidden, 1)

    def act(self, x):
        return x * torch.sigmoid(x)  # swish

    def forward(self, x):
        o = self.act(self.input_layer(x))
        for li in self.hidden_layers:
            o = self.act(li(o))
        return self.output_layer(o)

# =========================================================
# 3. AUTOGRAD UTILS
# =========================================================
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]

def diffusion_term_from_tx(v, tx, sigma):
    """
    tx = [t, x1, x2]
    return 0.5 * tr(sigma sigma^T Hess_x v)
    """
    dv_dtx = torch.autograd.grad(
        v, tx,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        retain_graph=True
    )[0]

    vx = dv_dtx[:, 1:3]

    h_rows = []
    for i in range(2):
        grad_i = vx[:, i:i+1]
        second = torch.autograd.grad(
            grad_i, tx,
            grad_outputs=torch.ones_like(grad_i),
            create_graph=True,
            retain_graph=True
        )[0][:, 1:3]
        h_rows.append(second)

    Hx = torch.stack(h_rows, dim=1)  # (batch, 2, 2)
    A = sigma @ sigma.T
    return 0.5 * torch.einsum('ij,bij->b', A, Hx).reshape(-1, 1)

# =========================================================
# 4. RICCATI SOLVER: exact benchmark
# =========================================================
class RiccatiSolver:
    def __init__(self, H, M, C, D, R, sigma, T, device):
        self.H = H.float()
        self.M = M.float()
        self.C = C.float()
        self.D = D.float()
        self.R = R.float()
        self.sigma = sigma.float()
        self.T = T
        self.device = device

        self.Dinv = torch.linalg.inv(self.D)
        self.S_list = []
        self.dt = None

    def solve(self, N=4000):
        self.dt = self.T / N
        S = self.R.clone()
        S_rev = [S.clone()]

        for _ in range(N):
            dS = -2 * self.H.T @ S + S @ self.M @ self.Dinv @ self.M.T @ S - self.C
            S = S - self.dt * dS
            S_rev.append(S.clone())

        self.S_list = list(reversed(S_rev))

    def S(self, t_scalar):
        idx = int(float(t_scalar) / self.T * (len(self.S_list) - 1))
        idx = max(0, min(idx, len(self.S_list) - 1))
        return self.S_list[idx]

    def value(self, t_scalar, x):
        """
        x: (batch, 2)
        returns: (batch, 1)
        """
        idx = int(float(t_scalar) / self.T * (len(self.S_list) - 1))
        idx = max(0, min(idx, len(self.S_list) - 1))
        S_t = self.S_list[idx]

        quad = torch.sum((x @ S_t) * x, dim=1, keepdim=True)

        const = 0.0
        A = self.sigma @ self.sigma.T
        for j in range(idx, len(self.S_list)):
            const += torch.trace(A @ self.S_list[j]).item() * self.dt

        const = torch.tensor([[const]], dtype=torch.float32, device=x.device)
        return quad + const

    def control(self, t_scalar, x):
        """
        x: (batch, 2)
        returns: (batch, 2)
        """
        S_t = self.S(t_scalar)
        K = self.Dinv @ self.M.T @ S_t
        return -(x @ K.T)

# =========================================================
# 5. PIA OBJECT
# critic minimizes:
#   R(theta) = R_eqn(theta) + R_boundary(theta)
# actor minimizes Hamiltonian
# =========================================================
class LQR_PIA:
    def __init__(self, critic, actor, H, M, C, D, R, sigma, T, device):
        self.critic = critic   # Net
        self.actor = actor     # FFN
        self.H = H
        self.M = M
        self.C = C
        self.D = D
        self.R = R
        self.sigma = sigma
        self.T = T
        self.device = device

    def pde_residual(self, size):
        t = torch.rand(size, 1, device=self.device) * self.T
        x = -2 + 4 * torch.rand(size, 2, device=self.device)

        tx = torch.cat([t, x], dim=1).clone().detach().requires_grad_(True)

        v = self.critic(tx)
        dv_dtx = grad(v, tx)

        vt = dv_dtx[:, 0:1]
        x_var = tx[:, 1:3]
        vx = dv_dtx[:, 1:3]
        a = self.actor(tx)

        drift = x_var @ self.H.T + a @ self.M.T
        diff = diffusion_term_from_tx(v, tx, self.sigma)

        xCx = torch.sum((x_var @ self.C) * x_var, dim=1, keepdim=True)
        aDa = torch.sum((a @ self.D) * a, dim=1, keepdim=True)

        residual = vt + diff + torch.sum(vx * drift, dim=1, keepdim=True) + xCx + aDa
        return residual

    def terminal_residual(self, size):
        x = -2 + 4 * torch.rand(size, 2, device=self.device)
        t = torch.ones(size, 1, device=self.device) * self.T
        tx = torch.cat([t, x], dim=1)

        vT = self.critic(tx)
        target = torch.sum((x @ self.R) * x, dim=1, keepdim=True)
        return vT - target

    def value_objective(self, size=2**8, return_parts=False):
        """
        R(theta) = R_eqn(theta) + R_boundary(theta)
        """
        eqn_res = self.pde_residual(size)
        bdry_res = self.terminal_residual(size)

        R_eqn = torch.mean(eqn_res ** 2)
        R_boundary = torch.mean(bdry_res ** 2)
        R_total = R_eqn + R_boundary

        if return_parts:
            return R_total, R_eqn, R_boundary
        return R_total

    def hamiltonian(self, size=2**8):
        t = torch.rand(size, 1, device=self.device) * self.T
        x = -2 + 4 * torch.rand(size, 2, device=self.device)

        tx = torch.cat([t, x], dim=1).clone().detach().requires_grad_(True)

        v = self.critic(tx)
        dv_dtx = grad(v, tx)

        x_var = tx[:, 1:3]
        vx = dv_dtx[:, 1:3]
        a = self.actor(tx)

        Hx = x_var @ self.H.T
        Ma = a @ self.M.T

        Hamil = (
            torch.sum(vx * Hx, dim=1, keepdim=True)
            + torch.sum(vx * Ma, dim=1, keepdim=True)
            + torch.sum((x_var @ self.C) * x_var, dim=1, keepdim=True)
            + torch.sum((a @ self.D) * a, dim=1, keepdim=True)
        )
        return torch.mean(Hamil)

# =========================================================
# 6. TRAINER
# no numpy
# no MSE
# =========================================================
class TrainPIA:
    def __init__(self, critic, actor, pia, ric, device):
        self.critic = critic
        self.actor = actor
        self.pia = pia
        self.ric = ric
        self.device = device

        # outer-iteration summaries
        self.outer_idx = []
        self.min_total_obj_list = []
        self.min_eqn_obj_list = []
        self.min_boundary_obj_list = []
        self.min_hamiltonian_list = []

        self.value_abs_error_list = []
        self.action_abs_error_list = []

        # epoch-level histories for DGM-style plots
        self.value_epoch_hist = []
        self.total_loss_hist = []
        self.pde_loss_hist = []
        self.terminal_loss_hist = []

        self.error_eval_epoch_hist = []
        self.value_mae_hist = []

    def evaluate_absolute_errors(self, n_test=300):
        x = -2 + 4 * torch.rand(n_test, 2, device=self.device)
        t = torch.zeros(n_test, 1, device=self.device)
        tx = torch.cat([t, x], dim=1)

        with torch.no_grad():
            v_true = self.ric.value(0.0, x)
            a_true = self.ric.control(0.0, x)

            v_nn = self.critic(tx)
            a_nn = self.actor(tx)

            value_abs_error = torch.mean(torch.abs(v_nn - v_true)).item()
            action_abs_error = torch.mean(torch.abs(a_nn - a_true)).item()

        return value_abs_error, action_abs_error

    def compute_value_mae(self, n_test=400):
        x = -2 + 4 * torch.rand(n_test, 2, device=self.device)
        t = torch.rand(n_test, 1, device=self.device) * self.pia.T
        tx = torch.cat([t, x], dim=1)

        with torch.no_grad():
            v_nn = self.critic(tx)

            v_true_list = []
            for i in range(n_test):
                ti = float(t[i].item())
                xi = x[i:i+1]
                v_true_list.append(self.ric.value(ti, xi))
            v_true = torch.cat(v_true_list, dim=0)

            mae = torch.mean(torch.abs(v_nn - v_true)).item()

        return mae

    def train(self, outer_iters=8, value_steps=3000, policy_steps=1500,
              lr_v=0.002, lr_a=0.001, eval_every=250):
        opt_v = optim.Adam(self.critic.parameters(), lr=lr_v)
        opt_a = optim.Adam(self.actor.parameters(), lr=lr_a)

        global_value_epoch = 0

        for k in range(outer_iters):
            print(f"\n=== Outer Iteration {k+1}/{outer_iters} ===")

            # -------------------------
            # critic update
            # -------------------------
            self.actor.eval()
            self.critic.train()

            min_total_obj = float("inf")
            min_eqn_obj = float("inf")
            min_boundary_obj = float("inf")

            for e in range(value_steps):
                global_value_epoch += 1

                opt_v.zero_grad()
                total_obj, eqn_obj, boundary_obj = self.pia.value_objective(size=2**8, return_parts=True)
                total_obj.backward()
                opt_v.step()

                min_total_obj = min(min_total_obj, total_obj.item())
                min_eqn_obj = min(min_eqn_obj, eqn_obj.item())
                min_boundary_obj = min(min_boundary_obj, boundary_obj.item())

                self.value_epoch_hist.append(global_value_epoch)
                self.total_loss_hist.append(total_obj.item())
                self.pde_loss_hist.append(eqn_obj.item())
                self.terminal_loss_hist.append(boundary_obj.item())

                if global_value_epoch % eval_every == 0:
                    value_mae = self.compute_value_mae(n_test=400)
                    self.error_eval_epoch_hist.append(global_value_epoch)
                    self.value_mae_hist.append(value_mae)

                if e % 300 == 299:
                    print(
                        f"Value step {e+1} | "
                        f"R_total = {total_obj.item():.6f}, "
                        f"R_eqn = {eqn_obj.item():.6f}, "
                        f"R_boundary = {boundary_obj.item():.6f}"
                    )

            # -------------------------
            # actor update
            # -------------------------
            self.critic.eval()
            for p in self.critic.parameters():
                p.requires_grad = False

            self.actor.train()
            min_hamiltonian = float("inf")

            for e in range(policy_steps):
                opt_a.zero_grad()
                hamil = self.pia.hamiltonian(size=2**8)
                hamil.backward()
                opt_a.step()

                min_hamiltonian = min(min_hamiltonian, hamil.item())

                if e % 300 == 299:
                    print(f"Policy step {e+1} | Hamiltonian = {hamil.item():.6f}")

            for p in self.critic.parameters():
                p.requires_grad = True

            # -------------------------
            # outer summary
            # -------------------------
            value_abs_error, action_abs_error = self.evaluate_absolute_errors()

            self.outer_idx.append(k + 1)
            self.min_total_obj_list.append(min_total_obj)
            self.min_eqn_obj_list.append(min_eqn_obj)
            self.min_boundary_obj_list.append(min_boundary_obj)
            self.min_hamiltonian_list.append(min_hamiltonian)
            self.value_abs_error_list.append(value_abs_error)
            self.action_abs_error_list.append(action_abs_error)

            print(
                f"Outer {k+1} summary | "
                f"min R_total = {min_total_obj:.6e}, "
                f"min R_eqn = {min_eqn_obj:.6e}, "
                f"min R_boundary = {min_boundary_obj:.6e}, "
                f"min Hamiltonian = {min_hamiltonian:.6e}, "
                f"value abs error = {value_abs_error:.6e}, "
                f"action abs error = {action_abs_error:.6e}"
            )

# =========================================================
# 7. STANDARD FIGURES
# 上图：原图 + inset
# 下图：error
# =========================================================
def plot_all_results(critic, actor, ric, trainer, device):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    x1_grid = torch.linspace(-2, 2, 400, device=device).reshape(-1, 1)
    x2_grid = torch.zeros_like(x1_grid)
    t_grid = torch.zeros_like(x1_grid)

    x = torch.cat([x1_grid, x2_grid], dim=1)
    tx = torch.cat([t_grid, x1_grid, x2_grid], dim=1)

    with torch.no_grad():
        v_nn = critic(tx).squeeze(1).cpu()
        a_nn = actor(tx).cpu()
        v_true = ric.value(0.0, x).squeeze(1).cpu()
        a_true = ric.control(0.0, x).cpu()

    eps = 1e-12
    v_abs_err = torch.abs(v_nn - v_true) + eps
    a1_abs_err = torch.abs(a_nn[:, 0] - a_true[:, 0]) + eps
    a2_abs_err = torch.abs(a_nn[:, 1] - a_true[:, 1]) + eps

    x1_list = x1_grid.squeeze(1).cpu().tolist()
    v_true_list = v_true.tolist()
    v_nn_list = v_nn.tolist()
    a1_true_list = a_true[:, 0].tolist()
    a1_nn_list = a_nn[:, 0].tolist()
    a2_true_list = a_true[:, 1].tolist()
    a2_nn_list = a_nn[:, 1].tolist()

    # =====================================================
    # Figure 1: value comparison
    # =====================================================
    fig, axes = plt.subplots(
        2, 1, figsize=(7, 7),
        gridspec_kw={'height_ratios': [3, 1.5]},
        sharex=True
    )

    axes[0].plot(x1_list, v_true_list, label='Exact value', linewidth=2)
    axes[0].plot(x1_list, v_nn_list, '--', label='Learned value', linewidth=2)
    axes[0].set_title('Value Function Comparison')
    axes[0].set_ylabel(r'$v(0, x_1, 0)$')
    axes[0].legend()
    axes[0].grid(True)

    # inset 放大
    axins_v = inset_axes(axes[0], width="38%", height="38%", loc='upper right')
    axins_v.plot(x1_list, v_true_list, linewidth=2)
    axins_v.plot(x1_list, v_nn_list, '--', linewidth=2)

    # 可调放大范围
    x1_left_v, x1_right_v = -0.3, 0.3
    y1_v, y2_v = 0.20, 0.24

    axins_v.set_xlim(x1_left_v, x1_right_v)
    axins_v.set_ylim(y1_v, y2_v)
    axins_v.grid(True)
    axins_v.tick_params(labelsize=8)

    mark_inset(axes[0], axins_v, loc1=2, loc2=4, fc="none", ec="0.5")

    # 下图 error
    axes[1].plot(x1_list, v_abs_err.tolist(), linewidth=2)
    axes[1].set_xlabel(r'$x_1$  (with $t=0,\ x_2=0$)')
    axes[1].set_ylabel(r'$|error|$')
    axes[1].set_title('Absolute Error in Value Function')
    axes[1].grid(True, which='both')

    plt.tight_layout()
    plt.savefig("fig1_value_comparison_inset.png", dpi=220, bbox_inches='tight')
    plt.close()

    # =====================================================
    # Figure 2: control comparison
    # =====================================================
    fig, axes = plt.subplots(
        2, 1, figsize=(7, 7),
        gridspec_kw={'height_ratios': [3, 1.5]},
        sharex=True
    )

    axes[0].plot(x1_list, a1_true_list, label='Exact $a_1$', linewidth=2)
    axes[0].plot(x1_list, a1_nn_list, '--', label='Learned $a_1$', linewidth=2)
    axes[0].plot(x1_list, a2_true_list, label='Exact $a_2$', linewidth=2)
    axes[0].plot(x1_list, a2_nn_list, '--', label='Learned $a_2$', linewidth=2)
    axes[0].set_title('Control Comparison')
    axes[0].set_ylabel(r'$a(0, x_1, 0)$')
    axes[0].legend()
    axes[0].grid(True)

    # inset 放大
    axins_a = inset_axes(axes[0], width="38%", height="38%", loc='upper right')
    axins_a.plot(x1_list, a1_true_list, linewidth=2)
    axins_a.plot(x1_list, a1_nn_list, '--', linewidth=2)
    axins_a.plot(x1_list, a2_true_list, linewidth=2)
    axins_a.plot(x1_list, a2_nn_list, '--', linewidth=2)

    # 可调放大范围
    x1_left_a, x1_right_a = -0.3, 0.3
    y1_a, y2_a = -0.02, 0.02

    axins_a.set_xlim(x1_left_a, x1_right_a)
    axins_a.set_ylim(y1_a, y2_a)
    axins_a.grid(True)
    axins_a.tick_params(labelsize=8)

    mark_inset(axes[0], axins_a, loc1=2, loc2=4, fc="none", ec="0.5")

    # 下图 error
    axes[1].plot(x1_list, a1_abs_err.tolist(), label=r'$|a_1-a_1^*|$', linewidth=2)
    axes[1].plot(x1_list, a2_abs_err.tolist(), label=r'$|a_2-a_2^*|$', linewidth=2)
    axes[1].set_xlabel(r'$x_1$  (with $t=0,\ x_2=0$)')
    axes[1].set_ylabel(r'$|error|$')
    axes[1].set_title('Absolute Error in Control')
    axes[1].legend()
    axes[1].grid(True, which='both')

    plt.tight_layout()
    plt.savefig("fig2_control_comparison_inset.png", dpi=220, bbox_inches='tight')
    plt.close()

    # =====================================================
    # Figure 3: minimum objective history
    # 保持不变
    # =====================================================
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    axes[0].plot(trainer.outer_idx, trainer.min_total_obj_list, marker='o', linewidth=2, label='min $R(\\theta)$')
    axes[0].plot(trainer.outer_idx, trainer.min_eqn_obj_list, marker='s', linewidth=2, label='min $R_{eqn}(\\theta)$')
    axes[0].plot(trainer.outer_idx, trainer.min_boundary_obj_list, marker='^', linewidth=2, label='min $R_{boundary}(\\theta)$')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Objective value')
    axes[0].set_title('Minimum Critic Objective Across Outer Iterations')
    axes[0].legend()
    axes[0].grid(True, which='both')

    axes[1].plot(trainer.outer_idx, trainer.min_hamiltonian_list, marker='o', linewidth=2, label='min Hamiltonian')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Outer iteration')
    axes[1].set_ylabel('Hamiltonian')
    axes[1].set_title('Minimum Hamiltonian Across Outer Iterations')
    axes[1].legend()
    axes[1].grid(True, which='both')

    plt.tight_layout()
    plt.savefig("fig3_minimum_objective_history.png", dpi=220, bbox_inches='tight')
    plt.close()

    print("Saved figures:")
    print(" - fig1_value_comparison_inset.png")
    print(" - fig2_control_comparison_inset.png")
    print(" - fig3_minimum_objective_history.png")
    print(f"Max abs value error         = {torch.max(v_abs_err).item():.6e}")
    print(f"Mean abs value error        = {torch.mean(v_abs_err).item():.6e}")
    print(f"Max abs control error (a1)  = {torch.max(a1_abs_err).item():.6e}")
    print(f"Mean abs control error (a1) = {torch.mean(a1_abs_err).item():.6e}")
    print(f"Max abs control error (a2)  = {torch.max(a2_abs_err).item():.6e}")
    print(f"Mean abs control error (a2) = {torch.mean(a2_abs_err).item():.6e}")

# =========================================================
# 8. DGM-STYLE FIGURES
# =========================================================
def plot_dgm_style_results(trainer):
    plt.figure(figsize=(8, 5))
    plt.plot(trainer.value_epoch_hist, trainer.total_loss_hist, label='total loss')
    plt.plot(trainer.value_epoch_hist, trainer.pde_loss_hist, label='PDE loss')
    plt.plot(trainer.value_epoch_hist, trainer.terminal_loss_hist, label='terminal loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('EX4 PIA training loss')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig("fig4_dgm_style_training_loss.png", dpi=220, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    err_plot = [max(v, 1e-12) for v in trainer.value_mae_hist]
    plt.semilogy(trainer.error_eval_epoch_hist, err_plot, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean absolute error vs Riccati')
    plt.title('EX4 PIA error against Riccati benchmark')
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig("fig5_dgm_style_value_error.png", dpi=220, bbox_inches='tight')
    plt.close()

    print("Saved DGM-style figures:")
    print(" - fig4_dgm_style_training_loss.png")
    print(" - fig5_dgm_style_value_error.png")

# =========================================================
# 9. MAIN
# =========================================================
if __name__ == "__main__":
    # problem parameters
    T = 1.0

    H = torch.tensor([[0.1, 0.0],
                      [0.0, 0.2]], dtype=torch.float32, device=device)

    M = torch.eye(2, dtype=torch.float32, device=device)
    C = torch.eye(2, dtype=torch.float32, device=device)
    D = torch.eye(2, dtype=torch.float32, device=device)
    R = torch.eye(2, dtype=torch.float32, device=device)

    sigma = 0.3 * torch.eye(2, dtype=torch.float32, device=device)

    # only two neural networks:
    # Net for value v(t,x)
    critic = Net(n_layer=3, n_hidden=128, dim=3).to(device)

    # FFN for control a(t,x)
    actor = FFN(
        sizes=[3, 64, 64, 2],
        activation=nn.Tanh,
        output_activation=nn.Identity,
        batch_norm=False
    ).to(device)

    # exact benchmark
    ric = RiccatiSolver(H, M, C, D, R, sigma, T, device)
    ric.solve(N=4000)

    # PIA
    pia = LQR_PIA(critic, actor, H, M, C, D, R, sigma, T, device)
    trainer = TrainPIA(critic, actor, pia, ric, device)

    # train
    trainer.train(
        outer_iters=8,
        value_steps=3000,
        policy_steps=1500,
        lr_v=0.002,
        lr_a=0.001,
        eval_every=250
    )

    # standard figures
    plot_all_results(critic, actor, ric, trainer, device)

    # DGM-style figures
    plot_dgm_style_results(trainer)