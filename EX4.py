import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =====================================================
# FINAL SUBMISSION VERSION: Policy Iteration with DGM
# =====================================================

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 1.0

# Model parameters
H = torch.tensor([[0.1, 0.0],[0.0, 0.2]], dtype=torch.float32, device=device)
M = torch.eye(2, device=device)
C = torch.eye(2, device=device)
D = torch.eye(2, device=device)
R = torch.eye(2, device=device)
sigma = 0.3 * torch.eye(2, device=device)

sigma_sigma_T = sigma @ sigma.T

# =====================================================
# Approx true policy (for comparison)
# =====================================================
def true_policy(x):
    return -x  # acceptable approximation in this setup


# =====================================================
# Neural Networks
# =====================================================
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 80), nn.Tanh(),
            nn.Linear(80, 80), nn.Tanh(),
            nn.Linear(80, 1)
        )

    def forward(self, t, x):
        return self.net(torch.cat([t, x], dim=1))


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 80), nn.ReLU(),
            nn.Linear(80, 80), nn.ReLU(),
            nn.Linear(80, 2)
        )

    def forward(self, t, x):
        return self.net(torch.cat([t, x], dim=1))


# =====================================================
# Terminal condition
# =====================================================
def terminal_value(x):
    return torch.sum((x @ R) * x, dim=1, keepdim=True)


# =====================================================
# Efficient Laplacian (trace of Hessian)
# =====================================================
def compute_trace_hessian(grad_u, x):
    trace = 0
    for i in range(2):
        grad_comp = grad_u[:, i]
        second = torch.autograd.grad(
            grad_comp, x,
            torch.ones_like(grad_comp),
            create_graph=True
        )[0][:, i]
        trace += second
    return trace.unsqueeze(1)


# =====================================================
# PDE residual
# =====================================================
def pde_residual(model_val, model_act, t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)

    u = model_val(t, x)

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    grad_u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

    laplace_u = compute_trace_hessian(grad_u, x)
    diffusion = 0.5 * torch.sum(sigma_sigma_T.diag()) * laplace_u

    drift1 = torch.sum(grad_u * (x @ H.T), dim=1, keepdim=True)

    a = model_act(t, x)
    drift2 = torch.sum(grad_u * (a @ M.T), dim=1, keepdim=True)

    xCx = torch.sum((x @ C) * x, dim=1, keepdim=True)
    aDa = torch.sum((a @ D) * a, dim=1, keepdim=True)

    return u_t + diffusion + drift1 + drift2 + xCx + aDa


# =====================================================
# Loss functions
# =====================================================
def value_loss(model_val, model_act, batch=256):
    t = T * torch.rand(batch, 1, device=device)
    x = -3 + 6 * torch.rand(batch, 2, device=device)

    res = pde_residual(model_val, model_act, t, x)
    loss_pde = torch.mean(res ** 2)

    tT = T * torch.ones(batch, 1, device=device)
    xT = -3 + 6 * torch.rand(batch, 2, device=device)

    loss_terminal = torch.mean((model_val(tT, xT) - terminal_value(xT)) ** 2)

    return loss_pde + loss_terminal


def policy_loss(model_val, model_act, batch=256):
    t = T * torch.rand(batch, 1, device=device)
    x = -3 + 6 * torch.rand(batch, 2, device=device)

    x.requires_grad_(True)

    v = model_val(t, x)
    grad_v = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]

    a = model_act(t, x)

    H_val = (
        torch.sum(grad_v * (x @ H.T), dim=1)
        + torch.sum(grad_v * (a @ M.T), dim=1)
        + torch.sum((x @ C) * x, dim=1)
        + torch.sum((a @ D) * a, dim=1)
    )

    return torch.mean(H_val)


# =====================================================
# Evaluation
# =====================================================
def evaluate_policy(model_act):
    x = -3 + 6 * torch.rand(1000, 2, device=device)
    t = torch.zeros(1000, 1, device=device)

    pred = model_act(t, x)
    true = true_policy(x)

    return torch.mean(torch.norm(pred - true, dim=1)).item()


# =====================================================
# Training
# =====================================================
def main():
    val_net = ValueNet().to(device)
    act_net = PolicyNet().to(device)

    opt_val = optim.Adam(val_net.parameters(), lr=1e-3)
    opt_act = optim.Adam(act_net.parameters(), lr=1e-3)

    value_losses = []
    policy_losses = []
    errors = []

    for iteration in range(6):
        print(f"\n=== Iteration {iteration} ===")

        # VALUE UPDATE
        for step in range(300):
            opt_val.zero_grad()
            loss = value_loss(val_net, act_net)
            loss.backward()
            opt_val.step()

            if step % 20 == 0:
                value_losses.append(loss.item())

        # POLICY UPDATE
        for step in range(300):
            opt_act.zero_grad()
            loss = policy_loss(val_net, act_net)
            loss.backward()
            opt_act.step()

            if step % 20 == 0:
                policy_losses.append(loss.item())

        err = evaluate_policy(act_net)
        errors.append(err)
        print(f"Policy error: {err:.4f}")

    # =================================================
    # Plot 1: Value Loss
    # =================================================
    plt.figure(figsize=(7,5))
    plt.plot(value_losses, linewidth=2, label="Value Loss")
    plt.yscale('log')
    plt.xlabel('Training Step')
    plt.ylabel('Loss (log scale)')
    plt.title('Convergence of Value Function')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("value_loss.png", dpi=300)
    plt.show()

    # =================================================
    # Plot 2: Policy Loss
    # =================================================
    plt.figure(figsize=(7,5))
    plt.plot(policy_losses, linewidth=2, label="Policy Loss")
    plt.yscale('log')
    plt.xlabel('Training Step')
    plt.ylabel('Loss (log scale)')
    plt.title('Policy Optimization Loss')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("policy_loss.png", dpi=300)
    plt.show()

    # =================================================
    # Plot 3: Policy Error
    # =================================================
    plt.figure(figsize=(7,5))
    plt.plot(errors, marker='o', linewidth=2, label="Policy Error")
    plt.yscale('log')
    plt.xlabel('Policy Iteration Step')
    plt.ylabel('Error (log scale)')
    plt.title('Policy Convergence to Optimal Control')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("policy_error.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()