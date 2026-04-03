import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 基础函数：梯度、拉普拉斯
# ----------------------------
def get_gradient(output, x):
    grad = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    return grad

def get_laplacian(grad, x):
    hess_diag = []
    for d in range(x.shape[1]):
        v = grad[:,d].view(-1,1)
        grad2 = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(v),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        hess_diag.append(grad2[:,d].view(-1,1))    
    hess_diag = torch.cat(hess_diag,1)
    laplacian = hess_diag.sum(1, keepdim=True)
    return laplacian

# ----------------------------
# DGM 网络 (value function)
# ----------------------------
class DGM_Layer(nn.Module):
    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super().__init__()
        act = nn.Tanh if activation=='Tanh' else nn.ReLU
        self.activation = act()
        self.gate_Z = nn.Sequential(nn.Linear(dim_x+dim_S, dim_S), self.activation)
        self.gate_G = nn.Sequential(nn.Linear(dim_x+dim_S, dim_S), self.activation)
        self.gate_R = nn.Sequential(nn.Linear(dim_x+dim_S, dim_S), self.activation)
        self.gate_H = nn.Sequential(nn.Linear(dim_x+dim_S, dim_S), self.activation)
    def forward(self, x, S):
        x_S = torch.cat([x,S],1)
        Z = self.gate_Z(x_S)
        G = self.gate_G(x_S)
        R = self.gate_R(x_S)
        H = self.gate_H(torch.cat([x, S*R],1))
        return (1-G)*H + Z*S

class Net_DGM(nn.Module):
    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super().__init__()
        act = nn.Tanh if activation=='Tanh' else nn.ReLU
        self.input_layer = nn.Sequential(nn.Linear(dim_x+1, dim_S), act())
        self.DGM1 = DGM_Layer(dim_x+1, dim_S, activation)
        self.DGM2 = DGM_Layer(dim_x+1, dim_S, activation)
        self.DGM3 = DGM_Layer(dim_x+1, dim_S, activation)
        self.output_layer = nn.Linear(dim_S, 1)
    def forward(self, t, x):
        tx = torch.cat([t, x], 1)
        S1 = self.input_layer(tx)
        S2 = self.DGM1(tx,S1)
        S3 = self.DGM2(tx,S2)
        S4 = self.DGM3(tx,S3)
        return self.output_layer(S4)

# ----------------------------
# FFN 网络 (policy function)
# ----------------------------
class FFN(nn.Module):
    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        layers = []
        for j in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[j], sizes[j+1]))
            if j < len(sizes)-2:
                layers.append(activation())
            else:
                layers.append(output_activation())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ----------------------------
# 示例 Option 类
# ----------------------------
class ExampleOption:
    def payoff(self, x):
        return x.sum(dim=1, keepdim=True)

# ----------------------------
# Policy Iteration
# ----------------------------
def policy_iteration_dgm_ffn(T, d, hidden_dgm, hidden_ffn, ts, max_updates, batch_size, device):
    net_dgm = Net_DGM(d, hidden_dgm).to(device)
    net_ffn = FFN([d+1]+hidden_ffn+[d]).to(device)
    
    optimizer_v = torch.optim.Adam(net_dgm.parameters(), lr=0.001)
    optimizer_a = torch.optim.Adam(net_ffn.parameters(), lr=0.001)
    option = ExampleOption()
    
    policy_errors = []
    
    for pi_iter in range(5):
        print(f"\n=== Policy Iteration {pi_iter+1} ===")
        
        # Step 1: 固定 policy 更新 value
        for it in range(max_updates):
            optimizer_v.zero_grad()
            t_rand = T*torch.rand(batch_size, 1, device=device, requires_grad=True)
            x_rand = 0.5 + 2*torch.rand(batch_size, d, device=device, requires_grad=True)
            a_curr = net_ffn(torch.cat([t_rand, x_rand],1))
            u = net_dgm(t_rand, x_rand)
            grad_u_x = get_gradient(u, x_rand)
            grad_u_t = get_gradient(u, t_rand)
            laplacian = get_laplacian(grad_u_x, x_rand)
            pde = grad_u_t + torch.sum(grad_u_x * a_curr,1,keepdim=True) + 0.5*laplacian - u
            loss_v = (pde**2).mean()
            loss_v.backward()
            optimizer_v.step()
        
        # Step 2: 固定 v 更新 policy
        for it in range(max_updates):
            optimizer_a.zero_grad()
            t_rand = T*torch.rand(batch_size,1,device=device)
            x_rand = 0.5 + 2*torch.rand(batch_size,d,device=device)
            t_rand.requires_grad_(True)
            x_rand.requires_grad_(True)
            u = net_dgm(t_rand, x_rand)
            grad_u_x = get_gradient(u, x_rand)
            a_pred = net_ffn(torch.cat([t_rand, x_rand],1))
            hamiltonian = torch.sum(grad_u_x * a_pred,1,keepdim=True) + (a_pred**2).sum(1,keepdim=True)
            loss_a = hamiltonian.mean()
            loss_a.backward()
            optimizer_a.step()
        
        # policy error (示例，使用 norm)
        x_test = torch.tensor([[1.0,1.0]], device=device)
        t_test = torch.tensor([[0.5]], device=device)
        a_err = net_ffn(torch.cat([t_test, x_test],1)).detach()
        policy_errors.append(a_err.norm().item())
        print(f"Policy output norm: {a_err.norm().item():.4f}")
    
    # 绘制收敛曲线
    plt.figure()
    plt.plot(range(1, len(policy_errors)+1), policy_errors, marker='o')
    plt.title("Policy iteration error convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Policy error (norm)")
    plt.show()
    
    # 绘制 value function 对比
    x_vals = torch.linspace(0,2,20).unsqueeze(1).repeat(1,d).to(device)
    t_val = torch.zeros(20,1, device=device)
    v_vals = net_dgm(t_val, x_vals).detach().cpu().numpy()
    plt.figure()
    plt.plot(np.linspace(0,2,20), v_vals)
    plt.title("Value function v(t,x)")
    plt.show()
    
    return net_dgm, net_ffn

# ----------------------------
# 运行示例
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1.0
d = 2
hidden_dgm = 32
hidden_ffn = [32,32]
ts = torch.linspace(0,T,10)
net_v, net_a = policy_iteration_dgm_ffn(T, d, hidden_dgm, hidden_ffn, ts, max_updates=500, batch_size=64, device=device)