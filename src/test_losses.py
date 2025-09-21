import torch
import torch.optim as optim
from tqdm import tqdm

x = torch.tensor(0.0, requires_grad=True, device="cuda")
# t = torch.linspace(0,2*torch.pi,100000)
# x = torch.complex(
#     torch.cos(t),
#     torch.sin(t),)

# p = torch.angle(x)/torch.pi

# print(p.min(),p.max())
# print(torch.atanh(torch.tensor(-1.0)),torch.atanh(torch.tensor(1.0)))
y = torch.atanh(torch.tensor(1.0 - 1e-7)).cuda()
print(y)
print(y.tanh(), y)
# print(torch.norm(x%(2*torch.pi)))
opt = optim.Adam([x], lr=1e-3)
for _ in tqdm(range(int(1e5))):
    opt.zero_grad()
    loss = (x - y) ** 2
    loss.backward()
    opt.step()

print(x)
print("loss", loss)
print(x.grad)
