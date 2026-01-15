# 测试代码
from ultralytics. nn.modules import BidirectionalGate, GateSelector
import torch

# 创建模块
gate = BidirectionalGate(c1=512, reduction=16)
selector0 = GateSelector(index=0)
selector1 = GateSelector(index=1)

# 测试
x1 = torch.randn(1, 512, 20, 20)
x2 = torch.randn(1, 512, 20, 20)

outputs = gate([x1, x2])
print(f"✅ BidirectionalGate: {len(outputs)} outputs")

vis = selector0(outputs)
ir = selector1(outputs)
print(f"✅ GateSelector[0]: {vis. shape}")
print(f"✅ GateSelector[1]: {ir.shape}")