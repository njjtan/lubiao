import torch
import torch.utils.data.distributed
# from model import efficientnet_b0 as create_model
import torch
from models.experimental import attempt_load
from torch.utils.mobile_optimizer import optimize_for_mobile

device = torch.device('cpu')
model_pth = r'F:\lubiao\runs\evolve\weights\best.pth'
mobile_pt =r'F:\lubiao\runs\evolve\weights\model2.ptl'
model = attempt_load(model_pth, map_location=device)
model.eval()
example = torch.rand(1, 3, 640, 640)
traced_script_module = torch.jit.trace(model, example, strict=False)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter(mobile_pt)