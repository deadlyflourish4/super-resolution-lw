import torch
from DIPnet import DIPNet  # hoặc DIPNetQAT nếu bạn muốn

model = DIPNet()
model.load_state_dict(torch.load("trained_model/student_dipnet_distilled.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 64, 64)  # hoặc size gốc của bạn

torch.onnx.export(
    model,
    dummy_input,
    "dipnet.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
print("✅ Xuất ONNX thành công")

