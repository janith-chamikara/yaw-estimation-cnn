import torch
import torch.onnx
from model import TunnelCNN

# --- CONFIGURATION ---
MODEL_PATH = "uav_yaw_net_best.pth"
ONNX_PATH = "tunnel_cnn.onnx"
DEVICE = torch.device("cpu")


def convert():
    print("1. Loading the PyTorch model...")
    model = TunnelCNN()

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    model.eval()

    print("2. Creating dummy input...")
    dummy_input = torch.randn(1, 3, 60, 80, requires_grad=True).to(DEVICE)

    print("3. Exporting to ONNX...")

    torch.onnx.export(
        model,                  # The model being run
        dummy_input,            # The dummy input
        ONNX_PATH,              # Where to save
        export_params=True,     # Store the trained weights inside the file
        # Name the input node (for easy access later)
        input_names=['input_image'],
        output_names=['yaw_angle'],    # Name the output node
    )

    print(f"✅ Success! Model saved to: {ONNX_PATH}")
    print("You can now transfer this file to your Raspberry Pi.")


if __name__ == "__main__":
    convert()
