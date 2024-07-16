import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def show_mask_on_image(img, mask):
    heatmap = plt.cm.jet(mask)[:, :, :3]  # Using matplotlib's colormap for heatmap
    cam = 1.0 * heatmap +  0.5 * np.float32(img)
    cam = cam / np.max(cam)
    return cam


class ViTAttentionGradRollout:
    def __init__(self, model: nn.Module, attention_layer_name='attention_dropout', discard_ratio=0.9) -> None:
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name == name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)
        self.attentions = []
        self.attention_gradients = []
        
    def get_attention(self, model, input, output):
        self.attentions.append(output.cpu())
    
    def get_attention_gradient(self, model, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input, labels):
        for name, param in self.model.named_parameters():
            param.grad = None
        output = self.model(input)

        preds = torch.argmax(output, dim=1)
        print(preds)
        label_mask = torch.zeros(output.size()).to(output.device)
        for label in labels:
            label_mask[:, label] = 1
        loss = (output*label_mask).sum()
        loss.backward()
        print(loss)
        
        return grad_rollout(self.attentions, self.attention_gradients)

def grad_rollout(attentions, gradients):
    result = torch.eye(attentions.attentions[0].size(-1))
    
    with torch.no_grad():
        for attentions, grads in zip(attentions, gradients):
            weights = grads
            attention_heads_fused = (attentions*weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0.0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1, keepdim=True)
            result = torch.matmul(a, result)
    
    eps = 1e-5
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask_torch = mask.reshape(width, width)
    mask_torch = mask_torch / (torch.max(mask_torch) + eps)
    return mask_torch

def visualize_grad_rollout(model: nn.Module, batch: tuple[torch.tensor]):
    
    vit_attn = ViTAttentionGradRollout(model)
    mask_torch = vit_attn(batch[0], batch[1])
    
    np_img = np.array(batch[0][0].permute(1, 2, 0).to("cpu"))

    # Resize mask to match image dimensions using PIL
    mask_resized_torch = torch.nn.functional.interpolate(mask_torch.unsqueeze(0).unsqueeze(0), scale_factor=16, mode='bilinear')
    mask_resized_numpy = mask_resized_torch.squeeze(0).squeeze(0).numpy()
    # Apply the mask on the image
    masked_img = show_mask_on_image(np_img, mask_resized_numpy)
    print(masked_img.max(), masked_img.min())
    # Display images using matplotlib
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    shw = plt.imshow(np_img)
    bar = plt.colorbar(shw)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    shw = plt.imshow(masked_img)
    bar = plt.colorbar(shw)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
