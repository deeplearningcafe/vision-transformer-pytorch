import torch
from utils.prepare_data import prepare_test
from utils.prepare_model import prepare_model, prepare_training
import utils.visualization
import hydra
import omegaconf

def debug_inference(model, batch: torch.tensor):
    output = model(batch[0])
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = loss_fn(output, batch[1])
    preds = torch.argmax(output, dim=1)
    acc = torch.sum(preds == batch[1].data) / preds.shape[0]
    print(f"Accuracy: {acc}")
    
    loss.backward()
    print(loss)
    print("*"*50)
    
    
    layers = ["mlp_head.predictor.weight", "blocks.1.mhsa.projection.weight",
          "blocks.0.mhsa.projection.weight", "blocks.0.mhsa.q.weight",
          "blocks.1.mhsa.q.weight", "blocks.11.mhsa.projection.weight",
          "blocks.11.mhsa.q.weight",]
    utils.visualization.plot_gradients(layers, model)
    
    print("*"*50)
    layers = ["blocks.0.mlp.projection_in.weight", "blocks.0.mlp.projection_out.weight",
            "blocks.5.mlp.projection_out.weight", "blocks.5.mlp.projection_in.weight",
          "blocks.11.mlp.projection_out.weight", "blocks.11.mlp.projection_in.weight",]
    utils.visualization.plot_gradients(layers, model)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(conf:omegaconf.DictConfig):
    utils.visualization.plot_logs(r"logs\log_output_20240712-202633.csv")
    conf.train.device = "cpu"
    model = prepare_model(conf)
    # model, optim, scheduler, loss_fn = prepare_training(conf)
    val_loader = prepare_test(conf)

    batch = next(iter(val_loader))
    debug_inference(model, batch)

if __name__ == "__main__":
    main()