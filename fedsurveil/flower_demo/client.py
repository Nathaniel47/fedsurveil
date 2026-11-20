"""Flower ClientApp: one site trains the global model on its own data and sends back the weights."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedsurveil.fl.client import local_train
from fedsurveil.flower_demo.common import load_site, new_model

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = context.run_config
    partition = int(context.node_config["partition-id"])
    num_sites = int(context.node_config["num-partitions"])
    x, y = load_site(
        str(cfg["dataset"]),
        partition,
        num_sites,
        float(cfg["alpha"]),
        int(cfg["seed"]),
        int(cfg["limit"]) or None,
    )
    model = new_model(str(cfg["dataset"]), int(cfg["seed"]))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    round_no = int(msg.content["config"]["server-round"])
    loss = local_train(
        model,
        x,
        y,
        epochs=int(cfg["local-epochs"]),
        lr=float(cfg["lr"]),
        batch_size=64,
        seed=int(cfg["seed"]) * 100003 + round_no * 1009 + partition,
        momentum=float(cfg["momentum"]),
    )
    reply = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord({"train_loss": float(loss), "num-examples": len(x)}),
        }
    )
    return Message(content=reply, reply_to=msg)
