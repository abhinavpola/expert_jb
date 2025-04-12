# Expert Jailbreaker

This is an attempt to train a model (Llama 4 Scout) using reinforcement learning (GRPO) to jailbreak other models.

## Usage

```
sudo docker build -t training .
sudo docker run --gpus all -v ~/grpo:/app/grpo -e WANDB_API_KEY=wandb_api_key -e OPENAI_API_KEY=openai_api_key training
```

At the moment, this OOMs on 1 x H100. Fix is WIP.

