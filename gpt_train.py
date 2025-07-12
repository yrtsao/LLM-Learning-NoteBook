import gpt
import gpt_model_function
import tiktoken
import torch



GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate_emb": 0.1,
    "drop_rate_ShortCut": 0.1,
    "drop_rate_mha": 0.1,
    "qkv_bias": False
}


model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
}

model_name = "gpt2-small (124M)"  # import OpenAI pre-Train weight
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

models_dir = "gpt2\\124M"
load_settings, load_params = gpt_model_function.load_gpt2(models_dir)

gpt_model = gpt.GPTModel(NEW_CONFIG)
gpt_model_function.load_weights_into_gpt(gpt_model, load_params)
#____修改Output從768 to 2 for classifier
#______ Freeze Model 修改前要做
# for param in gpt_model.parameters():
#     param.requires_grad = False
# num_class = 2 
# gpt_model.out_head = torch.nn.Linear(
#     in_features=GPT_CONFIG_124M['emb_dim'],
#     out_features=num_class
# )

# for param in gpt_model.trf_blocks[-1].parameters(): # 啟動trf_blocks最後一層的梯度
#     param.requires_grad = True
# for param in gpt_model.final_norm.parameters():
#     param.requires_grad = True
# ________________
device = torch.device("cuda")
gpt_model.to(device)
print(gpt_model)


tokenizer = tiktoken.get_encoding("gpt2")
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text=f.read()
total_character = len(raw_text)
total_tokens = len(tokenizer.encode(raw_text))
train_ratio = 0.9
split_idx = int(train_ratio * total_character)
# print(split_idx)
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]
# print("train_data:",train_data)
# print("_________________________________________________")
# print("val_data:",val_data)

train_loader = gpt_model_function.crate_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = gpt_model_function.crate_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
print("train_loader")
for x,y in train_loader:
    print(x.shape, y.shape)

print("val_loader")
for x,y in val_loader:
    print(x.shape, y.shape)


run_epochs = 5
start_context = "Every effort moves you"

optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=5e-4, weight_decay=0.1)
train_losses, val_losses, tokens_seen = gpt_model_function.train_model_simple(
    gpt_model, train_loader, val_loader, optimizer, device, num_eochs=run_epochs, eval_freq=5, eval_iter=5,
    start_context=start_context, tokenizer=tokenizer)

print("model saving.....")
torch.save({
    "model_state_dict": gpt_model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
    },
"model_and_optimizer.pth"
)
print("model saved. Create model_and_optimizer.pth")