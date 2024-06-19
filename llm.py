from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

# Load a small dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = dataset['text'][:1000]  # Use only the first 1000 lines for simplicity

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples, return_tensors='pt', truncation=True, padding=True, max_length=512)

tokenized_texts = [tokenize_function(text) for text in texts]

# Load a pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Prepare the tokenized data for training
inputs = [item['input_ids'] for item in tokenized_texts]
labels = [item['input_ids'] for item in tokenized_texts]

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # number of training epochs
    per_device_train_batch_size=2,   # batch size for training
    save_steps=10_000,               # number of updates steps before checkpoint
    save_total_limit=2,              # limit the total amount of checkpoints
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=inputs,                # training dataset
)

trainer.train()

results = trainer.evaluate()
print(results)

# Generate text
input_text = "who are you?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate continuation
outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

