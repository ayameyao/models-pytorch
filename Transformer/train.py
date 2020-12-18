import torch
import torch.nn as nn
import time
import math

from data import model, get_batch, bptt, device, ntokens, train_data, test_data, val_data
# from data import

criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train():
	model.train() 
	total_loss = 0.
	start_time = time.time()
	src_mask = model.generate_square_subsequent_mask(bptt).to(device)
	for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
		data, targets = get_batch(train_data, i)
		optimizer.zero_grad()
		if data.size(0) != bptt:
			src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
		output = model(data, src_mask)
		loss = criterion(output.view(-1, ntokens), targets)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) #对所有进行梯度调节，防止梯度爆炸
		optimizer.step()

		total_loss += loss.item()
		log_interval = 200
		if batch % log_interval == 0 and batch > 0:
			cur_loss = total_loss / log_interval
			elapsed = time.time() - start_time
			print('| epoch {:3d} | {:5d}/{:5d} batches | '
				  'lr {:02.2f} | ms/batch {:5.2f} | '
				  'loss {:5.2f} | ppl {:8.2f}'.format(
					epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
					elapsed * 1000 / log_interval,
					cur_loss, math.exp(cur_loss)))
			total_loss = 0
			start_time = time.time()

def evaluate(eval_model, data_source):
	eval_model.eval() 
	total_loss = 0.
	src_mask = model.generate_square_subsequent_mask(bptt).to(device)
	with torch.no_grad():
		for i in range(0, data_source.size(0) - 1, bptt):
			data, targets = get_batch(data_source, i)
			if data.size(0) != bptt:
				src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
			output = eval_model(data, src_mask)
			output_flat = output.view(-1, ntokens)
			total_loss += len(data) * criterion(output_flat, targets).item()
	return total_loss / (len(data_source) - 1)

best_val_loss = float("inf")
epochs = 1 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
	epoch_start_time = time.time()
	train()
	val_loss = evaluate(model, val_data)
	print('-' * 89)
	print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
		  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
									 val_loss, math.exp(val_loss)))
	print('-' * 89)

	if val_loss < best_val_loss:
		best_val_loss = val_loss
		best_model = model

	scheduler.step()


test_loss = evaluate(best_model, test_data)
print('=' * 100)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
	test_loss, math.exp(test_loss)))
print('=' * 100)