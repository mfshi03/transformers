import matplotlib.pyplot as plt

# Data

epochs = []
train_loss = []
train_ppl = []

with open("training.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        spl = line.split("|")
        epochs.append(int(spl[0].split(":")[1]))
        train_loss.append(float(spl[1].split(":")[1]))
        train_ppl.append(float(spl[2].split(":")[1]))

#epochs = [0, 1, 2, 3, 4, 5, 6, 7]
#train_loss = [4.231, 2.832, 2.510, 2.292, 2.125, 1.979, 1.849, 1.716]
#train_ppl = [68.778, 16.974, 12.310, 9.896, 8.370, 7.234, 6.354, 5.565]

# Creating the plot
plt.figure(figsize=(10, 6))

# Plotting both the train loss and the perplexity on the same graph but with different y-axes
plt.title("Training Loss and Perplexity Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Train Loss", color='blue')
plt.plot(epochs, train_loss, label="Train Loss", color='blue')

# Adding a second y-axis for the perplexity
plt2 = plt.twinx()
plt2.set_ylabel("Train Perplexity", color='red')
plt2.plot(epochs, train_ppl, label="Train Perplexity", color='red')

# Show the plot
plt.show()