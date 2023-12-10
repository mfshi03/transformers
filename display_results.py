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

print(epochs)
print(train_loss)
print(train_ppl)
# Creating the plot
plt.figure(figsize=(10, 6))

# Plotting the train loss
plt.title("Training Loss and Perplexity Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Train Loss", color='blue')
loss_plot, = plt.plot(epochs, train_loss, label="Train Loss", color='blue')

# Adding a second y-axis for the perplexity
plt2 = plt.twinx()
plt2.set_ylabel("Train Perplexity", color='red')
ppl_plot, = plt2.plot(epochs, train_ppl, label="Train Perplexity", color='red')

# Creating a legend that includes both plots
plt.legend(handles=[loss_plot, ppl_plot], loc='upper right')

# Show the plot
plt.show()