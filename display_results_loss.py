import matplotlib.pyplot as plt
# Data

epochs = []
train_loss = []
val_loss = []

with open("validation.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        spl = line.split("|")
        epochs.append(int(spl[0].split(":")[1]))
        val_loss.append(float(spl[1].split(":")[1]))
        #train_ppl.append(float(spl[2].split(":")[1]))

with open("training.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        spl = line.split("|")
        train_loss.append(float(spl[1].split(":")[1]))

# Creating the plot
plt.figure(figsize=(10, 6))

# Plotting the train loss and validation loss
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, train_loss, label="Train Loss", color='blue')
plt.plot(epochs, val_loss, label="Val Loss", color='red')

# Creating a legend
plt.legend(loc='upper right')

# Show the plot
plt.show()