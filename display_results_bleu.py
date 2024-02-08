import matplotlib.pyplot as plt

epochs = []
bleus = []

with open("validation.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        spl = line.split("|")
        epochs.append(int(spl[0].split(":")[1]))
        bleus.append(float(spl[3].split(":")[1]))
        #train_ppl.append(float(spl[2].split(":")[1]))

# Creating the plot
plt.figure(figsize=(10, 6))

# Plotting the train loss and validation loss
plt.title("BLEU Score for Validation Set Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("BLEU Score")
plt.plot(epochs, bleus, label="Average Epoch BLEU score", color='blue')

# Creating a legend
plt.legend(loc='upper right')

# Show the plot
plt.show()