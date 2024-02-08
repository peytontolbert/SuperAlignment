
import matplotlib.pyplot as plt
import json
def plot_and_save_metrics(
    metrics, filename="training_metrics.png", metrics_path="training_metrics.json"
):
    epochs = range(1, len(metrics["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics["train_loss"], label="Training Loss")
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss")
    plt.title("Loss Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # If you track accuracy, you can plot it here similarly.
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, metrics['train_accuracy'], label='Training Accuracy')
    # plt.plot(epochs, metrics['val_accuracy'], label='Validation Accuracy')
    # plt.title('Accuracy Metrics')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    # Save metrics to a JSON file for further analysis
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
