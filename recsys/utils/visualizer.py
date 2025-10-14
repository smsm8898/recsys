import matplotlib.pyplot as plt


def plot_history(history):
    metrics = list(history.keys())
    epochs = len(history[metrics[0]])
    
    _, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(range(1, epochs+1), history[metric], marker="o", label=metric)
        ax.set_title(metric.upper())
        ax.set_xlabel("Epochs")
        ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_metrics(results):
    metrics, models, ks = [], [], []
    for k in list(results.keys()):
        splitted = k.split("_")
        metrics.append(splitted[0])
        models.append(splitted[1])
        ks.append(int(splitted[2]))
    
    metrics, models, ks = sorted(set(metrics)), sorted(set(models)), sorted(set(ks))

    _, axes = plt.subplots(1, len(metrics), figsize=(18, 5), sharey=False)
    
    for i, metric_name in enumerate(metrics):
        ax = axes[i]
        for model_name in models:
            values = [results[f"{metric_name}_{model_name}_{k}"] for k in ks]
            ax.plot(ks, values, marker="o", label=model_name)
    
        ax.set_title(metric_name.upper())
        ax.set_xlabel("K")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    plt.close()
