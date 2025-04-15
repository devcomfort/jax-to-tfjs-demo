"""
Visualization Utilities

Provides various functions for visualizing model evaluation results.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import Dict, List, Tuple, Optional, Any, Union

# Logging setup
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MNIST class names
MNIST_CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def visualize_image_predictions(images: np.ndarray, 
                              true_labels: np.ndarray, 
                              predicted_labels: np.ndarray,
                              num_samples: int = 10,
                              output_dir: str = 'evaluation_results',
                              filename: str = 'prediction_samples.png') -> str:
    """
    Visualize images and their prediction results.
    
    Args:
        images: Array of images
        true_labels: Array of true labels
        predicted_labels: Array of predicted labels
        num_samples: Number of samples to visualize
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        str: Path to the saved image file
    """
    # Create directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Randomly select samples
    np.random.seed(42)  # Set seed for reproducibility
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    # Setup figure and subplots
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # Visualize each sample
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        # Display image
        image = images[idx].squeeze()
        axes[i].imshow(image, cmap='gray')
        
        # Get true and predicted labels
        true_label = true_labels[idx]
        pred_label = predicted_labels[idx]
        
        # Set color based on correctness (green if correct, red if wrong)
        color = 'green' if true_label == pred_label else 'red'
        
        # Set title
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
        axes[i].axis('off')
    
    # Adjust spacing
    plt.tight_layout()
    
    # Save image
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    
    logger.info(f"Prediction sample image saved to: {output_path}")
    
    return output_path

def visualize_confusion_matrix(confusion_matrix_data: np.ndarray,
                             output_dir: str = 'evaluation_results',
                             filename: str = 'confusion_matrix.png',
                             normalize: bool = False,
                             class_names: Optional[List[str]] = None) -> str:
    """
    Visualize confusion matrix.
    
    Args:
        confusion_matrix_data: Confusion matrix data
        output_dir: Output directory
        filename: Output filename
        normalize: Whether to normalize the matrix
        class_names: List of class names
        
    Returns:
        str: Path to the saved image file
    """
    # Create directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy array and normalize if needed
    cm = confusion_matrix_data.copy()
    
    # Set default class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Setup figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
               xticklabels=class_names, 
               yticklabels=class_names)
    
    plt.title(title, fontsize=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Save image
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix image saved to: {output_path}")
    
    return output_path

def visualize_roc_curve(true_labels: np.ndarray,
                      predicted_probs: np.ndarray,
                      output_dir: str = 'evaluation_results',
                      filename: str = 'roc_curve.png',
                      class_names: Optional[List[str]] = None) -> str:
    """
    Visualize ROC curve.
    
    Args:
        true_labels: True labels
        predicted_probs: Predicted probabilities
        output_dir: Output directory
        filename: Output filename
        class_names: List of class names
        
    Returns:
        str: Path to the saved image file
    """
    # Create directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of classes
    n_classes = predicted_probs.shape[1]
    
    # Set default class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    # Convert to one-hot encoding for multi-class
    y_true_bin = np.eye(n_classes)[true_labels.astype(int)]
    
    # Setup figure
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], predicted_probs[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'Class {class_names[i]}')
    
    # Plot diagonal reference line (random guess)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Setup graph
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=15)
    plt.legend(loc="lower right")
    
    # Save image
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curve image saved to: {output_path}")
    
    return output_path

def visualize_precision_recall_curve(true_labels: np.ndarray,
                                   predicted_probs: np.ndarray,
                                   output_dir: str = 'evaluation_results',
                                   filename: str = 'precision_recall_curve.png',
                                   class_names: Optional[List[str]] = None) -> str:
    """
    Visualize precision-recall curve.
    
    Args:
        true_labels: True labels
        predicted_probs: Predicted probabilities
        output_dir: Output directory
        filename: Output filename
        class_names: List of class names
        
    Returns:
        str: Path to the saved image file
    """
    # Create directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of classes
    n_classes = predicted_probs.shape[1]
    
    # Set default class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    # Convert to one-hot encoding for multi-class
    y_true_bin = np.eye(n_classes)[true_labels.astype(int)]
    
    # Setup figure
    plt.figure(figsize=(10, 8))
    
    # Plot precision-recall curve for each class
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], predicted_probs[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {class_names[i]}')
    
    # Setup graph
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=15)
    plt.legend(loc="lower left")
    
    # Save image
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Precision-recall curve image saved to: {output_path}")
    
    return output_path

def visualize_metrics_by_class(metrics: Dict[str, Any],
                             output_dir: str = 'evaluation_results',
                             filename: str = 'class_metrics.png',
                             class_names: Optional[List[str]] = None) -> str:
    """
    Visualize metrics by class as a bar graph.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Output directory
        filename: Output filename
        class_names: List of class names
        
    Returns:
        str: Path to the saved image file
    """
    # Create directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract class metrics
    class_precision = metrics.get('class_precision', [])
    class_recall = metrics.get('class_recall', [])
    class_f1 = metrics.get('class_f1', [])
    
    if not (class_precision and class_recall and class_f1):
        logger.warning("Class metrics not provided.")
        return ""
    
    # Number of classes
    n_classes = len(class_precision)
    
    # Set default class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    elif len(class_names) < n_classes:
        class_names = class_names + [str(i) for i in range(len(class_names), n_classes)]
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set x-axis positions
    x = np.arange(n_classes)
    width = 0.25
    
    # Draw bar graph
    ax.bar(x - width, class_precision, width, label='Precision')
    ax.bar(x, class_recall, width, label='Recall')
    ax.bar(x + width, class_f1, width, label='F1 Score')
    
    # Setup graph
    ax.set_ylabel('Score')
    ax.set_title('Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.7)
    
    # Save image
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Class metrics image saved to: {output_path}")
    
    return output_path

def visualize_all(true_labels: np.ndarray,
                predicted_labels: np.ndarray,
                predicted_probs: np.ndarray,
                images: Optional[np.ndarray] = None,
                output_dir: str = 'evaluation_results',
                class_names: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Run all visualization functions.
    
    Args:
        true_labels: True labels
        predicted_labels: Predicted labels
        predicted_probs: Predicted probabilities
        images: Image data (optional)
        output_dir: Output directory
        class_names: List of class names
        
    Returns:
        Dict[str, str]: Dictionary of generated visualization file paths
    """
    # Create directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results
    visualization_paths = {}
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_path = visualize_confusion_matrix(
        cm, output_dir=output_dir, 
        class_names=class_names
    )
    visualization_paths['confusion_matrix'] = cm_path
    
    # Normalized confusion matrix
    cm_norm_path = visualize_confusion_matrix(
        cm, output_dir=output_dir, 
        filename='normalized_confusion_matrix.png',
        normalize=True, class_names=class_names
    )
    visualization_paths['normalized_confusion_matrix'] = cm_norm_path
    
    # ROC curve
    roc_path = visualize_roc_curve(
        true_labels, predicted_probs, 
        output_dir=output_dir, class_names=class_names
    )
    visualization_paths['roc_curve'] = roc_path
    
    # Precision-recall curve
    pr_path = visualize_precision_recall_curve(
        true_labels, predicted_probs, 
        output_dir=output_dir, class_names=class_names
    )
    visualization_paths['precision_recall_curve'] = pr_path
    
    # Calculate metrics for class metrics
    from jax_to_tfjs.evaluation.metrics import calculate_metrics
    metrics = calculate_metrics(true_labels, predicted_labels, predicted_probs)
    
    # Class metrics visualization
    metrics_path = visualize_metrics_by_class(
        metrics, output_dir=output_dir, class_names=class_names
    )
    visualization_paths['class_metrics'] = metrics_path
    
    # Visualize sample predictions if images provided
    if images is not None:
        img_path = visualize_image_predictions(
            images, true_labels, predicted_labels, 
            output_dir=output_dir
        )
        visualization_paths['prediction_samples'] = img_path
    
    logger.info(f"All visualizations saved to {output_dir} directory.")
    
    return visualization_paths 