import os
import datetime
import logging
import matplotlib.pyplot as plt
import io
from PIL import Image


def get_logger(filepath: str = None, level: int = logging.INFO) -> logging.Logger:
    if filepath is None:
        filepath = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')
    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        open(filepath, 'w').close()  # reset log file if exists
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filepath),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def mels_to_image(mel_gt, mel_synth):
    # Create a figure with two vertical subplots.
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Plot the ground-truth mel-spectrogram.
    axs[0].imshow(mel_gt, origin='lower', aspect='auto')
    axs[0].set_title("Ground-truth")
    axs[0].axis('off')  # Hide axes for a cleaner look

    # Plot the synthesized mel-spectrogram.
    axs[1].imshow(mel_synth, origin='lower', aspect='auto')
    axs[1].set_title("Synthesized")
    axs[1].axis('off')

    plt.tight_layout()

    # Save the figure to a BytesIO buffer.
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    # Open the image from the buffer and convert to RGB.
    combined_img = Image.open(buf).convert("RGB")
    return combined_img
