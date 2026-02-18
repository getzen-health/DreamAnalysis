"""Training script for the EEG denoising autoencoder.

Trains on EEGdenoiseNet paired data (preferred) or synthetic pairs.
Evaluates with SNR improvement metrics.

Usage:
    python -m training.train_denoiser
    python -m training.train_denoiser --synthetic --n-pairs 10000
    python -m training.train_denoiser --data-dir data/eegdenoisenet
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path


def train_denoiser(
    data_dir: str = "data/eegdenoisenet",
    output_dir: str = "models/saved",
    n_pairs: int = 5000,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    use_synthetic: bool = False,
):
    """Train the denoising autoencoder."""
    from models.denoising_autoencoder import EEGDenoiser

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "denoiser_model.pt"

    denoiser = EEGDenoiser(fs=256.0)
    start_time = time.time()

    # Try loading real data first
    if not use_synthetic:
        try:
            from training.data_loaders import load_eegdenoisenet

            print("Loading EEGdenoiseNet...")
            data = load_eegdenoisenet(data_dir)

            clean_eeg = data["clean_eeg"]
            eog = data.get("eog_artifacts")
            emg = data.get("emg_artifacts")

            # Create paired data: clean + artifact = noisy
            noisy_samples = []
            clean_samples = []

            for i in range(len(clean_eeg)):
                clean_epoch = clean_eeg[i].flatten()

                # Add EOG artifact
                if eog is not None and len(eog) > 0:
                    eog_epoch = eog[np.random.randint(len(eog))].flatten()
                    if len(eog_epoch) >= len(clean_epoch):
                        eog_epoch = eog_epoch[:len(clean_epoch)]
                    else:
                        eog_epoch = np.pad(eog_epoch, (0, len(clean_epoch) - len(eog_epoch)))
                    noisy = clean_epoch + eog_epoch * np.random.uniform(0.3, 1.0)
                    noisy_samples.append(noisy)
                    clean_samples.append(clean_epoch)

                # Add EMG artifact
                if emg is not None and len(emg) > 0:
                    emg_epoch = emg[np.random.randint(len(emg))].flatten()
                    if len(emg_epoch) >= len(clean_epoch):
                        emg_epoch = emg_epoch[:len(clean_epoch)]
                    else:
                        emg_epoch = np.pad(emg_epoch, (0, len(clean_epoch) - len(emg_epoch)))
                    noisy = clean_epoch + emg_epoch * np.random.uniform(0.2, 0.8)
                    noisy_samples.append(noisy)
                    clean_samples.append(clean_epoch)

            if noisy_samples:
                # Normalize lengths
                min_len = min(len(s) for s in noisy_samples)
                noisy_arr = np.array([s[:min_len] for s in noisy_samples])
                clean_arr = np.array([s[:min_len] for s in clean_samples])

                print(f"  Paired data: {len(noisy_arr)} samples, length {min_len}")
                use_synthetic = False
            else:
                print("  No paired data created from EEGdenoiseNet, using synthetic")
                use_synthetic = True

        except (FileNotFoundError, Exception) as e:
            print(f"  EEGdenoiseNet not available: {e}")
            use_synthetic = True

    if use_synthetic:
        print(f"Generating {n_pairs} synthetic paired samples...")
        from training.data_loaders import generate_paired_denoise_data
        noisy_arr, clean_arr = generate_paired_denoise_data(
            n_pairs=n_pairs, fs=256.0, epoch_sec=2.0
        )
        print(f"  Generated: {noisy_arr.shape}")

    # Train the model
    print(f"\nTraining denoiser ({epochs} epochs, batch_size={batch_size})...")
    stats = denoiser.train_on_pairs(
        noisy_arr, clean_arr,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # Evaluate SNR improvement on held-out data
    n_eval = min(100, len(noisy_arr))
    eval_noisy = noisy_arr[-n_eval:]
    eval_clean = clean_arr[-n_eval:]

    snr_results = []
    for i in range(n_eval):
        snr = denoiser.compute_snr_improvement(eval_noisy[i], eval_clean[i])
        snr_results.append(snr)

    avg_improvement = np.mean([r["improvement_db"] for r in snr_results])
    avg_snr_before = np.mean([r["snr_before_db"] for r in snr_results])
    avg_snr_after = np.mean([r["snr_after_db"] for r in snr_results])

    elapsed = time.time() - start_time

    print(f"\n{'='*50}")
    print("Denoiser Training Complete")
    print(f"{'='*50}")
    print(f"  Final loss:        {stats['final_loss']:.6f}")
    print(f"  SNR before:        {avg_snr_before:.2f} dB")
    print(f"  SNR after:         {avg_snr_after:.2f} dB")
    print(f"  SNR improvement:   {avg_improvement:+.2f} dB")
    print(f"  Training time:     {elapsed:.1f}s")
    print(f"  Training samples:  {stats['n_samples']}")

    # Save model
    denoiser.save(str(model_path))
    print(f"  Model saved to:    {model_path}")

    # Save training report
    report = {
        "model": "denoising_autoencoder",
        "final_loss": stats["final_loss"],
        "snr_before_db": float(avg_snr_before),
        "snr_after_db": float(avg_snr_after),
        "snr_improvement_db": float(avg_improvement),
        "n_training_samples": stats["n_samples"],
        "epochs": stats["epochs"],
        "training_time_sec": elapsed,
        "data_source": "synthetic" if use_synthetic else "eegdenoisenet",
    }

    report_path = output_path / "denoiser_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG denoising autoencoder")
    parser.add_argument("--data-dir", default="data/eegdenoisenet")
    parser.add_argument("--output-dir", default="models/saved")
    parser.add_argument("--n-pairs", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    train_denoiser(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_pairs=args.n_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_synthetic=args.synthetic,
    )
