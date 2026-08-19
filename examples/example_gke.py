"""Run remote functions on a CPU pool, a TPU pool, and a GPU pool in turn.

Prerequisites:
1. A Kinetic cluster and an active profile. Run `kinetic init` one time.
   The profile supplies the project, the zone, and the cluster.
2. A node pool for each accelerator that this script uses. Add a pool
   with `kinetic pool add`, for example:

       kinetic pool add --accelerator tpu-v6e-2x4 --spot
       kinetic pool add --accelerator gpu-t4

   `kinetic pool list` shows the pools of the cluster. Change the
   `accelerator=` values below to match your pools, or remove the calls
   for the pools that you do not have. `kinetic accelerators` lists every
   accelerator name.

Note: `tpu-v6e-2x4` is a two-host slice, so Kinetic runs it on the
Pathways backend. The `spot=True` job needs a Spot node pool.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np

import kinetic


# Example 1: CPU-only execution (works with default cluster)
@kinetic.run(accelerator="cpu")
def simple_computation(x, y):
  """Simple addition that runs on remote CPU."""
  result = x + y
  print(f"Computing {x} + {y} = {result}")
  return result


# Example 2: Keras model training on TPU
@kinetic.run(accelerator="tpu-v6e-2x4", spot=True)
def train_simple_model_tpu():
  """Train a simple Keras model on remote TPU."""

  # Create a simple model
  model = keras.Sequential(
    [
      keras.layers.Dense(64, activation="relu", input_shape=(10,)),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(1),
    ]
  )

  model.compile(optimizer="adam", loss="mse")

  # Generate some dummy data
  x_train = np.random.randn(1000, 10)
  y_train = np.random.randn(1000, 1)

  # Train the model
  print("Training model on TPU...")
  history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

  print(f"Final loss: {history.history['loss'][-1]}")
  return history.history["loss"][-1]


# Example 3: GPU training (requires GPU node pool)
@kinetic.run(accelerator="gpu-t4")
def train_model_gpu():
  """Train a Keras model on remote GPU. Requires T4 GPU node pool."""
  model = keras.Sequential(
    [
      keras.layers.Dense(128, activation="relu", input_shape=(20,)),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(1),
    ]
  )

  model.compile(optimizer="adam", loss="mse")

  x_train = np.random.randn(5000, 20)
  y_train = np.random.randn(5000, 1)

  print("Training model on T4 GPU...")
  history = model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1)

  return history.history["loss"][-1]


def main():
  """Run examples."""
  print("=" * 60)
  print("Kinetic - GKE Examples")
  print("=" * 60)

  # Example 1: Simple computation (CPU)
  # print("\n--- Example 1: Simple Computation (CPU) ---")
  # print("Running simple_computation(10, 20) on GKE...")
  # result = simple_computation(10, 20)
  # print(f"Result: {result}")

  # Example 2: Model training on CPU
  print("\n--- Example 2: Keras Model Training (CPU) ---")
  print("Training a simple model on CPU...")
  final_loss = train_simple_model_tpu()
  print(f"Model trained. Final loss: {final_loss}")

  # Example 3: GPU training (requires GPU node pool)
  # Uncomment to run if you have T4 GPU nodes available
  # print("\n--- Example 3: Model Training on T4 GPU ---")
  # final_loss = train_model_gpu()
  # print(f"Model trained. Final loss: {final_loss}")

  print("\n" + "=" * 60)
  print("Examples completed!")
  print("=" * 60)


if __name__ == "__main__":
  # Prerequisites:
  # 1. Set KINETIC_PROJECT environment variable to your GCP project ID
  #    (if `project` param is not provided in the decorator)
  # 2. Ensure your GKE cluster has GPU nodes with the required accelerator type
  main()
