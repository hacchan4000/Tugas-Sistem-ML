from prometheus_client import start_http_server, Gauge
import random
import time

# Membuat metrik
current_prediction_value = Gauge("model_prediction_value", "Latest model prediction output")
model_latency_seconds = Gauge("model_latency_seconds", "Time taken to perform inference")

def get_dummy_prediction():
    """Simulasi prediksi model dan latencynya."""
    start = time.time()
    # Dummy prediksi (ganti nanti dengan real inference)
    pred = random.random()
    latency = time.time() - start
    return pred, latency

if __name__ == "__main__":
    # Menjalankan exporter di port 8000
    print("Prometheus exporter running at http://localhost:8000/metrics")
    start_http_server(8000)

    while True:
        pred, latency = get_dummy_prediction()
        current_prediction_value.set(pred)
        model_latency_seconds.set(latency)
        time.sleep(5)  # update tiap 5 detik
