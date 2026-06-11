import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None

tflitemohinh_path = "best_full_integer_quant.tflite"  

if tf is None:
    print("Cai tensorflow de kiem tra chi tiet IO.")
else:
    itp = tf.lite.Interpreter(model_path=tflitemohinh_path)
    itp.allocate_tensors()
    print("Input:", itp.get_input_details()[0]["dtype"], itp.get_input_details()[0]["quantization"])
    print("Output:", itp.get_output_details()[0]["dtype"], itp.get_output_details()[0]["quantization"])
