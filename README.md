**Efficiency analysis of CGHs with DNNs**

Code archive for diploma project in PW photonics engineering specialization.

You need to replace paths to corresponding path.

This repo consists of....
1. H5 model weight file from TensorFlow.
2. TF2ONNX_portal : Code for converting it to ONNX(32bits and 16bits), and OpenVINO(both convertion and execution).
3. ONNX16bit_8bit/ONNX16_inference : Run the inference in certain environment(CPU, CUDA or TensorRT, DirectML). Code for dynamic quantizaiton is added (static one may be included later)
4. GS-based/Fidoc_ASM_TFoptimized : Iterative phase retrieval for phase only hologram generation. Based on Fineup with don't care(Fidoc)
5. image_eval : Several image evaluation metrics are implemented. This code is for "import image_eval"
6. Validation960_half : numpy array it contains 50 images from DIV2K-validation dataset. https://data.vision.ee.ethz.ch/cvl/DIV2K/
