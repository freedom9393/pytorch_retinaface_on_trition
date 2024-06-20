<h1>Running Retinaface (Pytorch) on Triton Inference Server</h1>

<p>This repository contains instructions and scripts for running the <a href="https://github.com/biubug6/Pytorch_Retinaface)">Retinaface (Pytorch)</a> model on the <a href="https://github.com/triton-inference-server/server">Triton Inference Server</a></p>

<h2 id="prerequisites">What you need:</h2>
<ul>
  <li>Docker</li>
  <li>NVIDIA Docker (if running on GPU)</li>
  <li>Triton Inference Server</li>
</ul>

<h2 id="installation">Installation</h2>
<ol>
  <li>
    <strong>Clone the Repository</strong>
    <pre><code>git clone https://github.com/freedom9393/pytorch_retinaface_on_trition.git</code></pre>
  </li>
  <li>
    <strong>Download the Docker Image</strong>
    <pre><code>docker pull nvcr.io/nvidia/tensorrt:23.03-py3
</code></pre>
  </li>
</ol>

<h2 id="model-preparation">Directory structure</h2>
<ol>
  <li>
    <p>Create the following directory structure and place your model files accordingly:</p>
    <pre><code>models/
└── retinaface/
|   ├── 1/
|   │   └── model.onnx
|   └── config.pbtxt
├ client.py
├ config.py
├ functions.py
└ utils.py
    </code></pre>
  </li>
</ol>

<h2 id="triton-server-setup">Run Triton Inference Server with Docker</h2>
<ol>
  <li>
    <strong>Start Triton Inference Server</strong>
    <pre><code>docker run --gpus all --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v /path/to/your/model/repository:/models nvcr.io/nvidia/tritonserver:23.03-py3 tritonserver --model-repository=/models</code></pre>
  </li>
</ol>

<h2 id="license">License</h2>
<p>No License! Just do whatever you want 🤓</p>
