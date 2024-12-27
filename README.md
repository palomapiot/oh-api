# Online Harms API (Oh! API) 🛡️

This FastAPI application provides an endpoint to detect online harms in social media messages using the `unsloth`
library with the LLaMA-3.1 model.

The API takes a message as input and returns whether it qualifies as hate speech along with explanations for the
classification, as well as it qualifies as fake news or hyperpartisan content.

## Features 🌟

- Uses the `unsloth` library for natural language processing.
- Detects hate speech based on a defined set of criteria.
- Detects fake news based on a defined set of criteria.
- Detects hyperpartisan news based on a defined set of criteria.
- Provides detailed explanations for each message analyzed.
- Optimized for inference with 4-bit quantized models on GPU.

## Prerequisites 📋

- Docker
- An active Hugging Face account for model access

## Setup Instructions ⚙️

### Building the Docker Image 🐳

1. Clone the repository or download the application files.

```bash
git clone git@github.com:palomapiot/oh-api.git
cd oh-api
```

2. Build the Docker image.

```bash
docker build -t oh-api .
```

### Running the Docker Container 🚀

You can run the FastAPI application in a Docker container as follows:

```bash
docker run -d -p 8000:8000 -e HF_TOKEN="your_hugging_face_token" oh-api
```

Replace `your_hugging_face_token` with your Hugging Face API token.

### Accessing the API 🌐

Once the application is running, you can access the API at `http://localhost:8000/`.

## API Endpoints 📡

The API provides the following endpoints:

- `GET /`: Returns a welcome message.
- `POST /analyze`: Analyzes a social media message for online harms.

#### Request body 

```json
{
    "id": "unique_identifier",
    "prompt": "Your message to analyze"
}

```

#### Response

```json
{
    "id": "unique_identifier",
    "text": "response"
}
```

### Error Responses

- `400 Bad Request`: If the `prompt`is not provided.

```json
{
    "detail": "Prompt is required"
}
```

- `500 Internal Server Error`: If an error occurs during the analysis.

```json
{
    "detail": "Error message"
}
```

## Citation 📑

If you use this demo or any part of the code in this repository, please cite the following reference:

```bibtex
Soon! 🚀
```

If you use `Llama-3-8B-Distil-MetaHate` model, please cite the following reference:

```bibtex
@misc{piot2024efficientexplainablehatespeech,
      title={Towards Efficient and Explainable Hate Speech Detection via Model Distillation}, 
      author={Paloma Piot and Javier Parapar},
      year={2024},
      eprint={2412.13698},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.13698}, 
}
```

## Contributing 🤝

If you want to contribute to this project, feel free to open issues or submit pull requests. All contributions are
welcome!

## Disclaimer ⚠️

This repository includes content that may contain hate speech, offensive language, or other forms of inappropriate and objectionable material. The content present in the dataset or code is not created or endorsed by the authors or contributors of this project. It is collected from various sources and does not necessarily reflect the views or opinions of the project maintainers.  The purpose of using this repository is for research, analysis, or educational purposes only. The authors do not endorse or promote any harmful, discriminatory, or offensive behavior conveyed in the dataset.

Users are advised to exercise caution and sensitivity when interacting with or interpreting the repository. If you choose to use the datasets or models, it is recommended to handle the content responsibly and in compliance with ethical guidelines and applicable laws.  The project maintainers disclaim any responsibility for the content within the repository and cannot be held liable for how it is used or interpreted by others.

## Acknowledgements 🙏

Soon! 🚀

## License 📜

This project utilizes the Llama 3.1 Community License Agreement. By using or distributing any portion of the Llama
Materials, you agree to be bound by the terms and conditions set forth in the Agreement.

For full details, please refer to the following:

- [Llama 3.1 Community License Agreement](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct/blob/main/LICENSE)

## Contact 📬

For further questions, inquiries, or discussions related to this project, please feel free to reach out via email:

- **Email:** [paloma.piot@udc.es](mailto:paloma.piot@udc.es)

If you encounter any issues or have specific questions about the code, we recommend opening an [issue on GitHub](https://github.com/palomapiot/oh-api/issues) for better visibility and collaboration.