variable "DOCKERHUB_REPO" {
  default = "runpod"
}

variable "DOCKERHUB_IMG" {
  default = "worker-v1-vllm"
}

variable "RELEASE_VERSION" {
  default = "latest"
}

variable "HUGGINGFACE_ACCESS_TOKEN" {
  default = ""
}

variable "MODEL_NAME" {
  default = "datalab-to/chandra"
}

variable "BASE_PATH" {
  default = "/models"
}

group "default" {
  targets = ["worker-vllm"]
}

target "worker-vllm" {
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}"]
  context = "."
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64"]
  args = {
    MODEL_NAME = MODEL_NAME
    BASE_PATH  = BASE_PATH
  }
  secret = ["id=HF_TOKEN,env=HF_TOKEN"]
}