# Local LLM

## Setup

### Run LLM on your local machine

We can run open source LLM locally on our machine by using a powerful library named `ollama`.

Although it's name has llama it does support other open source LLMs too. Like Mistral, Gemmma and more.

## Steps:
1. First, we need to install ollama (I am doing on Linux)
- We can do direct install on Linux or use Docker to do so.
```sh
curl -fsSL https://ollama.com/install.sh | sh
```

2. Install your LLM. (I am doing `phi3` here, Pick [yours](https://ollama.com/library))
```sh
ollama pull phi3
```

3. Run your installed LLM.
```sh
ollama run phi3
# this will install model if not already install
```

4. To remove your model.
```sh
ollama rm phi3
```

### Questions

1. Can embedding model and LLM model can be different?
Answer: Yes, they can. Because when you use embedding model, we are basically coverting the source(context) text's chunks and our query text to embedding and fetching matching/similar N number of chunks of source text to the query text. So there is no direct 
relation between them. 