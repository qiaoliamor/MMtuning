# MMtuning
PEFT framework tailored for multimodal large language models（MM-LLMs）based

## Getting Started:

### Installation: 
Start by installing the required packages listed in the environment.yml file and then install our modified Python package with MMtuning-enabled PEFT.
```bash
## Create env
conda create --name <env> --file environment.yml

## pip install PEFT-MMtuning
pip install /path/to/your/local/package/PEFT-MMtuning/dist/peft-0.6.3.dev0.tar.gz
```

### Download Datasets: 
You can download either the ScienceQA or Visual7W dataset for testing purposes. The default code is configured to load the ScienceQA dataset.  
- ScienceQA: https://scienceqa.github.io/  
- Visual7W:https://ai.stanford.edu/~yukez/visual7w/
