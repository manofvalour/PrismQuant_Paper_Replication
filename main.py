
from src.prismquant_replication.prism_llama_model import prepare_prismquant_model, PrismQuantKVCacheQuantizer, PrismQuantQuantizer
from src.prismquant_replication.prismquant_engine import PrismQuant
from src.prismquant_replication.data_prep import load_wikitext_text, tokenize_text, calibration_batches
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"])

def main():
    ## loading the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer= AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    #loading the dataset
    data = load_wikitext_text('test')
    tokenized = tokenize_text(tokenizer, data)
    batched_data = calibration_batches(tokenized, samples=3, seqlen=2048)
    dataloaded_data =DataLoader(batched_data, batch_size=8)

    prism_model, tokenizer = prepare_prismquant_model(model_id='meta-llama/Llama-3.2-1B',
                            calib_data=dataloaded_data,
                            prism_class= PrismQuant(),
                            quantizer=PrismQuantQuantizer(),
                            kv_quantizer = PrismQuantKVCacheQuantizer())

if __name__== "__main__":
    main()