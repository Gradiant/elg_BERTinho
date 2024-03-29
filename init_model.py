from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
    pipeline,
)


class Initializer:
    def __init__(self):

        model_path = "dvilares/bertinho-gl-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, truncation=True, max_length=512
        )
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=10)
