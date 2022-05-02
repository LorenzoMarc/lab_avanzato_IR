import argparse

from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
tok = BartTokenizer.from_pretrained("facebook/bart-large")
example_english_phrase = "UN <mask> Says There Is No <mask> in Syria"
batch = tok(example_english_phrase, return_tensors="pt")
generated_ids = model.generate(batch["input_ids"],
                               max_length=300,
                               do_sample=True,
                               # top_k=5,
                               top_p=0.9,
                               # temperature=0.9,
                               # num_beams=6,
                               no_repeat_ngram_size=3,
                               num_return_sequences=3,
                               # repetition_penalty=1.3,
                               min_length=30
                               # early_stopping=True
                               )
res = tok.batch_decode(generated_ids, skip_special_tokens=True)
print(res)


