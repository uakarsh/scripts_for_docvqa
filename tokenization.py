import torch

def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class Collate(object):
    
    def __init__(self, model_name : str = "layout-lmv3_hf"):
        self.model_name = model_name
    
    
    def __call__(self, list_of_ds):
        input_ids = torch.stack([x[0] for x in list_of_ds], axis = 0)
        attn_mask = torch.stack([x[1] for x in list_of_ds], axis = 0)
        token_ids = torch.stack([x[2] for x in list_of_ds], axis = 0)
        start_ps = torch.stack([x[3] for x in list_of_ds], axis = 0)
        end_ps = torch.stack([x[4] for x in list_of_ds], axis = 0)
        ids = torch.stack([x[6] for x in list_of_ds], axis = 0)
        
        encoding = {'input_ids' : input_ids, "attention_mask" : attn_mask,
                   "token_type_ids" : token_ids, "start_positions" : start_ps, "end_positions" : end_ps, "unique_id" : ids}
        
        if self.model_name in ['bert_hf', 'roberta_hf']:
            return encoding
        
        else:
            bbox = torch.stack([x[5] for x in list_of_ds], axis = 0)
            encoding['bbox'] = bbox
            
            if self.model_name == "lilt_hf":
                return encoding
            
            pixels = torch.stack([x[7] for x in list_of_ds], axis = 0)
            encoding['pixel_values'] = pixels
            return encoding