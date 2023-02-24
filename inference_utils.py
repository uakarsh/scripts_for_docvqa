import torch
from torch.utils.data import TensorDataset


def store_ids(list_of_ids):
    ids = {}
    current_count = 0
    for i in range(len(list_of_ids)):
        current_id = list_of_ids[i].split("/")[-1]
        if current_id not in ids:
            ids[current_id] = current_count
            current_count += 1
            
    return ids


def convert_features_to_tensor(features):
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_bboxes = torch.tensor([f.boxes for f in features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_positions for f in features], dtype=torch.long) 
    all_end_positions = torch.tensor([f.end_positions for f in features], dtype=torch.long)
    # all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.long)
    # all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    
    img_id = [f.image_id for f in features]
    img_id = store_ids(img_id) ## Would be useful in calculating the scores
    transform_id = torch.tensor([img_id[f.image_id.split("/")[-1]] for f in features ])
    
    if features[0].img is not None:
        all_imgs = torch.stack([torch.from_numpy(x.img['pixel_values'][0]) for x in features])
        dataset = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions, all_bboxes, transform_id,
            all_imgs, 
        )
        
    else:
        dataset = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions, all_bboxes,transform_id
        )
    return dataset, img_id


def decode_for_batch(start_ps, end_ps, input_ids, tokenizer):
    decoded_statements = []
    batch_size = len(input_ids)
    
    for i in range(batch_size):
        current_id = input_ids[i]
        sent = current_id[start_ps[i] : end_ps[i] + 1].detach().cpu().tolist()
        sent = tokenizer.decode(sent)
        decoded_statements.append(sent)
        
    return decoded_statements


def get_pred_references(output, dl, tokenizer):
    pred_decode_sent = decode_for_batch(start_ps = dl['start_positions'].detach().cpu(), end_ps = dl['end_positions'].detach().cpu(), 
                                   input_ids = dl['input_ids'].detach().cpu(), tokenizer = tokenizer)
    
    act_decode_sent = decode_for_batch(start_ps = output.start_logits.argmax(axis = -1).detach().cpu(), end_ps = output.end_logits.argmax(axis = -1).detach().cpu(), 
                                    input_ids = dl['input_ids'].detach().cpu(), tokenizer = tokenizer)
    
    
    batch_size = len(pred_decode_sent)
    predictions = []
    references = []
    
    for i in range(batch_size):
        unique_id = str(dl['unique_id'].item())
        curr_pred = {"prediction_text" : act_decode_sent[i],
                    'id' : unique_id}
        
        curr_ref = {"answers" : {"answer_start" : [dl['start_positions'].item()], "text" : [pred_decode_sent[i]]},
                               'id' : unique_id}
        
        predictions.append(curr_pred)
        references.append(curr_ref)
        
    return predictions, references


