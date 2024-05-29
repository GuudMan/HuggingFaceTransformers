from datasets import load_from_disk
processed_datasets = load_from_disk("./processed_data")
print(processed_datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 5850
    })
    validation: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 1679
    })
})
"""

