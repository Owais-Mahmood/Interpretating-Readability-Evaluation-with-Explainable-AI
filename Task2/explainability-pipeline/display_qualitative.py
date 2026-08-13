import pandas as pd
pd.set_option('display.max_colwidth', None)
df = pd.read_csv('outputs/tables/qualitative_examples.csv')

import sys
strategy = sys.argv[1] if len(sys.argv) > 1 else 'Modulation'

for _, row in df[df['strategy'] == strategy].iterrows():
    print('===', row['label'], '===')
    print('Source:', row['source_text'])
    print('Simplified:', row['simplified_text'])
    print('Confidence:', row['confidence'])
    print('Deletion spans:', row['deletion_spans'])
    print('Insertion spans:', row['insertion_spans'])
    print('AttnLRP tokens:', row['AttnLRP_tokens'])
    print('GradientSHAP tokens:', row['GradientSHAP_tokens'])
    print('IG tokens:', row['Integrated Gradients_tokens'])
    print('RawAttn tokens:', row['Raw Attention_tokens'])
    print()