pretrained_models_path = {
    'bert-base-uncased': 'bert-base-uncased',
    # Set this to a local directory if you run offline (e.g. '/path/to/bert-large-uncased').
    'bert-large-uncased': 'bert-large-uncased',
}

video_feats_map = {
    'video_feats.pkl': 'resnet-50',     # MIntRec
    'swin_roi.pkl': 'swin-roi',         # MIntRec2.0
}

audio_feats_map = {
    'wavlm_feats.pkl': 'wavlm',         # MIntRec2.0
    'audio_feats.pkl': 'wav2vec2',      # MIntRec
}

feat_dims = {
    'text': {
        'bert-base-uncased': 768,
        'bert-large-uncased': 1024
    },
    'video': {
        'resnet-50': 256,
        'swin-roi': 256,
    },
    'audio': {
        'wavlm': 768,
        'wav2vec2': 768,
    }
}
