import os

# Current eeg_feat path
eeg_feat = "/space/emotion_data/microgestures/HCI_tagging/emotion_elicitation/bdf_content/eeg/SomeName_Something.npz"

# Split the path into parts
head, tail = os.path.split(eeg_feat)
eeg_feat_updated = os.path.join(head, 'features', tail)
eeg_feat_updated = eeg_feat_updated.replace("\\", "/")

print(eeg_feat_updated)