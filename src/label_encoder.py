from sklearn.preprocessing import LabelEncoder

def encode_labels(labels):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    return y, label_encoder

def mbti_to_binary_labels(mbti_types):
    # 각 축별로 0/1 라벨 생성
    ie = [0 if t[0].upper() == 'I' else 1 for t in mbti_types]
    ns = [0 if t[1].upper() == 'N' else 1 for t in mbti_types]
    tf = [0 if t[2].upper() == 'T' else 1 for t in mbti_types]
    jp = [0 if t[3].upper() == 'J' else 1 for t in mbti_types]
    return {'IE': ie, 'NS': ns, 'TF': tf, 'JP': jp}
