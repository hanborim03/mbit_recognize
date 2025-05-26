import json
import pandas as pd

def load_qna(file_path):
    """Load QnA data from tsv file and rename 'id' to 'idx'."""
    df = pd.read_csv(file_path, sep="\t")
    df['idx'] = df['id']
    df['a_FT'] = None  # instagram과 통일
    return df[['idx', 'question', 'answer', 'q_mbti', 'a_mbti', 'a_FT']]

def load_instagram(file_path, mbti_ft):
    """Load Instagram data and format it to match QnA structure."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        data = []
        if content.startswith('['):  # JSON 배열
            try:
                data = json.loads(content)
            except Exception as e:
                print(f"JSON 배열 파싱 오류: {e}")
        else:
            for line in content.splitlines():
                try:
                    if line.strip():
                        data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"오류 발생: {e} - 잘못된 줄은 건너뜁니다.")
                    continue

    if not data:
        print(f"경고: {file_path}에서 데이터를 불러오지 못했습니다.")
        return pd.DataFrame(columns=['idx', 'question', 'answer', 'q_mbti', 'a_mbti', 'a_FT'])

    df = pd.DataFrame(data)
    if 'query' not in df.columns or 'response' not in df.columns:
        print(f"경고: {file_path}에 'query' 또는 'response' 컬럼이 없습니다.")
        return pd.DataFrame(columns=['idx', 'question', 'answer', 'q_mbti', 'a_mbti', 'a_FT'])

    df['idx'] = df.index + 1_000_000  # 기존 QnA와 겹치지 않게 idx 설정
    df['question'] = df['query']
    df['answer'] = df['response']
    df['q_mbti'] = None
    df['a_mbti'] = None
    df['a_FT'] = mbti_ft  # 'f' 또는 't'

    return df[['idx', 'question', 'answer', 'q_mbti', 'a_mbti', 'a_FT']]

def extract_mbti_traits(mbti):
    """Extract MBTI traits from a 4-letter MBTI string."""
    if isinstance(mbti, str) and len(mbti) == 4:
        mbti = mbti.lower()
        return {
            'IE': mbti[0],
            'NS': mbti[1],
            'FT': mbti[2],
            'PJ': mbti[3]
        }
    else:
        return {
            'IE': None,
            'NS': None,
            'FT': None,
            'PJ': None
        }

def convert_mbti_to_labels(df):
    """Convert full MBTI to individual dimension labels (q_ and a_)."""
    # 질문자 MBTI
    q_traits = df['q_mbti'].apply(extract_mbti_traits).apply(pd.Series)
    q_traits.columns = ['q_IE', 'q_NS', 'q_FT', 'q_PJ']
    # 답변자 MBTI
    a_traits = df['a_mbti'].apply(extract_mbti_traits).apply(pd.Series)
    a_traits.columns = ['a_IE', 'a_NS', 'a_FT', 'a_PJ']
    # instagram 데이터의 a_FT를 우선 적용
    # (a_mbti에서 추출한 a_FT가 None이면, df['a_FT'] 사용)
    a_traits['a_FT'] = a_traits['a_FT'].combine_first(df['a_FT'])
    # 원본 df에서 a_FT 컬럼 제거 (중복 방지)
    df_no_aft = df.drop(columns=['a_FT']) if 'a_FT' in df.columns else df
    return pd.concat([df_no_aft, q_traits, a_traits], axis=1)

def merge_data(qna_file, multiple_qna_file, instagram_f_file, instagram_t_file):
    """Merge all QnA data from different sources."""
    qna_df = load_qna(qna_file)
    multiple_qna_df = load_qna(multiple_qna_file)
    instagram_f_df = load_instagram(instagram_f_file, 'f')
    instagram_t_df = load_instagram(instagram_t_file, 't')
    combined_df = pd.concat([qna_df, multiple_qna_df, instagram_f_df, instagram_t_df], ignore_index=True)
    if 'label' in combined_df.columns:
        combined_df.drop(columns=['label'], inplace=True)
    if 'a_FT' not in combined_df.columns:
        combined_df['a_FT'] = None
    return combined_df

def prepare_for_model(df):
    """Add MBTI binary dimension labels and select final columns."""
    df = convert_mbti_to_labels(df)
    final_cols = [
        'idx', 'question', 'answer', 'q_mbti', 'a_mbti',
        'q_IE', 'q_NS', 'q_FT', 'q_PJ',
        'a_IE', 'a_NS', 'a_FT', 'a_PJ'
    ]
    for col in final_cols:
        if col not in df.columns:
            df[col] = None
    return df[final_cols]

def save_data(data, output_file):
    data.to_csv(output_file, index=False, encoding='utf-8')

def main():
    print("데이터 로딩 시작")
    qna_file = 'data/qna_cleaned.tsv'
    multiple_qna_file = 'data/multiple_qna_cleaned.tsv'
    instagram_f_file = 'data/instagram_chatgpt_F.jsonl'
    instagram_t_file = 'data/instagram_chatgpt_T.jsonl'
    merged_data = merge_data(qna_file, multiple_qna_file, instagram_f_file, instagram_t_file)
    prepared_data = prepare_for_model(merged_data)
    output_file = '/home/hanborim/mbti_project/data/merged_data.csv'
    save_data(prepared_data, output_file)
    print("데이터 통합 완료:", output_file)

if __name__ == "__main__":
    main()
