# phoneme_feedback.py

from Levenshtein import editops

# 오류 유형별 피드백 정의
VARIANTS_FEEDBACK = {
    ('iy', 'ih_err'): "/i/는 혀를 앞쪽 윗입천장 가까이 올리고 입은 옆으로 벌리며 발음합니다.",
    ('ax', 'ax_err'): "/ə/는 입을 살짝 벌리고 혀를 편안하게 중간에 둔 채 발음하며, 힘을 주지 않고 소리냅니다.",
    ('t', 't_err'): "/t/는 혀끝을 윗잇몸에 대어 막았다가 공기를 빠르게 터뜨리며 소리를 냅니다. 이때 목소리는 울리지 않습니다.",
    ('d', 'd_err'): "/d/는 혀끝을 윗잇몸에 대어 막았다가 공기를 내보내며 성대를 울려 소리를 냅니다.",
    ('z', 's_err'): "/z/는 혀를 윗잇몸 가까이에 두고 공기를 흘려보내며 성대를 울려 소리를 냅니다.", 
    ('t', 'dh_err'): "/t/는 혀끝을 윗잇몸에 대어 막았다가 공기를 빠르게 터뜨리며 소리를 냅니다. 이때 목소리는 울리지 않습니다.",
    ('ax', 'ih_err'): "/ə/는 입을 살짝 벌리고 혀를 편안하게 중간에 둔 채 발음하며, 힘을 주지 않고 소리냅니다.",
    ('t', 'ih_err'): "/t/는 혀끝을 윗잇몸에 대어 막았다가 공기를 빠르게 터뜨리며 소리를 냅니다. 이때 목소리는 울리지 않습니다.",
    ('s', 's_err'): "/s/는 혀를 윗잇몸 가까이에 두고 공기를 지속적으로 내보내며 발음합니다. 소리가 끊기지 않도록 주의하세요.",
    ('ax', 't_err'): "/ə/는 입을 살짝 벌리고 혀를 편안하게 중간에 둔 채 발음하며, 힘을 주지 않고 소리냅니다."
    
    # ('uh', 'uw'): "/ʊ/는 혀를 뒤쪽 위로 올리고 입술을 둥글게 모아 부드럽게 발음합니다.",
    # ('ax', 'ae'): "/ə/는 입을 살짝 벌리고 혀를 편안하게 중간에 둔 채 발음하며, 힘을 주지 않고 소리냅니다.",
    # ('ah', 'ao'): "/ʌ/는 혀를 약간 뒤로 당기고 낮춘 채 입을 벌리며 소리를 냅니다.",
    # ('dh', 'd'): "/ð/는 혀끝을 윗니와 아랫니 사이에 두고 공기를 흘려보내며 성대를 울려 소리냅니다.",
    # ('aa', 'aw'): "/ɑ/는 혀를 최대한 뒤쪽 아래로 내리고 입을 크게 벌려 소리를 냅니다.",
    # ('ay', 'iy'): "/aɪ/는 /a/와 /ɪ/를 합쳐 발음하며, /a/는 혀를 최대한 뒤쪽 아래로 내리고 입을 크게 벌리며, /ɪ/는 혀를 앞쪽 윗입천장 가까이 올리고 입은 옆으로 벌리며 발음합니다."
}

SUBSTITUTION_FEEDBACK = {
    ('ax', 'ah'): "/ə/는 입을 살짝 벌리고 혀를 편안하게 중간에 둔 채 발음하며, 힘을 주지 않고 소리냅니다.",
    ('iy', 'ih'): "/i/는 혀를 앞쪽 윗입천장 가까이 올리고 입은 옆으로 벌리며 발음합니다.",
    ('ix', 'iy'): "/ɨ/는 혀를 입 중앙에 두고 약간 뒤쪽으로 올려 발음하며 입술은 편안하게 둡니다.",
    ('aa', 'ay'): "/ə/는 입을 살짝 벌리고 혀를 편안하게 중간에 둔 채 발음하며, 힘을 주지 않고 소리냅니다."

    # ('ax', 'er'): "/ə/는 입을 살짝 벌리고 혀를 편안하게 중간에 둔 채 발음하며, 힘을 주지 않고 소리냅니다.",
    # ('iy', 'ax'): "/i/는 혀를 앞쪽 윗입천장 가까이 올리고 입은 옆으로 벌리며 발음합니다."
}

INSERTION_FEEDBACK = {
    'hh': "/h/가 삽입되어 단어의 시작이나 중간에서 불필요한 기식이 추가될 수 있습니다. 자연스러운 흐름을 유지하세요.",
    'd': "/d/가 삽입되어 강세가 지나치게 강조될 수 있습니다. 단어 간 연결을 부드럽게 하도록 주의하세요.",
    'ah': "/ɑ/가 삽입되어 단어가 늘어지거나 예상치 못한 강세가 생길 수 있습니다. 원래의 리듬을 유지하세요.",
    'iy': "/i/가 삽입되면 발음이 뚜렷해지지만 과도하면 부자연스럽게 들릴 수 있습니다. 문장의 흐름을 고려하세요.",
    'l': "/L/이 삽입되면 혀끝이 윗잇몸에 닿아 추가적인 측면음이 생성될 수 있습니다. 원래 형태를 유지하세요.",
    'n': "/n/이 삽입되면 코로 공기가 지나가며 비음이 의도치 않게 추가될 수 있습니다. 불필요한 비음을 조절하세요."
}

DELETION_FEEDBACK = {
    'iy': "/i/는 혀를 앞으로 올려 발음하는 고모음으로, 삭제 시 단어가 명확하지 않을 수 있습니다.",

    'r': "/ɹ/는 혀를 구부려 발음하는 소리로, 삭제 시 단어가 부드러움이나 연속성을 잃을 수 있습니다.",
    'uh': "/ʊ/는 혀를 뒤쪽으로 올려 발음하는 둥근 소리로, 삭제 시 음색이 평탄하게 들릴 수 있습니다.",
    'ax': "/ə/는 중립적인 소리로, 삭제 시 문장의 리듬이 단조롭게 느껴질 수 있습니다.",
    's': "/s/는 혀끝에서 나는 날카로운 소리로, 삭제 시 단어의 선명도가 떨어질 수 있습니다.",
    'd': "/d/는 혀끝으로 막아 발음하는 소리로, 삭제 시 단어가 약하게 들릴 수 있습니다."
}

def map_phonemes_to_words(correct_phoneme, correct_sentence):
    """
    정답 문장의 단어와 음소를 매핑하는 함수
    - 단어별로 음소를 정확히 배분
    """
    phonemes = correct_phoneme.lower().split()
    words = correct_sentence.lower().split()
    
    word_phoneme_map = {}
    phoneme_index = 0
    total_phonemes = len(phonemes)

    for word in words:
        word_phonemes = []

        # 단어당 음소 배정 (비율 기반 정규화)
        avg_phonemes_per_word = max(1, total_phonemes // len(words))
        while phoneme_index < len(phonemes) and len(word_phonemes) < avg_phonemes_per_word:
            word_phonemes.append(phonemes[phoneme_index])
            phoneme_index += 1

        word_phoneme_map[word] = word_phonemes

    return word_phoneme_map

def find_word_for_phoneme(word_phoneme_map, phoneme_index):
    """
    주어진 phoneme_index가 속하는 단어를 찾음
    """
    cumulative_index = 0
    closest_word = None

    for word, phonemes in word_phoneme_map.items():
        if cumulative_index <= phoneme_index < cumulative_index + len(phonemes):
            return word
        if phoneme_index < cumulative_index + len(phonemes):
            closest_word = word  # 가장 가까운 단어 저장
        cumulative_index += len(phonemes)

    return closest_word  if closest_word else "" # 매칭되는 단어가 없을 경우 가장 가까운 단어 반환


def analyze_phoneme_errors(correct_phoneme, predicted_phoneme, correct_sentence):
    """
    정답 음소와 예측 음소를 비교하여 오류 유형을 분석하고 피드백을 제공합니다.
    오류 발생 시 해당 단어를 함께 제공.
    """
    correct_tokens = correct_phoneme.lower().split()
    predicted_tokens = predicted_phoneme.lower().split()
    word_phoneme_map = map_phonemes_to_words(correct_phoneme, correct_sentence)

    feedback_list = []
    ops = editops(correct_tokens, predicted_tokens)

    for op, i, j in ops:
        word = find_word_for_phoneme(word_phoneme_map, i)  # 오류가 발생한 단어 찾기

        if op == 'replace':  # 대체 오류 (Variants 또는 Substitution)
            pair = (correct_tokens[i], predicted_tokens[j])
            if pair in VARIANTS_FEEDBACK:
                feedback_list.append(f"- 발음 변이(Variants) 오류 발생: {pair} → {VARIANTS_FEEDBACK[pair]}")
            elif pair in SUBSTITUTION_FEEDBACK:
                feedback_list.append(f"- 음소 대체(Substitution) 오류 발생: {pair} → {SUBSTITUTION_FEEDBACK[pair]}")

        elif op == 'delete':  # 삭제 오류
            phoneme = correct_tokens[i]
            if phoneme in DELETION_FEEDBACK:
                feedback_list.append(f"- 음소 삭제(Deletion) 오류 발생: {phoneme} → {DELETION_FEEDBACK[phoneme]}")

        elif op == 'insert':  # 삽입 오류
            inserted_phoneme = predicted_tokens[j]
            if inserted_phoneme in INSERTION_FEEDBACK:
                feedback_list.append(f"- 음소 삽입(Insertion) 오류 발생 in '{word}': {inserted_phoneme} → {INSERTION_FEEDBACK[inserted_phoneme]}")

    # 피드백이 없으면 긍정적 메시지 추가
    if not feedback_list:
        feedback_list.append("문제있는 발음을 찾지 못하였습니다!")

    return feedback_list
